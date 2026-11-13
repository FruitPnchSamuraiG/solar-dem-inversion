#!/usr/bin/env python3
"""Does training enforce the bands where a DEM that obeys them exists?

A network is trained on a fixed set of pixels for which the solver's linear program is
feasible (experiments/constrained_toy.py wrote the set, with each pixel's LP optimum),
under one unconstrained stand-in for that program, and the band violations are recorded
through training. With g the distance outside the band in half-widths:

  quad       sum(x) + mu * sum relu(g)^2 with mu fixed (the loss used so far)
  quad_ramp  the same with mu raised geometrically during training (the penalty method)
  price      mu per pixel and channel, P_c / (2 eps): P_c = h_c / max_k D[c, k] is the
             single-channel price in half-widths, eps is lowered geometrically
  huber      the exact penalty, smoothed: slope capped at kappa * P_c, quadratic within
             kappa * eps of the band edge, eps lowered geometrically
  al         augmented Lagrangian with one multiplier stored per pixel, channel and side:
             sum [relu(y + 2 mu g)^2 - y^2] / (4 mu), y <- relu(y + 2 mu g) every --outer
             epochs after --al-warm epochs on the plain quadratic penalty; mu is multiplied
             by --mu-growth whenever an update leaves the mean violation above --tau times
             the previous one (the textbook rule)
  alnet      the same with the multipliers predicted from the input by a second network
             (primal-dual learning): the version that works with fresh pixels every epoch

Penalties and multipliers change only during the first 70% of the epochs; the rest trains
the final loss while the learning rate decays to zero. CPU or GPU.

    python3 experiments/constrained_toy_net.py --data output/experiments/constrained/toy \
        --form quad --mu 100 --output output/experiments/constrained/net_quad100.json
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from experiments.diagnose_bright_failures import INTENSITY_LABELS
from experiments.train_ablations import build_model
from experiments.train_scaled import NormalizedInput, harden_softplus, init_output_head, load_operators

FORMS = ("quad", "quad_ramp", "price", "huber", "al", "alnet")
INSIDE = (1e-1, 1e-2, 1e-3)
RAMP_SHARE = 0.7
Z_MAX = math.log1p(1e6)            # cap on the multiplier network's log price ratio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="stem written by constrained_toy.py (_train/_hold/_balanced.npz)")
    p.add_argument("--form", choices=FORMS, required=True)
    p.add_argument("--pixels", choices=["feasible", "all"], default="feasible",
                   help="train on pixels whose linear program is feasible, or on every pixel")
    p.add_argument("--n-train", type=int, default=1_000_000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip; 0 = off")
    p.add_argument("--optimizer", choices=["adam", "adamw", "nadam", "radam", "adafactor", "lion", "muon"],
                   default="adam",
                   help="muon: torch.optim.Muon on the hidden layers' weight matrices, Adam (lr --lr) on the "
                        "first and last layers and the biases (Muon is meant for hidden 2D weights only)")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--lion-lr", type=float, default=1e-4, help="Lion takes steps of +/- lr, about 3x smaller than Adam's")
    p.add_argument("--adafactor-lr", type=float, default=0.01, help="Adafactor's ceiling on its relative step (paper value)")
    p.add_argument("--muon-adjust", choices=["original", "match_rms_adamw"], default="original",
                   help="match_rms_adamw scales Muon's step to Adam's size, so --muon-lr can equal --lr")
    p.add_argument("--hidden", type=int, default=336)
    p.add_argument("--init-from", default="",
                   help="checkpoint from train_scaled.py to start from (its width replaces --hidden): "
                        "the trained quadratic-penalty model is the method of multipliers' first iteration")
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--mu", type=float, default=100.0, help="quad: the weight; quad_ramp, al, alnet: its start")
    p.add_argument("--mu-end", type=float, default=1e4, help="quad_ramp: the weight at the end of the ramp")
    p.add_argument("--eps", type=float, default=1.0, help="price, huber: tolerance in half-widths at the start")
    p.add_argument("--eps-end", type=float, default=1e-3)
    p.add_argument("--kappa", type=float, default=64.0, help="huber: slope cap over the single-channel price")
    p.add_argument("--outer", type=int, default=10, help="al, alnet: epochs between multiplier updates")
    p.add_argument("--al-warm", type=int, default=20,
                   help="al, alnet: epochs on the plain quadratic penalty before the first multiplier update")
    p.add_argument("--tau", type=float, default=0.5,
                   help="al, alnet: mu grows when an update leaves the mean violation above tau x the previous one")
    p.add_argument("--mu-growth", type=float, default=2.0)
    p.add_argument("--mu-max", type=float, default=1e3)
    p.add_argument("--y-max", type=float, default=1e5, help="al, alnet: safeguard on the multipliers (per half-width)")
    p.add_argument("--measure", choices=["mean", "p99"], default="mean",
                   help="al, alnet: violation measure for the mu rule and the stop: mean or 99th percentile, over "
                        "training pixels, of each pixel's largest violation (half-widths)")
    p.add_argument("--outer-mode", choices=["fixed", "settled"], default="fixed",
                   help="al, alnet: fixed = an outer step every --outer epochs; settled = an outer step once the "
                        "training loss has settled (relative fall under --settle over the last --outer epochs), the "
                        "textbook's condition that each round be solved before the multipliers or mu change")
    p.add_argument("--settle", type=float, default=0.02)
    p.add_argument("--floor-gain", type=float, default=0.0,
                   help="al, alnet: stop raising mu once a raise cut the violation measure by less than this factor "
                        "(theory predicts the factor of the raise itself, --mu-growth); the last raise is undone; 0 = off")
    p.add_argument("--stop-viol", type=float, default=0.0,
                   help="al, alnet: once the measure is at or below this (half-widths), mu and the multipliers "
                        "stop changing (the method's stopping test); 0 = never")
    p.add_argument("--dual-step", type=float, default=1.0,
                   help="al, alnet: fraction of the textbook multiplier step taken (y <- relu(y + step * 2 mu g))")
    p.add_argument("--dual-epochs", type=int, default=2, help="alnet: passes fitting the multiplier network per update")
    p.add_argument("--dual-hidden", type=int, default=256)
    p.add_argument("--dual-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023, Symbolic Discovery of Optimization Algorithms): each weight moves
    by +/- lr, the sign of an interpolation of momentum and gradient."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_((b1 * m + (1 - b1) * p.grad).sign_(), alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return None


def make_optimizers(args, model):
    """The optimizer(s) for one run; Muon runs Adam alongside for the non-hidden parameters."""
    params = list(model.parameters())
    if args.optimizer == "adam":
        return [torch.optim.Adam(params, lr=args.lr)]
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)]
    if args.optimizer == "nadam":
        return [torch.optim.NAdam(params, lr=args.lr)]
    if args.optimizer == "radam":
        return [torch.optim.RAdam(params, lr=args.lr)]
    if args.optimizer == "adafactor":
        return [torch.optim.Adafactor(params, lr=args.adafactor_lr)]
    if args.optimizer == "lion":
        return [Lion(params, lr=args.lion_lr)]
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    hidden = {id(m.weight) for m in linears[1:-1]}
    muon_params = [p_ for p_ in params if id(p_) in hidden]
    rest = [p_ for p_ in params if id(p_) not in hidden]
    print(f"Muon on {len(muon_params)} hidden weight matrices ({sum(p_.numel() for p_ in muon_params):,} weights), "
          f"Adam on the rest ({sum(p_.numel() for p_ in rest):,})", flush=True)
    return [torch.optim.Muon(muon_params, lr=args.muon_lr, weight_decay=0.0, adjust_lr_fn=args.muon_adjust),
            torch.optim.Adam(rest, lr=args.lr)]


def load_set(path, device, tolfac, lam_bar, select=None, n=None, rng=None):
    z = np.load(path)
    status = z["lp_status"]
    idx = np.arange(len(status)) if select is None else np.flatnonzero(select(status))
    if n is not None and len(idx) > n:
        idx = np.sort(rng.choice(idx, size=n, replace=False))
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a)[idx], dtype=dt, device=device)
    obs, err = t(z["obs"]), t(z["err"])
    h = tolfac * err
    return {"n": len(idx), "obs": obs, "h": h, "P": h * torch.as_tensor(lam_bar, dtype=torch.float32, device=device),
            "feasible": t(status == 0, torch.bool), "obj": t(z["lp_obj"]), "tmin": t(z["lp_tmin"]),
            "dem_lp": t(z["lp_dem_lp"]), "dem_solver": t(z["dem"]), "group": t(z["group"], torch.long),
            "y_true": torch.cat([t(z["lp_y_lo"]), t(z["lp_y_hi"])], dim=1)}


def make_net(n_out, hidden, device):
    core = build_model("mlp6", n_out, 1, 64, hidden=hidden, fourier_freqs=12)
    harden_softplus(core, -20.0)
    init_output_head(core, -3.0, 0.01)
    return NormalizedInput(core, "sqrt").to(device)


def gaps(x, D, obs, h):
    """Distance outside the band in half-widths, per channel: (below lower edge, above upper edge)."""
    r = (x @ D.T - obs) / h
    return -1.0 - r, r - 1.0


def multipliers(dual, obs, P):
    """The multiplier network's output: price ratios on a log scale -> multipliers per half-width."""
    return torch.cat([P, P], dim=1) * torch.expm1(dual(obs[:, :, None, None]).clamp(max=Z_MAX))


def penalty(form, g_lo, g_hi, st, P, Y=None):
    u2 = torch.relu(g_lo) ** 2 + torch.relu(g_hi) ** 2
    if form in ("quad", "quad_ramp"):
        return st["mu"] * u2.sum(1)
    if form == "price":
        return (P / (2.0 * st["eps"]) * u2).sum(1)
    if form == "huber":
        u = torch.relu(g_lo) + torch.relu(g_hi)          # at most one side is positive
        k, e = st["kappa"], st["eps"]
        return (P * torch.where(u <= k * e, u * u / (2.0 * e), k * u - 0.5 * k * k * e)).sum(1)
    g = torch.cat([g_lo, g_hi], dim=1)
    return ((torch.relu(Y + 2.0 * st["mu"] * g) ** 2 - Y ** 2) / (4.0 * st["mu"])).sum(1)


@torch.no_grad()
def predict(model, obs, chunk=65536):
    model.eval()
    out = [model(obs[s:s + chunk, :, None, None]) for s in range(0, len(obs), chunk)]
    model.train()
    return torch.cat(out)


@torch.no_grad()
def score(model, d, D, B, sel=None):
    """Band violations and distance from the linear program's optimum on one pixel set."""
    idx = torch.arange(d["n"], device=d["obs"].device) if sel is None else torch.nonzero(sel).flatten()
    if idx.numel() == 0:
        return None
    obs, h = d["obs"][idx], d["h"][idx]
    x = predict(model, obs)
    g_lo, g_hi = gaps(x, D, obs, h)
    u = torch.maximum(torch.relu(g_lo), torch.relu(g_hi))
    umax = u.max(1).values
    w, obj = x.sum(1), d["obj"][idx]
    dem = x @ B.T
    out = {"n": int(idx.numel()),
           "channels_outside_pct": {f"{e:g}": float(100 * (u > e).float().mean()) for e in INSIDE},
           "pixels_outside_pct": {f"{e:g}": float(100 * (umax > e).float().mean()) for e in INSIDE},
           "violation_mean": float(umax.mean()),
           "violation": {k: float(torch.quantile(umax, q)) for k, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))}
           | {"max": float(umax.max())},
           "below_band_share_pct": float(100 * (torch.relu(g_lo).sum() / u.sum().clamp(min=1e-30))),
           "weight_mean": float(w.mean())}
    zc = obs <= 1e-3                                     # channels zeroed by the deconvolution clamp
    if zc.any():                                         # their band is an upper limit only
        gz = g_hi[zc]
        out["zeroed_channels"] = {"n": int(zc.sum()),
                                  "above_band_pct": {f"{e:g}": float(100 * (gz > e).float().mean()) for e in INSIDE},
                                  "excess": {k: float(torch.quantile(gz.clamp(min=0)[:4_000_000], q))
                                             for k, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))}}
    ok = torch.isfinite(obj) & (obj > 1e-6)
    if ok.any():
        ratio = w[ok] / obj[ok]
        lp = d["dem_lp"][idx][ok]
        peak = dem[ok].max(1).values / lp.max(1).values.clamp(min=1e-30)
        out |= {"weight_over_lp_pooled": float(w[ok].sum() / obj[ok].sum()),
                "weight_over_lp_median": float(ratio.median()),
                "dem_mse_vs_lp": float(((dem[ok] - lp) ** 2).mean()),
                "dem_mae_vs_lp": float((dem[ok] - lp).abs().mean()),
                "peak_over_lp_median": float(peak.median())}
    sol = d["dem_solver"][idx]
    fin = torch.isfinite(sol).all(1)
    if fin.any():                                        # against the solver's stored DEM
        out |= {"dem_mse_vs_solver": float(((dem[fin] - sol[fin]) ** 2).mean()),
                "dem_mae_vs_solver": float((dem[fin] - sol[fin]).abs().mean()),
                "em_over_solver": float(dem[fin].sum() / sol[fin].sum().clamp(min=1e-30))}
    inf = ~d["feasible"][idx]
    if inf.any():                                        # how far above the least possible violation
        tm = d["tmin"][idx][inf]
        out["infeasible"] = {"n": int(inf.sum()), "violation_p50": float(umax[inf].median()),
                             "least_possible_p50": float(tm.median()),
                             "excess_over_least_p50": float((umax[inf] - tm).median()),
                             "excess_over_least_p90": float(torch.quantile(umax[inf] - tm, 0.9))}
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_flush_denormal(True)      # denormal floats make CPU epochs tens of times slower
    rng = np.random.default_rng(args.seed)
    n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if n_cpu:
        torch.set_num_threads(n_cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, B, n_basis, _ = load_operators(device)
    lam_bar = (1.0 / D.max(dim=1).values).cpu().numpy()
    # status 0 = a DEM fits every band; anything else is an infeasible pixel
    pick = (lambda s: s == 0) if args.pixels == "feasible" else None
    train = load_set(f"{args.data}_train.npz", device, args.tolfac, lam_bar, pick, args.n_train, rng)
    hold = load_set(f"{args.data}_hold.npz", device, args.tolfac, lam_bar)
    bal = load_set(f"{args.data}_balanced.npz", device, args.tolfac, lam_bar)
    N = train["n"]
    watch = torch.as_tensor(np.sort(rng.choice(N, size=min(N, 200_000), replace=False)), device=device)
    train_watch = {k: (v[watch] if torch.is_tensor(v) else v) for k, v in train.items()} | {"n": int(watch.numel())}
    # per-epoch scoring uses fixed subsets (the final table uses every held-out pixel)
    hw = torch.as_tensor(np.sort(rng.choice(hold["n"], size=min(hold["n"], 100_000), replace=False)), device=device)
    hold_watch = {k: (v[hw] if torch.is_tensor(v) else v) for k, v in hold.items()} | {"n": int(hw.numel())}
    print(f"device {device}, threads {torch.get_num_threads()}; train {N:,} pixels ({args.pixels}), hold {hold['n']:,}, "
          f"balanced {bal['n']:,}; form {args.form}", flush=True)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if ck.get("input_transform") != "sqrt" or ck.get("fourier_freqs") != 12 or ck.get("variant") != "mlp6":
            raise ValueError("--init-from expects an mlp6 checkpoint with sqrt input and 12 Fourier frequencies")
        args.hidden = ck.get("hidden") or 680
    model = make_net(n_basis, args.hidden, device)
    if args.init_from:
        model.inner.load_state_dict(ck["model"])
        print(f"started from {args.init_from} (epoch {ck.get('epoch')}, width {args.hidden})", flush=True)
    opts = make_optimizers(args, model)
    steps_per_epoch = N // args.batch
    total = args.epochs * steps_per_epoch
    ramp_epochs = max(1, int(round(RAMP_SHARE * args.epochs)))
    ramp_steps = ramp_epochs * steps_per_epoch

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        if step < ramp_steps:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * min((step - ramp_steps) / max(total - ramp_steps, 1), 1.0)))

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_at) for o in opts]
    st = {"mu": args.mu, "eps": args.eps, "kappa": args.kappa}
    Y = dual = dual_opt = None
    if args.form == "al":
        Y = torch.zeros(N, 12, device=device)
    if args.form == "alnet":
        dual = make_net(12, args.dual_hidden, device)
        dual_opt = torch.optim.Adam(dual.parameters(), lr=args.dual_lr)
    prev_v = None
    updates = []

    @torch.no_grad()
    def mean_violation():
        x = predict(model, train["obs"])
        g_lo, g_hi = gaps(x, D, train["obs"], train["h"])
        umax = torch.maximum(torch.relu(g_lo), torch.relu(g_hi)).max(1).values
        v = float(umax.mean()) if args.measure == "mean" else float(torch.quantile(umax[::max(1, len(umax) // 2_000_000)], 0.99))
        return x, g_lo, g_hi, v

    converged = {"epoch": None}
    floor = {"reached": None, "before_raise": None, "mu_before": None}

    def update_multipliers(epoch):
        """The outer step: y <- relu(y + 2 mu g), then the textbook rule for mu."""
        nonlocal Y, prev_v
        x, g_lo, g_hi, v = mean_violation()
        if converged["epoch"] is not None:
            return
        if args.stop_viol > 0 and v <= args.stop_viol:
            converged["epoch"] = epoch
            updates.append({"epoch": epoch, "measure": v, "mu_after": st["mu"], "stopped": True})
            print(f"   stopping test met after epoch {epoch}: violation measure {v:.4g} <= {args.stop_viol:g}; "
                  f"mu stays {st['mu']:g}", flush=True)
            return
        g = torch.cat([g_lo, g_hi], dim=1)
        if args.form == "al":
            Y = torch.relu(Y + args.dual_step * 2.0 * st["mu"] * g).clamp(max=args.y_max)
            y_mean = float(Y.mean())
        else:
            old = copy.deepcopy(dual).eval()
            P12 = torch.cat([train["P"], train["P"]], dim=1)
            with torch.no_grad():
                Y_old = torch.cat([multipliers(old, train["obs"][s:s + 65536], train["P"][s:s + 65536])
                                   for s in range(0, N, 65536)])
                target = torch.log1p(torch.relu(Y_old + args.dual_step * 2.0 * st["mu"] * g).clamp(max=args.y_max) / P12)
            fit = 0.0
            for _ in range(args.dual_epochs):
                perm = torch.randperm(N, device=device)
                for s in range(0, N - args.batch + 1, args.batch):
                    i = perm[s:s + args.batch]
                    dual_opt.zero_grad()
                    loss_d = ((dual(train["obs"][i][:, :, None, None]) - target[i]) ** 2).mean()
                    loss_d.backward()
                    dual_opt.step()
                    fit = loss_d.item()
            y_mean = float(torch.expm1(target).mean())
        if args.floor_gain > 0 and floor["before_raise"] is not None and floor["reached"] is None:
            # the round after a raise: did the raise buy the reduction theory predicts?
            if floor["before_raise"] / max(v, 1e-12) < args.floor_gain:
                floor["reached"] = epoch
                st["mu"] = floor["mu_before"]
                print(f"   network floor reached after epoch {epoch}: the raise cut the measure only "
                      f"{floor['before_raise'] / max(v, 1e-12):.2f}x; mu back to {st['mu']:g}, no further raises", flush=True)
            floor["before_raise"] = None
        grew = (prev_v is not None and v > args.tau * prev_v and st["mu"] < args.mu_max
                and floor["reached"] is None)
        if grew:
            floor["before_raise"], floor["mu_before"] = v, st["mu"]
            st["mu"] = min(st["mu"] * args.mu_growth, args.mu_max)
        prev_v = v
        updates.append({"epoch": epoch, "mean_violation": v, "mu_after": st["mu"], "multiplier_mean": y_mean}
                       | ({"dual_fit": fit} if args.form == "alnet" else {}))
        print(f"   multiplier update after epoch {epoch}: mean violation {v:.4g}, mu -> {st['mu']:g}, "
              f"mean multiplier {y_mean:.4g}", flush=True)

    history, step, t0 = [], 0, time.time()
    epoch_losses, round_start = [], {"epoch": 0}
    start = {"train": score(model, train_watch, D, B), "hold_feasible": score(model, hold_watch, D, B, hold_watch["feasible"]),
             "bright_feasible": score(model, bal, D, B, bal["feasible"] & (bal["group"] >= 6)),
             "core_feasible": score(model, bal, D, B, bal["feasible"] & (bal["group"] == 8))}
    print(f"before training: train outside>0.01 {start['train']['pixels_outside_pct']['0.01']:.2f}%, "
          f"hold w/LP {start['hold_feasible']['weight_over_lp_pooled']:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=device)
        tot = l1 = pen = 0.0
        for b in range(steps_per_epoch):
            i = perm[b * args.batch:(b + 1) * args.batch]
            p = min(step / max(ramp_steps, 1), 1.0)
            if args.form == "quad_ramp":
                st["mu"] = args.mu * (args.mu_end / args.mu) ** p
            if args.form in ("price", "huber"):
                st["eps"] = args.eps * (args.eps_end / args.eps) ** p
            obs, h = train["obs"][i], train["h"][i]
            x = model(obs[:, :, None, None])
            g_lo, g_hi = gaps(x, D, obs, h)
            Yb = None
            if args.form == "al":
                Yb = Y[i]
            elif args.form == "alnet":
                with torch.no_grad():
                    Yb = multipliers(dual, obs, train["P"][i])
            w = x.sum(1)
            q = penalty(args.form, g_lo, g_hi, st, train["P"][i], Yb)
            loss = (w + q).mean()
            for o in opts:
                o.zero_grad()
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for o in opts:
                o.step()
            for sc in scheds:
                sc.step()
            step += 1
            tot += loss.item()
            l1 += w.detach().mean().item()
            pen += q.detach().mean().item()
        epoch_losses.append(tot / steps_per_epoch)
        if args.form in ("al", "alnet") and args.al_warm <= epoch <= ramp_epochs:
            if args.outer_mode == "fixed":
                due = (epoch - args.al_warm) % args.outer == 0
            else:
                # settled: the loss fell by less than --settle over the last --outer epochs of this round
                w = epoch_losses[round_start["epoch"]:]
                due = len(w) > args.outer and (w[-args.outer - 1] - w[-1]) <= args.settle * abs(w[-args.outer - 1])
            if due:
                update_multipliers(epoch)
                round_start["epoch"] = len(epoch_losses)
        rec = {"epoch": epoch, "step": step, "lr": scheds[-1].get_last_lr()[0], "mu": st["mu"], "eps": st["eps"],
               "loss": tot / steps_per_epoch, "weight_term": l1 / steps_per_epoch, "penalty_term": pen / steps_per_epoch,
               "train": score(model, train_watch, D, B),
               "hold_feasible": score(model, hold_watch, D, B, hold_watch["feasible"]),
               "bright_feasible": score(model, bal, D, B, bal["feasible"] & (bal["group"] >= 6)),
               "core_feasible": score(model, bal, D, B, bal["feasible"] & (bal["group"] == 8)),
               "secs": time.time() - t0}
        if args.pixels == "all":
            rec["hold_all"] = score(model, hold_watch, D, B)
        history.append(rec)
        tr, ho, co = rec["train"], rec["hold_feasible"], rec["core_feasible"]
        print(f"epoch {epoch:3d}  loss {rec['loss']:.4f} (weight {rec['weight_term']:.4f}, penalty {rec['penalty_term']:.4g})  "
              f"mu {st['mu']:.4g} eps {st['eps']:.3g}  train outside>0.01: {tr['pixels_outside_pct']['0.01']:.2f}% "
              f"viol p99 {tr['violation']['p99']:.3g}  hold {ho['pixels_outside_pct']['0.01']:.2f}% w/LP {ho['weight_over_lp_pooled']:.4f}  "
              f"cores {co['pixels_outside_pct']['0.01']:.1f}% peak/LP {co['peak_over_lp_median']:.3f}  {rec['secs']:.0f}s", flush=True)

    final = {"train": history[-1]["train"], "hold_feasible": score(model, hold, D, B, hold["feasible"]),
             "hold_all": score(model, hold, D, B),
             "balanced_bands": {lab: {"feasible": score(model, bal, D, B, bal["feasible"] & (bal["group"] == g)),
                                      "infeasible": score(model, bal, D, B, ~bal["feasible"] & (bal["group"] == g))}
                                for g, lab in enumerate(INTENSITY_LABELS)}}
    if args.form in ("al", "alnet"):
        # Theory: the multipliers converge to the linear program's dual prices, which
        # training never sees. Stored multipliers are compared on the training pixels,
        # the multiplier network on held-out pixels.
        with torch.no_grad():
            if args.form == "al":
                Yc, d = Y, train
            else:
                d = {k: (v[hold["feasible"]] if torch.is_tensor(v) else v) for k, v in hold.items()}
                Yc = torch.cat([multipliers(dual.eval(), d["obs"][s:s + 65536], d["P"][s:s + 65536])
                                for s in range(0, len(d["obs"]), 65536)])
            yt = d["y_true"]
            P12 = torch.cat([d["P"], d["P"]], dim=1)
            act = yt > 1e-6 * P12
            ratio = (Yc[act] / yt[act]).clamp(min=1e-12)
            a, b = torch.log1p(Yc / P12).flatten(), torch.log1p(yt / P12).flatten()
            final["multipliers_vs_dual_prices"] = {
                "pixels": "train" if args.form == "al" else "hold_feasible",
                "active_constraints": int(act.sum()),
                "ratio_learned_over_true": {k: float(torch.quantile(ratio[:2_000_000], q))
                                            for k, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))},
                "within_factor_2_pct": float(100 * ((ratio > 0.5) & (ratio < 2)).float().mean()),
                "inactive_learned_below_tenth_of_scale_pct": float(100 * (Yc[~act] < 0.1 * P12[~act]).float().mean()),
                "log_correlation": float(torch.corrcoef(torch.stack([a, b]))[0, 1])}
    result = {"purpose": "band violations through training on pixels where the linear program is feasible",
              "args": vars(args), "device": str(device), "n_train": N, "steps_per_epoch": steps_per_epoch,
              "ramp_epochs": ramp_epochs, "params": sum(p.numel() for p in model.parameters()),
              "multiplier_updates": updates, "mu_final": st["mu"], "stopped_at_epoch": converged["epoch"],
              "floor_reached_at_epoch": floor["reached"],
              "before_training": start, "history": history, "final": final,
              "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=1)
    torch.save({"model": model.inner.state_dict(), "args": vars(args)}, os.path.splitext(args.output)[0] + ".pt")
    print(f"wrote {args.output} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
