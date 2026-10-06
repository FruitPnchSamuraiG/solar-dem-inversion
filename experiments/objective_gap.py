#!/usr/bin/env python3
"""Is the flare-core failure the network or the objective it is trained on?

The label-free BP model minimises, per pixel, the barrier objective

    L(x) = a * sum(x) + mu * sum_c [relu(lb_c - (Dx)_c)^2 + relu((Dx)_c - ub_c)^2] / s_c^2

over the 54 non-negative basis weights x (lb/ub = obs -/+ 1.4 err, s = 1.4 err),
exactly as src/losses.py:barrier_loss_batch. L is convex in x, so each pixel has
one best value L*. For a sample of test pixels in every brightness band (all
flare cores) this compares three answers under that same objective:

  net      the trained model's output
  optimum  the per-pixel minimiser of L (L-BFGS-B from both other answers)
  solver   the solver's DEM, written with the fewest basis weights
           (min sum x subject to Bx = DEM, an LP)

If the network sits far above L* at flare cores, it is not reaching its own
optimum: a training problem (sampling, steps, capacity). If it is close to L*
and the optimum itself under-fits 94 A, the objective prefers the small DEM
there and no training change can fix it. The optimum's DEM error against the
solver is the best any label-free model can do under this objective.

Alternative objectives are screened the same way, without training: their
per-pixel optima show whether a different objective would match the solver
better (`--variants`, e.g. mu10 = barrier weight x10, ibr1 = L1 weight divided
by max(1, I) with I the brightness multiple). Same for the ENet objective with
--loss enet. Runs on the staged split (needs the per-pixel errors), CPU only.

    python3 experiments/objective_gap.py --loss barrier \
        --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --models final=output/experiments/resample/scaled_mlp6_barrier_h336_resample.pt \
        --output output/experiments/diagnostics/bp_objective_gap.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from experiments.diagnose_bright_failures import (CHANNELS, INTENSITY_EDGES, INTENSITY_LABELS,
                                                  N_BINS)

MIN_OBS = 1e-3            # as src/zarr_data.py: the deconvolution-clamp floor
N_GROUPS = len(INTENSITY_LABELS)
CORE = N_GROUPS - 1       # >= 32x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="staged split with _x/_e/_y/_m zarrs")
    p.add_argument("--phase", default="test")
    p.add_argument("--loss", choices=["barrier", "enet"], default="barrier")
    p.add_argument("--models", nargs="+", required=True, help="label=checkpoint ...")
    p.add_argument("--supervised", default=None, help="supervised checkpoint (optional)")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--per-group", type=int, default=1500, help="target pixels per band")
    p.add_argument("--core-cap", type=int, default=6000, help="max flare-core pixels")
    p.add_argument("--variants", default="mu10,mu100,ibr1",
                   help="alternative objectives to screen: muK (barrier weight xK), "
                        "ibrP (L1 weight / max(1,I)^P), aK (L1 weight xK)")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--alpha-l1", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--enet-alpha", type=float, default=0.001)
    p.add_argument("--enet-lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


# ── pass 1: collect pixels by brightness band ─────────────────────────────────

def collect(job):
    cfg, start, stop, wid = job
    import zarr
    z = {k: zarr.open(os.path.join(cfg["root"], f"{cfg['phase']}_{k}.zarr"), mode="r")
         for k in ("x", "e", "y", "m")}
    thr = np.asarray(cfg["thresholds"], np.float32)[:, None, None]
    keep = np.asarray(cfg["keep"])
    rng = np.random.default_rng((cfg["seed"], wid))
    rows = {k: [] for k in ("obs", "err", "dem", "tol", "group", "inten")}
    pop = np.zeros(N_GROUPS, np.int64)
    for b in range(start, stop):
        obs = np.asarray(z["x"][:, :, :, b], np.float32)[:, ::2, ::2]
        err = np.asarray(z["e"][:, :, :, b], np.float32)[:, ::2, ::2]
        dem = np.asarray(z["y"][:N_BINS, :, :, b], np.float32)
        tol = np.asarray(z["m"][:, :, b])
        valid = (np.isfinite(obs).all(0) & np.isfinite(err).all(0) & (obs > MIN_OBS).all(0)
                 & (err > 0).all(0) & np.isin(tol, (1, 3, 5)) & np.isfinite(dem).all(0))
        ratio = (obs / thr).max(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            group = np.searchsorted(INTENSITY_EDGES, np.log10(ratio), side="right")
        group = np.clip(group, 0, N_GROUPS - 1)
        pop += np.bincount(group[valid], minlength=N_GROUPS)
        sel = valid & (rng.random(valid.shape) < keep[group])
        if sel.any():
            rows["obs"].append(obs[:, sel].T)
            rows["err"].append(err[:, sel].T)
            rows["dem"].append(dem[:, sel].T)
            rows["tol"].append(tol[sel])
            rows["group"].append(group[sel])
            rows["inten"].append(ratio[sel])
        if (b - start + 1) % 500 == 0:
            print(f"[w{wid}] {b - start + 1}/{stop - start} blocks", flush=True)
    out = {k: (np.concatenate(v) if v else None) for k, v in rows.items()}
    return out, pop


# ── objectives ────────────────────────────────────────────────────────────────

G = {}   # per-process operators, set by init_worker


def init_worker(cfg):
    from experiments.train_scaled import load_operators
    import torch
    D_t, B_t, _, _ = load_operators(torch.device("cpu"))
    G["D"] = D_t.numpy().astype(np.float64)
    G["B"] = B_t.numpy().astype(np.float64)
    G["cfg"] = cfg


def parse_variant(v):
    """-> (a multiplier, mu multiplier, brightness power). Terms combine with
    '_', e.g. mu30_ibr1 = barrier weight x30 and L1 weight / max(1, I)."""
    a_mult, mu_mult, power = 1.0, 1.0, 0.0
    if v == "base":
        return a_mult, mu_mult, power
    for term in v.split("_"):
        if term.startswith("mu"):
            mu_mult = float(term[2:])
        elif term.startswith("ibr"):
            power = float(term[3:])
        elif term.startswith("a"):
            a_mult = float(term[1:])
        else:
            raise ValueError(f"unknown variant term {term!r} in {v!r}")
    return a_mult, mu_mult, power


def parts(x, obs, err, cfg, a_mult=1.0, mu_mult=1.0, power=0.0, inten=None):
    """Objective terms for x [N,54] -> (reg [N], fit [N]); total = reg + fit."""
    D = G["D"]
    Dx = x @ D.T
    s = cfg["tolfac"] * err
    s2 = s ** 2 + 1e-8
    if cfg["loss"] == "barrier":
        lb, ub = obs - s, obs + s
        a = cfg["alpha_l1"] * a_mult
        if power and inten is not None:
            a = a / np.maximum(inten, 1.0) ** power
        reg = a * x.sum(1)
        fit = cfg["mu"] * mu_mult * (np.square(np.maximum(lb - Dx, 0)) / s2
                                     + np.square(np.maximum(Dx - ub, 0)) / s2).sum(1)
        return reg, fit
    C = D.shape[0]
    al, lam = cfg["enet_alpha"] * a_mult, cfg["enet_lam"]
    reg = al * lam * x.sum(1) + 0.5 * al * (1 - lam) * np.square(x).sum(1)
    fit = (0.5 / C) * (np.square(Dx - obs) / s2).sum(1) * mu_mult
    return reg, fit


def solve_chunk(job):
    """Per-pixel minimisers for every variant, plus the solver's fewest-weights x."""
    from scipy.optimize import linprog, minimize
    obs, err, dem, inten, starts, variants = job
    cfg, D, B = G["cfg"], G["D"], G["B"]
    n, k = len(obs), D.shape[1]
    x_sol = np.zeros((n, k))
    for i in range(n):
        target = np.maximum(dem[i].astype(np.float64), 0.0)
        r = linprog(np.ones(k), A_eq=B, b_eq=target, bounds=(0, None), method="highs")
        if r.status == 0:
            x_sol[i] = r.x
        else:                                   # spikes always represent a DEM
            x_sol[i, :B.shape[0]] = target
    out = {"solver": x_sol}
    stats = {}
    for v in variants:
        a_mult, mu_mult, power = parse_variant(v)
        xs = np.zeros((n, k))
        nonconv = 0
        for i in range(n):
            o, e = obs[i].astype(np.float64), err[i].astype(np.float64)
            s = cfg["tolfac"] * e
            s2 = s ** 2 + 1e-8

            if cfg["loss"] == "barrier":
                lb, ub = o - s, o + s
                a = cfg["alpha_l1"] * a_mult / (max(inten[i], 1.0) ** power if power else 1.0)
                m = cfg["mu"] * mu_mult

                def f(x):
                    Dx = D @ x
                    lo, hi = np.maximum(lb - Dx, 0), np.maximum(Dx - ub, 0)
                    val = a * x.sum() + m * ((lo ** 2 + hi ** 2) / s2).sum()
                    g = a + m * (D.T @ (2 * (hi - lo) / s2))
                    return val, g
            else:
                C = D.shape[0]
                al, lam = cfg["enet_alpha"] * a_mult, cfg["enet_lam"]

                def f(x):
                    r_ = D @ x - o
                    val = (0.5 / C) * mu_mult * (r_ ** 2 / s2).sum() + al * lam * x.sum() \
                        + 0.5 * al * (1 - lam) * (x ** 2).sum()
                    g = (1.0 / C) * mu_mult * (D.T @ (r_ / s2)) + al * lam + al * (1 - lam) * x
                    return val, g

            best, bval, ok = None, np.inf, False
            for x0 in [x_sol[i]] + [s_[i] for s_ in starts]:
                r = minimize(f, np.maximum(x0, 0), jac=True, method="L-BFGS-B",
                             bounds=[(0, None)] * k,
                             options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-15,
                                      "gtol": 1e-10, "maxcor": 30})
                if r.fun < bval:
                    best, bval, ok = r.x, r.fun, r.success
            xs[i] = best
            nonconv += int(not ok)
        out[v] = xs
        stats[v] = nonconv
    return out, stats


def predict_x(path, obs):
    import torch
    from src.scaled_eval import load_scaled_model
    n_basis = G["D"].shape[1]
    model, ckpt = load_scaled_model(path, n_basis, torch.device("cpu"))
    xs = []
    with torch.no_grad():
        for s in range(0, len(obs), 8192):
            o = obs[s:s + 8192]
            patch = np.zeros((len(o), 6, 9, 9), np.float32)
            patch[:, :, 4, 4] = o
            xs.append(model(torch.from_numpy(patch)).double().numpy())
    return np.concatenate(xs), ckpt


def predict_supervised_dem(path, obs):
    from experiments.diagnose_bright_failures import load_predictor
    predict, _ = load_predictor(path, 8)
    return predict(np.ascontiguousarray(obs.T[:, :, None]), 8192)[:, :, 0].T.astype(np.float64)


def min_l1_x(dems):
    from scipy.optimize import linprog
    B = G["B"]
    k = B.shape[1]
    xs = np.zeros((len(dems), k))
    for i, d in enumerate(np.maximum(dems, 0)):
        r = linprog(np.ones(k), A_eq=B, b_eq=d, bounds=(0, None), method="highs")
        xs[i] = r.x if r.status == 0 else np.r_[d, np.zeros(k - B.shape[0])]
    return xs


# ── summary ───────────────────────────────────────────────────────────────────

def describe(x, obs, err, inten, cfg, variant="base"):
    """Per-pixel quantities for one answer x under the base objective (and,
    for a variant optimum, under its own objective too)."""
    reg, fit = parts(x, obs, err, cfg)
    d = {"total": reg + fit, "reg": reg, "fit": fit}
    if variant != "base":
        a_mult, mu_mult, power = parse_variant(variant)
        r2, f2 = parts(x, obs, err, cfg, a_mult, mu_mult, power, inten)
        d["own_total"] = r2 + f2
    pred = x @ G["B"].T
    d["dem"] = pred
    d["recon"] = x @ G["D"].T
    return d


def group_summary(ans, sol_dem, obs, err, cfg, idx):
    if idx.size == 0:
        return None
    s = cfg["tolfac"] * err[idx]
    out = {"n": int(idx.size),
           "objective_mean": float(ans["total"][idx].mean()),
           "objective_median": float(np.median(ans["total"][idx])),
           "reg_mean": float(ans["reg"][idx].mean()),
           "fit_mean": float(ans["fit"][idx].mean()),
           "em_ratio_vs_solver": float(ans["dem"][idx].sum() / max(sol_dem[idx].sum(), 1e-30)),
           "dem_mse_vs_solver": float(np.square(ans["dem"][idx] - sol_dem[idx]).mean()),
           "recon_over_obs": {c: float(ans["recon"][idx, j].sum() / obs[idx, j].sum())
                              for j, c in enumerate(CHANNELS)},
           "in_band_pct": {c: float(100 * np.mean(np.abs(ans["recon"][idx, j] - obs[idx, j])
                                                  <= s[:, j] * (1 + 1e-6)))
                           for j, c in enumerate(CHANNELS)}}
    if "own_total" in ans:
        out["own_objective_mean"] = float(ans["own_total"][idx].mean())
    return out


def main():
    args = parse_args()
    import zarr
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload), np.float64)
    nb = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1]
    if args.max_blocks:
        nb = min(nb, args.max_blocks)
    variants = ["base"] + [v for v in args.variants.split(",") if v]
    if args.loss == "enet":
        variants = ["base"] + [v for v in variants[1:] if not v.startswith("ibr")]
    cfg = {"root": os.path.abspath(args.root), "phase": args.phase, "loss": args.loss,
           "thresholds": thresholds.tolist(), "seed": args.seed, "tolfac": args.tolfac,
           "alpha_l1": args.alpha_l1, "mu": args.mu, "enet_alpha": args.enet_alpha,
           "enet_lam": args.enet_lam}
    t0 = time.time()

    # Keep probabilities aim at --per-group pixels per band (population shares
    # from the final diagnostic); every flare core is kept, then capped below.
    share = np.array([3.85, 10.51, 33.02, 42.50, 6.53, 2.35, 1.15, 0.08, 0.0098]) / 100
    est_valid = nb * 128 * 128 * 0.85
    keep = np.minimum(1.0, args.per_group / (share * est_valid) * 1.5)
    keep[CORE] = 1.0
    cfg["keep"] = keep.tolist()
    chunk = math.ceil(nb / args.workers)
    jobs = [(cfg, s, min(s + chunk, nb), i) for i, s in enumerate(range(0, nb, chunk))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts_ = pool.map(collect, jobs)
    pop = sum(p for _, p in parts_)
    data = {k: np.concatenate([d[k] for d, _ in parts_ if d[k] is not None])
            for k in parts_[0][0]}
    rng = np.random.default_rng(args.seed)
    take = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(data["group"] == g)
        cap = args.core_cap if g == CORE else args.per_group
        take.append(rng.choice(idx, size=min(cap, idx.size), replace=False) if idx.size else idx)
    take = np.sort(np.concatenate(take))
    data = {k: v[take] for k, v in data.items()}
    obs = data["obs"].astype(np.float64)
    err = data["err"].astype(np.float64)
    dem = data["dem"].astype(np.float64)
    inten = data["inten"].astype(np.float64)
    group = data["group"]
    print(f"collected {len(obs):,} pixels from {nb:,} blocks in {time.time() - t0:.0f}s; "
          f"per band {np.bincount(group, minlength=N_GROUPS).tolist()}", flush=True)

    init_worker(cfg)
    answers, nets = {}, {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        x, ckpt = predict_x(path, obs.astype(np.float32))
        if ckpt["loss"] != args.loss:
            raise ValueError(f"{label} was trained with {ckpt['loss']}, not {args.loss}")
        nets[label] = x
    if args.supervised:
        answers["supervised"] = min_l1_x(predict_supervised_dem(args.supervised, obs.astype(np.float32)))

    # Per-pixel optima (and the solver's fewest-weights x), in parallel chunks.
    starts = list(nets.values())
    n_chunks = args.workers * 4
    bounds = np.linspace(0, len(obs), n_chunks + 1).astype(int)
    jobs = [(obs[a:b], err[a:b], dem[a:b], inten[a:b], [s[a:b] for s in starts], variants)
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    t1 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        solved = pool.map(solve_chunk, jobs)
    nonconv = {v: sum(st[v] for _, st in solved) for v in variants}
    supervised = answers.pop("supervised", None)
    for label, x in nets.items():
        answers[f"net_{label}"] = x
    answers["optimum"] = np.concatenate([o["base"] for o, _ in solved])
    answers["solver"] = np.concatenate([o["solver"] for o, _ in solved])
    if supervised is not None:
        answers["supervised"] = supervised
    for v in variants[1:]:
        answers[f"optimum_{v}"] = np.concatenate([o[v] for o, _ in solved])
    print(f"per-pixel solves in {time.time() - t1:.0f}s; not converged: {nonconv}", flush=True)

    sol_dem = answers["solver"] @ G["B"].T
    desc = {}
    for name, x in answers.items():
        v = name.split("optimum_", 1)[1] if name.startswith("optimum_") else "base"
        desc[name] = describe(x, obs, err, inten, cfg, variant=v)
    gap = {}
    opt_total = desc["optimum"]["total"]
    # The optimum started from every other answer, so it can never be worse.
    worse = {n: int(np.sum(d["total"] < opt_total - 1e-7 * np.abs(opt_total)))
             for n, d in desc.items() if not n.startswith("optimum")}
    print(f"answers scoring below the optimum (should be 0): {worse}", flush=True)
    for label in nets:
        rel = (desc[f"net_{label}"]["total"] - opt_total) / np.maximum(np.abs(opt_total), 1e-12)
        gap[label] = rel

    result = {"purpose": "network vs per-pixel optimum vs solver under the training objective",
              "loss": args.loss, "root": cfg["root"], "phase": args.phase,
              "models": dict(s.split("=", 1) for s in args.models),
              "supervised": args.supervised, "variants": variants,
              "settings": {k: cfg[k] for k in ("tolfac", "alpha_l1", "mu", "enet_alpha", "enet_lam")},
              "population_per_band": dict(zip(INTENSITY_LABELS, pop.tolist())),
              "not_converged": nonconv, "below_optimum": worse,
              "runtime_s": round(time.time() - t0, 1),
              "bands": {}}
    for g, lab in enumerate(INTENSITY_LABELS):
        idx = np.flatnonzero(group == g)
        band = {name: group_summary(d, sol_dem, obs, err, cfg, idx) for name, d in desc.items()}
        for label, rel in gap.items():
            if idx.size:
                band[f"net_{label}"]["gap_vs_optimum_median"] = float(np.median(rel[idx]))
                band[f"net_{label}"]["gap_vs_optimum_mean_abs"] = float(
                    (desc[f"net_{label}"]["total"][idx] - opt_total[idx]).mean())
                band[f"net_{label}"]["pct_above_optimum_by_5pct"] = float(100 * np.mean(rel[idx] > 0.05))
        result["bands"][lab] = band
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    print(render(result))
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as f:
        f.write(render(result) + "\n")


def render(r):
    L = [f"{r['loss']} objective; answers scored under the training objective "
         f"(tolfac {r['settings']['tolfac']}, a {r['settings']['alpha_l1']}, mu {r['settings']['mu']})",
         f"not converged: {r['not_converged']}"]
    names = list(next(b for b in r["bands"].values() if b.get("optimum")).keys())
    for lab, band in r["bands"].items():
        if not band.get("optimum"):
            continue
        L.append(f"\n== {lab}  (n={band['optimum']['n']}, population {r['population_per_band'][lab]:,})")
        L.append(f"{'answer':26s} {'objective':>11s} {'L1/reg':>9s} {'fit':>9s} {'94A rec':>8s} "
                 f"{'131A rec':>8s} {'EM/sol':>7s} {'DEM MSE':>10s} {'gap med':>8s}")
        for name in names:
            s = band.get(name)
            if not s:
                continue
            L.append(f"{name:26s} {s['objective_mean']:11.4g} {s['reg_mean']:9.4g} {s['fit_mean']:9.4g} "
                     f"{s['recon_over_obs']['94']:8.3f} {s['recon_over_obs']['131']:8.3f} "
                     f"{s['em_ratio_vs_solver']:7.3f} {s['dem_mse_vs_solver']:10.4g} "
                     f"{s.get('gap_vs_optimum_median', float('nan')):8.3f}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
