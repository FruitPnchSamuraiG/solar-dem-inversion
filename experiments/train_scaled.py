"""
Scaled unsupervised training on the staged 1,223-timestamp zarr.

Four planned runs: {mlp6, cnn} x {barrier, enet}. All four are AIA-only -- the
BP/ENet distinction lives entirely in the loss function, not in the inputs, and
training never reads the solver labels. The labels are used only at evaluation,
to report how far the network's answer sits from the solver's.

    python3 experiments/train_scaled.py --variant mlp6 --loss barrier \
        --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS

Notes on the metrics:
  * effective sparsity (Hoyer) is the primary number -- the whole point of the
    barrier loss is to reproduce BP's sparse solutions, and MAE-vs-solver was
    shown to be misleading (2026-07-02 ablation: the enet-loss model halves MAE
    while producing the least BP-like curves).
  * pixels are scored separately at tolLevel 1 vs 3/5. A level-3/5 label
    satisfied a band 3-5x wider than the nominal one, so agreement there is a
    weaker claim; pooling them hides that.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fullBP import getBasis
from src.losses import barrier_loss_batch, enet_loss_batch
from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS, MIN_OBS
from experiments.train_neural_field import effective_sparsity, pick_device
from experiments.train_ablations import build_model, VARIANTS


class ClampedSoftplus(nn.Module):
    """Softplus that cannot die.

    Every variant ends in nn.Softplus to enforce x >= 0. In float32 softplus(z)
    underflows to a denormal around z < -90, where its gradient is exactly 0 --
    an unrecoverable dead unit. Array 15088220 hit exactly that: a mean epoch-1
    loss of 1.2e10 followed by eleven epochs of a bit-identical 339.2775, i.e.
    zero gradient everywhere. Clamping the pre-activation keeps the output tiny
    (softplus(-20) ~ 2e-9, far below any DEM coefficient of interest) while
    leaving a finite gradient to climb back out on.
    """

    def __init__(self, floor=-20.0):
        super().__init__()
        self.floor = floor

    def forward(self, z):
        return F.softplus(z.clamp(min=self.floor))

    def extra_repr(self):
        return f"floor={self.floor}"


def harden_softplus(model, floor=-20.0):
    """Replace every nn.Softplus in `model` in place. Returns how many."""
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Softplus):
                setattr(mod, name, ClampedSoftplus(floor))
                n += 1
    return n


def init_output_head(model, bias=-3.0, weight_scale=0.01):
    """Start the network at a physically plausible DEM instead of a huge one.

    Every variant ends in Linear -> Softplus, and Softplus(0) = 0.693, so at
    default initialisation all 54 basis coefficients start near 0.7. With

        sum|D| per channel = [13.2, 117.6, 2541.4, 1478.7, 415.0, 33.5]

    that puts Dx for 171A at ~1780 against a typical ub of ~155 -- a 10x
    overshoot before the network has seen any data. barrier squares the excess
    and divides by sigma^2, which is small for the faint channels, giving the
    1.6e11 epoch-1 loss of array 15091457 and an immediate dead Softplus.

    Shrinking the final weights and setting a negative bias makes the initial
    output nearly constant at softplus(-3) ~ 0.049, i.e. Dx ~ 124 for 171A,
    comfortably inside the band. The network then grows away from that under
    warmup rather than having to climb back from saturation.

    log1p on the input (NormalizedInput) fixes conditioning; this fixes scale.
    Both are needed -- 15091457 had only the former and still collapsed.
    """
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is None:
        return None
    with torch.no_grad():
        last.weight.mul_(weight_scale)
        if last.bias is not None:
            last.bias.fill_(bias)
    return last


class NormalizedInput(nn.Module):
    """Wraps a variant so it sees a compressed version of the AIA patch.

    The architectures were validated on 128x128 on-disk crops, where the raw DN
    values span about one order of magnitude. Full-disk data spans roughly six:
    off-limb pixels sit at the MIN_OBS floor of 1e-3 while flare cores reach
    1e4. Feeding that straight into Conv2d/Linear makes the initial forward pass
    enormous for bright pixels, |Dx| overshoots ub by orders of magnitude, and
    the barrier's relu(Dx-ub)^2/sigma^2 term explodes.

    Only the network's *input representation* changes. The loss, lb/ub, D and
    every reported metric stay in physical units, so numbers remain directly
    comparable to the crop runs.
    """

    def __init__(self, inner, mode="log1p"):
        super().__init__()
        self.inner = inner
        self.mode = mode

    def forward(self, patch):
        if self.mode == "log1p":
            # patch is already floored at MIN_OBS > 0 by the dataloader's mask,
            # but clamp anyway so an unmasked caller cannot produce NaN.
            patch = torch.log1p(patch.clamp(min=0.0))
        return self.inner(patch)


def make_scheduler(optimizer, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine decay.

    Warmup exists to survive step 0: at initialisation the network's output is
    unrelated to the data, so the first few gradients are far larger than any
    seen later. Taking full-size Adam steps on them is what pushed the head into
    the dead Softplus region.
    """
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loss_fn(args):
    if args.loss == "barrier":
        return lambda x, D, obs, lb, ub: barrier_loss_batch(
            x, D, obs, lb, ub, a_l1=args.alpha_l1, a_l2=args.alpha_l2, mu=args.mu)
    if args.loss == "enet":
        return lambda x, D, obs, lb, ub: enet_loss_batch(
            x, D, obs, lb, ub, C=args.enet_C, alpha=args.enet_alpha, lam=args.enet_lam)
    raise ValueError(f"unknown loss: {args.loss}")


def load_operators(device, scale=10 ** 26):
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    return (torch.tensor(D).to(device), torch.tensor(B).to(device),
            D.shape[1], logT)


@torch.no_grad()
def evaluate(model, loader, D_t, B_t, loss_fn, device, n_bins):
    """Loss, sparsity and label agreement on held-out blocks, split by tolLevel."""
    model.eval()
    agg = {"loss": 0.0, "n_batches": 0}
    keys = ("sp_coef", "sp_dem", "sp_ref", "mae_dem", "mae_aia")
    per_level = {k: dict({"n": 0}, **{m: 0.0 for m in keys})
                 for k in (1, "relaxed")}

    for batch in loader:
        patch, obs, lb, ub, dem, tol = (t.to(device) for t in flatten_blocks(batch))
        x = model(patch)
        agg["loss"] += loss_fn(x, D_t, obs, lb, ub).item()
        agg["n_batches"] += 1

        # B is [n_temps, n_basis]; the stored label has 26 bins of which only the
        # first n_bins are real for AIA-only (the rest are stacked zeros).
        pred = (x @ B_t.T)[:, :n_bins]
        vals = {
            # Sparsity of the basis coefficients -- the quantity every earlier run
            # reported (ablation: BP 1.79, cnn 1.70, mlp6 1.89), so this is the one
            # that is comparable across the project's history.
            "sp_coef": effective_sparsity(x),
            # Sparsity in DEM-bin space, where the solver label also lives, so NN
            # and reference are measured on the same object. Not comparable to
            # sp_coef: 54 coefficients and 18 bins give different Hoyer scales.
            "sp_dem": effective_sparsity(pred),
            "sp_ref": effective_sparsity(dem),
            "mae_dem": (pred - dem).abs().mean(dim=1),
            # AIA resynthesis error, the "MAE" of the earlier runs.
            "mae_aia": (x @ D_t.T - obs).abs().mean(dim=1),
        }

        for key, sel in ((1, tol == 1), ("relaxed", (tol == 3) | (tol == 5))):
            n = int(sel.sum())
            if n == 0:
                continue
            per_level[key]["n"] += n
            for m in keys:
                per_level[key][m] += float(vals[m][sel].sum())

    out = {"loss": agg["loss"] / max(agg["n_batches"], 1)}
    for key, d in per_level.items():
        n = max(d["n"], 1)
        out[str(key)] = dict({"n": d["n"]}, **{m: d[m] / n for m in keys})
    return out


def train(args):
    device = pick_device()
    print(f"Device: {device}   Variant: {args.variant}   Loss: {args.loss}")
    print(f"Root: {args.root}")

    D_t, B_t, n_basis, logT = load_operators(device)
    print(f"D: {tuple(D_t.shape)}   basis: {n_basis}   temps: {len(logT)}")

    if args.enet_C < 0:
        args.enet_C = float(D_t.shape[0])          # n_obs, the solver's N
    if args.loss == "enet":
        print(f"ENet objective: C={args.enet_C} alpha={args.enet_alpha} "
              f"lam={args.enet_lam} (solver used alpha=1, l1_ratio=0.5)")
    if len(logT) < args.n_bins:
        # The label comparison needs one predicted value per scored bin. R's own
        # temperature grid is the ceiling; scoring past it is not meaningful.
        print(f"note: R covers {len(logT)} temperatures, scoring {len(logT)} bins "
              f"instead of {args.n_bins}")
        args.n_bins = len(logT)

    data_kw = dict(patch_size=args.patch_size, stride=args.stride, tolfac=args.tolfac,
                   pixels_per_block=args.pixels_per_block, n_bins=args.n_bins,
                   min_obs=args.min_obs)
    _, train_loader = make_loader(args.root, 'train', batch_blocks=args.batch_blocks,
                                  num_workers=args.num_workers, shuffle=True,
                                  max_blocks=args.max_train_blocks,
                                  with_labels=False, seed=args.seed, **data_kw)
    _, val_loader = make_loader(args.root, 'val', batch_blocks=args.batch_blocks,
                                num_workers=args.num_workers, shuffle=False,
                                max_blocks=args.max_val_blocks,
                                with_labels=True, seed=args.seed, **data_kw)

    perm = None
    if args.variant == "cnn_shuffled":
        perm = np.random.default_rng(args.seed).permutation(args.patch_size ** 2)
    core = build_model(args.variant, n_basis, args.patch_size, args.channels,
                       perm=perm, hidden=args.hidden)
    n_sp = harden_softplus(core, args.softplus_floor)
    init_output_head(core, args.init_bias, args.init_weight_scale)
    model = NormalizedInput(core, args.input_transform).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}  "
          f"input={args.input_transform}  softplus floor={args.softplus_floor} "
          f"({n_sp} replaced)  init bias={args.init_bias} "
          f"wscale={args.init_weight_scale}")

    # The first forward pass is what killed 15088220 and 15091457, so state it
    # plainly in the log rather than inferring it from the epoch-1 loss.
    with torch.no_grad():
        probe = next(iter(val_loader))
        p_patch = flatten_blocks(probe)[0][:4096].to(device)
        x0 = model(p_patch)
        Dx0 = x0 @ D_t.T
    print(f"init check: x med={x0.median():.4f} max={x0.max():.4f}  "
          f"|Dx| med={Dx0.abs().median():.3e} max={Dx0.abs().max():.3e}  "
          f"(want |Dx| comparable to obs, not orders above)")

    loss_fn = make_loss_fn(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps, args.lr)
    print(f"{len(train_loader):,} steps/epoch, {total_steps:,} total, "
          f"{args.warmup_steps:,} warmup")

    # The width suffix keeps a sweep's checkpoints from overwriting the 1.5M
    # baseline, whose tag stays exactly what array 15092854 wrote.
    tag = f"scaled_{args.variant}_{args.loss}"
    if args.hidden is not None:
        tag += f"_h{args.hidden}"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss, ep_sp, n = 0.0, 0.0, 0

        for batch in train_loader:
            patch, obs, lb, ub = (t.to(device, non_blocking=True)
                                  for t in flatten_blocks(batch))
            optimizer.zero_grad()
            x = model(patch)
            loss = loss_fn(x, D_t, obs, lb, ub)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            with torch.no_grad():
                ep_sp += effective_sparsity(x).mean().item()
            n += 1

        val = evaluate(model, val_loader, D_t, B_t, loss_fn, device, args.n_bins)
        rec = {"epoch": epoch + 1, "train_loss": ep_loss / n,
               "train_sparsity": ep_sp / n, "val": val, "secs": time.time() - t0}
        history.append(rec)
        tight = val["1"]
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={rec['train_loss']:.4f}  "
              f"val_loss={val['loss']:.4f}  "
              f"sp_coef={tight['sp_coef']:.2f} (hist: BP 1.79)  "
              f"sp_dem nn/ref={tight['sp_dem']:.2f}/{tight['sp_ref']:.2f}  "
              f"mae_aia={tight['mae_aia']:.3f}  {rec['secs']:.0f}s")

        # Save the bare variant's weights (not the NormalizedInput wrapper) so
        # existing eval code loads them unchanged; input_transform records the
        # representation those weights expect.
        torch.save({"model": core.state_dict(), "variant": args.variant,
                    "loss": args.loss, "patch_size": args.patch_size,
                    "stride": args.stride, "channels": args.channels,
                    "n_bins": args.n_bins, "perm": perm, "epoch": epoch + 1,
                    "hidden": args.hidden,
                    "input_transform": args.input_transform,
                    "softplus_floor": args.softplus_floor,
                    "args": vars(args)},
                   os.path.join(out_dir, f"{tag}.pt"))
        with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Fail loudly on the 15088220 failure mode rather than logging eleven
        # more identical epochs: an all-zero prediction has zero Hoyer sparsity,
        # and a bit-identical loss means the weights have stopped moving (block
        # sampling is deterministic, so epochs differ only in order).
        # Zero Hoyer sparsity means an identically-zero prediction. Check the
        # validation value too: 15091457 slipped past a train-only check for a
        # full epoch because a few surviving units kept the train mean just
        # above 0 while every reported metric was already dead.
        if rec["train_sparsity"] < 1e-3 or tight["sp_coef"] < 1e-3:
            raise RuntimeError(
                f"collapsed at epoch {epoch+1}: prediction is identically zero "
                f"(train sp {rec['train_sparsity']:.3e}, val sp "
                f"{tight['sp_coef']:.3e}). Check the init-check line above: if "
                f"|Dx| starts orders above the observations, the output head is "
                f"initialised too large.")
        # Relative, not exact: run 15091734_3 froze at 32.8399 but the values
        # differed past the printed digits, so an == comparison never fired.
        prev = history[-2]["train_loss"] if len(history) > 1 else None
        if prev is not None and abs(rec["train_loss"] - prev) <= 1e-9 * abs(prev):
            raise RuntimeError(
                f"collapsed at epoch {epoch+1}: train loss is bit-identical to "
                f"the previous epoch, so gradients are exactly zero.")

    print(f"\nSaved {os.path.join(out_dir, tag)}.pt")
    final = history[-1]["val"]
    print(f"Final val — tol1: {final['1']}\n           relaxed: {final['relaxed']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="staged zarr dir (…_DS)")
    p.add_argument("--variant", choices=VARIANTS, default="mlp6")
    p.add_argument("--loss", choices=["barrier", "enet"], default="barrier")
    p.add_argument("--hidden", type=int, default=None,
                   help="mlp6 hidden width; None = the 1.5M capacity-matched 680")
    p.add_argument("--out_dir", default="output/experiments")
    # data
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--stride", type=int, default=2,
                   help="AIA stride per patch step; 2 keeps the DEM's own footprint")
    p.add_argument("--pixels_per_block", type=int, default=512)
    p.add_argument("--batch_blocks", type=int, default=8)
    p.add_argument("--n_bins", type=int, default=N_AIA_BINS)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--min_obs", type=float, default=MIN_OBS,
                   help="drop pixels with any channel below this (deconvolution clamp)")
    p.add_argument("--max_train_blocks", type=int, default=None)
    p.add_argument("--max_val_blocks", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    # model / optim
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input_transform", choices=["none", "log1p"], default="log1p",
                   help="'none' reproduces the crop runs; full-disk data spans ~6 "
                        "decades and needs compressing (see NormalizedInput)")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="linear LR warmup; 0 disables")
    p.add_argument("--softplus_floor", type=float, default=-20.0,
                   help="clamp on the output pre-activation, guards against dead units")
    p.add_argument("--init_bias", type=float, default=-3.0,
                   help="output-layer bias; softplus(-3)~0.049 keeps initial Dx in-band")
    p.add_argument("--init_weight_scale", type=float, default=0.01,
                   help="shrink factor on the output layer's initial weights")
    # barrier
    p.add_argument("--alpha_l1", type=float, default=1.0)
    p.add_argument("--alpha_l2", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=1.0)
    # ENet defaults mirror the solver that produced the ENet labels, so the network
    # minimises the same objective it is scored against. fullBP's solveElasticNet is
    #   1/(2N)||Dx-y||^2 + a*l*||x||_1 + a(1-l)*0.5*||x||^2,  a=1, l=0.5, N=n_obs,
    # and it scales D and y by tol = meas - lb, i.e. our sigma. enet_loss_batch's
    # (0.5/C) prefactor therefore needs C = n_obs; --enet_C -1 resolves it from D.
    p.add_argument("--enet_C", type=float, default=-1.0,
                   help="-1 = use n_obs, matching the solver's 1/(2N) prefactor")
    p.add_argument("--enet_alpha", type=float, default=1.0,   # --fitlinearalpha
                   help="matches fullBP --fitlinearalpha")
    p.add_argument("--enet_lam", type=float, default=0.5,     # --fitlinearl1ratio
                   help="matches fullBP --fitlinearl1ratio")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
