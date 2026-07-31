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

from fullBP import getBasis
from src.losses import barrier_loss_batch, enet_loss_batch
from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS, MIN_OBS
from experiments.train_neural_field import effective_sparsity, pick_device
from experiments.train_ablations import build_model, VARIANTS


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
    per_level = {1: {"n": 0, "sp_nn": 0.0, "sp_ref": 0.0, "mae": 0.0},
                 "relaxed": {"n": 0, "sp_nn": 0.0, "sp_ref": 0.0, "mae": 0.0}}

    for batch in loader:
        patch, obs, lb, ub, dem, tol = (t.to(device) for t in flatten_blocks(batch))
        x = model(patch)
        agg["loss"] += loss_fn(x, D_t, obs, lb, ub).item()
        agg["n_batches"] += 1

        # B is [n_temps, n_basis]; the stored label has 26 bins of which only the
        # first n_bins are real for AIA-only (the rest are stacked zeros).
        pred = (x @ B_t.T)[:, :n_bins]
        sp_nn = effective_sparsity(pred)
        sp_ref = effective_sparsity(dem)
        mae = (pred - dem).abs().mean(dim=1)

        for key, sel in ((1, tol == 1), ("relaxed", (tol == 3) | (tol == 5))):
            n = int(sel.sum())
            if n == 0:
                continue
            per_level[key]["n"] += n
            per_level[key]["sp_nn"] += float(sp_nn[sel].sum())
            per_level[key]["sp_ref"] += float(sp_ref[sel].sum())
            per_level[key]["mae"] += float(mae[sel].sum())

    out = {"loss": agg["loss"] / max(agg["n_batches"], 1)}
    for key, d in per_level.items():
        n = max(d["n"], 1)
        out[str(key)] = {"n": d["n"], "nn_sparsity": d["sp_nn"] / n,
                         "ref_sparsity": d["sp_ref"] / n, "mae": d["mae"] / n}
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
    model = build_model(args.variant, n_basis, args.patch_size, args.channels,
                        perm=perm).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = make_loss_fn(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))

    tag = f"scaled_{args.variant}_{args.loss}"
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
              f"sp(nn/ref @tol1)={tight['nn_sparsity']:.2f}/{tight['ref_sparsity']:.2f}  "
              f"mae={tight['mae']:.3f}  {rec['secs']:.0f}s")

        torch.save({"model": model.state_dict(), "variant": args.variant,
                    "loss": args.loss, "patch_size": args.patch_size,
                    "stride": args.stride, "channels": args.channels,
                    "n_bins": args.n_bins, "perm": perm, "epoch": epoch + 1,
                    "args": vars(args)},
                   os.path.join(out_dir, f"{tag}.pt"))
        with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nSaved {os.path.join(out_dir, tag)}.pt")
    final = history[-1]["val"]
    print(f"Final val — tol1: {final['1']}\n           relaxed: {final['relaxed']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="staged zarr dir (…_DS)")
    p.add_argument("--variant", choices=VARIANTS, default="mlp6")
    p.add_argument("--loss", choices=["barrier", "enet"], default="barrier")
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
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
