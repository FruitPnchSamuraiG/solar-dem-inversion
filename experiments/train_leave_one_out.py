"""
Leave-one-timestamp-out (LOO) evaluation: the honest generalization test.

Per the 2026-07-10 meeting, the two model families going forward are the patch
CNN (`cnn`) and the capacity-matched center-pixel MLP (`mlp6`), and the open
question is whether the ~1.5M-param models overfit 4 images. All previous
validation held out *pixels* from the training images; this script holds out an
entire *timestamp*: for each fold, train on the other 3 timestamps and evaluate
on the never-seen one.

For each (fold, variant) it reports, on the held-out timestamp:
    - NN effective sparsity vs the BP reference (primary metric)
    - NN MAE (secondary; known to reward broad curves)
and, for contrast, the same metrics on the 3 training timestamps' held-out
val pixels — the gap between "unseen pixels of seen images" and "unseen image"
is the generalization penalty.

Also saves a per-fold pixel-curve plot (BP vs cnn vs mlp6 on held-out pixels)
and a JSON summary.

Run from project root (crunchy1 — needs all 4 timestamps under ./data):
    uv run python experiments/train_leave_one_out.py \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch
from experiments.train_neural_field import (
    AIAPatchDataset, effective_sparsity, pick_device, bp_sparsity_reference,
)
from experiments.train_neural_field_amortized import split_dataset, evaluate_val
from experiments.train_ablations import build_model

VARIANT_COLORS = {"cnn": "tab:blue", "mlp6": "tab:red"}


def train_fold(variant, train_subsets, D_t, args, device, n_basis):
    """Train one variant on the pooled train subsets (3 timestamps)."""
    full_train = ConcatDataset(train_subsets)
    loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    perm = None
    if variant == "cnn_shuffled":
        perm = np.random.default_rng(args.seed).permutation(args.patch_size ** 2)
    model = build_model(variant, n_basis, args.patch_size, args.channels, perm=perm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {variant}: {n_params:,} params, {len(full_train):,} train pixels")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_sparsity = 0.0, 0.0
        for patch, obs, lb, ub in loader:
            patch, obs, lb, ub = (patch.to(device), obs.to(device),
                                  lb.to(device), ub.to(device))
            optimizer.zero_grad()
            x = model(patch)
            loss = barrier_loss_batch(x, D_t, obs, lb, ub,
                                      a_l1=args.alpha_l1, a_l2=args.alpha_l2, mu=args.mu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_sparsity += effective_sparsity(x).mean().item()
        print(f"    epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss/len(loader):.4f}  "
              f"eff_sparsity={epoch_sparsity/len(loader):.2f}")
    return model


def plot_heldout_pixels(fold_tag, models, heldout_ds, D, B, logT, args):
    """BP vs both variants on random pixels of the held-out timestamp."""
    device = next(iter(models.values())).parameters().__next__().device
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(heldout_ds), size=min(args.n_pixels, len(heldout_ds)),
                        replace=False)

    fig, axes = plt.subplots(args.n_pixels, 1, figsize=(9, 3.5 * args.n_pixels))
    if args.n_pixels == 1:
        axes = [axes]

    for pi, idx in enumerate(chosen):
        patch, obs, lb, ub = heldout_ds[int(idx)]
        ax = axes[pi]
        row, col = heldout_ds.coords[int(idx)]

        with torch.no_grad():
            for variant, model in models.items():
                x = model(patch.unsqueeze(0).to(device))
                dem = np.maximum(B @ x.cpu().numpy()[0], 0)
                sp = effective_sparsity(x).item()
                ax.plot(logT, dem, color=VARIANT_COLORS[variant], lw=1.5, linestyle="--",
                        label=f"{variant}  sp={sp:.2f}", alpha=0.85)

        obs_np = obs.numpy().astype(np.float64)
        err = (ub.numpy().astype(np.float64) - obs_np) / args.tolfac
        x_bp = None
        for tolfac in [1.4, 2.0, 2.8, 5.0]:
            x_bp = solveLP((D.astype(np.float64), obs_np,
                            obs_np - tolfac * err, obs_np + tolfac * err, None))
            if x_bp is not None:
                break
        if x_bp is not None:
            bp_dem = np.maximum(B @ x_bp, 0)
            bp_sp = (np.sum(np.abs(x_bp)) ** 2) / (np.sum(x_bp ** 2) + 1e-6)
            ax.plot(logT, bp_dem, color="black", lw=2.5, label=f"BP  sp={bp_sp:.2f}", zorder=10)

        ax.set_title(f"Pixel ({row},{col})  171Å={obs[2]:.1f}", fontsize=8)
        ax.set_xlabel("log T", fontsize=7)
        ax.set_ylabel("DEM", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right")

    plt.suptitle(f"Leave-one-out: held-out {fold_tag} (never seen in training)", fontsize=11)
    plt.tight_layout()
    out = f"output/experiments/loo_heldout_{fold_tag}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def main(args):
    device = pick_device()
    print(f"Device: {device}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    D_t = torch.tensor(D).to(device)

    # load all timestamps once; per-fold we regroup them
    datasets, bp_refs = {}, {}
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        print(f"\nLoading {tag}...")
        datasets[tag] = AIAPatchDataset(data_dir, args.crop,
                                        patch_size=args.patch_size, tolfac=args.tolfac)
        if args.bp_compare > 0:
            bp_refs[tag] = bp_sparsity_reference(datasets[tag], D,
                                                 n_pixels=args.bp_compare, seed=args.seed,
                                                 n_jobs=args.bp_jobs)
            print(f"  BP sparsity reference: {bp_refs[tag]:.2f}")

    os.makedirs("output/experiments", exist_ok=True)
    summary = {}

    for heldout_tag in datasets:
        print(f"\n{'='*72}\nFOLD: hold out {heldout_tag}, train on the other "
              f"{len(datasets)-1}\n{'='*72}")

        train_subsets, insample_val = [], {}
        for tag, ds in datasets.items():
            if tag == heldout_tag:
                continue
            train_ds, val_ds = split_dataset(ds, args.val_frac, args.seed)
            train_subsets.append(train_ds)
            insample_val[tag] = val_ds

        fold_models = {}
        for variant in args.variants:
            print(f"\nTraining {variant} (fold: -{heldout_tag})")
            model = train_fold(variant, train_subsets, D_t, args, device, D.shape[1])
            fold_models[variant] = model

            ckpt_path = f"output/experiments/loo_{variant}_holdout_{heldout_tag}.pt"
            torch.save({"model": model.state_dict(), "variant": variant,
                        "heldout": heldout_tag, "D": D, "B": B, "logT": logT,
                        "patch_size": args.patch_size, "channels": args.channels,
                        "perm": None}, ckpt_path)
            print(f"  saved: {ckpt_path}")

            # eval: never-seen image vs held-out pixels of seen images
            heldout_res = evaluate_val(model, {heldout_tag: datasets[heldout_tag]},
                                       D, device, seed=args.seed)[heldout_tag]
            insample_res = evaluate_val(model, insample_val, D, device, seed=args.seed)
            insample_sp = float(np.mean([r["nn_sparsity"] for r in insample_res.values()]))
            insample_mae = float(np.mean([r["nn_mae"] for r in insample_res.values()]))

            summary.setdefault(heldout_tag, {})[variant] = {
                "heldout_sparsity": heldout_res["nn_sparsity"],
                "heldout_mae":      heldout_res["nn_mae"],
                "insample_sparsity": insample_sp,
                "insample_mae":      insample_mae,
                "bp_sparsity_heldout": bp_refs.get(heldout_tag, float("nan")),
            }

        plot_heldout_pixels(heldout_tag, fold_models, datasets[heldout_tag],
                            D, B, logT, args)

    # ── final table ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("LEAVE-ONE-TIMESTAMP-OUT SUMMARY  (sparsity: lower = closer to BP; "
          "gap = heldout - insample)")
    print(f"{'='*100}")
    hdr = (f"{'Held-out ts':<18} {'Variant':<8} {'BP sp':>7} {'heldout sp':>11} "
           f"{'insample sp':>12} {'sp gap':>8} {'heldout MAE':>12} {'insample MAE':>13}")
    print(hdr)
    print("-" * len(hdr))
    for tag, variants in summary.items():
        for variant, r in variants.items():
            gap = r["heldout_sparsity"] - r["insample_sparsity"]
            print(f"{tag:<18} {variant:<8} {r['bp_sparsity_heldout']:>7.2f} "
                  f"{r['heldout_sparsity']:>11.2f} {r['insample_sparsity']:>12.2f} "
                  f"{gap:>+8.2f} {r['heldout_mae']:>12.4f} {r['insample_mae']:>13.4f}")

    out_json = "output/experiments/loo_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs",  nargs="+", required=True)
    p.add_argument("--variants",   nargs="+", default=["cnn", "mlp6"],
                   help="the two model families going forward (2026-07-10 meeting)")
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--patch_size", type=int,   default=9)
    p.add_argument("--channels",   type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--alpha_l1",   type=float, default=1.0)
    p.add_argument("--alpha_l2",   type=float, default=0.0)
    p.add_argument("--mu",         type=float, default=1.0)
    p.add_argument("--bp_compare", type=int,   default=100)
    p.add_argument("--bp_jobs",    type=int,   default=-1,
                   help="processes for BP reference solves (0=serial, -1=all cores)")
    p.add_argument("--num_workers", type=int,  default=4,
                   help="DataLoader workers; match --cpus-per-task on SLURM")
    p.add_argument("--n_pixels",   type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
