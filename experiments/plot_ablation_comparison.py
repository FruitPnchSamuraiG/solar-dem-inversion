"""
Plot per-pixel DEM curves for all ablation variants vs BP on the same axes,
one PNG per timestamp. Also prints a combined sparsity/MAE table.

Run from project root after training the ablation variants:
    uv run python experiments/plot_ablation_comparison.py \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from fullBP import getBasis, solveLP
from experiments.train_neural_field import (
    AIAPatchDataset, effective_sparsity, pick_device,
)
from experiments.train_neural_field_amortized import split_dataset, evaluate_val
from experiments.train_ablations import load_ablation_model

ACTIVITY = {
    "20110906_2217": "X2.1 flare (AR 11283)",
    "20120603_0000": "Quiet sun",
    "20131113_0908": "Moderate activity",
    "20140910_1731": "X1.6 flare (AR 12158)",
}

RUNS = [
    ("cnn",          "output/experiments/ablation_cnn_barrier.pt",          "tab:blue"),
    ("mlp6",         "output/experiments/ablation_mlp6_barrier.pt",         "tab:red"),
    ("mlp_patch",    "output/experiments/ablation_mlp_patch_barrier.pt",    "tab:green"),
    ("cnn_shuffled", "output/experiments/ablation_cnn_shuffled_barrier.pt", "tab:orange"),
    ("cnn (enet)",   "output/experiments/ablation_cnn_enet.pt",             "tab:purple"),
]


def main(args):
    device = pick_device()

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)

    print("Loading models...")
    models = []
    for label, ckpt_path, color in RUNS:
        if not os.path.exists(ckpt_path):
            print(f"  missing: {ckpt_path}, skipping")
            continue
        model, _ = load_ablation_model(ckpt_path, D.shape[1], device)
        models.append((label, model, color))

    os.makedirs("output/experiments", exist_ok=True)

    # ── combined val table ────────────────────────────────────────────────────
    val_subsets = {}
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        ds = AIAPatchDataset(data_dir, args.crop, patch_size=9, tolfac=args.tolfac)
        _, val_ds = split_dataset(ds, args.val_frac, args.seed)
        val_subsets[tag] = val_ds

    all_results = {label: evaluate_val(m, val_subsets, D, device, seed=args.seed)
                   for label, m, _ in models}

    col = 13
    print(f"\nSparsity (lower = sparser = closer to BP):")
    print(f"{'Timestamp':<22} " + "  ".join(f"{l:>{col}}" for l, _, _ in models))
    for tag in val_subsets:
        row = f"{tag:<22} "
        for label, _, _ in models:
            row += f"{all_results[label][tag]['nn_sparsity']:>{col}.2f}  "
        print(row)

    print(f"\nMAE:")
    print(f"{'Timestamp':<22} " + "  ".join(f"{l:>{col}}" for l, _, _ in models))
    for tag in val_subsets:
        row = f"{tag:<22} "
        for label, _, _ in models:
            row += f"{all_results[label][tag]['nn_mae']:>{col}.4f}  "
        print(row)

    # ── per-pixel curve plots ─────────────────────────────────────────────────
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        print(f"\nPlotting {tag}...")
        val_ds = val_subsets[tag]

        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(val_ds), size=min(args.n_pixels, len(val_ds)), replace=False)

        fig, axes = plt.subplots(args.n_pixels, 1, figsize=(9, 3.5 * args.n_pixels))
        if args.n_pixels == 1:
            axes = [axes]

        for pi, idx in enumerate(chosen):
            patch, obs, lb, ub = val_ds[int(idx)]
            ax = axes[pi]
            row, col_ = val_ds.dataset.coords[val_ds.indices[int(idx)]]

            with torch.no_grad():
                for label, model, color in models:
                    x = model(patch.unsqueeze(0).to(device))
                    dem = np.maximum(B @ x.cpu().numpy()[0], 0)
                    sp = effective_sparsity(x).item()
                    ax.plot(logT, dem, color=color, lw=1.5, linestyle="--",
                            label=f"{label}  sp={sp:.2f}", alpha=0.85)

            obs_np = obs.numpy().astype(np.float64)
            ub_np  = ub.numpy().astype(np.float64)
            err    = (ub_np - obs_np) / args.tolfac
            x_bp   = None
            for tolfac in [1.4, 2.0, 2.8, 5.0]:
                lb_t = obs_np - tolfac * err
                ub_t = obs_np + tolfac * err
                x_bp = solveLP((D.astype(np.float64), obs_np, lb_t, ub_t, None))
                if x_bp is not None:
                    break
            if x_bp is not None:
                bp_dem = np.maximum(B @ x_bp, 0)
                bp_sp  = (np.sum(np.abs(x_bp))**2) / (np.sum(x_bp**2) + 1e-6)
                ax.plot(logT, bp_dem, color="black", lw=2.5,
                        label=f"BP  sp={bp_sp:.2f}", zorder=10)

            ax.set_title(f"Pixel ({row},{col_})  171Å={obs[2]:.1f}", fontsize=8)
            ax.set_xlabel("log T", fontsize=7)
            ax.set_ylabel("DEM", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc="upper right", ncol=2)

        plt.suptitle(f"Ablation comparison vs BP — {tag} ({ACTIVITY.get(tag,'')})", fontsize=11)
        plt.tight_layout()
        out = f"output/experiments/ablation_comparison_{tag}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--crop",      default="1800,1800,128,128")
    p.add_argument("--n_pixels",  type=int,   default=10)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--tolfac",    type=float, default=1.4)
    p.add_argument("--val_frac",  type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
