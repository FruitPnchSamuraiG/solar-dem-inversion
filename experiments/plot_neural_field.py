"""
Visualize per-pixel DEM curves from a trained patch-conditioned neural field
checkpoint (experiments/train_neural_field.py), overlaid against BP, for a
sample of pixels in the same crop the model was trained on.

Run from project root:
    uv run python experiments/plot_neural_field.py --data_dir ./data/20110906_2217 --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from fullBP import solveLP
from experiments.train_neural_field import AIAPatchDataset, PatchDEMNet, effective_sparsity


def main(args):
    tag = os.path.basename(args.data_dir.rstrip("/"))
    ckpt_path = args.ckpt or f"output/experiments/neural_field_{tag}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    D, B, logT = ckpt["D"], ckpt["B"], ckpt["logT"]

    model = PatchDEMNet(n_basis=B.shape[1], patch_size=ckpt["patch_size"],
                         channels=ckpt["channels"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loading AIA data (same crop used for training)...")
    dataset = AIAPatchDataset(args.data_dir, args.crop, patch_size=ckpt["patch_size"], tolfac=args.tolfac)

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(dataset.coords), size=min(args.n_pixels, len(dataset.coords)), replace=False)

    fig, axes = plt.subplots(args.n_pixels, 1, figsize=(8, 3.5 * args.n_pixels))
    if args.n_pixels == 1:
        axes = [axes]

    for pi, idx in enumerate(chosen):
        patch, obs, lb, ub = dataset[idx]
        ax = axes[pi]

        with torch.no_grad():
            x_nn = model(patch.unsqueeze(0))
        dem_nn = np.maximum(B @ x_nn.numpy()[0], 0)
        sparsity_nn = effective_sparsity(x_nn).item()
        mae_nn = np.mean(np.abs(D @ x_nn.numpy()[0] - obs.numpy()))
        ax.plot(logT, dem_nn, color='tab:cyan', lw=2.0, linestyle=':',
                label=f'neural field (NN)  MAE={mae_nn:.3f}  sparsity={sparsity_nn:.2f}', zorder=10)

        obs_np, ub_np = obs.numpy().astype(np.float64), ub.numpy().astype(np.float64)
        err = (ub_np - obs_np) / args.tolfac  # recover original per-channel sigma
        x_bp = None
        for tolfac in [1.4, 2.0, 2.8, 5.0]:
            lb_t, ub_t = obs_np - tolfac * err, obs_np + tolfac * err
            x_bp = solveLP((D.astype(np.float64), obs_np, lb_t, ub_t, None))
            if x_bp is not None:
                break
        if x_bp is not None:
            bp_dem = np.maximum(B @ x_bp, 0)
            bp_sparsity = (np.sum(np.abs(x_bp)) ** 2) / (np.sum(x_bp ** 2) + 1e-6)
            bp_mae = np.mean(np.abs(D @ x_bp - obs_np))
            ax.plot(logT, bp_dem, color='black', lw=2.5,
                    label=f'BP  MAE={bp_mae:.3f}  sparsity={bp_sparsity:.2f}', zorder=1)

        row, col = dataset.coords[idx]
        ax.set_title(f"Pixel ({row},{col})  171Å={obs[2]:.1f}", fontsize=9)
        ax.set_xlabel("log T", fontsize=8)
        ax.set_ylabel("DEM", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f"Neural field vs BP — {tag}", fontsize=11)
    plt.tight_layout()
    os.makedirs("output/experiments", exist_ok=True)
    out = f"output/experiments/neural_field_{tag}_pixel_dems.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/20110906_2217")
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--ckpt",       default=None)
    p.add_argument("--n_pixels",   type=int, default=10)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--tolfac",     type=float, default=1.4,
                    help="must match the tolfac used during training")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
