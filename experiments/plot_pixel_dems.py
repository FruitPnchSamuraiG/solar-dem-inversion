"""
For each timestamp, load saved .npy DEM files and plot per-pixel DEM curves
for 10 randomly sampled pixels, all loss formulations overlaid.

Run from project root:
    uv run python experiments/plot_pixel_dems.py
    uv run python experiments/plot_pixel_dems.py --n_pixels 10 --seed 42
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

LOSS_NAMES = ['barrier', 'barrier_fit', 'chisq_smooth', 'entropy', 'tikhonov']
COLORS     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

DATA_DIRS = [
    "./data/20110906_2217",
    "./data/20120603_0000",
    "./data/20131113_0908",
    "./data/20140910_1731",
]


def main(args):
    from src.utils import processIndAIAData
    from fullBP import getBasis

    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    scale = 10 ** 26
    R_scaled = (R * scale).astype(np.float32)
    n_temps = len(logT)

    out_dir = "output/experiments/pixel_dems"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for data_dir in DATA_DIRS:
        tag = os.path.basename(data_dir)
        npy_dir = f"output/experiments/run_all_losses/{tag}"
        if not os.path.exists(npy_dir):
            print(f"Skipping {tag} — no saved DEMs")
            continue

        # load AIA for observations
        data_args = SimpleNamespace(crop="1800,1800,256,256", corr_table="aia_corr.csv", pointing_file="")
        aia_cube, aia_errors, _ = processIndAIAData(data_dir, args=data_args)
        C, H, W = aia_cube.shape
        obs_all = aia_cube.reshape(C, -1).T
        err_all = aia_errors.reshape(C, -1).T
        valid_mask = np.all(np.isfinite(obs_all) & np.isfinite(err_all) & (obs_all > 0), axis=1)
        valid_idx = np.where(valid_mask)[0]

        # load saved DEMs
        dems = {}
        bp_path = os.path.join(npy_dir, "bp_dem.npy")
        if os.path.exists(bp_path):
            dems['BP'] = np.load(bp_path).reshape(n_temps, -1)
        for name in LOSS_NAMES:
            path = os.path.join(npy_dir, f"{name}_dem.npy")
            if os.path.exists(path):
                dems[name] = np.load(path).reshape(n_temps, -1)

        if 'BP' not in dems:
            print(f"Skipping {tag} — no BP .npy")
            continue

        # pick pixels
        chosen = rng.choice(valid_idx, size=min(args.n_pixels, len(valid_idx)), replace=False)

        fig, axes = plt.subplots(args.n_pixels, 1, figsize=(9, 3.5 * args.n_pixels))
        if args.n_pixels == 1:
            axes = [axes]

        for pi, pidx in enumerate(chosen):
            obs = obs_all[pidx]
            ax = axes[pi]

            # BP reference
            bp_dem = np.maximum(dems['BP'][:, pidx], 0)
            bp_resynth = R_scaled @ bp_dem
            bp_mae = np.mean(np.abs(bp_resynth - obs))
            ax.plot(logT, bp_dem, color='black', lw=2.5, label=f'BP  MAE={bp_mae:.3f}', zorder=1)

            # each differentiable loss
            for li, name in enumerate(LOSS_NAMES):
                if name not in dems:
                    continue
                dem = np.maximum(dems[name][:, pidx], 0)
                resynth = R_scaled @ dem
                mae = np.mean(np.abs(resynth - obs))
                ax.plot(logT, dem, color=COLORS[li], lw=1.8,
                        label=f'{name}  MAE={mae:.3f}', zorder=2 + li)

            ax.set_ylabel("DEM", fontsize=8)
            ax.set_xlabel("log T", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper right')
            row, col = divmod(pidx, W)
            ax.set_title(f"Pixel ({row},{col})  171Å={obs[2]:.1f}", fontsize=9)

        plt.suptitle(f"Per-pixel DEMs — {tag}", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{tag}_pixel_dems.png")
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pixels", type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
