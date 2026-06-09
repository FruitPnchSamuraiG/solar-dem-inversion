"""
Load saved .npy DEM files and print MAE vs observed + MAE vs BP table.
Run from project root after run_all_losses.py has completed.

    uv run python experiments/print_mae_table.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from types import SimpleNamespace
from fullBP import getBasis

LOSS_NAMES = ['BP', 'barrier', 'barrier_fit', 'chisq_smooth', 'entropy', 'tikhonov']

DATA_DIRS = [
    "./data/20110906_2217",
    "./data/20120603_0000",
    "./data/20131113_0908",
    "./data/20140910_1731",
]

def main():
    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R_scaled = (R * scale).astype(np.float32)
    n_temps = len(logT)

    from src.utils import processIndAIAData

    for data_dir in DATA_DIRS:
        tag = os.path.basename(data_dir)
        out_dir = f"output/experiments/run_all_losses/{tag}"
        if not os.path.exists(out_dir):
            print(f"Skipping {tag} — no output dir")
            continue

        data_args = SimpleNamespace(crop="1800,1800,256,256", corr_table="aia_corr.csv", pointing_file="")
        aia_cube, aia_errors, _ = processIndAIAData(data_dir, args=data_args)
        C, H, W = aia_cube.shape
        obs_all = aia_cube.reshape(C, -1).T
        err_all = aia_errors.reshape(C, -1).T
        valid = np.all(np.isfinite(obs_all) & np.isfinite(err_all) & (obs_all > 0), axis=1)
        obs_flat = aia_cube.reshape(C, -1)[:, valid]

        dems = {}
        for name in LOSS_NAMES:
            fname = os.path.join(out_dir, f"{'bp' if name == 'BP' else name}_dem.npy")
            if os.path.exists(fname):
                dems[name] = np.load(fname)
            else:
                print(f"  Missing: {fname}")

        if not dems:
            print(f"No .npy files found for {tag}")
            continue

        bp_dem = dems.get('BP')

        print(f"\n{'='*60}")
        print(f"Timestamp: {tag}  ({valid.sum():,} valid pixels)")
        print(f"  {'Loss':<20} {'MAE vs observed':>16} {'MAE vs BP':>12}")
        print(f"  {'-'*50}")
        for name, dem in dems.items():
            flat = np.maximum(dem.reshape(n_temps, -1)[:, valid], 0)
            rs = R_scaled @ flat
            mae_obs = np.nanmean(np.abs(rs - obs_flat))
            if bp_dem is not None and name != 'BP':
                bp_flat = np.maximum(bp_dem.reshape(n_temps, -1)[:, valid], 0)
                mae_bp = np.nanmean(np.abs(flat - bp_flat))
            else:
                mae_bp = 0.0
            print(f"  {name:<20} {mae_obs:>16.4f} {mae_bp:>12.4f}")

if __name__ == "__main__":
    main()
