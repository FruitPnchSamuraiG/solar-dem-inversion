"""
Run all loss formulations on all pixels of all timestamps.

For each timestamp + each loss: optimizes DEM per pixel in parallel,
saves DEM maps, produces mean-logT and resynth-diff visualizations.

Run from project root:
    uv run python experiments/run_all_losses.py
    uv run python experiments/run_all_losses.py --crop 1800,1800,256,256 --workers 64
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from types import SimpleNamespace
from multiprocessing import Pool
from functools import partial

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch


# ── loss formulations (same as loss_comparison.py) ───────────────────────────

def loss_barrier(x, D, obs, lb, ub, **kw):
    return barrier_loss_batch(x, D, obs, lb, ub, a_l1=1.0, a_l2=0.0,
                              mu=kw.get('mu', 1.0), alpha=0.0)

def loss_barrier_fit(x, D, obs, lb, ub, **kw):
    return barrier_loss_batch(x, D, obs, lb, ub, a_l1=1.0, a_l2=0.0,
                              mu=kw.get('mu', 1.0), alpha=0.1)

def loss_chisq_smooth(x, D, obs, lb, ub, **kw):
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8
    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()
    diff = x[:, 1:] - x[:, :-1]
    smooth = 0.1 * torch.sum(diff ** 2, dim=1).mean()
    pos = 10.0 * torch.sum(torch.relu(-x) ** 2, dim=1).mean()
    return fit + smooth + pos

def loss_entropy_fit(x, D, obs, lb, ub, **kw):
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8
    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()
    x_pos = x + 1e-8
    p = x_pos / x_pos.sum(dim=1, keepdim=True)
    entropy = -torch.sum(p * torch.log(p), dim=1).mean()
    return fit - 0.5 * entropy

def loss_tikhonov(x, D, obs, lb, ub, **kw):
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8
    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()
    l2  = 0.1 * torch.sum(x ** 2, dim=1).mean()
    pos = 10.0 * torch.sum(torch.relu(-x) ** 2, dim=1).mean()
    return fit + l2 + pos

LOSSES = {
    'barrier':        loss_barrier,
    'barrier_fit':    loss_barrier_fit,
    'chisq_smooth':   loss_chisq_smooth,
    'entropy':        loss_entropy_fit,
    'tikhonov':       loss_tikhonov,
}


# ── per-pixel optimizer ───────────────────────────────────────────────────────

def optimize_pixel(args_tuple):
    """Optimize one pixel with one loss — runs in a worker process."""
    obs_np, err_np, D_np, tolfac, loss_name, steps = args_tuple

    obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    D   = torch.tensor(D_np,   dtype=torch.float32)
    lb  = torch.tensor(obs_np - tolfac * err_np, dtype=torch.float32).unsqueeze(0)
    ub  = torch.tensor(obs_np + tolfac * err_np, dtype=torch.float32).unsqueeze(0)

    D_pinv = torch.linalg.pinv(D)
    x0 = torch.relu(obs @ D_pinv.T)
    x  = nn.Parameter(x0.clone())

    loss_fn = LOSSES[loss_name]
    opt = torch.optim.LBFGS([x], lr=0.01, max_iter=50,
                             line_search_fn='strong_wolfe',
                             tolerance_change=1e-12, tolerance_grad=1e-32)

    for _ in range(steps // 50):
        def closure():
            opt.zero_grad()
            loss = loss_fn(torch.relu(x), D, obs, lb, ub)
            loss.backward()
            return loss
        opt.step(closure)

    return torch.relu(x).detach().numpy()[0]  # [n_basis]


def run_bp_pixel(args_tuple):
    """Run BP solver on one pixel."""
    obs_np, err_np, D_np, B_np = args_tuple
    for tolfac in [1.4, 2.0, 2.8, 5.0]:
        lb = obs_np - tolfac * err_np
        ub = obs_np + tolfac * err_np
        x = solveLP((D_np, obs_np, lb, ub, None))
        if x is not None:
            return B_np @ x
    return np.zeros(B_np.shape[0])


# ── visualize ────────────────────────────────────────────────────────────────

def visualize_results(results, aia_cube, logT, R, out_dir, tag):
    """Save mean-logT maps and resynth-diff for all losses + BP."""
    C, H, W = aia_cube.shape
    n_temps = len(logT)
    wavelengths = [94, 131, 171, 193, 211, 335]
    os.makedirs(out_dir, exist_ok=True)

    def mean_logt(dem):
        d = np.maximum(dem, 0)
        total = d.sum(axis=0, keepdims=True) + 1e-8
        return (logT.reshape(-1, 1, 1) * d / total).sum(axis=0)

    def resynth_mae(dem):
        flat = np.maximum(dem.reshape(n_temps, -1), 0)
        rs = (R @ flat).reshape(C, H, W)
        return np.nanmean(np.abs(rs - aia_cube)), rs

    names = list(results.keys())

    # ── mean logT maps ────────────────────────────────────────────────────────
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    for i, name in enumerate(names):
        dem = results[name]
        im = axes[i].imshow(mean_logt(dem), cmap='inferno', vmin=5.5, vmax=7.5)
        mae, _ = resynth_mae(dem)
        axes[i].set_title(f"{name}\nMAE={mae:.3f}", fontsize=9)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    plt.suptitle(f"Mean logT — {tag}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_mean_logt.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # ── resynth diff ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, ncols, figsize=(3.5 * ncols, 4 * 6))
    if ncols == 1:
        axes = axes[:, np.newaxis]
    for i, name in enumerate(names):
        _, rs = resynth_mae(results[name])
        for ch in range(6):
            vrange = np.nanmax(np.abs(aia_cube[ch])) * 0.3
            diff = rs[ch] - aia_cube[ch]
            axes[ch, i].imshow(diff, cmap='bwr', vmin=-vrange, vmax=vrange)
            axes[ch, i].set_title(f"{name}\n{wavelengths[ch]}Å diff", fontsize=7)
            axes[ch, i].axis('off')
    plt.suptitle(f"Resynth diff — {tag}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_resynth_diff.png"), dpi=100, bbox_inches='tight')
    plt.close()

    # print similarity table
    bp_dem = results.get('BP')
    print(f"\n  {'Loss':<20} {'MAE vs observed':>16} {'MAE vs BP':>12}")
    print(f"  {'-'*50}")
    for name, dem in results.items():
        flat = np.maximum(dem.reshape(len(logT), -1)[:, valid.flatten()], 0)
        rs = (R_scaled @ flat)
        obs_flat = aia_cube.reshape(C, -1)[:, valid.flatten()]
        mae_obs = np.nanmean(np.abs(rs - obs_flat))
        if bp_dem is not None and name != 'BP':
            bp_flat = np.maximum(bp_dem.reshape(len(logT), -1)[:, valid.flatten()], 0)
            mae_bp = np.nanmean(np.abs(flat - bp_flat))
        else:
            mae_bp = 0.0
        print(f"  {name:<20} {mae_obs:>16.4f} {mae_bp:>12.4f}")

    print(f"  Saved plots: {out_dir}/{tag}_*.png")


# ── main ─────────────────────────────────────────────────────────────────────

def process_timestamp(data_dir, D, B, R, logT, args):
    tag = os.path.basename(data_dir.rstrip('/'))
    print(f"\n{'='*60}")
    print(f"Timestamp: {tag}")

    from src.utils import processIndAIAData
    data_args = SimpleNamespace(crop=args.crop, corr_table="aia_corr.csv", pointing_file="")
    aia_cube, aia_errors, _ = processIndAIAData(data_dir, args=data_args)
    C, H, W = aia_cube.shape
    print(f"  AIA cube: {aia_cube.shape}")

    # flatten valid pixels
    obs_all = aia_cube.reshape(C, -1).T.astype(np.float64)
    err_all = aia_errors.reshape(C, -1).T.astype(np.float64)
    valid = np.all(np.isfinite(obs_all) & np.isfinite(err_all) & (obs_all > 0), axis=1)
    obs_v, err_v = obs_all[valid], err_all[valid]
    n_valid = valid.sum()
    print(f"  Valid pixels: {n_valid:,} / {H*W:,}")

    out_dir = os.path.join("output/experiments/run_all_losses", tag)
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    def log_progress(name, done, total):
        pct = 100 * done / total
        print(f"  {name}: {done:,}/{total:,} pixels ({pct:.0f}%)", flush=True)

    # BP
    print(f"  Running BP...", flush=True)
    bp_args = [(obs_v[i], err_v[i], D, B) for i in range(n_valid)]
    with Pool(args.workers) as pool:
        bp_dems = []
        for i, result in enumerate(pool.imap(run_bp_pixel, bp_args), 1):
            bp_dems.append(result)
            if i % 5000 == 0 or i == n_valid:
                log_progress('BP', i, n_valid)
    bp_dem_flat = np.array(bp_dems).T
    dem_full = np.zeros((len(logT), H * W))
    dem_full[:, valid] = bp_dem_flat
    results['BP'] = dem_full.reshape(len(logT), H, W)
    np.save(os.path.join(out_dir, "bp_dem.npy"), results['BP'])
    print(f"  BP done.", flush=True)

    # each differentiable loss
    for loss_name in LOSSES:
        print(f"  Running {loss_name}...", flush=True)
        pixel_args = [(obs_v[i], err_v[i], D, args.tolfac, loss_name, args.steps)
                      for i in range(n_valid)]
        with Pool(args.workers) as pool:
            x_all = []
            for i, result in enumerate(pool.imap(optimize_pixel, pixel_args), 1):
                x_all.append(result)
                if i % 5000 == 0 or i == n_valid:
                    log_progress(loss_name, i, n_valid)
        x_all = np.array(x_all)
        dem_v = (B @ x_all.T)
        dem_full = np.zeros((len(logT), H * W))
        dem_full[:, valid] = dem_v
        results[loss_name] = dem_full.reshape(len(logT), H, W)
        np.save(os.path.join(out_dir, f"{loss_name}_dem.npy"), results[loss_name])
        print(f"  {loss_name} done.", flush=True)

    # visualize
    scale = 10 ** 26
    R_scaled = (R * scale).astype(np.float32)
    visualize_results(results, aia_cube, logT, R_scaled, out_dir, tag)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs='+',
                   default=["./data/20110906_2217", "./data/20120603_0000",
                            "./data/20131113_0908", "./data/20140910_1731"])
    p.add_argument("--crop",    default="1800,1800,256,256")
    p.add_argument("--steps",   type=int,   default=2000)
    p.add_argument("--tolfac",  type=float, default=1.4)
    p.add_argument("--workers", type=int,   default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R_raw = R.copy()
    R = (R * scale).astype(np.float64)
    B = getBasis(R, logT, alphas=[0.0, 0.1, 0.2])
    D = R @ B

    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            print(f"Skipping {data_dir} — not found")
            continue
        process_timestamp(data_dir, D, B, R_raw, logT, args)

    print("\nAll done. Results in output/experiments/run_all_losses/")
