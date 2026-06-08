"""
Multi-loss comparison: for N pixels, run BP (ground truth) and several
differentiable loss formulations, plot all DEM curves side by side.

Goal: find a differentiable loss that converges to similar curves as BP.

Run from project root:
    uv run python experiments/loss_comparison.py
    uv run python experiments/loss_comparison.py --n_pixels 10 --steps 3000
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

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch


# ── loss formulations ─────────────────────────────────────────────────────────

def loss_barrier(x, D, obs, lb, ub, **kw):
    """BP-inspired: L1 sparsity + soft barrier constraints (chi-squared normalised)."""
    return barrier_loss_batch(
        x, D, obs, lb, ub,
        a_l1=kw.get('a_l1', 1.0),
        a_l2=0.0,
        mu=kw.get('mu', 1.0),
        alpha=kw.get('alpha_fit', 0.0),
    )


def loss_barrier_fit(x, D, obs, lb, ub, **kw):
    """Barrier + explicit chi-squared fit term (pulls solution to center of band)."""
    return barrier_loss_batch(
        x, D, obs, lb, ub,
        a_l1=kw.get('a_l1', 1.0),
        a_l2=0.0,
        mu=kw.get('mu', 1.0),
        alpha=kw.get('alpha_fit', 0.1),
    )


def loss_chisq_smooth(x, D, obs, lb, ub, **kw):
    """
    Chi-squared fit + smoothness (L2 on DEM differences across temperature bins).
    No sparsity — prefers smooth DEMs over spiky ones.
    Physics motivation: plasma emission varies smoothly with temperature.
    """
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8

    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()

    # smoothness: penalise large jumps between adjacent temperature bins
    diff = x[:, 1:] - x[:, :-1]
    smooth = kw.get('lambda_smooth', 0.1) * torch.sum(diff ** 2, dim=1).mean()

    # positivity barrier
    pos = 10.0 * torch.sum(torch.relu(-x) ** 2, dim=1).mean()

    return fit + smooth + pos


def loss_entropy_fit(x, D, obs, lb, ub, **kw):
    """
    Maximum entropy: prefers DEMs spread across many temperature bins.
    Chi-squared fit + negative entropy regulariser.
    Physics motivation: without strong prior, prefer the most uninformative DEM.
    """
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8

    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()

    # entropy: -sum(p * log(p)) where p = x / sum(x)
    x_pos = x + 1e-8
    p = x_pos / x_pos.sum(dim=1, keepdim=True)
    entropy = -torch.sum(p * torch.log(p), dim=1).mean()

    # maximise entropy = minimise negative entropy
    lam = kw.get('lambda_entropy', 0.5)
    return fit - lam * entropy


def loss_tikhonov(x, D, obs, lb, ub, **kw):
    """
    Tikhonov (L2) regularisation + chi-squared fit + positivity barrier.
    Equivalent to ridge regression on DEM coefficients.
    Tends to find broad, smooth solutions.
    """
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8

    Dx = torch.matmul(x, D.T)
    fit = torch.sum((Dx - obs) ** 2 / sigma2, dim=1).mean()
    l2  = kw.get('lambda_l2', 0.1) * torch.sum(x ** 2, dim=1).mean()
    pos = 10.0 * torch.sum(torch.relu(-x) ** 2, dim=1).mean()

    return fit + l2 + pos


LOSSES = {
    'barrier (L1, no fit)':   loss_barrier,
    'barrier + fit':          loss_barrier_fit,
    'chi² + smooth':          loss_chisq_smooth,
    'max entropy':            loss_entropy_fit,
    'Tikhonov (L2)':          loss_tikhonov,
}

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


# ── optimizer ────────────────────────────────────────────────────────────────

def optimize_dem(obs_np, err_np, D_np, B_np, loss_fn, tolfac=1.4, steps=2000, **kw):
    """Directly optimize x (18 DEM values) using L-BFGS."""
    obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    D   = torch.tensor(D_np,   dtype=torch.float32)
    lb  = torch.tensor(obs_np - tolfac * err_np, dtype=torch.float32).unsqueeze(0)
    ub  = torch.tensor(obs_np + tolfac * err_np, dtype=torch.float32).unsqueeze(0)

    # initialise near pseudo-inverse solution
    D_pinv = torch.linalg.pinv(D)
    x0 = torch.relu(obs @ D_pinv.T)
    x  = nn.Parameter(x0.clone())

    opt = torch.optim.LBFGS([x], lr=0.01, max_iter=50,
                             line_search_fn='strong_wolfe',
                             tolerance_change=1e-12, tolerance_grad=1e-32)

    for _ in range(steps // 50):
        def closure():
            opt.zero_grad()
            loss = loss_fn(torch.relu(x), D, obs, lb, ub, **kw)
            loss.backward()
            return loss
        opt.step(closure)

    x_star = torch.relu(x).detach().numpy()[0]
    dem = B_np @ x_star
    resynth = D_np @ x_star  # same as R @ dem if using raw R
    return dem, resynth


# ── main ─────────────────────────────────────────────────────────────────────

def pick_pixels(aia_cube, aia_errors, n, seed=42):
    """Pick n diverse pixels: sample from different brightness quantiles."""
    C, H, W = aia_cube.shape
    obs = aia_cube.reshape(C, -1).T
    err = aia_errors.reshape(C, -1).T
    valid = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > 0), axis=1)
    idx = np.where(valid)[0]

    # spread across brightness quantiles of 171A (brightest, most representative)
    brightness = obs[idx, 2]  # 171A channel
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.1, 0.9, n)
    chosen = []
    for q in quantiles:
        threshold = np.quantile(brightness, q)
        close = np.argsort(np.abs(brightness - threshold))[:20]
        chosen.append(idx[rng.choice(close)])

    return chosen, obs, err


def main(args):
    os.makedirs("output/experiments", exist_ok=True)

    # load data
    print("Loading AIA data...")
    from src.utils import processIndAIAData
    data_args = SimpleNamespace(crop=args.crop, corr_table="aia_corr.csv", pointing_file="")
    aia_cube, aia_errors, _ = processIndAIAData(args.data_dir, args=data_args)
    print(f"AIA cube: {aia_cube.shape}")

    # response matrix
    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float64)
    B = getBasis(R, logT, alphas=[0.0, 0.1, 0.2])
    D = (R @ B)  # [6, n_basis]
    n_temps = len(logT)

    # pick pixels
    pixel_indices, obs_all, err_all = pick_pixels(aia_cube, aia_errors, args.n_pixels)
    print(f"Selected {len(pixel_indices)} pixels")

    # run comparison — one subplot per pixel, all curves overlaid
    fig, axes = plt.subplots(args.n_pixels, 1,
                             figsize=(8, 3.5 * args.n_pixels))
    if args.n_pixels == 1:
        axes = [axes]

    for pi, pidx in enumerate(pixel_indices):
        obs = obs_all[pidx].astype(np.float64)
        err = err_all[pidx].astype(np.float64)

        # BP ground truth
        bp_dem = None
        for tolfac in [1.4, 2.0, 2.8, 5.0]:
            lb = obs - tolfac * err
            ub = obs + tolfac * err
            x_bp = solveLP((D, obs, lb, ub, None))
            if x_bp is not None:
                bp_dem = B @ x_bp
                bp_resynth = R @ np.maximum(bp_dem, 0)
                bp_mae = np.mean(np.abs(bp_resynth - obs))
                break

        ax = axes[pi]
        ax.set_ylabel("DEM", fontsize=8)
        ax.set_xlabel("log T", fontsize=8)
        ax.tick_params(labelsize=7)

        # BP reference — light grey, behind everything
        if bp_dem is not None:
            ax.plot(logT, np.maximum(bp_dem, 0), color='black', lw=2.5,
                    alpha=0.25, label=f'BP  MAE={bp_mae:.3f}', zorder=1)
        else:
            ax.set_title(f"Pixel {pi} — BP failed", fontsize=9)

        # all differentiable losses overlaid — solid, dark, on top
        for li, (name, loss_fn) in enumerate(LOSSES.items()):
            dem, resynth = optimize_dem(obs, err, D, B, loss_fn,
                                        tolfac=tolfac, steps=args.steps)
            mae = np.mean(np.abs(resynth - obs))
            ax.plot(logT, np.maximum(dem, 0), color=COLORS[li], lw=2.0,
                    linestyle='-', alpha=1.0, label=f'{name}  MAE={mae:.3f}',
                    zorder=2 + li)

        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f"Pixel {pi+1}  (171Å brightness={obs[2]:.1f})", fontsize=9)

        print(f"Pixel {pi+1}/{args.n_pixels} done")

    plt.suptitle("DEM curves: BP vs differentiable loss formulations\n(dashed black = BP reference)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = "output/experiments/loss_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/20170910_1548")
    p.add_argument("--crop",       default="1800,1800,256,256")
    p.add_argument("--n_pixels",   type=int, default=6)
    p.add_argument("--steps",      type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
