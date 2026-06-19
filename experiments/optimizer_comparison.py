"""
Compare optimizer variants (L-BFGS, Adam, SGD) on the barrier and barrier_fit
losses, on a set of randomly/diversely sampled pixels. Plots DEM curves for
each optimizer x loss combination overlaid against BP.

Run from project root:
    uv run python experiments/optimizer_comparison.py
    uv run python experiments/optimizer_comparison.py --n_pixels 6 --steps 2000
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from types import SimpleNamespace

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch

NN_COLORS = {'barrier': 'tab:cyan', 'barrier_fit': 'tab:pink'}


def load_nn_models():
    """Load trained barrier/barrier_fit NN checkpoints if present."""
    from experiments.train_unsupervised import DEMNet
    models = {}
    for name in NN_COLORS:
        ckpt_path = f"output/experiments/{name}_nn_model.pt"
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DEMNet(n_basis=ckpt["B"].shape[1], hidden=ckpt.get("hidden", 256))
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[name] = (model, ckpt["B"])
    return models


def loss_barrier(x, D, obs, lb, ub, **kw):
    return barrier_loss_batch(x, D, obs, lb, ub, a_l1=1.0, a_l2=0.0, mu=1.0, alpha=0.0)

def loss_barrier_fit(x, D, obs, lb, ub, **kw):
    return barrier_loss_batch(x, D, obs, lb, ub, a_l1=1.0, a_l2=0.0, mu=1.0, alpha=0.1)

LOSSES = {'barrier': loss_barrier, 'barrier_fit': loss_barrier_fit}

OPTIMIZERS = ['lbfgs', 'adam', 'sgd']
COLORS = {'lbfgs': 'tab:blue', 'adam': 'tab:orange', 'sgd': 'tab:green'}
LINESTYLES = {'barrier': '-', 'barrier_fit': '--'}


def optimize_dem(obs_np, err_np, D_np, B_np, loss_fn, optimizer_name, tolfac=1.4, steps=2000):
    obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    D   = torch.tensor(D_np,   dtype=torch.float32)
    lb  = torch.tensor(obs_np - tolfac * err_np, dtype=torch.float32).unsqueeze(0)
    ub  = torch.tensor(obs_np + tolfac * err_np, dtype=torch.float32).unsqueeze(0)

    D_pinv = torch.linalg.pinv(D)
    x0 = torch.relu(obs @ D_pinv.T)
    x  = nn.Parameter(x0.clone())

    if optimizer_name == 'lbfgs':
        opt = torch.optim.LBFGS([x], lr=0.01, max_iter=50, line_search_fn='strong_wolfe',
                                 tolerance_change=1e-12, tolerance_grad=1e-32)
        for _ in range(steps // 50):
            def closure():
                opt.zero_grad()
                loss = loss_fn(torch.relu(x), D, obs, lb, ub)
                loss.backward()
                return loss
            opt.step(closure)
    elif optimizer_name == 'adam':
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(torch.relu(x), D, obs, lb, ub)
            loss.backward()
            opt.step()
    elif optimizer_name == 'sgd':
        opt = torch.optim.SGD([x], lr=0.01, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(torch.relu(x), D, obs, lb, ub)
            loss.backward()
            opt.step()

    x_star = torch.relu(x).detach().numpy()[0]
    dem = B_np @ x_star
    resynth = D_np @ x_star
    return dem, resynth


def pick_pixels(aia_cube, aia_errors, n, seed=42):
    C, H, W = aia_cube.shape
    obs = aia_cube.reshape(C, -1).T
    err = aia_errors.reshape(C, -1).T
    valid = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > 0), axis=1)
    idx = np.where(valid)[0]
    brightness = obs[idx, 2]
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

    print("Loading AIA data...")
    from src.utils import processIndAIAData
    data_args = SimpleNamespace(crop=args.crop, corr_table="aia_corr.csv", pointing_file="")
    aia_cube, aia_errors, _ = processIndAIAData(args.data_dir, args=data_args)
    print(f"AIA cube: {aia_cube.shape}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float64)
    B = getBasis(R, logT, alphas=[0.0, 0.1, 0.2])
    D = R @ B

    pixel_indices, obs_all, err_all = pick_pixels(aia_cube, aia_errors, args.n_pixels)
    print(f"Selected {len(pixel_indices)} pixels")

    nn_models = load_nn_models()
    if nn_models:
        print(f"Loaded NN checkpoints: {list(nn_models.keys())}")
    else:
        print("No NN checkpoints found — skipping NN overlay curves")

    fig, axes = plt.subplots(args.n_pixels, 1, figsize=(8, 3.5 * args.n_pixels))
    if args.n_pixels == 1:
        axes = [axes]

    for pi, pidx in enumerate(pixel_indices):
        obs = obs_all[pidx].astype(np.float64)
        err = err_all[pidx].astype(np.float64)

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

        if bp_dem is not None:
            ax.plot(logT, np.maximum(bp_dem, 0), color='black', lw=2.5,
                    label=f'BP  MAE={bp_mae:.3f}', zorder=1)

        for loss_name, loss_fn in LOSSES.items():
            for opt_name in OPTIMIZERS:
                dem, resynth = optimize_dem(obs, err, D, B, loss_fn, opt_name,
                                            tolfac=tolfac, steps=args.steps)
                mae = np.mean(np.abs(resynth - obs))
                ax.plot(logT, np.maximum(dem, 0), color=COLORS[opt_name],
                        linestyle=LINESTYLES[loss_name], lw=1.8,
                        label=f'{loss_name}/{opt_name}  MAE={mae:.3f}')

        for nn_name, (model, B_nn) in nn_models.items():
            with torch.no_grad():
                x_nn = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                dem_nn = B_nn @ x_nn.numpy()[0]
            dem_nn = np.maximum(dem_nn, 0)
            mae_nn = np.mean(np.abs(R @ dem_nn - obs))
            ax.plot(logT, dem_nn, color=NN_COLORS[nn_name], linestyle=':', lw=2.2,
                    label=f'{nn_name} (NN)  MAE={mae_nn:.3f}', zorder=10)

        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.set_title(f"Pixel {pi+1}  (171Å brightness={obs[2]:.1f})", fontsize=9)
        print(f"Pixel {pi+1}/{args.n_pixels} done")

    plt.suptitle("Optimizer comparison: L-BFGS vs Adam vs SGD vs NN\n"
                 "(barrier solid, barrier_fit dashed, NN dotted)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = "output/experiments/optimizer_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/20110906_2217")
    p.add_argument("--crop",     default="1800,1800,256,256")
    p.add_argument("--n_pixels", type=int, default=6)
    p.add_argument("--steps",    type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
