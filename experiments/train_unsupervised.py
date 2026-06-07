"""
Unsupervised DEM training with BarrierLoss.

No labels needed — only raw AIA images from JSOC.
The NN learns to satisfy the BP physics constraints directly.

Run from project root:
    uv run python experiments/train_unsupervised.py
    uv run python experiments/train_unsupervised.py --crop 1800,1800,256,256
    uv run python experiments/train_unsupervised.py --epochs 20 --batch_size 2048
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from types import SimpleNamespace

from fullBP import getBasis
from src.losses import barrier_loss_batch


# ── dataset ───────────────────────────────────────────────────────────────────

class AIAPixelDataset(Dataset):
    """Each sample is one pixel: (obs [6], err [6], lb [6], ub [6])."""

    def __init__(self, aia_cube, aia_errors, tolfac=1.4):
        C, H, W = aia_cube.shape
        # flatten to (N, C)
        obs = aia_cube.reshape(C, -1).T.astype(np.float32)    # [N, 6]
        err = aia_errors.reshape(C, -1).T.astype(np.float32)  # [N, 6]

        # drop pixels with any nan or non-positive obs
        valid = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > 0), axis=1)
        obs, err = obs[valid], err[valid]

        self.obs = torch.from_numpy(obs)
        self.err = torch.from_numpy(err)
        self.lb  = torch.from_numpy(obs - tolfac * err)
        self.ub  = torch.from_numpy(obs + tolfac * err)
        print(f"Dataset: {len(self.obs):,} valid pixels")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.err[idx], self.lb[idx], self.ub[idx]


# ── model ─────────────────────────────────────────────────────────────────────

class DEMNet(nn.Module):
    """MLP: 6 AIA channels → n_basis coefficients (positive via Softplus)."""

    def __init__(self, n_basis=54, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_basis),
            nn.Softplus(),  # ensures output > 0 smoothly
        )

    def forward(self, x):  # x: [B, 6]
        return self.net(x)  # [B, n_basis]


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cpu")
    print(f"Device: {device}")

    # load AIA data
    print("Loading AIA data...")
    from src.utils import processIndAIAData
    data_args = SimpleNamespace(crop=args.crop, corr_table="aia_corr.csv", pointing_file="")
    aia_cube, aia_errors, _ = processIndAIAData(args.data_dir, args=data_args)
    print(f"AIA cube: {aia_cube.shape}")

    # response matrix
    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)  # [6, n_basis]
    D_t = torch.tensor(D).to(device)
    B_t = torch.tensor(B).to(device)

    # dataset
    dataset = AIAPixelDataset(aia_cube, aia_errors, tolfac=args.tolfac)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)

    # model + optimizer
    model = DEMNet(n_basis=D.shape[1], hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))

    os.makedirs("output/experiments", exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for obs, err, lb, ub in loader:
            obs, lb, ub = obs.to(device), lb.to(device), ub.to(device)

            optimizer.zero_grad()
            x = model(obs)  # [B, n_basis]
            loss = barrier_loss_batch(
                x, D_t, obs, lb, ub,
                a_l1=args.alpha_l1,
                a_l2=args.alpha_l2,
                mu=args.mu,
                alpha=args.alpha_fit,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # save checkpoint
    ckpt = "output/experiments/unsupervised_model.pt"
    torch.save({"model": model.state_dict(), "D": D, "B": B, "logT": logT}, ckpt)
    print(f"Saved checkpoint: {ckpt}")

    # plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Unsupervised training loss")
    plt.tight_layout()
    plt.savefig("output/experiments/train_loss.png", dpi=150)

    return model, D, B, logT, aia_cube, aia_errors


# ── inference ────────────────────────────────────────────────────────────────

def run_nn_inference(model, B, logT, aia_cube, device):
    """Run trained NN on every pixel, return dem_cube [n_temps, H, W]."""
    model.eval()
    C, H, W = aia_cube.shape
    B_t = torch.tensor(B).to(device)
    obs_all = torch.tensor(aia_cube.reshape(C, -1).T.astype(np.float32))

    dem_all = []
    with torch.no_grad():
        for i in range(0, len(obs_all), 8192):
            chunk = obs_all[i:i+8192].to(device)
            x = model(chunk)
            dem = torch.matmul(x, B_t.T)
            dem_all.append(dem.cpu())

    dem_all = torch.cat(dem_all, dim=0).numpy()       # [N, n_temps]
    return dem_all.T.reshape(len(logT), H, W)         # [n_temps, H, W]


def run_bp_solver(aia_cube, aia_errors, D, B, tolfac=1.4):
    """Run BP solver on every pixel, return dem_cube [n_temps, H, W]."""
    from fullBP import invertDEMCube
    args = SimpleNamespace(fitfn='lp', parallel=-1, tolfac=tolfac,
                           zerochill=False, crop='', decimate=1, noisy=0)
    RData = {'R': D @ np.linalg.pinv(B), 'B': B}
    # invertDEMCube expects R and B separately — pass D as R with identity B
    n_temps = B.shape[0]
    RData = {'R': D, 'B': np.eye(D.shape[1])}
    dem_cube = invertDEMCube(aia_cube, aia_errors, RData, args)
    # convert basis coefficients back to DEM: dem = B @ x
    C, H, W = aia_cube.shape
    dem_flat = dem_cube.reshape(D.shape[1], -1)  # [n_basis, N]
    dem_temps = B @ dem_flat                      # [n_temps, N]
    return dem_temps.reshape(n_temps, H, W)


# ── visualize ─────────────────────────────────────────────────────────────────

def visualize(nn_dem, bp_dem, aia_cube, logT, D, B):
    """Side-by-side comparison: NN vs BP solver."""
    RData = np.load("RData.npz")
    R = (RData['R'] * 10**26).astype(np.float32)  # [6, 18]
    wavelengths = [94, 131, 171, 193, 211, 335]
    C, H, W = aia_cube.shape
    n_temps = len(logT)

    def mean_logt(dem):
        distr = dem / (1e-8 + dem.sum(axis=0, keepdims=True))
        return (logT.reshape(-1, 1, 1) * distr).sum(axis=0)

    def resynth(dem):
        flat = np.maximum(dem.reshape(n_temps, -1), 0)
        return (R @ flat).reshape(C, H, W)

    nn_resynth = resynth(nn_dem)
    bp_resynth = resynth(bp_dem) if bp_dem is not None else None

    nn_mae = np.nanmean(np.abs(nn_resynth - aia_cube))
    bp_mae = np.nanmean(np.abs(bp_resynth - aia_cube)) if bp_dem is not None else float('nan')
    print(f"Resynthesis MAE  — NN: {nn_mae:.4f}  |  BP: {bp_mae:.4f}")

    # ── 1. mean logT comparison ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2 if bp_dem is not None else 1, figsize=(12, 5))
    if bp_dem is None:
        axes = [axes]
    im = axes[0].imshow(mean_logt(nn_dem), cmap='inferno', vmin=5.5, vmax=7.5)
    axes[0].set_title("Mean logT — NN unsupervised"); axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    if bp_dem is not None:
        im = axes[1].imshow(mean_logt(bp_dem), cmap='inferno', vmin=5.5, vmax=7.5)
        axes[1].set_title("Mean logT — BP solver"); axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig("output/experiments/mean_logt.png", dpi=150)

    # ── 2. resynthesis diff ───────────────────────────────────────────────────
    ncols = 2 if bp_dem is not None else 1
    fig, axes = plt.subplots(6, ncols + 1, figsize=(5 * (ncols + 1), 4 * 6))
    for i in range(6):
        vrange = np.nanmax(np.abs(aia_cube[i])) * 0.3
        axes[i, 0].imshow(aia_cube[i] ** 0.5, cmap=f'sdoaia{wavelengths[i]}')
        axes[i, 0].set_title(f"{wavelengths[i]}Å — observed"); axes[i, 0].axis('off')
        diff = nn_resynth[i] - aia_cube[i]
        axes[i, 1].imshow(diff, cmap='bwr', vmin=-vrange, vmax=vrange)
        axes[i, 1].set_title(f"{wavelengths[i]}Å — NN diff"); axes[i, 1].axis('off')
        if bp_dem is not None:
            diff_bp = bp_resynth[i] - aia_cube[i]
            axes[i, 2].imshow(diff_bp, cmap='bwr', vmin=-vrange, vmax=vrange)
            axes[i, 2].set_title(f"{wavelengths[i]}Å — BP diff"); axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig("output/experiments/resynth_diff.png", dpi=100)

    # ── 3. DEM maps per temperature bin ──────────────────────────────────────
    ncols = 2 if bp_dem is not None else 1
    fig, axes = plt.subplots(n_temps, ncols, figsize=(5 * ncols, 3 * n_temps))
    for i in range(n_temps):
        vmax = np.nanmax(nn_dem[i]) ** 0.5
        if bp_dem is not None:
            vmax = max(vmax, np.nanmax(bp_dem[i]) ** 0.5)
        axes[i, 0].imshow(nn_dem[i] ** 0.5, cmap='inferno', vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"NN  logT={logT[i]:.1f}"); axes[i, 0].axis('off')
        if bp_dem is not None:
            axes[i, 1].imshow(bp_dem[i] ** 0.5, cmap='inferno', vmin=0, vmax=vmax)
            axes[i, 1].set_title(f"BP  logT={logT[i]:.1f}"); axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig("output/experiments/dem_bins.png", dpi=100)

    print("Saved: mean_logt.png, resynth_diff.png, dem_bins.png")


def infer_and_visualize(model, D, B, logT, aia_cube, aia_errors, args, device):
    nn_dem = run_nn_inference(model, B, logT, aia_cube, device)

    print("Running BP solver on same patch for comparison...")
    try:
        bp_dem = run_bp_solver(aia_cube, aia_errors, D, B, tolfac=args.tolfac)
    except Exception as e:
        print(f"BP solver failed: {e}")
        bp_dem = None

    visualize(nn_dem, bp_dem, aia_cube, logT, D, B)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="./data/20170910_1548")
    p.add_argument("--crop",        default="1800,1800,256,256",
                   help="sy,sx,h,w — use empty string for full image")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=4096)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--tolfac",      type=float, default=1.4)
    p.add_argument("--alpha_l1",    type=float, default=1.0)
    p.add_argument("--alpha_l2",    type=float, default=0.0)
    p.add_argument("--mu",          type=float, default=1.0)
    p.add_argument("--alpha_fit",   type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    model, D, B, logT, aia_cube, aia_errors = train(args)
    infer_and_visualize(model, D, B, logT, aia_cube, aia_errors, args, device)
