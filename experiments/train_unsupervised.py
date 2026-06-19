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
    """Each sample is one pixel: (obs [6], err [6], lb [6], ub [6]).
    Can pool pixels from multiple timestamps (data_dirs)."""

    def __init__(self, data_dirs, crop, tolfac=1.4):
        from src.utils import processIndAIAData
        obs_parts, err_parts = [], []
        for data_dir in data_dirs:
            data_args = SimpleNamespace(crop=crop, corr_table="aia_corr.csv", pointing_file="")
            aia_cube, aia_errors, _ = processIndAIAData(data_dir, args=data_args)
            C, H, W = aia_cube.shape
            obs = aia_cube.reshape(C, -1).T.astype(np.float32)
            err = aia_errors.reshape(C, -1).T.astype(np.float32)
            valid = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > 0), axis=1)
            obs_parts.append(obs[valid])
            err_parts.append(err[valid])
            print(f"  {os.path.basename(data_dir)}: {valid.sum():,} valid pixels")

        obs = np.concatenate(obs_parts, axis=0)
        err = np.concatenate(err_parts, axis=0)

        self.obs = torch.from_numpy(obs)
        self.err = torch.from_numpy(err)
        self.lb  = torch.from_numpy(obs - tolfac * err)
        self.ub  = torch.from_numpy(obs + tolfac * err)
        print(f"Dataset: {len(self.obs):,} total pixels pooled from {len(data_dirs)} timestamps")

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

    # response matrix
    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)  # [6, n_basis]
    D_t = torch.tensor(D).to(device)
    B_t = torch.tensor(B).to(device)

    # dataset — pooled across all data_dirs
    print("Loading AIA data...")
    dataset = AIAPixelDataset(args.data_dirs, args.crop, tolfac=args.tolfac)
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
            alpha_fit = 0.1 if args.loss_name == 'barrier_fit' else 0.0
            loss = barrier_loss_batch(
                x, D_t, obs, lb, ub,
                a_l1=args.alpha_l1,
                a_l2=args.alpha_l2,
                mu=args.mu,
                alpha=alpha_fit,
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
    ckpt = f"output/experiments/{args.loss_name}_nn_model.pt"
    torch.save({"model": model.state_dict(), "D": D, "B": B, "logT": logT,
                "hidden": args.hidden}, ckpt)
    print(f"Saved checkpoint: {ckpt}")

    # plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"Unsupervised training loss — {args.loss_name}")
    plt.tight_layout()
    plt.savefig(f"output/experiments/{args.loss_name}_train_loss.png", dpi=150)

    return model, D, B, logT


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


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs='+',
                   default=["./data/20110906_2217", "./data/20120603_0000",
                            "./data/20131113_0908", "./data/20140910_1731"])
    p.add_argument("--loss_name",   choices=['barrier', 'barrier_fit'], default='barrier')
    p.add_argument("--crop",        default="1800,1800,256,256",
                   help="sy,sx,h,w — use empty string for full image")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=4096)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--tolfac",      type=float, default=1.4)
    p.add_argument("--alpha_l1",    type=float, default=1.0)
    p.add_argument("--alpha_l2",    type=float, default=0.0)
    p.add_argument("--mu",          type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
