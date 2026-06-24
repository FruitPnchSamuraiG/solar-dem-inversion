"""
Neural field DEM model: per-image, patch-conditioned.

Instead of f(6 channel values at one pixel) -> DEM (the amortized channel-input
NN in train_unsupervised.py), this trains f(local KxK patch of 6-channel AIA
data centered on a pixel) -> DEM for that center pixel, fit jointly over a
single image via the barrier loss. The patch gives the network spatial context
(unlike the per-pixel-only NN, which had none) so the shared CNN weights act as
an implicit spatial smoothness prior, while still being able to resolve
per-pixel structure since the patch content differs pixel to pixel.

This is a per-image fit (one model per timestamp), not yet amortized across
timestamps -- the goal here is to first check whether patch context lets the
network reach BP-like sparse solutions at all, before adding cross-timestamp
amortization or neighborhood masking on top.

Run from project root:
    uv run python experiments/train_neural_field.py --data_dir ./data/20110906_2217 --crop 1800,1800,256,256
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from types import SimpleNamespace

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch


# ── dataset ───────────────────────────────────────────────────────────────────

class AIAPatchDataset(Dataset):
    """Each sample is one pixel's local patch: patch [6, K, K], plus that
    center pixel's obs/lb/ub for the barrier loss."""

    def __init__(self, data_dir, crop, patch_size=9, tolfac=1.4):
        from src.utils import processIndAIAData
        data_args = SimpleNamespace(crop=crop, corr_table="aia_corr.csv", pointing_file="")
        aia_cube, aia_errors, _ = processIndAIAData(data_dir, args=data_args)
        self.C, self.H, self.W = aia_cube.shape
        self.patch_size = patch_size
        pad = patch_size // 2

        obs = aia_cube.astype(np.float32)
        err = aia_errors.astype(np.float32)
        valid = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > 0), axis=0)  # [H, W]

        # pad the cube so every pixel (including edges) has a full patch
        obs_pad = np.pad(obs, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
        err_pad = np.pad(err, ((0, 0), (pad, pad), (pad, pad)), mode="edge")

        self.obs_pad = torch.from_numpy(obs_pad)
        self.err_pad = torch.from_numpy(err_pad)
        self.pad = pad
        self.tolfac = tolfac

        rows, cols = np.where(valid)
        self.coords = list(zip(rows.tolist(), cols.tolist()))
        print(f"Image: {self.C}x{self.H}x{self.W}  valid pixels: {len(self.coords):,}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row, col = self.coords[idx]
        p = self.pad
        patch_obs = self.obs_pad[:, row:row + 2 * p + 1, col:col + 2 * p + 1]  # [6, K, K]
        center_obs = self.obs_pad[:, row + p, col + p]
        center_err = self.err_pad[:, row + p, col + p]
        lb = center_obs - self.tolfac * center_err
        ub = center_obs + self.tolfac * center_err
        return patch_obs, center_obs, lb, ub


# ── model ─────────────────────────────────────────────────────────────────────

class PatchDEMNet(nn.Module):
    """Small CNN over a local AIA patch -> n_basis coefficients (positive via Softplus)."""

    def __init__(self, n_basis=54, patch_size=9, channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        flat_dim = channels * patch_size * patch_size
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, n_basis),
            nn.Softplus(),
        )

    def forward(self, patch):  # patch: [B, 6, K, K]
        feat = self.conv(patch)
        feat = feat.flatten(1)
        return self.head(feat)  # [B, n_basis]


# ── sparsity metric ────────────────────────────────────────────────────────────

def effective_sparsity(x, eps=1e-6):
    """Effective number of 'active' coefficients per row, via normalized L1/L2
    ratio (Hoyer-style): n_active = (sum|x|)^2 / sum(x^2). Lower = sparser.
    Returns per-row tensor [B]."""
    l1 = torch.sum(torch.abs(x), dim=1)
    l2 = torch.sum(x ** 2, dim=1) + eps
    return (l1 ** 2) / l2


# ── training loop ─────────────────────────────────────────────────────────────

def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda") + 1  # probe: fails if GPU arch unsupported by this torch build
        return torch.device("cuda")
    except RuntimeError:
        return torch.device("cpu")


def train(args):
    device = pick_device()
    print(f"Device: {device}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    D_t = torch.tensor(D).to(device)

    print("Loading AIA data...")
    dataset = AIAPatchDataset(args.data_dir, args.crop, patch_size=args.patch_size, tolfac=args.tolfac)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, pin_memory=True)

    model = PatchDEMNet(n_basis=D.shape[1], patch_size=args.patch_size,
                         channels=args.channels).to(device)
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
        epoch_sparsity = 0.0

        for patch, obs, lb, ub in loader:
            patch, obs, lb, ub = patch.to(device), obs.to(device), lb.to(device), ub.to(device)

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

        avg_loss = epoch_loss / len(loader)
        avg_sparsity = epoch_sparsity / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"eff_sparsity={avg_sparsity:.2f}  lr={scheduler.get_last_lr()[0]:.2e}")

    tag = os.path.basename(args.data_dir.rstrip("/"))
    ckpt = f"output/experiments/neural_field_{tag}.pt"
    torch.save({"model": model.state_dict(), "D": D, "B": B, "logT": logT,
                "patch_size": args.patch_size, "channels": args.channels}, ckpt)
    print(f"Saved checkpoint: {ckpt}")

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"Neural field training loss — {tag}")
    plt.tight_layout()
    plt.savefig(f"output/experiments/neural_field_{tag}_train_loss.png", dpi=150)

    return model, D, B, logT, dataset


# ── BP sparsity reference (for comparison printout) ───────────────────────────

def bp_sparsity_reference(dataset, D, n_pixels=200, seed=42):
    """Solve BP on a random subset of pixels and report effective sparsity,
    so the neural field's sparsity can be compared to BP's directly."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.coords), size=min(n_pixels, len(dataset.coords)), replace=False)
    sparsities = []
    for i in idx:
        row, col = dataset.coords[i]
        p = dataset.pad
        obs = dataset.obs_pad[:, row + p, col + p].numpy().astype(np.float64)
        err = dataset.err_pad[:, row + p, col + p].numpy().astype(np.float64)
        for tolfac in [1.4, 2.0, 2.8, 5.0]:
            lb, ub = obs - tolfac * err, obs + tolfac * err
            x_bp = solveLP((D.astype(np.float64), obs, lb, ub, None))
            if x_bp is not None:
                sparsities.append((np.sum(np.abs(x_bp)) ** 2) / (np.sum(x_bp ** 2) + 1e-6))
                break
    return float(np.mean(sparsities)) if sparsities else float("nan")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/20110906_2217")
    p.add_argument("--crop",       default="1800,1800,128,128",
                    help="sy,sx,h,w — smaller than the channel-NN default since patches add cost")
    p.add_argument("--patch_size", type=int, default=9, help="must be odd")
    p.add_argument("--channels",   type=int, default=64)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--alpha_l1",   type=float, default=1.0)
    p.add_argument("--alpha_l2",   type=float, default=0.0)
    p.add_argument("--mu",         type=float, default=1.0)
    p.add_argument("--bp_compare", type=int, default=200,
                    help="number of pixels to solve BP on for a sparsity reference; 0 to skip")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, D, B, logT, dataset = train(args)
    if args.bp_compare > 0:
        print("Solving BP on a comparison subset for sparsity reference...")
        bp_sp = bp_sparsity_reference(dataset, D, n_pixels=args.bp_compare)
        print(f"BP effective sparsity (lower = sparser) on {args.bp_compare} pixels: {bp_sp:.2f}")
