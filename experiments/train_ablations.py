"""
Ablations: why does the patch CNN work when the per-pixel MLP didn't?

Four model variants, all trained with the same loss, data, and budget, all
taking the same [B, 6, K, K] patch input (each variant internally uses only
what it is allowed to see), so training/eval code is identical:

    cnn           reference patch CNN (PatchDEMNet)          ~1.48M params
    mlp6          MLP on the 6 center-pixel channels only    ~1.43M params
                  -> distinguishes capacity vs. information
    mlp_patch     MLP on the flattened 486-dim patch (no convolution)
                  ~1.50M params -> distinguishes patch info vs. convolution
    cnn_shuffled  patch CNN with a FIXED random permutation of the 81
                  spatial positions -> distinguishes values vs. geometry

Loss is selectable: barrier (BP-style, default) or enet (Elastic Net,
unconstrained -- Athiray & Winebarger 2024).

Run from project root, e.g.:
    uv run python experiments/train_ablations.py --variant mlp_patch \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from fullBP import getBasis
from src.losses import barrier_loss_batch, enet_loss_batch
from experiments.train_neural_field import (
    AIAPatchDataset, PatchDEMNet, effective_sparsity, pick_device, bp_sparsity_reference,
)
from experiments.train_neural_field_amortized import split_dataset, evaluate_val


# ── ablation model variants ────────────────────────────────────────────────────

class CenterMLP(nn.Module):
    """Variant mlp6: MLP on the 6 center-pixel channels only, capacity-matched
    to PatchDEMNet (~1.43M params at hidden=680). Same information as the old
    channel-input DEMNet, ~7x the parameters."""

    def __init__(self, n_basis=54, patch_size=9, hidden=680):
        super().__init__()
        self.pad = patch_size // 2
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_basis),
            nn.Softplus(),
        )

    def forward(self, patch):  # patch: [B, 6, K, K]
        return self.net(patch[:, :, self.pad, self.pad])


class FlatPatchMLP(nn.Module):
    """Variant mlp_patch: MLP on the flattened 6*K*K patch. Same information
    as the CNN, no convolution (~1.50M params)."""

    def __init__(self, n_basis=54, patch_size=9):
        super().__init__()
        in_dim = 6 * patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.SiLU(),
            nn.Linear(1024, 768), nn.SiLU(),
            nn.Linear(768, 256), nn.SiLU(),
            nn.Linear(256, n_basis),
            nn.Softplus(),
        )

    def forward(self, patch):  # patch: [B, 6, K, K]
        return self.net(patch.flatten(1))


class ShuffledPatchCNN(nn.Module):
    """Variant cnn_shuffled: PatchDEMNet applied to a patch whose 81 spatial
    positions are permuted by one FIXED random permutation (same for every
    sample, saved in the checkpoint). Keeps the values/statistics of the
    neighborhood, destroys its geometry."""

    def __init__(self, n_basis=54, patch_size=9, channels=64, perm=None):
        super().__init__()
        self.cnn = PatchDEMNet(n_basis=n_basis, patch_size=patch_size, channels=channels)
        if perm is None:
            perm = np.random.default_rng(0).permutation(patch_size * patch_size)
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
        self.patch_size = patch_size

    def forward(self, patch):  # patch: [B, 6, K, K]
        b, c, k, _ = patch.shape
        shuffled = patch.flatten(2)[:, :, self.perm].view(b, c, k, k)
        return self.cnn(shuffled)


VARIANTS = ["cnn", "mlp6", "mlp_patch", "cnn_shuffled"]


def build_model(variant, n_basis, patch_size, channels, perm=None):
    if variant == "cnn":
        return PatchDEMNet(n_basis=n_basis, patch_size=patch_size, channels=channels)
    if variant == "mlp6":
        return CenterMLP(n_basis=n_basis, patch_size=patch_size)
    if variant == "mlp_patch":
        return FlatPatchMLP(n_basis=n_basis, patch_size=patch_size)
    if variant == "cnn_shuffled":
        return ShuffledPatchCNN(n_basis=n_basis, patch_size=patch_size,
                                channels=channels, perm=perm)
    raise ValueError(f"unknown variant: {variant}")


def load_ablation_model(ckpt_path, n_basis, device):
    """Rebuild an ablation model from its checkpoint (used by plot script)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["variant"], n_basis, ckpt["patch_size"],
                        ckpt["channels"], perm=ckpt.get("perm")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ── training ──────────────────────────────────────────────────────────────────

def make_loss_fn(args):
    if args.loss == "barrier":
        return lambda x, D_t, obs, lb, ub: barrier_loss_batch(
            x, D_t, obs, lb, ub, a_l1=args.alpha_l1, a_l2=args.alpha_l2, mu=args.mu)
    if args.loss == "enet":
        return lambda x, D_t, obs, lb, ub: enet_loss_batch(
            x, D_t, obs, lb, ub, C=args.enet_C, alpha=args.enet_alpha, lam=args.enet_lam)
    raise ValueError(f"unknown loss: {args.loss}")


def train(args):
    device = pick_device()
    print(f"Device: {device}")
    print(f"Variant: {args.variant}   Loss: {args.loss}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    D_t = torch.tensor(D).to(device)

    train_subsets, val_subsets, bp_refs = [], {}, {}
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        print(f"\nLoading {tag}...")
        ds = AIAPatchDataset(data_dir, args.crop, patch_size=args.patch_size, tolfac=args.tolfac)
        train_ds, val_ds = split_dataset(ds, args.val_frac, args.seed)
        print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")
        train_subsets.append(train_ds)
        val_subsets[tag] = val_ds
        if args.bp_compare > 0:
            print(f"  computing BP sparsity reference for {tag}...")
            bp_refs[tag] = bp_sparsity_reference(ds, D, n_pixels=args.bp_compare, seed=args.seed)
            print(f"  BP sparsity reference: {bp_refs[tag]:.2f}")

    full_train = ConcatDataset(train_subsets)
    print(f"\nTotal train pixels: {len(full_train):,}")
    loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    perm = None
    if args.variant == "cnn_shuffled":
        perm = np.random.default_rng(args.seed).permutation(args.patch_size ** 2)
    model = build_model(args.variant, D.shape[1], args.patch_size,
                        args.channels, perm=perm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    loss_fn = make_loss_fn(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))

    os.makedirs("output/experiments", exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_sparsity = 0.0, 0.0

        for patch, obs, lb, ub in loader:
            patch, obs, lb, ub = (patch.to(device), obs.to(device),
                                   lb.to(device), ub.to(device))
            optimizer.zero_grad()
            x = model(patch)
            loss = loss_fn(x, D_t, obs, lb, ub)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_sparsity += effective_sparsity(x).mean().item()

        avg_loss     = epoch_loss     / len(loader)
        avg_sparsity = epoch_sparsity / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"eff_sparsity={avg_sparsity:.2f}  lr={scheduler.get_last_lr()[0]:.2e}")

    tag = f"ablation_{args.variant}_{args.loss}"
    ckpt_path = f"output/experiments/{tag}.pt"
    torch.save({
        "model":      model.state_dict(),
        "variant":    args.variant,
        "loss":       args.loss,
        "D": D, "B": B, "logT": logT,
        "patch_size": args.patch_size,
        "channels":   args.channels,
        "perm":       perm,
    }, ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"Ablation training loss — {args.variant} / {args.loss}")
    plt.tight_layout()
    plt.savefig(f"output/experiments/{tag}_train_loss.png", dpi=150)

    print("\nEvaluating on held-out val pixels per timestamp...")
    val_results = evaluate_val(model, val_subsets, D, device, seed=args.seed)

    print(f"\n{'Timestamp':<22} {'Val pixels':>10} {'NN sparsity':>12} "
          f"{'BP sparsity':>12} {'NN MAE':>10}")
    print("-" * 72)
    for t, res in val_results.items():
        bp_sp = bp_refs.get(t, float("nan"))
        print(f"{t:<22} {res['n_val']:>10,} {res['nn_sparsity']:>12.2f} "
              f"{bp_sp:>12.2f} {res['nn_mae']:>10.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",    choices=VARIANTS, required=True)
    p.add_argument("--loss",       choices=["barrier", "enet"], default="barrier")
    p.add_argument("--data_dirs",  nargs="+", required=True)
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--patch_size", type=int,   default=9)
    p.add_argument("--channels",   type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--val_frac",   type=float, default=0.2)
    # barrier loss params
    p.add_argument("--alpha_l1",   type=float, default=1.0)
    p.add_argument("--alpha_l2",   type=float, default=0.0)
    p.add_argument("--mu",         type=float, default=1.0)
    # enet loss params
    p.add_argument("--enet_C",     type=float, default=1.0)
    p.add_argument("--enet_alpha", type=float, default=1.0)
    p.add_argument("--enet_lam",   type=float, default=0.9)
    p.add_argument("--bp_compare", type=int,   default=100)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
