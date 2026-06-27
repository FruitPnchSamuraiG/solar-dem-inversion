"""
Neural field DEM model: amortized across multiple timestamps (step 2).

Trains one PatchDEMNet over all provided AIA timestamps jointly. A random
fraction of pixels from each timestamp is held out for validation (not seen
during training) so we can check whether the model generalizes within the
training distribution.

Run from project root:
    uv run python experiments/train_neural_field_amortized.py \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch
from experiments.train_neural_field import (
    AIAPatchDataset, PatchDEMNet, effective_sparsity, pick_device, bp_sparsity_reference,
)


def split_dataset(dataset, val_frac, seed):
    """Split dataset into train/val Subsets by random pixel index."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return Subset(dataset, idx[n_val:].tolist()), Subset(dataset, idx[:n_val].tolist())


def evaluate_val(model, val_subsets, D, device, n_sample=500, seed=42):
    """
    Run NN inference on held-out val pixels for each timestamp.
    Returns dict: tag -> {nn_sparsity, nn_mae, n_val}
    """
    D_t = torch.tensor(D).to(device)
    model.eval()
    results = {}
    for tag, val_ds in val_subsets.items():
        rng = np.random.default_rng(seed)
        n = len(val_ds)
        chosen = rng.choice(n, size=min(n_sample, n), replace=False)

        nn_sparsities, nn_maes = [], []
        with torch.no_grad():
            for i in chosen:
                patch, obs, lb, ub = val_ds[int(i)]
                x = model(patch.unsqueeze(0).to(device))
                nn_sparsities.append(effective_sparsity(x).item())
                pred = (D_t @ x[0]).cpu().numpy()
                nn_maes.append(float(np.mean(np.abs(pred - obs.numpy()))))

        results[tag] = {
            "nn_sparsity": float(np.mean(nn_sparsities)),
            "nn_mae":      float(np.mean(nn_maes)),
            "n_val":       n,
        }
    return results


def train(args):
    device = pick_device()
    print(f"Device: {device}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    D_t = torch.tensor(D).to(device)

    # ── per-timestamp datasets + train/val split ───────────────────────────
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
    print(f"\nTotal train pixels across all timestamps: {len(full_train):,}")

    loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # ── model ──────────────────────────────────────────────────────────────
    model = PatchDEMNet(n_basis=D.shape[1], patch_size=args.patch_size,
                        channels=args.channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))

    os.makedirs("output/experiments", exist_ok=True)
    history = []

    # ── training loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_sparsity = 0.0, 0.0

        for patch, obs, lb, ub in loader:
            patch, obs, lb, ub = (patch.to(device), obs.to(device),
                                   lb.to(device), ub.to(device))
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

        avg_loss     = epoch_loss     / len(loader)
        avg_sparsity = epoch_sparsity / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"eff_sparsity={avg_sparsity:.2f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ── save checkpoint ────────────────────────────────────────────────────
    tags = [os.path.basename(d.rstrip("/")) for d in args.data_dirs]
    ckpt_tag = f"amortized_{len(tags)}ts"
    ckpt_path = f"output/experiments/neural_field_{ckpt_tag}.pt"
    torch.save({
        "model":      model.state_dict(),
        "D": D, "B": B, "logT": logT,
        "patch_size": args.patch_size,
        "channels":   args.channels,
        "tags":       tags,
    }, ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")

    # ── loss curve ─────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"Neural field amortized training loss ({len(tags)} timestamps)")
    plt.tight_layout()
    plt.savefig(f"output/experiments/neural_field_{ckpt_tag}_train_loss.png", dpi=150)

    # ── validation report ──────────────────────────────────────────────────
    print("\nEvaluating on held-out val pixels per timestamp...")
    val_results = evaluate_val(model, val_subsets, D, device, seed=args.seed)

    print(f"\n{'Timestamp':<22} {'Val pixels':>10} {'NN sparsity':>12} "
          f"{'BP sparsity':>12} {'NN MAE':>10}")
    print("-" * 72)
    for tag, res in val_results.items():
        bp_sp = bp_refs.get(tag, float("nan"))
        print(f"{tag:<22} {res['n_val']:>10,} {res['nn_sparsity']:>12.2f} "
              f"{bp_sp:>12.2f} {res['nn_mae']:>10.4f}")

    return model, D, B, logT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs",  nargs="+", required=True,
                   help="AIA data directories, one per timestamp")
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--patch_size", type=int,   default=9)
    p.add_argument("--channels",   type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--val_frac",   type=float, default=0.2,
                   help="fraction of pixels per timestamp held out for validation")
    p.add_argument("--alpha_l1",   type=float, default=1.0)
    p.add_argument("--alpha_l2",   type=float, default=0.0)
    p.add_argument("--mu",         type=float, default=1.0)
    p.add_argument("--bp_compare", type=int,   default=100,
                   help="pixels per timestamp for BP sparsity reference; 0 to skip")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
