#!/usr/bin/env python3
"""Evaluate an mlp6 checkpoint on Samuel's full BP or ENet test set.

Unlike this repo's staged split, Samuel's shared test roots contain only X and
solver Y.  They deliberately do not contain per-pixel error/tolerance arrays,
so ``eval_sweep.py`` cannot be used.  This script evaluates the solver-labelled
pixels directly (AIA ``[::2, ::2]`` aligned to the 128x128 DEM grid) and reports
the paper's DEM MSE, relative error, and W1 metrics.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.train_scaled import load_operators
from src.scaled_eval import load_scaled_model
from src.zarr_data import N_AIA_BINS

REL_FLOOR = 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="scaled mlp6 checkpoint")
    p.add_argument("--data", required=True, help="Samuel root with test_x/y.zarr")
    p.add_argument("--reference", required=True, help="BP or ENet label for output")
    p.add_argument("--output", required=True)
    p.add_argument("--pixel_batch", type=int, default=8192)
    p.add_argument("--max_blocks", type=int, default=None,
                   help="smoke-test cap only; omit for the complete test set")
    p.add_argument("--bright_thresholds", default=None,
                   help="JSON list of six precomputed top-5%% AIA thresholds")
    return p.parse_args()


def make_patches(obs, rows, cols, patch_size=9):
    """Return stride-2-aligned 9x9 patches for requested DEM-grid pixels."""
    p = patch_size // 2
    padded = np.pad(obs, ((0, 0), (p, p), (p, p)), mode="edge")
    offsets = np.arange(patch_size, dtype=np.int64)
    rr = rows[:, None] + offsets
    cc = cols[:, None] + offsets
    patch = padded[:, rr[:, :, None], cc[:, None, :]]
    return np.ascontiguousarray(patch.transpose(1, 0, 2, 3))


@torch.no_grad()
def predict_block(model, basis_t, x_block, pixel_batch):
    """Predict all 128x128 solver-grid pixels from one 256x256 AIA block."""
    obs = np.asarray(x_block[:, ::2, ::2], dtype=np.float32)
    h, w = obs.shape[1:]
    rows, cols = np.indices((h, w))
    rows, cols = rows.ravel(), cols.ravel()
    out = np.empty((len(rows), N_AIA_BINS), dtype=np.float32)
    device = basis_t.device
    for start in range(0, len(rows), pixel_batch):
        stop = min(start + pixel_batch, len(rows))
        patch = make_patches(obs, rows[start:stop], cols[start:stop])
        coeff = model(torch.from_numpy(patch).to(device, non_blocking=True))
        out[start:stop] = (coeff @ basis_t.T)[:, :N_AIA_BINS].cpu().numpy()
    return out.reshape(h, w, N_AIA_BINS).transpose(2, 0, 1), obs


def empty_acc():
    return {"sq": 0.0, "rel": 0.0, "w1": 0.0, "n_bins": 0, "n_px": 0, "n_w1": 0}


def add_metrics(acc, pred, truth, mask, logt):
    """Accumulate paper metrics over a [18,H,W] prediction/reference pair."""
    if not mask.any():
        return
    p = pred[:, mask].T.astype(np.float64, copy=False)
    y = truth[:, mask].T.astype(np.float64, copy=False)
    diff = p - y
    acc["sq"] += float(np.square(diff).sum())
    acc["rel"] += float((np.abs(diff) / (np.abs(y) + REL_FLOOR)).sum())
    acc["n_bins"] += int(p.size)
    acc["n_px"] += int(p.shape[0])

    # 1-D Wasserstein distance over a common logT support: L1 CDF distance.
    psum, ysum = p.sum(axis=1), y.sum(axis=1)
    ok = (psum > 0) & (ysum > 0)
    if ok.any():
        pcdf = np.cumsum(np.clip(p[ok], 0, None) / psum[ok, None], axis=1)
        ycdf = np.cumsum(np.clip(y[ok], 0, None) / ysum[ok, None], axis=1)
        acc["w1"] += float((np.abs(pcdf[:, :-1] - ycdf[:, :-1]) * np.diff(logt)[None]).sum())
        acc["n_w1"] += int(ok.sum())


def finish(acc):
    return {
        "dem_mse": acc["sq"] / max(acc["n_bins"], 1),
        "dem_rel_err_pct": 100 * acc["rel"] / max(acc["n_bins"], 1),
        "w1_dex": acc["w1"] / max(acc["n_w1"], 1),
        "n_pixels": acc["n_px"],
        "n_w1_pixels": acc["n_w1"],
    }


def main():
    args = parse_args()
    import zarr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, basis_t, n_basis, logt = load_operators(device)
    model, ckpt = load_scaled_model(args.model, n_basis, device)
    x = zarr.open(os.path.join(args.data, "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(args.data, "test_y.zarr"), mode="r")
    if x.shape[:3] != (6, 256, 256) or y.shape[:3] != (26, 128, 128):
        raise ValueError(f"unexpected shapes: X={x.shape}, Y={y.shape}")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"different block counts: X={x.shape[-1]}, Y={y.shape[-1]}")

    thresholds = None
    if args.bright_thresholds:
        with open(args.bright_thresholds) as f:
            thresholds = np.asarray(json.load(f), dtype=np.float32)
        if thresholds.shape != (6,):
            raise ValueError("--bright_thresholds must be a JSON list of six values")

    n_blocks = min(x.shape[-1], args.max_blocks) if args.max_blocks else x.shape[-1]
    acc = {"full": empty_acc()}
    if thresholds is not None:
        acc.update(bright=empty_acc(), quiet=empty_acc())
    print(f"device={device}; checkpoint={ckpt['variant']}/{ckpt['loss']}; blocks={n_blocks:,}")

    for i in range(n_blocks):
        pred, obs = predict_block(model, basis_t, x[:, :, :, i], args.pixel_batch)
        truth = np.asarray(y[:N_AIA_BINS, :, :, i], dtype=np.float32)
        valid = np.isfinite(truth).all(axis=0) & np.isfinite(pred).all(axis=0)
        add_metrics(acc["full"], pred, truth, valid, logt[:N_AIA_BINS])
        if thresholds is not None:
            bright = (obs >= thresholds[:, None, None]).any(axis=0) & valid
            add_metrics(acc["bright"], pred, truth, bright, logt[:N_AIA_BINS])
            add_metrics(acc["quiet"], pred, truth, valid & ~bright, logt[:N_AIA_BINS])
        if i % 100 == 0 or i + 1 == n_blocks:
            print(f"  {i + 1:,}/{n_blocks:,} blocks", flush=True)

    rows = {key: finish(value) for key, value in acc.items()}
    payload = {"model": os.path.abspath(args.model), "data": os.path.abspath(args.data),
               "reference": args.reference, "n_blocks": n_blocks,
               "bright_thresholds": None if thresholds is None else thresholds.tolist(),
               "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
