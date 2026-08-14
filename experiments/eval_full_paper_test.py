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

# Per-pixel DEM SSE spans many orders of magnitude.  Keeping all ~800M values
# merely to calculate quantiles would be wasteful, so use a very fine fixed
# log10 histogram instead.  A 0.0001-dex bin makes the reported quantiles much
# more precise than the displayed significant figures, while bin sums let us
# estimate how much of total SSE is in each upper tail.
SSE_LOG10_MIN = -12.0
SSE_LOG10_MAX = 12.0
SSE_HIST_BINS = 240_000


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
    return {"sq": 0.0, "rel": 0.0, "w1": 0.0, "n_bins": 0, "n_px": 0, "n_w1": 0,
            # Histogram count and SSE sum for each log10(per-pixel DEM SSE)
            # bin. Zeros and out-of-range values are retained separately so
            # the accounting remains exact even though quantiles are binned.
            "sse_hist_count": np.zeros(SSE_HIST_BINS, dtype=np.int64),
            "sse_hist_sum": np.zeros(SSE_HIST_BINS, dtype=np.float64),
            "sse_zero_count": 0, "sse_low_count": 0, "sse_low_sum": 0.0,
            "sse_high_count": 0, "sse_high_sum": 0.0,
            # Exact largest *per-pixel* DEM SSE.  Retaining one candidate makes
            # the leave-one-out diagnostic constant-memory over the full set.
            "worst": {"sq": -np.inf, "rel": 0.0, "w1": 0.0, "w1_valid": False,
                      "block": -1, "row": -1, "col": -1}}


def add_metrics(acc, pred, truth, mask, logt, block_index):
    """Accumulate paper metrics over a [18,H,W] prediction/reference pair."""
    if not mask.any():
        return
    p = pred[:, mask].T.astype(np.float64, copy=False)
    y = truth[:, mask].T.astype(np.float64, copy=False)
    diff = p - y
    per_px_sq = np.square(diff).sum(axis=1)
    per_px_rel = (np.abs(diff) / (np.abs(y) + REL_FLOOR)).sum(axis=1)
    acc["sq"] += float(per_px_sq.sum())
    acc["rel"] += float(per_px_rel.sum())
    acc["n_bins"] += int(p.size)
    acc["n_px"] += int(p.shape[0])

    # Streaming distribution summary for ordinary per-pixel DEM SSE. This is
    # deliberately distinct from the training barrier-loss percentiles.
    pos = per_px_sq > 0
    acc["sse_zero_count"] += int((~pos).sum())
    if pos.any():
        values = per_px_sq[pos]
        logs = np.log10(values)
        low = logs < SSE_LOG10_MIN
        high = logs >= SSE_LOG10_MAX
        acc["sse_low_count"] += int(low.sum())
        acc["sse_low_sum"] += float(values[low].sum())
        acc["sse_high_count"] += int(high.sum())
        acc["sse_high_sum"] += float(values[high].sum())
        in_range = ~(low | high)
        if in_range.any():
            indices = np.floor(
                (logs[in_range] - SSE_LOG10_MIN)
                / (SSE_LOG10_MAX - SSE_LOG10_MIN) * SSE_HIST_BINS
            ).astype(np.int64)
            indices = np.clip(indices, 0, SSE_HIST_BINS - 1)
            acc["sse_hist_count"] += np.bincount(indices, minlength=SSE_HIST_BINS)
            acc["sse_hist_sum"] += np.bincount(
                indices, weights=values[in_range], minlength=SSE_HIST_BINS)

    # 1-D Wasserstein distance over a common logT support: L1 CDF distance.
    psum, ysum = p.sum(axis=1), y.sum(axis=1)
    ok = (psum > 0) & (ysum > 0)
    per_px_w1 = np.zeros(len(p), dtype=np.float64)
    if ok.any():
        pcdf = np.cumsum(np.clip(p[ok], 0, None) / psum[ok, None], axis=1)
        ycdf = np.cumsum(np.clip(y[ok], 0, None) / ysum[ok, None], axis=1)
        per_px_w1[ok] = (np.abs(pcdf[:, :-1] - ycdf[:, :-1]) * np.diff(logt)[None]).sum(axis=1)
        acc["w1"] += float(per_px_w1.sum())
        acc["n_w1"] += int(ok.sum())

    # Recover the spatial location corresponding to the flattened valid set.
    # It is only retained for the single global worst pixel, not all pixels.
    worst_local = int(np.argmax(per_px_sq))
    if per_px_sq[worst_local] > acc["worst"]["sq"]:
        row, col = np.nonzero(mask)
        acc["worst"] = {
            "sq": float(per_px_sq[worst_local]),
            "rel": float(per_px_rel[worst_local]),
            "w1": float(per_px_w1[worst_local]),
            "w1_valid": bool(ok[worst_local]),
            "block": int(block_index), "row": int(row[worst_local]), "col": int(col[worst_local]),
        }


def sse_distribution(acc):
    """Return fine-histogram percentile and upper-tail diagnostics."""
    n = acc["n_px"]
    if not n:
        return None
    width = (SSE_LOG10_MAX - SSE_LOG10_MIN) / SSE_HIST_BINS
    counts = acc["sse_hist_count"]
    sums = acc["sse_hist_sum"]
    cumulative = acc["sse_zero_count"] + acc["sse_low_count"] + np.cumsum(counts)

    quantiles = {}
    for q in (50, 90, 99, 99.9, 99.99):
        rank = int(np.ceil(n * q / 100.0))
        if rank <= acc["sse_zero_count"]:
            value = 0.0
        elif rank <= acc["sse_zero_count"] + acc["sse_low_count"]:
            # This bucket is below 1e-12 and contributes negligibly here.
            value = 10 ** SSE_LOG10_MIN
        else:
            idx = int(np.searchsorted(cumulative, rank, side="left"))
            if idx >= SSE_HIST_BINS:
                value = 10 ** SSE_LOG10_MAX
            else:
                # The geometric midpoint is appropriate for a log-uniform bin.
                value = float(10 ** (SSE_LOG10_MIN + (idx + 0.5) * width))
        quantiles[f"p{q}"] = value

    # Accumulate from the upper end. A boundary bin is fractionally apportioned;
    # at 0.0001 dex this approximation is far below displayed precision.
    tails = {}
    total_sq = max(acc["sq"], 1e-300)
    for frac, label in ((0.01, "top_1_pct"), (0.001, "top_0.1_pct"),
                        (0.0001, "top_0.01_pct")):
        remaining = int(np.ceil(n * frac))
        selected_n = min(remaining, acc["sse_high_count"])
        selected_sq = (acc["sse_high_sum"] * selected_n / acc["sse_high_count"]
                       if acc["sse_high_count"] else 0.0)
        remaining -= selected_n
        threshold = 10 ** SSE_LOG10_MAX if selected_n else None
        for idx in range(SSE_HIST_BINS - 1, -1, -1):
            if remaining <= 0:
                break
            count = int(counts[idx])
            if not count:
                continue
            take = min(remaining, count)
            selected_sq += float(sums[idx]) * take / count
            remaining -= take
            threshold = float(10 ** (SSE_LOG10_MIN + idx * width))
        if remaining > 0 and acc["sse_low_count"]:
            take = min(remaining, acc["sse_low_count"])
            selected_sq += acc["sse_low_sum"] * take / acc["sse_low_count"]
            remaining -= take
        tails[label] = {
            "n_pixels": int(np.ceil(n * frac)),
            "min_sse_approx": threshold,
            "total_sse_share_pct_approx": 100 * selected_sq / total_sq,
        }
    return {
        "method": "streaming_log10_histogram",
        "bin_width_dex": width,
        "per_pixel_dem_sse_quantiles_approx": quantiles,
        "upper_tail_sse_share_approx": tails,
        "max_per_pixel_dem_sse": acc["worst"]["sq"],
    }


def finish(acc):
    base = {
        "dem_mse": acc["sq"] / max(acc["n_bins"], 1),
        "dem_rel_err_pct": 100 * acc["rel"] / max(acc["n_bins"], 1),
        "w1_dex": acc["w1"] / max(acc["n_w1"], 1),
        "n_pixels": acc["n_px"],
        "n_w1_pixels": acc["n_w1"],
        "per_pixel_dem_sse_distribution": sse_distribution(acc),
    }
    w = acc["worst"]
    if acc["n_px"] <= 1:
        base["leave_one_worst_pixel_out"] = None
        return base
    # Remove all 18 bin errors belonging to the single largest-SSE pixel.
    loo_sq = (acc["sq"] - w["sq"]) / max(acc["n_bins"] - N_AIA_BINS, 1)
    loo_rel = 100 * (acc["rel"] - w["rel"]) / max(acc["n_bins"] - N_AIA_BINS, 1)
    loo_w1 = (acc["w1"] - (w["w1"] if w["w1_valid"] else 0.0)) / max(
        acc["n_w1"] - int(w["w1_valid"]), 1)
    base["leave_one_worst_pixel_out"] = {
        "dem_mse": loo_sq,
        "dem_rel_err_pct": loo_rel,
        "w1_dex": loo_w1,
        "worst_pixel": w,
        "worst_pixel_mse_share_pct": 100 * w["sq"] / max(acc["sq"], 1e-12),
    }
    return base


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
        add_metrics(acc["full"], pred, truth, valid, logt[:N_AIA_BINS], i)
        if thresholds is not None:
            bright = (obs >= thresholds[:, None, None]).any(axis=0) & valid
            add_metrics(acc["bright"], pred, truth, bright, logt[:N_AIA_BINS], i)
            add_metrics(acc["quiet"], pred, truth, valid & ~bright, logt[:N_AIA_BINS], i)
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
