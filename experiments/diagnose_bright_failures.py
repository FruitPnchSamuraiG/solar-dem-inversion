#!/usr/bin/env python3
"""Explain where the label-free model disagrees with a solver reference.

This is deliberately a diagnostic, not another scorecard.  Bright is Samuel's
union mask (any of six observed AIA values crosses its top-5% threshold).  The
script preserves that full-population accounting, while also reporting the
individual threshold patterns and a reproducible sample of DEM peak shapes.

Peak shape is sampled because prominence-aware peak finding is per-curve.  The
error/trigger accounting is nevertheless complete over every requested block.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.eval_full_paper_test import N_AIA_BINS
from experiments.train_scaled import load_operators
from src.scaled_eval import load_scaled_model

CHANNELS = ("94", "131", "171", "193", "211", "335")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--reference", required=True, choices=("bp", "enet"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--output", required=True)
    p.add_argument("--pixel-batch", type=int, default=8192)
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--shape-samples-per-block", type=int, default=64,
                   help="valid pixels sampled per block for peak-shape analysis")
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


@torch.no_grad()
def predict_block(model, basis_t, x_block, pixel_batch):
    """Return [18,128,128] DEM prediction and aligned [6,128,128] AIA."""
    obs = np.asarray(x_block[:, ::2, ::2], dtype=np.float32)
    h, w = obs.shape[1:]
    rows, cols = np.indices((h, w))
    rows, cols = rows.ravel(), cols.ravel()
    out = np.empty((len(rows), N_AIA_BINS), dtype=np.float32)
    p = 4
    padded = np.pad(obs, ((0, 0), (p, p), (p, p)), mode="edge")
    offsets = np.arange(9, dtype=np.int64)
    for start in range(0, len(rows), pixel_batch):
        stop = min(start + pixel_batch, len(rows))
        rr = rows[start:stop, None] + offsets
        cc = cols[start:stop, None] + offsets
        patch = padded[:, rr[:, :, None], cc[:, None, :]]
        patch = np.ascontiguousarray(patch.transpose(1, 0, 2, 3))
        coeff = model(torch.from_numpy(patch).to(basis_t.device, non_blocking=True))
        out[start:stop] = (coeff @ basis_t.T)[:, :N_AIA_BINS].cpu().numpy()
    return out.reshape(h, w, N_AIA_BINS).transpose(2, 0, 1), obs


def empty_error():
    return {"n_pixels": 0, "dem_sse": 0.0,
            "signed_error_by_logt": np.zeros(N_AIA_BINS, dtype=np.float64),
            "absolute_error_by_logt": np.zeros(N_AIA_BINS, dtype=np.float64),
            "squared_error_by_logt": np.zeros(N_AIA_BINS, dtype=np.float64)}


def add_error(acc, diff, mask):
    if not mask.any():
        return
    d = diff[:, mask].astype(np.float64, copy=False)
    acc["n_pixels"] += int(d.shape[1])
    acc["dem_sse"] += float(np.square(d).sum())
    acc["signed_error_by_logt"] += d.sum(axis=1)
    acc["absolute_error_by_logt"] += np.abs(d).sum(axis=1)
    acc["squared_error_by_logt"] += np.square(d).sum(axis=1)


def finish_error(acc, total_sse):
    n = max(acc["n_pixels"], 1)
    return {
        "n_pixels": acc["n_pixels"],
        "dem_mse": acc["dem_sse"] / (n * N_AIA_BINS),
        "dem_sse_share_pct": 100 * acc["dem_sse"] / max(total_sse, 1e-300),
        "mean_signed_error_by_logt": (acc["signed_error_by_logt"] / n).tolist(),
        "mae_by_logt": (acc["absolute_error_by_logt"] / n).tolist(),
        "mse_by_logt": (acc["squared_error_by_logt"] / n).tolist(),
    }


def empty_shape():
    return {"n": 0, "ref_multimodal": 0, "model_multimodal": 0,
            "ref_multi_model_unimodal": 0, "model_multi_ref_unimodal": 0,
            "sse": 0.0}


def interior_peak_count(curve, prominence=0.15):
    """Prominent local maxima excluding poorly constrained boundary bins."""
    from scipy.signal import find_peaks
    curve = np.asarray(curve, dtype=np.float64)
    if not np.isfinite(curve).all() or curve.max() <= 0:
        return 0
    # Interior only: positions 1..16. This avoids treating endpoint pile-up as
    # a separate thermal component.
    peaks, _ = find_peaks(curve, prominence=prominence * curve.max())
    return int(((peaks >= 1) & (peaks <= len(curve) - 2)).sum())


def add_shapes(acc, pred, truth, sse, indices):
    for idx in indices:
        r_multi = interior_peak_count(truth[:, idx]) >= 2
        m_multi = interior_peak_count(pred[:, idx]) >= 2
        acc["n"] += 1
        acc["ref_multimodal"] += int(r_multi)
        acc["model_multimodal"] += int(m_multi)
        acc["ref_multi_model_unimodal"] += int(r_multi and not m_multi)
        acc["model_multi_ref_unimodal"] += int(m_multi and not r_multi)
        acc["sse"] += float(sse[idx])


def shape_result(acc):
    n = max(acc["n"], 1)
    return {"n_sampled": acc["n"],
            "reference_multimodal_pct": 100 * acc["ref_multimodal"] / n,
            "model_multimodal_pct": 100 * acc["model_multimodal"] / n,
            "reference_multi_model_unimodal_pct": 100 * acc["ref_multi_model_unimodal"] / n,
            "model_multi_reference_unimodal_pct": 100 * acc["model_multi_ref_unimodal"] / n,
            "sample_mean_per_pixel_sse": acc["sse"] / n}


def mask_name(bits):
    selected = [f"{CHANNELS[c]}A" for c in range(6) if bits & (1 << c)]
    return "+".join(selected) if selected else "quiet"


def main():
    args = parse_args()
    import zarr

    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload), dtype=np.float32)
    if thresholds.shape != (6,):
        raise ValueError("thresholds must be six values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, basis_t, n_basis, logt = load_operators(device)
    model, ckpt = load_scaled_model(args.model, n_basis, device)
    x = zarr.open(os.path.join(args.data, "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(args.data, "test_y.zarr"), mode="r")
    n_blocks = min(x.shape[-1], args.max_blocks) if args.max_blocks else x.shape[-1]
    rng = np.random.default_rng(args.seed)

    all_groups = {name: empty_error() for name in ("full", "bright", "quiet")}
    exact = {bits: empty_error() for bits in range(1, 64)}
    inclusive = {channel: empty_error() for channel in CHANNELS}
    active_count = {str(k): empty_error() for k in range(1, 7)}
    largest_ratio = {channel: empty_error() for channel in CHANNELS}
    shapes = {name: empty_shape() for name in ("full", "bright", "quiet")}
    shape_active = {str(k): empty_shape() for k in range(1, 7)}

    print(f"device={device}; blocks={n_blocks:,}; shape samples/block={args.shape_samples_per_block}")
    for block in range(n_blocks):
        pred, obs = predict_block(model, basis_t, x[:, :, :, block], args.pixel_batch)
        truth = np.asarray(y[:N_AIA_BINS, :, :, block], dtype=np.float32)
        valid = np.isfinite(pred).all(axis=0) & np.isfinite(truth).all(axis=0) & np.isfinite(obs).all(axis=0)
        diff = pred - truth
        hit = obs >= thresholds[:, None, None]
        bits = np.tensordot((1 << np.arange(6, dtype=np.uint8)), hit.astype(np.uint8), axes=(0, 0)).astype(np.uint8)
        bright = valid & (bits != 0)
        quiet = valid & ~bright
        add_error(all_groups["full"], diff, valid)
        add_error(all_groups["bright"], diff, bright)
        add_error(all_groups["quiet"], diff, quiet)
        for bit in range(1, 64):
            add_error(exact[bit], diff, valid & (bits == bit))
        for c, channel in enumerate(CHANNELS):
            add_error(inclusive[channel], diff, valid & hit[c])
        active = hit.sum(axis=0)
        ratios = obs / thresholds[:, None, None]
        lead = np.argmax(ratios, axis=0)
        for k in range(1, 7):
            add_error(active_count[str(k)], diff, valid & (active == k))
        for c, channel in enumerate(CHANNELS):
            add_error(largest_ratio[channel], diff, bright & (lead == c))

        valid_flat = np.flatnonzero(valid.ravel())
        take = min(args.shape_samples_per_block, len(valid_flat))
        if take:
            chosen = rng.choice(valid_flat, size=take, replace=False)
            p = pred.reshape(N_AIA_BINS, -1)
            t = truth.reshape(N_AIA_BINS, -1)
            per_sse = np.square(p[:, chosen] - t[:, chosen]).sum(axis=0)
            bright_flat = bright.ravel()[chosen]
            active_flat = active.ravel()[chosen]
            add_shapes(shapes["full"], p[:, chosen], t[:, chosen], per_sse,
                       np.arange(take))
            add_shapes(shapes["bright"], p[:, chosen], t[:, chosen], per_sse,
                       np.flatnonzero(bright_flat))
            add_shapes(shapes["quiet"], p[:, chosen], t[:, chosen], per_sse,
                       np.flatnonzero(~bright_flat))
            for k in range(1, 7):
                idx = np.flatnonzero(active_flat == k)
                add_shapes(shape_active[str(k)], p[:, chosen], t[:, chosen], per_sse, idx)
        if block % 100 == 0 or block + 1 == n_blocks:
            print(f"  {block + 1:,}/{n_blocks:,}", flush=True)

    total_sse = all_groups["full"]["dem_sse"]
    exact_out = {mask_name(bits): finish_error(acc, total_sse) for bits, acc in exact.items() if acc["n_pixels"]}
    output = {
        "purpose": "Bright-failure association diagnostic; not a causal inference",
        "model": os.path.abspath(args.model), "reference": args.reference,
        "checkpoint": {k: ckpt.get(k) for k in ("variant", "loss", "hidden", "epoch")},
        "data": os.path.abspath(args.data), "n_blocks": n_blocks,
        "channels_angstrom": [int(c) for c in CHANNELS], "bright_thresholds": thresholds.tolist(),
        "definition": {"bright": "any observed AIA channel >= its threshold", "quiet": "valid complement",
                       "shape": "interior prominent peaks only; >=2 is multimodal",
                       "shape_prominence_fraction": 0.15,
                       "shape_sampling": f"uniform {args.shape_samples_per_block} valid pixels per block, seed {args.seed}"},
        "full_population": {k: finish_error(v, total_sse) for k, v in all_groups.items()},
        "bright_inclusive_by_trigger_channel": {k: finish_error(v, total_sse) for k, v in inclusive.items()},
        "bright_exact_trigger_pattern": exact_out,
        "bright_by_number_of_trigger_channels": {k: finish_error(v, total_sse) for k, v in active_count.items()},
        "bright_by_largest_intensity_over_threshold_channel": {k: finish_error(v, total_sse) for k, v in largest_ratio.items()},
        "shape_sample": {k: shape_result(v) for k, v in shapes.items()},
        "shape_sample_by_number_of_trigger_channels": {k: shape_result(v) for k, v in shape_active.items()},
        "logt": logt[:N_AIA_BINS].tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
