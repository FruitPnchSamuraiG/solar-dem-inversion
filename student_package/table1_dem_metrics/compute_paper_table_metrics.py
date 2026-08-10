#!/usr/bin/env python3
"""
compute_paper_table_metrics.py  (minimal version)

Populates tab:comparison — for one (model, reference-solver dataset) pair,
computes DEM MSE, DEM Rel. Err. (%), and W1 (dex) against the reference
solver's ground truth, stratified by Full / Bright / Quiet pixels, over the
full test split. This is the trimmed-down version: only the three metrics
actually cited in the paper table are computed (the full lab codebase has
extra relative-error variants used only for internal sanity checks).

Metric definitions:
  - DEM MSE: mean over (valid pixels x 18 bins) of (pred - gt)^2
  - DEM Rel. Err. (%): mean over (valid pixels x 18 bins) of
             |pred - gt| / (|gt| + rel_floor), rel_floor=0.1
  - W1 (dex): per-pixel 1-D Wasserstein distance between pred and gt DEM,
             treated as distributions over the shared logT grid (18 bins,
             5.5-7.2, 0.1 dex spacing). Computed via the closed-form CDF
             L1-distance (exact match to scipy.stats.wasserstein_distance
             when both distributions share support points), vectorized —
             a per-pixel scipy loop is infeasible at ~3.2e9 pixels.

Bright/quiet: any AIA channel >= that channel's own top-5% intensity
threshold, threshold computed from the SAME test split being evaluated
(not borrowed from another split/dataset).

Usage:
    python3 compute_paper_table_metrics.py \
        --model results/models/<run>/model_best.pth \
        --data /scratch/vp2435/workspace/dem/data/bp_AIA_hofdeconv_full_DS \
        --rdata RData.npz \
        --variant methodbp --reference "Basis Pursuit (BP)" \
        --output table_comparison_bp.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import zarr

# NOTE: this script expects to be run from (or with this repo's root on
# PYTHONPATH alongside) the demdemo repo, since it imports from src/. Adjust
# this path if you relocate the script relative to the repo root.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model import FreqClassNonReLU, FreqClassNonReLUNoPos, BasicNetworkFreqClass

_MODEL_REGISTRY = {
    'FreqClassNonReLU': FreqClassNonReLU,
    'FreqClassNonReLUNoPos': FreqClassNonReLUNoPos,
    'BasicNetworkFreqClass': BasicNetworkFreqClass,
}

N_BINS = 18
TOP_PCT = 5.0
REL_FLOOR = 0.1
EPS = 1e-12


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_name = ckpt['model_name']
    model_args = ckpt.get('args', {})
    if hasattr(model_args, '__dict__'):
        model_args = vars(model_args)
    ModelClass = _MODEL_REGISTRY[model_name]
    n_bins = model_args.get('n_bins', 64)
    model = ModelClass(nIn=6, nOut=26, n_bins=n_bins)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, {'model_name': model_name, 'n_bins': int(n_bins), 'epoch': int(ckpt.get('epoch', -1))}


def compute_thresholds(x_zarr, top_pct=TOP_PCT, n_hist_bins=500_000, hist_min=-300.0, hist_max=30_000.0, bs=64):
    """Streaming per-channel top-pct% threshold over the full split, so this
    script is self-contained and always thresholds against the exact split
    it evaluates."""
    N = x_zarr.shape[3]
    edges = np.linspace(hist_min, hist_max, n_hist_bins + 1, dtype=np.float64)
    bin_width = edges[1] - edges[0]
    counts = np.zeros((6, n_hist_bins), dtype=np.int64)
    finite_counts = np.zeros(6, dtype=np.int64)

    for i in range(0, N, bs):
        i1 = min(i + bs, N)
        aia = x_zarr[:, :, :, i:i1].transpose(3, 0, 1, 2).astype(np.float32)
        px = aia.transpose(0, 2, 3, 1).reshape(-1, 6)
        for c in range(6):
            vals = px[:, c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            finite_counts[c] += vals.size
            clipped = np.clip(vals, hist_min, np.nextafter(hist_max, hist_min))
            bin_idx = ((clipped - hist_min) / bin_width).astype(np.int64)
            counts[c] += np.bincount(bin_idx, minlength=n_hist_bins)

    thresholds = np.zeros(6, dtype=np.float32)
    for c in range(6):
        n_total = int(finite_counts[c])
        K = max(1, int(np.ceil(n_total * top_pct / 100.0)))
        rev_cdf = np.cumsum(counts[c][::-1])
        rev_idx = int(np.searchsorted(rev_cdf, K, side='left'))
        idx = n_hist_bins - 1 - min(rev_idx, n_hist_bins - 1)
        thresholds[c] = edges[idx]
    return thresholds


def upsample_y_with_nan(dem_np, target_h, target_w):
    """Replicates zarrDataset.__getitem__'s handling of the 'compact' Y format:
    Y is stored at native (lower) DEM-solver resolution when the solver only
    runs on a decimated grid (e.g. BP is expensive, solved on every dec-th
    pixel). Placed back on the X grid at every dec-th pixel; all other pixels
    are NaN (not interpolated) so they're excluded from the valid mask.
    dem_np: [C, h, w, B] (native resolution). Returns [C, target_h, target_w, B]."""
    C, h, w, B = dem_np.shape
    if h == target_h and w == target_w:
        return dem_np
    assert target_h % h == 0 and target_w % w == 0, \
        f"Y spatial size {(h, w)} must divide X spatial size {(target_h, target_w)}"
    dec = target_h // h
    full = np.full((C, target_h, target_w, B), np.nan, dtype=dem_np.dtype)
    full[:, ::dec, ::dec, :] = dem_np
    return full


def w1_per_pixel(pred, gt, dT, eps=EPS):
    """pred, gt: [B, 18, H, W] tensors (already masked to nonneg/finite where valid).
    dT: [17] tensor of logT bin spacings.
    Returns [B, H, W] W1 distance in dex. Caller must apply a validity mask
    (sum(pred) > 0 and sum(gt) > 0) since this normalizes by the sums."""
    p = pred / torch.clamp(pred.sum(dim=1, keepdim=True), min=eps)
    q = gt / torch.clamp(gt.sum(dim=1, keepdim=True), min=eps)
    cdf_p = torch.cumsum(p, dim=1)
    cdf_q = torch.cumsum(q, dim=1)
    return (torch.abs(cdf_p - cdf_q)[:, :-1] * dT[None, :, None, None]).sum(dim=1)


def _accum():
    return {'sq_err': 0.0, 'rel_err': 0.0, 'w1_sum': 0.0,
            'n_bins_valid': 0, 'n_pixels_valid': 0, 'n_w1_valid': 0}


@torch.no_grad()
def evaluate(model, x_zarr, y_zarr, thresholds, logT, device, batch_size, rel_floor=REL_FLOOR, n_samples=None):
    N = x_zarr.shape[3]
    if n_samples is not None:
        N = min(N, n_samples)

    dT = torch.from_numpy(np.diff(logT).astype(np.float32)).to(device)
    thr_t = torch.from_numpy(thresholds).to(device)

    acc = {k: _accum() for k in ('all', 'bright', 'quiet')}

    for i in range(0, N, batch_size):
        i1 = min(i + batch_size, N)
        aia_raw = x_zarr[:, :, :, i:i1].astype(np.float32)             # [6, H, W, B]
        dem_raw = y_zarr[:, :, :, i:i1].astype(np.float32)[:N_BINS]    # [18, h, w, B] (h,w may be < H,W)
        dem_raw = upsample_y_with_nan(dem_raw, aia_raw.shape[1], aia_raw.shape[2])

        aia_np = aia_raw.transpose(3, 0, 1, 2)   # [B,6,H,W]
        dem_np = dem_raw.transpose(3, 0, 1, 2)   # [B,18,H,W]

        aia_t = torch.from_numpy(aia_np).to(device)
        gt_t = torch.from_numpy(dem_np).to(device)
        aia_in = torch.clamp(aia_t, min=0.0)

        out = model(aia_in)
        pred_t = (out[0] if isinstance(out, tuple) else out)[:, :N_BINS]

        pixel_valid = torch.isfinite(gt_t).all(dim=1) & torch.isfinite(pred_t).all(dim=1)  # [B,H,W]
        bright = (aia_t >= thr_t[None, :, None, None]).any(dim=1) & pixel_valid
        quiet = ~(aia_t >= thr_t[None, :, None, None]).any(dim=1) & pixel_valid

        gt_clean = gt_t.nan_to_num(0.0)
        pred_clean = pred_t.nan_to_num(0.0)

        abs_err = (pred_clean - gt_clean).abs()
        sq_err = (pred_clean - gt_clean) ** 2
        rel_err = abs_err / (gt_clean.abs() + rel_floor)

        # W1 needs strictly-positive mass in both pred and gt to be well-defined
        w1_valid = pixel_valid & (gt_clean.sum(dim=1) > 0) & (pred_clean.sum(dim=1) > 0)
        w1_map = w1_per_pixel(pred_clean.clamp(min=0.0), gt_clean.clamp(min=0.0), dT)  # [B,H,W]

        for mask, key in ((pixel_valid, 'all'), (bright, 'bright'), (quiet, 'quiet')):
            m4d = mask[:, None, :, :]
            acc[key]['sq_err'] += (sq_err * m4d).sum().item()
            acc[key]['rel_err'] += (rel_err * m4d).sum().item()
            acc[key]['n_bins_valid'] += int(mask.sum().item()) * N_BINS
            acc[key]['n_pixels_valid'] += int(mask.sum().item())

            w1_mask = mask & w1_valid
            acc[key]['w1_sum'] += (w1_map * w1_mask).sum().item()
            acc[key]['n_w1_valid'] += int(w1_mask.sum().item())

        if (i // batch_size) % 200 == 0:
            print(f'  {i1}/{N} patches ({100 * i1 / N:.1f}%)')

    results = {}
    for key, a in acc.items():
        n_bins = max(a['n_bins_valid'], 1)
        results[key] = {
            'dem_mse': a['sq_err'] / n_bins,
            'dem_rel_err_pct': 100.0 * a['rel_err'] / n_bins,
            'w1_dex': a['w1_sum'] / max(a['n_w1_valid'], 1),
            'n_pixels': a['n_pixels_valid'],
            'n_w1_pixels': a['n_w1_valid'],
        }
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, help='dir with test_x.zarr / test_y.zarr (reference-solver DEM as GT)')
    p.add_argument('--rdata', default='RData.npz')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--rel_floor', type=float, default=REL_FLOOR)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--variant', required=True, help='e.g. methodbp, methoden (row label)')
    p.add_argument('--reference', required=True, help='e.g. "Basis Pursuit (BP)", "ElasticNet"')
    p.add_argument('--thresholds_json', default=None,
                   help='optional precomputed aia_thresholds.json; must contain a "test" key '
                        'computed on THIS --data. If omitted, computed fresh from --data test_x.zarr.')
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    logT = np.load(args.rdata)['logT'][:N_BINS].astype(np.float64)

    x_zarr = zarr.open(os.path.join(args.data, 'test_x.zarr'), mode='r')
    y_zarr = zarr.open(os.path.join(args.data, 'test_y.zarr'), mode='r')
    print(f'test patches: {x_zarr.shape[3]}')

    if args.thresholds_json is not None:
        with open(args.thresholds_json) as f:
            thresholds = np.array(json.load(f)['test'], dtype=np.float32)
        print(f'loaded thresholds: {thresholds.round(3)}')
    else:
        print('computing AIA top-5% thresholds from this test split ...')
        thresholds = compute_thresholds(x_zarr)
        print(f'thresholds: {thresholds.round(3)}')

    model, meta = load_model(args.model, device)
    print(f'loaded model: {meta}')

    results = evaluate(
        model, x_zarr, y_zarr, thresholds, logT, device,
        batch_size=args.batch_size, rel_floor=args.rel_floor, n_samples=args.n_samples,
    )

    print(f"\n=== {args.variant} vs {args.reference} ===")
    for stratum_label, key in [('Full', 'all'), ('Bright', 'bright'), ('Quiet', 'quiet')]:
        r = results[key]
        print(f"  {stratum_label:6s}  DEM MSE={r['dem_mse']:.4f}  Rel.Err={r['dem_rel_err_pct']:.1f}%  "
              f"W1={r['w1_dex']:.4f} dex  (n={r['n_pixels']:,}, n_w1={r['n_w1_pixels']:,})")

    payload = {
        'variant': args.variant,
        'reference': args.reference,
        'model': os.path.abspath(args.model),
        'data': os.path.abspath(args.data),
        'thresholds': thresholds.tolist(),
        'rows': [
            {'pixel_type': label, **results[key]}
            for label, key in [('Full', 'all'), ('Bright', 'bright'), ('Quiet', 'quiet')]
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nsaved -> {args.output}')


if __name__ == '__main__':
    main()
