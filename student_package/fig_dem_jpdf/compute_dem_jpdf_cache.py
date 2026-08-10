#!/usr/bin/env python3
"""
compute_dem_jpdf_cache.py

GPU pass for fig:dem_jpdf — streams a model's predictions against its
reference solver's ground truth over the full test split, pools every
(AIA-sensitive bin, valid pixel) pair into one log-log 2D histogram, and
caches it. Mirrors scripts/compute_paper_table_metrics.py's model/data
loading (same compact-Y-format handling via upsample_y_with_nan) and
scripts/dumpVisuals.py::dumpDEMJointPDF's masking/quantity convention
(clip to a positivity floor, log10 both axes) but pools across all 18 bins
and the whole test set into a single density instead of per-bin/per-patch
plots.

Usage:
    python3 scripts/compute_dem_jpdf_cache.py \
        --model results/models/<run>/model_best.pth \
        --data /scratch/vp2435/workspace/dem/data/bp_AIA_hofdeconv_full_DS \
        --variant methodbp \
        --output paperplots/cache/dem_jpdf_methodbp.npz
"""
import argparse
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
TR_MASK = 1e-1     # positivity floor, matches dumpDEMJointPDF's tr_mask
FLOOR = 1e-8       # matches dumpDEMJointPDF's np.maximum floor before masking


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


def upsample_y_with_nan(dem_np, target_h, target_w):
    """Same as compute_paper_table_metrics.py — places compact (lower-res)
    solver ground truth back onto the AIA grid at every dec-th pixel, NaN
    elsewhere (no interpolation)."""
    C, h, w, B = dem_np.shape
    if h == target_h and w == target_w:
        return dem_np
    assert target_h % h == 0 and target_w % w == 0, \
        f"Y spatial size {(h, w)} must divide X spatial size {(target_h, target_w)}"
    dec = target_h // h
    full = np.full((C, target_h, target_w, B), np.nan, dtype=dem_np.dtype)
    full[:, ::dec, ::dec, :] = dem_np
    return full


@torch.no_grad()
def accumulate_histogram(model, x_zarr, y_zarr, device, batch_size, edges, n_samples=None):
    N = x_zarr.shape[3]
    if n_samples is not None:
        N = min(N, n_samples)

    H2D = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    log_edges = np.log10(edges)
    n_valid = 0
    n_total = 0
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0

    for i in range(0, N, batch_size):
        i1 = min(i + batch_size, N)
        aia_raw = x_zarr[:, :, :, i:i1].astype(np.float32)             # [6, H, W, B]
        dem_raw = y_zarr[:, :, :, i:i1].astype(np.float32)[:N_BINS]    # [18, h, w, B]
        dem_raw = upsample_y_with_nan(dem_raw, aia_raw.shape[1], aia_raw.shape[2])

        aia_np = aia_raw.transpose(3, 0, 1, 2)   # [B,6,H,W]
        dem_np = dem_raw.transpose(3, 0, 1, 2)   # [B,18,H,W]

        aia_t = torch.from_numpy(aia_np).to(device)
        gt_t = torch.from_numpy(dem_np).to(device)
        aia_in = torch.clamp(aia_t, min=0.0)

        out = model(aia_in)
        pred_t = (out[0] if isinstance(out, tuple) else out)[:, :N_BINS]

        gt = gt_t.cpu().numpy().ravel()
        pd = pred_t.cpu().numpy().ravel()
        n_total += gt.size

        finite = np.isfinite(gt) & np.isfinite(pd)
        gt = np.maximum(gt[finite], FLOOR)
        pd = np.maximum(pd[finite], FLOOR)

        mask = (gt > TR_MASK) & (pd > TR_MASK)
        gt_v, pd_v = gt[mask], pd[mask]
        n_valid += gt_v.size

        if gt_v.size:
            x = np.log10(gt_v)
            y = np.log10(pd_v)
            h, _, _ = np.histogram2d(x, y, bins=[log_edges, log_edges])
            H2D += h
            sum_x += x.sum(); sum_y += y.sum()
            sum_x2 += (x * x).sum(); sum_y2 += (y * y).sum()
            sum_xy += (x * y).sum()

        if (i // batch_size) % 200 == 0:
            print(f'  {i1}/{N} patches ({100 * i1 / N:.1f}%)  valid so far: {n_valid:,}', flush=True)

    # R^2 from accumulated moments (equivalent to np.corrcoef(x,y)[0,1]**2 over the pooled set)
    n = max(n_valid, 1)
    mean_x, mean_y = sum_x / n, sum_y / n
    cov_xy = sum_xy / n - mean_x * mean_y
    var_x = sum_x2 / n - mean_x ** 2
    var_y = sum_y2 / n - mean_y ** 2
    r2 = (cov_xy ** 2) / (var_x * var_y) if var_x > 0 and var_y > 0 else 0.0

    return H2D, n_valid, n_total, r2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, help='dir with test_x.zarr / test_y.zarr')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--variant', required=True, help='e.g. methodbp, methoden')
    p.add_argument('--lo', type=float, default=1e-1, help='histogram lower edge (native/scaled DEM units)')
    p.add_argument('--hi', type=float, default=1e4, help='histogram upper edge (native/scaled DEM units)')
    p.add_argument('--n_hist_bins', type=int, default=150)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    x_zarr = zarr.open(os.path.join(args.data, 'test_x.zarr'), mode='r')
    y_zarr = zarr.open(os.path.join(args.data, 'test_y.zarr'), mode='r')
    print(f'test patches: {x_zarr.shape[3]}')

    model, meta = load_model(args.model, device)
    print(f'loaded model: {meta}')

    edges = np.logspace(np.log10(args.lo), np.log10(args.hi), args.n_hist_bins + 1)

    H2D, n_valid, n_total, r2 = accumulate_histogram(
        model, x_zarr, y_zarr, device, args.batch_size, edges, n_samples=args.n_samples)

    print(f'\n=== {args.variant} ===')
    print(f'valid pixel-bin pairs: {n_valid:,} / {n_total:,}  R^2={r2:.4f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        H2D=H2D, edges=edges, r2=r2, n_valid=n_valid, n_total=n_total,
        variant=args.variant, model=os.path.abspath(args.model), data=os.path.abspath(args.data),
    )
    print(f'saved -> {args.output}')


if __name__ == '__main__':
    main()
