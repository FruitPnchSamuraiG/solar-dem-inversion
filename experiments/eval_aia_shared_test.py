#!/usr/bin/env python3
"""Paired AIA reconstruction metrics on the shared test grid, not viewer frames.

Retains all shared-test blocks (including repeated clean inputs for the five
solver targets). The reference-finite mask therefore has the same population
as DEM evaluation. Both models are scored on one common mask per track.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def accumulate(acc, reconstruction, observed, mask):
    residual = reconstruction[:, mask].astype(np.float64) - observed[:, mask]
    acc['abs'] += np.abs(residual).sum(axis=1)
    acc['sq'] += np.square(residual).sum(axis=1)
    acc['n_pixels'] += int(mask.sum())


def finish(acc):
    n = acc['n_pixels']
    return {'n_pixels': n, 'n_channel_pixels': 6 * n,
            'mae': float(acc['abs'].sum() / (6 * n)) if n else None,
            'mse': float(acc['sq'].sum() / (6 * n)) if n else None,
            'per_channel_mae': (acc['abs'] / n).tolist() if n else None,
            'per_channel_mse': (acc['sq'] / n).tolist() if n else None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=['bp', 'enet'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--label-free', required=True)
    parser.add_argument('--supervised', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--pixel-batch', type=int, default=8192)
    args = parser.parse_args()
    if args.max_blocks is not None and args.max_blocks <= 0:
        parser.error('--max-blocks must be positive')
    from experiments.eval_full_paper_test import predict_block
    from experiments.export_supervised_full_disk import predict
    from experiments.train_scaled import load_operators
    from src.scaled_eval import load_scaled_model
    from student_package.table1_dem_metrics.compute_paper_table_metrics import load_model
    device = torch.device(args.device)
    _, basis, n_basis, _ = load_operators(device)
    label_free, _ = load_scaled_model(args.label_free, n_basis, device)
    supervised, _ = load_model(args.supervised, device)
    with np.load('RData.npz') as rdata:
        response = np.asarray(rdata['R'][:, :18], dtype=np.float64) * 1e26
    if response.shape != (6, 18):
        raise ValueError(f'Unexpected AIA response shape: {response.shape}')
    with open('eval_and_enet_specs/aia_thresholds.json') as stream:
        thresholds = np.asarray(json.load(stream)['test'], dtype=np.float32)
    x = zarr.open(str(Path(args.data) / 'test_x.zarr'), mode='r')
    y = zarr.open(str(Path(args.data) / 'test_y.zarr'), mode='r')
    if x.shape[:3] != (6, 256, 256) or y.shape[:3] != (26, 128, 128) or x.shape[-1] != y.shape[-1]:
        raise ValueError(f'Unexpected shared-test shapes: {x.shape}, {y.shape}')
    count = min(x.shape[-1], args.max_blocks or x.shape[-1])
    acc = {model: {group: {'abs': np.zeros(6), 'sq': np.zeros(6), 'n_pixels': 0}
                   for group in ('full', 'bright', 'quiet')}
           for model in ('label_free', 'supervised')}
    excluded = 0
    for i in range(count):
        dem, observed = predict_block(label_free, basis, x[:, :, :, i], args.pixel_batch)
        # Supplied supervised models are pointwise; predict() verifies this.
        sup_dem = predict(supervised, observed, args.pixel_batch, progress=False)
        reference = np.asarray(y[:18, :, :, i], dtype=np.float32)
        reconstructions = {name: np.einsum('ct,thw->chw', response, values)
                           for name, values in [('label_free', dem), ('supervised', sup_dem)]}
        base = np.isfinite(reference).all(axis=0) & np.isfinite(observed).all(axis=0)
        valid = base.copy()
        for reconstruction in reconstructions.values():
            valid &= np.isfinite(reconstruction).all(axis=0)
        excluded += int((base & ~valid).sum())
        bright = (observed >= thresholds[:, None, None]).any(axis=0)
        for name, reconstruction in reconstructions.items():
            for group, mask in [('full', valid), ('bright', valid & bright), ('quiet', valid & ~bright)]:
                accumulate(acc[name][group], reconstruction, observed, mask)
        if i % 100 == 0 or i + 1 == count:
            print(f'{i+1:,}/{count:,} blocks', flush=True)
    payload = {'track': args.track, 'data': os.path.abspath(args.data),
               'checkpoints': {'label_free': os.path.abspath(args.label_free),
                               'supervised': os.path.abspath(args.supervised)},
               'n_blocks': count, 'available_blocks': int(x.shape[-1]),
               'complete_test': count == x.shape[-1],
               'scope': 'shared test blocks including five targets per clean input; stride-2 AIA grid',
               'mask': 'finite 18-bin reference, six observations, and both reconstructions; identical within track',
               'excluded_nonfinite_prediction_pixels': excluded,
               'channels_angstrom': [94, 131, 171, 193, 211, 335],
               'bright_thresholds': thresholds.tolist(),
               'rows': {name: {group: finish(value) for group, value in groups.items()}
                        for name, groups in acc.items()}}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, destination)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
