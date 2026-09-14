#!/usr/bin/env python3
"""Predict Samuel's checkpoints on the exact AIA grid of an existing viewer export."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from student_package.table1_dem_metrics.compute_paper_table_metrics import (
    load_model, predict_regression_only,
)


@torch.inference_mode()
def predict(model, aia, pixel_batch=8192):
    # These supplied models are pointwise. Guard this assumption so a future
    # spatial checkpoint cannot silently lose its neighbourhood context.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) and (
            layer.kernel_size != (1, 1) or layer.stride != (1, 1)
            or layer.padding != (0, 0)
        ):
            raise ValueError("Full-disk point batching requires only 1x1 convolutions")
    if pixel_batch <= 0 or aia.ndim != 3 or aia.shape[0] != 6:
        raise ValueError("Expected positive pixel_batch and AIA shape [6,H,W]")
    h, w = aia.shape[1:]
    points = np.ascontiguousarray(aia.reshape(6, -1))
    result = np.empty((18, h * w), dtype=np.float32)
    device = next(model.parameters()).device
    for start in range(0, h * w, pixel_batch):
        stop = min(start + pixel_batch, h * w)
        batch = torch.from_numpy(points[:, start:stop]).to(device)[None, :, None, :]
        pred = predict_regression_only(model, batch)[0, :18, 0, :]
        if not torch.isfinite(pred).all():
            raise ValueError(f"Nonfinite supervised prediction at batch {start}:{stop}")
        result[:, start:stop] = pred.cpu().numpy()
        if start == 0 or stop == h * w or start % (pixel_batch * 100) == 0:
            print(f"  {stop:,}/{h*w:,} pixels", flush=True)
    return result.reshape(18, h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Existing label-free viewer export NPZ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pixel-batch", type=int, default=8192)
    args = parser.parse_args()
    with np.load(args.input) as data:
        aia = np.asarray(data["aia"], dtype=np.float32)
        logt = np.asarray(data["logT"])
    if logt.shape != (18,) or not np.isfinite(aia).all():
        raise ValueError("Expected an 18-bin logT grid and finite AIA observations")
    model, meta = load_model(args.model, torch.device(args.device))
    print(f"Checkpoint: {args.model}; metadata={meta}; AIA={aia.shape}", flush=True)
    dem = predict(model, aia, args.pixel_batch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Complete output replaces any previous export only after writing succeeds.
    temporary = args.output + ".tmp.npz"
    np.savez_compressed(temporary, dem=dem, aia=aia, logT=logt,
                        model=np.array(os.path.abspath(args.model)),
                        input=np.array(os.path.abspath(args.input)))
    os.replace(temporary, args.output)
    print(f"Wrote {args.output}: {dem.shape}", flush=True)


if __name__ == "__main__":
    main()
