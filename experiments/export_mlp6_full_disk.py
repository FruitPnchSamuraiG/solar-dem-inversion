#!/usr/bin/env python3
"""Export an AIA-only mlp6 checkpoint as an 18-bin full-disk DEM cube.

This bridges our scaled unsupervised checkpoints to Samuel's static visualizer.
The solver and training data use a stride-2 DEM grid: a 4096x4096 AIA image
becomes a 2048x2048 DEM map.  ``mlp6`` uses only the centre six-channel AIA
observation, so inference is a batched matrix of those grid pixels; it does not
need the patch-CNN's 9x9 neighbourhood extraction.

Input is one of the compressed full-disk NPZ files used by the staging pipeline
(``AIACube`` + ``AIACubeShape``), typically also containing ``DEMCube`` for the
matching BP/ENet solver reference.  The output is an ordinary NPZ, consumable by
``student_package/visuals_pipeline/dumpVisuals.py`` through its ``dem``/``aia``
keys and by a small renderer wrapper.
"""
import argparse
import os
import sys

import numpy as np
import torch
from numcodecs import Blosc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.train_scaled import load_operators
from src.scaled_eval import load_scaled_model
from src.zarr_data import N_AIA_BINS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="scaled mlp6 h232 checkpoint")
    p.add_argument("--input_npz", required=True, help="full-disk staged solver NPZ")
    p.add_argument("--output", required=True)
    p.add_argument("--pixel_batch", type=int, default=65_536)
    p.add_argument("--stride", type=int, default=2,
                   help="AIA-to-DEM decimation used by the solver (normally 2)")
    return p.parse_args()


def decode_array(npz, key, shape_key):
    """Read either a plain floating array or the project's Blosc NPZ payload."""
    value = npz[key]
    if value.dtype in (np.float32, np.float64):
        return np.asarray(value, dtype=np.float32)
    raw = value.item() if value.ndim == 0 else value.tobytes()
    return np.frombuffer(Blosc(cname="zstd", clevel=4, shuffle=2).decode(raw),
                         dtype=np.float32).reshape(tuple(npz[shape_key]))


@torch.no_grad()
def predict(model, basis_t, aia, pixel_batch):
    """Predict a [18,H,W] cube from AIA already aligned to the DEM grid."""
    h, w = aia.shape[1:]
    points = np.ascontiguousarray(aia.transpose(1, 2, 0).reshape(-1, 6))
    out = np.empty((len(points), N_AIA_BINS), dtype=np.float32)
    device = basis_t.device
    for start in range(0, len(points), pixel_batch):
        stop = min(start + pixel_batch, len(points))
        # CenterMLP indexes the centre of a 9x9 input patch. Repeating each
        # centre observation across that patch is exactly equivalent because
        # mlp6 reads only the centre.
        centre = torch.from_numpy(points[start:stop]).to(device, non_blocking=True)
        coeff = model(centre[:, :, None, None].expand(-1, -1, 9, 9))
        out[start:stop] = (coeff @ basis_t.T)[:, :N_AIA_BINS].cpu().numpy()
        if start == 0 or stop == len(points) or (start // pixel_batch) % 20 == 0:
            print(f"  {stop:,}/{len(points):,} DEM pixels", flush=True)
    return out.reshape(h, w, N_AIA_BINS).transpose(2, 0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, basis_t, n_basis, logt = load_operators(device)
    model, ckpt = load_scaled_model(args.model, n_basis, device)
    if ckpt["variant"] != "mlp6":
        raise ValueError(f"this exporter is intentionally mlp6-only, got {ckpt['variant']}")

    with np.load(args.input_npz, mmap_mode="r") as data:
        if "AIACube" not in data or "AIACubeShape" not in data:
            raise ValueError("input needs compressed AIACube and AIACubeShape")
        aia_full = decode_array(data, "AIACube", "AIACubeShape")[:6]
        reference = None
        if "DEMCube" in data and "DEMCubeShape" in data:
            reference = decode_array(data, "DEMCube", "DEMCubeShape")[:N_AIA_BINS]

    aia = np.ascontiguousarray(aia_full[:, ::args.stride, ::args.stride])
    if reference is not None and reference.shape[1:] != aia.shape[1:]:
        raise ValueError(f"reference DEM grid {reference.shape[1:]} does not match AIA grid {aia.shape[1:]}")
    print(f"device={device}; model={ckpt['variant']}/{ckpt['loss']}; AIA={aia.shape}")
    dem = predict(model, basis_t, aia, args.pixel_batch)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {"dem": dem, "aia": aia, "logT": np.asarray(logt[:N_AIA_BINS]),
               "model": np.array(os.path.abspath(args.model)),
               "input": np.array(os.path.abspath(args.input_npz))}
    if reference is not None:
        payload["reference_dem"] = reference
    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output}: dem={dem.shape}, reference={'yes' if reference is not None else 'no'}")


if __name__ == "__main__":
    main()
