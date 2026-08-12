#!/usr/bin/env python3
"""Render 18-bin AIA-only DEM NPZ exports into compare.html asset folders."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WAVELENGTHS = (94, 131, 171, 193, 211, 335)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="NPZ from export_mlp6_full_disk.py")
    p.add_argument("--output", required=True, help="asset folder for one model/timestamp")
    p.add_argument("--source", choices=("model", "reference"), default="model")
    return p.parse_args()


def save_image(path, image, cmap="inferno", vmin=None, vmax=None):
    valid = image[np.isfinite(image)]
    if vmin is None:
        vmin = float(np.percentile(valid, 1)) if valid.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, 99.9)) if valid.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-8
    plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)


def jpdf(path, observed, synthesized):
    x = np.maximum(observed.ravel(), 0.5)
    y = np.maximum(synthesized.ravel(), 0.5)
    good = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(4, 4))
    if good.any():
        lo = min(np.log10(x[good]).min(), np.log10(y[good]).min())
        hi = max(np.log10(x[good]).max(), np.log10(y[good]).max())
        ax.hexbin(x[good], y[good], xscale="log", yscale="log", gridsize=200,
                  bins="log", extent=(lo, hi, lo, hi))
        ax.plot((10**lo, 10**hi), (10**lo, 10**hi), "k--", linewidth=0.8)
    ax.set(xlabel="observed AIA", ylabel="DEM-resynthesized AIA")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    with np.load(args.input) as d:
        if args.source == "reference":
            if "reference_dem" not in d:
                raise ValueError("input has no reference_dem; export from a solver NPZ first")
            dem = d["reference_dem"]
        else:
            dem = d["dem"]
        aia, logt = d["aia"], d["logT"]
    if dem.shape[0] != 18 or aia.shape[0] != 6:
        raise ValueError(f"expected 18-bin DEM and 6-channel AIA, got {dem.shape}, {aia.shape}")

    os.makedirs(args.output, exist_ok=True)
    for i, layer in enumerate(dem):
        save_image(os.path.join(args.output, f"dem_{i}.png"), layer)

    mass = np.clip(dem, 0, None)
    denom = mass.sum(axis=0)
    mean = (mass * logt[:, None, None]).sum(axis=0) / np.maximum(denom, 1e-12)
    std = np.sqrt((mass * (logt[:, None, None] - mean[None]) ** 2).sum(axis=0) /
                  np.maximum(denom, 1e-12))
    save_image(os.path.join(args.output, "mean_logt.png"), mean, vmin=float(logt.min()), vmax=float(logt.max()))
    save_image(os.path.join(args.output, "std_logt.png"), std, vmin=0.0, vmax=0.5)

    rdata = np.load("RData.npz")
    response = (rdata["R"][:, :18] * 1e26).astype(np.float32)
    resynth = np.tensordot(response, dem, axes=(1, 0))
    for i, wavelength in enumerate(WAVELENGTHS):
        cmap = f"sdoaia{wavelength}"
        scale = np.sqrt(np.maximum(aia[i], 0))
        vmin, vmax = np.percentile(scale[np.isfinite(scale)], (20, 99.99))
        save_image(os.path.join(args.output, f"aia_{i}_resynth.png"), np.sqrt(np.maximum(resynth[i], 0)), cmap, vmin, vmax)
        save_image(os.path.join(args.output, f"aia_{i}.png"), scale, cmap, vmin, vmax)
        jpdf(os.path.join(args.output, f"aia_{i}_resynth_jpdf.png"), aia[i], resynth[i])

    metrics = {"mae": float(np.mean(np.abs(resynth - aia))),
               "mse": float(np.mean((resynth - aia) ** 2)),
               "source": args.source, "n_dem_bins": 18}
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"rendered {args.source} assets -> {args.output}")


if __name__ == "__main__":
    main()
