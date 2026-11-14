#!/usr/bin/env python3
"""Fit the shared test set's calibration scale relative to ours, per channel, from the solver's
own clean DEMs, so its noise can be rebuilt (src/aia_noise.py, ratio=...).

The shared test was staged with a slightly different calibration: its readings are counts / s'
with s' = k s (s from our solver output files), so its noise is estimate_error(max(x s', 0.5))
/ s'. In the clean target slot (slot 0 of each image's five) the solver's DEM sits exactly on a
band edge in its binding channels, |R y - x| = 1.4 sigma' (up to the 12-bit rounding of stored
DEMs, ~2e-3 half-widths). For each channel, k is the value at which the 99.9th percentile of
|R y - x| / (1.4 sigma'(k)) is 1.002; sigma' falls monotonically with k, so bisection finds it.
Then reports the share of clean-slot pixels inside the band by brightness, with and without k.

    python3 experiments/fit_shared_noise.py --out results/plots/27_refine_20261114/shared_noise_fit.json
"""
import argparse
import json
import os
import sys

import numpy as np
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.aia_noise import NoiseModel  # noqa: E402

SHARED = "/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS"
BRIGHT = np.array([3.1212, 16.9986, 448.8948, 498.3444, 190.7388, 10.3326])
WAVES = (94, 131, 171, 193, 211, 335)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_step", type=int, default=4)
    p.add_argument("--block_step", type=int, default=4)
    p.add_argument("--target", type=float, default=1.002)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    stamps = open(f"{SHARED}/test_datetimes.txt").read().split()
    npz = f"{os.environ['SCRATCH']}/dem/data/lp_AIA_hofdeconv_full"
    base = NoiseModel(npz, stamps)
    R = np.load(os.path.join(ROOT, "RData.npz"))["R"][:, :18].astype(np.float64) * 1e26
    xs, ys = (zarr.open(f"{SHARED}/test_{k}.zarr", mode="r") for k in "xy")
    per = xs.shape[-1] // len(stamps)
    X, G, S = [], [], []
    for im in range(0, len(stamps), args.image_step):
        for j in range(0, 64, args.block_step):
            b = im * per + j                                   # slot 0: the clean solve
            x = np.asarray(xs[:, ::2, ::2, b], np.float64).reshape(6, -1).T
            y = np.asarray(ys[:18, :, :, b], np.float64).reshape(18, -1).T
            ok = np.isfinite(y).all(1) & np.isfinite(x).all(1)
            if ok.any():
                X.append(x[ok]); G.append(np.abs(y[ok] @ R.T - x[ok]) / 1.4)
                S.append(np.repeat(base.scale[im][None], ok.sum(), axis=0))
    X, G, S = np.concatenate(X), np.concatenate(G), np.concatenate(S)
    print(f"clean-slot pixels: {len(X):,}", flush=True)

    def sig(c, k, idx=slice(None)):
        return base.sigma_raw(X[idx, c], S[idx, c] * k, c)

    k_fit = np.ones(6)
    for c in range(6):
        lo, hi = 0.5, 2.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            q = np.quantile(G[:, c] / sig(c, mid), 0.999)
            lo, hi = (mid, hi) if q < args.target else (lo, mid)
        k_fit[c] = 0.5 * (lo + hi)
        print(f"{WAVES[c]:>4}: k = {k_fit[c]:.4f}", flush=True)

    inten = (X / BRIGHT).max(1)
    r0 = np.stack([G[:, c] / sig(c, 1.0) for c in range(6)], 1).max(1)
    r1 = np.stack([G[:, c] / sig(c, k_fit[c]) for c in range(6)], 1).max(1)
    rows = []
    for lo_i, hi_i in ((0, 1), (1, 3.2), (3.2, 10), (10, 32), (32, np.inf)):
        m = (inten >= lo_i) & (inten < hi_i)
        if m.any():
            rows.append({"intensity": [lo_i, None if np.isinf(hi_i) else hi_i], "pixels": int(m.sum()),
                         "inside_ours": float((r0[m] <= 1.01).mean()), "inside_fitted": float((r1[m] <= 1.01).mean()),
                         "worst_channel_p99_fitted": float(np.quantile(r1[m], 0.99))})
            print(rows[-1], flush=True)
    with open(args.out, "w") as f:
        json.dump({"ratio": k_fit.tolist(), "channels": WAVES, "target_p999": args.target,
                   "pixels": int(len(X)), "by_intensity": rows}, f, indent=1)


if __name__ == "__main__":
    main()
