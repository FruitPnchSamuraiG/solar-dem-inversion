#!/usr/bin/env python3
"""Check rebuilt noise estimates (src/aia_noise.py) against the solver's own DEMs.

With the right noise, the solver's DEM from the clean image fits every channel inside its
band, |R y - reading| <= 1.4 sigma, at every pixel it solved at the tight band. For a sample
of blocks this prints the share of valid pixels inside the band, by target slot (the shared
test holds five targets per image: one clean solve, four re-solves of noisy copies, which
need not fit the clean readings), and for our own staging both with the stored noise and with
the rebuilt one (they must agree).

    python3 experiments/check_shared_noise.py --images 10
"""
import argparse
import os
import sys

import numpy as np
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.aia_noise import NoiseModel  # noqa: E402

N_BINS = 18
SHARED = "/projects/rps/dff6142/fouheylab/solar_dem/bp_AIA_hofdeconv_full_DS"


def inside(x, y, sig, R, tol=1e-2):
    """Share of pixels with finite DEM whose reconstruction is inside the band in all channels, up to
    tol half-widths: the solver stores its DEMs bit-rounded to 12 bits, which puts its own
    reconstructions up to ~2e-3 half-widths outside the edge they sit on."""
    obs = x[:, ::2, ::2].reshape(6, -1).T.astype(np.float64)
    dem = y[:N_BINS].reshape(N_BINS, -1).T.astype(np.float64)
    ok = np.isfinite(dem).all(1) & np.isfinite(obs).all(1)
    if not ok.any():
        return np.nan, 0
    rec = dem[ok] @ R.T
    h = 1.4 * sig[ok]
    return float((np.abs(rec - obs[ok]) <= h * (1 + tol)).all(1).mean()), int(ok.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--blocks", type=int, default=3, help="blocks per target slot per image")
    args = p.parse_args()
    scratch = os.environ["SCRATCH"]
    ours = f"{scratch}/dem/data/lp_AIA_hofdeconv_full_DS"
    stamps = open(f"{SHARED}/test_datetimes.txt").read().split()
    noise = NoiseModel(f"{scratch}/dem/data/lp_AIA_hofdeconv_full", stamps)
    R = np.load(os.path.join(ROOT, "RData.npz"))["R"][:, :N_BINS].astype(np.float64) * 1e26
    xs, ys = (zarr.open(f"{SHARED}/test_{k}.zarr", mode="r") for k in "xy")
    xo, yo, eo = (zarr.open(f"{ours}/test_{k}.zarr", mode="r") for k in "xye")
    per_s, per_o = xs.shape[-1] // len(stamps), xo.shape[-1] // len(stamps)
    images = np.linspace(0, len(stamps) - 1, args.images).astype(int)
    by_slot = {}
    control = []
    for im in images:
        for slot in range(per_s // 64):
            for j in range(args.blocks):
                b = im * per_s + slot * 64 + j * (64 // args.blocks)
                x = np.asarray(xs[:, :, :, b], dtype=np.float32)
                sig = noise.sigma(x[:, ::2, ::2].reshape(6, -1).T, im)
                f, n = inside(x, np.asarray(ys[:, :, :, b], dtype=np.float32), sig, R)
                if n:
                    by_slot.setdefault(slot, []).append(f)
        for j in range(args.blocks):
            b = im * per_o + j * (64 // args.blocks)
            x = np.asarray(xo[:, :, :, b], dtype=np.float32)
            y = np.asarray(yo[:, :, :, b], dtype=np.float32)
            e = np.asarray(eo[:, :, :, b], dtype=np.float32)[:, ::2, ::2].reshape(6, -1).T.astype(np.float64)
            f_stored, n = inside(x, y, e, R)
            f_rebuilt, _ = inside(x, y, noise.sigma(x[:, ::2, ::2].reshape(6, -1).T, im), R)
            if n:
                control.append((f_stored, f_rebuilt))
    print(f"images {list(images)}; shared test {per_s} blocks per image (target slots of 64)")
    for slot, v in sorted(by_slot.items()):
        v = np.asarray(v)
        print(f"  shared slot {slot}: blocks {len(v)}, share inside the band: median {np.median(v):.4f}, "
              f"min {v.min():.4f}")
    c = np.asarray(control)
    print(f"  our staging ({len(c)} blocks): stored noise median {np.median(c[:, 0]):.4f} min {c[:, 0].min():.4f}; "
          f"rebuilt noise median {np.median(c[:, 1]):.4f} min {c[:, 1].min():.4f}")


if __name__ == "__main__":
    main()
