#!/usr/bin/env python3
"""Build a per-pixel training cache from a staged zarr split, so training stops waiting on data.

The block loader (src/zarr_data.py) decompresses a whole 256x256 block of readings and noise
estimates (3 MB) to draw 512 pixels from it: about 176 GB of decompression per epoch, which
leaves the GPU idle most of the time. The centre-pixel MLP needs, per pixel, only the six
readings, the six noise estimates and the solver's tolerance flag. This script writes exactly
those for every usable pixel of the stride-2 grid, grouped by block in block order, once:

    obs.npy   [N, 6] float32   readings (DN/s), as the loader's obs_c[:, i, j]
    err.npy   [N, 6] float32   noise estimates, as the loader's err_c[:, i, j]
    tol.npy   [N]    uint8     solver tolLevel (1 tight, 3 / 5 relaxed)
    offsets.npy [n_blocks + 1] int64   pixels of block b are rows offsets[b]:offsets[b+1]
    meta.json

A pixel is kept by the same rule as ZarrPatchBlockDataset._valid_mask for --clamp_mode
(finite reading and noise, solved at tolLevel 1, 3 or 5; plus reading > min_obs in every
channel for exclude, noise > 0 for keep), in the order np.nonzero gives, so the cache holds
exactly the pixels the block loader can draw. CPU only.

    python3 experiments/build_pixel_cache.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --phase train --clamp_mode keep --out $SCRATCH/dem/cache/lp_train_keep
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

MIN_OBS = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--phase", default="train")
    p.add_argument("--clamp_mode", choices=["exclude", "mask", "keep"], required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min_obs", type=float, default=MIN_OBS)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    return p.parse_args()


def valid_mask(obs_c, err_c, tol, clamp_mode, min_obs):
    """As src/zarr_data.py ZarrPatchBlockDataset._valid_mask."""
    ok = np.isfinite(obs_c) & np.isfinite(err_c)
    if clamp_mode == "exclude":
        ok &= obs_c > min_obs
    if clamp_mode == "keep":
        ok &= err_c > 0
    return np.all(ok, axis=0) & np.isin(tol, (1, 3, 5))


def work(job):
    """One worker's block range -> chunk files with its pixels, in block order."""
    cfg, start, stop, wid = job
    import zarr
    z = {k: zarr.open(os.path.join(cfg["root"], f"{cfg['phase']}_{k}.zarr"), mode="r") for k in ("x", "e", "m")}
    s = cfg["stride"]
    obs_l, err_l, tol_l, counts = [], [], [], []
    for b in range(start, stop):
        obs = np.asarray(z["x"][:, :, :, b], dtype=np.float32)[:, ::s, ::s]
        err = np.asarray(z["e"][:, :, :, b], dtype=np.float32)[:, ::s, ::s]
        tol = np.asarray(z["m"][:, :, b])
        rows, cols = np.nonzero(valid_mask(obs, err, tol, cfg["clamp_mode"], cfg["min_obs"]))
        obs_l.append(np.ascontiguousarray(obs[:, rows, cols].T))
        err_l.append(np.ascontiguousarray(err[:, rows, cols].T))
        tol_l.append(tol[rows, cols].astype(np.uint8))
        counts.append(len(rows))
        if (b - start + 1) % 500 == 0:
            print(f"[w{wid}] {b - start + 1}/{stop - start} blocks", flush=True)
    base = os.path.join(cfg["out"], f"chunk{wid:03d}")
    np.save(base + "_obs.npy", np.concatenate(obs_l) if obs_l else np.zeros((0, 6), np.float32))
    np.save(base + "_err.npy", np.concatenate(err_l) if err_l else np.zeros((0, 6), np.float32))
    np.save(base + "_tol.npy", np.concatenate(tol_l) if tol_l else np.zeros(0, np.uint8))
    return wid, np.asarray(counts, np.int64)


def main():
    args = parse_args()
    import zarr
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    nb = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1]
    if args.max_blocks:
        nb = min(nb, args.max_blocks)
    cfg = {"root": os.path.abspath(args.root), "phase": args.phase, "clamp_mode": args.clamp_mode,
           "stride": args.stride, "min_obs": args.min_obs, "out": args.out}
    chunk = math.ceil(nb / args.workers)
    jobs = [(cfg, s, min(s + chunk, nb), i) for i, s in enumerate(range(0, nb, chunk))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = sorted(pool.map(work, jobs))
    counts = np.concatenate([c for _, c in parts])
    offsets = np.zeros(nb + 1, np.int64)
    offsets[1:] = np.cumsum(counts)
    n = int(offsets[-1])
    print(f"{n:,} pixels from {nb:,} blocks in {time.time() - t0:.0f}s; joining chunks", flush=True)

    # Join the chunks into one file per array without holding everything in memory twice.
    for name, shape, dtype in (("obs", (n, 6), np.float32), ("err", (n, 6), np.float32), ("tol", (n,), np.uint8)):
        out = np.lib.format.open_memmap(os.path.join(args.out, f"{name}.npy"), mode="w+", dtype=dtype, shape=shape)
        at = 0
        for wid, _ in parts:
            path = os.path.join(args.out, f"chunk{wid:03d}_{name}.npy")
            a = np.load(path)
            out[at:at + len(a)] = a
            at += len(a)
            os.remove(path)
        out.flush()
        del out
        assert at == n, (name, at, n)
    np.save(os.path.join(args.out, "offsets.npy"), offsets)
    meta = {"root": cfg["root"], "phase": args.phase, "clamp_mode": args.clamp_mode, "stride": args.stride,
            "min_obs": args.min_obs, "n_blocks": nb, "n_pixels": n,
            "empty_blocks": int((counts == 0).sum()), "runtime_s": round(time.time() - t0, 1),
            "rule": "ZarrPatchBlockDataset._valid_mask for this clamp_mode, pixels in np.nonzero order per block"}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
