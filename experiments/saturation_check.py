#!/usr/bin/env python3
"""Are the brightest flare-core readings saturated?

AIA's detector saturates near 16,000 counts. The solver files keep each channel's scale factor
(exposure time x degradation), so a reading in DN/s times the scale factor estimates the raw counts
(before deconvolution, which moves light around, so this is approximate). A pixel is called
saturated when any channel's estimate reaches --sat counts.

  1. the tight-band flare cores of experiments/flare_pixel_analysis.py (pixels.npz): share saturated,
     by failure type, and the share of our and the supervised model's squared DEM error they hold
  2. every pixel at 32x the Bright threshold or above in the test split, by BP's status (tight band,
     widened, unsolved): share saturated

Staged test blocks are 256 x 256 AIA pixels (128 x 128 on the DEM grid), 64 per image drawn as the
staging script draws them (random.seed(image index), shuffle), images in sorted date order; the mapping
is checked against the solver files.

    python3 experiments/saturation_check.py --pixels output/experiments/flare_pixels/pixels.npz --out <dir>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CHANNELS = (94, 131, 171, 193, 211, 335)
TYPES = ("good", "tie", "fits_more", "smeared", "too_little", "too_much", "wrong_peak", "other")
SC = os.environ.get("SCRATCH", "")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pixels", required=True)
    p.add_argument("--root", default=os.path.join(SC, "dem/data/lp_AIA_hofdeconv_full_DS"))
    p.add_argument("--npz_dir", default=os.path.join(SC, "dem/data/lp_AIA_hofdeconv_full"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--sat", type=float, default=15000.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", required=True)
    return p.parse_args()


def scan(job):
    root, blocks, thr = job
    import zarr
    zx = zarr.open(os.path.join(root, "test_x.zarr"), mode="r")
    zm = zarr.open(os.path.join(root, "test_m.zarr"), mode="r")
    rows = []
    for b in blocks:
        obs = np.asarray(zx[:, :, :, b], np.float32)[:, ::2, ::2]
        tol = np.asarray(zm[:, :, b])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nan_to_num((obs / np.asarray(thr, np.float32)[:, None, None]).max(0))
        ii, jj = np.nonzero(ratio >= 32)
        for i, j in zip(ii, jj):
            rows.append((b, i, j, int(tol[i, j]), *obs[:, i, j].tolist()))
    return rows


def main():
    args = parse_args()
    import zarr
    from experiments.band_maps import load_solver
    dates = [l.strip() for l in open(os.path.join(args.root, "test_datetimes.txt")) if l.strip()]
    nb = zarr.open(os.path.join(args.root, "test_x.zarr"), mode="r").shape[-1]
    per = nb // len(dates)
    assert per * len(dates) == nb, (nb, len(dates))
    import random

    def order(i):                       # dataset/stage_hofdeconv_full.py: random.seed(i), shuffle, keep PER_IMAGE
        random.seed(i)
        blocks = [(bi * 256, bj * 256) for bi in range(16) for bj in range(16)]
        random.shuffle(blocks)
        return blocks[:per]
    orders = {}
    sf = np.stack([np.load(os.path.join(args.npz_dir, f"{d}.npz"))["scaleFactor"].reshape(-1)[:6] for d in dates])

    def full_pos(b, i, j):
        di, loc = divmod(int(b), per)
        if di not in orders:
            orders[di] = order(di)
        sy, sx = orders[di][loc]
        return di, sy // 2 + int(i), sx // 2 + int(j)

    P = np.load(args.pixels)
    where = P["where"]
    # check the mapping on three dates
    check = []
    for k in np.linspace(0, len(where) - 1, 3).astype(int):
        di, I, J = full_pos(*where[k])
        aia = load_solver(os.path.join(args.npz_dir, f"{dates[di]}.npz"))[0]
        check.append(float(np.abs(aia[:, I, J] - P["obs"][k]).max() / max(np.abs(P["obs"][k]).max(), 1e-6)))
    print("mapping check (max relative difference, should be ~0):", check, flush=True)
    di = np.array([full_pos(*w)[0] for w in where])
    counts = P["obs"] * sf[di]
    sat = (counts >= args.sat).any(1)
    sq_o = ((P["ours"] - P["bp"]) ** 2).sum(1)
    sq_s = ((P["sup"] - P["bp"]) ** 2).sum(1)
    types = P["type"].astype(str)
    res = {"sat_counts": args.sat, "mapping_check": check,
           "tight_band_flare_cores": {
               "n": int(len(sat)), "saturated_pct": float(100 * sat.mean()),
               "saturated_by_channel_pct": {str(c): float(100 * (counts[:, k] >= args.sat).mean()) for k, c in enumerate(CHANNELS)},
               "share_of_our_sq_error_pct": float(100 * sq_o[sat].sum() / sq_o.sum()),
               "share_of_supervised_sq_error_pct": float(100 * sq_s[sat].sum() / sq_s.sum()),
               "saturated_pct_by_type": {t: float(100 * sat[types == t].mean()) for t in TYPES if (types == t).any()},
               "saturated_pct_extreme_top1": float(100 * sat[P["extreme"].astype(bool)].mean()),
               "saturated_pct_missing_hot": float(100 * sat[P["missing_hot"].astype(bool)].mean())
               if P["missing_hot"].astype(bool).any() else None,
               "our_dem_mse_unsaturated_vs_all": [float(sq_o[~sat].mean() / 18), float(sq_o.mean() / 18)],
               "supervised_dem_mse_unsaturated_vs_all": [float(sq_s[~sat].mean() / 18), float(sq_s.mean() / 18)]}}
    payload = json.load(open(args.thresholds))
    thr = np.asarray(payload.get("test", payload), np.float64).tolist()
    chunks = [c.tolist() for c in np.array_split(np.arange(nb), args.workers * 4)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = sum(pool.map(scan, [(args.root, c, thr) for c in chunks]), [])
    A = np.array(rows, np.float64)
    tol = A[:, 3].astype(int)
    c_all = A[:, 4:10] * sf[(A[:, 0] // per).astype(int)]
    s_all = (c_all >= args.sat).any(1)
    res["all_flare_cores_by_bp_status"] = {
        name: {"n": int(sel.sum()), "saturated_pct": float(100 * s_all[sel].mean()) if sel.any() else None}
        for name, sel in (("tight", tol == 1), ("widened", tol > 1), ("unsolved", tol == 0))}
    res["unsolved_flare_cores_among_saturated_pct"] = float(100 * (tol[s_all] == 0).mean()) if s_all.any() else None
    res["dates_with_saturated_pixels"] = int(len(np.unique((A[s_all, 0] // per).astype(int))))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "saturation.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
