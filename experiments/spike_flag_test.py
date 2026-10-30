#!/usr/bin/env python3
"""Can a simple input check catch the wild reconstructions without touching real pixels?

A first version flagged pixels with clamped channels and a neighbour spike; it caught under
half the outliers and flagged up to 14% of pixels, so it was dropped. This one uses only
the observed AIA: one channel far above the others. For each (F, K) on the eight website
disks: pixels flagged, wild reconstructions caught, flagged pixels with a solver DEM, and
flagged pixels in the brightest bands, where a flag would hit real flare pixels.

    python experiments/spike_flag_test.py bp_mlp6_h680_clamp bp_mlp6_h336
"""
import json
import os
import sys

import numpy as np

V = os.path.join(os.environ["SCRATCH"], "dem/visuals/matched_enet_alpha0p001/exports")
R = np.load("RData.npz")["R"][:, :18] * 1e26
with open("experiments/visualizer_dates.txt") as f:
    dates = [l.strip() for l in f if l.strip() and not l.startswith("#")]
with open("eval_and_enet_specs/aia_thresholds.json") as f:
    p = json.load(f)
thr = np.asarray(p.get("test", p) if isinstance(p, dict) else p, float)[:, None, None]
# Wild: some channel reconstructed above 1e3 DN/s and more than 100 times the observation.
# (An absolute cut on 171 A alone also counts real flare pixels.)
# Rule: in units of each channel's Bright threshold, the largest channel is at least F
# and more than K times the second largest. Plasma lights several channels together, so a
# single dominant channel is not something any DEM can make.
GRID = [(f, k) for f in (0.1, 0.2, 0.5, 1.0) for k in (5, 10, 20, 50, 100)]
tot = {g: np.zeros(5, np.int64) for g in GRID}  # flagged, wild caught, flagged with solver DEM, >=10x, >=32x
wild_all = wild_solved = 0
left_max = {g: 0.0 for g in GRID}
for d in dates:
    z0 = np.load(f"{V}/{sys.argv[1]}/dem_{d}.npz")
    aia = np.nan_to_num(z0["aia"].astype(np.float64))
    if aia.shape[-1] != 2048:
        aia = aia[:, ::2, ::2]
    ref = z0["reference_dem"]
    solved = np.isfinite(ref).all(0)
    dem = np.nan_to_num(z0["dem"].astype(np.float64))
    rec = np.einsum("ct,thw->chw", R, dem)
    over = rec / np.maximum(aia, 1.0)
    wild = ((rec > 1e3) & (over > 100)).any(0)
    wild_all += int(wild.sum()); wild_solved += int((wild & solved).sum())
    norm = np.sort(np.maximum(aia, 0) / thr, axis=0)
    top, second = norm[-1], norm[-2]
    inten = top
    ratio = top / np.maximum(second, 1e-3)
    rec_over = np.where(wild, over.max(0), 0)
    print(f"{d}: wild {int(wild.sum())}, with solver DEM {int((wild & solved).sum())}, largest overshoot {rec_over.max():.3g}", flush=True)
    for g in GRID:
        f, k = g
        flag = (top >= f) & (ratio > k)
        tot[g] += np.array([flag.sum(), (wild & flag).sum(), (flag & solved).sum(),
                            (flag & (inten >= 10)).sum(), (flag & (inten >= 32)).sum()])
        left = rec_over[wild & ~flag]
        left_max[g] = max(left_max[g], float(left.max()) if left.size else 0.0)
print(f"\nwild pixels on 8 disks: {wild_all} ({wild_solved} with a solver DEM) of {8 * 2048 * 2048:,}")
print("F, K | flagged | share of pixels | wild caught | flagged with a solver DEM | flagged >=10x | >=32x | largest overshoot left")
for g, t in tot.items():
    print(f"  F>={g[0]:<3} K>{g[1]:>3} | {t[0]:>9,} | {100 * t[0] / (8 * 2048 * 2048):6.3f}% | {t[1]:>4} ({100 * t[1] / max(wild_all, 1):5.1f}%) | {t[2]:>9,} | {t[3]:>6,} | {t[4]:>5,} | {left_max[g]:.3g}")
