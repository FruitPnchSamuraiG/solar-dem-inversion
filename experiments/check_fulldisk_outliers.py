#!/usr/bin/env python3
"""Full-disk DEM error and reconstruction outliers of two viewer runs, per website date.

The shared test set scores 64 blocks per image; the viewer shows whole disks, including
pixels the shared test has no BP answer for. This compares the exported DEMs of two runs
against the solver DEM stored with them: full-disk DEM MSE, how much of it the worst
pixels hold, and how far the reconstructed 171 A goes.

    python experiments/check_fulldisk_outliers.py bp_mlp6_h336 bp_mlp6_h680_clamp
"""
import os
import sys

import numpy as np

V = os.path.join(os.environ["SCRATCH"], "dem/visuals/matched_enet_alpha0p001/exports")
R = np.load("RData.npz")["R"][:, :18] * 1e26
with open("experiments/visualizer_dates.txt") as f:
    dates = [l.strip() for l in f if l.strip() and not l.startswith("#")]
for d in dates:
    print("==", d, flush=True)
    for run in sys.argv[1:]:
        z = np.load(f"{V}/{run}/dem_{d}.npz")
        dem, ref = z["dem"].astype(np.float64), z["reference_dem"].astype(np.float64)
        ok = np.isfinite(ref).all(0) & np.isfinite(dem).all(0)
        sse = np.where(ok, ((dem - ref) ** 2).sum(0), 0.0)
        o = np.sort(sse.ravel())[::-1]
        rec = np.einsum("ct,thw->chw", R, np.nan_to_num(dem))
        em = np.nan_to_num(dem).sum(0)
        print(f"  {run:20s} MSE {sse.sum() / (ok.sum() * 18):8.3f}  top1 {100 * o[0] / o.sum():5.1f}%  "
              f"top10 {100 * o[:10].sum() / o.sum():5.1f}%  | 171A recon max {rec[2].max():.3g}, "
              f">2e3 {(rec[2] > 2e3).sum()}, >1e4 {(rec[2] > 1e4).sum()}  | max EM {em.max():.3g} "
              f"(solver {np.nanmax(ref.sum(0)):.3g})", flush=True)

# Who are the reconstruction outliers? Their solver DEM, tolLevel and the six observations.
print("\n## pixels with reconstructed 171 A above 1e4 DN/s", flush=True)
D = os.environ["SCRATCH"] + "/dem/data/lp_AIA_hofdeconv_full"
for run in sys.argv[1:]:
    n_all = n_nan = 0
    rows = []
    for d in dates:
        z = np.load(f"{V}/{run}/dem_{d}.npz")
        if run == sys.argv[1] and d == dates[0]:
            print("export keys", z.files, flush=True)
        dem, ref = np.nan_to_num(z["dem"].astype(np.float64)), z["reference_dem"]
        rec = np.einsum("ct,thw->chw", R, dem)[2]
        ii, jj = np.nonzero(rec > 1e4)
        n_all += ii.size
        bad = ~np.isfinite(ref[:, ii, jj]).all(0)
        n_nan += int(bad.sum())
        for k in ("aia", "obs", "aia_cube", "AIACube"):
            if k in z.files:
                obs = z[k]
                rows += [np.round(obs[:, i, j], 2).tolist() + [f"{rec[i, j]:.3g}"] for i, j in zip(ii[:3], jj[:3])]
                break
    print(f"{run}: {n_all} pixels on 8 disks, {n_nan} with no solver DEM ({100 * n_nan / max(n_all, 1):.1f}%)", flush=True)
    for r in rows[:8]:
        print("   obs 94..335, recon171:", r, flush=True)
