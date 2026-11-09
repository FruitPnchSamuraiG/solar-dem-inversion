#!/usr/bin/env python3
"""Score full-disk DEMs of one date against that date's solver DEM.

Reads the solver's output file (DEM, AIA and the per-pixel tolerance level) and one or
more full-disk exports on the same grid, and reports for each model the DEM error
against the solver by brightness band (Quiet, Bright 1 to 32x, flare cores 32x and up),
for all solved pixels and separately for pixels the solver fitted at 1.4 sigma and
pixels it had to relax; the emission and per-channel reconstruction at flare cores;
and the pixels the self-check flags. Predictions are clipped at zero.

    python3 experiments/date_metrics.py --solver <solver.npz> --exports ours=<npz> sup10m=<npz> \
        --output out.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.export_mlp6_full_disk import decode_array  # noqa: E402
from experiments.full_validation import BRIGHT  # noqa: E402

CHANNELS = ("94", "131", "171", "193", "211", "335")


def w1(p, q, dlogt=0.1):
    p = p / np.maximum(p.sum(0, keepdims=True), 1e-30)
    q = q / np.maximum(q.sum(0, keepdims=True), 1e-30)
    return np.abs(np.cumsum(p - q, 0)).sum(0) * dlogt


def group_scores(pred, ref, rec, obs, sel):
    if sel.sum() == 0:
        return None
    p, r = pred[:, sel], ref[:, sel]
    sse = ((p - r) ** 2).sum(0)
    return {"n": int(sel.sum()), "dem_mse": float(sse.sum() / (18 * sel.sum())),
            "dem_mae": float(np.abs(p - r).mean()), "median_pixel_sse": float(np.median(sse)),
            "em_ratio": float(p.sum() / max(r.sum(), 1e-30)), "w1_median": float(np.median(w1(p, r))),
            "recon_over_obs": {c: float(rec[k, sel].sum() / max(obs[k, sel].sum(), 1e-30))
                               for k, c in enumerate(CHANNELS)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solver", required=True, help="solver output NPZ (fullBP.py)")
    ap.add_argument("--exports", nargs="+", required=True, help="label=export NPZ with a 'dem' cube")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with np.load(args.solver) as d:
        ref = decode_array(d, "DEMCube", "DEMCubeShape")[:18].astype(np.float64)
        aia = decode_array(d, "AIACube", "AIACubeShape")[:6, ::2, ::2].astype(np.float64)
        tol = np.asarray(d["tolLevel"])
    R = np.load("RData.npz")["R"][:, :18] * 1e26
    inten = (aia / np.asarray(BRIGHT)[:, None, None]).max(0)
    solved = np.isfinite(ref).all(0) & np.isfinite(aia).all(0) & (tol > 0)
    bands = {"all": np.ones_like(solved), "quiet": inten < 1, "bright": (inten >= 1) & (inten < 32),
             "flare_core": inten >= 32}
    ref_rec = np.einsum("ct,thw->chw", R, np.nan_to_num(ref))
    out = {"solver": os.path.abspath(args.solver), "grid": list(ref.shape[1:]),
           "pixels": {"solved": int(solved.sum()), "relaxed": int((solved & (tol > 1)).sum()),
                      "unsolved_on_grid": int((~solved).sum()), "flare_core_solved": int((solved & bands["flare_core"]).sum())},
           "max_reading_dn_s": {c: float(np.nanmax(aia[k])) for k, c in enumerate(CHANNELS)},
           "solver_recon_over_obs_flare_core": group_scores(np.nan_to_num(ref), np.nan_to_num(ref), ref_rec, aia,
                                                            solved & bands["flare_core"]),
           "models": {}}
    for spec in args.exports:
        label, path = spec.split("=", 1)
        with np.load(path) as d:
            dem = np.clip(np.nan_to_num(np.asarray(d["dem"], dtype=np.float64)[:18]), 0, None)
        if dem.shape != ref.shape:
            raise ValueError(f"{label}: grid {dem.shape} does not match the solver's {ref.shape}")
        rec = np.einsum("ct,thw->chw", R, dem)
        flagged = ((rec > 1e3) & (rec > 100 * np.maximum(aia, 1))).any(0)
        res = {"self_check_flagged": int(flagged.sum()), "self_check_flagged_solved": int((flagged & solved).sum())}
        for name, b in bands.items():
            for part, cond in (("", solved), ("_feasible", solved & (tol == 1)), ("_relaxed", solved & (tol > 1)),
                               ("_checked", solved & ~flagged)):
                res[name + part] = group_scores(dem, np.nan_to_num(ref), rec, aia, b & cond)
        out["models"][label] = res
        a, c = res["all"], res["flare_core"]
        print(f"{label:10s} DEM MSE {a['dem_mse']:.4g} (checked {res['all_checked']['dem_mse']:.4g}), median "
              f"{a['median_pixel_sse']:.4f}, EM {a['em_ratio']:.3f}; flare cores n={c['n'] if c else 0}: "
              f"MSE {c['dem_mse'] if c else float('nan'):.4g}, EM {c['em_ratio'] if c else float('nan'):.3f}, "
              f"94A {c['recon_over_obs']['94'] if c else float('nan'):.3f}; flagged {res['self_check_flagged']}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
