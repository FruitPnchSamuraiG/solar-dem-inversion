#!/usr/bin/env python3
"""A known-answer test: invert synthetic AIA readings whose DEM we chose.

Every other score in this project compares a DEM with a solver's DEM, which no one
claims is the true one. Here the truth is known. We draw DEMs from three families,
predict their six AIA readings with the response matrix, add noise drawn from AIA's
noise model (the median staged error at each intensity, per channel), clip negative
readings at zero as the pipeline does, and invert the noisy readings with both
solvers (BP's linear program with its 3x and 5x retries; Elastic Net with alpha
0.001 and l1 ratio 0.5, as the reference runs), our two networks and the supervised
networks. Every answer is scored against the true DEM.

Families:
  gauss   one Gaussian in logT, centre 5.8 to 7.0, width (sigma) 0.05 to 0.25 dex
  two     a cool component (centre 5.9 to 6.4) plus a hot one (6.7 to 7.1), hot/cool
          emission ratio 0.01 to 1 (log-uniform), widths 0.05 to 0.2 dex
  solver  real BP DEMs from the staged test split, a fifth of them flare cores; these
          lie in the solvers' own basis, so they favor BP-like answers
Total emission for gauss and two is log-uniform between the 1st and 99.9th
percentiles of real BP DEM totals.

    python3 experiments/synthetic_dem_test.py --root $ROOT --output out.json \
        --models bp_ours=<ckpt> enet_ours=<ckpt> --supervised bp_sup10m=<pth> enet_sup10m=<pth>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.full_validation import BRIGHT, label_free  # noqa: E402
from experiments.train_scaled import load_operators  # noqa: E402
from src.zarr_data import MIN_OBS  # noqa: E402

TOLFAC = 1.4
RETRIES = (1.0, 3.0, 5.0)        # BP widens its band 3x and then 5x where no DEM fits
MIN_TOL_ENET = 0.5               # fullBP's _minimumDNs floor on the Elastic Net scaling
BANDS = (("quiet", 0, 1), ("bright", 1, 32), ("flare_core", 32, np.inf))
G = {}


def fit_noise(root, phase, n_blocks, rng):
    """Median sigma^2 in log-intensity bins per channel, from staged readings and errors."""
    zx = zarr.open(f"{root}/{phase}_x.zarr", mode="r")
    ze = zarr.open(f"{root}/{phase}_e.zarr", mode="r")
    pick = rng.choice(zx.shape[-1], size=min(n_blocks, zx.shape[-1]), replace=False)
    o = np.concatenate([np.asarray(zx[:, ::2, ::2, b]).reshape(6, -1) for b in pick], 1)
    e = np.concatenate([np.asarray(ze[:, ::2, ::2, b]).reshape(6, -1) for b in pick], 1)
    edges = np.linspace(-2, 4.5, 66)
    model = []
    for c in range(6):
        ok = np.isfinite(o[c]) & np.isfinite(e[c]) & (o[c] > MIN_OBS) & (e[c] > 0)
        lo = np.log10(o[c, ok])
        idx = np.digitize(lo, edges)
        xs, ys = [], []
        for k in np.unique(idx):
            sel = idx == k
            if sel.sum() >= 50:
                xs.append(np.median(lo[sel]))
                ys.append(np.log10(np.median(e[c, ok][sel] ** 2)))
        model.append((np.array(xs), np.array(ys)))
    return model


def sigma(model, o):
    """Noise sigma for readings o [N, 6] from the fitted curves (flat beyond the data)."""
    out = np.empty_like(o)
    for c, (xs, ys) in enumerate(model):
        out[:, c] = np.sqrt(10 ** np.interp(np.log10(np.maximum(o[:, c], 1e-3)), xs, ys))
    return out


def init(D):
    G["D"] = D


def solve_bp(args):
    from scipy.optimize import linprog
    o, s = args
    D = G["D"]
    A = np.vstack([D, -D])
    for level in RETRIES:
        h = TOLFAC * level * s
        r = linprog(np.ones(D.shape[1]), A_ub=A, b_ub=np.concatenate([o + h, -(o - h)]),
                    bounds=(0, None), method="highs")
        if r.status == 0:
            return r.x, level
    return np.full(D.shape[1], np.nan), 0.0


def solve_enet(args):
    from sklearn.linear_model import ElasticNet
    import warnings
    o, s = args
    D = G["D"]
    tol = np.maximum(TOLFAC * s, MIN_TOL_ENET)
    m = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=False, max_iter=10000, positive=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(D / tol[:, None], o / tol)
    return m.coef_


def draw(family, n, logt, em_range, rng, real):
    if family == "solver":
        return real[rng.choice(len(real), size=n, replace=False)]
    em = 10 ** rng.uniform(*np.log10(em_range), size=n)

    def gauss(c, w):
        g = np.exp(-0.5 * ((logt[None] - c[:, None]) / w[:, None]) ** 2)
        return g / g.sum(1, keepdims=True)
    if family == "gauss":
        return em[:, None] * gauss(rng.uniform(5.8, 7.0, n), rng.uniform(0.05, 0.25, n))
    ratio = 10 ** rng.uniform(-2, 0, n)
    cool = gauss(rng.uniform(5.9, 6.4, n), rng.uniform(0.05, 0.2, n))
    hot = gauss(rng.uniform(6.7, 7.1, n), rng.uniform(0.05, 0.2, n))
    return em[:, None] * (cool + ratio[:, None] * hot) / (1 + ratio[:, None])


def real_dems(root, n_blocks, rng, thr, R):
    """Solved BP DEMs from the staged test split: 40% quiet, 40% bright, 20% flare cores."""
    zy = zarr.open(f"{root}/test_y.zarr", mode="r")
    zx = zarr.open(f"{root}/test_x.zarr", mode="r")
    zm = zarr.open(f"{root}/test_m.zarr", mode="r")
    pick = rng.choice(zy.shape[-1], size=min(n_blocks, zy.shape[-1]), replace=False)
    dems, inten = [], []
    for b in pick:
        y = np.asarray(zy[:18, :, :, b]).reshape(18, -1).T
        o = np.asarray(zx[:, ::2, ::2, b]).reshape(6, -1).T
        t = np.asarray(zm[:, :, b]).reshape(-1)
        ok = np.isfinite(y).all(1) & (t == 1) & (y.sum(1) > 0)
        dems.append(y[ok])
        inten.append((o[ok] / thr).max(1))
    dems, inten = np.concatenate(dems), np.concatenate(inten)
    return dems, inten


def w1(p, q, dlogt=0.1):
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-30)
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-30)
    return np.abs(np.cumsum(p - q, 1)).sum(1) * dlogt


def scores(pred, truth, o_true, s_true, R, logt):
    ok = np.isfinite(pred).all(1)
    p, t = np.clip(pred[ok], 0, None), truth[ok]
    em_p, em_t = p.sum(1), t.sum(1)
    rec = p @ R.T
    shift = np.abs(logt[p.argmax(1)] - logt[t.argmax(1)])
    return {"n": int(ok.sum()), "n_missing": int((~ok).sum()),
            "dem_mse": float(np.mean((p - t) ** 2)), "dem_mae": float(np.mean(np.abs(p - t))),
            "dem_rel_l1_pooled": float(np.abs(p - t).sum() / t.sum()),
            "em_ratio_pooled": float(em_p.sum() / em_t.sum()),
            "em_abs_log10_median": float(np.median(np.abs(np.log10(np.maximum(em_p, 1e-30) / em_t)))),
            "w1_median": float(np.median(w1(p, t))), "w1_mean": float(np.mean(w1(p, t))),
            "peak_shift_median": float(np.median(shift)), "peak_within_0p1_pct": float(100 * np.mean(shift <= 0.1 + 1e-9)),
            "aia_chi_rms_vs_true": float(np.sqrt(np.mean(((rec - o_true[ok]) / s_true[ok]) ** 2)))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="staged BP split (for the noise model and real DEMs)")
    p.add_argument("--models", nargs="*", default=[], help="label=label-free checkpoint")
    p.add_argument("--supervised", nargs="*", default=[], help="label=supervised .pth (clipped at zero)")
    p.add_argument("--n", type=int, default=4000, help="pixels per family")
    p.add_argument("--noise_blocks", type=int, default=300)
    p.add_argument("--real_blocks", type=int, default=600)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    D_t, B_t, n_basis, logt = load_operators(torch.device("cpu"))
    D, B = D_t.double().numpy(), B_t.double().numpy()
    logt = np.asarray(logt[:18], dtype=np.float64)
    R = np.load("RData.npz")["R"][:, :18] * 1e26
    thr = np.asarray(BRIGHT)
    noise = fit_noise(args.root, "val", args.noise_blocks, rng)
    real, real_inten = real_dems(args.root, args.real_blocks, rng, thr, R)
    em_range = tuple(np.percentile(real.sum(1), [1, 99.9]))
    # Stratify the real family: 40% quiet, 40% bright, 20% flare cores.
    strata = []
    for share, sel in ((0.4, real_inten < 1), (0.4, (real_inten >= 1) & (real_inten < 32)), (0.2, real_inten >= 32)):
        idx = np.flatnonzero(sel)
        strata.append(real[rng.choice(idx, size=min(int(share * args.n), idx.size), replace=False)])
    real_strat = np.concatenate(strata)
    print(f"noise model and {len(real):,} real DEMs in {time.time() - t0:.0f} s; EM range {em_range}", flush=True)

    families = {f: draw(f, args.n, logt, em_range, rng, real_strat) for f in ("gauss", "two")}
    families["solver"] = real_strat
    truth = np.concatenate(list(families.values()))
    fam = np.concatenate([[f] * len(v) for f, v in families.items()])
    o_true = truth @ R.T
    s_true = sigma(noise, o_true)
    o = np.maximum(o_true + s_true * rng.standard_normal(o_true.shape), 0)
    s = sigma(noise, o)
    inten = (o_true / thr).max(1)
    in_band = np.mean((np.abs(o - o_true) <= TOLFAC * s).all(1))
    print(f"{len(truth):,} pixels; the true DEM is inside every 1.4-sigma band at {100 * in_band:.1f}% of them", flush=True)

    answers = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=init, initargs=(D,)) as pool:
        t1 = time.time()
        bp = pool.map(solve_bp, list(zip(o, s)), chunksize=64)
        answers["bp_solver"] = np.array([w for w, _ in bp]) @ B.T
        levels = np.array([lv for _, lv in bp])
        print(f"BP solves {time.time() - t1:.0f} s; level counts {dict(zip(*np.unique(levels, return_counts=True)))}", flush=True)
        t1 = time.time()
        answers["enet_solver"] = np.array(pool.map(solve_enet, list(zip(o, s)), chunksize=64)) @ B.T
        print(f"ENet solves {time.time() - t1:.0f} s", flush=True)
    ot = torch.as_tensor(o, dtype=torch.float32)
    for spec in args.models:
        label, path = spec.split("=", 1)
        with torch.no_grad():
            w = label_free(path, n_basis, torch.device("cpu"))(ot).double().numpy()
        answers[label] = w @ B.T
    if args.supervised:
        from student_package.table1_dem_metrics.compute_paper_table_metrics import load_model, predict_regression_only
        for spec in args.supervised:
            label, path = spec.split("=", 1)
            model, _ = load_model(path, torch.device("cpu"))
            with torch.no_grad():
                pred = predict_regression_only(model, ot.T[None, :, None, :])[0, :18, 0, :].T
            answers[label] = pred.clamp(min=0).double().numpy()

    out = {"settings": vars(args), "em_range": list(em_range), "truth_in_all_bands_pct": float(100 * in_band),
           "bp_solver_levels": {str(k): int(v) for k, v in zip(*np.unique(levels, return_counts=True))},
           "noise_model": {c: {"log10_obs": xs.tolist(), "log10_sigma2": ys.tolist()}
                           for c, (xs, ys) in zip(("94", "131", "171", "193", "211", "335"), noise)},
           "results": {}}
    for f in list(families) + ["all"]:
        for band, lo, hi in (("all", 0, np.inf),) + BANDS:
            sel = ((fam == f) | (f == "all")) & (inten >= lo) & (inten < hi)
            if sel.sum() < 20:
                continue
            key = f"{f}/{band}"
            out["results"][key] = {"n": int(sel.sum()),
                                   **{name: scores(a[sel], truth[sel], o_true[sel], s_true[sel], R, logt)
                                      for name, a in answers.items()}}
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    lines = []
    for key, res in out["results"].items():
        lines.append(f"\n== {key} (n={res['n']})")
        lines.append(f"{'method':14s} {'DEM MSE':>11s} {'DEM MAE':>9s} {'EM ratio':>9s} {'|log EM|':>9s} "
                     f"{'W1 med':>7s} {'peak<=0.1':>9s} {'AIA chi':>8s}")
        for name, r in res.items():
            if isinstance(r, dict):
                lines.append(f"{name:14s} {r['dem_mse']:11.4g} {r['dem_mae']:9.4f} {r['em_ratio_pooled']:9.3f} "
                             f"{r['em_abs_log10_median']:9.3f} {r['w1_median']:7.3f} {r['peak_within_0p1_pct']:8.1f}% "
                             f"{r['aia_chi_rms_vs_true']:8.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
