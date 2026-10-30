#!/usr/bin/env python3
"""Where does the band weight mu come from?  Penalty-method theory, checked per pixel.

The label-free BP loss is the quadratic exterior penalty of the solver's LP:

    L(x) = sum(x) + mu * sum_c relu(|(Dx)_c - o_c| - t s_c)^2 / (t s_c)^2,   x >= 0,

with t s_c the half band-width (t = 1.4, s_c the noise).  At the optimum an
active channel never sits exactly inside its band: the marginal penalty of the
violation balances the marginal sparsity cost.  With v_c the violation in DN/s
and lambda_c the LP's dual price of channel c (weight units per DN/s of band
edge), the balance 2 mu v_c / (t s_c)^2 = lambda_c gives, in half band-widths,

    v_c / (t s_c) = lambda_c * t s_c / (2 mu).

So mu buys a tolerated band violation that falls as 1/mu and grows with the
noise (brighter pixels have wider bands) and with the price (94 A, the channel
with the weakest response, is the dear one).  Turned round, a tolerance eps
needs mu >= max_c lambda_c t s_c / (2 eps): a number set by the response matrix
and the noise model, with no solver DEM involved.

This script checks that on the validation population of the mu screen (same
sampling as objective_gap.py).  Per pixel it solves BP's LP with HiGHS and keeps
the duals, solves the penalised problem at several mu, measures the violation
per channel, compares it with the prediction, and tabulates the mu each
tolerance needs.  It also reports how often the solver's own DEM sits inside
the 1.4 sigma band, split by the solver's tolLevel (1 tight, 3 and 5 relaxed),
and, with --models, scores trained networks the same way.  CPU only.

    python3 experiments/mu_theory.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --phase val --output output/experiments/diagnostics/bp_mu_theory_val.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from experiments.diagnose_bright_failures import CHANNELS, INTENSITY_LABELS
from experiments.objective_gap import (CORE, N_GROUPS, G, collect, describe, group_summary,
                                       init_worker, predict_x, solve_chunk)

MUS = (1, 3, 10, 30, 100, 300, 1000)
EPS = (0.5, 0.2, 0.1, 0.05)        # tolerances, in half band-widths
PCTS = ((50, "p50"), (90, "p90"), (99, "p99"), (100, "max"))
TINY = 1e-6                        # violations below this many half-widths count as inside
# Population share per brightness band in the full diagnostic, as objective_gap.py.
SHARE = np.array([3.85, 10.51, 33.02, 42.50, 6.53, 2.35, 1.15, 0.08, 0.0098]) / 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="staged split with _x/_e/_y/_m zarrs")
    p.add_argument("--phase", default="val")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--per-group", type=int, default=1000, help="target pixels per band")
    p.add_argument("--core-cap", type=int, default=6000, help="max flare-core pixels")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--alpha-l1", type=float, default=1.0)
    p.add_argument("--mus", default=",".join(str(m) for m in MUS))
    p.add_argument("--models", nargs="*", default=[],
                   help="label=checkpoint: trained networks scored the same way (optional)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


# ── BP's LP and its duals ─────────────────────────────────────────────────────

def lp_duals(D, obs, err, t):
    """min sum(x) s.t. obs - t err <= Dx <= obs + t err, x >= 0.

    Returns (status, objective, x, lam_lo, lam_hi).  lam_lo[c] is the sparsity
    cost of raising channel c's lower edge by one DN/s, lam_hi[c] that of
    lowering its upper edge; both are >= 0 and at most one side of a channel is
    active.  status is scipy's: 0 solved, 2 infeasible."""
    from scipy.optimize import linprog
    s = t * err
    k = D.shape[1]
    r = linprog(np.ones(k), A_ub=np.vstack([D, -D]),
                b_ub=np.concatenate([obs + s, -(obs - s)]),
                bounds=(0, None), method="highs")
    nan = np.full(len(obs), np.nan)
    if r.status != 0:
        return r.status, np.nan, None, nan, nan
    # marginals are d objective / d b_ub, never positive for a minimisation
    m = np.asarray(r.ineqlin.marginals, dtype=np.float64)
    return 0, float(r.fun), r.x, np.maximum(-m[len(obs):], 0.0), np.maximum(-m[:len(obs)], 0.0)


def lp_chunk(job):
    obs, err, tol, t = job
    D = G["D"]
    n, c = obs.shape
    out = {"status": np.full(n, -1, np.int8), "obj": np.full(n, np.nan),
           "x": np.zeros((n, D.shape[1])),
           "lam_lo": np.full((n, c), np.nan), "lam_hi": np.full((n, c), np.nan),
           "status_tol": np.full(n, -1, np.int8), "obj_tol": np.full(n, np.nan),
           "lam_lo_tol": np.full((n, c), np.nan), "lam_hi_tol": np.full((n, c), np.nan)}
    for i in range(n):
        st, ob, x, lo, hi = lp_duals(D, obs[i], err[i], t)
        out["status"][i], out["obj"][i], out["lam_lo"][i], out["lam_hi"][i] = st, ob, lo, hi
        if x is not None:
            out["x"][i] = x
        if tol[i] > 1:                       # the solver relaxed this pixel's band
            st, ob, _, lo, hi = lp_duals(D, obs[i], err[i], t * float(tol[i]))
            out["status_tol"][i], out["obj_tol"][i] = st, ob
            out["lam_lo_tol"][i], out["lam_hi_tol"][i] = lo, hi
    return out


# ── summaries ─────────────────────────────────────────────────────────────────

def pcts(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: None for _, k in PCTS}
    return {k: float(np.percentile(a, q)) for q, k in PCTS}


def share(values, labels):
    values = np.asarray(values)
    if values.size == 0:
        return {lab: None for lab in labels}
    return {lab: float(100 * np.mean(values == i)) for i, lab in enumerate(labels)}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def main():
    args = parse_args()
    import zarr
    mus = [float(m) for m in args.mus.split(",")]
    high = [m for m in mus if m >= 10]
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload), np.float64)
    nb = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1]
    if args.max_blocks:
        nb = min(nb, args.max_blocks)
    cfg = {"root": os.path.abspath(args.root), "phase": args.phase, "loss": "barrier",
           "thresholds": thresholds.tolist(), "seed": args.seed, "tolfac": args.tolfac,
           "alpha_l1": args.alpha_l1, "mu": 1.0, "enet_alpha": 0.001, "enet_lam": 0.5,
           "l1_power": 0.0}
    t0 = time.time()

    # Same population as the mu screen: --per-group pixels per band, every flare core
    # kept and capped below.
    est_valid = nb * 128 * 128 * 0.85
    keep = np.minimum(1.0, args.per_group / (SHARE * est_valid) * 1.5)
    keep[CORE] = 1.0
    cfg["keep"] = keep.tolist()
    chunk = math.ceil(nb / args.workers)
    jobs = [(cfg, s, min(s + chunk, nb), i) for i, s in enumerate(range(0, nb, chunk))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts_ = pool.map(collect, jobs)
    pop = sum(p for _, p in parts_)
    data = {k: np.concatenate([d[k] for d, _ in parts_ if d[k] is not None])
            for k in parts_[0][0]}
    rng = np.random.default_rng(args.seed)
    take = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(data["group"] == g)
        cap = args.core_cap if g == CORE else args.per_group
        take.append(rng.choice(idx, size=min(cap, idx.size), replace=False) if idx.size else idx)
    take = np.sort(np.concatenate(take))
    data = {k: v[take] for k, v in data.items()}
    obs, err, dem = (data[k].astype(np.float64) for k in ("obs", "err", "dem"))
    inten, group, tol = data["inten"].astype(np.float64), data["group"], data["tol"].astype(np.int64)
    n = len(obs)
    print(f"collected {n:,} pixels from {nb:,} blocks in {time.time() - t0:.0f}s; "
          f"per band {np.bincount(group, minlength=N_GROUPS).tolist()}", flush=True)

    init_worker(cfg)
    D, B = G["D"], G["B"]
    bound_single = 1.0 / D.max(axis=1)        # price when only that channel is active
    n_chunks = args.workers * 4
    edges = np.linspace(0, n, n_chunks + 1).astype(int)
    pairs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    # One pool for both solve stages: every worker pays the torch import once
    # (init_worker loads the operators through train_scaled).
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        # 1. BP's LP at the tight band (and at the solver's relaxed band where it relaxed).
        t1 = time.time()
        lp = pool.map(lp_chunk, [(obs[a:b], err[a:b], tol[a:b], args.tolfac) for a, b in pairs])
        lp = {k: np.concatenate([d[k] for d in lp]) for k in lp[0]}
        feasible = lp["status"] == 0
        print(f"LPs in {time.time() - t1:.0f}s; feasible at {args.tolfac} sigma: "
              f"{100 * feasible.mean():.1f}%", flush=True)

        # 2. Trained networks (optional), scored the same way.
        nets = {}
        for spec in args.models:
            label, path = spec.split("=", 1)
            x, ckpt = predict_x(path, obs.astype(np.float32))
            if ckpt["loss"] != "barrier":
                raise ValueError(f"{label} was trained with {ckpt['loss']}, not barrier")
            nets[label] = x

        # 3. Penalised optima at every mu, warm-started from the LP solution, the solver's
        # fewest-weights x and the networks (solve_chunk keeps the best).
        variants = [f"mu{m:g}" for m in mus]
        starts = [np.where(feasible[:, None], lp["x"], 0.0)] + list(nets.values())
        jobs = [(obs[a:b], err[a:b], dem[a:b], inten[a:b], [s[a:b] for s in starts], variants)
                for a, b in pairs]
        t2 = time.time()
        solved = pool.map(solve_chunk, jobs)
    nonconv = {v: int(sum(st[v] for _, st in solved)) for v in variants}
    x_sol = np.concatenate([o["solver"] for o, _ in solved])
    X = {m: np.concatenate([o[v] for o, _ in solved]) for m, v in zip(mus, variants)}
    print(f"per-pixel optima in {time.time() - t2:.0f}s; not converged: {nonconv}", flush=True)

    # 4. Violations, measured and predicted.
    s = args.tolfac * err                     # half band-width in DN/s
    lb, ub = obs - s, obs + s

    def violation(x):
        rec = x @ D.T
        below, above = np.maximum(lb - rec, 0) / s, np.maximum(rec - ub, 0) / s
        side = np.where(below > TINY, -1, np.where(above > TINY, 1, 0))
        return np.maximum(below, above), side

    V, S = {}, {}
    for m in mus:
        V[m], S[m] = violation(X[m])
    lam_act = np.fmax(lp["lam_lo"], lp["lam_hi"])        # NaN where the LP is infeasible
    price_hw = lam_act * s / 2.0                          # mu x predicted violation, half-widths

    def predicted(m):
        lam_side = np.where(S[m] < 0, lp["lam_lo"], np.where(S[m] > 0, lp["lam_hi"], 0.0))
        return lam_side * s / (2 * m), lam_side

    sol_dem = x_sol @ B.T
    rec_sol = x_sol @ D.T
    in14 = np.abs(rec_sol - obs) <= s * (1 + 1e-6)
    intol = np.abs(rec_sol - obs) <= s * tol[:, None] * (1 + 1e-6)
    DESC = {m: describe(X[m], obs, err, inten, cfg, variant=f"mu{m:g}") for m in mus}
    DESC["solver"] = describe(x_sol, obs, err, inten, cfg)
    NETV = {label: violation(x) for label, x in nets.items()}
    NETD = {label: describe(x, obs, err, inten, cfg, variant="mu100") for label, x in nets.items()}

    def screen(d, idx):
        gs = group_summary(d, sol_dem, obs, err, cfg, idx)
        return {"dem_mse_vs_solver": gs["dem_mse_vs_solver"], "em_ratio_vs_solver": gs["em_ratio_vs_solver"],
                "recon94_over_obs": gs["recon_over_obs"]["94"], "in_band_pct": gs["in_band_pct"],
                "objective_mean_mu1": gs["objective_mean"]}

    bands = {}
    for g, lab in list(enumerate(INTENSITY_LABELS)) + [(None, "all")]:
        idx = np.arange(n) if g is None else np.flatnonzero(group == g)
        if idx.size == 0:
            continue
        fe = feasible[idx]
        fi = idx[fe]
        b = {"n": int(idx.size), "lp_feasible_pct": float(100 * fe.mean()),
             "lp_status_counts": {int(k): int(v) for k, v in zip(*np.unique(lp["status"][idx], return_counts=True))},
             "solver_in_band_1.4_pct": {c: float(100 * in14[idx, ci].mean()) for ci, c in enumerate(CHANNELS)},
             "solver_in_band_1.4_all6_pct": float(100 * in14[idx].all(1).mean()),
             "solver_in_own_band_all6_pct": float(100 * intol[idx].all(1).mean()),
             "by_tol": {}}
        for tv in (1, 3, 5):
            j = idx[tol[idx] == tv]
            if j.size == 0:
                continue
            e = {"n": int(j.size), "share_pct": float(100 * j.size / idx.size),
                 "lp_feasible_1.4_pct": float(100 * feasible[j].mean()),
                 "solver_in_band_1.4_all6_pct": float(100 * in14[j].all(1).mean()),
                 "solver_in_own_band_all6_pct": float(100 * intol[j].all(1).mean()),
                 "solver_in_band_1.4_pct": {c: float(100 * in14[j, ci].mean()) for ci, c in enumerate(CHANNELS)}}
            if tv > 1:
                e["lp_feasible_own_band_pct"] = float(100 * np.mean(lp["status_tol"][j] == 0))
                e["solver_in_own_band_pct"] = {c: float(100 * intol[j, ci].mean()) for ci, c in enumerate(CHANNELS)}
            b["by_tol"][str(tv)] = e
        # prices
        b["lambda_active"] = {c: pcts(lam_act[fi, ci]) for ci, c in enumerate(CHANNELS)}
        b["lambda_positive_pct"] = {c: (float(100 * np.mean(lam_act[fi, ci] > 1e-9)) if fi.size else None)
                                    for ci, c in enumerate(CHANNELS)}
        b["lambda_94_above_single_channel_bound_pct"] = (
            float(100 * np.mean(lam_act[fi, 0] > bound_single[0])) if fi.size else None)
        ph = price_hw[fi]
        if fi.size:
            b["price_halfwidth"] = {"max_over_channels": pcts(ph.max(1)),
                                    "dearest_channel_share": share(ph.argmax(1), CHANNELS),
                                    "per_channel": {c: pcts(ph[:, ci]) for ci, c in enumerate(CHANNELS)}}
            b["mu_needed"] = {f"{e:g}": pcts(ph.max(1) / e) for e in EPS}
        # violations of the penalised optimum
        b["violation"] = {}
        for m in mus:
            v = V[m][idx]
            vmax, am = v.max(1), v.argmax(1)
            vh, lam_side = predicted(m)
            vh, lam_side = vh[idx], lam_side[idx]
            active = fe[:, None] & (lam_side > 1e-9)
            msk = (v > TINY) & active
            n_viol = int((v > TINY).sum())
            n_inactive = int(((v > TINY) & fe[:, None] & ~(lam_side > 1e-9)).sum())
            ratio = v[msk] / vh[msk]
            b["violation"][f"{m:g}"] = {
                "max_violation_halfwidths": pcts(vmax),
                "within_eps_pct": {f"{e:g}": float(100 * np.mean(vmax <= e)) for e in EPS},
                "violated_channel_share": share(am[vmax > TINY], CHANNELS),
                "ratio_measured_over_predicted": {"n": int(msk.sum()),
                                                  "p10": float(np.percentile(ratio, 10)) if ratio.size else None,
                                                  "p50": float(np.percentile(ratio, 50)) if ratio.size else None,
                                                  "p90": float(np.percentile(ratio, 90)) if ratio.size else None},
                "violations_not_active_in_lp_pct": float(100 * n_inactive / max(n_viol, 1)),
                "mean_violation_94_halfwidths": float(v[:, 0].mean()),
                "screen": screen(DESC[m], idx)}
        b["screen_solver"] = screen(DESC["solver"], idx)
        # the 1/mu law: mu x violation should be constant per pixel-channel once mu is large
        stack = np.stack([V[m][idx] for m in high])
        ok = (stack > TINY).all(0) & fe[:, None]
        if ok.any():
            mv = stack * np.array(high)[:, None, None]
            cv = mv.std(0) / mv.mean(0)
            b["inverse_mu_law"] = {"mus": high, "n_pixel_channels": int(ok.sum()),
                                   "cv_of_mu_times_violation": pcts(cv[ok]),
                                   "persist_pct": float(100 * ok.sum() / max(((V[high[0]][idx] > TINY) & fe[:, None]).sum(), 1))}
        for label in nets:
            v = NETV[label][0][idx]
            vmax = v.max(1)
            b[f"net_{label}"] = {"max_violation_halfwidths": pcts(vmax),
                                 "within_eps_pct": {f"{e:g}": float(100 * np.mean(vmax <= e)) for e in EPS},
                                 "violated_channel_share": share(v.argmax(1)[vmax > TINY], CHANNELS),
                                 "screen": screen(NETD[label], idx)}
        bands[lab] = b

    overall = {"mu_needed_max": {f"{e:g}": (float(np.nanmax(price_hw.max(1)) / e) if feasible.any() else None) for e in EPS},
               "mu_needed_argmax_band": INTENSITY_LABELS[int(group[np.nanargmax(np.where(feasible, price_hw.max(1), -1))])],
               "mu_needed_argmax_channel": CHANNELS[int(price_hw[np.nanargmax(np.where(feasible, price_hw.max(1), -1))].argmax())],
               "single_channel_bound": {c: float(bound_single[ci]) for ci, c in enumerate(CHANNELS)}}

    result = {"purpose": "band weight from penalty-method theory, checked per pixel against BP's LP duals",
              "root": cfg["root"], "phase": args.phase, "loss": "barrier",
              "settings": {"tolfac": args.tolfac, "alpha_l1": args.alpha_l1, "mus": mus, "eps": list(EPS),
                           "per_group": args.per_group, "core_cap": args.core_cap, "seed": args.seed},
              "models": dict(s_.split("=", 1) for s_ in args.models),
              "population_per_band": dict(zip(INTENSITY_LABELS, pop.tolist())),
              "n_per_band": dict(zip(INTENSITY_LABELS, np.bincount(group, minlength=N_GROUPS).tolist())),
              "not_converged": nonconv, "overall": overall, "bands": bands,
              "runtime_s": round(time.time() - t0, 1)}
    result = clean(result)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    text = render(result)
    print(text)
    stem = os.path.splitext(args.output)[0]
    with open(stem + ".txt", "w") as f:
        f.write(text + "\n")
    arrays = {"mus": np.array(mus), "group": group, "tol": tol, "inten": inten, "obs": obs, "err": err,
              "dem": dem, "x_sol": x_sol, "lp_status": lp["status"], "lp_obj": lp["obj"],
              "lam_lo": lp["lam_lo"], "lam_hi": lp["lam_hi"], "lp_status_tol": lp["status_tol"],
              "lp_obj_tol": lp["obj_tol"], "lam_lo_tol": lp["lam_lo_tol"], "lam_hi_tol": lp["lam_hi_tol"],
              "price_halfwidth": price_hw,
              "viol": np.stack([V[m] for m in mus]), "side": np.stack([S[m] for m in mus]),
              "x_opt": np.stack([X[m] for m in mus]).astype(np.float32),
              "dem_mse": np.stack([np.square(DESC[m]["dem"] - sol_dem).mean(1) for m in mus]),
              "em_ratio": np.stack([DESC[m]["dem"].sum(1) / np.maximum(sol_dem.sum(1), 1e-30) for m in mus]),
              "recon94_ratio": np.stack([DESC[m]["recon"][:, 0] / np.maximum(obs[:, 0], 1e-30) for m in mus])}
    for label, x in nets.items():
        arrays[f"net_{label}_x"] = x.astype(np.float32)
        arrays[f"net_{label}_viol"] = NETV[label][0]
    np.savez_compressed(stem + ".npz", **arrays)
    print(f"\nwrote {args.output} (+ .txt, .npz) in {time.time() - t0:.0f}s")


# ── text rendering ────────────────────────────────────────────────────────────

def fmt(v, spec=".3g", width=None):
    s_ = "-" if v is None else format(v, spec)
    return s_.rjust(width) if width else s_


def render(r):
    st = r["settings"]
    L = [f"Band-weight theory check: barrier objective, {r['phase']} split, t = {st['tolfac']}, "
         f"mus {st['mus']}, {sum(r['n_per_band'].values()):,} pixels, {r['runtime_s']} s",
         "Prediction: violation (half band-widths) = lambda_c x t sigma_c / (2 mu), lambda_c the LP's dual price.",
         "Single-channel price bound 1/max_k D[c,k] (weight units per DN/s): "
         + ", ".join(f"{c} A {v:.3f}" for c, v in r["overall"]["single_channel_bound"].items()),
         "mu needed over every sampled pixel, by tolerance: "
         + ", ".join(f"eps {e}: {fmt(v, '.0f')}" for e, v in r["overall"]["mu_needed_max"].items())
         + f" (set by a {r['overall']['mu_needed_argmax_band']} pixel, {r['overall']['mu_needed_argmax_channel']} A)",
         f"L-BFGS-B not converged per mu: {r['not_converged']}"]
    for lab, b in r["bands"].items():
        tols = ", ".join(f"tol {k}: {e['share_pct']:.0f}% of pixels, solver inside 1.4 sigma {e['solver_in_band_1.4_all6_pct']:.0f}%"
                         + (f", inside its own band {e['solver_in_own_band_all6_pct']:.0f}%" if k != "1" else "")
                         for k, e in b["by_tol"].items())
        L.append(f"\n== {lab}: n = {b['n']:,}; LP feasible at 1.4 sigma {b['lp_feasible_pct']:.1f}%; "
                 f"solver DEM inside 1.4 sigma (all six channels) {b['solver_in_band_1.4_all6_pct']:.0f}%")
        L.append(f"   {tols}")
        L.append("   mu     max violation p50   p90   p99   max | within 0.5  0.2  0.1 0.05 | v/vhat p10  p50  p90 (n)   | not in LP | DEM MSE  EM    94A  94A in band")
        for m, v in b["violation"].items():
            mv, w, rt, sc = v["max_violation_halfwidths"], v["within_eps_pct"], v["ratio_measured_over_predicted"], v["screen"]
            L.append(f"   {m:>5} {fmt(mv['p50'], '.3f', 19)} {fmt(mv['p90'], '.3f', 5)} {fmt(mv['p99'], '.3f', 5)} {fmt(mv['max'], '.2f', 5)} |"
                     f" {fmt(w['0.5'], '.0f', 10)} {fmt(w['0.2'], '.0f', 4)} {fmt(w['0.1'], '.0f', 4)} {fmt(w['0.05'], '.0f', 4)} |"
                     f" {fmt(rt['p10'], '.2f', 10)} {fmt(rt['p50'], '.2f', 4)} {fmt(rt['p90'], '.2f', 4)} ({rt['n']:>5}) |"
                     f" {fmt(v['violations_not_active_in_lp_pct'], '.0f', 8)}% |"
                     f" {fmt(sc['dem_mse_vs_solver'], '.3g', 7)} {fmt(sc['em_ratio_vs_solver'], '.2f', 5)}"
                     f" {fmt(sc['recon94_over_obs'], '.2f', 5)} {fmt(sc['in_band_pct']['94'], '.0f', 4)}%")
        ss = b["screen_solver"]
        L.append(f"   solver DEM: DEM MSE 0, EM 1.00, 94A {fmt(ss['recon94_over_obs'], '.2f')}, 94A in band {fmt(ss['in_band_pct']['94'], '.0f')}%")
        for k, v in b.items():
            if k.startswith("net_"):
                mv, w, sc = v["max_violation_halfwidths"], v["within_eps_pct"], v["screen"]
                L.append(f"   {k}: max violation p50 {fmt(mv['p50'], '.3f')} p90 {fmt(mv['p90'], '.3f')} p99 {fmt(mv['p99'], '.3f')} "
                         f"max {fmt(mv['max'], '.2f')}; within 0.1: {fmt(w['0.1'], '.0f')}%; DEM MSE {fmt(sc['dem_mse_vs_solver'], '.3g')}, "
                         f"EM {fmt(sc['em_ratio_vs_solver'], '.2f')}, 94A {fmt(sc['recon94_over_obs'], '.2f')}; "
                         f"violated channel: " + ", ".join(f"{c} {fmt(p, '.0f')}%" for c, p in v["violated_channel_share"].items()))
        la = b["lambda_active"]
        L.append("   lambda (active side, weight units per DN/s) p50 / p99 / max: "
                 + "; ".join(f"{c} {fmt(la[c]['p50'], '.3f')} / {fmt(la[c]['p99'], '.3f')} / {fmt(la[c]['max'], '.3f')}" for c in la))
        L.append(f"   lambda_94 above the single-channel bound: {fmt(b['lambda_94_above_single_channel_bound_pct'], '.0f')}%; "
                 f"94 A active at {fmt(b['lambda_positive_pct']['94'], '.0f')}% of pixels")
        if "mu_needed" in b:
            L.append("   mu needed (p50 / p90 / p99 / max): "
                     + " | ".join(f"eps {e}: {fmt(v['p50'], '.1f')} / {fmt(v['p90'], '.1f')} / {fmt(v['p99'], '.0f')} / {fmt(v['max'], '.0f')}"
                                  for e, v in b["mu_needed"].items()))
            L.append("   dearest channel: " + ", ".join(f"{c} {fmt(p, '.0f')}%" for c, p in b["price_halfwidth"]["dearest_channel_share"].items()))
        if "inverse_mu_law" in b:
            il = b["inverse_mu_law"]
            L.append(f"   1/mu law over mu >= 10: CV of mu x violation p50 {fmt(il['cv_of_mu_times_violation']['p50'], '.3f')}, "
                     f"p90 {fmt(il['cv_of_mu_times_violation']['p90'], '.3f')} over {il['n_pixel_channels']:,} pixel-channels "
                     f"({il['persist_pct']:.0f}% of mu = 10 violations persist to mu = {il['mus'][-1]:g})")
    return "\n".join(L)


if __name__ == "__main__":
    main()
