#!/usr/bin/env python3
"""The solver's constrained problem and the unconstrained losses that stand in for it.

Per pixel the solver's linear program is

    min sum(x)   s.t.   lb <= D x <= ub,   x >= 0,      lb, ub = obs -/+ h,   h = t sigma (t = 1.4),

54 non-negative basis weights x, six channels. A network cannot be held to a rule on its
output, so training turns the rule into a charge. With g the distance outside the band in
half-widths h (g_lo = (lb - Dx)/h, g_hi = (Dx - ub)/h; inside when both <= 0), the textbook
stand-ins are

    quadratic penalty      sum(x) + mu * sum relu(g)^2                exact only as mu -> infinity
    exact (l1) penalty     sum(x) + sum rho * relu(g)                 exact once rho exceeds the dual price
    augmented Lagrangian   sum(x) + sum [relu(y + 2 mu g)^2 - y^2] / (4 mu),   then y <- relu(y + 2 mu g)
                                                                      exact at finite mu; y -> dual price

This script measures on real pixels, with no network, what each one does:

  collect  three samples from the staged split, zeroed channels kept with their band:
           a uniform draw of training pixels, a uniform draw of validation pixels, and a
           validation sample balanced over brightness (every flare core)
  lp       the linear program per pixel (HiGHS): feasible or not, optimum, dual prices;
           where no DEM fits, the smallest band multiple that admits one and the
           least-squares violation
  forms    on the balanced sample: the exact penalty at several prices and the augmented
           Lagrangian over its outer iterations, each solved to its optimum

Prices are written against one scale per channel, P_c = h_c / max_k D[c, k]: the weight
saved per half-width of channel c left out when that channel alone is tight. CPU only.

    python3 experiments/constrained_toy.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --output output/experiments/constrained/toy.json
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

from experiments.diagnose_bright_failures import CHANNELS, INTENSITY_EDGES, INTENSITY_LABELS, N_BINS

MIN_OBS = 1e-3                      # at or below: the deconvolution clamp (src/zarr_data.py)
N_GROUPS = len(INTENSITY_LABELS)
CORE = N_GROUPS - 1
KEYS = ("obs", "err", "dem", "tol", "group", "inten")
# Population share per brightness band in the earlier diagnostics (zero-channel pixels
# excluded there); only used to aim the balanced sample, which is capped per band after.
SHARE = np.array([3.85, 10.51, 33.02, 42.50, 6.53, 2.35, 1.15, 0.08, 0.0098]) / 100
KAPPAS = (1, 2, 4, 16, 64, 256, 1024)
# mu: one fixed weight; ada: starts there and is multiplied by 10 at a pixel whenever an
# outer iteration fails to cut that pixel's violation to a quarter (the textbook rule)
AL_SETTINGS = (("mu", 10.0), ("mu", 100.0), ("ada", 10.0))
AL_ITERS = 8
INSIDE = (1e-1, 1e-2, 1e-3)         # "inside the band" tolerances, half-widths
RELAX = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # band multiples 1.05x ... 3x (the solver's first step), 5x
LBFGS = {"maxiter": 20000, "maxfun": 200000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 30, "maxls": 200}
G = {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="staged split with _x/_e/_y/_m zarrs")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--n-train", type=int, default=2_000_000, help="uniform training pixels")
    p.add_argument("--train-blocks", type=int, default=3000, help="training blocks they come from")
    p.add_argument("--n-hold", type=int, default=300_000, help="uniform validation pixels")
    p.add_argument("--val-blocks", type=int, default=None, help="validation blocks scanned (default all)")
    p.add_argument("--per-group", type=int, default=1000, help="balanced sample: pixels per band")
    p.add_argument("--core-cap", type=int, default=3000, help="balanced sample: max flare cores")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-npz", action="store_true",
                   help="rebuild the summaries from the pixel sets an earlier run wrote (no solving)")
    p.add_argument("--output", required=True)
    return p.parse_args()


# ── collect ───────────────────────────────────────────────────────────────────

def collect(job):
    """One worker's blocks -> (uniform rows, balanced rows, valid pixels per band)."""
    cfg, phase, blocks, wid, p_uni, keep = job
    import zarr
    z = {k: zarr.open(os.path.join(cfg["root"], f"{phase}_{k}.zarr"), mode="r") for k in ("x", "e", "y", "m")}
    thr = np.asarray(cfg["thresholds"], np.float32)[:, None, None]
    rng = np.random.default_rng((cfg["seed"], wid, 0 if phase == "train" else 1))
    rows = {name: {k: [] for k in KEYS} for name in ("uni", "bal")}
    pop = np.zeros(N_GROUPS, np.int64)
    for n, b in enumerate(blocks):
        obs = np.asarray(z["x"][:, :, :, b], np.float32)[:, ::2, ::2]
        err = np.asarray(z["e"][:, :, :, b], np.float32)[:, ::2, ::2]
        dem = np.asarray(z["y"][:N_BINS, :, :, b], np.float32)
        tol = np.asarray(z["m"][:, :, b])
        # every pixel with a reading and a noise estimate: solver-unsolved pixels and
        # pixels with a zeroed channel stay in
        valid = np.isfinite(obs).all(0) & np.isfinite(err).all(0) & (err > 0).all(0)
        ratio = (obs / thr).max(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            group = np.searchsorted(INTENSITY_EDGES, np.log10(ratio), side="right")
        group = np.clip(group, 0, N_GROUPS - 1)
        pop += np.bincount(group[valid], minlength=N_GROUPS)
        picks = {"uni": valid & (rng.random(valid.shape) < p_uni)}
        if keep is not None:
            picks["bal"] = valid & (rng.random(valid.shape) < np.asarray(keep)[group])
        for name, sel in picks.items():
            if sel.any():
                r = rows[name]
                r["obs"].append(obs[:, sel].T)
                r["err"].append(err[:, sel].T)
                r["dem"].append(dem[:, sel].T)
                r["tol"].append(tol[sel])
                r["group"].append(group[sel])
                r["inten"].append(ratio[sel])
        if (n + 1) % 500 == 0:
            print(f"[{phase} w{wid}] {n + 1}/{len(blocks)} blocks", flush=True)
    out = {name: ({k: np.concatenate(v) for k, v in r.items()} if r["obs"] else None)
           for name, r in rows.items()}
    return out, pop


def gather(cfg, phase, blocks, p_uni, keep, workers):
    chunks = [c for c in np.array_split(np.asarray(blocks), workers) if len(c)]
    jobs = [(cfg, phase, c.tolist(), i, p_uni, keep) for i, c in enumerate(chunks)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(collect, jobs)
    pop = sum(p for _, p in parts)
    out = {}
    for name in ("uni", "bal"):
        have = [d[name] for d, _ in parts if d[name] is not None]
        out[name] = {k: np.concatenate([h[k] for h in have]) for k in KEYS} if have else None
    return out, pop


# ── the linear program ────────────────────────────────────────────────────────

def init_worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.train_scaled import load_operators
    D_t, B_t, _, _ = load_operators(torch.device("cpu"))
    G["D"] = D_t.numpy().astype(np.float64)
    G["B"] = B_t.numpy().astype(np.float64)
    G["cfg"] = cfg


def scaled(obs, err, t):
    """Rows in half-widths: the band is lo <= A x <= hi with hi - lo = 2."""
    h = t * err
    return G["D"] / h[:, None], obs / h - 1.0, obs / h + 1.0, h


def lp_pixel(A, lo, hi):
    """min sum(x) s.t. lo <= A x <= hi, x >= 0 -> (status, optimum, x, y_lo, y_hi).

    y_lo[c] is the weight it costs to raise channel c's lower edge by one half-width,
    y_hi[c] to lower its upper edge; both >= 0 (scipy status 0 solved, 2 infeasible)."""
    from scipy.optimize import linprog
    c = A.shape[0]
    r = linprog(np.ones(A.shape[1]), A_ub=np.vstack([A, -A]), b_ub=np.concatenate([hi, -lo]),
                bounds=(0, None), method="highs")
    if r.status != 0:
        return r.status, np.nan, None, None, None
    m = -np.asarray(r.ineqlin.marginals, dtype=np.float64)
    return 0, float(r.fun), r.x, np.maximum(m[c:], 0.0), np.maximum(m[:c], 0.0)


def min_relax(A, lo, hi):
    """Smallest t >= 0 such that lo - t <= A x <= hi + t has a solution x >= 0: the band
    multiple 1 + t that first admits a DEM (the solver's ladder is 3x and 5x, t = 2 and 4)."""
    from scipy.optimize import linprog
    c, k = A.shape
    one = np.ones((c, 1))
    cost = np.zeros(k + 1)
    cost[-1] = 1.0
    r = linprog(cost, A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.concatenate([hi, -lo]),
                bounds=(0, None), method="highs")
    if r.status != 0:
        return np.nan, np.zeros(k)
    return float(r.x[-1]), r.x[:k]


def solve(f, x0, tol=1e-6):
    """Minimise a convex, once-differentiable f over x >= 0 -> (x, converged).

    SLSQP then an L-BFGS-B polish, repeated while the point is not stationary.
    L-BFGS-B alone is not safe here: on the stiffest pixels (curvature across a
    band edge orders above the slope along it) its line search gives up far from
    the minimum and can still report success. Convergence is judged by the
    projected gradient, whose natural scale is 1 (the slope of sum(x))."""
    from scipy.optimize import minimize
    bounds = [(0, None)] * len(x0)

    def stationarity(x):
        g = f(x)[1]
        return float(np.abs(np.where(x > 0, g, np.minimum(g, 0.0))).max())

    best_x = np.maximum(np.asarray(x0, np.float64), 0)
    best_f = f(best_x)[0]
    for _ in range(4):
        for method, opts in (("SLSQP", {"maxiter": 3000, "ftol": 1e-15}), ("L-BFGS-B", LBFGS)):
            r = minimize(f, best_x, jac=True, method=method, bounds=bounds, options=opts)
            x = np.maximum(r.x, 0)
            fx = f(x)[0]
            if fx < best_f:
                best_x, best_f = x, fx
        if stationarity(best_x) <= tol:
            return best_x, True
    return best_x, False


def min_l2(A, lo, hi, x0):
    """Least-squares violation: min over x >= 0 of sum relu(g)^2 -> signed violation per
    channel in half-widths (negative below the band, positive above)."""
    def f(x):
        r = A @ x
        a, b = np.maximum(lo - r, 0), np.maximum(r - hi, 0)
        return float((a * a + b * b).sum()), 2.0 * (A.T @ (b - a))

    x, _ = solve(f, np.maximum(x0, 0))
    rec = A @ x
    return np.maximum(rec - hi, 0) - np.maximum(lo - rec, 0)


def lp_chunk(job):
    obs, err, keep_x = job
    t, B = G["cfg"]["tolfac"], G["B"]
    n, c = obs.shape
    k = G["D"].shape[1]
    out = {"status": np.full(n, -1, np.int8), "obj": np.full(n, np.nan, np.float32),
           "y_lo": np.zeros((n, c), np.float32), "y_hi": np.zeros((n, c), np.float32),
           "tmin": np.zeros(n, np.float32), "u2": np.zeros((n, c), np.float32),
           "dem_lp": np.full((n, B.shape[0]), np.nan, np.float32)}
    if keep_x:
        out["x0"] = np.zeros((n, k), np.float32)
    for i in range(n):
        A, lo, hi, _ = scaled(obs[i].astype(np.float64), err[i].astype(np.float64), t)
        st, ob, x, ylo, yhi = lp_pixel(A, lo, hi)
        out["status"][i] = st
        if st == 0:
            out["obj"][i], out["y_lo"][i], out["y_hi"][i] = ob, ylo, yhi
            out["dem_lp"][i] = B @ x
        else:
            tm, x = min_relax(A, lo, hi)
            out["tmin"][i] = tm
            out["u2"][i] = min_l2(A, lo, hi, x)
        if keep_x:
            out["x0"][i] = x
    return out


def run_lp(pool, data, workers, keep_x=False):
    n = len(data["obs"])
    edges = np.linspace(0, n, max(workers * 8, 1) + 1).astype(int)
    jobs = [(data["obs"][a:b], data["err"][a:b], keep_x) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    parts = pool.map(lp_chunk, jobs)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ── the stand-ins, each at its exact optimum ──────────────────────────────────

def al_step(A, lo, hi, y, m):
    """One augmented-Lagrangian solve, done through its dual.

    Minimising sum(x) + sum [relu(y + 2 m g)^2 - y^2] / (4 m) over x >= 0 and then
    setting y <- relu(y + 2 m g) is the proximal step on the linear program's dual:

        y_new = argmax over y' >= 0, A^T (y'_lo - y'_hi) <= 1 of
                lo . y'_lo - hi . y'_hi - sum (y' - y)^2 / (4 m).

    Twelve variables and strictly concave, where the primal form has 54 variables,
    flat directions and curvature that stalls quasi-Newton line searches.
    Returns (y_new [lower six, upper six], converged)."""
    from scipy.optimize import minimize
    c = A.shape[0]
    lin = np.concatenate([lo, -hi])
    mm = np.concatenate([m, m])
    Gm = np.hstack([A.T, -A.T])                              # [basis, 2c]
    cons = {"type": "ineq", "fun": lambda z: 1.0 - Gm @ z, "jac": lambda z: -Gm}
    r = minimize(lambda z: (((z - y) ** 2) / (4 * mm)).sum() - lin @ z, y,
                 jac=lambda z: (z - y) / (2 * mm) - lin, method="SLSQP",
                 bounds=[(0, None)] * (2 * c), constraints=[cons], options={"maxiter": 1000, "ftol": 1e-15})
    z = np.maximum(r.x, 0)
    # status 8 (no further descent along the search direction) is how SLSQP stops at
    # the optimum under this tolerance; the primal form reproduces these iterates
    return z, r.status in (0, 8) and float((Gm @ z).max()) <= 1 + 1e-7


def forms_chunk(job):
    from scipy.optimize import linprog
    obs, err, x0 = job
    t = G["cfg"]["tolfac"]
    D = G["D"]
    n, c = obs.shape
    k = D.shape[1]
    lam_bar = 1.0 / D.max(axis=1)                 # single-channel price, weight per DN/s
    out = {"pen_vmax": np.zeros((n, len(KAPPAS)), np.float32),
           "pen_sum": np.zeros((n, len(KAPPAS)), np.float32),
           "al_vmax": np.zeros((n, len(AL_SETTINGS), AL_ITERS), np.float32),
           "al_sum": np.zeros((n, len(AL_SETTINGS), AL_ITERS), np.float32),
           "al_fail": np.zeros((n, len(AL_SETTINGS)), np.int16)}
    eye = np.eye(c)
    zero = np.zeros((c, c))
    for i in range(n):
        A, lo, hi, h = scaled(obs[i].astype(np.float64), err[i].astype(np.float64), t)
        P = lam_bar * h                           # the price scale, weight per half-width
        # exact penalty: min sum(x) + rho (s_lo + s_hi), lo - s_lo <= A x <= hi + s_hi
        A_ub = np.block([[A, zero, -eye], [-A, -eye, zero]])
        b_ub = np.concatenate([hi, -lo])
        for j, kap in enumerate(KAPPAS):
            rho = kap * P
            r = linprog(np.concatenate([np.ones(k), rho, rho]), A_ub=A_ub, b_ub=b_ub,
                        bounds=(0, None), method="highs")
            if r.status == 0:
                out["pen_vmax"][i, j] = r.x[k:].max()
                out["pen_sum"][i, j] = r.x[:k].sum()
            else:
                out["pen_vmax"][i, j] = out["pen_sum"][i, j] = np.nan
        # augmented Lagrangian from y = 0: the first solve is the quadratic penalty itself
        for j, (kind, val) in enumerate(AL_SETTINGS):
            m = np.full(c, val)
            y = np.zeros(2 * c)
            lin = np.concatenate([lo, -hi])
            prev = np.inf
            for it in range(AL_ITERS):
                y_new, ok = al_step(A, lo, hi, y, m)
                out["al_fail"][i, j] += int(not ok)
                mm = np.concatenate([m, m])
                # y_new = relu(y + 2 m g): the violation is the part of the step that raised y
                v = float((np.maximum(y_new - y, 0) / (2 * mm)).max())
                out["al_vmax"][i, j, it] = v
                # sum(x) from the equal primal and dual values of the solve
                out["al_sum"][i, j, it] = lin @ y_new - (((y_new - y) ** 2) / (4 * mm)).sum() \
                    - ((y_new ** 2 - y ** 2) / (4 * mm)).sum()
                y = y_new
                if kind == "ada" and v > 1e-6 and v > 0.25 * prev:
                    m = m * 10.0
                prev = v
    return out


# ── summaries ─────────────────────────────────────────────────────────────────

def pct(a, qs=(50, 90, 99, 100)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {("max" if q == 100 else f"p{q}"): None for q in qs}
    return {("max" if q == 100 else f"p{q}"): float(np.percentile(a, q)) for q in qs}


def share(mask):
    mask = np.asarray(mask)
    return float(100 * mask.mean()) if mask.size else None


def population(data, lp, t):
    """What the population looks like to the linear program (a uniform sample)."""
    obs, err, tol, group = data["obs"], data["err"], data["tol"], data["group"]
    feas = lp["status"] == 0
    # HiGHS reports most infeasible programs as status 2 and a few as status 4 (numerical
    # trouble); both were then given the smallest band multiple that admits a DEM
    infe = lp["status"] != 0
    zero = obs <= MIN_OBS
    anyz = zero.any(1)
    brightz = zero[:, 2] | zero[:, 3]                       # 171 or 193 A zeroed
    P = (t * err) / G["D"].max(axis=1)[None, :]
    q_lo, q_hi = lp["y_lo"] / P, lp["y_hi"] / P             # price over the single-channel scale
    qmax = np.maximum(q_lo, q_hi).max(1)
    out = {"n": int(len(obs)),
           "lp_feasible_pct": share(feas), "lp_infeasible_pct": share(infe),
           "lp_infeasible_reported_as_numerical_pct": share(lp["status"] == 4),
           "zero_channel_pct": share(anyz), "zero_171_or_193_pct": share(brightz),
           "zero_per_channel_pct": {c: share(zero[:, j]) for j, c in enumerate(CHANNELS)},
           "feasible_pct_given": {"no_zero_channel": share(feas[~anyz]), "zero_channel": share(feas[anyz]),
                                  "zero_171_or_193": share(feas[brightz])},
           "solver_flag_pct": {str(v): share(tol == v) for v in (0, 1, 3, 5)},
           "lp_feasible_pct_by_solver_flag": {str(v): share(feas[tol == v]) for v in (0, 1, 3, 5)},
           "bands": {}}
    for g, lab in enumerate(INTENSITY_LABELS):
        j = group == g
        if j.any():
            out["bands"][lab] = {"n": int(j.sum()), "share_pct": share(j), "lp_feasible_pct": share(feas[j]),
                                 "zero_channel_pct": share(anyz[j])}
    # dual prices at feasible pixels
    f = np.flatnonzero(feas)
    out["price_ratio"] = {
        "max_over_channels": pct(qmax[f], (50, 90, 99, 99.9, 100)),
        "lower_edge": {c: pct(q_lo[f, j], (50, 90, 99, 100)) for j, c in enumerate(CHANNELS)},
        "upper_edge": {c: pct(q_hi[f, j], (50, 90, 99, 100)) for j, c in enumerate(CHANNELS)},
        "active_lower_pct": {c: share(lp["y_lo"][f, j] > 1e-9) for j, c in enumerate(CHANNELS)},
        "active_upper_pct": {c: share(lp["y_hi"][f, j] > 1e-9) for j, c in enumerate(CHANNELS)},
        "no_active_channel_pct": share(qmax[f] <= 1e-9),
        "exact_if_price_at_most_kappa_pct": {str(k): share(qmax[f] <= k) for k in KAPPAS}}
    # how far from feasible the infeasible pixels are
    i = np.flatnonzero(infe)
    if i.size:
        tm, u2 = lp["tmin"][i], lp["u2"][i]
        l2 = np.sqrt(np.square(u2).sum(1))
        worst = np.abs(u2).argmax(1)
        out["infeasible"] = {
            "n": int(i.size),
            "band_multiple_needed_minus_1": pct(tm, (10, 50, 90, 99, 100)),
            "pct_feasible_within": {f"{1 + r:g}x": share(tm <= r) for r in RELAX},
            "l2_violation_halfwidths": pct(l2, (10, 50, 90, 99, 100)),
            "worst_channel_share_pct": {c: share(worst == j) for j, c in enumerate(CHANNELS)},
            "worst_channel_below_band_pct": share(u2[np.arange(i.size), worst] < 0),
            "zero_channel_pct": share(anyz[i]),
            "by_solver_flag": {str(v): {"n": int((tol[i] == v).sum()),
                                        "band_multiple_needed_minus_1": pct(tm[tol[i] == v], (50, 90, 100))}
                               for v in (0, 1, 3, 5)},
            "by_band": {lab: {"n": int((group[i] == g).sum()),
                              "band_multiple_needed_minus_1": pct(tm[group[i] == g], (50, 90, 100))}
                        for g, lab in enumerate(INTENSITY_LABELS) if (group[i] == g).any()}}
    return out


def forms_summary(data, lp, fm, pop):
    """The stand-ins on the balanced sample, per band and weighted to the population."""
    group = data["group"]
    feas = lp["status"] == 0
    obj = lp["obj"].astype(np.float64)
    w_band = pop / pop.sum()

    def block(idx):
        b = {"n": int(idx.size), "exact_penalty": {}, "augmented_lagrangian": {}}
        if idx.size == 0:
            return b
        for j, kap in enumerate(KAPPAS):
            v = fm["pen_vmax"][idx, j]
            b["exact_penalty"][str(kap)] = {
                "inside_pct": {f"{e:g}": share(v <= e) for e in INSIDE},
                "max_violation": pct(v, (50, 99, 100)),
                "weight_over_lp": pct(fm["pen_sum"][idx, j] / np.maximum(obj[idx], 1e-30), (50,))["p50"]}
        for j, (kind, val) in enumerate(AL_SETTINGS):
            its = []
            for it in range(AL_ITERS):
                v = fm["al_vmax"][idx, j, it]
                its.append({"inside_pct": {f"{e:g}": share(v <= e) for e in INSIDE},
                            "max_violation": pct(v, (50, 99, 100)),
                            "weight_over_lp": pct(fm["al_sum"][idx, j, it] / np.maximum(obj[idx], 1e-30), (50,))["p50"]})
            b["augmented_lagrangian"][f"{kind}{val:g}"] = {"iterations": its,
                                                         "solves_not_converged": int(fm["al_fail"][idx, j].sum())}
        return b

    out = {"bands": {}, "feasible_weighted": {"exact_penalty": {}, "augmented_lagrangian": {}}}
    for g, lab in enumerate(INTENSITY_LABELS):
        out["bands"][lab] = {"feasible": block(np.flatnonzero((group == g) & feas)),
                             "infeasible": block(np.flatnonzero((group == g) & ~feas))}
    # population-weighted share of feasible pixels inside the band (band weights x band shares)
    fw = np.array([w_band[g] * np.mean(feas[group == g]) if (group == g).any() else 0.0 for g in range(N_GROUPS)])
    fw = fw / max(fw.sum(), 1e-30)

    def weighted(get):
        tot = 0.0
        for g, lab in enumerate(INTENSITY_LABELS):
            blk = out["bands"][lab]["feasible"]
            if blk["n"]:
                tot += fw[g] * get(blk)
        return float(tot)

    for kap in KAPPAS:
        out["feasible_weighted"]["exact_penalty"][str(kap)] = {
            f"{e:g}": weighted(lambda b, e=e, kap=kap: b["exact_penalty"][str(kap)]["inside_pct"][f"{e:g}"]) for e in INSIDE}
    for kind, val in AL_SETTINGS:
        key = f"{kind}{val:g}"
        out["feasible_weighted"]["augmented_lagrangian"][key] = [
            {f"{e:g}": weighted(lambda b, e=e, it=it, key=key: b["augmented_lagrangian"][key]["iterations"][it]["inside_pct"][f"{e:g}"])
             for e in INSIDE} for it in range(AL_ITERS)]
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def fmt(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def render(r):
    f = fmt
    L = [f"Constrained problem vs its unconstrained stand-ins (t = {r['settings']['tolfac']}, zeroed channels keep their band); "
         f"{r['runtime_s']} s"]
    for name in ("uniform_train", "uniform_val"):
        p = r[name]
        L.append(f"\n== {name}: {p['n']:,} pixels")
        L.append(f"   a DEM fits all six bands (LP feasible): {f(p['lp_feasible_pct'], '.2f')}%   none fits: "
                 f"{f(p['lp_infeasible_pct'], '.2f')}%")
        L.append(f"   pixels with a zeroed channel: {f(p['zero_channel_pct'])}% (171 or 193 A: {f(p['zero_171_or_193_pct'])}%); per channel "
                 + ", ".join(f"{c} {f(v)}%" for c, v in p["zero_per_channel_pct"].items()))
        g = p["feasible_pct_given"]
        L.append(f"   feasible share: no zeroed channel {f(g['no_zero_channel'], '.2f')}%, a zeroed channel {f(g['zero_channel'], '.2f')}%, "
                 f"171 or 193 A zeroed {f(g['zero_171_or_193'], '.2f')}%")
        L.append("   solver flag (share of pixels -> LP feasible at 1.4 sigma): "
                 + ", ".join(f"{k}: {f(p['solver_flag_pct'][k], '.2f')}% -> {f(p['lp_feasible_pct_by_solver_flag'][k])}%"
                             for k in ("1", "3", "5", "0")))
        L.append("   by brightness (share, feasible, zeroed channel): "
                 + "; ".join(f"{lab} {f(b['share_pct'], '.3g')}% {f(b['lp_feasible_pct'], '.0f')}% {f(b['zero_channel_pct'], '.0f')}%"
                             for lab, b in p["bands"].items()))
        pr = p["price_ratio"]
        m = pr["max_over_channels"]
        L.append(f"   dual price over the single-channel scale, largest channel: p50 {f(m['p50'], '.2f')}, p90 {f(m['p90'], '.2f')}, "
                 f"p99 {f(m['p99'], '.1f')}, p99.9 {f(m['p99.9'], '.0f')}, max {f(m['max'], '.0f')}; no channel at its edge: "
                 f"{f(pr['no_active_channel_pct'])}%")
        L.append("   exact penalty would be exact at price kappa x scale for: "
                 + ", ".join(f"kappa {k}: {f(v, '.2f')}%" for k, v in pr["exact_if_price_at_most_kappa_pct"].items()))
        L.append("   channel at its lower / upper edge: "
                 + ", ".join(f"{c} {f(pr['active_lower_pct'][c], '.0f')}/{f(pr['active_upper_pct'][c], '.0f')}%" for c in CHANNELS))
        if "infeasible" in p:
            q = p["infeasible"]
            t = q["band_multiple_needed_minus_1"]
            L.append(f"   infeasible pixels ({q['n']:,}): band multiple needed p10 {1 + t['p10']:.2f}x, p50 {1 + t['p50']:.2f}x, "
                     f"p90 {1 + t['p90']:.2f}x, p99 {1 + t['p99']:.1f}x, max {1 + t['max']:.0f}x")
            L.append("   feasible once the band is widened to: "
                     + ", ".join(f"{k} {f(v)}%" for k, v in q["pct_feasible_within"].items()))
            l2 = q["l2_violation_halfwidths"]
            L.append(f"   least-squares violation (half-widths): p10 {f(l2['p10'], '.3f')}, p50 {f(l2['p50'], '.3f')}, p90 {f(l2['p90'], '.2f')}, "
                     f"p99 {f(l2['p99'], '.1f')}, max {f(l2['max'], '.0f')}; worst channel "
                     + ", ".join(f"{c} {f(v, '.0f')}%" for c, v in q["worst_channel_share_pct"].items())
                     + f"; below its band {f(q['worst_channel_below_band_pct'], '.0f')}%; with a zeroed channel {f(q['zero_channel_pct'], '.0f')}%")
            L.append("   by solver flag (n, band multiple needed p50 / p90 / max): "
                     + "; ".join(f"{k}: {v['n']:,}, " + " / ".join(
                         "-" if v['band_multiple_needed_minus_1'][s] is None else f"{1 + v['band_multiple_needed_minus_1'][s]:.2f}x"
                         for s in ("p50", "p90", "max")) for k, v in q["by_solver_flag"].items()))
    fm = r["forms"]
    L.append("\n== stand-ins at their exact optimum, balanced validation sample; share of FEASIBLE pixels inside every band "
             "(within 0.1 / 0.01 / 0.001 half-widths), weighted to the population")
    L.append("   exact penalty, price = kappa x single-channel scale:")
    for k, v in fm["feasible_weighted"]["exact_penalty"].items():
        L.append(f"      kappa {k:>5}: {f(v['0.1'], '.2f')} / {f(v['0.01'], '.2f')} / {f(v['0.001'], '.2f')}%")
    L.append("   augmented Lagrangian, by outer iteration (iteration 1 is the plain quadratic penalty):")
    for k, its in fm["feasible_weighted"]["augmented_lagrangian"].items():
        L.append(f"      {k:>7}: " + "  ".join(f"[{i + 1}] {f(v['0.1'], '.1f')}/{f(v['0.01'], '.1f')}/{f(v['0.001'], '.1f')}"
                                               for i, v in enumerate(its)))
    for lab, b in fm["bands"].items():
        fe, inf = b["feasible"], b["infeasible"]
        if not fe["n"] and not inf["n"]:
            continue
        L.append(f"\n   -- {lab}: {fe['n']} feasible, {inf['n']} infeasible")
        if fe["n"]:
            L.append("      exact penalty, feasible inside within 0.001 (max violation p99; weight / LP): "
                     + ", ".join(f"k{k} {f(v['inside_pct']['0.001'], '.0f')}% ({f(v['max_violation']['p99'], '.2g')}; {f(v['weight_over_lp'], '.3f')})"
                                 for k, v in fe["exact_penalty"].items()))
            for k, v in fe["augmented_lagrangian"].items():
                L.append(f"      AL {k:>7}, feasible inside within 0.001 by iteration: "
                         + " ".join(f"{f(it['inside_pct']['0.001'], '.0f')}%" for it in v["iterations"])
                         + f"; p99 violation " + " ".join(f(it['max_violation']['p99'], '.2g') for it in v["iterations"])
                         + f"; weight / LP {f(v['iterations'][-1]['weight_over_lp'], '.4f')}; unconverged solves {v['solves_not_converged']}")
        if inf["n"]:
            L.append("      infeasible, exact penalty max violation p50: "
                     + ", ".join(f"k{k} {f(v['max_violation']['p50'], '.3g')}" for k, v in inf["exact_penalty"].items()))
            for k, v in inf["augmented_lagrangian"].items():
                L.append(f"      infeasible, AL {k:>7} max violation p50 by iteration: "
                         + " ".join(f(it['max_violation']['p50'], '.3g') for it in v["iterations"]))
    return "\n".join(L)


def main():
    args = parse_args()
    if args.from_npz:
        return from_npz(args)
    import zarr
    t0 = time.time()
    with open(args.thresholds) as fh:
        payload = json.load(fh)
    thresholds = np.asarray(payload.get("test", payload), np.float64)
    cfg = {"root": os.path.abspath(args.root), "thresholds": thresholds.tolist(), "seed": args.seed,
           "tolfac": args.tolfac}
    rng = np.random.default_rng(args.seed)
    nb_train = zarr.open(os.path.join(args.root, "train_x.zarr"), mode="r").shape[-1]
    nb_val = zarr.open(os.path.join(args.root, "val_x.zarr"), mode="r").shape[-1]
    train_blocks = np.sort(rng.choice(nb_train, size=min(args.train_blocks, nb_train), replace=False))
    val_blocks = np.arange(nb_val if args.val_blocks is None else min(args.val_blocks, nb_val))
    px = 128 * 128 * 0.9                                   # rough valid pixels per block
    p_train = min(1.0, 1.15 * args.n_train / (len(train_blocks) * px))
    p_val = min(1.0, 1.15 * args.n_hold / (len(val_blocks) * px))
    keep = np.minimum(1.0, 1.5 * args.per_group / (SHARE * len(val_blocks) * px))
    keep[CORE] = 1.0

    tr, pop_train = gather(cfg, "train", train_blocks, p_train, None, args.workers)
    va, pop_val = gather(cfg, "val", val_blocks, p_val, keep.tolist(), args.workers)

    def cut(d, n):
        idx = np.sort(rng.choice(len(d["obs"]), size=min(n, len(d["obs"])), replace=False))
        return {k: v[idx] for k, v in d.items()}

    train, hold = cut(tr["uni"], args.n_train), cut(va["uni"], args.n_hold)
    bal = va["bal"]
    take = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(bal["group"] == g)
        cap = args.core_cap if g == CORE else args.per_group
        take.append(rng.choice(idx, size=min(cap, idx.size), replace=False) if idx.size else idx)
    take = np.sort(np.concatenate(take))
    bal = {k: v[take] for k, v in bal.items()}
    print(f"collected train {len(train['obs']):,} (from {len(train_blocks):,} blocks), hold {len(hold['obs']):,}, "
          f"balanced {len(bal['obs']):,} {np.bincount(bal['group'], minlength=N_GROUPS).tolist()} in {time.time() - t0:.0f}s",
          flush=True)

    init_worker(cfg)
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        t1 = time.time()
        lp_bal = run_lp(pool, bal, args.workers, keep_x=True)
        lp_hold = run_lp(pool, hold, args.workers)
        lp_train = run_lp(pool, train, args.workers)
        print(f"linear programs in {time.time() - t1:.0f}s; feasible: train {100 * np.mean(lp_train['status'] == 0):.2f}%, "
              f"hold {100 * np.mean(lp_hold['status'] == 0):.2f}%, balanced {100 * np.mean(lp_bal['status'] == 0):.2f}%", flush=True)
        t2 = time.time()
        n = len(bal["obs"])
        edges = np.linspace(0, n, args.workers * 8 + 1).astype(int)
        jobs = [(bal["obs"][a:b], bal["err"][a:b], lp_bal["x0"][a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        parts = pool.map(forms_chunk, jobs)
        fm = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        print(f"stand-ins in {time.time() - t2:.0f}s", flush=True)

    write(args, cfg, t0, train, hold, bal, lp_train, lp_hold, lp_bal, fm, pop_val, pop_train,
          len(train_blocks), len(val_blocks))


def write(args, cfg, t0, train, hold, bal, lp_train, lp_hold, lp_bal, fm, pop_val, pop_train, n_tb, n_vb,
          save=True):
    result = {"purpose": "the solver's linear program against its unconstrained stand-ins, per pixel, no network",
              "root": cfg["root"],
              "settings": {"tolfac": args.tolfac, "seed": args.seed, "min_obs": MIN_OBS, "kappas": list(KAPPAS),
                           "al_settings": [f"{k}{v:g}" for k, v in AL_SETTINGS], "al_iterations": AL_ITERS,
                           "train_blocks": int(n_tb), "val_blocks": int(n_vb)},
              "single_channel_price_per_dn": {c: float(1.0 / G["D"].max(axis=1)[j]) for j, c in enumerate(CHANNELS)},
              "valid_pixels_per_band_val": dict(zip(INTENSITY_LABELS, np.asarray(pop_val).tolist())),
              "valid_pixels_per_band_train_blocks": dict(zip(INTENSITY_LABELS, np.asarray(pop_train).tolist())),
              "uniform_train": population(train, lp_train, args.tolfac),
              "uniform_val": population(hold, lp_hold, args.tolfac),
              "balanced_n_per_band": dict(zip(INTENSITY_LABELS, np.bincount(bal["group"], minlength=N_GROUPS).tolist())),
              "forms": forms_summary(bal, lp_bal, fm, np.asarray(pop_val, np.float64)),
              "runtime_s": round(time.time() - t0, 1)}
    result = clean(result)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=1)
    text = render(result)
    print(text)
    stem = os.path.splitext(args.output)[0]
    with open(stem + ".txt", "w") as fh:
        fh.write(text + "\n")
    if save:
        for name, d, lp in (("train", train, lp_train), ("hold", hold, lp_hold), ("balanced", bal, lp_bal)):
            extra = {f"forms_{k}": v for k, v in fm.items()} if name == "balanced" else {}
            if name == "balanced":
                extra |= {"pop_val": np.asarray(pop_val), "pop_train": np.asarray(pop_train),
                          "n_blocks": np.array([n_tb, n_vb])}
            np.savez_compressed(f"{stem}_{name}.npz", **d, **{f"lp_{k}": v for k, v in lp.items()}, **extra)
    print(f"\nwrote {args.output} (+ .txt{', _train/_hold/_balanced.npz' if save else ''}) in {time.time() - t0:.0f}s")


def from_npz(args):
    """Summaries again from the pixel sets of an earlier run."""
    t0 = time.time()
    cfg = {"root": os.path.abspath(args.root), "tolfac": args.tolfac, "seed": args.seed}
    init_worker(cfg)
    stem = os.path.splitext(args.output)[0]
    sets, lps, fm, extra = {}, {}, {}, {}
    for name in ("train", "hold", "balanced"):
        z = np.load(f"{stem}_{name}.npz")
        sets[name] = {k: z[k] for k in KEYS}
        lps[name] = {k[3:]: z[k] for k in z.files if k.startswith("lp_")}
        if name == "balanced":
            fm = {k[6:]: z[k] for k in z.files if k.startswith("forms_")}
            extra = {k: z[k] for k in ("pop_val", "pop_train", "n_blocks") if k in z.files}
    pop_val = extra.get("pop_val", np.bincount(sets["hold"]["group"], minlength=N_GROUPS))
    pop_train = extra.get("pop_train", np.bincount(sets["train"]["group"], minlength=N_GROUPS))
    nb = extra.get("n_blocks", np.array([0, 0]))
    write(args, cfg, t0, sets["train"], sets["hold"], sets["balanced"], lps["train"], lps["hold"], lps["balanced"],
          fm, pop_val, pop_train, nb[0], nb[1], save=False)


if __name__ == "__main__":
    main()
