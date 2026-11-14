"""Exact per-pixel solve of the band problem, for the pixels a network gets worst.

At a flagged pixel the network's answer is replaced by the optimum of the pixel's own
problem: least total emission with every channel inside its band (reading +/- 1.4 sigma; a
zeroed channel keeps its band around zero). That is the problem the network is trained on
(its loss is the method of multipliers for it), so this is test-time optimization run to
convergence, done directly as the small linear program it is (54 weights, 12 band edges).
Only the pixel's readings and noise are used, never a solver DEM.

The band ladder is basis pursuit's own: the tight band if any DEM fits it, else 3x, else
5x. Which rung applies is decided by our own feasibility check (the smallest band multiple
that admits a DEM), not read from the solver's output. Pixels that fit no band up to 5x
keep the network's answer (the solver leaves those unsolved).

    w, level, need = solve_pixels(obs, half, D, workers=15)
"""
import multiprocessing as mp

import numpy as np

LADDER = (1, 3, 5)
_G = {}


def _init(D):
    _G["D"] = np.asarray(D, dtype=np.float64)


def _lp(A, lo, hi):
    """min sum(w) s.t. lo <= A w <= hi, w >= 0 -> w, or None if HiGHS finds no solution."""
    from scipy.optimize import linprog
    r = linprog(np.ones(A.shape[1]), A_ub=np.vstack([A, -A]), b_ub=np.concatenate([hi, -lo]),
                bounds=(0, None), method="highs")
    return r.x if r.status == 0 else None


def _min_band(A, lo, hi):
    """Smallest band multiple 1 + s (s >= 0, in tight half-widths) that admits a DEM w >= 0."""
    from scipy.optimize import linprog
    c, k = A.shape
    one = np.ones((c, 1))
    cost = np.zeros(k + 1)
    cost[-1] = 1.0
    r = linprog(cost, A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.concatenate([hi, -lo]),
                bounds=(0, None), method="highs")
    return 1.0 + float(r.x[-1]) if r.status == 0 else np.inf


def _chunk(job):
    obs, half = job
    D = _G["D"]
    n = len(obs)
    w = np.zeros((n, D.shape[1]))
    level = np.zeros(n, dtype=np.int8)
    need = np.full(n, np.nan)
    for i in range(n):
        o, h = obs[i], half[i]
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(h)) and np.all(h > 0)):
            continue
        need[i] = _min_band(D / h[:, None], o / h - 1.0, o / h + 1.0)
        for m in LADDER:
            if need[i] <= m * (1 + 1e-9):
                hm = m * h
                x = _lp(D / hm[:, None], o / hm - 1.0, o / hm + 1.0)
                if x is not None:
                    w[i], level[i] = np.maximum(x, 0.0), m
                    break
    return w, level, need


def solve_pixels(obs, half, D, workers, chunk=1000):
    """obs, half: [n, 6] in DN/s (half = 1.4 sigma); D: [6, k] channel response of each basis
    element. Returns w [n, k] (zeros where unsolved), level [n] (band multiple used; 0 = none
    up to 5x) and need [n] (smallest admitting band multiple)."""
    obs = np.asarray(obs, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    k = np.asarray(D).shape[1]
    if len(obs) == 0:
        return np.zeros((0, k)), np.zeros(0, np.int8), np.zeros(0)
    jobs = [(obs[s:s + chunk], half[s:s + chunk]) for s in range(0, len(obs), chunk)]
    with mp.get_context("spawn").Pool(max(1, workers), initializer=_init, initargs=(D,)) as pool:
        parts = pool.map(_chunk, jobs)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))
