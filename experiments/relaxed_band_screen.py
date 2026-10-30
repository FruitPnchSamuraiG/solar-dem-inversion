#!/usr/bin/env python3
"""Does relaxing the band where the solver relaxed it remove the cost of a stiff band?

The band-weight theory check (`mu_theory.py`) found that on validation the best answer
under our loss moves monotonically toward the solver as mu grows at every pixel where
the 1.4 sigma band is feasible, and away from it at the ~9% of pixels where no DEM fits
the band. There the solver widens its own band (to 3x, then 5x, `fullBP.py:515`) while
our loss keeps 1.4 sigma and, when stiff, fits the inconsistent observation instead.

This screen re-solves the best answer at those relaxed pixels with the solver's own band
(t = 1.4 x tolLevel) and recomputes the mu screen (slide 19 of the deck: population-
weighted DEM error, Quiet DEM MSE, flare-core 94 A and emission) for the fixed and the
relaxed band. Feasible pixels have the same band either way, so their optima are reused
from the theory check's arrays.

    python experiments/relaxed_band_screen.py --npz output/meeting_followup/bp_mu_theory_val.npz \
        --json results/plots/23_meeting_followup_20261024/bp_mu_theory_val.json \
        --output results/plots/23_meeting_followup_20261024/bp_relaxed_band_screen.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = ['<0.03x', '0.03-0.1x', '0.1-0.3x', '0.3-1x', '1-1.8x', '1.8-3.2x', '3.2-10x',
          '10-32x', '>=32x']
G = {}


def init_worker():
    import torch
    from experiments.train_scaled import load_operators
    D_t, B_t, _, _ = load_operators(torch.device("cpu"))
    G["D"] = D_t.numpy().astype(np.float64)
    G["B"] = B_t.numpy().astype(np.float64)


def solve(job):
    """Penalised best answer for one pixel at each mu, band half-width s = t * sigma."""
    from scipy.optimize import minimize
    o, s, starts, mus = job
    D = G["D"]
    lb, ub, s2 = o - s, o + s, s ** 2 + 1e-8
    out = []
    for mu in mus:
        def f(x):
            Dx = D @ x
            lo, hi = np.maximum(lb - Dx, 0), np.maximum(Dx - ub, 0)
            return (x.sum() + mu * ((lo ** 2 + hi ** 2) / s2).sum(),
                    1.0 + mu * (D.T @ (2 * (hi - lo) / s2)))
        best, bval = None, np.inf
        for x0 in starts:
            r = minimize(f, np.maximum(x0, 0), jac=True, method="L-BFGS-B",
                         bounds=[(0, None)] * D.shape[1],
                         options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-15,
                                  "gtol": 1e-10, "maxcor": 30})
            if r.fun < bval:
                best, bval = r.x, r.fun
        out.append(best)
    return np.stack(out)


def screen(x_opt, dem, obs, group, pop, D, B):
    """The slide-19 columns for one set of per-pixel answers [N, 54]."""
    pred = x_opt @ B.T
    mse = np.square(pred - dem).mean(1)
    band_mse = np.array([mse[group == g].mean() if (group == g).any() else 0.0 for g in range(9)])
    w = pop / pop.sum()
    quiet = slice(0, 4)
    core = group == 8
    rec94 = (x_opt[core] @ D.T)[:, 0]
    return {"dem_error": float((w * band_mse).sum()),
            "quiet_mse": float((w[quiet] * band_mse[quiet]).sum() / w[quiet].sum()),
            "core_94": float(rec94.sum() / obs[core, 0].sum()),
            "core_emission": float(pred[core].sum() / dem[core].sum()),
            "band_mse": dict(zip(LABELS, band_mse.round(6).tolist()))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz", required=True, help="arrays from mu_theory.py")
    p.add_argument("--json", required=True, help="summary from mu_theory.py (population per band)")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tolfac", type=float, default=1.4)
    args = p.parse_args()
    t0 = time.time()

    z = np.load(args.npz, allow_pickle=True)
    with open(args.json) as f:
        per_band = json.load(f)["population_per_band"]
    pop = np.array([per_band[k] for k in LABELS], float)
    mus = [float(m) for m in z["mus"]]
    obs, err, dem = (z[k].astype(np.float64) for k in ("obs", "err", "dem"))
    tol, group, x_sol, x_fixed = z["tol"], z["group"], z["x_sol"], z["x_opt"]

    relaxed = np.flatnonzero(tol > 1)
    jobs = []
    for i in relaxed:
        s = args.tolfac * float(tol[i]) * err[i]
        jobs.append((obs[i], s, [x_sol[i]] + [x_fixed[m, i] for m in range(len(mus))], mus))
    print(f"{len(relaxed):,} of {len(tol):,} pixels relaxed by the solver; re-solving at "
          f"mu {mus} with its own band", flush=True)
    with mp.Pool(args.workers, initializer=init_worker) as pool:
        sols = pool.map(solve, jobs, chunksize=8)
    init_worker()
    D, B = G["D"], G["B"]

    x_relaxed = x_fixed.copy()
    for k, i in enumerate(relaxed):
        x_relaxed[:, i] = sols[k]
    result = {"purpose": __doc__.split("\n")[0], "npz": args.npz, "mus": mus,
              "n_pixels": int(len(tol)), "n_relaxed": int(len(relaxed)),
              "relaxed_by_band": {LABELS[g]: int(((group == g) & (tol > 1)).sum()) for g in range(9)},
              "population_per_band": dict(zip(LABELS, pop.tolist())),
              "fixed": {}, "relaxed": {}, "runtime_s": None}
    for m, mu in enumerate(mus):
        result["fixed"][f"{mu:g}"] = screen(x_fixed[m], dem, obs, group, pop, D, B)
        result["relaxed"][f"{mu:g}"] = screen(x_relaxed[m], dem, obs, group, pop, D, B)
    result["runtime_s"] = round(time.time() - t0, 1)

    lines = [f"Relaxed-band screen on {len(tol):,} validation pixels ({len(relaxed):,} relaxed by the "
             f"solver); population-weighted as the mu screen",
             f"{'mu':>6} | {'DEM error':>21} | {'Quiet MSE':>19} | {'core 94 A':>15} | {'core emission':>15}",
             f"{'':>6} | {'fixed':>10} {'relaxed':>10} | {'fixed':>9} {'relaxed':>9} | "
             f"{'fixed':>7} {'relax':>7} | {'fixed':>7} {'relax':>7}"]
    for mu in mus:
        a, b = result["fixed"][f"{mu:g}"], result["relaxed"][f"{mu:g}"]
        lines.append(f"{mu:>6g} | {a['dem_error']:>10.3f} {b['dem_error']:>10.3f} | "
                     f"{a['quiet_mse']:>9.4f} {b['quiet_mse']:>9.4f} | "
                     f"{a['core_94']:>7.1%} {b['core_94']:>7.1%} | "
                     f"{a['core_emission']:>7.1%} {b['core_emission']:>7.1%}")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
