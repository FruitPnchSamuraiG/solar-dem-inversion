#!/usr/bin/env python3
"""The exact range of the penalty law, pixel by pixel.

Above a threshold mu_0 the minimiser of the BP loss keeps the LP's support and
binding edges, and w_mu = w* - q / (2 mu) on the support, with
q = D_{A,S}^{-1} (y_A h_A^2). So mu_0 follows from the LP alone: the largest mu at
which an element reaches zero (q_k / 2 w*_k) or a free channel reaches an edge.
This script solves the LP at the saved validation pixels of experiments/mu_theory.py,
computes mu_0, and compares the law with the saved minimisers on either side of it.
It also reports the violations those minimisers have at mu = 100.

    .venv/bin/python experiments/law_threshold_check.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)

from experiments.train_scaled import load_operators  # noqa: E402
from experiments.mu_theory import lp_duals  # noqa: E402

T = 1.4
TOL = 1e-9
OUT = "results/plots/24_paper_20261031/law_threshold_check.json"


def threshold(D, x, lo, hi, obs, h):
    """(mu_0, element thresholds, S) for one LP solution, or None if degenerate."""
    y = lo - hi                       # positive at a binding lower edge
    S = np.flatnonzero(x > TOL)
    A = np.flatnonzero(np.abs(y) > TOL)
    if len(S) != len(A) or len(S) == 0:
        return None
    M = D[np.ix_(A, S)]
    if np.linalg.cond(M) > 1e12:
        return None
    q = np.linalg.solve(M, y[A] * h[A] ** 2)
    mu_el = np.where(q > 0, q / (2 * x[S]), 0.0)
    mu0 = mu_el.max()
    free = np.setdiff1d(np.arange(len(obs)), A)
    if len(free):
        drift = D[np.ix_(free, S)] @ q       # the free channels move by -drift / (2 mu)
        rec = D[free] @ x
        room_down = rec - (obs[free] - h[free])
        room_up = (obs[free] + h[free]) - rec
        need = np.where(drift > 0, drift / (2 * np.maximum(room_down, 1e-12)),
                        -drift / (2 * np.maximum(room_up, 1e-12)))
        mu0 = max(mu0, need.max())
    return mu0, mu_el, S, q, A, mu_el.max() >= 0.999 * mu0


def main():
    z = np.load("output/meeting_followup/bp_mu_theory_val.npz")
    D = load_operators("cpu")[0].numpy().astype(np.float64)
    mus, group, obs, err, status = (z["mus"].astype(float), z["group"], z["obs"], z["err"],
                                    z["lp_status"])
    viol, price_h, x_opt = z["viol"], z["price_halfwidth"], z["x_opt"]
    logt_el = np.tile(5.5 + 0.1 * np.arange(18), 3)
    hot = logt_el >= 6.6
    n = len(obs)
    mu0 = np.full(n, np.nan)
    hot_first = np.full(n, np.nan)          # largest threshold among hot elements
    n_edges = np.zeros(n, dtype=int)
    set_by_element = np.zeros(n, dtype=bool)
    paths = {}
    for i in np.flatnonzero(status == 0):
        st, _, x, lo, hi = lp_duals(D, obs[i], err[i], T)
        if st != 0:
            continue
        h = T * err[i]
        n_edges[i] = int((np.abs(lo - hi) > TOL).sum())
        r = threshold(D, x, lo, hi, obs[i], h)
        if r is None:
            continue
        mu0[i], mu_el, S, q, A, set_by_element[i] = r
        paths[i] = (x, S, q)
        hs = hot[S]
        hot_first[i] = mu_el[hs].max() if hs.any() else 0.0

    feas = status == 0
    ok = np.isfinite(mu0)
    core = group == group.max()
    out = {"n_pixels": int(n), "n_lp_solved": int(feas.sum()), "n_nondegenerate": int(ok.sum())}

    def pct(a, q=(10, 50, 90, 99)):
        return dict(zip((f"p{p}" for p in q), (float(v) for v in np.percentile(a, q))))

    bands = {"all": np.ones(n, bool), "below_cutoff": group <= 3, "bright_1_32": (group >= 4) & (group <= 7),
             "flare_cores": core}
    for g in range(group.max() + 1):
        bands[f"band_{g}"] = group == g
    out["mu0"] = {}
    for name, m in bands.items():
        s = m & ok
        out["mu0"][name] = {"n": int(s.sum()), "percentiles": pct(mu0[s]),
                            "share_le": {str(int(t)): float(np.mean(mu0[s] <= t)) for t in (1, 10, 30, 100, 300, 1000)}}

    # the law on either side of mu_0, per channel with a positive prediction
    rows = []
    for k, mu in enumerate(mus):
        vhat = price_h / mu
        for side, sel in (("above", ok & (mu >= 1.05 * mu0)), ("below", ok & (mu <= 0.5 * mu0))):
            v, p = viol[k][sel], vhat[sel]
            use = (p > 1e-4)
            if use.sum() == 0:
                continue
            ratio = v[use] / p[use]
            rows.append({"mu": float(mu), "side": side, "n": int(use.sum()),
                         "median": float(np.median(ratio)),
                         "within_5pct": float(np.mean(np.abs(ratio - 1) < 0.05))})
    out["law_vs_mu0"] = rows

    # hot plasma at flare cores: does the element leave where the LP says?
    s = core & ok
    out["hot_threshold_flare_cores"] = {"percentiles": pct(hot_first[s]),
                                        "share_above_1": float(np.mean(hot_first[s] > 1))}
    hot_w = lambda x: x[..., hot].sum(-1)
    k1, k100 = list(mus).index(1.0), list(mus).index(100.0)
    base = hot_w(x_opt[list(mus).index(1000.0)]).astype(np.float64)
    keep = hot_w(x_opt[k1]) / np.maximum(base, 1e-9)
    out["hot_weight_kept_at_mu1"] = {
        "hot_threshold_above_1": float(np.median(keep[s & (hot_first > 1)])),
        "hot_threshold_below_1": float(np.median(keep[s & (hot_first <= 1)]))}

    # what ends the law at flare cores: an element or a free channel, a hot element or not
    out["what_sets_mu0"] = {
        "flare_cores_element_share": float(np.mean(set_by_element[s])),
        "flare_cores_hot_element_first_share": float(np.mean(hot_first[s] >= 0.999 * mu0[s])),
        "flare_cores_mu0_above_1_share": float(np.mean(mu0[s] > 1)),
        "all_element_share": float(np.mean(set_by_element[ok]))}

    # pixel-channels above mu_0 that miss the law by 5%: is the closed form or L-BFGS-B better?
    def loss(w, o, h, mu):
        r = D @ w
        lo, hi = np.maximum(o - h - r, 0), np.maximum(r - o - h, 0)
        return w.sum() + mu * ((lo ** 2 + hi ** 2) / (h ** 2 + 1e-8)).sum()

    out["misses_above_mu0"] = {}
    for mu in (100.0, 1000.0):
        k = list(mus).index(mu)
        sel = ok & (mu >= 1.05 * mu0)
        vhat = price_h / mu
        use = (vhat > 1e-4) & sel[:, None]
        ratio = np.where(use, viol[k] / np.where(vhat > 0, vhat, 1), np.nan)
        miss = use & (np.abs(ratio - 1) >= 0.05)
        idx = np.flatnonzero(miss.any(1))
        better = 0
        for i in idx:
            x, S, q = paths[i]
            w = np.zeros(54)
            w[S] = x[S] - q / (2 * mu)
            h = T * err[i]
            better += loss(w, obs[i], h, mu) <= loss(x_opt[k][i].astype(np.float64), obs[i], h, mu)
        out["misses_above_mu0"][str(int(mu))] = {"pixel_channels": int(miss.sum()), "pixels": int(len(idx)),
                                                 "closed_form_at_or_below_saved_loss": int(better)}

    # measured violations of the minimiser at mu = 100, feasible flare cores
    m = core & feas
    w = viol[k100][m].max(1)
    out["max_violation_mu100_feasible_flare_cores"] = pct(w, (50, 90))
    out["predicted_max_mu100_feasible_flare_cores"] = pct(price_h[m].max(1) / 100, (50, 90))
    out["n_binding_edges_feasible_flare_cores"] = {str(v): int(c) for v, c in
                                                   zip(*np.unique(n_edges[m], return_counts=True))}
    # local slope of the median violation against mu
    med = [float(np.median(viol[k][m].max(1))) for k in range(len(mus))]
    out["median_violation_flare_cores_by_mu"] = dict(zip(map(str, mus), med))
    out["local_slope"] = [float(np.log(med[i + 1] / med[i]) / np.log(mus[i + 1] / mus[i]))
                          for i in range(len(mus) - 1)]
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
