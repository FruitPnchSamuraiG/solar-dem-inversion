#!/usr/bin/env python3
"""The method of multipliers on single pixels, no network: figures of the method at work.

For a few pixels of a toy set (default: the hard set's flare cores and bright pixels whose LP is
feasible), run the exact per-pixel method of multipliers on BP's problem:

    w_k = argmin_{w >= 0}  sum(w) + sum_edges [relu(y + 2 mu g(w))^2 - y^2] / (4 mu)
    y  <- relu(y + 2 mu g(w_k))

g is the distance outside each of the 12 band edges in half-widths. Each round records the
multipliers, the gaps, the DEM and the total weight, for mu in --mus. The LP's own optimum and dual
prices (from the toy set) are the reference: the multipliers should reach the dual prices and the
DEM the LP's DEM in a few rounds, at any mu (mu sets the speed).

    python3 experiments/al_trajectory.py --data output/experiments/constrained/toy_hard_train.npz --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.constrained_toy import solve  # noqa: E402

CHANNELS = (94, 131, 171, 193, 211, 335)
EDGE_NAMES = [f"{c} low" for c in CHANNELS] + [f"{c} high" for c in CHANNELS]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--n_core", type=int, default=3)
    p.add_argument("--n_bright", type=int, default=3)
    p.add_argument("--mus", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", required=True)
    return p.parse_args()


def run_pixel(D, B, o, h, mu, rounds):
    A = D / h[:, None]
    lo, hi = o / h - 1.0, o / h + 1.0
    y = np.zeros(12)
    w = np.zeros(D.shape[1])
    recs = []
    for k in range(rounds):
        def f(x, y=y):
            r = A @ x
            g = np.concatenate([lo - r, r - hi])
            q = np.maximum(y + 2 * mu * g, 0.0)
            val = x.sum() + ((q ** 2 - y ** 2) / (4 * mu)).sum()
            grad = 1.0 + A.T @ (q[6:] - q[:6])
            return val, grad
        w, _ = solve(f, w)
        r = A @ w
        g = np.concatenate([lo - r, r - hi])
        y_new = np.maximum(y + 2 * mu * g, 0.0)
        recs.append({"round": k, "multipliers_used": y.tolist(), "gap": g.tolist(), "max_gap": float(max(g.max(), 0.0)),
                     "weight": float(w.sum()), "dem": (B @ w).tolist()})
        y = y_new
    return recs


def main():
    args = parse_args()
    import torch
    from experiments.train_scaled import load_operators
    D_t, B_t, _, logT = load_operators(torch.device("cpu"))
    D, B = D_t.numpy().astype(np.float64), B_t.numpy().astype(np.float64)[:18]
    z = np.load(args.data)
    feas = z["lp_status"] == 0
    rng = np.random.default_rng(args.seed)
    pick = []
    for sel, n in ((feas & (z["group"] == 8), args.n_core), (feas & (z["group"] >= 6) & (z["group"] < 8), args.n_bright)):
        idx = np.flatnonzero(sel)
        pick += list(rng.choice(idx, size=min(n, idx.size), replace=False))
    os.makedirs(args.out, exist_ok=True)
    out = []
    for i in pick:
        o, h = z["obs"][i].astype(np.float64), args.tolfac * z["err"][i].astype(np.float64)
        item = {"index": int(i), "group": int(z["group"][i]), "inten": float(z["inten"][i]),
                "lp_objective": float(z["lp_obj"][i]), "lp_dem": z["lp_dem_lp"][i].tolist(),
                "dual_prices": np.concatenate([z["lp_y_lo"][i], z["lp_y_hi"][i]]).tolist(), "runs": {}}
        for mu in args.mus:
            item["runs"][f"{mu:g}"] = run_pixel(D, B, o, h, mu, args.rounds)
        out.append(item)
        print(f"pixel {i} ({item['inten']:.0f}x Bright): max gap by round at mu=10: "
              + " ".join(f"{r['max_gap']:.2g}" for r in item["runs"].get("10", [])), flush=True)
    json.dump({"settings": vars(args), "edges": EDGE_NAMES, "logT": list(map(float, logT[:18])), "pixels": out},
              open(os.path.join(args.out, "al_trajectory.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lg = np.asarray(logT[:18], np.float64)
    for it in out:
        run = it["runs"]["10"] if "10" in it["runs"] else next(iter(it["runs"].values()))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        cm = plt.cm.viridis(np.linspace(0, 0.9, len(run)))
        for r, c in zip(run, cm):
            ax[0].plot(lg, r["dem"], color=c, lw=1, alpha=0.8)
        ax[0].plot(lg, it["lp_dem"], "k--", lw=2, label="BP's problem, exact (LP)")
        ax[0].set(xlabel="logT", ylabel="DEM", title="DEM by round (dark to light), mu 10")
        ax[0].legend(fontsize=8, frameon=False)
        for mu, rr in it["runs"].items():
            ax[1].semilogy([r["round"] for r in rr], [max(r["max_gap"], 1e-6) for r in rr], "o-", ms=3, label=f"mu {mu}")
        ax[1].set(xlabel="round", ylabel="worst distance outside the band (half-widths)", title="band miss by round")
        ax[1].legend(fontsize=8, frameon=False)
        dual = np.asarray(it["dual_prices"])
        Y = np.array([r["multipliers_used"] for r in run])
        active = np.flatnonzero(dual > 1e-6 * max(dual.max(), 1e-12))
        for e in active:
            line, = ax[2].plot(range(len(run)), Y[:, e], "o-", ms=3, label=EDGE_NAMES[e])
            ax[2].axhline(dual[e], color=line.get_color(), ls="--", lw=1)
        ax[2].set(xlabel="round", ylabel="multiplier (fine per half-width)", title="fines (solid) vs BP's dual prices (dashed), mu 10")
        ax[2].legend(fontsize=7, frameon=False)
        fig.suptitle(f"pixel {it['index']} ({it['inten']:.0f}x Bright)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"al_pixel_{it['index']}.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
