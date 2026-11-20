#!/usr/bin/env python3
"""Figures and a table for the hard-pixel toy (experiments/submit_hard_toy.sh).

Reads net_hard_<run>.json from --dir and writes to --out:
  compare_<axis>.png   one setting changed at a time against the reference run (mu, learning
                       rate, round length): training pixels outside their band, worst-1% band miss,
                       weight over the LP's, DEM error against the LP's DEM, over epochs
  pixel_<i>.png        the reference run at one tracked pixel: its DEM at a few epochs against the
                       LP's, the worst band miss by epoch, and the fines by epoch against BP's dual
                       prices (the LP's exact multipliers)
  pixels_ref.png       four tracked pixels side by side (band miss and fines), for the paper
  summary.txt          final numbers per run

    python3 experiments/plot_hard_toy.py --dir output/experiments/constrained --out results/plots/30_hard_toy_20261120
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = ("ref", "mu1", "mu100", "nofines", "lr1e-4", "lr1e-3", "round5", "round50", "ref_long", "lr1e-3_long",
        "bpflags", "tight", "trunc1", "trunc3", "trunc10")
SET = "hard"
LABEL = {"ref": "reference (mu 10, lr 3e-4, round 20 ep)", "mu1": "mu 1", "mu100": "mu 100",
         "nofines": "no fines (penalty mu 10 only)", "lr1e-4": "lr 1e-4", "lr1e-3": "lr 1e-3",
         "round5": "round every 5 epochs", "round50": "round every 50 epochs",
         "ref_long": "reference, 3000 epochs", "lr1e-3_long": "lr 1e-3, 3000 epochs",
         "bpflags": "BP's flags (final model's way)", "tight": "no flags, tight band everywhere",
         "trunc1": "no flags, penalty capped at 1", "trunc3": "no flags, capped at 3", "trunc10": "no flags, capped at 10"}
COLOR = {"ref": "black", "mu1": "tab:blue", "mu100": "tab:orange", "nofines": "tab:red",
         "lr1e-4": "tab:blue", "lr1e-3": "tab:orange", "round5": "tab:blue", "round50": "tab:orange",
         "ref_long": "tab:green", "lr1e-3_long": "tab:purple"}
AXES = {"mu": ("ref", "mu1", "mu100", "nofines"), "lr": ("ref", "lr1e-4", "lr1e-3"),
        "length": ("ref", "ref_long", "lr1e-3", "lr1e-3_long"),
        "round": ("ref", "round5", "round50")}
CHANNELS = (94, 131, 171, 193, 211, 335)
EDGES = [f"{c} low" for c in CHANNELS] + [f"{c} high" for c in CHANNELS]
GROUP = {6: "3.2-10x Bright", 7: "10-32x Bright", 8: "flare core"}


def load(d):
    out = {}
    for r in RUNS:
        p = os.path.join(d, f"net_{SET}_{r}.json")
        if os.path.exists(p):
            out[r] = json.load(open(p))
    return out


def series(run, key, sub):
    h = [x for x in run["history"] if key in x]
    return [x["epoch"] for x in h], [sub(x[key]) for x in h]


METRICS = (("pixels outside band (> 0.01 half-width), %", lambda s: s["pixels_outside_pct"]["0.01"], False),
           ("worst-1% band miss (half-widths)", lambda s: s["violation"]["p99"], True),
           ("total weight / LP's", lambda s: s["weight_over_lp_pooled"], False),
           ("DEM error vs LP's DEM (MAE)", lambda s: s["dem_mae_vs_lp"], True))


def compare(runs, axis, out):
    names = [r for r in AXES[axis] if r in runs]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (title, f, log) in zip(axs, METRICS):
        for r in names:
            x, y = series(runs[r], "train", f)
            ax.plot(x, y, color=COLOR[r], lw=1.6 if r == "ref" else 1.2, label=LABEL[r])
        ax.set(xlabel="epoch", title=title)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"hard-pixel toy, training pixels: changing the {axis}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"compare_{axis}.png"), dpi=120)
    plt.close(fig)


def pixel_arrays(run, k):
    t = run["tracked"]
    rec = t["records"]
    ep = np.array([r["epoch"] for r in rec])
    gap = np.array([r["gap"][k] for r in rec])
    Y = np.array([r["multipliers"][k] if r["multipliers"] is not None else [0.0] * 12 for r in rec])
    dem = np.array([r["dem"][k] for r in rec])
    return ep, gap, Y, dem


def draw_fines(ax, ep, Y, dual):
    active = np.flatnonzero(dual > 1e-3 * max(dual.max(), 1e-12))
    for e in active:
        line, = ax.plot(ep, Y[:, e], lw=1.3, label=EDGES[e])
        ax.axhline(dual[e], color=line.get_color(), ls="--", lw=1)
    ax.set(xlabel="epoch", ylabel="fine per half-width")
    ax.legend(fontsize=6, frameon=False)


def pixel_figs(run, logT, out):
    t = run["tracked"]
    for k, idx in enumerate(t["index"]):
        ep, gap, Y, dem = pixel_arrays(run, k)
        dual = np.asarray(t["dual_prices"][k])
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        snaps = [e for e in (0, 20, 50, 100, 200, len(ep) - 1) if e < len(ep)]
        cm = plt.cm.viridis(np.linspace(0, 0.9, len(snaps)))
        for s, c in zip(snaps, cm):
            ax[0].plot(logT, dem[s], color=c, lw=1, label=f"epoch {ep[s]}")
        ax[0].plot(logT, t["dem_lp"][k], "k--", lw=2, label="LP (BP's problem, exact)")
        ax[0].set(xlabel="logT", ylabel="DEM", title="network's DEM over training")
        ax[0].legend(fontsize=6, frameon=False)
        ax[1].semilogy(ep, np.maximum(gap.max(1), 1e-4), color="black", lw=1.3)
        ax[1].axhline(0.01, color="0.6", ls=":", lw=1)
        ax[1].set(xlabel="epoch", ylabel="worst band miss (half-widths, floor 1e-4)", title="band miss")
        draw_fines(ax[2], ep, Y, dual)
        ax[2].set_title("fines (solid) vs BP's dual prices (dashed)")
        fig.suptitle(f"reference run, training pixel {idx} ({GROUP.get(t['group'][k], t['group'][k])})", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"pixel_{idx}.png"), dpi=110)
        plt.close(fig)


def paper_fig(run, out, n=4):
    t = run["tracked"]
    order = sorted(range(len(t["index"])), key=lambda k: -t["group"][k])[:n]
    fig, axs = plt.subplots(2, len(order), figsize=(4 * len(order), 6.2), sharex=True)
    for col, k in enumerate(order):
        ep, gap, Y, _ = pixel_arrays(run, k)
        axs[0, col].semilogy(ep, np.maximum(gap.max(1), 1e-4), color="black", lw=1.3)
        axs[0, col].axhline(0.01, color="0.6", ls=":", lw=1)
        axs[0, col].set_title(f"pixel {t['index'][k]} ({GROUP.get(t['group'][k], '')})", fontsize=9)
        draw_fines(axs[1, col], ep, Y, np.asarray(t["dual_prices"][k]))
    axs[0, 0].set_ylabel("worst band miss (half-widths)")
    fig.suptitle("one network trained on hard pixels: worst band miss (top) and fines by epoch (bottom; dashed: "
                 "BP's dual prices, the per-pixel answer)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pixels_ref.png"), dpi=120)
    plt.close(fig)


def summary(runs, out):
    L = ["run | train: outside >0.01 % / p99 miss / weight vs LP / DEM MAE vs LP | held-out bright: outside % / "
         "weight vs LP / DEM MAE | flare cores (balanced): outside % / peak vs LP (median) | minutes"]
    for r, run in runs.items():
        f = run["final"]
        tr, ho = f["train"], f["hold_feasible"]
        co = f["balanced_bands"][list(f["balanced_bands"])[8]]["feasible"]
        L.append(f"{r:8s} | {tr['pixels_outside_pct']['0.01']:6.2f} / {tr['violation']['p99']:.3g} / "
                 f"{tr['weight_over_lp_pooled']:.3f} / {tr['dem_mae_vs_lp']:.3g} | "
                 f"{ho['pixels_outside_pct']['0.01']:6.2f} / {ho['weight_over_lp_pooled']:.3f} / {ho['dem_mae_vs_lp']:.3g} | "
                 f"{co['pixels_outside_pct']['0.01']:6.2f} / {co.get('peak_over_lp_median', float('nan')):.3f} | "
                 f"{run['runtime_s'] / 60:.0f}")
    flagged = [r for r, run in runs.items() if "by_flag_train" in run["final"]]
    if any(runs[r]["args"].get("pixels") == "all" for r in flagged):
        L.append("\nby BP's flag, training pixels (miss in TIGHT half-widths; BP's band = 3x/5x where BP widened):")
        L.append("run | class: n / inside tight % / inside BP's band % / worst miss p50, p90 / DEM MAE vs LP / vs BP")
        for r in flagged:
            for cls, o in runs[r]["final"]["by_flag_train"].items():
                L.append(f"{r:8s} | {cls:8s}: {o['n']:6d} / {o['inside_tight_pct']:6.2f} / {o['inside_bp_band_pct']:6.2f} / "
                         f"{o['worst_miss_p50']:.3g}, {o['worst_miss_p90']:.3g} / {o.get('dem_mae_vs_lp', float('nan')):.3g} / "
                         f"{o.get('dem_mae_vs_bp', float('nan')):.3g}")
    text = "\n".join(L)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", default="hard", help="hard / mix / mixall")
    args = p.parse_args()
    global SET
    SET = args.set
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.dir)
    AXES["flags"] = ("bpflags", "tight", "trunc1", "trunc3", "trunc10")
    COLOR.update({"bpflags": "black", "tight": "tab:red", "trunc1": "tab:blue", "trunc3": "tab:orange", "trunc10": "tab:green"})
    for axis in AXES:
        if any(r in runs for r in AXES[axis]):
            compare(runs, axis, args.out)
    if "ref" in runs and runs["ref"].get("tracked"):
        logT = np.linspace(5.5, 7.2, 18)
        pixel_figs(runs["ref"], logT, args.out)
        paper_fig(runs["ref"], args.out)
    summary(runs, args.out)


if __name__ == "__main__":
    main()
