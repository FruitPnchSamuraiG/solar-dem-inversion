#!/usr/bin/env python3
"""How stiff should the BP band be? Per-pixel best answers vs trained networks.

Reads experiments/objective_gap.py outputs on the validation split: the
per-pixel optimum of the BP loss at each band stiffness (mu), the trained
networks, the supervised model and the solver. Four panels against stiffness:
overall DEM error (bands weighted by their pixel counts), Quiet DEM error,
flare-core emission over the solver's, and flare-core 94 A reconstruction over
the observation. Blue line = best possible answer under each loss (no
training); orange dots = trained networks at the stiffness they were trained
with; grey lines = supervised model and solver.

    python3 experiments/plot_stiffness.py --sweep A.json [B.json ...] \
        --nets nets.json --net-mu final=1 mu100=100 --out fig.png
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
QUIET = ("<0.03x", "0.03-0.1x", "0.1-0.3x", "0.3-1x")


def metrics(r, name):
    pop = r["population_per_band"]
    b = r["bands"]
    tot = sum(pop.values())
    overall = sum(pop[k] * b[k][name]["dem_mse_vs_solver"] for k in pop) / tot
    quiet = sum(pop[k] * b[k][name]["dem_mse_vs_solver"] for k in QUIET) / sum(pop[k] for k in QUIET)
    core = b[">=32x"][name]
    return {"overall": overall, "quiet": quiet, "core_em": core["em_ratio_vs_solver"],
            "core_94": core["recon_over_obs"]["94"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", nargs="+", required=True)
    p.add_argument("--nets", default=None, help="objective_gap JSON scoring trained networks")
    p.add_argument("--net-mu", nargs="*", default=[], help="label=mu for trained networks to plot")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    curve, ref = {}, {}
    for path in args.sweep:
        r = json.load(open(path))
        for name in r["bands"][">=32x"]:
            m = re.fullmatch(r"optimum(?:_mu([\d.]+))?", name)
            if m:
                curve[float(m.group(1) or 1)] = metrics(r, name)
            elif name in ("supervised", "solver"):
                ref[name] = metrics(r, name)
    mus = sorted(curve)
    nets = []
    if args.nets:
        r = json.load(open(args.nets))
        for spec in args.net_mu:
            label, mu = spec.split("=")
            nets.append((label, float(mu), metrics(r, f"net_{label}")))
        for name in ("supervised", "solver"):
            ref.setdefault(name, metrics(r, name))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    panels = [("overall", "DEM error, all pixels", True, None),
              ("quiet", "DEM error, Quiet pixels", False, None),
              ("core_em", "Flare-core emission / solver's", False, (0, 1.12)),
              ("core_94", "Flare-core 94 Å / observed", False, (0, 1.12))]
    for ax, (key, title, logy, ylim) in zip(axes.ravel(), panels):
        ys = [curve[m][key] for m in mus]
        ax.plot(mus, ys, color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter(mus, ys, s=36, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
        for label, mu, m in nets:
            ax.scatter([mu], [m[key]], s=64, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=5)
        # Reference lines are labelled at the left, where the curve sits far
        # from them in every panel; on 94 A the solver and supervised lines are
        # close, so one label goes above its line and the other below.
        sup = ref.get("supervised", {}).get(key)
        if sup is not None:
            ax.axhline(sup, color=MUTED, lw=1, zorder=2)
            below = key == "core_94"
            right = key == "quiet"          # there the curve crosses the line on the left
            ax.annotate("supervised model", xy=(0.97 if right else 0.03, sup),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, -3 if below else 3), textcoords="offset points",
                        ha="right" if right else "left",
                        va="top" if below else "bottom", fontsize=8.5, color=INK2)
        if key == "core_94" and "solver" in ref:
            ax.axhline(ref["solver"][key], color=BASE, lw=1, zorder=2)
            ax.annotate("solver", xy=(0.03, ref["solver"][key]), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                        fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
            ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100)
                     if min(ys) / 1.5 <= t <= max(ys) * 1.5]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
        ax.grid(True, which="major", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([1, 3, 10, 30, 100, 300, 1000])
        ax.set_xticklabels(["1", "3", "10", "30", "100", "300", "1000"])
    for ax in axes[1]:
        ax.set_xlabel("band stiffness (barrier weight, today = 1)")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                          label="best possible answer under that loss (no training)")]
    if nets:
        handles.append(plt.Line2D([], [], color=ORANGE, lw=0, marker="o", ms=8, mec=SURFACE, mew=1.5,
                                  label="trained network"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
