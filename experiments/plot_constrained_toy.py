#!/usr/bin/env python3
"""Figures for the constrained-problem study (experiments/constrained_toy*.py).

    python3 experiments/plot_constrained_toy.py results/plots/26_constrained_theory_20261112

Reads toy.json (per-pixel linear programs and stand-ins) and any net_*.json (network
runs) in that folder and writes fig1..fig3 next to them.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 6.5, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def end_label(ax, x, y, text, color, dx=6, dy=0, ha="left"):
    """Direct label at a line's end; the text stays in ink, a dot carries the series colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=9, color=INK)


def fig_infeasible(r, out):
    """How far from feasible are the pixels the solver cannot fit at the tight band?"""
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    for key, name, color in (("uniform_train", "training pixels", BLUE), ("uniform_val", "validation pixels", ORANGE)):
        q = r[key]["infeasible"]["pct_feasible_within"]
        xs = [float(k[:-1]) for k in q]
        ax.plot(xs, list(q.values()), "-o", color=color, label=f"{name} ({r[key]['infeasible']['n']:,} infeasible)",
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    for mult, text in ((3, "solver's first retry (3x)"), (5, "second retry (5x)")):
        ax.axvline(mult, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(text, (mult, 62), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK2)
    q = r["uniform_train"]["infeasible"]
    med = 1 + q["band_multiple_needed_minus_1"]["p50"]
    ax.annotate(f"half are feasible once the\nband is {med:.2f}x as wide", (med, 50), xytext=(1.36, 31),
                fontsize=9, color=INK, va="center", arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    ax.set_xscale("log")
    ax.set_xticks([1.05, 1.25, 1.5, 2, 3, 5])
    ax.set_xticklabels(["1.05x", "1.25x", "1.5x", "2x", "3x", "5x"])
    ax.minorticks_off()
    ax.set_ylim(0, 103)
    ax.set_xlabel("band width, as a multiple of 1.4 sigma")
    ax.set_ylabel("infeasible pixels that become feasible (%)")
    ax.set_title("Pixels with no DEM inside the bands miss by a little")
    ax.legend(loc="lower left", bbox_to_anchor=(0.27, 0.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_standins(r, out):
    """The exact penalty and the method of multipliers at their per-pixel optimum."""
    fw = r["forms"]["feasible_weighted"]
    core = r["forms"]["bands"][">=32x"]["feasible"]
    tol = "0.001"
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.1), dpi=200, sharey=True)
    ks = [int(k) for k in fw["exact_penalty"]]
    y_all = [fw["exact_penalty"][str(k)][tol] for k in ks]
    y_core = [core["exact_penalty"][str(k)]["inside_pct"][tol] for k in ks]
    a.plot(ks, y_all, "-o", color=BLUE, label="all feasible pixels", markeredgecolor=SURFACE, markeredgewidth=1.5)
    a.plot(ks, y_core, "-o", color=ORANGE, label="flare cores (>= 32x Bright)", markeredgecolor=SURFACE, markeredgewidth=1.5)
    a.set_xscale("log", base=2)
    a.set_xticks(ks)
    a.set_xticklabels([str(k) for k in ks])
    a.minorticks_off()
    a.set_xlabel("price charged, as a multiple of the single-channel price")
    a.set_ylabel("feasible pixels inside every band (%)")
    a.set_title("Exact penalty: the price needed spans three decades")
    a.legend(loc="lower right", fontsize=9)
    its = list(range(1, len(fw["augmented_lagrangian"]["mu100"]) + 1))
    series = (("mu100", "weight 100", BLUE), ("mu10", "weight 10", AQUA), ("ada10", "weight 10, raised when stuck", ORANGE))
    for key, name, color in series:
        ys = [v[tol] for v in fw["augmented_lagrangian"][key]]
        b.plot(its, ys, "-o", color=color, label=name, markeredgecolor=SURFACE, markeredgewidth=1.5)
    yc = [it["inside_pct"][tol] for it in core["augmented_lagrangian"]["ada10"]["iterations"]]
    b.plot(its, yc, "--o", color=ORANGE, linewidth=1.4, label="flare cores, weight 10 raised when stuck",
           markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.set_xticks(its)
    b.set_xticklabels(["1\n(the loss\nused today)"] + [str(i) for i in its[1:]])
    b.set_xlabel("multiplier updates (outer iterations)")
    b.set_title("Method of multipliers: exact after a few updates")
    b.legend(loc="lower right", fontsize=9)
    a.set_ylim(0, 103)
    fig.suptitle("Share of feasible pixels whose per-pixel optimum sits inside all six bands (within 0.001 half-widths)",
                 x=0.012, y=0.985, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out)
    plt.close(fig)


NET_ORDER = (("quad100", "fixed weight 100 (today's loss)", BLUE), ("ramp", "weight raised during training", ORANGE),
             ("al", "multipliers, stored per pixel", AQUA), ("alnet", "multipliers from a second network", YELLOW),
             ("huber", "smoothed exact penalty", MAGENTA), ("price", "weight from noise and price scale", GREEN),
             ("alfix", "multipliers per pixel, weight held at 100", "#6250d6"))


def fig_training(runs, out, title, key="train"):
    """Band violations through training, one line per stand-in."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=200)
    panels = ((lambda h: h[key]["pixels_outside_pct"]["0.1"], "pixels outside a band by > 0.1 half-widths (%)", "log"),
              (lambda h: h[key]["violation_mean"], "mean violation (half-widths)", "log"),
              (lambda h: h["hold_feasible"]["weight_over_lp_median"], "total weight / solver's optimum (median held-out pixel)", "linear"))
    for ax, (get, ylabel, scale) in zip(axes, panels):
        for name, label, color in NET_ORDER:
            if name not in runs:
                continue
            hist = runs[name]["history"]
            xs = [h["epoch"] for h in hist]
            ys = [get(h) for h in hist]
            if "before_training" in runs[name] and key in runs[name]["before_training"]:
                xs, ys = [0] + xs, [get({key: runs[name]["before_training"][key],
                                         "hold_feasible": runs[name]["before_training"]["hold_feasible"]})] + ys
            ax.plot(xs, ys, color=color, label=label)
        ax.set_yscale(scale)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
    axes[2].axhline(1.0, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ramp = next(iter(runs.values()))["ramp_epochs"]
    for ax in axes:
        ax.axvline(ramp, color=GRID, linewidth=1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=9, frameon=False)
    fig.suptitle(title, x=0.012, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11 if len(labels) > 4 else 0.07, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def fig_tradeoff(runs, r, out, title):
    """Held-out band violations against total weight, epoch by epoch: the solver sits at
    (1, 0); a network can only buy fewer violations with more weight."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    finals = []
    for name, label, color in NET_ORDER:
        if name not in runs:
            continue
        run = runs[name]
        warm = bool(run["args"].get("init_from"))
        pts = []
        if warm and "before_training" in run:
            b = run["before_training"]["hold_feasible"]
            pts.append((b["weight_over_lp_median"], b["pixels_outside_pct"]["0.01"]))
        hist = run["history"] if warm else run["history"][len(run["history"]) // 2:]
        pts += [(h["hold_feasible"]["weight_over_lp_median"], h["hold_feasible"]["pixels_outside_pct"]["0.01"])
                for h in hist]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.45)
        f = run["final"]["hold_feasible"]
        ax.plot([f["weight_over_lp_median"]], [f["pixels_outside_pct"]["0.01"]], "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        finals.append((f["weight_over_lp_median"], f["pixels_outside_pct"]["0.01"]))
        finals.extend(pts[:1] if warm else [])
    start = next((run["before_training"]["hold_feasible"] for run in runs.values() if "before_training" in run), None)
    if start is not None and any("init_from" in run["args"] and run["args"]["init_from"] for run in runs.values()):
        ax.plot([start["weight_over_lp_median"]], [start["pixels_outside_pct"]["0.01"]], "s", color=INK2, markersize=7)
        ax.annotate("current final model", (start["weight_over_lp_median"], start["pixels_outside_pct"]["0.01"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.plot([1.0], [0.0], "*", color=INK, markersize=13, clip_on=False, zorder=5)
    ax.annotate("the solver", (1.0, 0.0), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK)
    q = 100 - r["forms"]["feasible_weighted"]["augmented_lagrangian"]["mu100"][0]["0.01"]
    fx, fy = zip(*finals)
    ax.set_xlim(min(0.85, min(fx) - 0.1), max(fx) * 1.15 + 0.15)
    ax.set_ylim(0, max(max(fy), q) * 1.25 + 1)
    ax.axhline(q, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(f"today's loss solved exactly per pixel: {q:.1f}% outside", (ax.get_xlim()[1], q), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("total weight / solver's optimum (median held-out pixel)")
    ax.set_ylabel("held-out feasible pixels outside a band by > 0.01 half-widths (%)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12 if len(labels) > 4 else 0.08, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def table(groups):
    """One line per run: where it ended on training and held-out feasible pixels."""
    head = (f"{'run':22s} {'epochs':>6s} | {'train outside >0.1 / 0.01 / 0.001 (%)':>38s} | "
            f"{'held-out outside >0.1 / 0.01':>28s} | {'viol p99':>8s} | {'weight/LP pooled, median':>24s} | "
            f"{'DEM MAE vs LP':>13s} | {'core peak/LP':>12s} | {'zeroed ch. above':>16s}")
    lines = [head, "-" * len(head)]
    for suffix, runs in sorted(groups.items()):
        for name, _, _ in NET_ORDER:
            if name not in runs:
                continue
            r = runs[name]
            tr, ho = r["final"]["train"], r["final"]["hold_feasible"]
            core = r["final"]["balanced_bands"][">=32x"]["feasible"] or {}
            zc = ho.get("zeroed_channels", {}).get("above_band_pct", {}).get("0.01")
            tag = name + (f"_{suffix}" if suffix else "")
            lines.append(
                f"{tag:22s} {len(r['history']):6d} | "
                f"{tr['pixels_outside_pct']['0.1']:11.2f} / {tr['pixels_outside_pct']['0.01']:6.2f} / "
                f"{tr['pixels_outside_pct']['0.001']:6.2f}      | "
                f"{ho['pixels_outside_pct']['0.1']:13.2f} / {ho['pixels_outside_pct']['0.01']:6.2f}     | "
                f"{tr['violation']['p99']:8.3g} | {ho.get('weight_over_lp_pooled', float('nan')):12.3f}, {ho.get('weight_over_lp_median', float('nan')):9.3f} | "
                f"{ho.get('dem_mae_vs_lp', float('nan')):13.4g} | {core.get('peak_over_lp_median', float('nan')):12.3f} | "
                f"{'-' if zc is None else format(zc, '.2f') + '%':>16s}")
            mv = r["final"].get("multipliers_vs_dual_prices")
            if mv:
                q = mv["ratio_learned_over_true"]
                lines.append(f"{'':22s}   multipliers vs the solver's dual prices ({mv['pixels']}): learned/true p10 {q['p10']:.2f}, "
                             f"p50 {q['p50']:.2f}, p90 {q['p90']:.2f}; within 2x {mv['within_factor_2_pct']:.0f}%; "
                             f"log correlation {mv['log_correlation']:.2f}")
    return "\n".join(lines)


def main():
    folder = sys.argv[1]
    with open(os.path.join(folder, "toy.json")) as fh:
        r = json.load(fh)
    fig_infeasible(r, os.path.join(folder, "fig1_infeasible_band_multiple.png"))
    fig_standins(r, os.path.join(folder, "fig2_standins_per_pixel.png"))
    groups = {}
    for path in sorted(glob.glob(os.path.join(folder, "net_*.json"))):
        name = os.path.basename(path)[4:-5]
        base, _, suffix = name.partition("_")
        with open(path) as fh:
            groups.setdefault(suffix, {})[base] = json.load(fh)
    titles = {"": "Trained from scratch on feasible pixels: band violations through training",
              "warm": "Starting from the current model, feasible pixels: band violations through training",
              "warmall": "Starting from the current model, every pixel: band violations through training"}
    for i, (suffix, runs) in enumerate(sorted(groups.items())):
        fig_training(runs, os.path.join(folder, f"fig3_training_violations{'_' + suffix if suffix else ''}.png"),
                     titles.get(suffix, suffix))
        fig_tradeoff(runs, r, os.path.join(folder, f"fig4_violation_vs_weight{'_' + suffix if suffix else ''}.png"),
                     "Fewer band violations cost extra weight" + (" (from the current model)" if suffix == "warm" else
                                                                  " (trained from scratch)" if not suffix else f" ({suffix})"))
    if groups:
        text = table(groups)
        print(text)
        with open(os.path.join(folder, "net_summary.txt"), "w") as fh:
            fh.write(text + "\n")
    print("figures written to", folder)


if __name__ == "__main__":
    main()
