#!/usr/bin/env python3
"""Figures for results/bright_failure_story_20260929.md.

Reads the four diagnose_bright_failures.py outputs committed in
results/plots/13_bright_diagnostic_20260928/ and writes three PNGs beside them.
No server access needed.

    uv run python experiments/plot_bright_failures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/plots/13_bright_diagnostic_20260928")
LF, SUP, PIX = "#2a78d6", "#eb6834", "#a3a19a"      # validated pair + neutral
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def load(track, supervised=False):
    name = f"{track}_bright_failure{'_supervised' if supervised else ''}.json"
    with open(os.path.join(DIR, name)) as f:
        return json.load(f)


def merged(rows, keys):
    """Pixel share, error share and relative error of several bands combined."""
    n = sum(rows[k]["n_pixels"] for k in keys)
    sse = sum(rows[k]["dem_mse"] * rows[k]["n_pixels"] * 18 for k in keys)
    ref = sum(rows[k]["dem_mse"] * rows[k]["n_pixels"] * 18 / rows[k]["relative_sse"] for k in keys)
    return {"pix": sum(rows[k]["pct_of_pixels"] for k in keys),
            "err": sum(rows[k]["sse_share_pct"] for k in keys), "rel": sse / ref}


BUCKETS = [("Quiet\n(below 1x)", ["<0.03x", "0.03-0.1x", "0.1-0.3x", "0.3-1x"]),
           ("Ordinary Bright\n(1x to 10x)", ["1-1.8x", "1.8-3.2x", "3.2-10x"]),
           ("Very bright\n(10x to 32x)", ["10-32x"]),
           ("Flare core\n(32x and above)", [">=32x"])]


def fig_error_share():
    lf = load("bp")["by_intensity_over_threshold"]
    sp = load("bp", True)["by_intensity_over_threshold"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.grid(axis="y", visible=False)
    y = np.arange(len(BUCKETS))[::-1]
    h = 0.24
    series = [("Share of pixels", PIX, lambda k: merged(lf, k)["pix"]),
              ("Share of label-free error", LF, lambda k: merged(lf, k)["err"]),
              ("Share of supervised error", SUP, lambda k: merged(sp, k)["err"])]
    for i, (label, color, value) in enumerate(series):
        vals = [value(keys) for _, keys in BUCKETS]
        pos = y + (1 - i) * (h + 0.03)
        ax.barh(pos, vals, height=h, color=color, label=label, edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(v * 1.15, p, f"{v:.2g}%" if v < 1 else f"{v:.1f}%", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.004, 400)
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_yticks(y)
    ax.set_yticklabels([b for b, _ in BUCKETS])
    ax.set_xlabel("Percent (log scale)")
    fig.suptitle("BP track: one pixel in 10,000 holds about 90% of the DEM error",
                 x=0.01, ha="left", fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.01, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(DIR, "fig1_error_share_by_brightness.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


BANDS = ["<0.03x", "0.03-0.1x", "0.1-0.3x", "0.3-1x", "1-1.8x", "1.8-3.2x", "3.2-10x", "10-32x", ">=32x"]
BAND_TICKS = ["<0.03", "0.03-0.1", "0.1-0.3", "0.3-1", "1-1.8", "1.8-3.2", "3.2-10", "10-32", "32+"]


def fig_relative_error():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    x = np.arange(len(BANDS))
    for ax, track, title in zip(axes, ("bp", "enet"), ("BP track", "ENet track")):
        for data, color, name in ((load(track), LF, "Label-free"), (load(track, True), SUP, "Supervised")):
            rows = data["by_intensity_over_threshold"]
            vals = [rows[b]["relative_sse"] for b in BANDS]
            ax.plot(x, vals, color=color, lw=2, marker="o", ms=5.5, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.text(x[-1] + 0.2, vals[-1], name, color=INK2, fontsize=8.5, va="center")
        ax.axvline(3.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(3.6, 1.07, "Bright cutoff", color=INK2, fontsize=8.5, va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(BAND_TICKS, rotation=35, ha="right")
        ax.set_xlim(-0.4, len(BANDS) + 0.9)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Brightness: largest channel / its cutoff")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Relative DEM error\n(error size / DEM size, squared)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Supervised improves as pixels brighten; label-free turns worse above ~3x",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(DIR, "fig2_relative_error_by_brightness.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def peak_centres(rows):
    """Geometric centre of each reference-peak bin, in bin order, skipping <=0."""
    edges = np.round(np.arange(-3.0, 3.51, 0.5), 2)
    centres = [10 ** (edges[0] - 0.25)] + [10 ** (a + 0.25) for a in edges[:-1]] + [10 ** (edges[-1] + 0.25)]
    keys = [k for k in rows if k != "<=0"]
    return keys, centres[:len(keys)]


def fig_peak_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, track, title in zip(axes, ("bp", "enet"), ("BP track", "ENet track")):
        for data, color, name in ((load(track), LF, "Label-free"), (load(track, True), SUP, "Supervised")):
            rows = data["calibration_by_reference_peak"]
            keys, xs = peak_centres(rows)
            pts = [(x, rows[k]["peak_ratio_pred_over_ref"]) for k, x in zip(keys, xs)
                   if rows[k] and rows[k].get("peak_ratio_pred_over_ref")]
            ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=5.5, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(6e-4, 0.82, "perfect match", color=INK2, fontsize=8.5, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.07, 800)
        ax.set_xlabel("Solver DEM peak height (log scale)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Predicted peak / solver peak\n(log scale)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].annotate("supervised puts plasma\nwhere the solver has almost none",
                     xy=(2e-3, 20), xytext=(1.2e-2, 60), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0].annotate("label-free hits a ceiling\nabove a peak of ~100",
                     xy=(560, 0.21), xytext=(0.6, 0.09), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.suptitle("Label-free hits a ceiling on the tallest DEMs; on BP, supervised over-predicts the faintest",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(DIR, "fig3_peak_height_calibration.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (fig_error_share(), fig_relative_error(), fig_peak_calibration()):
        print("wrote", path)
