#!/usr/bin/env python3
"""Figures for results/bright_failure_story_20260929.md.

Reads the four diagnose_bright_failures.py outputs committed in
results/plots/13_bright_diagnostic_20260928/ and writes three PNGs beside them.
No server access needed.

    uv run python experiments/plot_bright_failures.py

For another label-free run, point --lf_dir/--tag at its outputs (supervised
always comes from the 13_ folder); --plain drops the titles and annotations,
which quote the original run's numbers:

    uv run python experiments/plot_bright_failures.py --plain \
        --lf_dir results/plots/16_resample_20261002 --tag resample \
        --out_dir results/plots/16_resample_20261002
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/plots/13_bright_diagnostic_20260928")
LF_DIR, SUFFIX, OUT_DIR, PLAIN = DIR, "", DIR, False   # overridden by the CLI
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
    if supervised:
        path = os.path.join(DIR, f"{track}_bright_failure_supervised.json")
    else:
        path = os.path.join(LF_DIR, f"{track}_bright_failure{SUFFIX}.json")
    with open(path) as f:
        return json.load(f)


def headline(fig, text, **kw):
    if not PLAIN:
        fig.suptitle(text, x=0.01, ha="left", fontweight="bold", **kw)


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
    headline(fig, "BP track: one pixel in 10,000 holds about 90% of the DEM error")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.01, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(OUT_DIR, "fig1_error_share_by_brightness.png")
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
    headline(fig, "Supervised improves as pixels brighten; label-free turns worse above ~3x")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_relative_error_by_brightness.png")
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
        ax.set_ylim(0.03 if PLAIN else 0.07, 800)
        ax.set_xlabel("Solver DEM peak height (log scale)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Predicted peak / solver peak\n(log scale)")
    axes[0].legend(loc="upper right", fontsize=9)
    if not PLAIN:
        axes[0].annotate("supervised puts plasma\nwhere the solver has almost none",
                         xy=(2e-3, 20), xytext=(1.2e-2, 60), fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        axes[0].annotate("label-free hits a ceiling\nabove a peak of ~100",
                         xy=(560, 0.21), xytext=(0.6, 0.09), fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    headline(fig, "Label-free hits a ceiling on the tallest DEMs; on BP, supervised over-predicts the faintest")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_peak_height_calibration.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


SOLVER = "#3b3a37"
TRACK_NAME = {"bp": "BP", "enet": "ENet"}


def examples(track):
    path = os.path.join(LF_DIR, f"{track}_aia_fit{SUFFIX}_examples.npz")
    return np.load(path) if os.path.exists(path) else None


def pick(ex, group, n, seed=0):
    """n examples drawn at random (seeded) from the stored random sample,
    skipping repeats of the same observed pixel under another solver target."""
    obs = ex[f"{group}_obs"]
    _, first = np.unique(np.round(obs, 5), axis=0, return_index=True)
    order = np.random.default_rng(seed).permutation(np.sort(first))
    return order[:n]


def fig_example_curves(track="bp"):
    ex = examples(track)
    if ex is None:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logt = np.load(os.path.join(root, "RData.npz"))["logT"][:18]
    fc = pick(ex, "flare_core", 8, seed=1)
    ob = pick(ex, "ordinary_bright", 4, seed=2)
    panels = [("flare_core", i) for i in fc] + [("ordinary_bright", i) for i in ob]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6))
    for ax, (group, i) in zip(axes.ravel(), panels):
        for key, color, name, lw in (("ref", SOLVER, "Solver", 2.2), ("label_free", LF, "Label-free", 2),
                                     ("supervised", SUP, "Supervised", 2)):
            ax.plot(logt, ex[f"{group}_{key}"][i], color=color, lw=lw, label=name)
        lead = CHANNEL_NAMES[int(np.argmax(ex[f"{group}_obs"][i] / THRESHOLDS))]
        ax.set_title(f"{'Flare core' if group == 'flare_core' else 'Ordinary Bright'}: "
                     f"{ex[f'{group}_score'][i]:.0f}x, led by {lead}", loc="left", fontsize=9.5)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("logT")
    for ax in axes[:, 0]:
        ax.set_ylabel("DEM")
    axes[0, 0].legend(fontsize=8.5, loc="upper left")
    headline(fig, f"{TRACK_NAME[track]} track: randomly drawn pixels. Top two rows flare cores (32x and above), "
               "bottom row ordinary Bright (1x to 10x)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"fig4_example_curves_{track}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


CHANNEL_NAMES = ["94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "335 Å"]
THRESHOLDS = np.array([3.1212, 16.9986, 448.8948, 498.3444, 190.7388, 10.3326])


def fig_aia_fit():
    paths = {t: os.path.join(LF_DIR, f"{t}_aia_fit{SUFFIX}.json") for t in ("bp", "enet")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    x = np.arange(6)
    w = 0.26
    for ax, track in zip(axes, ("bp", "enet")):
        with open(paths[track]) as f:
            res = json.load(f)["results"]
        for j, (src, color, name) in enumerate((("solver", SOLVER, "Solver"), ("label_free", LF, "Label-free"),
                                                ("supervised", SUP, "Supervised"))):
            ch = res[src]["by_band"][">=32x"]["channels"]
            vals = [ch[c]["recon_over_obs"] for c in ("94", "131", "171", "193", "211", "335")]
            ax.bar(x + (j - 1) * (w + 0.02), vals, width=w, color=color, label=name, edgecolor=SURFACE)
        ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{TRACK_NAME[track]} track", loc="left")
    axes[0].set_ylabel("Reconstructed / observed AIA\n(flare cores, 32x and above)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.01, 0.99 if PLAIN else 0.92))
    headline(fig, "At flare cores, how well does each DEM reproduce the observed AIA? (1 = perfect)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(OUT_DIR, "fig5_aia_fit_flare_cores.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf_dir", default=DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--plain", action="store_true")
    a = ap.parse_args()
    LF_DIR, SUFFIX, PLAIN = a.lf_dir, (f"_{a.tag}" if a.tag else ""), a.plain
    OUT_DIR = a.out_dir or a.lf_dir
    for path in (fig_error_share(), fig_relative_error(), fig_peak_calibration(),
                 fig_example_curves("bp"), fig_example_curves("enet"), fig_aia_fit()):
        if path:
            print("wrote", path)
