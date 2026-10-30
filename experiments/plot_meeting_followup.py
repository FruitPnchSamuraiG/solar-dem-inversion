#!/usr/bin/env python3
"""Figures for the 21 Oct follow-up analyses (report sections 3.3, 7 and 8).

Reads the summaries in results/plots/23_meeting_followup_20261024/ and the per-pixel
arrays pulled from Torch into output/meeting_followup/ (mu_theory.py, blowup_census.py,
jpdf_pixels.py, final_layer.py), and writes PNGs next to the summaries:

    fig1_violation_vs_mu.png      measured band violation against the 1/mu law
    fig2_relaxed_band_screen.png  the mu screen with the fixed and the solver's band
    fig3_blowup_distance.png      distance from the worst pixels to the training data
    fig4_lone_spike_example.png   one lone 94 A spike: observation and reconstructions
    fig5_final_layer.png          output layer: spectrum, sorted correlation, usage
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "plots", "23_meeting_followup_20261024")
ARR = os.path.join(ROOT, "output", "meeting_followup")
LABELS = ['<0.03x', '0.03-0.1x', '0.1-0.3x', '0.3-1x', '1-1.8x', '1.8-3.2x', '3.2-10x',
          '10-32x', '>=32x']
CH = ["94", "131", "171", "193", "211", "335"]
# reference categorical slots (dataviz palette, light mode) and text inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 170, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(RES, name))
    plt.close(fig)
    print("wrote", name)


def fig_violation():
    z = np.load(os.path.join(ARR, "bp_mu_theory_val.npz"), allow_pickle=True)
    mus, viol, ph = z["mus"].astype(float), z["viol"], z["price_halfwidth"]
    group, tol = z["group"], z["tol"]
    feas = tol == 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    sets = [("Bright, 1 to 10x", np.isin(group, (4, 5, 6)), C1),
            ("10 to 32x", group == 7, C3), ("Flare cores, 32x and up", group == 8, C2)]
    for name, m, col in sets:
        sel = m & feas
        med = [np.median(viol[k][sel].max(1)) for k in range(len(mus))]
        p90 = [np.percentile(viol[k][sel].max(1), 90) for k in range(len(mus))]
        a.plot(mus, med, "-o", color=col, lw=1.6, ms=4, label=f"{name}, median")
        a.plot(mus, p90, "--", color=col, lw=1.2, alpha=0.8, label=f"{name}, 90th pct")
    ref = np.array([3, 1000.0])
    a.plot(ref, 0.5 * 3 / ref, ":", color=INK2, lw=1.2)
    a.text(30, 0.5 * 3 / 30 * 1.25, "slope 1/mu", color=INK2, fontsize=8, rotation=-28)
    a.set(xscale="log", yscale="log", xlabel="band weight mu",
          ylabel="largest violation (half band-widths)",
          title="Violation falls as 1/mu at feasible pixels")
    a.legend(fontsize=7, loc="lower left")
    k = list(mus).index(100.0)
    v, vhat = viol[k][feas], ph[feas] / 100.0
    ok = (v > 1e-4) & (vhat > 1e-4)
    hb = b.hexbin(vhat[ok], v[ok], gridsize=45, xscale="log", yscale="log", bins="log",
                  cmap="Blues", mincnt=1, linewidths=0)
    lim = [1e-4, 30]
    b.plot(lim, lim, "-", color=INK2, lw=1)
    b.set(xlim=lim, ylim=lim, xlabel="predicted  lambda t sigma / (2 mu)",
          ylabel="measured violation", title="Measured against predicted, mu = 100")
    ratio = v[ok] / vhat[ok]
    b.text(1.5e-4, 8, f"median ratio {np.median(ratio):.2f}\n{ok.sum():,} pixel-channels",
           fontsize=8, color=INK)
    fig.colorbar(hb, ax=b, label="pixel-channels")
    save(fig, "fig1_violation_vs_mu.png")


def fig_relaxed():
    r = json.load(open(os.path.join(RES, "bp_relaxed_band_screen.json")))
    mus = [float(m) for m in r["mus"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    rows = [("dem_error", "DEM error (population-weighted)", "log"),
            ("quiet_mse", "Quiet DEM MSE", "log"),
            ("core_94", "flare-core 94 A reproduced", "linear")]
    ref = {"dem_error": 6.10, "quiet_mse": 0.054, "core_94": 0.91}   # supervised model, slide 19
    for ax, (key, title, ys) in zip(axes, rows):
        for var, name, col in (("fixed", "fixed 1.4 sigma band", C2),
                               ("relaxed_noise", "the solver's band", C1)):
            ax.plot(mus, [r[var][f"{m:g}"][key] for m in mus], "-o", color=col, lw=1.6, ms=4,
                    label=name)
        ax.axhline(ref[key], color=INK2, ls=":", lw=1.1, label="supervised model")
        ax.set(xscale="log", yscale=ys, xlabel="band weight mu", title=title)
        if key == "core_94":
            ax.axhline(0.94, color=C3, ls="--", lw=1.1, label="solver itself")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("Best answer under the loss on validation: relaxing the band where the solver "
                 "did removes the trade-off", fontsize=9.5, color=INK, y=1.02)
    save(fig, "fig2_relaxed_band_screen.png")


def fig_blowup():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, track, names in ((axes[0], "bp", [("h336_mu100_cont", "409k (adopted)"),
                                              ("h680_mu100_ep120", "1.5M"),
                                              ("h1800_mu100", "10M")]),
                             (axes[1], "enet", [("h336_sqrtff", "409k (final)"),
                                                ("h680_log1p", "1.4M, log1p"),
                                                ("h1800_sqrtff", "10M")])):
        z = np.load(os.path.join(ARR, f"{track}_blowup_census.npz"), allow_pickle=True)
        data, ticks = [], []
        cb, cd = z["controls_band"], z["controls_d1"]
        for g, lab in ((3, "controls\n0.3-1x"), (5, "controls\n1.8-3.2x"), (8, "controls\n32x+")):
            data.append(cd[cb == g]); ticks.append(lab)
        for key, lab in names:
            sse, d1 = z[f"worst_{key}_sse"], z[f"worst_{key}_d1"]
            keyin = z[f"worst_{key}_input_key"]
            order = np.argsort(-sse)
            _, first = np.unique(keyin[order], return_index=True)
            top = order[np.sort(first)][:100]
            data.append(d1[top]); ticks.append(f"worst 100\n{lab}")
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.5),
                        medianprops=dict(color=INK, lw=1.3))
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#d9d8d3" if k < 3 else [C1, C2, C4][k - 3])
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(ticks) + 1), ticks, fontsize=7)
        ax.set(yscale="log", title=("BP" if track == "bp" else "ElasticNet") +
               ": distance to the nearest training pixel")
    axes[0].set_ylabel("standardised log-intensity distance")
    save(fig, "fig3_blowup_distance.png")


def fig_spike():
    z = np.load(os.path.join(ARR, "bp_jpdf_pixels.npz"), allow_pickle=True)
    obs, med = z["low_171_obs"], z["low_171_med"]
    rl, rs, rsup = z["low_171_rec_lf"], z["low_171_rec_sol"], z["low_171_rec_sup"]
    lone = ((obs > 1e-3).sum(1) == 1) & (obs[:, 0] > 1e-3)
    k = int(np.flatnonzero(lone)[np.argmax(rl[lone][:, 2])])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.0))
    x = np.arange(6)
    floor = 0.05
    a.bar(x - 0.2, np.maximum(obs[k], floor), 0.38, color=C1, label="this pixel")
    a.bar(x + 0.2, np.maximum(med[k], floor), 0.38, color="#c3c2b7", label="median of its 8 neighbours")
    a.set(yscale="log", xticks=x, xticklabels=[c + " A" for c in CH], ylabel="observed DN/s",
          title="A lone 94 A reading; five channels clamped to zero")
    a.legend(fontsize=7)
    for arr, name, col, off in ((rl[k], "label-free (409k)", C1, -0.27), (rsup[k], "supervised", C2, 0.0),
                                (rs[k], "solver", C3, 0.27)):
        b.bar(x + off, np.maximum(arr, floor), 0.26, color=col, label=name)
    b.set(yscale="log", xticks=x, xticklabels=[c + " A" for c in CH], ylabel="reconstructed DN/s",
          title="What each DEM reproduces there")
    b.legend(fontsize=7)
    fig.text(0.5, -0.04, f"Values below {floor} DN/s drawn at {floor}. Shared test set, BP track.",
             ha="center", fontsize=7, color=INK2)
    save(fig, "fig4_lone_spike_example.png")


def fig_final_layer():
    z = np.load(os.path.join(ARR, "final_layer.npz"), allow_pickle=True)
    temps = np.round(np.arange(5.5, 7.21, 0.1), 1)
    labels = [f"{s} {t:.1f}" for s in ("spike", "0.1", "0.2") for t in temps]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.4, 4.1),
                                  gridspec_kw={"width_ratios": [1, 1.25, 1.1]})
    for key, name, col in (("bp", "BP adopted", C1), ("enet", "ElasticNet", C3),
                           ("bp_old", "BP old (mu = 1, log1p)", C2)):
        s = z[f"{key}__s"]
        a.plot(np.arange(1, len(s) + 1), s / s[0], "-", color=col, lw=1.6, label=name)
    a.set(yscale="log", xlabel="singular value index", ylabel="relative to the largest",
          title="Output layer (336 to 54): singular values")
    a.legend(fontsize=7)
    corr = z["bp__out_corr"]
    e1 = z["bp__out_corr_vectors"][:, 0]          # leading eigenvector (eig_desc order)
    e1 = e1 if e1.sum() >= 0 else -e1
    order = np.argsort(-e1)
    im = b.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    step = 3
    b.set_xticks(range(0, 54, step), [labels[i] for i in order[::step]], rotation=90, fontsize=6)
    b.set_yticks(range(0, 54, step), [labels[i] for i in order[::step]], fontsize=6)
    b.grid(False)
    b.set_title("BP adopted: correlation of the 54 outputs,\nsorted by the leading eigenvector")
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.02, label="correlation over pixels")
    w_net, w_sol = z["bp__sol_x_network"], z["sol_x_solver"]
    for w, name, col, off in ((w_net, "network", C1, -0.2), (w_sol, "solver", C3, 0.2)):
        by_t = w.sum(0).reshape(3, 18).sum(0) / w.sum()
        shape = w.sum(0).reshape(3, 18).sum(1) / w.sum()
        c.bar(temps + off * 0.1, by_t, 0.04, color=col,
              label=f"{name}: spike {shape[0]:.0%}, 0.1 dex {shape[1]:.0%}, 0.2 dex {shape[2]:.0%}")
    c.set(xlabel="centre logT of the basis element", ylabel="share of all weight",
          title="Where the weight goes, BP adopted vs solver\n(same 20k validation pixels)")
    c.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_final_layer.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["violation", "relaxed", "blowup", "spike", "final_layer"]
    for name in which:
        globals()[f"fig_{name}"]()
