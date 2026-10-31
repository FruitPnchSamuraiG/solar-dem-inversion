#!/usr/bin/env python3
"""Figures for the paper, drawn only from saved results (no model or solver runs).

    .venv/bin/python paper/make_figures.py

Needs output/meeting_followup/bp_mu_theory_val.npz (pulled from Torch) and the
viewer images in ~/dem_preview. Writes to results/plots/24_paper_20261031/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARR = os.path.join(ROOT, "output", "meeting_followup")
OUT = os.path.join(ROOT, "results", "plots", "24_paper_20261031")
VIEW = os.path.expanduser("~/dem_preview/results/test")
DATE = "20150930_053501"

BLUE, ORANGE, GREEN = "#2a78b5", "#d9652b", "#3a9a6e"
INK, INK2, GRID = "#1b1f24", "#5b6470", "#d9dde3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3, "ytick.minor.width": 0.3, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def fig_law():
    z = np.load(os.path.join(ARR, "bp_mu_theory_val.npz"))
    mus, viol, ph = z["mus"].astype(float), z["viol"], z["price_halfwidth"]
    group, status, obs, err, lam, rr = (z["group"], z["lp_status"], z["obs"], z["err"],
                                        z["lam_lo"], z["recon94_ratio"])
    feas = status == 0
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.875, 2.2), width_ratios=[1.1, 1, 1],
                                  layout="constrained")

    sets = [("1 to 10 times cutoff", np.isin(group, (4, 5, 6)), BLUE, "o"),
            ("10 to 32 times", group == 7, GREEN, "s"),
            ("flare cores", group == 8, ORANGE, "^")]
    for name, m, col, mark in sets:
        sel = m & feas
        med = [np.median(viol[k][sel].max(1)) for k in range(len(mus))]
        p90 = [np.percentile(viol[k][sel].max(1), 90) for k in range(len(mus))]
        a.plot(mus, med, "-", marker=mark, color=col, lw=1.2, ms=3, label=name, zorder=3)
        a.plot(mus, p90, "--", color=col, lw=0.8, zorder=2)
    ref = np.array([100, 1000.0])
    a.plot(ref, 300 / ref, ":", color=INK2, lw=0.9, zorder=2)
    a.text(170, 300 / 170 * 1.3, r"$\propto 1/\mu$", color=INK2, fontsize=7)
    a.set(xscale="log", yscale="log", xlabel=r"penalty weight $\mu$",
          ylabel="largest violation (half-widths)")
    a.set_title("(a) Feasible pixels", loc="left")
    a.legend(frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.1)
    clean(a)

    k = list(mus).index(100.0)
    v, vhat = viol[k][feas], ph[feas] / 100.0
    ok = (v > 1e-4) & (vhat > 1e-4)
    b.hexbin(vhat[ok], v[ok], gridsize=38, xscale="log", yscale="log", bins="log",
             cmap="Blues", mincnt=1, linewidths=0, zorder=2)
    lim = [1e-4, 30]
    b.plot(lim, lim, "-", color=INK2, lw=0.7, zorder=3)
    b.set(xlim=lim, ylim=lim, xlabel=r"predicted $\lambda_c h_c/2\mu$",
          ylabel="measured violation")
    b.set_title(r"(b) Law at $\mu=100$", loc="left")
    b.text(1.6e-4, 5, f"median ratio {np.median(v[ok] / vhat[ok]):.2f}\n{ok.sum():,} pixel-channels",
           fontsize=6.5, color=INK, va="center")
    clean(b)

    h = 1.4 * err[:, 0]
    low = obs[:, 0] - h
    la = lam[:, 0]
    m = (group == 8) & feas & np.isfinite(la) & (la > 0) & (low > 0)
    total = obs[m, 0].sum()
    pred = [np.maximum(low[m] - la[m] * h[m] ** 2 / (2 * mu), 0).sum() / total for mu in mus]
    meas = [(rr[k][m] * obs[m, 0]).sum() / total for k in range(len(mus))]
    c.plot(mus, meas, "-", marker="o", color=ORANGE, lw=1.2, ms=3, label="minimizer", zorder=3)
    c.plot(mus, pred, "--", marker="^", color=BLUE, lw=1.0, ms=3, label="law, clipped at zero", zorder=3)
    c.set(xscale="log", ylim=(0, 1), xlabel=r"penalty weight $\mu$",
          ylabel=r"94 $\mathrm{\AA}$ reproduced / observed")
    c.set_title(r"(c) Flare cores", loc="left")
    c.legend(frameon=False, loc="lower right", handlelength=2.2, borderaxespad=0.1)
    clean(c)

    fig.savefig(os.path.join(OUT, "fig_law.pdf"))
    plt.close(fig)
    print("n flare cores in (c):", int(m.sum()), "points in (b):", int(ok.sum()))


def fig_disk():
    """Full-disk mean logT and a zoom on the hot bin, one held-out image, BP track."""
    runs = [("bp_solver", "Solver (BP)"), ("bp_mlp6_h680_clamp", "Ours (label-free)"),
            ("bp_supervised", "Supervised")]
    box = (0.655, 0.205, 0.875, 0.425)      # zoom on the north-west active region, frame fractions

    def load(run, name, zoom=False, size=620):
        im = Image.open(os.path.join(VIEW, run, DATE, name)).convert("RGB")
        if zoom:
            w, h = im.size
            im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
        return np.asarray(im.resize((size, size), Image.LANCZOS))

    fig, axes = plt.subplots(2, 4, figsize=(6.875, 3.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.0, hspace=0.0)
    axes[0, 0].imshow(load("bp_solver", "aia_3.png"))
    axes[1, 0].imshow(load("bp_solver", "aia_0.png", zoom=True))
    for j, (run, name) in enumerate(runs, start=1):
        axes[0, j].imshow(load(run, "mean_logt.png"))
        axes[1, j].imshow(load(run, "dem_14.png", zoom=True))
        axes[0, j].set_title(name, pad=2)
    axes[0, 0].set_title("Observed", pad=2)
    labels = {(0, 0): r"193 $\mathrm{\AA}$", (1, 0): r"94 $\mathrm{\AA}$ (zoom)",
              (0, 1): r"mean $\log T$", (1, 1): r"DEM at $\log T=6.9$ (zoom)"}
    for (i, j), text in labels.items():
        axes[i, j].text(0.03, 0.97, text, transform=axes[i, j].transAxes, color="white",
                        fontsize=7, va="top", ha="left")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    for row, cmap, lim, ticks, label in (
            (0, "inferno", (5.5, 7.2), [5.5, 6, 6.5, 7], r"mean $\log T$"),
            (1, "turbo", (0, 30), [0, 10, 20, 30], r"$\sqrt{\mathrm{DEM}}$")):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*lim))
        cb = fig.colorbar(sm, ax=axes[row, 1:].tolist(), fraction=0.016, pad=0.008, aspect=28, ticks=ticks)
        cb.ax.tick_params(labelsize=6, length=1.5, width=0.3, pad=1)
        cb.outline.set_linewidth(0.3)
        cb.set_label(label, fontsize=6.5, labelpad=1)
    fig.savefig(os.path.join(OUT, "fig_disk.png"), dpi=300)
    plt.close(fig)


def fig_curves():
    """Solver DEM against the loss's minimizer at mu = 1 and 100, at random flare cores."""
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    os.chdir(ROOT)
    from experiments.train_scaled import load_operators
    B = load_operators("cpu")[1].numpy().astype(np.float64)
    z = np.load(os.path.join(ARR, "bp_mu_theory_val.npz"))
    mus, group, status, lam, dem, x_opt, obs = (z["mus"].astype(float), z["group"], z["lp_status"],
                                                z["lam_lo"], z["dem"], z["x_opt"], z["obs"])
    pool = np.flatnonzero((group == 8) & (status == 0) & np.isfinite(lam[:, 0]) & (lam[:, 0] > 0))
    pick = np.random.default_rng(0).choice(pool, 3, replace=False)
    logt = 5.5 + 0.1 * np.arange(18)
    k1, k100 = list(mus).index(1.0), list(mus).index(100.0)
    fig, axes = plt.subplots(1, 3, figsize=(3.25, 1.32), sharey=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)
    for n, (ax, i) in enumerate(zip(axes, pick)):
        ax.fill_between(logt, dem[i], color="#c9ced6", lw=0, zorder=1, label="solver")
        ax.plot(logt, B @ x_opt[k1][i].astype(np.float64), "--", color=ORANGE, lw=1.0, label=r"$\mu=1$", zorder=3)
        ax.plot(logt, B @ x_opt[k100][i].astype(np.float64), "-", color=BLUE, lw=1.0, label=r"$\mu=100$", zorder=4)
        ax.set_xlim(5.5, 7.2)
        ax.set_ylim(bottom=0)
        ax.set_xticks([6, 6.5, 7])
        ax.tick_params(labelsize=6, length=1.5, pad=1)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
        ax.set_xlabel(r"$\log T$", fontsize=7, labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("DEM", fontsize=7, labelpad=1)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1.4, borderaxespad=0.0, labelspacing=0.15)
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))
    plt.close(fig)
    print("curve pixels:", pick.tolist())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_law()
    fig_disk()
    fig_curves()
    with open(os.path.join(OUT, "README.txt"), "w") as f:
        json.dump({"date": DATE, "source": "paper/make_figures.py"}, f)
