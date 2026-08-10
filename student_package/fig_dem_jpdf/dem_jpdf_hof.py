#!/usr/bin/env python3
"""
dem_jpdf_hof.py

fig:dem_jpdf — joint density of predicted vs. reference DEM, one panel per
method: DEMNet_BP vs Basis Pursuit (left), DEMNet_EN vs ElasticNet (right).
Pooled over every (AIA-sensitive bin, valid test-set pixel) pair, log-log,
same masking/quantity convention as scripts/dumpVisuals.py::dumpDEMJointPDF
(positivity floor, log10 both axes, diagonal = perfect agreement, R^2
annotated) but summarizing the whole test split as one density per method
instead of per-bin/per-patch plots.

EN uses the pre-fix placeholder checkpoint (model_best.pth = epoch_49, the
last checkpoint before the pre-fix resynthesis-loss bug's phase; see
reference_dem_model_variants memory) until the corrected retrain (job
15288474) lands -- swap MODEL path below and rerun the cache job when it does.

Data (built by scripts/compute_dem_jpdf_cache.py via slurm/dem_jpdf_cache.sh):
  paperplots/cache/dem_jpdf_methodbp.npz
  paperplots/cache/dem_jpdf_methoden.npz

Run: launchfaster python3 paperplots/dem_jpdf_hof.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(__file__))
import matplotconfig  # noqa: F401  serif/Computer Modern via the 'science' style

BASE = os.path.join(os.path.dirname(__file__), '..')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')

CACHES = [
    (os.path.join(CACHE_DIR, 'dem_jpdf_methodbp.npz'), r'DEMNet$_{\rm BP}$ vs Basis Pursuit',
     r'Basis Pursuit DEM', r'DEMNet$_{\rm BP}$ Predictions'),
    (os.path.join(CACHE_DIR, 'dem_jpdf_methoden.npz'), r'DEMNet$_{\rm EN}$ vs ElasticNet',
     r'ElasticNet DEM', r'DEMNet$_{\rm EN}$ Predictions'),
]

TEXT_COLOR = "#000000"
FS_LABEL = 13
FS_TICK = FS_LABEL + 2
FS_TITLE = 14


def plot_panel(ax, cache_path, title, xlabel, ylabel):
    d = np.load(cache_path, allow_pickle=True)
    H2D, edges, r2 = d['H2D'], d['edges'], float(d['r2'])

    H_masked = np.ma.masked_where(H2D.T == 0, H2D.T)
    pcm = ax.pcolormesh(edges, edges, H_masked, norm=mcolors.LogNorm(vmin=1),
                         cmap='turbo', shading='flat')

    lo, hi = edges[0], edges[-1]
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, alpha=0.7, zorder=5)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax.tick_params(labelsize=FS_TICK - 3, colors=TEXT_COLOR, width=1.4, length=5, which="major")
    ax.tick_params(which="minor", length=2, width=1.0, colors=TEXT_COLOR)

    ax.set_xlabel(f'{xlabel} [scaled units]', fontsize=FS_LABEL, color=TEXT_COLOR)
    ax.set_ylabel(f'{ylabel} [scaled units]', fontsize=FS_LABEL, color=TEXT_COLOR)
    ax.set_title(title, fontsize=FS_TITLE, color=TEXT_COLOR)
    ax.text(0.05, 0.93, r'$R^2 = %.2f$' % r2, transform=ax.transAxes,
            fontsize=FS_LABEL, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color(TEXT_COLOR)

    return pcm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))

    for ax, (cache_path, title, xlabel, ylabel) in zip(axes, CACHES):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'missing JPDF cache: {cache_path} -- run slurm/dem_jpdf_cache.sh first')
        pcm = plot_panel(ax, cache_path, title, xlabel, ylabel)
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Pixel-bin count', fontsize=FS_LABEL - 1)
        cb.ax.tick_params(labelsize=FS_TICK - 4)

    fig.tight_layout()

    out_name = 'dem_jpdf_hof'
    for ext in ['.png', '.pdf']:
        out = os.path.join(OUTPUT_DIR, out_name + ext)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'saved -> {out}')


if __name__ == '__main__':
    main()
