#!/usr/bin/env python3
"""Click a pixel on a full-disk band map to see its DEMs and resynthesis.

Reads viewer_<date>.npz written by experiments/band_maps.py --viewer, or viewer_ar<k>.npz written by
experiments/active_regions.py (these also carry the noise and the supervised model's DEM, so the
readings are drawn with error bars and every DEM's resynthesis in DN/s). Left: the network's worst
channel against the band BP used (grey inside, orange to dark red further outside), or BP's own
status with --show status. Right, for the clicked pixel: our DEM and BP's against temperature,
and (reconstruction - reading) / half-width per channel for both, the band drawn as +-1.

    python3 experiments/band_viewer.py viewer_20151022_053428.npz
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CHANNELS = (94, 131, 171, 193, 211, 335)
STATUS = {0: "unsolved", 1: "tight band", 3: "widened 3x", 5: "widened 5x"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("npz")
    p.add_argument("--show", choices=["worst", "status"], default="worst")
    args = p.parse_args()
    d = np.load(args.npz)
    H, W = (int(v) for v in d["shape"])
    idx = d["idx"].astype(np.int64)
    where = np.full(H * W, -1, np.int64)
    where[idx] = np.arange(len(idx))
    logT = d["logT"]
    img = np.full(H * W, np.nan)
    if args.show == "worst":
        img[idx] = d["cat"]
        cmap, labels = ListedColormap(["#d9d9d9", "#fdae61", "#f46d43", "#a50026"]), ["inside", "<=1.5", "<=3", ">3"]
        vals = (0, 1, 2, 3)
    else:
        img[idx] = np.searchsorted([0, 1, 3, 5], d["tol"])
        cmap, labels = ListedColormap(["black", "#d9d9d9", "#74add1", "#313695"]), [STATUS[k] for k in (0, 1, 3, 5)]
        vals = (0, 1, 2, 3)

    full = "err" in d.files
    fig = plt.figure(figsize=(16, 8.5 if full else 7))
    ax_map = fig.add_axes([0.03, 0.08, 0.42, 0.85])
    if full:
        ax_dem = fig.add_axes([0.53, 0.70, 0.44, 0.25])
        ax_obs = fig.add_axes([0.53, 0.38, 0.44, 0.25])
        ax_res = fig.add_axes([0.53, 0.06, 0.44, 0.25])
    else:
        ax_dem = fig.add_axes([0.55, 0.56, 0.42, 0.37])
        ax_res = fig.add_axes([0.55, 0.08, 0.42, 0.37])
    R = None
    if full:
        import os
        R = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RData.npz"))["R"][:, :18] * 1e26
    im = ax_map.imshow(img.reshape(H, W), origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax_map, fraction=0.04, pad=0.01, ticks=vals)
    cb.set_ticklabels(labels)
    ax_map.set_title(f"{args.npz}: {args.show} (click a pixel)", fontsize=9)
    marker, = ax_map.plot([], [], "c+", ms=14, mew=2)

    def show(i, j):
        k = where[i * W + j]
        ax_dem.cla(); ax_res.cla()
        if k < 0:
            ax_dem.set_title(f"pixel ({i},{j}): no valid reading")
            fig.canvas.draw_idle(); return
        ours, bp = d["ours"][k].astype(float), d["bp"][k].astype(float)
        miss, miss_bp = d["miss"][k].astype(float), d["miss_bp"][k].astype(float)
        ax_dem.plot(logT, bp, "k-", lw=2, label="BP")
        ax_dem.plot(logT, ours, "-", color="tab:blue", lw=1.6, label="ours")
        if full and np.any(d["sup"][k]):
            ax_dem.plot(logT, d["sup"][k], "--", color="tab:orange", lw=1.4, label="supervised")
        if full:
            ax_obs.cla()
            o, s_ = d["obs"][k].astype(float), d["err"][k].astype(float)
            h = 1.4 * s_ * float(d["level"][k])
            for c in range(6):
                ax_obs.add_patch(plt.Rectangle((c - 0.32, o[c] - h[c]), 0.64, 2 * h[c], color="0.85", zorder=0))
            ax_obs.errorbar(np.arange(6), o, yerr=s_, fmt="o", color="k", capsize=4, label="reading +-1 sigma", zorder=3)
            # ours: the stored miss gives its resynthesis exactly; BP's and the supervised model's via R
            ax_obs.plot(np.arange(6), o + miss * h, "o", color="tab:blue", mfc="none", ms=7, mew=1.6, label="ours", zorder=4)
            if np.isfinite(bp).all():
                ax_obs.plot(np.arange(6) - 0.2, R @ bp, "x", color="k", ms=7, mew=1.6, label="BP", zorder=4)
            if np.any(d["sup"][k]):
                ax_obs.plot(np.arange(6) + 0.2, R @ d["sup"][k].astype(float), "s", color="tab:orange", mfc="none",
                            ms=7, mew=1.6, label="supervised", zorder=4)
            ax_obs.set_yscale("symlog", linthresh=1.0)
            ax_obs.set_xticks(np.arange(6), [str(c) for c in CHANNELS])
            ax_obs.set(ylabel="DN/s", title="readings (grey: BP's band) and resynthesis")
            ax_obs.legend(fontsize=7, frameon=False, ncol=4)
        ax_dem.set(xlabel="logT", ylabel="DEM")
        ax_dem.legend(fontsize=8, frameon=False)
        ax_dem.set_title(f"pixel ({i},{j}) | BP: {STATUS.get(int(d['tol'][k]), '?')} | readings "
                         + " ".join(f"{v:.0f}" for v in d["obs"][k]), fontsize=8)
        x = np.arange(6)
        ax_res.axhspan(-1, 1, color="0.88", zorder=0)
        ax_res.axhline(0, color="0.6", lw=0.8)
        ax_res.bar(x - 0.18, miss_bp, 0.36, color="k", label="BP")
        ax_res.bar(x + 0.18, miss, 0.36, color="tab:blue", label="ours")
        lim = max(3.0, np.nanmax(np.abs(np.r_[miss, miss_bp])) * 1.1)
        ax_res.set_ylim(-min(lim, 50), min(lim, 50))
        ax_res.set_xticks(x, [str(c) for c in CHANNELS])
        ax_res.set(xlabel="channel (A)", ylabel="(recon - reading) / half-width")
        ax_res.legend(fontsize=8, frameon=False)
        marker.set_data([j], [i])
        fig.canvas.draw_idle()

    def onclick(ev):
        if ev.inaxes is ax_map and ev.xdata is not None:
            show(int(round(ev.ydata)), int(round(ev.xdata)))

    fig.canvas.mpl_connect("button_press_event", onclick)
    show(H // 2, W // 2)
    plt.show()


if __name__ == "__main__":
    main()
