#!/usr/bin/env python3
"""Full-disk maps of where the network's DEM stays inside BP's bands, beside BP's own status.

For each date, from BP's solver output (readings, noise and the per-pixel flag on the 2048 grid:
AIA pixel (2i, 2j) for DEM pixel (i, j)) and the network's DEM at every pixel:

  miss_c = (reconstruction_c - reading_c) / half-width_c, half-width = 1.4 sigma (the tight band)

Pixels BP solved with a widened band are measured against that band as well (miss / level), so
"inside" always means inside the band BP itself used. Figures per date:

  status.png        BP's flag: tight band / widened 3x / widened 5x / unsolved
  worst.png         the network's worst channel: inside / up to 1.5 / up to 3 / beyond 3 half-widths
  channels.png      signed miss per channel (blue too dim, red too bright), clipped at +-5
  bp_worst.png      the same worst-channel map for BP's own DEM (inside by construction, up to rounding)

plus counts per (BP flag x our category) in summary.json and, with --viewer, a compact
viewer_<date>.npz for experiments/band_viewer.py (click a pixel: its DEMs and resynthesis).

    python3 experiments/band_maps.py --model <ckpt> --dates 20151022_053428 --out <dir> --viewer
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
N_BINS = 18
CHANNELS = (94, 131, 171, 193, 211, 335)
CATS = ("inside", "<=1.5", "<=3", ">3")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz_dir", default=os.path.join(os.environ.get("SCRATCH", ""), "dem/data/lp_AIA_hofdeconv_full"))
    p.add_argument("--model", required=True)
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_solver(path):
    from numcodecs import Blosc
    c = Blosc(cname="zstd", clevel=4, shuffle=2)
    d = np.load(path, allow_pickle=True)

    def dec(key, shape_key):
        raw = d[key].item() if d[key].ndim == 0 else d[key]
        return np.frombuffer(c.decode(raw), dtype=np.float32).reshape(tuple(d[shape_key]))
    aia = dec("AIACube", "AIACubeShape")[:6, ::2, ::2]
    err = dec("AIAErrors", "AIACubeShape")[:6, ::2, ::2]
    dem = dec("DEMCube", "DEMCubeShape")[:N_BINS]
    return aia, err, dem, np.asarray(d["tolLevel"])


def predict(path, obs):
    import torch
    from experiments.train_scaled import load_operators
    from src.scaled_eval import load_scaled_model
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    dev = torch.device("cpu")
    D_t, B_t, n_basis, _ = load_operators(dev)
    model, ck = load_scaled_model(path, n_basis, dev)
    k, c = ck["patch_size"], ck["patch_size"] // 2
    W = np.empty((len(obs), D_t.shape[1]), np.float32)
    with torch.no_grad():
        for s in range(0, len(obs), 65536):
            o = torch.as_tensor(obs[s:s + 65536])
            patch = torch.zeros(len(o), 6, k, k)
            patch[:, :, c, c] = o
            W[s:s + len(o)] = model(patch).numpy()
    return W, D_t.numpy().astype(np.float64), B_t.numpy()[:N_BINS].astype(np.float64)


def save_map(path, img, cmap, vmin, vmax, title, ticks=None, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if ticks is not None:
        cb.set_ticks(ticks); cb.set_ticklabels(labels)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap
    os.makedirs(args.out, exist_ok=True)
    R = np.load(os.path.join(ROOT, "RData.npz"))["R"][:, :N_BINS].astype(np.float64) * 1e26
    summary = {}
    for date in args.dates:
        aia, err, dem_bp, tol = load_solver(os.path.join(args.npz_dir, f"{date}.npz"))
        H, W_ = tol.shape
        obs = aia.reshape(6, -1).T
        e = err.reshape(6, -1).T
        valid = np.isfinite(obs).all(1) & np.isfinite(e).all(1) & (e > 0).all(1)
        idx = np.flatnonzero(valid)
        w, D, B = predict(args.model, obs[idx].astype(np.float32))
        level = np.where(tol.ravel()[idx] > 1, tol.ravel()[idx], 1).astype(np.float64)
        h = args.tolfac * e[idx] * level[:, None]          # the band BP itself used
        miss = (w.astype(np.float64) @ D.T - obs[idx]) / h
        bp = dem_bp.reshape(N_BINS, -1).T[idx].astype(np.float64)
        miss_bp = np.where(np.isfinite(bp).all(1)[:, None], (np.nan_to_num(bp) @ R.T - obs[idx]) / h, np.nan)
        worst = np.abs(miss).max(1)
        cat = np.digitize(worst, [1.02, 1.5, 3.0])           # 0 inside .. 3 beyond 3
        img = np.full(H * W_, np.nan); img[idx] = cat; img = img.reshape(H, W_)
        st = np.full(H * W_, np.nan); st[valid] = np.searchsorted([0, 1, 3, 5], tol.ravel()[valid]); st = st.reshape(H, W_)
        cm4 = ListedColormap(["#d9d9d9", "#fdae61", "#f46d43", "#a50026"])
        save_map(os.path.join(args.out, f"{date}_worst.png"), img, cm4, -0.5, 3.5,
                 f"{date}: network's worst channel vs the band BP used (half-widths)", [0, 1, 2, 3],
                 ["inside", "<= 1.5", "<= 3", "> 3"])
        save_map(os.path.join(args.out, f"{date}_status.png"), st, ListedColormap(["black", "#d9d9d9", "#74add1", "#313695"]),
                 -0.5, 3.5, f"{date}: BP status", [0, 1, 2, 3], ["unsolved", "tight band", "widened 3x", "widened 5x"])
        wb = np.full(H * W_, np.nan); wb[idx] = np.digitize(np.nanmax(np.abs(miss_bp), 1), [1.02, 1.5, 3.0]); wb = wb.reshape(H, W_)
        save_map(os.path.join(args.out, f"{date}_bp_worst.png"), wb, cm4, -0.5, 3.5,
                 f"{date}: BP's own DEM, worst channel vs its band", [0, 1, 2, 3], ["inside", "<= 1.5", "<= 3", "> 3"])
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
        for c, ax in enumerate(axs.ravel()):
            m = np.full(H * W_, np.nan); m[idx] = np.clip(miss[:, c], -5, 5); m = m.reshape(H, W_)
            im = ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
            ax.set_title(f"{CHANNELS[c]} A: (recon - reading) / half-width", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="negative: too dim, positive: too bright (clipped at 5)")
        fig.savefig(os.path.join(args.out, f"{date}_channels.png"), dpi=120); plt.close(fig)
        t = tol.ravel()[idx]
        table = {}
        for lv, name in ((1, "tight"), (3, "widened 3x"), (5, "widened 5x"), (0, "unsolved")):
            m = t == lv
            table[name] = {"pixels": int(m.sum()), **{CATS[k]: int((m & (cat == k)).sum()) for k in range(4)}}
        summary[date] = {"valid_pixels": int(len(idx)), "by_bp_status": table,
                         "worst_channel_when_outside": {str(CHANNELS[c]): int(((cat > 0) & (np.abs(miss).argmax(1) == c)).sum())
                                                        for c in range(6)}}
        print(date, json.dumps(summary[date]), flush=True)
        if args.viewer:
            f16 = lambda a: a.astype(np.float16)
            np.savez_compressed(os.path.join(args.out, f"viewer_{date}.npz"), shape=np.array([H, W_]), idx=idx.astype(np.int32),
                                obs=obs[idx].astype(np.float32), half=h.astype(np.float32), tol=t.astype(np.uint8),
                                ours=f16(w.astype(np.float64) @ B.T), bp=f16(np.nan_to_num(bp)), miss=f16(miss),
                                miss_bp=f16(np.nan_to_num(miss_bp)), cat=cat.astype(np.uint8),
                                logT=np.load(os.path.join(ROOT, "RData.npz"))["logT"][:N_BINS])
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
