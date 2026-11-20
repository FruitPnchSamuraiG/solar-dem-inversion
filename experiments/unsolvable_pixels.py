#!/usr/bin/env python3
"""What are the pixels BP cannot solve even with its widest (5x) band?

For each such pixel (solver flag 0) on the full disk of the given dates:
  need       the smallest band multiple that admits any DEM (a feasibility LP, analysis only)
  culprit    a channel whose removal makes the tight band feasible, if exactly one does
  spike      per channel, the reading over the median of its 5x5 neighbourhood at full AIA
             resolution: a lone spike in one channel points to a cosmic ray or hot pixel
  where      distance from disk centre in solar radii (limb ~1)
Writes summary.json / summary.txt and cutouts of a few pixels (each channel, 21x21 full-resolution
pixels, the pixel at the centre).

    python3 experiments/unsolvable_pixels.py --dates 20151022_053428 ... --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CHANNELS = (94, 131, 171, 193, 211, 335)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz_dir", default=os.path.join(os.environ.get("SCRATCH", ""), "dem/data/lp_AIA_hofdeconv_full"))
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--n_cutouts", type=int, default=6)
    p.add_argument("--out", required=True)
    return p.parse_args()


def decode(d, key, shape_key):
    from numcodecs import Blosc
    raw = d[key].item() if d[key].ndim == 0 else d[key]
    return np.frombuffer(Blosc(cname="zstd", clevel=4, shuffle=2).decode(raw), dtype=np.float32).reshape(tuple(d[shape_key]))


def min_band(D, o, h):
    from scipy.optimize import linprog
    A, lo, hi = D / h[:, None], o / h - 1.0, o / h + 1.0
    c, k = A.shape
    one = np.ones((c, 1))
    cost = np.zeros(k + 1); cost[-1] = 1.0
    r = linprog(cost, A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.concatenate([hi, -lo]), bounds=(0, None), method="highs")
    return 1.0 + float(r.x[-1]) if r.status == 0 else np.inf


def main():
    args = parse_args()
    import torch
    from experiments.train_scaled import load_operators
    D = load_operators(torch.device("cpu"))[0].numpy().astype(np.float64)
    os.makedirs(args.out, exist_ok=True)
    rows, cut = [], []
    for date in args.dates:
        d = np.load(os.path.join(args.npz_dir, f"{date}.npz"), allow_pickle=True)
        aia = decode(d, "AIACube", "AIACubeShape")[:6]
        err = decode(d, "AIAErrors", "AIACubeShape")[:6]
        tol = np.asarray(d["tolLevel"])
        H = tol.shape[0]
        # disk centre and radius from the 171 A image on the 2048 grid
        img = np.nan_to_num(aia[2, ::2, ::2])
        on = img > np.percentile(img, 60)
        yy, xx = np.nonzero(on)
        cy, cx = yy.mean(), xx.mean()
        rad = np.sqrt(on.sum() / np.pi)
        for i, j in zip(*np.nonzero(tol == 0)):
            o = aia[:, 2 * i, 2 * j].astype(np.float64)
            e = err[:, 2 * i, 2 * j].astype(np.float64)
            if not (np.isfinite(o).all() and np.isfinite(e).all() and (e > 0).all()):
                continue
            h = args.tolfac * e
            need = min_band(D, o, h)
            drop = [c for c in range(6) if min_band(np.delete(D, c, 0), np.delete(o, c), np.delete(h, c)) <= 1.0 + 1e-9]
            spike = []
            for c in range(6):
                y0, x0 = 2 * i, 2 * j
                patch = aia[c, max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3].astype(np.float64)
                med = np.nanmedian(np.where((np.arange(patch.size) == patch.size // 2).reshape(patch.shape), np.nan, patch))
                spike.append(float(o[c] / med) if med > 0 else np.inf)
            rows.append({"date": date, "pixel": [int(i), int(j)], "readings": o.tolist(), "need": need,
                         "culprits": [CHANNELS[c] for c in drop], "spike": spike,
                         "r_over_rsun": float(np.hypot(i - cy, j - cx) / rad)})
            if len(cut) < args.n_cutouts and len(drop) == 1:
                cut.append((date, int(i), int(j), aia[:, max(0, 2 * i - 10):2 * i + 11, max(0, 2 * j - 10):2 * j + 11].copy(),
                            drop[0]))
    n = len(rows)
    need = np.array([r["need"] for r in rows])
    one = [r for r in rows if len(r["culprits"]) == 1]
    spk = np.array([max(r["spike"]) for r in rows])
    culprit_spike = [r["spike"][CHANNELS.index(r["culprits"][0])] for r in one]
    summ = {"pixels": n, "dates": args.dates,
            "need_band_multiple": {"median": float(np.median(need)), "p90": float(np.quantile(need, 0.9)),
                                   "share_needing_over_10x_pct": float(100 * (need > 10).mean())},
            "single_culprit_channel_pct": 100 * len(one) / max(n, 1),
            "no_single_culprit_pct": 100 * sum(1 for r in rows if not r["culprits"]) / max(n, 1),
            "culprit_channel_counts": {str(c): sum(1 for r in one if r["culprits"][0] == c) for c in CHANNELS},
            "culprit_reading_over_neighbour_median": {"median": float(np.median(culprit_spike)) if one else None,
                                                      "share_over_3x_pct": float(100 * (np.array(culprit_spike) > 3).mean()) if one else None},
            "max_spike_over_3x_pct": float(100 * (spk > 3).mean()),
            "r_over_rsun": {"median": float(np.median([r["r_over_rsun"] for r in rows])),
                            "off_disk_pct": float(100 * np.mean([r["r_over_rsun"] > 1.0 for r in rows]))},
            "pixels_detail": rows}
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    txt = {k: v for k, v in summ.items() if k != "pixels_detail"}
    open(os.path.join(args.out, "summary.txt"), "w").write(json.dumps(txt, indent=1) + "\n")
    print(json.dumps(txt, indent=1), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, (date, i, j, patch, c0) in enumerate(cut):
        fig, axs = plt.subplots(1, 6, figsize=(15, 2.8))
        for c, ax in enumerate(axs):
            ax.imshow(patch[c], origin="lower", cmap="magma", interpolation="nearest")
            ax.plot(patch.shape[2] // 2, patch.shape[1] // 2, "c+", ms=10)
            ax.set_title(f"{CHANNELS[c]} A" + ("  (culprit)" if c == c0 else ""), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{date} pixel ({i},{j}): unsolvable by BP; 21x21 full-resolution cutouts", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"cutout_{k}.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
