#!/usr/bin/env python3
"""40 active regions from the test dates: band maps, and DEM line plots of their hard pixels.

Scan: on each test date, Bright pixels (any channel at or above its Bright threshold) are grouped into
connected regions (lightly smoothed); regions touching the image frame are skipped. One region per date
at most: the --n_flaring regions with the brightest peak, then the largest by Bright area, until --n.

Per region (a --box x --box crop of the 2048 grid, AIA pixel (2i, 2j) for DEM pixel (i, j)), with our
network, the supervised model and BP's own DEM:
  ar_<k>.png       171 A image, BP's status, our worst channel and the supervised model's worst channel
                   against the band BP used (inside / <= 1.5 / <= 3 / > 3 half-widths), the picked pixels
                   numbered on every panel
  viewer_ar<k>.npz for experiments/band_viewer.py (click a pixel: DEMs, readings with error bars,
                   resynthesis)
Picked pixels per region (the hard cases, plus one ordinary pixel for reference):
  worst_dem    the 3 tight-band pixels with our largest squared DEM difference from BP
  worst_shape  the tight-band Bright pixel with our largest relative DEM difference
  worst_band   the tight-band pixel where our resynthesis misses its band the most
  brightest    the brightest pixel (any BP status)
  widened      the brightest pixel BP solved only with a widened band (3x / 5x), if any
  unsolved     a pixel BP could not solve, if any
  typical      a random tight-band Bright pixel
For each: the DEM as a line (BP, ours, supervised), the six readings with error bars (1 sigma; grey box
= BP's band) and each DEM's resynthesis in DN/s, and the resynthesis minus reading in sigma.
Pages in by_region.pdf (region overview, then its pixels) and by_case.pdf; numbers in summary.json.

    python3 experiments/active_regions.py --model <ckpt> --supervised paper_model/checkpoints/BP_5_Wide_10.1M.pth \
        --out <dir>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.band_maps import load_solver, predict  # noqa: E402

N_BINS = 18
CHANNELS = (94, 131, 171, 193, 211, 335)
STATUS = {0: "unsolved", 1: "tight band", 3: "widened 3x", 5: "widened 5x"}
CASES = ("worst_dem", "worst_shape", "worst_band", "brightest", "widened", "unsolved", "typical")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz_dir", default=os.path.join(os.environ.get("SCRATCH", ""), "dem/data/lp_AIA_hofdeconv_full"))
    p.add_argument("--dates_file", default=os.path.join(os.environ.get("SCRATCH", ""),
                                                        "dem/data/lp_AIA_hofdeconv_full_DS/test_datetimes.txt"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--model", required=True)
    p.add_argument("--supervised", default=None)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--n_flaring", type=int, default=15)
    p.add_argument("--box", type=int, default=192)
    p.add_argument("--min_area", type=int, default=400, help="Bright pixels a region needs")
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def scan(job):
    """One date -> candidate regions (centre, Bright area, peak brightness)."""
    path, date, thr = job
    from numcodecs import Blosc
    from scipy import ndimage
    d = np.load(path, allow_pickle=True)
    raw = d["AIACube"].item() if d["AIACube"].ndim == 0 else d["AIACube"]
    aia = np.frombuffer(Blosc(cname="zstd", clevel=4, shuffle=2).decode(raw), np.float32).reshape(tuple(d["AIACubeShape"]))
    aia = aia[:6, ::2, ::2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nan_to_num((aia / np.asarray(thr, np.float32)[:, None, None]).max(0), nan=0.0, posinf=0.0)
    bright = ratio >= 1.0
    grown = ndimage.uniform_filter(bright.astype(np.float32), 15) > 0.15
    lab, n = ndimage.label(grown)
    H, W = ratio.shape
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        if sl[0].start < 20 or sl[1].start < 20 or sl[0].stop > H - 20 or sl[1].stop > W - 20:
            continue                                   # touches the frame: edge artifacts
        m = (lab[sl] == k) & bright[sl]
        area = int(m.sum())
        if area < 400:
            continue
        r = np.where(m, ratio[sl], 0.0)
        ii, jj = np.nonzero(m)
        wgt = np.sqrt(r[ii, jj])
        out.append({"date": date, "cy": int(sl[0].start + np.average(ii, weights=wgt)),
                    "cx": int(sl[1].start + np.average(jj, weights=wgt)), "area": area,
                    "peak": float(r.max()), "n_core": int((r >= 32).sum())})
    return out


def choose(cands, n, n_flaring, min_area):
    cands = [c for c in cands if c["area"] >= min_area]
    used, pick = set(), []
    for key, k in (("peak", n_flaring), ("area", n)):
        for c in sorted(cands, key=lambda c: -c[key]):
            if len(pick) >= (n_flaring if key == "peak" else n):
                break
            if c["date"] not in used:
                used.add(c["date"]); pick.append(c | {"chosen_by": key})
    return pick


def predict_supervised(path, obs):
    import torch
    from student_package.table1_dem_metrics.compute_paper_table_metrics import load_model, predict_regression_only
    model, _ = load_model(path, torch.device("cpu"))
    out = []
    with torch.no_grad():
        for s in range(0, len(obs), 4096):
            o = torch.as_tensor(np.clip(obs[s:s + 4096], 0, None))
            out.append(predict_regression_only(model, o.T[None, :, None, :])[0, :N_BINS, 0, :].T.clamp(min=0).double().numpy())
    return np.concatenate(out)


def pick_pixels(P, rng):
    tight = P["tol"] == 1
    bright = P["ratio"] >= 1.0
    sq = ((P["ours"] - np.nan_to_num(P["bp"])) ** 2).sum(1)
    rel = np.abs(P["ours"] - np.nan_to_num(P["bp"])).sum(1) / np.maximum(np.nan_to_num(P["bp"]).sum(1), 1e-6)
    worst = np.abs(P["miss"]).max(1)
    picks = []

    def add(case, idx):
        for i in np.atleast_1d(idx):
            if int(i) not in [p for _, p in picks]:
                picks.append((case, int(i)))
    t = np.flatnonzero(tight)
    if t.size:
        add("worst_dem", t[np.argsort(-sq[t])[:3]])
        tb = np.flatnonzero(tight & bright)
        if tb.size:
            add("worst_shape", tb[np.argmax(rel[tb])])
            add("typical", rng.choice(tb))
        add("worst_band", t[np.argmax(worst[t])])
    add("brightest", np.argmax(P["ratio"]))
    w = np.flatnonzero(P["tol"] > 1)
    if w.size:
        add("widened", w[np.argmax(P["ratio"][w])])
    u = np.flatnonzero(P["tol"] == 0)
    if u.size:
        add("unsolved", u[np.argmax(P["ratio"][u])])
    order = {c: k for k, c in enumerate(CASES)}
    return sorted(picks, key=lambda x: order[x[0]])


def pixel_figure(plt, P, i, logT, title):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.1), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    if np.isfinite(P["bp"][i]).all():
        ax[0].plot(logT, P["bp"][i], "k-", lw=2, label="BP")
    ax[0].plot(logT, P["ours"][i], "-", color="tab:blue", lw=1.6, label="ours")
    if P.get("sup") is not None:
        ax[0].plot(logT, P["sup"][i], "--", color="tab:orange", lw=1.4, label="supervised")
    ax[0].set(xlabel="logT", ylabel="DEM", title="DEM")
    ax[0].legend(fontsize=8, frameon=False)
    x = np.arange(6)
    o, s, lev = P["obs"][i], P["err"][i], P["level"][i]
    h = 1.4 * s * lev
    for c in range(6):
        ax[1].add_patch(plt.Rectangle((c - 0.32, o[c] - h[c]), 0.64, 2 * h[c], color="0.85", zorder=0))
    ax[1].errorbar(x, o, yerr=s, fmt="o", color="k", ms=5, capsize=4, lw=1.5, label="reading +-1 sigma", zorder=3)
    marks = [("rec_bp", "x", "k", "BP"), ("rec_ours", "o", "tab:blue", "ours"), ("rec_sup", "s", "tab:orange", "supervised")]
    for k, (key, m, col, lab) in enumerate(marks):
        if P.get(key) is None or not np.isfinite(P[key][i]).all():
            continue
        ax[1].plot(x + (k - 1) * 0.2, P[key][i], m, color=col, mfc="none" if m != "x" else col, ms=7, mew=1.6,
                   label=f"{lab} resynthesis", zorder=4)
    ax[1].set_yscale("symlog", linthresh=1.0)
    ax[1].set_xticks(x, [f"{c}" for c in CHANNELS])
    ax[1].set(xlabel="channel (A)", ylabel="DN/s", title="readings (grey: BP's band) and resynthesis")
    ax[1].legend(fontsize=7, frameon=False)
    ax[2].axhspan(-1.4 * lev, 1.4 * lev, color="0.85", zorder=0)
    ax[2].axhline(0, color="0.5", lw=0.8)
    lim = 4.0
    for k, (key, m, col, lab) in enumerate(marks):
        if P.get(key) is None or not np.isfinite(P[key][i]).all():
            continue
        z = (P[key][i] - o) / s
        lim = max(lim, float(np.abs(z).max()) * 1.15)
        ax[2].plot(x + (k - 1) * 0.2, z, m, color=col, mfc="none" if m != "x" else col, ms=7, mew=1.6, label=lab)
    lim = min(lim, 60)
    ax[2].set_ylim(-lim, lim)
    ax[2].set_xticks(x, [f"{c}" for c in CHANNELS])
    ax[2].set(xlabel="channel (A)", ylabel="(resynthesis - reading) / sigma",
              title="miss in error bars (grey: BP's band; clipped at 60)")
    ax[2].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    os.makedirs(args.out, exist_ok=True)
    payload = json.load(open(args.thresholds))
    thr = np.asarray(payload.get("test", payload), np.float64).tolist()
    dates = [l.strip() for l in open(args.dates_file) if l.strip()]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cands = sum(pool.map(scan, [(os.path.join(args.npz_dir, f"{d}.npz"), d, thr) for d in dates]), [])
    regions = choose(cands, args.n, args.n_flaring, args.min_area)
    print(f"{len(cands)} candidate regions on {len(dates)} dates; chose {len(regions)}", flush=True)
    R = np.load(os.path.join(ROOT, "RData.npz"))["R"][:, :N_BINS].astype(np.float64) * 1e26
    logT = np.linspace(5.5, 7.2, N_BINS)
    rng = np.random.default_rng(args.seed)
    cm4 = ListedColormap(["#d9d9d9", "#fdae61", "#f46d43", "#a50026"])
    cmS = ListedColormap(["black", "#d9d9d9", "#74add1", "#313695"])
    summary, pages = [], []
    pdf_r = PdfPages(os.path.join(args.out, "by_region.pdf"))
    tot = {m: np.zeros((4, 4), np.int64) for m in ("ours", "sup")}
    for k, reg in enumerate(sorted(regions, key=lambda r: r["date"])):
        aia, err, dem_bp, tol = load_solver(os.path.join(args.npz_dir, f"{reg['date']}.npz"))
        H, W_ = tol.shape
        b = args.box // 2
        y0, x0 = int(np.clip(reg["cy"] - b, 0, H - args.box)), int(np.clip(reg["cx"] - b, 0, W_ - args.box))
        sl = (slice(y0, y0 + args.box), slice(x0, x0 + args.box))
        obs = aia[:, sl[0], sl[1]].reshape(6, -1).T.astype(np.float64)
        e = err[:, sl[0], sl[1]].reshape(6, -1).T.astype(np.float64)
        t = tol[sl].ravel()
        valid = np.isfinite(obs).all(1) & np.isfinite(e).all(1) & (e > 0).all(1)
        idx = np.flatnonzero(valid)
        w, D, B = predict(args.model, obs[idx].astype(np.float32))
        w = w.astype(np.float64)
        P = {"obs": obs[idx], "err": e[idx], "tol": t[idx],
             "level": np.where(t[idx] > 1, t[idx], 1).astype(np.float64),
             "ours": w @ B.T, "rec_ours": w @ D.T,
             "bp": dem_bp[:, sl[0], sl[1]].reshape(N_BINS, -1).T[idx].astype(np.float64)}
        with np.errstate(invalid="ignore"):
            P["ratio"] = (P["obs"] / np.asarray(thr)).max(1)
        P["rec_bp"] = np.where(np.isfinite(P["bp"]).all(1)[:, None], np.nan_to_num(P["bp"]) @ R.T, np.nan)
        P["sup"] = predict_supervised(args.supervised, P["obs"].astype(np.float32)) if args.supervised else None
        P["rec_sup"] = P["sup"] @ R.T if P["sup"] is not None else None
        h = args.tolfac * P["err"] * P["level"][:, None]
        P["miss"] = (P["rec_ours"] - P["obs"]) / h
        cats = {"ours": np.digitize(np.abs(P["miss"]).max(1), [1.02, 1.5, 3.0])}
        if P["sup"] is not None:
            cats["sup"] = np.digitize(np.abs((P["rec_sup"] - P["obs"]) / h).max(1), [1.02, 1.5, 3.0])
        st = np.searchsorted([0, 1, 3, 5], P["tol"])
        for m, c in cats.items():
            np.add.at(tot[m], (st, c), 1)
        picks = pick_pixels(P, rng)
        # overview page
        def img(v):
            a = np.full(args.box * args.box, np.nan); a[idx] = v; return a.reshape(args.box, args.box)
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
        axs[0].imshow(np.log10(np.clip(aia[2][sl], 1, None)), origin="lower", cmap="magma")
        axs[0].set_title("171 A (log)", fontsize=9)
        axs[1].imshow(img(st), origin="lower", cmap=cmS, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[1].set_title("BP: black unsolved, grey tight, blue widened", fontsize=9)
        axs[2].imshow(img(cats["ours"]), origin="lower", cmap=cm4, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[2].set_title("ours: grey inside, orange <=1.5, red <=3, dark >3", fontsize=9)
        if "sup" in cats:
            axs[3].imshow(img(cats["sup"]), origin="lower", cmap=cm4, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[3].set_title("supervised, same colours", fontsize=9)
        flat = np.flatnonzero(valid)
        for n, (case, i) in enumerate(picks, start=1):
            yy, xx = divmod(int(flat[i]), args.box)
            for ax in axs:
                ax.plot(xx, yy, "c+", ms=9, mew=1.5)
                ax.text(xx + 3, yy + 3, str(n), color="cyan", fontsize=8, weight="bold")
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        tb = P["tol"] == 1
        frac = {m: float(100 * (c[tb] == 0).mean()) if tb.any() else float("nan") for m, c in cats.items()}
        fig.suptitle(f"region {k + 1}: {reg['date']} centre ({reg['cy']},{reg['cx']}), peak {reg['peak']:.0f}x Bright, "
                     f"{reg['n_core']} flare-core pixels | tight-band pixels inside: ours {frac['ours']:.1f}%"
                     + (f", supervised {frac['sup']:.1f}%" if "sup" in frac else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"ar_{k + 1:02d}.png"), dpi=100)
        pdf_r.savefig(fig); plt.close(fig)
        rows = []
        for n, (case, i) in enumerate(picks, start=1):
            yy, xx = divmod(int(flat[i]), args.box)
            bp_ok = np.isfinite(P["bp"][i]).all()
            relv = float(np.abs(P["ours"][i] - P["bp"][i]).sum() / max(P["bp"][i].sum(), 1e-6)) if bp_ok else None
            title = (f"region {k + 1} pixel {n} [{case}] | {reg['date']} ({y0 + yy},{x0 + xx}) | BP: {STATUS[int(P['tol'][i])]} | "
                     f"{P['ratio'][i]:.1f}x Bright" + (f" | rel. DEM difference ours vs BP {relv:.2f}" if relv is not None else ""))
            fig = pixel_figure(plt, P, i, logT, title)
            pdf_r.savefig(fig); plt.close(fig)
            pages.append((case, k + 1, n, P, i, title))
            rows.append({"n": n, "case": case, "pixel": [y0 + yy, x0 + xx], "status": STATUS[int(P["tol"][i])],
                         "ratio": float(P["ratio"][i]), "rel_dem_diff": relv,
                         "worst_miss_ours": float(np.abs(P["miss"][i]).max())})
        np.savez_compressed(os.path.join(args.out, f"viewer_ar{k + 1:02d}.npz"),
                            shape=np.array([args.box, args.box]), idx=np.flatnonzero(valid).astype(np.int32),
                            cat=cats["ours"].astype(np.int8), tol=P["tol"].astype(np.int8),
                            ours=P["ours"].astype(np.float32), bp=P["bp"].astype(np.float32),
                            sup=(P["sup"] if P["sup"] is not None else np.zeros_like(P["ours"])).astype(np.float32),
                            obs=P["obs"].astype(np.float32), err=P["err"].astype(np.float32),
                            level=P["level"].astype(np.float32), miss=P["miss"].astype(np.float32),
                            miss_bp=((P["rec_bp"] - P["obs"]) / h).astype(np.float32), logT=logT)
        summary.append(reg | {"region": k + 1, "box": [y0, x0, args.box], "inside_tight_pct": frac,
                              "status_counts": {STATUS[s_]: int((P["tol"] == s_).sum()) for s_ in STATUS}, "pixels": rows})
        print(f"region {k + 1} {reg['date']}: {len(picks)} pixels, inside ours {frac['ours']:.1f}%", flush=True)
        # keep memory down: only what the case pages need
        for key in list(P):
            if key not in ("obs", "err", "level", "bp", "ours", "sup", "rec_bp", "rec_ours", "rec_sup"):
                P.pop(key)
        for key in list(P):
            if P[key] is not None:
                P[key] = P[key][[i for _, i in picks]]
        pages[-len(picks):] = [(c, r, n, P, j, ti) for j, (c, r, n, _, _, ti) in enumerate(pages[-len(picks):])]
    pdf_r.close()
    with PdfPages(os.path.join(args.out, "by_case.pdf")) as pdf:
        for case in CASES:
            for c, r, n, P, i, title in pages:
                if c == case:
                    fig = pixel_figure(plt, P, i, logT, title)
                    pdf.savefig(fig); plt.close(fig)
    names = ("inside", "<=1.5", "<=3", ">3")
    totals = {m: {STATUS[s_]: dict(zip(names, tot[m][j].tolist())) for j, s_ in enumerate((0, 1, 3, 5))} for m in tot
              if tot[m].sum()}
    json.dump({"regions": summary, "totals": totals, "candidates": len(cands), "settings": vars(args)},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
