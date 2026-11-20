#!/usr/bin/env python3
"""Pixel-by-pixel look at where the network's DEM differs from BP's, at flare cores BP solves
at the normal band.

Every pixel of our staged split at or above --min_inten times the Bright threshold (default 32:
flare cores) whose solver flag is 1 (solved at the tight band, so one LP and one optimum) is
scored on a few measures:

  rel_err     sum over bins of (DEM - BP DEM)^2 / sum of BP DEM^2
  miss_c      (reconstruction - reading) / half-width per channel; inside the band is |miss| <= 1
  worst       max_c |miss_c|, and which channel
  emission    sum of the DEM over bins, relative to BP's
  hot         DEM mass at logT >= 6.8, relative to BP's hot mass (and BP's hot share)
  peak        logT of the DEM maximum minus BP's, and the number of peaks

and sorted into one type each, in this order:

  good             rel_err < 0.05
  tie              inside every band (|miss| <= 1.02, the solver's own DEMs are stored 12-bit) and
                   no more emission than BP (<= 1.05x): as good an answer to BP's problem
  fits_more        inside every band but more emission than BP: fits, but not the cheapest
  wrong_peak       outside a band, peak off by >= 0.2 dex or more peaks than BP
  smeared          outside a band, total emission within 10% of BP's but peak under 0.8x BP's
  too_little       outside a band, emission under 0.9x BP's
  too_much         outside a band, emission over 1.1x BP's
  other            outside a band, none of the above

Three flags cut across the types: missing_hot (BP has >= 5% of its DEM at logT >= 6.8 and we
have under half of it), extreme (top 1% brightness among these pixels) and lookalike (another
pixel with readings within 4.7% in every channel whose BP DEM differs by > 25%).
For each type: size, share of the network's and the supervised model's squared DEM error, and
figures (DEM against temperature; reconstruction in band half-widths per channel) for the
pixel at the type's median error and its worst two.

    python3 experiments/flare_pixel_analysis.py --model <ckpt> --supervised <pth> --out <dir>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N_BINS = 18
CHANNELS = (94, 131, 171, 193, 211, 335)
BRIGHT = np.array([3.1212, 16.9986, 448.8948, 498.3444, 190.7388, 10.3326])
TYPES = ("good", "tie", "fits_more", "smeared", "too_little", "too_much", "wrong_peak", "other")
LABELS = {"good": "matches BP", "tie": "inside every band, no more emission than BP",
          "fits_more": "inside every band, more emission than BP",
          "smeared": "outside a band, same total but a lower, broader peak",
          "too_little": "outside a band, too little emission", "too_much": "outside a band, too much emission",
          "wrong_peak": "outside a band, peak shifted or extra peak", "other": "outside a band, other"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=os.path.join(os.environ.get("SCRATCH", ""), "dem/data/lp_AIA_hofdeconv_full_DS"))
    p.add_argument("--phase", default="test")
    p.add_argument("--model", required=True)
    p.add_argument("--supervised", default=None)
    p.add_argument("--min_inten", type=float, default=32.0)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--n_show", type=int, default=2, help="worst pixels shown per type, besides the median one")
    p.add_argument("--out", required=True)
    return p.parse_args()


def collect(job):
    root, phase, start, stop, min_inten = job
    import zarr
    z = {k: zarr.open(os.path.join(root, f"{phase}_{k}.zarr"), mode="r") for k in "xemy"}
    out = {"obs": [], "err": [], "bp": [], "where": []}
    n_core = np.zeros(6, np.int64)            # flare-core pixels by solver flag 0..5
    for b in range(start, stop):
        x = np.asarray(z["x"][:, ::2, ::2, b], np.float32)
        inten = np.where(np.isfinite(x).all(0), (np.nan_to_num(x).clip(min=0) / BRIGHT[:, None, None]).max(0), 0)
        core = inten >= min_inten
        if not core.any():
            continue
        tol = np.asarray(z["m"][:, :, b])
        n_core += np.bincount(tol[core].astype(np.int64), minlength=6)[:6]
        sel = core & (tol == 1)
        if not sel.any():
            continue
        e = np.asarray(z["e"][:, ::2, ::2, b], np.float32)
        y = np.asarray(z["y"][:N_BINS, :, :, b], np.float32)
        ii, jj = np.nonzero(sel)
        ok = np.isfinite(y[:, ii, jj]).all(0) & (e[:, ii, jj] > 0).all(0)
        ii, jj = ii[ok], jj[ok]
        out["obs"].append(x[:, ii, jj].T); out["err"].append(e[:, ii, jj].T); out["bp"].append(y[:, ii, jj].T)
        out["where"].append(np.stack([np.full(len(ii), b), ii, jj], 1))
    return {k: (np.concatenate(v) if v else None) for k, v in out.items()}, n_core


def predict_ours(path, obs):
    import torch
    from experiments.train_scaled import load_operators
    from src.scaled_eval import load_scaled_model
    dev = torch.device("cpu")
    D_t, B_t, n_basis, logT = load_operators(dev)
    model, ck = load_scaled_model(path, n_basis, dev)
    k, c = ck["patch_size"], ck["patch_size"] // 2
    W = []
    with torch.no_grad():
        for s in range(0, len(obs), 4096):
            o = torch.as_tensor(obs[s:s + 4096])
            patch = torch.zeros(len(o), 6, k, k)
            patch[:, :, c, c] = o
            W.append(model(patch).double().numpy())
    W = np.concatenate(W)
    D, B = D_t.double().numpy(), B_t.double().numpy()[:N_BINS]
    return W @ B.T, W @ D.T, np.asarray(logT[:N_BINS], np.float64)


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


def n_peaks(dem):
    from src.scaled_eval import count_peaks
    return np.array([count_peaks(d)[0] for d in dem])


def lookalikes(obs, bp, tol=0.02, rel=0.25):
    from scipy.spatial import cKDTree
    z = np.log10(np.clip(obs, 1e-3, None))
    tree = cKDTree(z)
    pairs = tree.query_pairs(tol, p=np.inf, output_type="ndarray")
    flag = np.zeros(len(obs), bool)
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        d = ((bp[a] - bp[b]) ** 2).sum(1) / np.maximum(np.minimum((bp[a] ** 2).sum(1), (bp[b] ** 2).sum(1)), 1e-30)
        hit = d > rel
        flag[a[hit]] = True
        flag[b[hit]] = True
    return flag, int(len(pairs))


def figure(path, i, P, logT, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.plot(logT, P["bp"][i], "k-", lw=2, label="BP")
    a1.plot(logT, P["ours"][i], "-", color="tab:blue", lw=1.6, label="ours")
    if P.get("sup") is not None:
        a1.plot(logT, P["sup"][i], "--", color="tab:orange", lw=1.3, label="supervised")
    a1.set(xlabel="logT", ylabel="DEM", title="DEM")
    a1.legend(fontsize=8, frameon=False)
    x = np.arange(6)
    a2.axhspan(-1, 1, color="0.88", zorder=0, label="band")
    a2.axhline(0, color="0.6", lw=0.8)
    w = 0.26
    a2.bar(x - w, P["miss_bp"][i], w, color="k", label="BP")
    a2.bar(x, P["miss"][i], w, color="tab:blue", label="ours")
    if P.get("miss_sup") is not None:
        a2.bar(x + w, P["miss_sup"][i], w, color="tab:orange", label="supervised")
    a2.set_xticks(x, [f"{c}" for c in CHANNELS])
    lim = max(3.0, float(np.nanmax(np.abs(np.concatenate([P["miss"][i], P["miss_bp"][i]])))) * 1.1)
    a2.set_ylim(-min(lim, 30), min(lim, 30))
    a2.set(xlabel="channel (A)", ylabel="(reconstruction - reading) / half-width", title="resynthesis vs band")
    a2.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    import zarr
    os.makedirs(args.out, exist_ok=True)
    nb = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1]
    if args.max_blocks:
        nb = min(nb, args.max_blocks)
    stamps = open(os.path.join(args.root, f"{args.phase}_datetimes.txt")).read().split()
    per_image = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1] // len(stamps)
    step = max(1, nb // (args.workers * 8))
    jobs = [(args.root, args.phase, s, min(s + step, nb), args.min_inten) for s in range(0, nb, step)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(collect, jobs)
    n_core = sum(p[1] for p in parts)
    C = {k: np.concatenate([p[0][k] for p in parts if p[0][k] is not None]) for k in ("obs", "err", "bp", "where")}
    obs, err, bp = C["obs"].astype(np.float64), C["err"].astype(np.float64), C["bp"].astype(np.float64)
    n = len(obs)
    print(f"flare cores (>= {args.min_inten:g}x Bright) by solver flag 0..5: {n_core.tolist()}; "
          f"solved at the tight band and scored: {n:,}", flush=True)

    ours, rec, logT = predict_ours(args.model, obs.astype(np.float32))
    R = np.load(os.path.join(ROOT, "RData.npz"))["R"][:, :N_BINS].astype(np.float64) * 1e26
    h = args.tolfac * err
    P = {"bp": bp, "ours": ours, "miss": (rec - obs) / h, "miss_bp": (bp @ R.T - obs) / h}
    sup = None
    if args.supervised:
        sup = predict_supervised(args.supervised, obs.astype(np.float32))
        P["sup"], P["miss_sup"] = sup, (sup @ R.T - obs) / h

    sse = ((ours - bp) ** 2).sum(1)
    norm = np.maximum((bp ** 2).sum(1), 1e-30)
    rel = sse / norm
    worst = np.abs(P["miss"]).max(1)
    worst_ch = np.abs(P["miss"]).argmax(1)
    emis = ours.sum(1) / np.maximum(bp.sum(1), 1e-30)
    hot = logT >= 6.8
    hot_bp = bp[:, hot].sum(1)
    hot_share = hot_bp / np.maximum(bp.sum(1), 1e-30)
    hot_ratio = ours[:, hot].sum(1) / np.maximum(hot_bp, 1e-30)
    shift = logT[ours.argmax(1)] - logT[bp.argmax(1)]
    pk_ours, pk_bp = n_peaks(ours), n_peaks(bp)
    inten = (np.clip(obs, 0, None) / BRIGHT).max(1)
    extreme = inten >= np.quantile(inten, 0.99)
    look, n_pairs = lookalikes(obs, bp)

    inside = worst <= 1.02
    t = np.full(n, "other", dtype=object)
    peak_ratio = ours.max(1) / np.maximum(bp.max(1), 1e-30)
    missing_hot = (hot_share >= 0.05) & (hot_ratio < 0.5)
    out_ = ~inside
    t[out_ & (emis > 1.1)] = "too_much"
    t[out_ & (emis < 0.9)] = "too_little"
    t[out_ & (np.abs(emis - 1) <= 0.1) & (peak_ratio < 0.8)] = "smeared"
    t[out_ & ((np.abs(shift) >= 0.2) | (pk_ours > pk_bp))] = "wrong_peak"
    t[inside & (emis > 1.05)] = "fits_more"
    t[inside & (emis <= 1.05)] = "tie"
    t[rel < 0.05] = "good"

    tot = sse.sum()
    sse_sup = ((sup - bp) ** 2).sum(1) if sup is not None else None
    rows = []
    for ty in TYPES:
        m = t == ty
        if not m.any():
            rows.append({"type": ty, "n": 0})
            continue
        ch = np.bincount(worst_ch[m], minlength=6)
        rows.append({"type": ty, "label": LABELS[ty], "n": int(m.sum()), "share_pct": 100 * m.mean(),
                     "error_share_pct": 100 * sse[m].sum() / tot,
                     "sup_error_share_pct": None if sse_sup is None else 100 * sse_sup[m].sum() / sse_sup.sum(),
                     "median_rel_err": float(np.median(rel[m])),
                     "median_rel_err_sup": None if sup is None else float(np.median((sse_sup / norm)[m])),
                     "median_worst_miss": float(np.median(worst[m])),
                     "worst_channel_counts": {str(CHANNELS[c]): int(ch[c]) for c in range(6) if ch[c]},
                     "median_emission_ratio": float(np.median(emis[m])),
                     "median_peak_ratio": float(np.median(peak_ratio[m])),
                     "too_dim_pct_of_worst": 100 * float((np.take_along_axis(P["miss"][m], worst_ch[m][:, None], 1) < 0).mean()),
                     "extreme_pct": 100 * extreme[m].mean(), "lookalike_pct": 100 * look[m].mean()})
    flags = {f: {"n": int(v.sum()), "error_share_pct": 100 * sse[v].sum() / tot,
                 "types": {ty: int((v & (t == ty)).sum()) for ty in TYPES}}
             for f, v in (("missing_hot", missing_hot), ("extreme", extreme), ("lookalike", look))}

    shown = {}
    for ty in TYPES:
        idx = np.flatnonzero(t == ty)
        if not len(idx):
            continue
        order = idx[np.argsort(sse[idx])]
        pick = [("median", order[len(order) // 2])] + [(f"worst{k + 1}", order[-1 - k]) for k in range(min(args.n_show, len(order) - 1))]
        shown[ty] = []
        for tag, i in pick:
            b, ii, jj = (int(v) for v in C["where"][i])
            name = f"{ty}_{tag}.png"
            title = (f"{LABELS[ty]} ({tag}) | {stamps[b // per_image]} block {b} px ({ii},{jj}) | "
                     f"{inten[i]:.0f}x Bright | rel. DEM error {rel[i]:.2f} | worst miss {worst[i]:.1f} at "
                     f"{CHANNELS[worst_ch[i]]} A | emission {emis[i]:.2f}x BP | peak {peak_ratio[i]:.2f}x BP")
            figure(os.path.join(args.out, name), i, P, logT, title)
            shown[ty].append({"file": name, "image": stamps[b // per_image], "block": b, "pixel": [ii, jj],
                              "inten": float(inten[i]), "rel_err": float(rel[i]), "worst_miss": float(worst[i]),
                              "worst_channel": CHANNELS[worst_ch[i]], "emission_ratio": float(emis[i]),
                              "hot_ratio": float(hot_ratio[i]), "peak_ratio": float(peak_ratio[i]),
                              "peak_shift": float(shift[i])})

    summary = {"settings": {k: v for k, v in vars(args).items()}, "flare_cores_by_solver_flag": n_core.tolist(),
               "scored": n, "lookalike_pairs_within_2pct": n_pairs, "types": rows, "flags": flags, "shown": shown,
               "overall": {"median_rel_err": float(np.median(rel)),
                           "median_rel_err_sup": None if sup is None else float(np.median(sse_sup / norm)),
                           "inside_every_band_pct": 100 * inside.mean(),
                           "inside_every_band_pct_sup": None if sup is None else
                           100 * (np.abs(P["miss_sup"]).max(1) <= 1.02).mean(),
                           "median_emission_ratio": float(np.median(emis))}}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "pixels.npz"), where=C["where"], obs=obs, err=err, bp=bp, ours=ours,
                        sup=sup if sup is not None else np.zeros(0), type=t.astype(str), rel=rel, worst=worst,
                        emis=emis, hot_ratio=hot_ratio, peak_ratio=peak_ratio, shift=shift, extreme=extreme,
                        lookalike=look, missing_hot=missing_hot, inten=inten)
    L = [f"flare cores (>= {args.min_inten:g}x Bright), solved by BP at the tight band: {n:,} "
         f"(by solver flag 0/1/3/5: {n_core[0]}/{n_core[1]}/{n_core[3]}/{n_core[5]})",
         f"inside every band: ours {summary['overall']['inside_every_band_pct']:.1f}%"
         + ("" if sup is None else f", supervised {summary['overall']['inside_every_band_pct_sup']:.1f}%"),
         f"median relative DEM error: ours {summary['overall']['median_rel_err']:.3f}"
         + ("" if sup is None else f", supervised {summary['overall']['median_rel_err_sup']:.3f}"), "",
         f"{'type':12s} {'pixels':>7s} {'share':>6s} {'our err':>8s} {'sup err':>8s} {'med rel':>8s} {'worst miss':>10s} {'emission':>9s} {'peak':>6s} {'dim%':>5s}  worst channel"]
    for r in rows:
        if not r["n"]:
            continue
        L.append(f"{r['type']:12s} {r['n']:7d} {r['share_pct']:5.1f}% {r['error_share_pct']:7.1f}% "
                 f"{(r['sup_error_share_pct'] or 0):7.1f}% {r['median_rel_err']:8.3f} {r['median_worst_miss']:10.2f} "
                 f"{r['median_emission_ratio']:9.2f} {r['median_peak_ratio']:6.2f} {r['too_dim_pct_of_worst']:5.0f}  {r['worst_channel_counts']}")
    L += ["", f"missing_hot: {flags['missing_hot']}", f"extreme (top 1% brightness): {flags['extreme']}",
          f"lookalike: {flags['lookalike']}"]
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
