#!/usr/bin/env python3
"""Which pixels do the large models blow up on, and why?

Follow-up to the 21 Oct meeting. The 680-wide BP model on the stiff band and
both 10M models (1,800 wide) fail on a handful of flare-core pixels; the
ElasticNet 722k and 1.43M log1p models fail on a few bright pixels; the
336-wide models do not. One pass over the shared test set with every model of
a track answers, per model:

1. Tail concentration: total DEM SSE against the solver, the share held by the
   worst 0.01%, 0.1% and 1% of pixels (from a fine log histogram) and the exact
   share held by the worst 100, 1,000 and 5,000 pixels.
2. Whether the models fail on the same pixels: Jaccard overlap of the worst
   sets, keyed by (block, i, j) and by the AIA input itself (the five solver
   targets repeat the clean AIA, so one input can appear up to five times).
3. What the worst pixels are: brightness band, deconvolution-clamp incidence
   (a channel at or below MIN_OBS), channel ratios against random pixels of
   the same band, which DEM bins carry the excess, whether the solver's DEM is
   multi-peaked there, the 171 and 94 A reconstruction ratios, and the other
   models' error on the same pixels.
4. Whether they sit where training data is sparse: the distance in
   standardised log-input space to the nearest of a stratified sample of
   training pixels (every pixel above 10x the Bright cutoff is kept, fainter
   bands subsampled) and per-channel extrapolation flags, against random test
   pixels of the same band.

Evaluation only; no training. Resumable: --start/--stop with --part write a
part file, --merge combines the parts and runs the analysis.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from experiments.diagnose_bright_failures import (CHANNELS, INTENSITY_EDGES, INTENSITY_LABELS,  # noqa: E402
                                                  N_BINS, interior_peak_counts, load_predictor,
                                                  split_ranges)
from src.zarr_data import MIN_OBS  # noqa: E402

SHARED = "/projects/rps/dff6142/fouheylab/solar_dem/{track}_AIA_hofdeconv_full_DS"
TRAIN_ROOT = os.path.join(os.environ.get("SCRATCH", ""), "dem/data/lp_AIA_hofdeconv_full_DS")
MODELS = {
    "bp": OrderedDict([
        ("h336_mu100_cont", "output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt"),
        ("h336_mu100_ep120", "output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120.pt"),
        ("h680_mu100_ep120", "output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_mu100_ep120.pt"),
        ("h1800_mu100", "output/experiments/followup/scaled_mlp6_barrier_h1800_sqrt_ff12_resample_mu100.pt"),
        ("h1800_mu1", "output/experiments/followup/scaled_mlp6_barrier_h1800_sqrt_ff12_resample.pt"),
        ("h336_mu1", "output/experiments/input_encoding/scaled_mlp6_barrier_h336_sqrt_ff12_resample.pt"),
    ]),
    "enet": OrderedDict([
        ("h336_sqrtff", "output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt"),
        ("h1800_sqrtff", "output/experiments/followup/scaled_mlp6_enet_h1800_sqrt_ff12_resample.pt"),
        ("h336_log1p", "output/experiments/resample/scaled_mlp6_enet_h336_resample.pt"),
        ("h480_log1p", "output/experiments/resample/scaled_mlp6_enet_h480_resample.pt"),
        ("h680_log1p", "output/experiments/resample/scaled_mlp6_enet_resample.pt"),
    ]),
}
CACHE = 320                      # one timestamp: 64 blocks x 5 targets
HIST_MIN, HIST_MAX, HIST_W = -12.0, 12.0, 0.02
N_HIST = int(round((HIST_MAX - HIST_MIN) / HIST_W))
# Training-sample keep probability per brightness band: everything at 10x the
# Bright cutoff and above, fainter bands thinned so ~3,000 blocks give ~1.5M.
KEEP_TRAIN = np.array([0.02, 0.02, 0.02, 0.02, 0.05, 0.1, 0.3, 1.0, 1.0])
RATIO_CHANNELS = [0, 1, 3, 4, 5]          # 94, 131, 193, 211, 335 over 171 A
N_BANDS = len(INTENSITY_LABELS)
TOP_SETS = (100, 1000, 5000)
W_FIELDS = ["block", "i", "j", "band", "clamp", "sse", "sse_all", "r171_all", "r94_all", "obs", "ref", "pred"]
C_FIELDS = ["block", "i", "j", "band", "clamp", "sse_all", "r171_all", "r94_all", "obs", "ref"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--track", choices=("bp", "enet"), required=True)
    p.add_argument("--models", nargs="*", help="name=checkpoint ... (default: the track's list above)")
    p.add_argument("--data", help="shared test root (default: the track's)")
    p.add_argument("--train-root", default=TRAIN_ROOT, help="our staged split with train_x.zarr")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--output", required=True, help="JSON path; .txt, .npz and part files share the stem")
    p.add_argument("--max-blocks", type=int)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--part", action="store_true", help="write a part file for this block range and stop")
    p.add_argument("--merge", action="store_true", help="combine part files and run the analysis")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--pixel-batch", type=int, default=16384)
    p.add_argument("--topk", type=int, default=5000)
    p.add_argument("--controls", type=int, default=1000, help="random valid pixels kept per band")
    p.add_argument("--train-blocks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-missing", action="store_true")
    return p.parse_args()


# ── pass over the test set ────────────────────────────────────────────────────

def band_of(obs, thr):
    """Brightness band 0..8 of [6, ...] observations (aia_fit_by_brightness convention)."""
    bright = (obs >= thr).any(axis=0)
    ratio = np.where(np.isfinite(obs), obs / thr, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log10(ratio.max(axis=0))
    logr = np.where(np.isfinite(logr), logr, -np.inf)
    band = np.searchsorted(INTENSITY_EDGES, logr, side="right")
    return np.where(bright, np.maximum(band, 4), np.minimum(band, 3)).astype(np.int8)


class TopK:
    """Running top-k by score: append candidates, prune with argpartition."""

    def __init__(self, k, fields):
        self.k, self.fields = k, fields
        self.buf = {f: [] for f in fields}
        self.n, self.floor = 0, -np.inf

    def add(self, rows):
        m = len(rows["sse"])
        if m == 0:
            return
        for f in self.fields:
            self.buf[f].append(np.asarray(rows[f]))
        self.n += m
        if self.n > 4 * self.k:
            self.prune()

    def prune(self):
        if self.n == 0:
            return
        cat = {f: np.concatenate(self.buf[f]) for f in self.fields}
        if self.n > self.k:
            sel = np.argpartition(-cat["sse"], self.k - 1)[:self.k]
            cat = {f: v[sel] for f, v in cat.items()}
        self.buf = {f: [v] for f, v in cat.items()}
        self.n = len(cat["sse"])
        if self.n >= self.k:
            self.floor = float(cat["sse"].min())

    def arrays(self):
        self.prune()
        return None if self.n == 0 else {f: self.buf[f][0] for f in self.fields}


class Reservoir:
    """Per-band uniform sample of size k (algorithm R, vectorised per block)."""

    def __init__(self, k, n_groups, fields, rng):
        self.k, self.fields, self.rng = k, fields, rng
        self.seen = np.zeros(n_groups, np.int64)
        self.store = [None] * n_groups

    def offer(self, g, m):
        """-> (positions in the batch to keep, their slots)."""
        t = self.seen[g] + np.arange(1, m + 1)
        slot = np.floor(self.rng.random(m) * t).astype(np.int64)
        filling = t <= self.k
        slot = np.where(filling, t - 1, slot)
        take = np.flatnonzero(filling | (slot < self.k))
        self.seen[g] += m
        return take, slot[take]

    def write(self, g, slots, rows):
        if self.store[g] is None:
            self.store[g] = {f: np.zeros((self.k,) + np.asarray(v).shape[1:], dtype=np.asarray(v).dtype)
                             for f, v in rows.items()}
        for f, v in rows.items():
            self.store[g][f][slots] = v

    def filled(self, g):
        return int(min(self.k, self.seen[g]))


def run_range(job):
    cfg, start, stop, wid = job
    import zarr
    preds = OrderedDict()
    for name, path in cfg["models"].items():
        preds[name], _ = load_predictor(path, cfg["threads"])
    names = list(preds)
    M = len(names)
    x = zarr.open(os.path.join(cfg["data"], "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(cfg["data"], "test_y.zarr"), mode="r")
    with np.load(os.path.join(ROOT, "RData.npz")) as rdata:
        response = np.asarray(rdata["R"][:, :N_BINS], dtype=np.float64) * 1e26
    thr = np.asarray(cfg["thresholds"], np.float32)[:, None, None]
    rng = np.random.default_rng((cfg["seed"], wid))
    top = [TopK(cfg["topk"], W_FIELDS) for _ in names]
    ctrl = Reservoir(cfg["controls"], N_BANDS, C_FIELDS, rng)
    total = np.zeros(M)
    n_valid = np.zeros(M, np.int64)
    n_nonfinite = np.zeros(M, np.int64)
    hist = np.zeros((M, N_HIST + 2), np.int64)        # [zero SSE, bins, overflow]
    hist_sse = np.zeros((M, N_HIST + 2))
    band_n = np.zeros(N_BANDS, np.int64)
    cache = OrderedDict()
    t0 = time.time()
    for b in range(start, stop):
        xb = np.ascontiguousarray(np.asarray(x[:, :, :, b], dtype=np.float32))
        obs = np.ascontiguousarray(xb[:, ::2, ::2])
        key = hashlib.blake2b(obs.tobytes(), digest_size=16).hexdigest()
        if key in cache:
            pred = cache.pop(key)
        else:
            pred = [p(obs, cfg["pixel_batch"]) for p in preds.values()]
        cache[key] = pred
        if len(cache) > CACHE:
            cache.popitem(last=False)
        ref = np.asarray(y[:N_BINS, :, :, b], dtype=np.float32)
        H, W = ref.shape[1:]
        P = H * W
        o = obs.reshape(6, P).astype(np.float64)
        r = ref.reshape(N_BINS, P).astype(np.float64)
        base = np.isfinite(r).all(0) & np.isfinite(o).all(0)
        band = band_of(obs, thr).reshape(P)
        clamp = np.zeros(P, np.uint8)
        for c in range(6):
            clamp |= ((o[c] <= MIN_OBS).astype(np.uint8) << c)
        ii, jj = np.divmod(np.arange(P), W)
        sse_all = np.full((M, P), np.nan)
        r171 = np.full((M, P), np.nan)
        r94 = np.full((M, P), np.nan)
        pr = []
        for m in range(M):
            pm = pred[m].reshape(N_BINS, P).astype(np.float64)
            pr.append(pm)
            fin = np.isfinite(pm).all(0)
            valid = base & fin
            n_nonfinite[m] += int((base & ~fin).sum())
            d = np.where(valid, pm - r, 0.0)
            s = np.where(valid, np.einsum("tp,tp->p", d, d), np.nan)
            sse_all[m] = s
            rec = response @ np.where(np.isfinite(pm), pm, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r171[m] = np.where(valid & (o[2] > 0), rec[2] / o[2], np.nan)
                r94[m] = np.where(valid & (o[0] > 0), rec[0] / o[0], np.nan)
            sv = s[valid]
            total[m] += sv.sum()
            n_valid[m] += sv.size
            pos = sv > 0
            hist[m, 0] += int((~pos).sum())
            lg = np.log10(sv[pos])
            k = np.clip(np.floor((lg - HIST_MIN) / HIST_W).astype(np.int64) + 1, 1, N_HIST + 1)
            np.add.at(hist[m], k, 1)
            np.add.at(hist_sse[m], k, sv[pos])
        band_n += np.bincount(band[base], minlength=N_BANDS)
        for m in range(M):
            s = sse_all[m]
            cand = np.flatnonzero(np.isfinite(s) & (s > top[m].floor))
            if cand.size == 0:
                continue
            top[m].add({"block": np.full(cand.size, b, np.int32), "i": ii[cand].astype(np.int16),
                        "j": jj[cand].astype(np.int16), "band": band[cand], "clamp": clamp[cand],
                        "sse": s[cand], "sse_all": sse_all[:, cand].T.copy(),
                        "r171_all": r171[:, cand].T.copy(), "r94_all": r94[:, cand].T.copy(),
                        "obs": o[:, cand].T.astype(np.float32), "ref": r[:, cand].T.astype(np.float32),
                        "pred": pr[m][:, cand].T.astype(np.float32)})
        allv = base & np.isfinite(sse_all).all(0)
        for g in range(N_BANDS):
            idx = np.flatnonzero(allv & (band == g))
            if idx.size == 0:
                continue
            take, slots = ctrl.offer(g, idx.size)
            if take.size == 0:
                continue
            q = idx[take]
            ctrl.write(g, slots, {"block": np.full(q.size, b, np.int32), "i": ii[q].astype(np.int16),
                                  "j": jj[q].astype(np.int16), "band": band[q], "clamp": clamp[q],
                                  "sse_all": sse_all[:, q].T.copy(), "r171_all": r171[:, q].T.copy(),
                                  "r94_all": r94[:, q].T.copy(), "obs": o[:, q].T.astype(np.float32),
                                  "ref": r[:, q].T.astype(np.float32)})
        done = b - start + 1
        if done % 250 == 0 or b + 1 == stop:
            print(f"[w{wid}] {done:,}/{stop - start:,} blocks  {done / (time.time() - t0):.2f} blocks/s",
                  flush=True)
    return {"names": names, "total": total, "n_valid": n_valid, "n_nonfinite": n_nonfinite,
            "hist": hist, "hist_sse": hist_sse, "band_n": band_n,
            "top": [t.arrays() for t in top],
            "ctrl": [(ctrl.store[g], int(ctrl.seen[g]), ctrl.filled(g)) for g in range(N_BANDS)],
            "range": (start, stop)}


def merge_parts(parts, topk, controls, seed):
    names = parts[0]["names"]
    M = len(names)
    out = {"names": names, "total": sum(p["total"] for p in parts),
           "n_valid": sum(p["n_valid"] for p in parts), "n_nonfinite": sum(p["n_nonfinite"] for p in parts),
           "hist": sum(p["hist"] for p in parts), "hist_sse": sum(p["hist_sse"] for p in parts),
           "band_n": sum(p["band_n"] for p in parts),
           "ranges": sorted(p["range"] for p in parts)}
    worst = []
    for m in range(M):
        pieces = [p["top"][m] for p in parts if p["top"][m] is not None]
        if not pieces:
            worst.append(None)
            continue
        cat = {f: np.concatenate([q[f] for q in pieces]) for f in W_FIELDS}
        order = np.argsort(-cat["sse"], kind="stable")[:topk]
        worst.append({f: v[order] for f, v in cat.items()})
    out["worst"] = worst
    rng = np.random.default_rng(seed + 1)
    ctrl = []
    for g in range(N_BANDS):
        pool, weights = [], []
        for p in parts:
            store, seen, filled = p["ctrl"][g]
            if store is None or filled == 0:
                continue
            pool.append({f: store[f][:filled] for f in C_FIELDS})
            weights.append(np.full(filled, seen / filled))
        if not pool:
            ctrl.append(None)
            continue
        cat = {f: np.concatenate([q[f] for q in pool]) for f in C_FIELDS}
        w = np.concatenate(weights)
        n = min(controls, len(w))
        keep = np.sort(rng.choice(len(w), size=n, replace=False, p=w / w.sum()))
        ctrl.append({f: v[keep] for f, v in cat.items()})
    out["ctrl"] = ctrl
    return out


# ── analysis ──────────────────────────────────────────────────────────────────

def tail_shares(hist, hist_sse, n_valid, total):
    """Share of total SSE held by the worst fraction of pixels, from the log histogram."""
    res = {}
    for frac in (1e-4, 1e-3, 1e-2):
        want = frac * n_valid
        got, sse = 0.0, 0.0
        for k in range(hist.shape[0] - 1, 0, -1):
            if got + hist[k] >= want:
                sse += hist_sse[k] * (want - got) / max(hist[k], 1)
                break
            got += hist[k]
            sse += hist_sse[k]
        res[f"worst_{frac:g}"] = float(sse / total) if total > 0 else None
    return res


def input_keys(obs):
    return np.array([hashlib.blake2b(np.ascontiguousarray(row, dtype=np.float32).tobytes(),
                                     digest_size=8).hexdigest() for row in obs])


def distinct_prefix(keys, n):
    """Indices of the first n rows with distinct input keys (rows already sorted)."""
    seen, idx = set(), []
    for t, k in enumerate(keys):
        if k in seen:
            continue
        seen.add(k)
        idx.append(t)
        if len(idx) == n:
            break
    return np.asarray(idx, dtype=np.int64)


def quantiles(v, qs=(10, 50, 90)):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


def log_ratios(obs):
    """log10 of each channel over 171 A, observations floored at MIN_OBS; -> [P, 5]."""
    o = np.maximum(np.asarray(obs, dtype=np.float64), MIN_OBS)
    return np.log10(o[:, RATIO_CHANNELS] / o[:, 2:3])


def characterise(w, sel, names, m, ctrl_ratios, logt):
    """Describe one model's worst pixels `sel` (indices into its worst arrays)."""
    obs, ref, pred = w["obs"][sel].astype(np.float64), w["ref"][sel].astype(np.float64), w["pred"][sel].astype(np.float64)
    d = pred - ref
    sse_bin = (d ** 2).sum(0)
    band = w["band"][sel]
    clamp = w["clamp"][sel]
    ref_peaks = interior_peak_counts(ref)
    pred_peaks = interior_peak_counts(pred)
    em_ref, em_pred = ref.sum(1), pred.sum(1)
    peak_ratio = pred.max(1) / np.maximum(ref.max(1), 1e-30)
    lr = log_ratios(obs)
    ratio_tab = {}
    for g in np.unique(band):
        sub = lr[band == g]
        if sub.shape[0] < 10:
            continue
        ratio_tab[INTENSITY_LABELS[g]] = {
            "n": int(sub.shape[0]),
            "worst": {CHANNELS[c]: quantiles(sub[:, k]) for k, c in enumerate(RATIO_CHANNELS)},
            "controls": ctrl_ratios.get(INTENSITY_LABELS[g])}
    others = {}
    own = np.median(w["sse"][sel])
    for k, n in enumerate(names):
        s = w["sse_all"][sel][:, k]
        others[n] = {"median_sse": float(np.nanmedian(s)), "ratio_to_own_median": float(np.nanmedian(s) / own) if own > 0 else None,
                     "share_above_1pct_of_own": float(np.nanmean(s > 0.01 * w["sse"][sel]))}
    excess_hot = sse_bin[logt >= 6.55].sum() / max(sse_bin.sum(), 1e-30)
    low_ref = ref < 0.01 * ref.max(1, keepdims=True)
    return {
        "n": int(sel.size),
        "sse_sum": float(w["sse"][sel].sum()),
        "band_counts": {INTENSITY_LABELS[g]: int((band == g).sum()) for g in range(N_BANDS) if (band == g).any()},
        "clamp_any_pct": float(100 * (clamp > 0).mean()),
        "clamp_by_channel_pct": {CHANNELS[c]: float(100 * ((clamp >> c) & 1).mean()) for c in range(6)},
        "obs_quantiles": {CHANNELS[c]: quantiles(obs[:, c]) for c in range(6)},
        "channel_ratios_log10_over_171": ratio_tab,
        "excess_mean_by_bin": d.mean(0).tolist(),
        "mean_abs_err_by_bin": np.abs(d).mean(0).tolist(),
        "sse_share_by_bin": (sse_bin / max(sse_bin.sum(), 1e-30)).tolist(),
        "sse_share_logt_ge_6.6": float(excess_hot),
        "sse_share_where_solver_below_1pct_of_its_peak": float((d ** 2)[low_ref].sum() / max(sse_bin.sum(), 1e-30)),
        "solver_multipeaked_pct": float(100 * (ref_peaks >= 2).mean()),
        "model_multipeaked_pct": float(100 * (pred_peaks >= 2).mean()),
        "solver_peak_bin_counts": np.bincount(ref.argmax(1), minlength=N_BINS).tolist(),
        "model_peak_bin_counts": np.bincount(pred.argmax(1), minlength=N_BINS).tolist(),
        "emission_ratio_pred_over_solver": quantiles(em_pred / np.maximum(em_ref, 1e-30)),
        "peak_ratio_pred_over_solver": quantiles(peak_ratio),
        "recon_171_over_obs": quantiles(w["r171_all"][sel][:, m]),
        "recon_94_over_obs": quantiles(w["r94_all"][sel][:, m]),
        "other_models_at_these_pixels": others,
    }


def read_train(job):
    """Stratified sample of training pixels: log10 observations and band."""
    cfg, blocks, wid = job
    import zarr
    x = zarr.open(os.path.join(cfg["train_root"], "train_x.zarr"), mode="r")
    thr = np.asarray(cfg["thresholds"], np.float32)[:, None, None]
    rng = np.random.default_rng((cfg["seed"], 100 + wid))
    feats, bands = [], []
    for b in blocks:
        obs = np.ascontiguousarray(np.asarray(x[:, :, :, b], dtype=np.float32)[:, ::2, ::2])
        valid = np.isfinite(obs).all(0) & (obs > MIN_OBS).all(0)
        band = band_of(obs, thr)
        keep = valid & (rng.random(valid.shape) < KEEP_TRAIN[band])
        if keep.any():
            feats.append(np.log10(obs[:, keep].T))
            bands.append(band[keep])
    if not feats:
        return np.zeros((0, 6), np.float32), np.zeros(0, np.int8)
    return np.concatenate(feats), np.concatenate(bands)


def nn_stats(d1, d10, band, ctrl_d1, ctrl_band):
    out = {}
    for g in np.unique(band):
        wsel = band == g
        if wsel.sum() < 10:
            continue
        c = ctrl_d1[ctrl_band == g]
        p99 = float(np.percentile(c, 99)) if c.size else None
        out[INTENSITY_LABELS[g]] = {
            "n": int(wsel.sum()),
            "worst_d1": quantiles(d1[wsel]), "worst_d10": quantiles(d10[wsel]),
            "controls_d1": quantiles(c) if c.size else None,
            "controls_d1_p99": p99,
            "worst_pct_beyond_controls_p99": float(100 * (d1[wsel] > p99).mean()) if p99 is not None else None}
    return out


def extrapolation(obs, tmin, tmax, tp):
    o = np.asarray(obs, dtype=np.float64)
    above = o > tmax[None, :]
    below = o < tmin[None, :]
    high = o > tp[None, :]
    return {"outside_training_range_pct": float(100 * (above | below).any(1).mean()),
            "above_training_max_by_channel_pct": {CHANNELS[c]: float(100 * above[:, c].mean()) for c in range(6)},
            "below_training_min_by_channel_pct": {CHANNELS[c]: float(100 * below[:, c].mean()) for c in range(6)},
            "above_training_p99.999_any_pct": float(100 * high.any(1).mean()),
            "above_training_p99.999_by_channel_pct": {CHANNELS[c]: float(100 * high[:, c].mean()) for c in range(6)}}


def analyse(res, cfg, args):
    from scipy.spatial import cKDTree
    names = res["names"]
    M = len(names)
    logt = np.load(os.path.join(ROOT, "RData.npz"))["logT"][:N_BINS]
    out = {"models": {}, "overlap": {}, "controls": {}}
    # control ratio quantiles per band
    ctrl_ratios = {}
    ctrl_obs, ctrl_band = [], []
    for g in range(N_BANDS):
        c = res["ctrl"][g]
        if c is None:
            continue
        lr = log_ratios(c["obs"])
        ctrl_ratios[INTENSITY_LABELS[g]] = {CHANNELS[ch]: quantiles(lr[:, k]) for k, ch in enumerate(RATIO_CHANNELS)}
        ctrl_obs.append(c["obs"])
        ctrl_band.append(c["band"])
    ctrl_obs = np.concatenate(ctrl_obs) if ctrl_obs else np.zeros((0, 6), np.float32)
    ctrl_band = np.concatenate(ctrl_band) if ctrl_band else np.zeros(0, np.int8)
    out["controls"]["n_per_band"] = {INTENSITY_LABELS[g]: (0 if res["ctrl"][g] is None else int(len(res["ctrl"][g]["band"])))
                                     for g in range(N_BANDS)}
    out["controls"]["clamp_any_pct_by_band"] = {INTENSITY_LABELS[g]: float(100 * (res["ctrl"][g]["clamp"] > 0).mean())
                                                for g in range(N_BANDS) if res["ctrl"][g] is not None}
    out["controls"]["median_sse_by_band"] = {
        INTENSITY_LABELS[g]: {n: float(np.nanmedian(res["ctrl"][g]["sse_all"][:, k])) for k, n in enumerate(names)}
        for g in range(N_BANDS) if res["ctrl"][g] is not None}

    # training sample and tree
    t0 = time.time()
    import zarr
    n_train = int(zarr.open(os.path.join(cfg["train_root"], "train_x.zarr"), mode="r").shape[-1])
    rng = np.random.default_rng(args.seed + 2)
    blocks = np.sort(rng.choice(n_train, size=min(args.train_blocks, n_train), replace=False))
    chunks = np.array_split(blocks, max(1, min(args.workers, len(blocks))))
    jobs = [(cfg, ch.tolist(), i) for i, ch in enumerate(chunks) if len(ch)]
    if len(jobs) == 1:
        tr = [read_train(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            tr = pool.map(read_train, jobs)
    feats = np.concatenate([t[0] for t in tr]).astype(np.float64)
    tband = np.concatenate([t[1] for t in tr])
    print(f"training sample: {feats.shape[0]:,} pixels from {len(blocks):,} blocks in {time.time() - t0:.0f}s", flush=True)
    mean, std = feats.mean(0), feats.std(0) + 1e-9
    tree = cKDTree((feats - mean) / std)
    tobs = 10 ** feats
    tmin, tmax = tobs.min(0), tobs.max(0)
    tp = np.percentile(tobs, 99.999, axis=0)
    out["training_sample"] = {
        "n_pixels": int(feats.shape[0]), "n_blocks": int(len(blocks)),
        "keep_probability_by_band": dict(zip(INTENSITY_LABELS, KEEP_TRAIN.tolist())),
        "n_by_band": {INTENSITY_LABELS[g]: int((tband == g).sum()) for g in range(N_BANDS)},
        "obs_min": tmin.tolist(), "obs_max": tmax.tolist(), "obs_p99.999": tp.tolist(),
        "log10_mean": mean.tolist(), "log10_std": std.tolist(),
        "feature": "log10(observed DN/s) per channel, standardised by the sample's mean and std"}

    def query(obs):
        z = (np.log10(np.maximum(np.asarray(obs, np.float64), MIN_OBS)) - mean) / std
        dist, _ = tree.query(z, k=10, workers=-1)
        return dist[:, 0], dist.mean(1)

    c_d1, c_d10 = query(ctrl_obs) if len(ctrl_obs) else (np.zeros(0), np.zeros(0))
    out["controls"]["nn_d1_by_band"] = {INTENSITY_LABELS[g]: quantiles(c_d1[ctrl_band == g])
                                        for g in range(N_BANDS) if (ctrl_band == g).any()}
    out["controls"]["extrapolation_by_band"] = {INTENSITY_LABELS[g]: extrapolation(ctrl_obs[ctrl_band == g], tmin, tmax, tp)
                                                for g in range(N_BANDS) if (ctrl_band == g).any()}
    arrays = {"names": np.array(names), "hist_edges": HIST_MIN + HIST_W * np.arange(N_HIST + 1),
              "hist": res["hist"], "hist_sse": res["hist_sse"], "band_n": res["band_n"],
              "controls_obs": ctrl_obs, "controls_band": ctrl_band, "controls_d1": c_d1, "controls_d10": c_d10,
              "train_obs_min": tmin, "train_obs_max": tmax, "train_obs_p99999": tp,
              "train_log10_mean": mean, "train_log10_std": std}
    for g in range(N_BANDS):
        if res["ctrl"][g] is not None:
            for f in ("sse_all", "r171_all", "r94_all", "ref", "clamp", "block", "i", "j"):
                arrays[f"controls_{INTENSITY_LABELS[g]}_{f}"] = res["ctrl"][g][f]

    keysets = {}
    for m, n in enumerate(names):
        w = res["worst"][m]
        entry = {"total_sse": float(res["total"][m]), "n_valid": int(res["n_valid"][m]),
                 "mse": float(res["total"][m] / max(res["n_valid"][m], 1)),
                 "n_nonfinite_predictions": int(res["n_nonfinite"][m]),
                 "n_zero_sse": int(res["hist"][m, 0]),
                 "tail_share_from_histogram": tail_shares(res["hist"][m], res["hist_sse"][m], res["n_valid"][m], res["total"][m]),
                 "pixels_with_sse_above": {f"1e{e}": int(res["hist"][m, int((e - HIST_MIN) / HIST_W) + 1:].sum())
                                           for e in (2, 4, 6, 8)}}
        if w is None:
            out["models"][n] = entry
            continue
        keys = input_keys(w["obs"])
        exact = {}
        for K in TOP_SETS:
            k = min(K, len(w["sse"]))
            exact[f"top{K}"] = {"k": k, "sse_share": float(w["sse"][:k].sum() / max(res["total"][m], 1e-30)),
                                "distinct_inputs": int(len(set(keys[:k].tolist()))),
                                "min_sse": float(w["sse"][k - 1])}
        entry["worst_sets"] = exact
        entry["worst_pixel"] = {"sse": float(w["sse"][0]), "block": int(w["block"][0]), "i": int(w["i"][0]),
                                "j": int(w["j"][0]), "band": INTENSITY_LABELS[int(w["band"][0])],
                                "obs": w["obs"][0].tolist(), "solver_dem": w["ref"][0].tolist(),
                                "pred_dem": w["pred"][0].tolist(),
                                "sse_of_each_model": dict(zip(names, w["sse_all"][0].tolist()))}
        for n_distinct in (100, 1000):
            sel = distinct_prefix(keys, n_distinct)
            ch = characterise(w, sel, names, m, ctrl_ratios, logt)
            d1, d10 = query(w["obs"][sel])
            ch["nn_distance_by_band"] = nn_stats(d1, d10, w["band"][sel], c_d1, ctrl_band)
            ch["extrapolation"] = extrapolation(w["obs"][sel], tmin, tmax, tp)
            entry[f"worst_{n_distinct}_distinct_inputs"] = ch
        d1_all, d10_all = query(w["obs"])
        for f in W_FIELDS:
            arrays[f"worst_{n}_{f}"] = w[f]
        arrays[f"worst_{n}_input_key"] = keys
        arrays[f"worst_{n}_d1"] = d1_all
        arrays[f"worst_{n}_d10"] = d10_all
        keysets[n] = {"pixel": {K: set(zip(w["block"][:K].tolist(), w["i"][:K].tolist(), w["j"][:K].tolist()))
                                for K in TOP_SETS},
                      "input": {K: set(keys[:K].tolist()) for K in TOP_SETS}}
        out["models"][n] = entry
    for how in ("pixel", "input"):
        out["overlap"][how] = {}
        for K in TOP_SETS:
            tab = {}
            for a in keysets:
                for b2 in keysets:
                    if a >= b2:
                        continue
                    A, B = keysets[a][how][K], keysets[b2][how][K]
                    tab[f"{a} vs {b2}"] = float(len(A & B) / max(len(A | B), 1))
            out["overlap"][how][f"top{K}_jaccard"] = tab
    out["overlap"]["note"] = ("pixel: keyed by (block, i, j); input: keyed by the six observed values, "
                              "which merges the repeats of one clean AIA input across solver targets")
    return out, arrays


def render(out):
    L = [f"# Blow-up census, {out['track'].upper()} track ({out['n_blocks']:,} blocks, "
         f"{'complete' if out['complete_test'] else 'PARTIAL'} test set)"]
    names = list(out["models"])
    L.append("\n## Tail concentration of DEM SSE")
    head = f"{'model':<18}{'MSE':>10}{'nonfinite':>10}{'w0.01%':>8}{'w0.1%':>8}{'w1%':>8}{'top100':>8}{'top1k':>8}{'top5k':>8}{'distinct1k':>11}"
    L += [head, "-" * len(head)]
    for n in names:
        e = out["models"][n]
        ts = e["tail_share_from_histogram"]
        ws = e.get("worst_sets", {})
        g = lambda k, f: (f"{ws[k][f]:>8.4f}" if k in ws else f"{'':>8}")
        L.append(f"{n:<18}{e['mse']:>10.4f}{e['n_nonfinite_predictions']:>10,}"
                 f"{ts['worst_0.0001']:>8.4f}{ts['worst_0.001']:>8.4f}{ts['worst_0.01']:>8.4f}"
                 f"{g('top100', 'sse_share')}{g('top1000', 'sse_share')}{g('top5000', 'sse_share')}"
                 f"{ws.get('top1000', {}).get('distinct_inputs', ''):>11}")
    L.append("\n## Overlap of the worst sets (Jaccard), keyed by input")
    for K in TOP_SETS:
        tab = out["overlap"]["input"].get(f"top{K}_jaccard", {})
        if tab:
            L.append(f"  top {K}: " + "; ".join(f"{k} {v:.2f}" for k, v in tab.items()))
    L.append("\n## The worst 1,000 distinct inputs per model")
    for n in names:
        c = out["models"][n].get("worst_1000_distinct_inputs")
        if not c:
            continue
        bands = ", ".join(f"{k} {v}" for k, v in c["band_counts"].items())
        L.append(f"\n### {n}")
        L.append(f"  bands: {bands}")
        L.append(f"  clamp (a channel <= {MIN_OBS:g}): {c['clamp_any_pct']:.1f}% of pixels; by channel "
                 + ", ".join(f"{k} {v:.1f}%" for k, v in c["clamp_by_channel_pct"].items()))
        L.append(f"  solver multi-peaked {c['solver_multipeaked_pct']:.1f}%, model multi-peaked {c['model_multipeaked_pct']:.1f}%; "
                 f"emission pred/solver median {c['emission_ratio_pred_over_solver']['p50']:.2f}, "
                 f"peak ratio median {c['peak_ratio_pred_over_solver']['p50']:.2f}")
        L.append(f"  SSE share in bins logT >= 6.6: {100 * c['sse_share_logt_ge_6.6']:.1f}%; "
                 f"where the solver is below 1% of its peak: {100 * c['sse_share_where_solver_below_1pct_of_its_peak']:.1f}%")
        r171, r94 = c["recon_171_over_obs"], c["recon_94_over_obs"]
        if r171 and r94:
            L.append(f"  reconstruction / observed: 171 A median {r171['p50']:.2f} (p90 {r171['p90']:.2f}); "
                     f"94 A median {r94['p50']:.2f} (p90 {r94['p90']:.2f})")
        L.append("  other models' median SSE at these pixels, relative to this model's: "
                 + ", ".join(f"{k} {v['ratio_to_own_median']:.3g}" for k, v in c["other_models_at_these_pixels"].items()
                             if v["ratio_to_own_median"] is not None))
        for band, st in c["nn_distance_by_band"].items():
            L.append(f"  nearest training pixel, {band} (n={st['n']}): worst median {st['worst_d1']['p50']:.3f} "
                     f"p90 {st['worst_d1']['p90']:.3f}; controls median "
                     f"{st['controls_d1']['p50'] if st['controls_d1'] else float('nan'):.3f} "
                     f"p99 {st['controls_d1_p99'] if st['controls_d1_p99'] is not None else float('nan'):.3f}; "
                     f"worst beyond controls' p99: {st['worst_pct_beyond_controls_p99']:.1f}%")
        ex = c["extrapolation"]
        L.append(f"  outside the training range: {ex['outside_training_range_pct']:.1f}%; above the training "
                 f"99.999th percentile in some channel: {ex['above_training_p99.999_any_pct']:.1f}%")
        for band, tab in c["channel_ratios_log10_over_171"].items():
            if tab["controls"] is None:
                continue
            L.append(f"  log10(channel / 171 A) medians, {band}: worst "
                     + ", ".join(f"{ch} {q['p50']:+.2f}" for ch, q in tab["worst"].items() if q)
                     + " | controls "
                     + ", ".join(f"{ch} {q['p50']:+.2f}" for ch, q in tab["controls"].items() if q))
    L.append("\n## Controls (random valid test pixels per band)")
    for band, q in out["controls"]["nn_d1_by_band"].items():
        if q:
            L.append(f"  {band:<10} nearest training pixel median {q['p50']:.3f} p90 {q['p90']:.3f}; "
                     f"clamp {out['controls']['clamp_any_pct_by_band'].get(band, float('nan')):.1f}%")
    return "\n".join(L)


def main():
    args = parse_args()
    track = args.track
    data = args.data or SHARED.format(track=track)
    if args.models:
        models = OrderedDict(s.split("=", 1) for s in args.models)
    else:
        models = OrderedDict(MODELS[track])
    missing = [f"{n}={p}" for n, p in models.items() if not os.path.isfile(p)]
    if missing:
        msg = "missing checkpoints: " + ", ".join(missing)
        if not args.skip_missing:
            raise FileNotFoundError(msg)
        print(msg, flush=True)
        models = OrderedDict((n, p) for n, p in models.items() if os.path.isfile(p))
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload) if isinstance(payload, dict) else payload, dtype=np.float64)
    import zarr
    available = int(zarr.open(os.path.join(data, "test_x.zarr"), mode="r").shape[-1])
    n_blocks = min(available, args.max_blocks) if args.max_blocks else available
    stem = os.path.splitext(args.output)[0]
    cfg = {"models": OrderedDict((n, os.path.abspath(p)) for n, p in models.items()),
           "data": os.path.abspath(data), "train_root": os.path.abspath(args.train_root),
           "thresholds": thresholds.tolist(), "threads": args.threads_per_worker,
           "pixel_batch": args.pixel_batch, "topk": args.topk, "controls": args.controls, "seed": args.seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t0 = time.time()
    if args.merge:
        files = sorted(glob.glob(f"{stem}_part*.pkl"))
        if not files:
            raise FileNotFoundError(f"no part files {stem}_part*.pkl")
        parts = []
        for fn in files:
            with open(fn, "rb") as f:
                parts.extend(pickle.load(f))
        print(f"merging {len(files)} part files ({len(parts)} worker results)", flush=True)
    else:
        start, stop = args.start, min(args.stop or n_blocks, n_blocks)
        ranges = [(s + start, e + start) for s, e in split_ranges(stop - start, args.workers)]
        jobs = [(cfg, s, e, i) for i, (s, e) in enumerate(ranges)]
        print(f"blocks {start:,}-{stop:,} of {available:,}; {len(models)} models: {list(models)}; ranges {ranges}", flush=True)
        if len(jobs) == 1:
            parts = [run_range(jobs[0])]
        else:
            with mp.get_context("spawn").Pool(len(jobs)) as pool:
                parts = pool.map(run_range, jobs)
        if args.part:
            fn = f"{stem}_part{start:06d}_{stop:06d}.pkl"
            with open(fn, "wb") as f:
                pickle.dump(parts, f, protocol=4)
            print(f"wrote {fn} in {time.time() - t0:.0f}s", flush=True)
            return
    res = merge_parts(parts, args.topk, args.controls, args.seed)
    covered = sum(e - s for s, e in res["ranges"])
    print(f"pass done: {covered:,} blocks in {time.time() - t0:.0f}s; analysing", flush=True)
    out, arrays = analyse(res, cfg, args)
    out.update({"purpose": "Census of the pixels on which the large models blow up; an association study",
                "track": track, "data": cfg["data"],
                "n_blocks": covered, "available_blocks": available, "complete_test": covered == available,
                "block_ranges": res["ranges"], "channels_angstrom": [int(c) for c in CHANNELS],
                "bright_thresholds": thresholds.tolist(), "bands": INTENSITY_LABELS,
                "valid_pixels_by_band": dict(zip(INTENSITY_LABELS, res["band_n"].tolist())),
                "definitions": {
                    "valid": "finite solver DEM (18 bins) and finite observations; per model also a finite prediction",
                    "sse": "sum over the 18 bins of (prediction - solver)^2 per pixel",
                    "clamp": f"an observed channel at or below {MIN_OBS:g} DN/s (the deconvolution positivity clamp)",
                    "band": "max over channels of observed / Bright threshold, as in aia_fit_by_brightness.py",
                    "worst_sets": "top-k pixels by SSE; distinct inputs merge repeats of one clean AIA input",
                    "nn_distance": "Euclidean distance in standardised log10-observation space to the nearest "
                                   "pixel of the stratified training sample (d1) and the mean over the 10 nearest (d10)",
                    "multipeaked": "zero-padded curve, >= 2 interior peaks with prominence >= 0.15 x curve max"},
                "runtime_s": round(time.time() - t0, 1)})
    # merge the per-model entries computed in analyse with the checkpoint paths
    for n in cfg["models"]:
        out["models"][n] = {**out["models"].get(n, {}), "checkpoint": cfg["models"][n]}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(stem + ".npz", **arrays)
    text = render(out)
    with open(stem + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.output} (+ .txt, .npz) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
