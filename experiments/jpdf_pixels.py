#!/usr/bin/env python3
"""What are the pixels at the low end of the AIA JPDFs, and the clamped ones?

The viewer's AIA JPDFs (observed against reconstructed intensity, log axes,
both floored at 0.5 DN/s) show a column of pixels at observed ~1 DN/s whose
label-free reconstruction sits far above the diagonal; the supervised model
shows no such column.  This census asks what those pixels are.  Working
hypothesis: the deconvolution positivity clamp (a bright channel at or near
zero after Hofmeister deconvolution; see zarr_data.MIN_OBS), excluded from our
training but present in the test set, which the solver fitted and the
supervised model learned.  Cosmic rays would show in the opposite corner
(observed far above every reconstruction) as isolated spikes.

Mode "shared": one pass over the shared test set (test_x, test_y) for the
label-free model, the supervised model and the solver.  Per channel it counts
the low cluster (observed < 1 DN/s and reconstruction > 10x the floored
observation) and the high cluster (observed > 20 DN/s and > 10x the floored
reconstruction), cross-tabulated with the clamp (observed <= MIN_OBS), the
brightness band, spatial isolation against the eight full-resolution
neighbours and the other two sources, and it recomputes the headline DEM and
AIA metrics under three masks (all valid pixels; no channel clamped; not in the
label-free low cluster).  The all-valid numbers must reproduce the known
headline values; a mismatch is a bug, not a result.

Mode "tol": one pass over our own test split, which also carries the per-pixel
errors and the solver's tolLevel: the same quantities by tolLevel (0 unsolved,
1 tight, 3 and 5 relaxed) with in-band shares at 1.4 sigma and at the solver's
own relaxed band, and by position on the Sun (disk, limb, off-limb), replaying
the staging's block draw to recover each block's place on the 4096 grid.

Evaluation only; no training.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.diagnose_bright_failures import (CHANNELS, INTENSITY_EDGES, INTENSITY_LABELS,
                                                  N_BINS, load_predictor, split_ranges)
from src.zarr_data import MIN_OBS

SOURCES = ("label_free", "supervised", "solver")
MODELS = SOURCES[:2]
KINDS = ("low", "high")
MASKS = ("all_valid", "no_clamp", "not_lf_low")
SUBSETS = ("full", "bright", "quiet")
METRIC_STATS = (["n", "dem_sse", "dem_abs", "em_abs_err", "em_truth"]
                + [f"aia_sq_{c}" for c in CHANNELS] + [f"aia_abs_{c}" for c in CHANNELS])
MI = {s: i for i, s in enumerate(METRIC_STATS)}
COMBOS = ["none", "solver", "supervised", "supervised+solver", "label_free",
          "label_free+solver", "label_free+supervised", "all"]   # bits: 4 lf, 2 sup, 1 solver
LOW_OBS, HIGH_OBS, FACTOR, FLOOR = 1.0, 20.0, 10.0, 0.5
DIP, SPIKE = 0.2, 5.0
LR_MIN, LR_W, N_LR = -2.0, 0.1, 70        # log10(reconstruction / observation), both floored
TOL_LEVELS = (0, 1, 3, 5)
TOL_INDEX = {t: i for i, t in enumerate(TOL_LEVELS)}
REGIONS = ("disk", "limb", "off_limb")
SUN_CENTRE, SUN_RADIUS, LIMB_HALF = 2048.0, 1600.0, 100.0
TOLFAC = 1.4
CACHE = 256          # distinct AIA blocks whose predictions are kept for repeats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("shared", "tol"), default="shared")
    p.add_argument("--label-free")
    p.add_argument("--supervised")
    p.add_argument("--data")
    p.add_argument("--reference", choices=("bp", "enet"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--datetimes", help="tol mode: timestamps of the split (default <data>/test_datetimes.txt)")
    p.add_argument("--output", help="JSON path; .txt and (shared mode) .npz go to the same stem")
    p.add_argument("--max-blocks", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--pixel-batch", type=int, default=16384)
    p.add_argument("--examples", type=int, default=200, help="low-cluster pixels kept per channel")
    p.add_argument("--examples-high", type=int, default=100, help="high-cluster pixels kept per channel")
    p.add_argument("--summarize", help="re-print the summary of an existing JSON")
    args = p.parse_args()
    if not args.summarize:
        missing = [k for k in ("label_free", "supervised", "data", "reference", "output")
                   if not getattr(args, k)]
        if missing:
            p.error("required: " + ", ".join("--" + k.replace("_", "-") for k in missing))
    return args


# ── shared per-block helpers ──────────────────────────────────────────────────

def brightness_band(obs, thr):
    """9 brightness bands of diagnose_bright_failures, consistent with Bright."""
    bright = (obs >= thr).any(axis=0)
    ratio = np.where(np.isfinite(obs), obs / thr, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log10(ratio.max(axis=0))
    logr = np.where(np.isfinite(logr), logr, -np.inf)
    band = np.searchsorted(INTENSITY_EDGES, logr, side="right")
    return np.where(bright, np.maximum(band, 4), np.minimum(band, 3)), bright


def neighbour_median(xb):
    """Median of the eight full-resolution neighbours of each DEM-grid pixel.

    DEM pixel (i, j) sits at full-resolution (2i, 2j).  Returns (med [6,h,w],
    ok [h,w]); ok is False on the first DEM row and column, whose neighbours
    would fall outside the block."""
    C, H, W = xb.shape
    shifts = [xb[:, 1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    med_int = np.median(np.stack(shifts), axis=0)        # full-res rows/cols 1 .. H-2
    med = np.full((C, H // 2, W // 2), np.nan, np.float32)
    med[:, 1:, 1:] = med_int[:, 1::2, 1::2]              # row 2i <-> interior index 2i-1
    ok = np.zeros((H // 2, W // 2), bool)
    ok[1:, 1:] = True
    return med, ok


def isolation(obs, med, ok):
    """dip: observation below 0.2x its neighbours; spike: above 5x.  [6,h,w] each."""
    ratio = np.maximum(obs, 0) / np.maximum(med, FLOOR)
    return (ratio < DIP) & ok[None], (ratio > SPIKE) & ok[None]


def cluster_flags(obs, rec):
    """low: observed < 1 and reconstruction > 10x the floored observation;
    high: observed > 20 and > 10x the floored reconstruction.  [S,6,h,w] each."""
    fo = np.maximum(obs, FLOOR)
    low = np.stack([(obs < LOW_OBS) & (rec[s] > FACTOR * fo) for s in SOURCES])
    high = np.stack([(obs > HIGH_OBS) & (obs > FACTOR * np.maximum(rec[s], FLOOR)) for s in SOURCES])
    return low, high


def log_ratio(rec_c, obs_c):
    return np.log10(np.maximum(rec_c, FLOOR) / np.maximum(obs_c, FLOOR))


def add_metrics(table, masks, bright, dems, rec, obs, ref):
    """table [len(masks), 3 subsets, 2 models, stats]; sums for DEM and AIA metrics."""
    o_all = obs.reshape(6, -1).astype(np.float64)
    r_all = ref.reshape(N_BINS, -1).astype(np.float64)
    for mi, m in enumerate(masks):
        for si, sub in enumerate(SUBSETS):
            sel = m if sub == "full" else (m & bright if sub == "bright" else m & ~bright)
            idx = np.flatnonzero(sel.ravel())
            if not idx.size:
                continue
            o, r_ = o_all[:, idx], r_all[:, idx]
            for ki, s in enumerate(MODELS):
                d = dems[s].reshape(N_BINS, -1)[:, idx].astype(np.float64)
                dd = rec[s].reshape(6, -1)[:, idx] - o
                row = table[mi, si, ki]
                row[MI["n"]] += idx.size
                row[MI["dem_sse"]] += np.square(d - r_).sum()
                row[MI["dem_abs"]] += np.abs(d - r_).sum()
                row[MI["em_abs_err"]] += np.abs(d.sum(0) - r_.sum(0)).sum()
                row[MI["em_truth"]] += np.abs(r_.sum(0)).sum()
                row[MI["aia_sq_94"]:MI["aia_sq_94"] + 6] += np.square(dd).sum(1)
                row[MI["aia_abs_94"]:MI["aia_abs_94"] + 6] += np.abs(dd).sum(1)


def reservoir_update(store, seen, rng, cap, cand, make):
    """Algorithm R over a batch of candidate pixel indices; make(p) builds a record."""
    m = int(cand.size)
    if m == 0:
        return seen
    k0 = 0
    while len(store) < cap and k0 < m:
        store.append(make(int(cand[k0])))
        k0 += 1
    if k0 < m:
        t = seen + k0 + np.arange(1, m - k0 + 1)
        u = np.floor(rng.random(m - k0) * t).astype(np.int64)
        for k in np.flatnonzero(u < cap):
            store[u[k]] = make(int(cand[k0 + k]))
    return seen + m


def merge_examples(results, pos, cap, seed):
    """Weighted merge of per-worker reservoirs (results[i][pos] is {key: (store, seen)})."""
    merged = {}
    for key in results[0][pos]:
        pool, weights = [], []
        for r in results:
            store, seen = r[pos][key]
            pool += store
            weights += [seen / max(len(store), 1)] * len(store)
        if pool:
            rng = np.random.default_rng(seed)
            w = np.asarray(weights, np.float64)
            keep = rng.choice(len(pool), size=min(len(pool), cap), replace=False, p=w / w.sum())
            pool = [pool[i] for i in sorted(keep)]
        merged[key] = pool
    return merged


def new_shared_acc():
    S, C, B = len(SOURCES), len(CHANNELS), len(INTENSITY_LABELS)
    return {"n_total": 0, "n_valid": 0, "excluded": 0, "clamp_any_valid": 0, "clamp_any_all": 0,
            "clamp_any_invalid": 0,
            "clamp": np.zeros((C, 3)),                 # valid & clamped, valid & exactly zero, all & clamped
            "clamp_by_band": np.zeros((C + 1, B)),     # per channel, then any channel (valid pixels)
            "n_by_band": np.zeros(B),
            "cluster_n": np.zeros((2, S, C)),
            "cluster_clamp": np.zeros((2, S, C, 3)),   # same channel clamped, any clamped, same channel zero
            "cluster_band": np.zeros((2, S, C, B)),
            "cluster_iso": np.zeros((2, S, C, 3)),     # dip, spike, isolation defined
            "base_n": np.zeros((2, C)),
            "threeway": np.zeros((2, C, 8)),
            "logratio_hist": np.zeros((S, C, N_LR)),   # over the low base population (observed < 1)
            "logratio_sum": np.zeros((2, S, C)),       # each source's log ratio over the label-free cluster
            "metrics": np.zeros((len(MASKS), len(SUBSETS), len(MODELS), len(METRIC_STATS)))}


def merge_acc(into, other):
    for k, v in other.items():
        into[k] = into[k] + v
    return into


def run_shared_range(job):
    cfg, start, stop, wid = job
    import zarr
    lf_predict, _ = load_predictor(cfg["label_free"], cfg["threads"])
    sup_predict, _ = load_predictor(cfg["supervised"], cfg["threads"])
    with np.load("RData.npz") as rdata:
        response = np.asarray(rdata["R"][:, :N_BINS], dtype=np.float64) * 1e26
    x = zarr.open(os.path.join(cfg["data"], "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(cfg["data"], "test_y.zarr"), mode="r")
    thr = np.asarray(cfg["thresholds"], dtype=np.float32)[:, None, None]
    acc = new_shared_acc()
    cache = OrderedDict()
    rng = np.random.default_rng(1000 + wid)
    caps = {"low": cfg["examples"], "high": cfg["examples_high"]}
    examples = {(kind, ch): ([], 0) for kind in KINDS for ch in CHANNELS}
    t0 = time.time()
    for b in range(start, stop):
        xb = np.ascontiguousarray(np.asarray(x[:, :, :, b], dtype=np.float32))
        key = hashlib.blake2b(xb.tobytes(), digest_size=16).hexdigest()
        obs = np.ascontiguousarray(xb[:, ::2, ::2])
        if key in cache:
            lf, sup = cache.pop(key)
        else:
            lf = lf_predict(obs, cfg["pixel_batch"])
            sup = sup_predict(obs, cfg["pixel_batch"])
        cache[key] = (lf, sup)
        if len(cache) > CACHE:
            cache.popitem(last=False)
        ref = np.asarray(y[:N_BINS, :, :, b], dtype=np.float32)
        dems = {"label_free": lf, "supervised": sup, "solver": ref}
        rec = {s: np.einsum("ct,thw->chw", response, d.astype(np.float64)) for s, d in dems.items()}
        base = np.isfinite(ref).all(axis=0) & np.isfinite(obs).all(axis=0)
        valid = (base & np.isfinite(lf).all(axis=0) & np.isfinite(sup).all(axis=0)
                 & np.isfinite(rec["label_free"]).all(axis=0) & np.isfinite(rec["supervised"]).all(axis=0))
        acc["excluded"] += int((base & ~valid).sum())
        acc["n_total"] += valid.size
        acc["n_valid"] += int(valid.sum())
        band, bright = brightness_band(obs, thr)

        clamp = obs <= MIN_OBS
        zero = obs == 0
        clamp_any = clamp.any(axis=0)
        acc["clamp"][:, 0] += (clamp & valid[None]).reshape(6, -1).sum(1)
        acc["clamp"][:, 1] += (zero & valid[None]).reshape(6, -1).sum(1)
        acc["clamp"][:, 2] += clamp.reshape(6, -1).sum(1)
        acc["clamp_any_valid"] += int((clamp_any & valid).sum())
        acc["clamp_any_all"] += int(clamp_any.sum())
        acc["clamp_any_invalid"] += int((clamp_any & ~valid).sum())
        acc["n_by_band"] += np.bincount(band[valid], minlength=len(INTENSITY_LABELS))
        for c in range(6):
            acc["clamp_by_band"][c] += np.bincount(band[clamp[c] & valid], minlength=len(INTENSITY_LABELS))
        acc["clamp_by_band"][6] += np.bincount(band[clamp_any & valid], minlength=len(INTENSITY_LABELS))

        med, ok = neighbour_median(xb)
        dip, spike = isolation(obs, med, ok)
        low, high = cluster_flags(obs, rec)
        vflat = valid.ravel()
        oflat = obs.reshape(6, -1)
        bflat = band.ravel()
        lr = {s: log_ratio(rec[s].reshape(6, -1), oflat) for s in SOURCES}
        for ki, flags in enumerate((low, high)):
            f = flags.reshape(len(SOURCES), 6, -1) & vflat[None, None]
            acc["cluster_n"][ki] += f.sum(-1)
            acc["cluster_clamp"][ki, :, :, 0] += (f & clamp.reshape(6, -1)[None]).sum(-1)
            acc["cluster_clamp"][ki, :, :, 1] += (f & clamp_any.ravel()[None, None]).sum(-1)
            acc["cluster_clamp"][ki, :, :, 2] += (f & zero.reshape(6, -1)[None]).sum(-1)
            acc["cluster_iso"][ki, :, :, 0] += (f & dip.reshape(6, -1)[None]).sum(-1)
            acc["cluster_iso"][ki, :, :, 1] += (f & spike.reshape(6, -1)[None]).sum(-1)
            acc["cluster_iso"][ki, :, :, 2] += (f & ok.ravel()[None, None]).sum(-1)
            basepop = vflat[None] & ((oflat < LOW_OBS) if ki == 0 else (oflat > HIGH_OBS))
            acc["base_n"][ki] += basepop.sum(-1)
            code = 4 * f[0].astype(np.int64) + 2 * f[1] + f[2]
            for c in range(6):
                acc["threeway"][ki, c] += np.bincount(code[c][basepop[c]], minlength=8)
                for s in range(len(SOURCES)):
                    acc["cluster_band"][ki, s, c] += np.bincount(bflat[f[s, c]], minlength=len(INTENSITY_LABELS))
            for s, name in enumerate(SOURCES):
                # index, do not multiply: the solver's log ratio is NaN on unsolved pixels
                for c in range(6):
                    acc["logratio_sum"][ki, s, c] += lr[name][c][f[0, c]].sum()
        basepop = vflat[None] & (oflat < LOW_OBS)
        for s, name in enumerate(SOURCES):
            for c in range(6):
                vals = lr[name][c][basepop[c]]
                hidx = np.clip(np.floor((vals - LR_MIN) / LR_W), 0, N_LR - 1).astype(np.int64)
                acc["logratio_hist"][s, c] += np.bincount(hidx, minlength=N_LR)

        lf_low_any = low[0].any(axis=0)
        add_metrics(acc["metrics"], [valid, valid & ~clamp_any, valid & ~lf_low_any],
                    bright, dems, rec, obs, ref)

        flat = {"obs": oflat, "med": med.reshape(6, -1), "clamp": clamp.reshape(6, -1),
                "ref": ref.reshape(N_BINS, -1), "lf": lf.reshape(N_BINS, -1), "sup": sup.reshape(N_BINS, -1),
                "rec_lf": rec["label_free"].reshape(6, -1), "rec_sup": rec["supervised"].reshape(6, -1),
                "rec_sol": rec["solver"].reshape(6, -1)}
        for ki, (kind, flags) in enumerate(zip(KINDS, (low, high))):
            f = flags.reshape(len(SOURCES), 6, -1) & vflat[None, None]
            for c, ch in enumerate(CHANNELS):
                cand = np.flatnonzero(f[0, c])
                if not cand.size:
                    continue

                def make(p, b=b, c=c, f=f):
                    return {"block": b, "i": p // obs.shape[2], "j": p % obs.shape[2], "band": int(bflat[p]),
                            "member": np.array([f[s, c, p] for s in range(len(SOURCES))]),
                            **{k: v[:, p].copy() for k, v in flat.items()}}
                store, seen = examples[(kind, ch)]
                examples[(kind, ch)] = (store, reservoir_update(store, seen, rng, caps[kind], cand, make))

        done = b - start + 1
        if done % 500 == 0 or b + 1 == stop:
            print(f"[w{wid}] {done:,}/{stop - start:,} blocks  {done / (time.time() - t0):.1f} blocks/s", flush=True)
    return acc, examples


# ── tol mode (our own split: errors, tolLevel, block positions) ───────────────

def block_origins(datetimes_path, n_blocks, per_image=64, block=256, n_side=16):
    """Replay dataset/stage_hofdeconv_full.py's block draw: for file index i
    (sorted timestamps) random.seed(i), shuffle the 16x16 block grid and keep
    the first 64; block index = i * 64 + c.  Returns (sy, sx) per block."""
    with open(datetimes_path) as f:
        stamps = sorted(line.strip() for line in f if line.strip())
    if len(stamps) * per_image != n_blocks:
        raise ValueError(f"{len(stamps)} timestamps x {per_image} != {n_blocks} blocks")
    origins = np.zeros((n_blocks, 2), np.int64)
    for i in range(len(stamps)):
        random.seed(i)
        blocks = [(bi * block, bj * block) for bi in range(n_side) for bj in range(n_side)]
        random.shuffle(blocks)
        for c, (sy, sx) in enumerate(blocks[:per_image]):
            origins[i * per_image + c] = (sy, sx)
    return origins


T_STATS = (["n", "dem_sse_lf", "dem_sse_sup", "dem_abs_lf", "dem_abs_sup", "em_abs_lf", "em_abs_sup", "em_truth"]
           + [f"aia_sq_{s}_{c}" for s in ("lf", "sup", "sol") for c in CHANNELS]
           + [f"inband_tight_{s}_{c}" for s in ("lf", "sup", "sol") for c in CHANNELS]
           + [f"inband_own_{s}_{c}" for s in ("lf", "sup", "sol") for c in CHANNELS])
TI = {s: i for i, s in enumerate(T_STATS)}
R_STATS = (["n_all", "n_valid", "n_171", "sum_171", "n_171_faint", "n_clamp_any", "n_tol0"]
           + [f"clamp_{c}" for c in CHANNELS]
           + [f"low_any_{s}" for s in SOURCES] + [f"high_any_{s}" for s in SOURCES]
           + ["dip_any", "spike_any", "iso_defined"])
RI = {s: i for i, s in enumerate(R_STATS)}


def new_tol_acc():
    return {"n_total": 0, "n_other_tol": 0,
            "tol_region": np.zeros((len(TOL_LEVELS), len(REGIONS))),          # all pixels
            "tol_band": np.zeros((len(TOL_LEVELS), len(INTENSITY_LABELS))),   # valid observations
            # DEM-valid pixels (finite solver DEM and observations): all three sources
            "tol_dem": np.zeros((len(TOL_LEVELS), len(SUBSETS), len(T_STATS))),
            # observation-valid pixels (no solver DEM needed): label-free and supervised AIA fit
            "tol_aia": np.zeros((len(TOL_LEVELS), len(SUBSETS), len(T_STATS))),
            "region": np.zeros((len(REGIONS), len(R_STATS))),
            "region_band": np.zeros((len(REGIONS), len(INTENSITY_LABELS))),
            "region_cluster": np.zeros((len(REGIONS), 2, len(SOURCES), len(CHANNELS)))}


def add_tol_stats(row, idx, dems, rec, obs, ref, sig, sig_own, with_solver):
    o = obs.reshape(6, -1)[:, idx].astype(np.float64)
    s_t = sig.reshape(6, -1)[:, idx]
    s_o = sig_own.reshape(6, -1)[:, idx]
    row[TI["n"]] += idx.size
    if with_solver:
        r_ = ref.reshape(N_BINS, -1)[:, idx].astype(np.float64)
        for tag, name in (("lf", "label_free"), ("sup", "supervised")):
            d = dems[name].reshape(N_BINS, -1)[:, idx].astype(np.float64)
            row[TI[f"dem_sse_{tag}"]] += np.square(d - r_).sum()
            row[TI[f"dem_abs_{tag}"]] += np.abs(d - r_).sum()
            row[TI[f"em_abs_{tag}"]] += np.abs(d.sum(0) - r_.sum(0)).sum()
        row[TI["em_truth"]] += np.abs(r_.sum(0)).sum()
    for tag, name in (("lf", "label_free"), ("sup", "supervised"), ("sol", "solver")):
        if tag == "sol" and not with_solver:
            continue
        dd = np.abs(rec[name].reshape(6, -1)[:, idx] - o)
        row[TI[f"aia_sq_{tag}_94"]:TI[f"aia_sq_{tag}_94"] + 6] += np.square(dd).sum(1)
        row[TI[f"inband_tight_{tag}_94"]:TI[f"inband_tight_{tag}_94"] + 6] += (dd <= s_t).sum(1)
        row[TI[f"inband_own_{tag}_94"]:TI[f"inband_own_{tag}_94"] + 6] += (dd <= s_o).sum(1)


def run_tol_range(job):
    cfg, start, stop, wid = job
    import zarr
    lf_predict, _ = load_predictor(cfg["label_free"], cfg["threads"])
    sup_predict, _ = load_predictor(cfg["supervised"], cfg["threads"])
    with np.load("RData.npz") as rdata:
        response = np.asarray(rdata["R"][:, :N_BINS], dtype=np.float64) * 1e26
    z = {k: zarr.open(os.path.join(cfg["data"], f"test_{k}.zarr"), mode="r") for k in ("x", "e", "y", "m")}
    thr = np.asarray(cfg["thresholds"], dtype=np.float32)[:, None, None]
    origins = np.asarray(cfg["origins"])
    ii, jj = np.mgrid[0:128, 0:128]
    acc = new_tol_acc()
    t0 = time.time()
    for b in range(start, stop):
        xb = np.ascontiguousarray(np.asarray(z["x"][:, :, :, b], dtype=np.float32))
        obs = np.ascontiguousarray(xb[:, ::2, ::2])
        err = np.ascontiguousarray(np.asarray(z["e"][:, :, :, b], dtype=np.float32)[:, ::2, ::2])
        ref = np.asarray(z["y"][:N_BINS, :, :, b], dtype=np.float32)
        tol = np.asarray(z["m"][:, :, b])
        lf = lf_predict(obs, cfg["pixel_batch"])
        sup = sup_predict(obs, cfg["pixel_batch"])
        dems = {"label_free": lf, "supervised": sup, "solver": ref}
        rec = {s: np.einsum("ct,thw->chw", response, d.astype(np.float64)) for s, d in dems.items()}
        valid_aia = (np.isfinite(obs).all(0) & np.isfinite(err).all(0) & (err > 0).all(0)
                     & np.isfinite(lf).all(0) & np.isfinite(sup).all(0)
                     & np.isfinite(rec["label_free"]).all(0) & np.isfinite(rec["supervised"]).all(0))
        valid_dem = valid_aia & np.isfinite(ref).all(0)
        band, bright = brightness_band(obs, thr)
        tgroup = np.full(tol.shape, -1, np.int64)
        for t, g in TOL_INDEX.items():
            tgroup[tol == t] = g
        acc["n_total"] += tol.size
        acc["n_other_tol"] += int((tgroup < 0).sum())
        sy, sx = origins[b]
        r = np.hypot(sy + 2 * ii - SUN_CENTRE, sx + 2 * jj - SUN_CENTRE)
        region = np.where(r < SUN_RADIUS - LIMB_HALF, 0, np.where(r < SUN_RADIUS + LIMB_HALF, 1, 2))
        sig = TOLFAC * err
        sig_own = sig * np.maximum(tol, 1)[None].astype(np.float32)

        clamp = obs <= MIN_OBS
        clamp_any = clamp.any(0)
        med, ok = neighbour_median(xb)
        dip, spike = isolation(obs, med, ok)
        low, high = cluster_flags(obs, rec)
        src_valid = np.stack([valid_aia, valid_aia, valid_dem])

        for g in range(len(TOL_LEVELS)):
            tg = tgroup == g
            acc["tol_region"][g] += np.bincount(region[tg], minlength=len(REGIONS))
            acc["tol_band"][g] += np.bincount(band[tg & valid_aia], minlength=len(INTENSITY_LABELS))
            for si, sub in enumerate(SUBSETS):
                sel = tg if sub == "full" else (tg & bright if sub == "bright" else tg & ~bright)
                idx = np.flatnonzero((sel & valid_dem).ravel())
                if idx.size:
                    add_tol_stats(acc["tol_dem"][g, si], idx, dems, rec, obs, ref, sig, sig_own, True)
                idx = np.flatnonzero((sel & valid_aia).ravel())
                if idx.size:
                    add_tol_stats(acc["tol_aia"][g, si], idx, dems, rec, obs, ref, sig, sig_own, False)

        fin171 = np.isfinite(obs[2])
        for gi in range(len(REGIONS)):
            rg = region == gi
            row = acc["region"][gi]
            row[RI["n_all"]] += int(rg.sum())
            row[RI["n_valid"]] += int((rg & valid_aia).sum())
            row[RI["n_171"]] += int((rg & fin171).sum())
            row[RI["sum_171"]] += float(obs[2][rg & fin171].sum())
            row[RI["n_171_faint"]] += int((rg & fin171 & (obs[2] < 5.0)).sum())
            row[RI["n_clamp_any"]] += int((rg & clamp_any).sum())
            row[RI["n_tol0"]] += int((rg & (tol == 0)).sum())
            for c, ch in enumerate(CHANNELS):
                row[RI[f"clamp_{ch}"]] += int((rg & clamp[c]).sum())
            for s, name in enumerate(SOURCES):
                row[RI[f"low_any_{name}"]] += int((rg & src_valid[s] & low[s].any(0)).sum())
                row[RI[f"high_any_{name}"]] += int((rg & src_valid[s] & high[s].any(0)).sum())
                acc["region_cluster"][gi, 0, s] += (low[s] & (rg & src_valid[s])[None]).reshape(6, -1).sum(1)
                acc["region_cluster"][gi, 1, s] += (high[s] & (rg & src_valid[s])[None]).reshape(6, -1).sum(1)
            row[RI["dip_any"]] += int((rg & valid_aia & dip.any(0)).sum())
            row[RI["spike_any"]] += int((rg & valid_aia & spike.any(0)).sum())
            row[RI["iso_defined"]] += int((rg & valid_aia & ok).sum())
            acc["region_band"][gi] += np.bincount(band[rg & valid_aia], minlength=len(INTENSITY_LABELS))

        done = b - start + 1
        if done % 500 == 0 or b + 1 == stop:
            print(f"[w{wid}] {done:,}/{stop - start:,} blocks  {done / (time.time() - t0):.1f} blocks/s", flush=True)
    return acc, None


# ── summaries ─────────────────────────────────────────────────────────────────

def pct(a, b):
    return float(100.0 * a / b) if b else None


def metric_row(row):
    n = row[MI["n"]]
    if not n:
        return None
    sq = row[MI["aia_sq_94"]:MI["aia_sq_94"] + 6]
    ab = row[MI["aia_abs_94"]:MI["aia_abs_94"] + 6]
    return {"n_pixels": int(n),
            "dem_mse": float(row[MI["dem_sse"]] / (n * N_BINS)),
            "dem_mae": float(row[MI["dem_abs"]] / (n * N_BINS)),
            "em_error_pct": pct(row[MI["em_abs_err"]], row[MI["em_truth"]]),
            "aia_mse": float(sq.sum() / (6 * n)), "aia_mae": float(ab.sum() / (6 * n)),
            "aia_mse_by_channel": {c: float(sq[k] / n) for k, c in enumerate(CHANNELS)},
            "aia_mae_by_channel": {c: float(ab[k] / n) for k, c in enumerate(CHANNELS)}}


def summarize_shared(acc):
    out = {"pixels": {"total": int(acc["n_total"]), "valid": int(acc["n_valid"]),
                      "excluded_nonfinite_prediction": int(acc["excluded"]),
                      "valid_by_band": dict(zip(INTENSITY_LABELS, acc["n_by_band"].astype(int).tolist()))}}
    nv = acc["n_valid"]
    out["clamp"] = {
        "definition": f"observed <= MIN_OBS = {MIN_OBS:g} DN/s in that channel",
        "any_channel": {"valid": int(acc["clamp_any_valid"]), "pct_of_valid": pct(acc["clamp_any_valid"], nv),
                        "all_pixels": int(acc["clamp_any_all"]),
                        "solver_unsolved_or_invalid": int(acc["clamp_any_invalid"])},
        "by_channel": {c: {"valid": int(acc["clamp"][k, 0]), "pct_of_valid": pct(acc["clamp"][k, 0], nv),
                           "exact_zero_valid": int(acc["clamp"][k, 1]), "all_pixels": int(acc["clamp"][k, 2])}
                       for k, c in enumerate(CHANNELS)},
        "pct_of_band_any_channel": {b: pct(acc["clamp_by_band"][6, i], acc["n_by_band"][i])
                                    for i, b in enumerate(INTENSITY_LABELS)},
        "pct_of_band_by_channel": {c: {b: pct(acc["clamp_by_band"][k, i], acc["n_by_band"][i])
                                       for i, b in enumerate(INTENSITY_LABELS)} for k, c in enumerate(CHANNELS)}}
    clusters = {}
    for ki, kind in enumerate(KINDS):
        per = {}
        for k, c in enumerate(CHANNELS):
            entry = {"base_population": int(acc["base_n"][ki, k]),
                     "threeway": {COMBOS[m]: int(acc["threeway"][ki, k, m]) for m in range(8)},
                     "sources": {}}
            for s, name in enumerate(SOURCES):
                n = acc["cluster_n"][ki, s, k]
                entry["sources"][name] = {
                    "n": int(n), "pct_of_valid": pct(n, nv),
                    "pct_same_channel_clamped": pct(acc["cluster_clamp"][ki, s, k, 0], n),
                    "pct_any_channel_clamped": pct(acc["cluster_clamp"][ki, s, k, 1], n),
                    "pct_same_channel_exact_zero": pct(acc["cluster_clamp"][ki, s, k, 2], n),
                    "pct_dip": pct(acc["cluster_iso"][ki, s, k, 0], acc["cluster_iso"][ki, s, k, 2]),
                    "pct_spike": pct(acc["cluster_iso"][ki, s, k, 1], acc["cluster_iso"][ki, s, k, 2]),
                    "band_counts": dict(zip(INTENSITY_LABELS, acc["cluster_band"][ki, s, k].astype(int).tolist())),
                    "mean_log10_recon_over_obs_on_label_free_cluster":
                        float(acc["logratio_sum"][ki, s, k] / acc["cluster_n"][ki, 0, k])
                        if acc["cluster_n"][ki, 0, k] else None}
            per[c] = entry
        clusters[kind] = per
    out["clusters"] = clusters
    edges = (LR_MIN + LR_W * np.arange(N_LR + 1)).round(2).tolist()
    out["low_observation_log_ratio_hist"] = {
        "definition": "pixels with observed < 1 DN/s: log10(max(recon, 0.5) / max(observed, 0.5)), bin edges below",
        "edges": edges,
        "counts": {name: {c: acc["logratio_hist"][s, k].astype(int).tolist() for k, c in enumerate(CHANNELS)}
                   for s, name in enumerate(SOURCES)}}
    out["metrics"] = {mask: {sub: {name: metric_row(acc["metrics"][mi, si, ki]) for ki, name in enumerate(MODELS)}
                             for si, sub in enumerate(SUBSETS)} for mi, mask in enumerate(MASKS)}
    return out


def render_shared(out):
    px, cl, me = out["pixels"], out["clamp"], out["metrics"]
    L = [f"# JPDF low-end, clamp and spike census: {out['reference'].upper()} track, shared test set",
         f"blocks {out['n_blocks']:,} of {out['available_blocks']:,}; valid pixels {px['valid']:,} "
         f"of {px['total']:,}; non-finite predictions excluded {px['excluded_nonfinite_prediction']:,}",
         "", "## Clamp (observed <= 1e-3 DN/s)",
         f"any channel: {cl['any_channel']['valid']:,} valid pixels ({cl['any_channel']['pct_of_valid']:.2f}%), "
         f"{cl['any_channel']['solver_unsolved_or_invalid']:,} more where the solver DEM is not finite"]
    for c in CHANNELS:
        r = cl["by_channel"][c]
        L.append(f"  {c:>4} A: {r['valid']:>11,} ({r['pct_of_valid']:5.2f}%)  exact zeros {r['exact_zero_valid']:,}")
    L.append("clamped share by band (any channel): " + "  ".join(
        f"{b} {v:.1f}%" for b, v in cl["pct_of_band_any_channel"].items() if v is not None))
    for kind, title in (("low", "Low cluster: observed < 1 DN/s, reconstruction > 10x the floored observation"),
                        ("high", "High cluster: observed > 20 DN/s and > 10x the floored reconstruction")):
        L += ["", f"## {title}"]
        head = f"{'chan':<6}{'base pop':>12}" + "".join(f"{s:>14}" for s in SOURCES) + "   label-free cluster: same-ch clamp / any clamp / dip / spike"
        L += [head, "-" * len(head)]
        for c in CHANNELS:
            e = out["clusters"][kind][c]
            lf = e["sources"]["label_free"]
            L.append(f"{c + 'A':<6}{e['base_population']:>12,}" + "".join(
                f"{e['sources'][s]['n']:>14,}" for s in SOURCES)
                + f"   {lf['pct_same_channel_clamped'] or 0:5.1f}% / {lf['pct_any_channel_clamped'] or 0:5.1f}% / "
                f"{lf['pct_dip'] or 0:5.1f}% / {lf['pct_spike'] or 0:5.1f}%")
        L.append("three-way membership within the base population (label-free / supervised / solver):")
        for c in CHANNELS:
            t = out["clusters"][kind][c]["threeway"]
            L.append(f"  {c:>4} A: " + "  ".join(f"{k} {v:,}" for k, v in t.items() if v))
        L.append("mean log10(reconstruction / observation) on the label-free cluster, by source:")
        for c in CHANNELS:
            e = out["clusters"][kind][c]["sources"]
            vals = [e[s]["mean_log10_recon_over_obs_on_label_free_cluster"] for s in SOURCES]
            if vals[0] is not None:
                L.append(f"  {c:>4} A: " + "  ".join(f"{s} {v:+.2f}" for s, v in zip(SOURCES, vals)))
        L.append("label-free cluster by brightness band:")
        for c in CHANNELS:
            bc = out["clusters"][kind][c]["sources"]["label_free"]["band_counts"]
            tot = sum(bc.values())
            if tot:
                L.append(f"  {c:>4} A: " + "  ".join(f"{b} {100 * v / tot:.0f}%" for b, v in bc.items() if v))
    L += ["", "## Headline metrics under three masks (label-free / supervised)",
          "all_valid must reproduce the known numbers; no_clamp drops pixels with any channel at or below 1e-3; "
          "not_lf_low drops pixels in the label-free low cluster in any channel"]
    head = f"{'mask':<12}{'subset':<8}{'pixels':>14}{'DEM MSE':>18}{'DEM MAE':>16}{'EM err %':>16}{'AIA MSE':>18}{'AIA MAE':>16}"
    L += [head, "-" * len(head)]
    for mask in MASKS:
        for sub in SUBSETS:
            a, b = me[mask][sub]["label_free"], me[mask][sub]["supervised"]
            if a is None:
                continue
            L.append(f"{mask:<12}{sub:<8}{a['n_pixels']:>14,}"
                     f"{a['dem_mse']:>9.4f}/{b['dem_mse']:<8.4f}"
                     f"{a['dem_mae']:>8.4f}/{b['dem_mae']:<7.4f}"
                     f"{a['em_error_pct']:>8.2f}/{b['em_error_pct']:<7.2f}"
                     f"{a['aia_mse']:>9.2f}/{b['aia_mse']:<8.2f}"
                     f"{a['aia_mae']:>8.3f}/{b['aia_mae']:<7.3f}")
    h = out["low_observation_log_ratio_hist"]
    L += ["", "## Pixels with observed < 1 DN/s: share of each source more than one decade above the floored observation"]
    edges = np.asarray(h["edges"])
    for c in CHANNELS:
        parts = []
        for s in SOURCES:
            counts = np.asarray(h["counts"][s][c], np.float64)
            tot = counts.sum()
            above = counts[edges[:-1] >= 1.0].sum()
            parts.append(f"{s} {100 * above / tot:5.1f}%" if tot else f"{s} n/a")
        L.append(f"  {c:>4} A: " + "  ".join(parts))
    return "\n".join(L)


def tol_row(row, with_solver):
    n = row[TI["n"]]
    if not n:
        return None
    d = {"n_pixels": int(n)}
    if with_solver:
        d.update({"dem_mse": {"label_free": float(row[TI["dem_sse_lf"]] / (n * N_BINS)),
                              "supervised": float(row[TI["dem_sse_sup"]] / (n * N_BINS))},
                  "dem_mae": {"label_free": float(row[TI["dem_abs_lf"]] / (n * N_BINS)),
                              "supervised": float(row[TI["dem_abs_sup"]] / (n * N_BINS))},
                  "em_error_pct": {"label_free": pct(row[TI["em_abs_lf"]], row[TI["em_truth"]]),
                                   "supervised": pct(row[TI["em_abs_sup"]], row[TI["em_truth"]])}})
    tags = (("lf", "label_free"), ("sup", "supervised"), ("sol", "solver")) if with_solver \
        else (("lf", "label_free"), ("sup", "supervised"))
    d["aia_mse"] = {name: float(row[TI[f"aia_sq_{t}_94"]:TI[f"aia_sq_{t}_94"] + 6].sum() / (6 * n)) for t, name in tags}
    d["inband_tight_pct"] = {name: {c: pct(row[TI[f"inband_tight_{t}_{c}"]], n) for c in CHANNELS} for t, name in tags}
    d["inband_own_pct"] = {name: {c: pct(row[TI[f"inband_own_{t}_{c}"]], n) for c in CHANNELS} for t, name in tags}
    return d


def summarize_tol(acc):
    out = {"pixels": {"total": int(acc["n_total"]), "other_tol_values": int(acc["n_other_tol"])},
           "tol_levels": {}, "regions": {}}
    for g, t in enumerate(TOL_LEVELS):
        out["tol_levels"][str(t)] = {
            "n_all": int(acc["tol_region"][g].sum()), "pct_of_all": pct(acc["tol_region"][g].sum(), acc["n_total"]),
            "by_region": dict(zip(REGIONS, acc["tol_region"][g].astype(int).tolist())),
            "band_counts": dict(zip(INTENSITY_LABELS, acc["tol_band"][g].astype(int).tolist())),
            "dem_valid": {sub: tol_row(acc["tol_dem"][g, si], True) for si, sub in enumerate(SUBSETS)},
            "obs_valid": {sub: tol_row(acc["tol_aia"][g, si], False) for si, sub in enumerate(SUBSETS)}}
    for gi, reg in enumerate(REGIONS):
        row = acc["region"][gi]
        n_all, n_valid = row[RI["n_all"]], row[RI["n_valid"]]
        out["regions"][reg] = {
            "n_all": int(n_all), "n_valid": int(n_valid),
            "mean_171": float(row[RI["sum_171"]] / row[RI["n_171"]]) if row[RI["n_171"]] else None,
            "pct_171_below_5": pct(row[RI["n_171_faint"]], row[RI["n_171"]]),
            "pct_tol0": pct(row[RI["n_tol0"]], n_all),
            "pct_clamp_any": pct(row[RI["n_clamp_any"]], n_all),
            "pct_clamp_by_channel": {c: pct(row[RI[f"clamp_{c}"]], n_all) for c in CHANNELS},
            "pct_low_any_channel": {s: pct(row[RI[f"low_any_{s}"]], n_valid) for s in SOURCES},
            "pct_high_any_channel": {s: pct(row[RI[f"high_any_{s}"]], n_valid) for s in SOURCES},
            "pct_dip_any": pct(row[RI["dip_any"]], row[RI["iso_defined"]]),
            "pct_spike_any": pct(row[RI["spike_any"]], row[RI["iso_defined"]]),
            "band_counts": dict(zip(INTENSITY_LABELS, acc["region_band"][gi].astype(int).tolist())),
            "cluster_counts": {kind: {s: dict(zip(CHANNELS, acc["region_cluster"][gi, ki, si].astype(int).tolist()))
                                      for si, s in enumerate(SOURCES)} for ki, kind in enumerate(KINDS)}}
    return out


def render_tol(out):
    L = [f"# By solver tolLevel and position on the Sun: {out['reference'].upper()} track, our test split",
         f"blocks {out['n_blocks']:,} of {out['available_blocks']:,}; pixels {out['pixels']['total']:,}; "
         f"other tolLevel values {out['pixels']['other_tol_values']:,}",
         "", "## tolLevel (0 unsolved, 1 tight band, 3 and 5 relaxed); DEM metrics on pixels with a solver DEM",
         f"{'tol':<5}{'pixels':>13}{'share':>8}{'bright':>8}{'DEM MSE lf/sup':>20}{'EM% lf/sup':>16}{'AIA MSE lf/sup/sol':>26}"
         f"{'in 1.4s lf/sup/sol':>22}{'in own band lf/sup/sol':>26}"]
    for t, e in out["tol_levels"].items():
        dv = e["dem_valid"]["full"]
        ov = e["obs_valid"]["full"]
        if ov is None:
            L.append(f"{t:<5}{e['n_all']:>13,}{e['pct_of_all']:>7.2f}%   (no valid observations)")
            continue
        nb = e["dem_valid"]["bright"]
        bright_share = 100 * nb["n_pixels"] / dv["n_pixels"] if dv and nb else 0.0
        if dv:
            ib = lambda key: "/".join(f"{np.mean([dv[key][s][c] for c in CHANNELS]):.0f}" for s in SOURCES)
            L.append(f"{t:<5}{e['n_all']:>13,}{e['pct_of_all']:>7.2f}%{bright_share:>7.1f}%"
                     f"{dv['dem_mse']['label_free']:>10.4f}/{dv['dem_mse']['supervised']:<9.4f}"
                     f"{dv['em_error_pct']['label_free']:>8.1f}/{dv['em_error_pct']['supervised']:<7.1f}"
                     f"{dv['aia_mse']['label_free']:>9.1f}/{dv['aia_mse']['supervised']:.1f}/{dv['aia_mse']['solver']:<8.1f}"
                     f"{ib('inband_tight_pct'):>22}{ib('inband_own_pct'):>26}")
        else:
            ib = lambda key: "/".join(f"{np.mean([ov[key][s][c] for c in CHANNELS]):.0f}" for s in MODELS)
            L.append(f"{t:<5}{e['n_all']:>13,}{e['pct_of_all']:>7.2f}%   no solver DEM; AIA MSE lf/sup "
                     f"{ov['aia_mse']['label_free']:.1f}/{ov['aia_mse']['supervised']:.1f}; in 1.4 sigma {ib('inband_tight_pct')}")
    L.append("in-band shares are means over the six channels; per-channel values are in the JSON")
    L += ["", "## Position on the Sun (block positions replayed from the staging draw; centre 2048, radius 1600 px)",
          "check: off-limb pixels must be faint in 171 A",
          f"{'region':<10}{'pixels':>13}{'mean 171':>10}{'171<5':>8}{'tol0':>8}{'clamp any':>11}"
          + "".join(f"{'low ' + s[:3]:>10}" for s in SOURCES) + "".join(f"{'high ' + s[:3]:>10}" for s in SOURCES)
          + f"{'dip':>8}{'spike':>8}"]
    for reg, e in out["regions"].items():
        if not e["n_all"]:
            continue
        L.append(f"{reg:<10}{e['n_all']:>13,}{(e['mean_171'] or 0):>10.1f}{(e['pct_171_below_5'] or 0):>7.1f}%"
                 f"{(e['pct_tol0'] or 0):>7.1f}%{(e['pct_clamp_any'] or 0):>10.2f}%"
                 + "".join(f"{(e['pct_low_any_channel'][s] or 0):>9.2f}%" for s in SOURCES)
                 + "".join(f"{(e['pct_high_any_channel'][s] or 0):>9.3f}%" for s in SOURCES)
                 + f"{(e['pct_dip_any'] or 0):>7.2f}%{(e['pct_spike_any'] or 0):>7.3f}%")
    L.append("clamp by channel and region (% of pixels): " + "; ".join(
        f"{reg} " + " ".join(f"{c}:{(e['pct_clamp_by_channel'][c] or 0):.2f}" for c in CHANNELS)
        for reg, e in out["regions"].items() if e["n_all"]))
    return "\n".join(L)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            out = json.load(f)
        print(render_tol(out) if out.get("mode") == "tol" else render_shared(out))
        return
    import zarr
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload) if isinstance(payload, dict) else payload, dtype=np.float64)
    available = int(zarr.open(os.path.join(args.data, "test_x.zarr"), mode="r").shape[-1])
    n_blocks = min(available, args.max_blocks) if args.max_blocks else available
    cfg = {"label_free": os.path.abspath(args.label_free), "supervised": os.path.abspath(args.supervised),
           "data": os.path.abspath(args.data), "thresholds": thresholds.tolist(),
           "threads": args.threads_per_worker, "pixel_batch": args.pixel_batch,
           "examples": args.examples, "examples_high": args.examples_high}
    if args.mode == "tol":
        stamps = args.datetimes or os.path.join(args.data, "test_datetimes.txt")
        cfg["origins"] = block_origins(stamps, available).tolist()
    ranges = split_ranges(n_blocks, args.workers)
    jobs = [(cfg, s, e, i) for i, (s, e) in enumerate(ranges)]
    print(f"{args.mode} mode; blocks {n_blocks:,} of {available:,}; ranges {ranges}", flush=True)
    worker = run_shared_range if args.mode == "shared" else run_tol_range
    t0 = time.time()
    if len(jobs) == 1:
        results = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(worker, jobs)
    acc = results[0][0]
    for r in results[1:]:
        merge_acc(acc, r[0])
    out = {"mode": args.mode, "reference": args.reference, "label_free": cfg["label_free"],
           "supervised": cfg["supervised"], "data": cfg["data"], "n_blocks": n_blocks,
           "available_blocks": available, "complete": n_blocks == available,
           "channels_angstrom": [int(c) for c in CHANNELS], "bright_thresholds": thresholds.tolist(),
           "bands": INTENSITY_LABELS,
           "definitions": {"clamp": f"observed <= {MIN_OBS:g} DN/s", "min_obs": MIN_OBS,
                           "low_cluster": f"observed < {LOW_OBS:g} and reconstruction > {FACTOR:g} x max(observed, {FLOOR:g})",
                           "high_cluster": f"observed > {HIGH_OBS:g} and observed > {FACTOR:g} x max(reconstruction, {FLOOR:g})",
                           "dip_spike": f"observed / median of the 8 full-resolution neighbours < {DIP:g} (dip) or > {SPIKE:g} (spike)",
                           "valid": "finite solver DEM, six observations, both model DEMs and reconstructions",
                           "metrics": "DEM MSE and MAE over valid pixels x 18 bins against the solver; EM error = "
                                      "sum |EM_pred - EM_solver| / sum EM_solver; AIA pooled over six channels",
                           "regions": f"distance from ({SUN_CENTRE:g}, {SUN_CENTRE:g}) on the 4096 grid: disk < "
                                      f"{SUN_RADIUS - LIMB_HALF:g}, limb to {SUN_RADIUS + LIMB_HALF:g}, off-limb beyond"},
           "runtime_s": round(time.time() - t0, 1)}
    out.update(summarize_shared(acc) if args.mode == "shared" else summarize_tol(acc))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    stem = os.path.splitext(args.output)[0]
    if args.mode == "shared":
        merged = merge_examples(results, 1, max(args.examples, args.examples_high), 7)
        arrays = {}
        for (kind, ch), pool in merged.items():
            cap = args.examples if kind == "low" else args.examples_high
            pool = pool[:cap]
            for field in ("block", "i", "j", "band", "member", "obs", "med", "clamp", "ref", "lf", "sup",
                          "rec_lf", "rec_sup", "rec_sol"):
                arrays[f"{kind}_{ch}_{field}"] = np.array([item[field] for item in pool])
        np.savez_compressed(stem + ".npz", **arrays)
    text = render_shared(out) if args.mode == "shared" else render_tol(out)
    with open(stem + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.output} (+ .txt{', .npz' if args.mode == 'shared' else ''}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
