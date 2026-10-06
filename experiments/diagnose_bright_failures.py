#!/usr/bin/env python3
"""Explain where the label-free model disagrees with the solver reference.

One pass over the complete shared test set (every valid pixel, no sampling)
answers six questions, separately for each track:

1. Which AIA channel thresholds make a pixel Bright, and which trigger
   patterns carry the DEM error.  Bright is Samuel's union mask: any of the six
   observed AIA values at or above its fixed top-5% threshold.
2. Whether Bright pixels are *relatively* worse or merely carry larger DEMs.
   Squared error scales with DEM amplitude, so "relative SSE" divides the
   group's squared error by the squared norm of its reference DEMs.
3. What kind of error it is.  Per pixel the squared error splits exactly into a
   part *along* the reference curve (right shape, wrong overall scale) and a
   part *orthogonal* to it (emission placed at the wrong temperatures, e.g. a
   sharp solver spike spread over neighbouring bins).
4. Whether the model shrinks toward typical DEMs ("regression to the mean"):
   predicted versus reference peak height, emission and width, binned by the
   reference peak height.
5. Whether multi-peaked references explain the error.  Peaks use the project's
   standing definition: zero-padded curve, interior bins 1..16 only, prominence
   at least 15% of the curve maximum.
6. How much of the error is the solver's own irreproducibility.  The shared
   test set repeats each clean AIA input for five solver targets (clean solve
   plus four noise re-solves).  Blocks with bit-identical AIA are grouped, and
   for those pixels the mean squared error over targets splits exactly into
   |pred - mean_k y_k|^2 (model bias) + var_k(y_k) (solver scatter).  No
   deterministic predictor of the solver output can go below the scatter term.

Everything here is an association.  It cannot prove that a channel or a peak
shape physically causes the error.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

CHANNELS = ("94", "131", "171", "193", "211", "335")
N_BINS = 18
PROMINENCE = 0.15
# 64 spatial blocks x 5 solver targets per timestamp.  Worker ranges are
# aligned to this so a timestamp's repeated inputs normally stay in one worker.
BLOCK_ALIGN = 320

SCALARS = (["n", "sse", "along_sse", "orth_sse", "ref_sq", "pred_sq",
            "em_pred", "em_ref", "em_abs_err", "peak_pred", "peak_ref",
            "n_shape", "w1", "shift", "abs_shift", "width_pred", "width_ref",
            "log_peak_ratio", "log_em_ratio", "ref_multi", "model_multi",
            "bright", "n_active"]
           + [f"lead_{c}" for c in CHANNELS])
COL = {name: i for i, name in enumerate(SCALARS)}
SIGNED = len(SCALARS)
ABSOLUTE = SIGNED + N_BINS
SQUARED = ABSOLUTE + N_BINS
N_COLS = SQUARED + N_BINS

PAIR_COLS = ["n", "n_targets", "floor", "bias", "bias_along", "bias_orth",
             "mean_target_sse", "ybar_sq"]
PCOL = {name: i for i, name in enumerate(PAIR_COLS)}

# Per-group log10(per-pixel SSE) histogram, for medians and tail shares.
HIST_MIN, HIST_MAX, HIST_W = -12.0, 12.0, 0.02
N_HIST = int(round((HIST_MAX - HIST_MIN) / HIST_W))

# log10(max over channels of observed / threshold).  0 is the Bright boundary.
INTENSITY_EDGES = np.array([-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
INTENSITY_LABELS = ["<0.03x", "0.03-0.1x", "0.1-0.3x", "0.3-1x", "1-1.8x",
                    "1.8-3.2x", "3.2-10x", "10-32x", ">=32x"]
REFPEAK_EDGES = np.round(np.arange(-3.0, 3.51, 0.5), 2)
SSE_EDGES = np.round(np.arange(-8.0, 8.01, 0.25), 2)
SHAPE_LABELS = [f"ref {'multi' if r else 'uni'} / model {'multi' if m else 'uni'} / {'Bright' if b else 'Quiet'}"
                for r in (0, 1) for m in (0, 1) for b in (0, 1)]

SCHEMES = OrderedDict([("pattern", (64, True)), ("dominant", (7, True)),
                       ("intensity", (9, True)), ("shape", (8, True)),
                       ("ref_peak", (len(REFPEAK_EDGES) + 2, False)),
                       ("sse_tail", (len(SSE_EDGES) + 2, False))])
PAIR_SCHEMES = OrderedDict([("pattern", 64), ("intensity", 9)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model")
    p.add_argument("--data", help="root holding test_x.zarr and test_y.zarr")
    p.add_argument("--reference", choices=("bp", "enet"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--output", help="JSON path; a .txt summary is written beside it")
    p.add_argument("--max-blocks", type=int, default=None, help="smoke-test cap")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--pixel-batch", type=int, default=16384)
    p.add_argument("--targets-per-input", type=int, default=5)
    p.add_argument("--max-pending", type=int, default=160,
                   help="open repeated-input groups per worker before eviction")
    p.add_argument("--summarize", help="re-print the summary of an existing JSON")
    p.add_argument("--compare", nargs=2, metavar=("LABEL_FREE_JSON", "SUPERVISED_JSON"),
                   help="print label-free vs supervised side by side from two outputs")
    args = p.parse_args()
    if not (args.summarize or args.compare):
        missing = [k for k in ("model", "data", "reference", "output") if not getattr(args, k)]
        if missing:
            p.error("required: " + ", ".join("--" + k for k in missing))
    return args


# ── per-pixel quantities ──────────────────────────────────────────────────────

def interior_peak_counts(curves, frac=PROMINENCE):
    """Vectorised ``src.scaled_eval.count_peaks`` restricted to interior bins.

    Same as scipy.signal.find_peaks on the zero-padded curve with
    prominence >= frac * max: a peak is a rising edge followed by a (possibly
    flat) top and a strictly lower sample, located at the plateau midpoint;
    prominence scans outward from it until a strictly higher sample.  Only
    peaks in bins 1..T-2 are counted.  Returns counts for [P, T] curves.
    """
    P, T = curves.shape
    counts = np.zeros(P, dtype=np.int16)
    x = np.zeros((P, T + 2), dtype=np.float64)
    x[:, 1:-1] = curves
    mx = curves.max(axis=1)
    usable = np.isfinite(mx) & (mx > 0)
    run_end = np.empty((P, T + 2), dtype=np.int64)     # last index of equal run
    run_end[:, T + 1] = T + 1
    for k in range(T, -1, -1):
        run_end[:, k] = np.where(x[:, k] == x[:, k + 1], run_end[:, k + 1], k)
    rows_all, pos_all = [], []
    for i in range(1, T + 1):
        end = run_end[:, i]
        r = np.flatnonzero(usable & (x[:, i] > x[:, i - 1]) & (end < T + 1))
        if r.size == 0:
            continue
        end = end[r]
        down = x[r, end + 1] < x[r, i]
        rows_all.append(r[down])
        pos_all.append((i + end[down]) // 2)
    if not rows_all:
        return counts
    rows, pos = np.concatenate(rows_all), np.concatenate(pos_all)
    interior = (pos >= 2) & (pos <= T - 1)             # padded index -> bins 1..T-2
    rows, pos = rows[interior], pos[interior]
    xi = x[rows, pos]
    mins = []
    for step in (-1, 1):
        low = xi.copy()
        active = np.ones(rows.size, dtype=bool)
        for j in range(1, T + 2):
            q = pos + step * j
            inside = (q >= 0) & (q <= T + 1)
            v = x[rows, np.clip(q, 0, T + 1)]
            active &= inside & (v <= xi)
            if not active.any():
                break
            low = np.where(active, np.minimum(low, v), low)
        mins.append(low)
    prominence = xi - np.maximum(mins[0], mins[1])
    np.add.at(counts, rows[prominence >= frac * mx[rows]], 1)
    return counts


def pixel_features(p, y, logt, bits, lead):
    """Feature matrix [P, N_COLS] for predicted and reference DEMs [P, 18]."""
    P = p.shape[0]
    F = np.zeros((P, N_COLS), dtype=np.float64)
    d = p - y
    sse = np.square(d).sum(axis=1)
    yy = np.square(y).sum(axis=1)
    py = (p * y).sum(axis=1)
    safe_yy = np.where(yy > 0, yy, 1.0)
    along = np.where(yy > 0, np.square(py - yy) / safe_yy, 0.0)
    orth = np.clip(sse - along, 0.0, None)

    pc, yc = np.clip(p, 0, None), np.clip(y, 0, None)
    ps, ys = pc.sum(axis=1), yc.sum(axis=1)
    ok = (ps > 0) & (ys > 0)
    safe_ps, safe_ys = np.where(ok, ps, 1.0), np.where(ok, ys, 1.0)
    wp, wy = pc / safe_ps[:, None], yc / safe_ys[:, None]
    mu_p, mu_y = wp @ logt, wy @ logt
    sd_p = np.sqrt(np.clip(wp @ np.square(logt) - np.square(mu_p), 0, None))
    sd_y = np.sqrt(np.clip(wy @ np.square(logt) - np.square(mu_y), 0, None))
    w1 = (np.abs(np.cumsum(wp, axis=1)[:, :-1] - np.cumsum(wy, axis=1)[:, :-1])
          * np.diff(logt)[None]).sum(axis=1)
    peak_p, peak_y = p.max(axis=1), y.max(axis=1)
    safe_peak_p = np.where(ok & (peak_p > 0), peak_p, 1.0)
    safe_peak_y = np.where(ok & (peak_y > 0), peak_y, 1.0)

    ref_multi = interior_peak_counts(y) >= 2
    model_multi = interior_peak_counts(p) >= 2
    bright = bits != 0
    n_active = np.zeros(P, dtype=np.float64)
    for c in range(6):
        n_active += (bits >> c) & 1

    F[:, COL["n"]] = 1.0
    F[:, COL["sse"]] = sse
    F[:, COL["along_sse"]] = along
    F[:, COL["orth_sse"]] = orth
    F[:, COL["ref_sq"]] = yy
    F[:, COL["pred_sq"]] = np.square(p).sum(axis=1)
    F[:, COL["em_pred"]] = p.sum(axis=1)
    F[:, COL["em_ref"]] = y.sum(axis=1)
    F[:, COL["em_abs_err"]] = np.abs(p.sum(axis=1) - y.sum(axis=1))
    F[:, COL["peak_pred"]] = peak_p
    F[:, COL["peak_ref"]] = peak_y
    F[:, COL["n_shape"]] = ok
    F[:, COL["w1"]] = np.where(ok, w1, 0.0)
    F[:, COL["shift"]] = np.where(ok, mu_p - mu_y, 0.0)
    F[:, COL["abs_shift"]] = np.where(ok, np.abs(mu_p - mu_y), 0.0)
    F[:, COL["width_pred"]] = np.where(ok, sd_p, 0.0)
    F[:, COL["width_ref"]] = np.where(ok, sd_y, 0.0)
    F[:, COL["log_peak_ratio"]] = np.where(ok, np.log10(safe_peak_p / safe_peak_y), 0.0)
    F[:, COL["log_em_ratio"]] = np.where(ok, np.log10(safe_ps / safe_ys), 0.0)
    F[:, COL["ref_multi"]] = ref_multi
    F[:, COL["model_multi"]] = model_multi
    F[:, COL["bright"]] = bright
    F[:, COL["n_active"]] = n_active
    for c, name in enumerate(CHANNELS):
        F[:, COL[f"lead_{name}"]] = bright & (lead == c)
    F[:, SIGNED:ABSOLUTE] = d
    F[:, ABSOLUTE:SQUARED] = np.abs(d)
    F[:, SQUARED:] = np.square(d)
    return F, sse, ref_multi, model_multi


def bin_positive(values, edges):
    """0 for non-positive values, else 1 + position among log10 edges."""
    out = np.zeros(values.shape, dtype=np.int64)
    pos = values > 0
    out[pos] = 1 + np.searchsorted(edges, np.log10(values[pos]), side="right")
    return out


# ── accumulators ──────────────────────────────────────────────────────────────

def new_acc():
    acc = {"sums": {}, "hist_n": {}, "hist_sse": {}, "pair": {},
           "stats": {"blocks": 0, "predicted_blocks": 0, "valid_pixels": 0,
                     "groups_by_n": {}, "evicted_groups": 0}}
    for name, (groups, with_hist) in SCHEMES.items():
        acc["sums"][name] = np.zeros((groups, N_COLS))
        if with_hist:
            acc["hist_n"][name] = np.zeros((groups, N_HIST), dtype=np.int64)
            acc["hist_sse"][name] = np.zeros((groups, N_HIST))
    for name, groups in PAIR_SCHEMES.items():
        acc["pair"][name] = np.zeros((groups, len(PAIR_COLS)))
    return acc


def merge_acc(into, other):
    for part in ("sums", "hist_n", "hist_sse", "pair"):
        for name, value in other[part].items():
            into[part][name] += value
    for key in ("blocks", "predicted_blocks", "valid_pixels", "evicted_groups"):
        into["stats"][key] += other["stats"][key]
    for n, count in other["stats"]["groups_by_n"].items():
        into["stats"]["groups_by_n"][n] = into["stats"]["groups_by_n"].get(n, 0) + count


def group_add(table, groups, F):
    if groups.size == 0:
        return
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    table[g[starts]] += np.add.reduceat(F[order], starts, axis=0)


def hist_add(counts, sums, groups, sse):
    if groups.size == 0:
        return
    with np.errstate(divide="ignore"):
        idx = np.floor((np.log10(sse) - HIST_MIN) / HIST_W)
    idx = np.clip(np.where(np.isfinite(idx), idx, 0), 0, N_HIST - 1).astype(np.int64)
    flat = groups.astype(np.int64) * N_HIST + idx
    size = counts.size
    counts += np.bincount(flat, minlength=size).reshape(counts.shape)
    sums += np.bincount(flat, weights=sse, minlength=size).reshape(sums.shape)


def finalize_group(acc, grp):
    """Solver-scatter decomposition for one set of bit-identical AIA inputs."""
    n = grp["n"]
    acc["stats"]["groups_by_n"][n] = acc["stats"]["groups_by_n"].get(n, 0) + 1
    if n < 2:
        return
    idx = np.flatnonzero(grp["valid"].ravel())
    if idx.size == 0:
        return
    sy = grp["sum_y"].reshape(N_BINS, -1)[:, idx].T
    sy2 = grp["sum_y2"].reshape(N_BINS, -1)[:, idx].T
    ybar = sy / n
    floor = np.clip(sy2 / n - np.square(ybar), 0, None).sum(axis=1)
    p = grp["pred"].reshape(N_BINS, -1)[:, idx].T.astype(np.float64)
    bias = np.square(p - ybar).sum(axis=1)
    yy = np.square(ybar).sum(axis=1)
    py = (p * ybar).sum(axis=1)
    along = np.where(yy > 0, np.square(py - yy) / np.where(yy > 0, yy, 1.0), 0.0)
    F = np.column_stack([np.ones(idx.size), np.full(idx.size, float(n)), floor, bias,
                         along, np.clip(bias - along, 0, None),
                         grp["sse_sum"].ravel()[idx] / n, yy])
    group_add(acc["pair"]["pattern"], grp["bits"].ravel()[idx], F)
    group_add(acc["pair"]["intensity"], grp["igroup"].ravel()[idx], F)


def combine_group(into, other):
    into["n"] += other["n"]
    into["valid"] &= other["valid"]
    for key in ("sum_y", "sum_y2", "sse_sum"):
        into[key] += other[key]


# ── worker ────────────────────────────────────────────────────────────────────

def load_predictor(path, threads):
    """Return (predict(obs[6,H,W]) -> DEM[18,H,W], description).

    Label-free mlp6 checkpoints come from train_scaled.py; supervised ones are
    Samuel's .pth files, recognised by their ``model_name`` key and run exactly
    as the published Table-1 script does (regression branch only, AIA clamped
    at zero inside predict_regression_only).
    """
    import torch
    torch.set_num_threads(threads)
    # DEM_EVAL_DEVICE=cuda runs label-free checkpoints on the GPU (needed for the
    # ~10M-parameter models, ~25x the CPU cost of the 360k ones); supervised
    # checkpoints always run on the CPU exactly as the published script does.
    device = torch.device(os.environ.get("DEM_EVAL_DEVICE", "cpu"))
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_name" in raw:
        from student_package.table1_dem_metrics.compute_paper_table_metrics import (
            load_model, predict_regression_only)
        model, meta = load_model(path, torch.device("cpu"))

        def predict(obs, pixel_batch):
            C, H, W = obs.shape
            points = np.ascontiguousarray(obs.reshape(C, -1))
            out = np.empty((N_BINS, H * W), dtype=np.float32)
            with torch.no_grad():
                for s in range(0, H * W, pixel_batch):
                    e = min(s + pixel_batch, H * W)
                    batch = torch.from_numpy(points[:, s:e])[None, :, None, :]
                    out[:, s:e] = predict_regression_only(model, batch)[0, :N_BINS, 0, :].numpy()
            return out.reshape(N_BINS, H, W)
        return predict, {"kind": "supervised", **meta}

    from experiments.train_scaled import load_operators
    from src.scaled_eval import load_scaled_model
    _, basis_t, n_basis, _ = load_operators(device)
    model, ckpt = load_scaled_model(path, n_basis, device)
    if ckpt["variant"] != "mlp6":
        raise ValueError(f"only the center-pixel mlp6 is supported, got {ckpt['variant']}")

    def predict(obs, pixel_batch):
        return predict_dem(model, basis_t, obs, ckpt["patch_size"], pixel_batch)
    return predict, {"kind": "label_free", **{k: ckpt.get(k) for k in ("variant", "loss", "hidden", "epoch")}}


def predict_dem(model, basis_t, obs, patch_size, pixel_batch):
    import torch
    C, H, W = obs.shape
    flat = obs.reshape(C, -1).T
    out = np.empty((H * W, N_BINS), dtype=np.float32)
    # mlp6 reads only the patch centre, so the other patch pixels stay zero.
    k, c = patch_size, patch_size // 2
    with torch.no_grad():
        for s in range(0, H * W, pixel_batch):
            e = min(s + pixel_batch, H * W)
            patch = np.zeros((e - s, C, k, k), dtype=np.float32)
            patch[:, :, c, c] = flat[s:e]
            coeff = model(torch.from_numpy(patch).to(basis_t.device))
            out[s:e] = (coeff @ basis_t.T)[:, :N_BINS].cpu().numpy()
    return out.reshape(H, W, N_BINS).transpose(2, 0, 1)


def run_range(job):
    cfg, start, stop, wid = job
    import zarr
    predict, _ = load_predictor(cfg["model"], cfg["threads"])
    logt = np.asarray(np.load("RData.npz")["logT"][:N_BINS], dtype=np.float64)
    x = zarr.open(os.path.join(cfg["data"], "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(cfg["data"], "test_y.zarr"), mode="r")
    thr = np.asarray(cfg["thresholds"], dtype=np.float32)[:, None, None]
    weights = (1 << np.arange(6, dtype=np.int64))[:, None, None]

    acc = new_acc()
    pending = OrderedDict()
    t0 = time.time()
    for b in range(start, stop):
        xb = np.ascontiguousarray(np.asarray(x[:, :, :, b], dtype=np.float32))
        key = hashlib.blake2b(xb.tobytes(), digest_size=16).hexdigest()
        grp = pending.get(key)
        obs = np.ascontiguousarray(xb[:, ::2, ::2])
        if grp is None:
            pred = predict(obs, cfg["pixel_batch"])
            acc["stats"]["predicted_blocks"] += 1
        else:
            pred = grp["pred"]
        truth = np.asarray(y[:N_BINS, :, :, b], dtype=np.float32)
        # Same validity rule as eval_full_paper_test.py, so counts reconcile
        # with the published Full/Bright/Quiet rows.
        valid = np.isfinite(truth).all(axis=0) & np.isfinite(pred).all(axis=0)

        hit = obs >= thr
        bits = (hit * weights).sum(axis=0)
        bright_map = bits != 0
        ratio = np.where(np.isfinite(obs), obs / thr, -np.inf)
        lead = ratio.argmax(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            logr = np.log10(ratio.max(axis=0))
        logr = np.where(np.isfinite(logr), logr, -np.inf)
        igroup = np.searchsorted(INTENSITY_EDGES, logr, side="right")
        igroup = np.where(bright_map, np.maximum(igroup, 4), np.minimum(igroup, 3))

        idx = np.flatnonzero(valid.ravel())
        sse_map = np.zeros(valid.size)
        if idx.size:
            P = pred.reshape(N_BINS, -1)[:, idx].T.astype(np.float64)
            Y = truth.reshape(N_BINS, -1)[:, idx].T.astype(np.float64)
            bb = bits.ravel()[idx]
            ld = lead.ravel()[idx]
            F, sse, ref_multi, model_multi = pixel_features(P, Y, logt, bb, ld)
            keys = {"pattern": bb,
                    "dominant": np.where(bb != 0, ld, 6),
                    "intensity": igroup.ravel()[idx],
                    "shape": ref_multi.astype(np.int64) * 4 + model_multi.astype(np.int64) * 2 + (bb != 0),
                    "ref_peak": bin_positive(Y.max(axis=1), REFPEAK_EDGES),
                    "sse_tail": bin_positive(sse, SSE_EDGES)}
            for name, (_, with_hist) in SCHEMES.items():
                group_add(acc["sums"][name], keys[name], F)
                if with_hist:
                    hist_add(acc["hist_n"][name], acc["hist_sse"][name], keys[name], sse)
            sse_map[idx] = sse
            acc["stats"]["valid_pixels"] += int(idx.size)

        if grp is None:
            shape = (N_BINS,) + valid.shape
            grp = {"pred": pred, "valid": valid.copy(), "n": 0, "bits": bits,
                   "igroup": igroup, "sum_y": np.zeros(shape), "sum_y2": np.zeros(shape),
                   "sse_sum": np.zeros(valid.shape)}
            pending[key] = grp
        tz = np.where(valid[None], truth, 0).astype(np.float64)
        grp["sum_y"] += tz
        grp["sum_y2"] += np.square(tz)
        grp["sse_sum"] += sse_map.reshape(valid.shape)
        grp["valid"] &= valid
        grp["n"] += 1
        if grp["n"] == cfg["targets"]:
            finalize_group(acc, pending.pop(key))
        while len(pending) > cfg["max_pending"]:
            _, old = pending.popitem(last=False)
            acc["stats"]["evicted_groups"] += 1
            finalize_group(acc, old)

        acc["stats"]["blocks"] += 1
        done = b - start + 1
        if done % 500 == 0 or b + 1 == stop:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[w{wid}] {done:,}/{stop - start:,} blocks  {rate:.1f} blocks/s", flush=True)

    # Groups still open may continue in a neighbouring worker's range.  Return
    # them for merging unless eviction shows repeats are not local, in which
    # case shipping them back would only move a lot of memory.
    if acc["stats"]["evicted_groups"]:
        for grp in pending.values():
            finalize_group(acc, grp)
        return acc, []
    return acc, list(pending.items())


# ── summaries ─────────────────────────────────────────────────────────────────

def hist_stats(hist_n, hist_sse, n, total):
    cum = np.cumsum(hist_n)
    centre = lambda k: float(10 ** (HIST_MIN + (k + 0.5) * HIST_W))
    out = {f"sse_p{q:g}": centre(int(np.searchsorted(cum, q / 100 * n))) for q in (50, 90, 99, 99.9)}
    want, got, tail = 0.01 * n, 0.0, 0.0
    for k in range(N_HIST - 1, -1, -1):
        c = hist_n[k]
        if c == 0:
            continue
        take = min(c, want - got)
        tail += hist_sse[k] * take / c
        got += take
        if got >= want:
            break
    out["worst_1pct_sse_share_pct"] = 100 * tail / total if total > 0 else None
    return out


def summarize(row, total_n, total_sse, hist_n=None, hist_sse=None, vectors=True):
    n = row[COL["n"]]
    if n <= 0:
        return None
    ratio = lambda a, b: float(a / b) if b != 0 else None
    sse = row[COL["sse"]]
    ns = row[COL["n_shape"]]
    bright = row[COL["bright"]]
    out = {
        "n_pixels": int(round(n)),
        "pct_of_pixels": 100 * n / total_n,
        "dem_mse": sse / (n * N_BINS),
        "sse_share_pct": 100 * sse / total_sse if total_sse > 0 else None,
        "relative_sse": ratio(sse, row[COL["ref_sq"]]),
        "along_reference_share_pct": ratio(100 * row[COL["along_sse"]], sse),
        "orthogonal_share_pct": ratio(100 * row[COL["orth_sse"]], sse),
        "em_ratio_pred_over_ref": ratio(row[COL["em_pred"]], row[COL["em_ref"]]),
        "em_abs_rel_err_pct": ratio(100 * row[COL["em_abs_err"]], abs(row[COL["em_ref"]])),
        "peak_ratio_pred_over_ref": ratio(row[COL["peak_pred"]], row[COL["peak_ref"]]),
        "geomean_peak_ratio": float(10 ** (row[COL["log_peak_ratio"]] / ns)) if ns else None,
        "geomean_em_ratio": float(10 ** (row[COL["log_em_ratio"]] / ns)) if ns else None,
        "mean_logt_shift_dex": ratio(row[COL["shift"]], ns),
        "mean_abs_logt_shift_dex": ratio(row[COL["abs_shift"]], ns),
        "width_pred_dex": ratio(row[COL["width_pred"]], ns),
        "width_ref_dex": ratio(row[COL["width_ref"]], ns),
        "w1_dex": ratio(row[COL["w1"]], ns),
        "ref_multimodal_pct": 100 * row[COL["ref_multi"]] / n,
        "model_multimodal_pct": 100 * row[COL["model_multi"]] / n,
        "bright_pct": 100 * bright / n,
        "mean_trigger_channels": row[COL["n_active"]] / n,
    }
    if bright > 0:
        out["dominant_channel_pct_of_bright"] = {
            c: 100 * row[COL[f"lead_{c}"]] / bright for c in CHANNELS}
    if hist_n is not None:
        out.update(hist_stats(hist_n, hist_sse, n, sse))
    if vectors:
        out["mean_signed_error_by_logt"] = (row[SIGNED:ABSOLUTE] / n).tolist()
        out["mae_by_logt"] = (row[ABSOLUTE:SQUARED] / n).tolist()
        out["mse_by_logt"] = (row[SQUARED:] / n).tolist()
    return out


def summarize_pair(row):
    n = row[PCOL["n"]]
    if n <= 0:
        return None
    mts, floor, bias = row[PCOL["mean_target_sse"]], row[PCOL["floor"]], row[PCOL["bias"]]
    return {"n_pixels": int(round(n)),
            "mean_targets_per_pixel": row[PCOL["n_targets"]] / n,
            "model_mse": mts / (n * N_BINS),
            "solver_scatter_mse": floor / (n * N_BINS),
            "model_bias_mse": bias / (n * N_BINS),
            "scatter_share_pct": 100 * floor / mts if mts > 0 else None,
            "bias_along_share_pct": 100 * row[PCOL["bias_along"]] / bias if bias > 0 else None,
            "bias_orthogonal_share_pct": 100 * row[PCOL["bias_orth"]] / bias if bias > 0 else None,
            "identity_check_rel": abs(mts - floor - bias) / mts if mts > 0 else None}


def pattern_label(bits):
    names = [f"{CHANNELS[c]}" for c in range(6) if bits & (1 << c)]
    return "+".join(names) if names else "quiet"


def build_output(acc, meta):
    S, Hn, Hs = acc["sums"], acc["hist_n"], acc["hist_sse"]
    pat, pat_n, pat_s = S["pattern"], Hn["pattern"], Hs["pattern"]
    total_n = pat[:, COL["n"]].sum()
    total_sse = pat[:, COL["sse"]].sum()
    bright_rows = np.arange(1, 64)
    view = lambda sel: summarize(pat[sel].sum(axis=0), total_n, total_sse,
                                 pat_n[sel].sum(axis=0), pat_s[sel].sum(axis=0))

    popcount = np.array([bin(b).count("1") for b in range(64)])
    out = dict(meta)
    out["population"] = {"full": view(np.arange(64)), "bright": view(bright_rows),
                         "quiet": view(np.array([0]))}
    out["by_trigger_channel_inclusive"] = {
        c: view(np.array([b for b in range(64) if b & (1 << i)])) for i, c in enumerate(CHANNELS)}
    out["by_number_of_trigger_channels"] = {
        str(k): view(np.flatnonzero(popcount == k)) for k in range(7) if pat[popcount == k, 0].sum() > 0}
    exact = {pattern_label(b): summarize(pat[b], total_n, total_sse, pat_n[b], pat_s[b], vectors=False)
             for b in bright_rows if pat[b, COL["n"]] > 0}
    out["by_trigger_pattern_exact"] = dict(sorted(
        exact.items(), key=lambda kv: -(kv[1]["sse_share_pct"] or 0)))
    out["by_dominant_channel"] = {
        (CHANNELS[g] if g < 6 else "quiet"): summarize(S["dominant"][g], total_n, total_sse,
                                                        Hn["dominant"][g], Hs["dominant"][g])
        for g in range(7) if S["dominant"][g, COL["n"]] > 0}
    out["by_intensity_over_threshold"] = {
        INTENSITY_LABELS[g]: summarize(S["intensity"][g], total_n, total_sse,
                                       Hn["intensity"][g], Hs["intensity"][g])
        for g in range(9) if S["intensity"][g, COL["n"]] > 0}
    out["by_dem_shape"] = {
        SHAPE_LABELS[g]: summarize(S["shape"][g], total_n, total_sse, Hn["shape"][g], Hs["shape"][g])
        for g in range(8) if S["shape"][g, COL["n"]] > 0}
    rp_labels = (["<=0"] + [f"<{10 ** REFPEAK_EDGES[0]:.2g}"]
                 + [f"{10 ** a:.2g} to {10 ** b:.2g}" for a, b in zip(REFPEAK_EDGES[:-1], REFPEAK_EDGES[1:])]
                 + [f">={10 ** REFPEAK_EDGES[-1]:.2g}"])
    out["calibration_by_reference_peak"] = {
        rp_labels[g]: summarize(S["ref_peak"][g], total_n, total_sse, vectors=False)
        for g in range(len(rp_labels)) if S["ref_peak"][g, COL["n"]] > 0}

    tail = S["sse_tail"]
    edges = np.r_[-np.inf, SSE_EDGES]
    tail_out = {}
    for pct in (0.1, 1.0):
        want, rows, g = pct / 100 * total_n, np.zeros(N_COLS), tail.shape[0] - 1
        while g > 0 and rows[COL["n"]] < want:
            rows = rows + tail[g]
            g -= 1
        threshold = float(10 ** edges[g]) if g >= 1 else 0.0
        row = summarize(rows, total_n, total_sse, vectors=True)
        if row:
            row["pixel_sse_at_least"] = threshold
        tail_out[f"worst_{pct:g}pct_approx"] = row
    rest = tail.sum(axis=0) - (tail[g + 1:].sum(axis=0))
    tail_out["remainder"] = summarize(rest, total_n, total_sse, vectors=True)
    out["worst_tail_composition"] = tail_out

    pair = acc["pair"]
    pv = lambda sel: summarize_pair(pair["pattern"][sel].sum(axis=0))
    out["solver_scatter"] = {
        "note": ("model MSE over repeated targets = model bias (distance to the mean solver DEM) "
                 "+ solver scatter (variance of the solver DEM across targets)"),
        "groups_by_targets": {str(k): v for k, v in sorted(acc["stats"]["groups_by_n"].items())},
        "evicted_groups": acc["stats"]["evicted_groups"],
        "population": {"full": pv(np.arange(64)), "bright": pv(bright_rows), "quiet": pv(np.array([0]))},
        "by_intensity_over_threshold": {
            INTENSITY_LABELS[g]: summarize_pair(pair["intensity"][g]) for g in range(9)
            if pair["intensity"][g, 0] > 0},
    }
    out["stats"] = {k: v for k, v in acc["stats"].items() if k != "groups_by_n"}
    return out


def fmt(v, spec):
    if v is None:
        return "-"
    return format(v, spec)


def table(title, rows, cols):
    lines = [f"\n## {title}"]
    head = f"{'group':<30}" + "".join(f" {label:>{w}}" for label, _, w, _ in cols)
    lines += [head, "-" * len(head)]
    for name, row in rows.items():
        if row is None:
            continue
        lines.append(f"{name[:30]:<30}" + "".join(
            f" {fmt(row.get(key), spec):>{w}}" for _, key, w, spec in cols))
    return lines


MAIN_COLS = [("pix%", "pct_of_pixels", 7, ".2f"), ("SSE%", "sse_share_pct", 7, ".1f"),
             ("MSE", "dem_mse", 9, ".3g"), ("relSSE", "relative_sse", 9, ".3g"),
             ("orth%", "orthogonal_share_pct", 7, ".0f"), ("EMr", "em_ratio_pred_over_ref", 7, ".3g"),
             ("pkR", "peak_ratio_pred_over_ref", 7, ".3g"), ("wP", "width_pred_dex", 6, ".2f"),
             ("wR", "width_ref_dex", 6, ".2f"), ("shift", "mean_logt_shift_dex", 7, "+.2f"),
             ("refM%", "ref_multimodal_pct", 7, ".1f"), ("modM%", "model_multimodal_pct", 7, ".1f"),
             ("p50", "sse_p50", 9, ".3g"), ("top1%", "worst_1pct_sse_share_pct", 7, ".1f")]
PAIR_TABLE = [("pix%", None, 7, ".2f"), ("modelMSE", "model_mse", 10, ".3g"),
              ("scatterMSE", "solver_scatter_mse", 11, ".3g"), ("biasMSE", "model_bias_mse", 10, ".3g"),
              ("scatter%", "scatter_share_pct", 9, ".1f"), ("biasOrth%", "bias_orthogonal_share_pct", 10, ".0f"),
              ("check", "identity_check_rel", 9, ".1e")]


def render_summary(out):
    kind = (out.get("model_info") or {}).get("kind", "label_free").replace("_", "-")
    L = [f"# Bright-failure diagnostic: {out['reference'].upper()} track, {kind} model",
         f"model {out['model']}", f"blocks {out['n_blocks']:,}   valid pixel-targets {out['stats']['valid_pixels']:,}",
         "",
         "Columns: pix% share of pixels | SSE% share of total DEM squared error | MSE per bin |",
         "relSSE error energy / reference energy | orth% of SSE orthogonal to the reference curve",
         "(temperature placement) vs along it (scale) | EMr, pkR aggregate predicted/reference emission",
         "and peak height | wP, wR EM-weighted logT width (dex) | shift mean logT pred-ref (dex) |",
         "refM%, modM% multi-peaked | p50 median per-pixel SSE | top1% SSE share of the worst 1%"]
    L += table("Population", out["population"], MAIN_COLS)
    L += table("By intensity (max over channels of observed/threshold)", out["by_intensity_over_threshold"], MAIN_COLS)
    L += table("Bright, by channel with the largest observed/threshold", out["by_dominant_channel"], MAIN_COLS)
    L += table("Bright, inclusive by trigger channel (overlapping)", out["by_trigger_channel_inclusive"], MAIN_COLS)
    L += table("By number of trigger channels", out["by_number_of_trigger_channels"], MAIN_COLS)
    top = dict(list(out["by_trigger_pattern_exact"].items())[:12])
    L += table("Top exact trigger patterns by SSE share", top, MAIN_COLS)
    L += table("By DEM shape (reference x model x Bright)", out["by_dem_shape"], MAIN_COLS)
    L += table("Regression-to-mean check, by reference peak height", out["calibration_by_reference_peak"],
               [c for c in MAIN_COLS if c[1] not in ("sse_p50", "worst_1pct_sse_share_pct")]
               + [("gPkR", "geomean_peak_ratio", 7, ".3g"), ("gEMr", "geomean_em_ratio", 7, ".3g")])
    tail = out["worst_tail_composition"]
    L += table("Worst per-pixel SSE tail (bin-edge approximate)", tail,
               [c for c in MAIN_COLS if c[1] not in ("sse_p50", "worst_1pct_sse_share_pct")]
               + [("bright%", "bright_pct", 8, ".1f")])
    for name in ("worst_0.1pct_approx", "worst_1pct_approx"):
        row = tail.get(name)
        if row and row.get("dominant_channel_pct_of_bright"):
            dom = ", ".join(f"{c}:{v:.0f}%" for c, v in row["dominant_channel_pct_of_bright"].items())
            L.append(f"  {name}: pixel SSE >= {row['pixel_sse_at_least']:.3g}; dominant channel of Bright part: {dom}")
    sc = out["solver_scatter"]
    L.append(f"\n## Solver scatter (repeated AIA inputs)\ngroups by number of targets: {sc['groups_by_targets']}; "
             f"evicted: {sc['evicted_groups']}")
    total = (sc["population"]["full"] or {}).get("n_pixels") or 0
    for part in ("population", "by_intensity_over_threshold"):
        rows = {k: (dict(v, **{"pix%": 100 * v["n_pixels"] / total}) if v and total else v)
                for k, v in sc[part].items()}
        cols = [("pix%", "pix%", 7, ".2f")] + PAIR_TABLE[1:]
        L += table(f"Solver scatter, {part.replace('_', ' ')}", rows, cols)
    L.append("\nSigned DEM error by logT bin (pred - ref, mean per pixel):")
    logt = out["logt"]
    for name in ("bright", "quiet"):
        row = out["population"][name]
        if row:
            L.append(f"  {name:<7}" + " ".join(f"{v:+.2f}" for v in row["mean_signed_error_by_logt"]))
    L.append("  logT   " + " ".join(f"{t:5.2f}" for t in logt))
    return "\n".join(L)


# ── main ──────────────────────────────────────────────────────────────────────

COMPARE_COLS = [("MSE", "dem_mse", ".3g"), ("relSSE", "relative_sse", ".3f"),
                ("EMr", "em_ratio_pred_over_ref", ".3f"), ("pkR", "peak_ratio_pred_over_ref", ".3f"),
                ("orth%", "orthogonal_share_pct", ".0f"), ("modM%", "model_multimodal_pct", ".1f"),
                ("p50", "sse_p50", ".3g")]


def render_compare(lf, sup):
    """Label-free (LF) and supervised (SUP) side by side on the same pixels."""
    L = [f"# Label-free vs supervised, {lf['reference'].upper()} track",
         "Same pixels, Bright mask and validity rule in both runs. LF/SUP is the ratio of",
         "DEM MSE; below 1 means the label-free model is closer to the solver.",
         f"LF pixels {lf['stats']['valid_pixels']:,}; SUP pixels {sup['stats']['valid_pixels']:,}"]
    sections = [("Population", "population"), ("By intensity over threshold", "by_intensity_over_threshold"),
                ("Bright, by dominant channel", "by_dominant_channel"), ("By DEM shape", "by_dem_shape"),
                ("By reference peak height", "calibration_by_reference_peak")]
    for title, key in sections:
        L.append(f"\n## {title}")
        head = f"{'group':<30} {'pix%':>6} {'LF/SUP':>7}" + "".join(
            f" {'LF ' + c:>11} {'SUP ' + c:>11}" for c, _, _ in COMPARE_COLS)
        L += [head, "-" * len(head)]
        for name, a in lf[key].items():
            b = sup[key].get(name)
            if not a or not b:
                continue
            ratio = a["dem_mse"] / b["dem_mse"] if b["dem_mse"] else None
            L.append(f"{name[:30]:<30} {a['pct_of_pixels']:>6.2f} {fmt(ratio, '.3g'):>7}" + "".join(
                f" {fmt(a.get(k), spec):>11} {fmt(b.get(k), spec):>11}" for _, k, spec in COMPARE_COLS))
    for name, out in (("LF", lf), ("SUP", sup)):
        sc = out["solver_scatter"]["population"]
        for part in ("full", "bright", "quiet"):
            row = sc.get(part)
            if row:
                L.append(f"{name:<4} {part:<7} model MSE {row['model_mse']:.4g} = bias {row['model_bias_mse']:.4g}"
                         f" + solver scatter {row['solver_scatter_mse']:.4g} ({row['scatter_share_pct']:.1f}%)")
    return "\n".join(L)


def split_ranges(n_blocks, workers):
    chunk = math.ceil(n_blocks / max(workers, 1))
    if n_blocks > BLOCK_ALIGN * workers:
        chunk = math.ceil(chunk / BLOCK_ALIGN) * BLOCK_ALIGN
    return [(s, min(s + chunk, n_blocks)) for s in range(0, n_blocks, chunk)]


def main():
    args = parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            print(render_summary(json.load(f)))
        return
    if args.compare:
        with open(args.compare[0]) as f, open(args.compare[1]) as g:
            print(render_compare(json.load(f), json.load(g)))
        return
    import zarr
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload) if isinstance(payload, dict) else payload,
                            dtype=np.float64)
    if thresholds.shape != (6,) or not (thresholds > 0).all():
        raise ValueError("thresholds must be six positive values")
    x = zarr.open(os.path.join(args.data, "test_x.zarr"), mode="r")
    available = int(x.shape[-1])
    n_blocks = min(available, args.max_blocks) if args.max_blocks else available
    cfg = {"model": os.path.abspath(args.model), "data": os.path.abspath(args.data),
           "thresholds": thresholds.tolist(), "threads": args.threads_per_worker,
           "pixel_batch": args.pixel_batch, "targets": args.targets_per_input,
           "max_pending": args.max_pending}
    ranges = split_ranges(n_blocks, args.workers)
    jobs = [(cfg, s, e, i) for i, (s, e) in enumerate(ranges)]
    print(f"blocks {n_blocks:,} of {available:,}; {len(jobs)} worker range(s): {ranges}", flush=True)

    t0 = time.time()
    if len(jobs) == 1:
        results = [run_range(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(run_range, jobs)
    acc, leftovers = new_acc(), OrderedDict()
    for part, left in results:
        merge_acc(acc, part)
        for key, grp in left:
            if key in leftovers:
                combine_group(leftovers[key], grp)
            else:
                leftovers[key] = grp
    for grp in leftovers.values():
        finalize_group(acc, grp)

    _, model_info = load_predictor(cfg["model"], 1)
    logt = np.load("RData.npz")["logT"]
    meta = {"purpose": "Bright-failure association diagnostic; not a causal inference",
            "reference": args.reference, "model": cfg["model"], "model_info": model_info,
            "data": cfg["data"],
            "n_blocks": n_blocks, "available_blocks": available, "complete_test": n_blocks == available,
            "bright_thresholds": thresholds.tolist(), "channels_angstrom": [int(c) for c in CHANNELS],
            "definitions": {
                "bright": "any observed AIA channel >= its threshold; quiet is the valid complement",
                "valid": "finite reference DEM and finite prediction (as eval_full_paper_test.py)",
                "multimodal": "zero-padded curve, >=2 interior peaks with prominence >= 0.15 x curve max",
                "along/orthogonal": "exact split of per-pixel SSE: component of (pred-ref) along ref, and the rest",
                "relative_sse": "group SSE / group sum of squared reference DEM",
                "intensity": "max over channels of observed / threshold",
                "solver_scatter": "bit-identical AIA blocks grouped; per-bin variance of solver DEM across targets"},
            "runtime_s": round(time.time() - t0, 1),
            "logt": np.asarray(logt[:N_BINS], dtype=float).tolist()}
    out = build_output(acc, meta)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    text = render_summary(out)
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.output} and its .txt summary in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
