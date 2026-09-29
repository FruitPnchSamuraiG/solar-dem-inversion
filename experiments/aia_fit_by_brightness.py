#!/usr/bin/env python3
"""Does each DEM reproduce the observed AIA, by brightness band?

Follow-up to diagnose_bright_failures.py.  At flare cores the label-free model
predicts far less emission than the solver.  This asks whether it also fails to
reproduce the six observed AIA intensities there, i.e. whether it is failing its
own training objective (a training problem) or fits the AIA with a different,
smaller DEM (the objective allows the wrong answer).

For the label-free model, the supervised model and the solver itself, each DEM
is projected through the AIA response (the viewer's 18-bin, 1e26 scaling, as in
eval_aia_shared_test.py) and compared with the observation, per channel and per
brightness band (max over channels of observed / Bright threshold).  The mask
matches eval_aia_shared_test.py, so pooled MAE/MSE reconcile with the website.

It also keeps a seeded random sample of pixels per band (reservoir sampling)
with all three DEMs, for example-curve figures.

Evaluation only; no training.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.diagnose_bright_failures import (CHANNELS, INTENSITY_EDGES, INTENSITY_LABELS,
                                                  N_BINS, load_predictor, split_ranges)

SOURCES = ("label_free", "supervised", "solver")
STATS = ("n", "obs", "rec", "abs", "sq", "logr", "n_logr", "under", "over")
SI = {s: i for i, s in enumerate(STATS)}
SAMPLE_GROUPS = {"quiet": (0, 1, 2, 3), "ordinary_bright": (4, 5, 6),
                 "very_bright": (7,), "flare_core": (8,)}
CACHE = 256          # distinct AIA blocks whose predictions are kept for repeats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label-free")
    p.add_argument("--supervised")
    p.add_argument("--data")
    p.add_argument("--reference", choices=("bp", "enet"))
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--output", help="JSON path; examples go to the same stem + .npz")
    p.add_argument("--max-blocks", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--pixel-batch", type=int, default=16384)
    p.add_argument("--samples", type=int, default=40, help="pixels kept per band group per worker")
    p.add_argument("--summarize", help="re-print the summary of an existing JSON")
    args = p.parse_args()
    if not args.summarize:
        missing = [k for k in ("label_free", "supervised", "data", "reference", "output")
                   if not getattr(args, k)]
        if missing:
            p.error("required: " + ", ".join("--" + k.replace("_", "-") for k in missing))
    return args


def reservoir_add(res, seen, rng, k, items):
    """Algorithm R over a batch of candidate items (list of dicts)."""
    m = len(items)
    if m == 0:
        return seen
    t = seen + np.arange(1, m + 1)
    u = np.floor(rng.random(m) * t).astype(np.int64)
    for i in range(m):
        if len(res) < k:
            res.append(items[i]())
        elif u[i] < k:
            res[u[i]] = items[i]()
    return seen + m


def run_range(job):
    cfg, start, stop, wid = job
    import zarr
    lf_predict, _ = load_predictor(cfg["label_free"], cfg["threads"])
    sup_predict, _ = load_predictor(cfg["supervised"], cfg["threads"])
    with np.load("RData.npz") as rdata:
        response = np.asarray(rdata["R"][:, :N_BINS], dtype=np.float64) * 1e26
    x = zarr.open(os.path.join(cfg["data"], "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(cfg["data"], "test_y.zarr"), mode="r")
    thr = np.asarray(cfg["thresholds"], dtype=np.float32)[:, None, None]
    acc = np.zeros((len(SOURCES), len(INTENSITY_LABELS), len(CHANNELS), len(STATS)))
    cache = OrderedDict()
    rng = np.random.default_rng(1000 + wid)
    samples = {g: [] for g in SAMPLE_GROUPS}
    seen = {g: 0 for g in SAMPLE_GROUPS}
    excluded = 0
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
        valid = base & np.isfinite(rec["label_free"]).all(axis=0) & np.isfinite(rec["supervised"]).all(axis=0)
        excluded += int((base & ~valid).sum())
        bright = (obs >= thr).any(axis=0)
        ratio = np.where(np.isfinite(obs), obs / thr, -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            logr = np.log10(ratio.max(axis=0))
        logr = np.where(np.isfinite(logr), logr, -np.inf)
        band = np.searchsorted(INTENSITY_EDGES, logr, side="right")
        band = np.where(bright, np.maximum(band, 4), np.minimum(band, 3))

        idx = np.flatnonzero(valid.ravel())
        if idx.size:
            g = band.ravel()[idx]
            o = obs.reshape(6, -1)[:, idx].astype(np.float64)
            for si, s in enumerate(SOURCES):
                r = rec[s].reshape(6, -1)[:, idx]
                d = r - o
                pos = (o > 0) & (r > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lr = np.where(pos, np.log10(np.where(pos, r, 1) / np.where(pos, o, 1)), 0.0)
                cols = {"n": np.ones_like(o), "obs": o, "rec": r, "abs": np.abs(d), "sq": d * d,
                        "logr": lr, "n_logr": pos.astype(float),
                        "under": (r < 0.9 * o).astype(float), "over": (r > 1.1 * o).astype(float)}
                for c in range(6):
                    for stat, values in cols.items():
                        acc[si, :, c, SI[stat]] += np.bincount(g, weights=values[c],
                                                               minlength=len(INTENSITY_LABELS))

            flat = {n: v.reshape(v.shape[0], -1) for n, v in (("obs", obs), ("lf", lf), ("sup", sup), ("ref", ref))}
            score = ratio.max(axis=0).ravel()
            for name, bands in SAMPLE_GROUPS.items():
                cand = idx[np.isin(band.ravel()[idx], bands)]
                items = [(lambda p=p: {"block": b, "pixel": int(p), "score": float(score[p]),
                                       "obs": flat["obs"][:, p].copy(), "ref": flat["ref"][:, p].copy(),
                                       "label_free": flat["lf"][:, p].copy(),
                                       "supervised": flat["sup"][:, p].copy()}) for p in cand]
                seen[name] = reservoir_add(samples[name], seen[name], rng, cfg["samples"], items)

        done = b - start + 1
        if done % 500 == 0 or b + 1 == stop:
            print(f"[w{wid}] {done:,}/{stop - start:,} blocks  {done / (time.time() - t0):.1f} blocks/s", flush=True)
    return acc, samples, seen, excluded


def summarize(acc):
    out = {}
    for si, s in enumerate(SOURCES):
        rows = {}
        for gi, label in enumerate(INTENSITY_LABELS):
            a = acc[si, gi]
            if a[0, SI["n"]] == 0:
                continue
            rows[label] = {"n_pixels": int(a[0, SI["n"]]), "channels": {
                ch: {"recon_over_obs": a[c, SI["rec"]] / a[c, SI["obs"]] if a[c, SI["obs"]] else None,
                     "geomean_recon_over_obs": float(10 ** (a[c, SI["logr"]] / a[c, SI["n_logr"]]))
                     if a[c, SI["n_logr"]] else None,
                     "mae_over_mean_obs": a[c, SI["abs"]] / a[c, SI["obs"]] if a[c, SI["obs"]] else None,
                     "pct_under_10pct": 100 * a[c, SI["under"]] / a[c, SI["n"]],
                     "pct_over_10pct": 100 * a[c, SI["over"]] / a[c, SI["n"]]}
                for c, ch in enumerate(CHANNELS)}}
        tot = acc[si].sum(axis=0)
        bright = acc[si, 4:].sum(axis=0)
        pooled = lambda t: {"n_pixels": int(t[0, SI["n"]]),
                            "mae": float(t[:, SI["abs"]].sum() / t[:, SI["n"]].sum()),
                            "mse": float(t[:, SI["sq"]].sum() / t[:, SI["n"]].sum())}
        out[s] = {"by_band": rows, "pooled_full": pooled(tot), "pooled_bright": pooled(bright)}
    return out


def render(out):
    L = [f"# AIA reconstruction by brightness band: {out['reference'].upper()} track",
         "Aggregate reconstructed / observed intensity per channel (1.00 = perfect).",
         "Pooled full-population MAE/MSE should reproduce the website's AIA numbers."]
    for s in SOURCES:
        p = out["results"][s]["pooled_full"]
        L.append(f"  {s:<11} pooled MAE {p['mae']:.4f}  MSE {p['mse']:.4f}  (n={p['n_pixels']:,})")
    for s in SOURCES:
        L.append(f"\n## {s}: reconstructed / observed")
        head = f"{'band':<11}{'pixels':>13}" + "".join(f"{c + 'A':>8}" for c in CHANNELS) + "   under-by-10% (94A,131A)"
        L += [head, "-" * len(head)]
        for band, row in out["results"][s]["by_band"].items():
            ch = row["channels"]
            L.append(f"{band:<11}{row['n_pixels']:>13,}" + "".join(
                f"{ch[c]['recon_over_obs']:>8.3f}" for c in CHANNELS)
                + f"   {ch['94']['pct_under_10pct']:5.1f}% {ch['131']['pct_under_10pct']:5.1f}%")
    L.append(f"\nexcluded non-finite prediction pixels: {out['excluded_nonfinite_prediction_pixels']:,}")
    return "\n".join(L)


def main():
    args = parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            print(render(json.load(f)))
        return
    import zarr
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload) if isinstance(payload, dict) else payload, dtype=np.float64)
    available = int(zarr.open(os.path.join(args.data, "test_x.zarr"), mode="r").shape[-1])
    n_blocks = min(available, args.max_blocks) if args.max_blocks else available
    cfg = {"label_free": os.path.abspath(args.label_free), "supervised": os.path.abspath(args.supervised),
           "data": os.path.abspath(args.data), "thresholds": thresholds.tolist(),
           "threads": args.threads_per_worker, "pixel_batch": args.pixel_batch, "samples": args.samples}
    ranges = split_ranges(n_blocks, args.workers)
    jobs = [(cfg, s, e, i) for i, (s, e) in enumerate(ranges)]
    print(f"blocks {n_blocks:,} of {available:,}; ranges {ranges}", flush=True)
    t0 = time.time()
    if len(jobs) == 1:
        results = [run_range(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(run_range, jobs)
    acc = sum(r[0] for r in results)
    excluded = sum(r[3] for r in results)
    out = {"reference": args.reference, "label_free": cfg["label_free"], "supervised": cfg["supervised"],
           "data": cfg["data"], "n_blocks": n_blocks, "available_blocks": available,
           "complete_test": n_blocks == available, "channels_angstrom": [int(c) for c in CHANNELS],
           "bright_thresholds": thresholds.tolist(), "bands": INTENSITY_LABELS,
           "mask": "finite reference DEM, six observations and both model reconstructions (as eval_aia_shared_test.py)",
           "excluded_nonfinite_prediction_pixels": excluded,
           "runtime_s": round(time.time() - t0, 1), "results": summarize(acc)}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    arrays = {}
    for name in SAMPLE_GROUPS:
        pool = [item for r in results for item in r[1][name]]
        seen = sum(r[2][name] for r in results)
        # Each worker kept a uniform sample of its own range; weight workers by
        # how many candidates they saw so the merged sample stays uniform.
        weights = np.concatenate([np.full(len(r[1][name]), r[2][name] / max(len(r[1][name]), 1))
                                  for r in results]) if pool else np.zeros(0)
        if pool:
            rng = np.random.default_rng(7)
            keep = rng.choice(len(pool), size=min(len(pool), args.samples), replace=False,
                              p=weights / weights.sum())
            pool = [pool[i] for i in sorted(keep)]
        arrays[f"{name}_seen"] = np.array(seen)
        for field in ("block", "pixel", "score", "obs", "ref", "label_free", "supervised"):
            arrays[f"{name}_{field}"] = np.array([item[field] for item in pool])
    np.savez_compressed(os.path.splitext(args.output)[0] + "_examples.npz", **arrays)
    text = render(out)
    with open(os.path.splitext(args.output)[0] + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.output} (+ .txt, _examples.npz) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
