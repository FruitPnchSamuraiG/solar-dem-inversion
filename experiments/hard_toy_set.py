#!/usr/bin/env python3
"""Hard pixel sets for the network toy (experiments/constrained_toy_net.py --data <stem>).

The toy sets from experiments/constrained_toy.py are mostly ordinary pixels (a uniform draw holds
~200 flare cores in 2M). This draws a sample balanced over the bright bands from the TRAINING
split, about --n-core flare cores (32x the Bright threshold or above), --n-mid at 10-32x and
--n-low at 3.2-10x, with each pixel's linear program solved (optimum, DEM, dual prices, smallest
feasible band) exactly as the toy does, and keeps only those BP solved at the tight band. The
held-out sets are the existing toy validation sets cut to the same bands:

    <stem>_train.npz     the hard training sample (new)
    <stem>_hold.npz      <toy>_hold.npz, bands 3.2x and up
    <stem>_balanced.npz  <toy>_balanced.npz, bands 3.2x and up (every flare core it holds)

With --mix <out2> (after the hard set exists) it also writes two mixed sets, held-out sets being the
toy's full ones (every band):
    <out2>_train.npz     the hard set + --n-easy ordinary pixels (below 3.2x) BP solved at the tight
                         band, from the toy's uniform training sample (David: 10k hard and 10k easy)
    <out2>all_train.npz  the same + --n-relaxed pixels BP had to widen (3x / 5x) + --n-unsolved pixels
                         BP could not solve, drawn from the training split: for losses that use no BP
                         flag (train with --pixels all)

    python3 experiments/hard_toy_set.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --toy output/experiments/constrained/toy --out output/experiments/constrained/toy_hard
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from experiments.constrained_toy import CORE, N_GROUPS, gather, init_worker, run_lp  # noqa: E402

LOW, MID = CORE - 2, CORE - 1          # bands 3.2-10x and 10-32x


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--toy", required=True, help="stem of the existing toy sets (_hold/_balanced.npz)")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--blocks", type=int, default=20000, help="training blocks scanned")
    p.add_argument("--n-core", type=int, default=3000)
    p.add_argument("--n-mid", type=int, default=3000)
    p.add_argument("--n-low", type=int, default=4000)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--mix", default=None, help="write the mixed sets under this stem (the hard set must exist)")
    p.add_argument("--n-easy", type=int, default=10000)
    p.add_argument("--n-relaxed", type=int, default=2000)
    p.add_argument("--n-unsolved", type=int, default=2000)
    return p.parse_args()


def collect_flags(job):
    """Pixels BP widened (with probability p_rel) and every pixel it could not solve, from some blocks."""
    cfg, blocks, wid, p_rel = job
    import zarr
    from experiments.constrained_toy import INTENSITY_EDGES, KEYS, N_BINS
    z = {k: zarr.open(os.path.join(cfg["root"], f"train_{k}.zarr"), mode="r") for k in ("x", "e", "y", "m")}
    thr = np.asarray(cfg["thresholds"], np.float32)[:, None, None]
    rng = np.random.default_rng((cfg["seed"], wid, 7))
    rows = {k: [] for k in KEYS}
    for b in blocks:
        tol = np.asarray(z["m"][:, :, b])
        sel = (tol == 0) | (np.isin(tol, (3, 5)) & (rng.random(tol.shape) < p_rel))
        if not sel.any():
            continue
        obs = np.asarray(z["x"][:, :, :, b], np.float32)[:, ::2, ::2]
        err = np.asarray(z["e"][:, :, :, b], np.float32)[:, ::2, ::2]
        sel &= np.isfinite(obs).all(0) & np.isfinite(err).all(0) & (err > 0).all(0)
        if not sel.any():
            continue
        dem = np.asarray(z["y"][:N_BINS, :, :, b], np.float32)
        ratio = (obs / thr).max(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            group = np.clip(np.searchsorted(INTENSITY_EDGES, np.log10(ratio), side="right"), 0, len(INTENSITY_EDGES))
        for k, v in (("obs", obs[:, sel].T), ("err", err[:, sel].T), ("dem", dem[:, sel].T), ("tol", tol[sel]),
                     ("group", group[sel]), ("inten", ratio[sel])):
            rows[k].append(v)
    return {k: np.concatenate(v) for k, v in rows.items()} if rows["obs"] else None


def make_mix(args, cfg, rng):
    import zarr
    common = ("obs", "err", "dem", "tol", "group", "inten", "lp_status", "lp_obj", "lp_tmin", "lp_dem_lp", "lp_y_lo", "lp_y_hi")
    hard = np.load(f"{args.out}_train.npz")
    toy = np.load(f"{args.toy}_train.npz")
    easy_idx = np.flatnonzero((toy["group"] < LOW) & (toy["tol"] == 1))
    easy_idx = np.sort(rng.choice(easy_idx, size=min(args.n_easy, easy_idx.size), replace=False))
    mix = {k: np.concatenate([hard[k], toy[k][easy_idx]]) for k in common}
    np.savez_compressed(f"{args.mix}_train.npz", **mix)
    nb = zarr.open(os.path.join(args.root, "train_x.zarr"), mode="r").shape[-1]
    blocks = np.sort(rng.choice(nb, size=min(args.blocks, nb), replace=False))
    p_rel = min(1.0, 3 * args.n_relaxed / (0.085 * 128 * 128 * len(blocks)))
    chunks = [c.tolist() for c in np.array_split(blocks, args.workers) if len(c)]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = [p for p in pool.map(collect_flags, [(cfg, c, i, p_rel) for i, c in enumerate(chunks)]) if p is not None]
    fl = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    take = []
    for sel, n in ((fl["tol"] == 0, args.n_unsolved), (np.isin(fl["tol"], (3, 5)), args.n_relaxed)):
        idx = np.flatnonzero(sel)
        take.append(rng.choice(idx, size=min(n, idx.size), replace=False) if idx.size else idx)
    fl = {k: v[np.sort(np.concatenate(take))] for k, v in fl.items()}
    init_worker(cfg)
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        lp = run_lp(pool, fl, args.workers)
    fl |= {f"lp_{k}": v for k, v in lp.items()}
    allset = {k: np.concatenate([mix[k], fl[k]]) for k in common}
    np.savez_compressed(f"{args.mix}all_train.npz", **allset)
    for suffix in ("", "all"):
        for part in ("hold", "balanced"):
            src, dst = f"{args.toy}_{part}.npz", f"{args.mix}{suffix}_{part}.npz"
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    summ = {"mix_train": {"hard": int(len(hard["obs"])), "easy": int(easy_idx.size)},
            "all_extra": {"unsolved": int((fl["tol"] == 0).sum()), "relaxed": int(np.isin(fl["tol"], (3, 5)).sum()),
                          "lp_feasible_among_unsolved": int(((fl["tol"] == 0) & (fl["lp_status"] == 0)).sum()),
                          "lp_feasible_among_relaxed": int((np.isin(fl["tol"], (3, 5)) & (fl["lp_status"] == 0)).sum()),
                          "unsolved_least_violation_p50": float(np.nanmedian(fl["lp_tmin"][fl["tol"] == 0])),
                          "relaxed_least_violation_p50": float(np.nanmedian(fl["lp_tmin"][fl["tol"] > 1]))}}
    json.dump(summ, open(f"{args.mix}_sets.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


def cut_bright(path, out):
    z = np.load(path)
    n = len(z["group"])
    keep = z["group"] >= LOW
    arrays = {k: (z[k][keep] if z[k].ndim and z[k].shape[0] == n else z[k]) for k in z.files}
    np.savez_compressed(out, **arrays)
    return {int(g): int(((z["group"] == g) & keep & (z["lp_status"] == 0)).sum()) for g in (LOW, MID, CORE)}


def main():
    args = parse_args()
    import zarr
    t0 = time.time()
    payload = json.load(open(args.thresholds))
    thresholds = np.asarray(payload.get("test", payload), np.float64)
    cfg = {"root": os.path.abspath(args.root), "thresholds": thresholds.tolist(), "seed": args.seed, "tolfac": args.tolfac}
    rng = np.random.default_rng(args.seed)
    if args.mix:
        return make_mix(args, cfg, rng)
    nb = zarr.open(os.path.join(args.root, "train_x.zarr"), mode="r").shape[-1]
    blocks = np.sort(rng.choice(nb, size=min(args.blocks, nb), replace=False))
    # draw probabilities aimed at ~2x the targets (the bands' shares of all pixels), capped below
    px = 128 * 128 * 0.9 * len(blocks)
    keep = np.zeros(N_GROUPS)
    keep[LOW] = min(1.0, 2 * args.n_low / (0.0116 * px))
    keep[MID] = min(1.0, 2 * args.n_mid / (0.00077 * px))
    keep[CORE] = 1.0
    data, _ = gather(cfg, "train", blocks, 0.0, keep.tolist(), args.workers)
    bal = data["bal"]
    take = []
    for g, n in ((LOW, args.n_low), (MID, args.n_mid), (CORE, args.n_core)):
        idx = np.flatnonzero(bal["group"] == g)
        take.append(rng.choice(idx, size=min(n, idx.size), replace=False) if idx.size else idx)
    take = np.sort(np.concatenate(take))
    hard = {k: v[take] for k, v in bal.items()}
    print(f"drawn {len(take):,} from {len(blocks):,} training blocks: "
          f"{np.bincount(hard['group'], minlength=N_GROUPS)[LOW:].tolist()} (3.2-10x / 10-32x / >=32x)", flush=True)
    init_worker(cfg)
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        lp = run_lp(pool, hard, args.workers, keep_x=True)
    tight = hard["tol"] == 1
    hard = {k: v[tight] for k, v in hard.items()}
    lp = {k: v[tight] for k, v in lp.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(f"{args.out}_train.npz", **hard, **{f"lp_{k}": v for k, v in lp.items()})
    feas = lp["status"] == 0
    summary = {"train_blocks": int(len(blocks)), "train_tight_band": int(tight.sum()),
               "train_lp_feasible_per_band": {int(g): int(((hard["group"] == g) & feas).sum()) for g in (LOW, MID, CORE)},
               "hold_lp_feasible_per_band": cut_bright(f"{args.toy}_hold.npz", f"{args.out}_hold.npz"),
               "balanced_lp_feasible_per_band": cut_bright(f"{args.toy}_balanced.npz", f"{args.out}_balanced.npz"),
               "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(f"{args.out}_sets.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
