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
    return p.parse_args()


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
