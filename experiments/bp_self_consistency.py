"""
Is the bimodal deficit ours, or is it label noise?

The 2026-08-03 sweep evaluation found the network's mae_dem is ~3x worse where
BP is bimodal (0.29 vs 0.10), and -- the informative part -- that this penalty is
essentially CONSTANT from 1.43M parameters down to 20k (2.89x-3.11x). Capacity is
not the bottleneck. That leaves two candidates: the amortization itself, or the
target.

BP solves each pixel independently from a noisy observation. If re-solving the
same pixel under one realistic photon-noise draw moves BP's own answer as much as
our prediction differs from it, then "match BP exactly" is not a well-posed
target and mae_dem is partly measuring solver irreproducibility rather than model
error. The perturbation spaghetti in the final figures suggests exactly this at
bimodal pixels -- the gray re-solves span 0 to 60 where BP's clean answer sits
at 15.

Three quantities per pixel, all in DEM units so they are directly comparable:

  * BP self-scatter   mean_k |BP(obs + noise_k) - BP(obs)|
        how far BP moves under one noise draw. The reproducibility floor: no
        emulator can be expected to beat it.
  * NN deviation      |NN(obs) - BP(obs)|
        what mae_dem measures.
  * NN vs ensemble    |NN(obs) - mean_k BP(obs + noise_k)|
        the network is trained on the objective, not the label, so it has no
        reason to reproduce one particular solve. If it sits closer to the mean
        of BP's noise ensemble than to any single solve, it is estimating the
        expectation -- arguably the better answer, and invisible to mae_dem.

Reported separately for unimodal and bimodal pixels, and restricted to the hot
bins where the deficit concentrates.

    python3 experiments/bp_self_consistency.py \
        --bp_root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --sweep_dir output/experiments/sweep --base_dir output/experiments
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch

from src.zarr_data import N_AIA_BINS
from src.scaled_eval import BlockReader, load_scaled_model
from experiments.train_neural_field import pick_device
from experiments.train_scaled import load_operators
from experiments.eval_scaled import find_ckpt
from experiments.bimodal_scaled import count_peaks_batch, solve_bp


def sample_pixels(reader, block_ids, args, rng):
    """Equal numbers of unimodal and bimodal pixels from the bright tail.

    Bright because that is where bimodality lives (30.3% prevalence in the top
    decile vs 7.4% in the faintest) and where the DEM is large enough that a
    scatter measurement means something.
    """
    pools = {"uni": [], "bi": []}
    for b in block_ids:
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        curves = np.ascontiguousarray(dem[:, rows, cols].T)
        _, npk = count_peaks_batch(curves, args.prominence, interior=True)
        bright = obs[:, rows, cols].sum(axis=0)
        thr = np.quantile(bright, args.bright_q)
        for g, sel in (("uni", npk <= 1), ("bi", npk == 2)):
            for k in np.nonzero(sel & (bright >= thr))[0]:
                pools[g].append((int(b), int(rows[k]), int(cols[k]),
                                 float(bright[k])))

    out = {}
    for g, pool in pools.items():
        if not pool:
            out[g] = []
            continue
        idx = rng.choice(len(pool), size=min(args.n_study, len(pool)),
                         replace=False)
        # block order: read_block decompresses a whole chunk, so grouping avoids
        # re-reading the same block once per pixel
        out[g] = sorted([pool[int(i)] for i in idx], key=lambda h: h[0])
    return out


@torch.no_grad()
def measure(reader, picks, models, B_t, D64, B_np, hot, args, device, rng):
    rows = []
    cur_block, cache = None, None
    for group, hits in picks.items():
        for (b, i, j, bright) in hits:
            if b != cur_block:
                cache = reader.read_block(b)
                cur_block = b
            obs_a, err_a, tol_a, dem_a = cache
            obs = obs_a[:, i, j].astype(np.float64)
            err = err_a[:, i, j].astype(np.float64)

            x_bp, _ = solve_bp(D64, obs, err)
            if x_bp is None:
                continue
            dem_bp = np.maximum(B_np @ x_bp, 0)[:N_AIA_BINS]

            solves = []
            for _ in range(args.n_perturb):
                x_p, _ = solve_bp(D64, obs + rng.normal(0, 1, size=obs.shape) * err,
                                  err)
                if x_p is not None:
                    solves.append(np.maximum(B_np @ x_p, 0)[:N_AIA_BINS])
            if len(solves) < args.min_solves:
                continue
            solves = np.array(solves)
            ens = solves.mean(axis=0)

            px = reader.gather(obs_a, err_a, tol_a, dem_a, [i], [j])
            patch = px["patch"].to(device)
            preds = {k: ((m(patch) @ B_t.T)[:, :N_AIA_BINS]).cpu().numpy()[0]
                     for k, m in models.items()}

            rec = {"group": group, "block": b, "i": i, "j": j, "bright": bright,
                   "n_solves": len(solves),
                   # BP against itself: the reproducibility floor
                   "bp_scatter": float(np.abs(solves - dem_bp).mean()),
                   "bp_scatter_hot": float(np.abs(solves - dem_bp)[:, hot].mean()),
                   # how much of BP's own answer is noise-driven, as a fraction
                   "bp_scale": float(np.abs(dem_bp).mean()),
                   }
            for k, p in preds.items():
                name = f"{k[0]}_{k[1]}"
                rec[f"nn_{name}"] = float(np.abs(p - dem_bp).mean())
                rec[f"nn_{name}_hot"] = float(np.abs(p - dem_bp)[hot].mean())
                rec[f"ens_{name}"] = float(np.abs(p - ens).mean())
                rec[f"ens_{name}_hot"] = float(np.abs(p - ens)[hot].mean())
            rows.append(rec)
    return rows


def report(rows, model_names, args):
    print(f"\n{'='*98}")
    print("BP AGAINST ITSELF -- is the bimodal deficit ours or the label's?")
    print(f"{'='*98}")
    if not rows:
        print("no pixels measured")
        return {}

    out = {}
    for group, label in (("uni", "UNIMODAL"), ("bi", "BIMODAL")):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        scat = np.mean([r["bp_scatter"] for r in sub])
        scat_h = np.mean([r["bp_scatter_hot"] for r in sub])
        scale = np.mean([r["bp_scale"] for r in sub])
        print(f"\n{label}  (n={len(sub)}, mean |BP| = {scale:.4f})")
        print(f"  BP self-scatter under one noise draw : {scat:.4f}"
              f"   (hot bins {scat_h:.4f})")
        print(f"  -> {100*scat/max(scale,1e-12):.1f}% of BP's own signal is "
              f"noise-driven")
        print(f"\n  {'model':>16} {'|NN-BP|':>9} {'ratio':>7} "
              f"{'|NN-ens|':>9} {'ratio':>7}   {'hot |NN-BP|':>12} {'ratio':>7}")
        g = {"bp_scatter": float(scat), "bp_scatter_hot": float(scat_h),
             "bp_scale": float(scale), "n": len(sub), "models": {}}
        for name in model_names:
            dev = np.mean([r[f"nn_{name}"] for r in sub])
            devh = np.mean([r[f"nn_{name}_hot"] for r in sub])
            ens = np.mean([r[f"ens_{name}"] for r in sub])
            print(f"  {name:>16} {dev:>9.4f} {dev/max(scat,1e-12):>6.2f}x "
                  f"{ens:>9.4f} {ens/max(scat,1e-12):>6.2f}x   "
                  f"{devh:>12.4f} {devh/max(scat_h,1e-12):>6.2f}x")
            g["models"][name] = {"nn_dev": float(dev), "nn_dev_hot": float(devh),
                                 "ens_dev": float(ens),
                                 "ratio": float(dev / max(scat, 1e-12))}
        out[group] = g

    print("\n  ratio < 1 means the network is closer to BP than BP is to itself "
          "under noise --\n  i.e. inside the solver's own reproducibility "
          "envelope, and mae_dem at those\n  pixels is measuring label noise "
          "rather than model error. ratio >> 1 means the\n  gap is genuinely "
          "ours. |NN-ens| tests whether the network is estimating the\n  "
          "expectation over noise rather than any single solve.")
    return out


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    D64 = D_t.cpu().numpy().astype(np.float64)
    B_np = B_t.cpu().numpy()
    hot = np.asarray(logT[:N_AIA_BINS]) >= args.hot_logT

    models = {}
    for loss in ("barrier", "enet"):
        models[(loss, "base")], _ = load_scaled_model(
            find_ckpt(args.base_dir, "mlp6", loss), n_basis, device)
        models[(loss, "small")], _ = load_scaled_model(
            find_ckpt(args.sweep_dir, "mlp6", loss, suffix=f"_h{args.width}"),
            n_basis, device)
    names = [f"{k[0]}_{k[1]}" for k in models]
    print(f"Loaded {len(models)} models (baseline 1.43M and h{args.width})")

    reader = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(reader), size=min(args.n_blocks, len(reader)),
                           replace=False)
    picks = sample_pixels(reader, block_ids, args, rng)
    print(f"Sampled {len(picks.get('uni', []))} unimodal and "
          f"{len(picks.get('bi', []))} bimodal pixels; "
          f"{args.n_perturb} BP re-solves each")

    rows = measure(reader, picks, models, B_t, D64, B_np, hot, args, device, rng)
    summary = report(rows, names, args)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "bp_self_consistency.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "pixels": rows}, f, indent=2)
    print(f"\nSummary -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--sweep_dir", default="output/experiments/sweep")
    p.add_argument("--base_dir", default="output/experiments")
    p.add_argument("--out_dir", default="output/experiments/final")
    p.add_argument("--width", type=int, default=232)
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--n_blocks", type=int, default=20)
    p.add_argument("--n_study", type=int, default=80,
                   help="pixels per group (unimodal / bimodal)")
    p.add_argument("--n_perturb", type=int, default=30)
    p.add_argument("--min_solves", type=int, default=10)
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--bright_q", type=float, default=0.90,
                   help="sample from above this brightness quantile")
    p.add_argument("--hot_logT", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
