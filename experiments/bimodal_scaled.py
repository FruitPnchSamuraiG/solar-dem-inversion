"""
Bimodal DEM census on the held-out test split, and what the scaled networks
predict at those pixels.

This reruns the 2026-07-11 diagnostic at scale and against the new models. That
one scanned 1,000 pixels per timestamp across 4 crop timestamps and found
bimodality in 5.1-9.1% of pixels, mostly unstable under photon noise, with only
6/120 studied pixels stable -- all in flare timestamps. Everything about that
result was small-sample and used the pre-scaling patch-CNN checkpoint, so none
of it carries over automatically.

Three phases:

  1. CENSUS -- count peaks in the *stored* solver DEM over many test blocks. The
     solver already ran on every pixel, so prevalence needs no re-solving and
     the sample is orders of magnitude larger than before. Broken down by
     tolLevel and by brightness, since the earlier finding was that stable
     bimodality concentrates in flare regions.

  2. STABILITY -- at a sample of bimodal pixels, perturb the observation by
     N(0, err) and re-solve BP. If the second peak survives the perturbation it
     is signal; if it flips in and out, BP is tie-breaking between near-equal
     optima at the noise level and there is nothing for the network to learn.

  3. MODELS -- what all four scaled networks predict at those same pixels: peak
     count, and barrier loss against BP's own solution. A network whose unimodal
     answer scores *better* on the objective than BP's bimodal one is not making
     an error; the two are degenerate and BP broke the tie differently.

Run from the project root on Torch:

    python3 experiments/bimodal_scaled.py \
        --bp_root   $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --enet_root $SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS \
        --ckpt_dir  output/experiments
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fullBP import solveLP
from src.losses import barrier_loss_batch
from src.zarr_data import N_AIA_BINS
from src.scaled_eval import (BlockReader, count_peaks, describe_ckpt,
                             load_scaled_model)
from experiments.train_neural_field import pick_device
from experiments.train_scaled import load_operators
from experiments.eval_scaled import RUNS, STYLE, find_ckpt

C_PERTURB = "#b9b9b9"


def count_peaks_batch(dems, prominence_frac=0.15, interior=False):
    """Peak count for [P, T] curves.

    Two stages because the census runs over hundreds of thousands of pixels and
    scipy's find_peaks is per-curve: a vectorised local-maximum count first
    (cheap, ignores prominence), then the real prominence-aware count only on
    curves with more than one raw maximum. Curves with <=1 raw maximum cannot
    gain a peak by adding a prominence threshold, so the shortcut is exact.

    With interior=True, also returns the count restricted to bins 1..T-2. The
    first and last logT bins sit at the ends of AIA's temperature response,
    where six EUV channels have almost no discriminating power, and the padding
    above makes a curve that merely rises into a boundary bin register a peak
    there. The test-split plots show most of BP's second peaks are exactly that
    -- emission dumped at logT 5.5 or 7.2 -- which is a known inversion artifact
    rather than a second thermal component, so the two must be counted apart.
    """
    P, T = dems.shape
    out = np.zeros(P, dtype=np.int16)
    inn = np.zeros(P, dtype=np.int16)
    pos = dems.max(axis=1) > 0
    if not pos.any():
        return (out, inn) if interior else out

    z = np.zeros((P, 1), dtype=dems.dtype)
    padded = np.concatenate([z, dems, z], axis=1)
    ismax = ((padded[:, 1:-1] > padded[:, :-2]) &
             (padded[:, 1:-1] >= padded[:, 2:]))          # [P, T]
    raw = ismax.sum(axis=1)

    single = pos & (raw == 1)
    out[single] = 1
    inn[single] = ismax[single, 1:T - 1].sum(axis=1)
    for k in np.nonzero(pos & (raw >= 2))[0]:
        n, where = count_peaks(dems[k], prominence_frac)
        out[k] = n
        inn[k] = int(((where >= 1) & (where <= T - 2)).sum())
    return (out, inn) if interior else out


def solve_bp(D64, obs, err, tolfacs=(1.4, 2.0, 2.8, 5.0)):
    """BP with the escalating tolerance schedule -- the same one dropping
    --zerochill restored, which is why 99.98% of pixels have a solution."""
    for tolfac in tolfacs:
        x = solveLP((D64, obs, obs - tolfac * err, obs + tolfac * err, None))
        if x is not None:
            return x, tolfac
    return None, None


# ── phase 1 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _network_peaks(reader, obs, err, tol, rows, cols, models, B_t, device,
                   n_bins, prominence, chunk=8192):
    """Peak count for every network at every valid pixel of one block.

    Recall alone cannot answer "can the network predict bimodality": a network
    that emitted two peaks everywhere would match BP at every bimodal pixel and
    be worthless. So the census scores each network over *all* valid pixels,
    which gives false positives and hence precision.
    """
    px = reader.gather(obs, err, tol, None, rows, cols)
    patch = px["patch"]
    out = {}
    for key, model in models.items():
        peaks = np.empty(len(rows), dtype=np.int16)
        inner = np.empty(len(rows), dtype=np.int16)
        for s in range(0, len(rows), chunk):
            x = model(patch[s:s + chunk].to(device))
            pred = (x @ B_t.T)[:, :n_bins].cpu().numpy()
            peaks[s:s + chunk], inner[s:s + chunk] = count_peaks_batch(
                pred, prominence, interior=True)
        out[key] = (peaks, inner)
    return out


def census(reader, block_ids, prominence, n_bins, models=None, B_t=None,
           device=None):
    print(f"\n{'='*78}\nPHASE 1 -- BIMODAL CENSUS on stored BP solutions\n{'='*78}")
    tot = {"n": 0, "bimodal": 0}
    by_tol = {1: [0, 0], 3: [0, 0], 5: [0, 0]}
    bright_all, peaks_all = [], []
    hits = []          # (block, i, j) of bimodal pixels, for phase 2
    # confusion counts per network: [true pos, false pos, false neg, true neg],
    # kept twice -- once against any second peak, once against interior-only
    conf = {k: np.zeros(4, dtype=np.int64) for k in (models or {})}
    conf_in = {k: np.zeros(4, dtype=np.int64) for k in (models or {})}
    interior_all = []

    for b in block_ids:
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        curves = np.ascontiguousarray(dem[:, rows, cols].T)      # [P, n_bins]
        npk, npk_in = count_peaks_batch(curves, prominence, interior=True)
        bright = obs[:, rows, cols].sum(axis=0)
        tl = tol[rows, cols]
        interior_all.append(npk_in)

        if models:
            ref_bi, ref_bi_in = npk >= 2, npk_in >= 2
            nn_peaks = _network_peaks(reader, obs, err, tol, rows, cols,
                                      models, B_t, device, n_bins, prominence)
            for key, (pk, pk_in) in nn_peaks.items():
                for tgt, nn_bi, bp_bi in ((conf, pk >= 2, ref_bi),
                                          (conf_in, pk_in >= 2, ref_bi_in)):
                    tgt[key] += np.array([
                        int((nn_bi & bp_bi).sum()), int((nn_bi & ~bp_bi).sum()),
                        int((~nn_bi & bp_bi).sum()), int((~nn_bi & ~bp_bi).sum())])

        tot["n"] += len(rows)
        tot["bimodal"] += int((npk >= 2).sum())
        for lvl in by_tol:
            sel = tl == lvl
            by_tol[lvl][0] += int(sel.sum())
            by_tol[lvl][1] += int((npk[sel] >= 2).sum())
        bright_all.append(bright)
        peaks_all.append(npk)
        for k in np.nonzero(npk >= 2)[0]:
            hits.append((int(b), int(rows[k]), int(cols[k]), float(bright[k]),
                         int(tl[k]), int(npk[k])))

    bright_all = np.concatenate(bright_all)
    peaks_all = np.concatenate(peaks_all)
    interior_all = np.concatenate(interior_all)
    frac = tot["bimodal"] / max(tot["n"], 1)
    n_in = int((interior_all >= 2).sum())
    frac_in = n_in / max(tot["n"], 1)
    print(f"Scanned {tot['n']:,} valid pixels over {len(block_ids)} test blocks")
    print(f"  bimodal (>=2 peaks): {tot['bimodal']:,}  ({100*frac:.2f}%)")
    print(f"  of which INTERIOR (both peaks off the boundary bins): "
          f"{n_in:,}  ({100*frac_in:.2f}% of all pixels)")
    print(f"  boundary-only second peak: {tot['bimodal']-n_in:,}  "
          f"({100*(tot['bimodal']-n_in)/max(tot['bimodal'],1):.1f}% of the bimodal set)"
          f"\n    -- emission at logT 5.5 or 7.2, the ends of AIA's response, is a"
          f"\n       known inversion artifact rather than a second thermal component")
    print(f"  (2026-07-11 on 4 crop timestamps, 1,000 px each: 5.1-9.1%)")

    print("\nBy tolLevel:")
    for lvl, (n, nb) in by_tol.items():
        if n:
            print(f"  tol={lvl}: {nb:,}/{n:,} = {100*nb/n:.2f}% bimodal")

    print("\nBy brightness decile (sum over 6 channels):")
    edges = np.quantile(bright_all, np.linspace(0, 1, 11))
    for k in range(10):
        sel = (bright_all >= edges[k]) & (bright_all <= edges[k + 1])
        if sel.sum():
            print(f"  d{k+1:>2} [{edges[k]:>10.3g},{edges[k+1]:>10.3g}]: "
                  f"{100*(peaks_all[sel] >= 2).mean():>5.2f}%  (n={sel.sum():,})")

    out = {"n_scanned": int(tot["n"]), "n_bimodal": int(tot["bimodal"]),
           "frac_bimodal": float(frac),
           "by_tol": {str(k): v for k, v in by_tol.items()}}

    for label, table, base, dkey in (
            ("ANY second peak", conf, frac, "predict"),
            ("INTERIOR peaks only", conf_in, frac_in, "predict_interior")):
        if not table:
            continue
        print(f"\nCan the networks PREDICT bimodality -- {label}? "
              f"({tot['n']:,} pixels, BP base rate {100*base:.2f}%)")
        print(f"  {'network':>13}  {'rate':>6}  {'recall':>7}  {'prec':>6}   "
              f"{'TP':>7} {'FP':>7} {'FN':>7}")
        out[dkey] = {}
        for key, (tp, fp, fn, tn) in table.items():
            name = f"{key[0]}_{key[1]}"
            rate = (tp + fp) / max(tp + fp + fn + tn, 1)
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            print(f"  {name:>13}  {100*rate:>5.2f}%  {100*rec:>6.2f}%  "
                  f"{100*prec:>5.2f}%   {tp:>7,} {fp:>7,} {fn:>7,}")
            out[dkey][name] = {"rate": float(rate), "recall": float(rec),
                               "precision": float(prec), "tp": int(tp),
                               "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        print(f"  precision against the {100*base:.2f}% base rate is the test: at "
              f"the base rate\n  the network is guessing; well above it, it has "
              f"learned where these live.")
    out["n_interior_bimodal"] = n_in
    out["frac_interior_bimodal"] = float(frac_in)

    return out, hits


# ── phase 2 ───────────────────────────────────────────────────────────────────

def stability(reader, hits, D64, args, rng):
    print(f"\n{'='*78}\nPHASE 2 -- PERTURBATION STABILITY\n{'='*78}")
    # Bias the study sample toward the bright tail: the 2026-07-11 result was
    # that every stable bimodal pixel sat in a flare region, so a uniform draw
    # would spend the budget where the answer is already known.
    hits_sorted = sorted(hits, key=lambda h: -h[3])
    n_bright = min(len(hits_sorted), args.n_study // 2)
    pick = hits_sorted[:n_bright]
    rest = hits_sorted[n_bright:]
    if rest and len(pick) < args.n_study:
        extra = rng.choice(len(rest), size=min(args.n_study - len(pick), len(rest)),
                           replace=False)
        pick += [rest[int(k)] for k in extra]

    # Group by block: read_block decompresses a whole 256x256 chunk, and `pick`
    # is in brightness order, so iterating it directly would re-read the same
    # blocks repeatedly.
    records = []
    cur_block, cache = None, None
    for (b, i, j, bright, tl, npk) in sorted(pick, key=lambda h: h[0]):
        if b != cur_block:
            cache = reader.read_block(b)
            cur_block = b
        obs_a, err_a, tol_a, dem_a = cache
        obs = obs_a[:, i, j].astype(np.float64)
        err = err_a[:, i, j].astype(np.float64)

        x_bp, tf = solve_bp(D64, obs, err)
        if x_bp is None:
            continue

        n_still, perturbed = 0, []
        for _ in range(args.n_perturb):
            obs_p = obs + rng.normal(0, 1, size=obs.shape) * err
            x_p, _ = solve_bp(D64, obs_p, err)
            if x_p is None:
                continue
            dem_p = np.maximum(args.B @ x_p, 0)
            perturbed.append(dem_p)
            if count_peaks(dem_p, args.prominence)[0] >= 2:
                n_still += 1
        stab = n_still / max(len(perturbed), 1)
        records.append({"block": b, "i": i, "j": j, "bright": bright,
                        "tol": tl, "peaks": npk, "stab": stab,
                        "x_bp": x_bp, "tolfac": tf, "perturbed": perturbed,
                        "dem_bp": np.maximum(args.B @ x_bp, 0)})

    stab_arr = np.array([r["stab"] for r in records])
    if not records:
        # --n_study 0 runs the census and the cross-tab only, which is the cheap
        # path: phase 2 is the one that re-solves the LP.
        print("no pixels studied (--n_study 0); skipping the perturbation phases")
        return records
    print(f"Studied {len(records)} bimodal pixels, {args.n_perturb} noise draws each")
    print(f"  mean fraction still bimodal: {stab_arr.mean():.2f}")
    print(f"  stable (>=0.8): {(stab_arr >= 0.8).sum()}   "
          f"marginal (0.2-0.8): {((stab_arr > 0.2) & (stab_arr < 0.8)).sum()}   "
          f"noise (<=0.2): {(stab_arr <= 0.2).sum()}")
    print(f"  (2026-07-11: mean 0.40-0.46, 6/120 stable)")
    return records


# ── phase 3 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def model_behaviour(reader, records, models, D_t, B_t, device, args):
    print(f"\n{'='*78}\nPHASE 3 -- WHAT THE NETWORKS PREDICT THERE\n{'='*78}")
    cur_block, cache = None, None
    for r in records:
        if r["block"] != cur_block:
            cache = reader.read_block(r["block"])
            cur_block = r["block"]
        obs_a, err_a, tol_a, dem_a = cache
        px = reader.gather(obs_a, err_a, tol_a, dem_a, [r["i"]], [r["j"]])
        patch = px["patch"].to(device)
        obs_b, lb_b, ub_b = (px["obs"].to(device), px["lb"].to(device),
                             px["ub"].to(device))

        x_bp_t = torch.tensor(r["x_bp"], dtype=torch.float32,
                              device=device).unsqueeze(0)
        r["loss_bp"] = barrier_loss_batch(x_bp_t, D_t, obs_b, lb_b, ub_b).item()
        r["nn"] = {}
        for key in RUNS:
            x = models[key](patch)
            dem = (x @ B_t.T)[:, :N_AIA_BINS].cpu().numpy()[0]
            r["nn"][f"{key[0]}_{key[1]}"] = {
                "dem": dem,
                "peaks": int(count_peaks(dem, args.prominence)[0]),
                "loss": barrier_loss_batch(x, D_t, obs_b, lb_b, ub_b).item(),
            }

    for key in RUNS:
        name = f"{key[0]}_{key[1]}"
        pk = np.array([r["nn"][name]["peaks"] for r in records])
        d = np.array([r["nn"][name]["loss"] - r["loss_bp"] for r in records])
        print(f"{name:>13}: bimodal at {int((pk >= 2).sum())}/{len(pk)} of BP's "
              f"bimodal pixels   mean(loss_NN - loss_BP) = {d.mean():+.3f}   "
              f"better in {int((d < 0).sum())}/{len(d)}")
    print("  (barrier loss is the BP-side objective, so it is the fair "
          "comparison for all four; ~0 or negative => degenerate tie-breaking, "
          "not a network error)")

    stable = [r for r in records if r["stab"] >= 0.8]
    print(f"\nRestricted to the {len(stable)} pixels stable under noise:")
    for key in RUNS:
        name = f"{key[0]}_{key[1]}"
        if not stable:
            break
        pk = np.array([r["nn"][name]["peaks"] for r in stable])
        d = np.array([r["nn"][name]["loss"] - r["loss_bp"] for r in stable])
        print(f"{name:>13}: bimodal at {int((pk >= 2).sum())}/{len(pk)}   "
              f"mean(loss_NN - loss_BP) = {d.mean():+.3f}   "
              f"better in {int((d < 0).sum())}/{len(d)}")


def plot_records(records, logT, out_dir, args):
    """Most-stable pixels first -- those are the ones that would matter."""
    recs = sorted(records, key=lambda r: -r["stab"])[:args.n_plot]
    if not recs:
        return []
    fig, axes = plt.subplots(len(recs), 1, figsize=(9.5, 3.2 * len(recs)),
                             squeeze=False)
    for ax, r in zip(axes[:, 0], recs):
        for dem_p in r["perturbed"]:
            ax.plot(logT, dem_p, color=C_PERTURB, lw=0.6, alpha=0.5, zorder=1)
        ax.plot(logT, r["dem_bp"], label=f"BP (stability {r['stab']:.2f})",
                **STYLE["ref"])
        for key in RUNS:
            name = f"{key[0]}_{key[1]}"
            if "nn" not in r:
                continue
            st = dict(STYLE[key[0]])
            if key[1] == "enet":
                st["alpha"] = 0.55
            ax.plot(logT, r["nn"][name]["dem"], label=name, **st)
        ax.set_title(f"block {r['block']} px ({r['i']},{r['j']})  "
                     f"tol={r['tol']}  sum(obs)={r['bright']:.3g}  "
                     f"gray: {len(r['perturbed'])} noise-perturbed BP solves",
                     fontsize=7.5)
        ax.set_xlabel("log T", fontsize=7)
        ax.set_ylabel("DEM", fontsize=7)
        ax.tick_params(labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.18, lw=0.5)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle("Bimodal BP pixels on held-out test data", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    path = os.path.join(out_dir, "bimodal_scaled.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"\n  wrote {path}")
    return [path]


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    D64 = D_t.cpu().numpy().astype(np.float64)
    args.B = B_t.cpu().numpy()
    print(f"D: {tuple(D_t.shape)}   basis: {n_basis}   temps: {len(logT)}")

    models = {}
    for key in RUNS:
        path = find_ckpt(args.ckpt_dir, *key, suffix=args.ckpt_suffix)
        m, c = load_scaled_model(path, n_basis, device)
        models[key] = m
        print(f"  loaded {describe_ckpt(c)}")

    os.makedirs(args.out_dir, exist_ok=True)
    reader = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(reader), size=min(args.n_blocks, len(reader)),
                           replace=False)

    stats, hits = census(reader, block_ids, args.prominence, N_AIA_BINS,
                         models=models, B_t=B_t, device=device)
    if not hits:
        print("no bimodal pixels found; nothing to study")
        return

    records = stability(reader, hits, D64, args, rng)
    figs = []
    if records:
        model_behaviour(reader, records, models, D_t, B_t, device, args)
        figs = plot_records(records, logT, args.out_dir, args)

    stats["stability"] = [
        {k: v for k, v in r.items()
         if k not in ("x_bp", "perturbed", "dem_bp", "nn")}
        | {"nn": {n: {"peaks": d["peaks"], "loss": d["loss"]}
                  for n, d in r.get("nn", {}).items()},
           "loss_bp": r.get("loss_bp")}
        for r in records]
    stats["figures"] = figs
    path = os.path.join(args.out_dir, "bimodal_scaled_summary.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--enet_root", required=False)
    p.add_argument("--ckpt_dir", default="output/experiments")
    p.add_argument("--out_dir", default="output/experiments/bimodal_scaled")
    p.add_argument("--ckpt_suffix", default="",
                   help="select a sweep width, e.g. _h160")
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--n_blocks", type=int, default=20,
                   help="test blocks for the census (~16k pixels each)")
    p.add_argument("--n_study", type=int, default=60,
                   help="bimodal pixels taken forward to the perturbation study")
    p.add_argument("--n_perturb", type=int, default=30)
    p.add_argument("--n_plot", type=int, default=8)
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
