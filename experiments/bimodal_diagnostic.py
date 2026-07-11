"""
Bimodal DEM diagnostic: is BP's second peak real signal or noise degeneracy?

Per the 2026-07-10 meeting: BP produces bimodal DEMs *because* two narrow
spikes are sparse (fewer active bins than one broad peak), while the NN — a
deterministic point-estimate regressor — averages across near-identical inputs
with different BP optima and collapses to a single blended peak. Before
building a distribution-output head to fix this, this script answers two
questions:

  1. STABILITY: at pixels where BP is bimodal, rerun BP on noise-perturbed
     inputs (obs + N(0, err)). If the second peak flips in and out across
     perturbations, bimodality is noise-level degeneracy and not worth chasing.
     If it persists, it is real signal.

  2. DEGENERACY: compare the barrier-loss value of BP's bimodal solution vs
     the NN's unimodal prediction at the same pixel. If the values are
     near-equal, both are valid optima of the underdetermined problem and the
     NN merely tie-breaks differently — a fundamentally different situation
     from the NN being *worse* on the objective.

Peak counting is done on the DEM curve (B @ x over logT), zero-padded at both
ends so boundary peaks count, with a relative prominence threshold.

Run from project root (crunchy1 — needs the training timestamps under ./data):
    uv run python experiments/bimodal_diagnostic.py \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128 \
        --ckpt output/experiments/ablation_cnn_barrier.pt
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fullBP import getBasis, solveLP
from src.losses import barrier_loss_batch
from experiments.train_neural_field import AIAPatchDataset, pick_device
from experiments.train_ablations import load_ablation_model


def count_peaks(dem, prominence_frac=0.15):
    """Number of peaks in a DEM curve, with prominence relative to the curve
    max. Zero-padded at both ends so a maximum at the first/last logT bin
    counts as a peak."""
    if dem.max() <= 0:
        return 0, np.array([], dtype=int)
    padded = np.concatenate([[0.0], dem, [0.0]])
    peaks, _ = find_peaks(padded, prominence=prominence_frac * dem.max())
    return len(peaks), peaks - 1  # shift back to unpadded indices


def solve_bp(D64, obs, err, tolfacs=(1.4, 2.0, 2.8, 5.0)):
    """BP with the standard escalating tolerance schedule; returns x or None."""
    for tolfac in tolfacs:
        x = solveLP((D64, obs, obs - tolfac * err, obs + tolfac * err, None))
        if x is not None:
            return x
    return None


def main(args):
    device = pick_device()
    print(f"Device: {device}")

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)
    D64 = D.astype(np.float64)
    D_t = torch.tensor(D).to(device)

    model = None
    if args.ckpt:
        print(f"Loading NN checkpoint: {args.ckpt}")
        model, _ = load_ablation_model(args.ckpt, D.shape[1], device)

    os.makedirs("output/experiments", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = {}

    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        print(f"\n{'='*72}\n{tag}\n{'='*72}")
        ds = AIAPatchDataset(data_dir, args.crop, patch_size=args.patch_size,
                             tolfac=args.tolfac)

        # ── phase 1: scan a pixel sample for bimodal BP solutions ────────────
        scan_idx = rng.choice(len(ds), size=min(args.n_scan, len(ds)), replace=False)
        bimodal = []  # (dataset idx, x_bp, n_peaks)
        n_solved = 0
        for i in scan_idx:
            row, col = ds.coords[int(i)]
            p = ds.pad
            obs = ds.obs_pad[:, row + p, col + p].numpy().astype(np.float64)
            err = ds.err_pad[:, row + p, col + p].numpy().astype(np.float64)
            x_bp = solve_bp(D64, obs, err)
            if x_bp is None:
                continue
            n_solved += 1
            n_pk, _ = count_peaks(np.maximum(B @ x_bp, 0), args.prominence)
            if n_pk >= 2:
                bimodal.append((int(i), x_bp, n_pk))

        frac = len(bimodal) / max(n_solved, 1)
        print(f"Scanned {n_solved} solved pixels: {len(bimodal)} bimodal "
              f"({100*frac:.1f}%)")
        if not bimodal:
            summary[tag] = {"n_scanned": n_solved, "n_bimodal": 0}
            continue

        # ── phase 2: perturbation stability at bimodal pixels ────────────────
        study = bimodal[:args.n_study]
        stab_fracs, records = [], []
        for idx, x_bp, n_pk in study:
            row, col = ds.coords[idx]
            p = ds.pad
            obs = ds.obs_pad[:, row + p, col + p].numpy().astype(np.float64)
            err = ds.err_pad[:, row + p, col + p].numpy().astype(np.float64)

            n_still_bimodal, perturbed_dems = 0, []
            for _ in range(args.n_perturb):
                obs_p = obs + rng.normal(0, 1, size=obs.shape) * err
                x_p = solve_bp(D64, obs_p, err)
                if x_p is None:
                    continue
                dem_p = np.maximum(B @ x_p, 0)
                perturbed_dems.append(dem_p)
                n_pk_p, _ = count_peaks(dem_p, args.prominence)
                if n_pk_p >= 2:
                    n_still_bimodal += 1

            stab = n_still_bimodal / max(len(perturbed_dems), 1)
            stab_fracs.append(stab)
            records.append((idx, x_bp, stab, perturbed_dems))

        stab_arr = np.array(stab_fracs)
        print(f"Perturbation stability over {len(study)} bimodal pixels "
              f"({args.n_perturb} noise draws each):")
        print(f"  mean fraction still bimodal: {stab_arr.mean():.2f}")
        print(f"  stable (>=0.8): {(stab_arr >= 0.8).sum()}   "
              f"marginal (0.2-0.8): {((stab_arr > 0.2) & (stab_arr < 0.8)).sum()}   "
              f"noise (<=0.2): {(stab_arr <= 0.2).sum()}")

        # ── phase 3: NN-vs-BP barrier-loss degeneracy check ──────────────────
        loss_rows = []
        if model is not None:
            for idx, x_bp, stab, _ in records:
                patch, obs_t, lb_t, ub_t = ds[idx]
                with torch.no_grad():
                    x_nn = model(patch.unsqueeze(0).to(device))
                    obs_b = obs_t.unsqueeze(0).to(device)
                    lb_b = lb_t.unsqueeze(0).to(device)
                    ub_b = ub_t.unsqueeze(0).to(device)
                    loss_nn = barrier_loss_batch(x_nn, D_t, obs_b, lb_b, ub_b).item()
                    x_bp_t = torch.tensor(x_bp, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                    loss_bp = barrier_loss_batch(x_bp_t, D_t, obs_b, lb_b, ub_b).item()
                loss_rows.append((idx, stab, loss_bp, loss_nn))

            print(f"\nBarrier loss at BP's bimodal solution vs NN's prediction:")
            print(f"{'pixel':>8} {'stability':>10} {'loss(BP)':>10} {'loss(NN)':>10} "
                  f"{'NN-BP':>8}")
            for idx, stab, l_bp, l_nn in loss_rows:
                print(f"{idx:>8} {stab:>10.2f} {l_bp:>10.3f} {l_nn:>10.3f} "
                      f"{l_nn - l_bp:>+8.3f}")
            diffs = np.array([l_nn - l_bp for _, _, l_bp, l_nn in loss_rows])
            print(f"  mean(loss_NN - loss_BP) = {diffs.mean():+.3f}  "
                  f"(≈0 ⇒ degenerate tie-breaking; >0 ⇒ NN genuinely worse)")

        # ── plot: BP original + perturbed spaghetti + NN, per bimodal pixel ──
        n_plot = min(args.n_plot, len(records))
        fig, axes = plt.subplots(n_plot, 1, figsize=(9, 3.5 * n_plot))
        if n_plot == 1:
            axes = [axes]
        for ax, (idx, x_bp, stab, perturbed_dems) in zip(axes, records[:n_plot]):
            row, col = ds.coords[idx]
            for dem_p in perturbed_dems:
                ax.plot(logT, dem_p, color="gray", lw=0.6, alpha=0.35)
            ax.plot(logT, np.maximum(B @ x_bp, 0), color="black", lw=2.5,
                    label=f"BP (stab={stab:.2f})", zorder=10)
            if model is not None:
                patch, _, _, _ = ds[idx]
                with torch.no_grad():
                    x_nn = model(patch.unsqueeze(0).to(device))
                ax.plot(logT, np.maximum(B @ x_nn.cpu().numpy()[0], 0),
                        color="tab:blue", lw=1.8, linestyle="--", label="NN", zorder=9)
            ax.set_title(f"Pixel ({row},{col}) — gray: {len(perturbed_dems)} "
                         f"noise-perturbed BP solves", fontsize=8)
            ax.set_xlabel("log T", fontsize=7)
            ax.set_ylabel("DEM", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
        plt.suptitle(f"Bimodal BP pixels under noise perturbation — {tag}", fontsize=11)
        plt.tight_layout()
        out = f"output/experiments/bimodal_diagnostic_{tag}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nSaved plot: {out}")

        summary[tag] = {
            "n_scanned":        n_solved,
            "n_bimodal":        len(bimodal),
            "bimodal_frac":     frac,
            "n_studied":        len(study),
            "mean_stability":   float(stab_arr.mean()),
            "n_stable":         int((stab_arr >= 0.8).sum()),
            "n_marginal":       int(((stab_arr > 0.2) & (stab_arr < 0.8)).sum()),
            "n_noise":          int((stab_arr <= 0.2).sum()),
            "loss_rows": [{"pixel": i, "stability": s, "loss_bp": lb, "loss_nn": ln}
                          for i, s, lb, ln in loss_rows],
        }

    out_json = "output/experiments/bimodal_diagnostic_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_json}")

    print(f"\n{'='*72}\nOVERALL")
    for tag, s in summary.items():
        if s.get("n_bimodal", 0) == 0:
            print(f"{tag:<18} no bimodal pixels in scan of {s['n_scanned']}")
        else:
            print(f"{tag:<18} bimodal {100*s['bimodal_frac']:.1f}% of pixels, "
                  f"mean stability {s['mean_stability']:.2f} "
                  f"(stable {s['n_stable']}/{s['n_studied']})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs",  nargs="+", required=True)
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--ckpt",       default="output/experiments/ablation_cnn_barrier.pt",
                   help="ablation-format checkpoint for the NN comparison; '' to skip")
    p.add_argument("--patch_size", type=int,   default=9)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--prominence", type=float, default=0.15,
                   help="peak prominence threshold as fraction of curve max")
    p.add_argument("--n_scan",     type=int,   default=1000,
                   help="pixels per timestamp to scan for bimodality")
    p.add_argument("--n_study",    type=int,   default=30,
                   help="bimodal pixels per timestamp for the perturbation study")
    p.add_argument("--n_perturb",  type=int,   default=20,
                   help="noise draws per studied pixel")
    p.add_argument("--n_plot",     type=int,   default=8)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
