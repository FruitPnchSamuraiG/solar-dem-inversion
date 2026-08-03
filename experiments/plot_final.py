"""
Final production-model figures: the chosen h232 (176k) model against the 1.43M
baseline and the solver, on the untouched test split.

Two figures:

  1. DEM curves. Each row is one pixel shown twice -- left against BP's own
     solution, right against ENet's. Three lines per panel: the solver, the
     1.43M baseline, and the chosen 176k model. The question the figure has to
     answer is whether the 8x smaller model tracks the baseline, so both are
     drawn rather than the small one alone.

  2. Bimodal examples showing two genuine, comparable temperature components:
     both peaks well inside the logT range, separated by 3-8 bins, with the
     weaker peak at least a quarter the height of the taller. See
     `find_clean_bimodal` for why each of those bounds exists -- every one of
     them replaced a criterion that selected the wrong pixels.

    python3 experiments/plot_final.py \
        --bp_root   $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --enet_root $SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS \
        --sweep_dir output/experiments/sweep \
        --base_dir  output/experiments
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

from src.zarr_data import N_AIA_BINS
from src.scaled_eval import BlockReader, count_peaks, load_scaled_model
from experiments.train_neural_field import pick_device
from experiments.train_scaled import load_operators
from experiments.eval_scaled import find_ckpt, pick_pixels
from experiments.bimodal_scaled import count_peaks_batch, solve_bp

# Solver is ink; the two networks are the validated blue/orange CVD pair, each
# with its own linestyle so identity never rests on hue alone.
STYLE = {
    "ref":  dict(color="#1a1a1a", lw=2.2, ls="-",  zorder=10),
    "base": dict(color="#1F6FB2", lw=1.7, ls="--", zorder=9),
    "small": dict(color="#E8762C", lw=1.7, ls="-.", zorder=8),
}
C_PERTURB = "#b9b9b9"


def load_pair(base_dir, sweep_dir, loss, width, n_basis, device):
    """The 1.43M baseline and the chosen width, for one loss."""
    base_path = find_ckpt(base_dir, "mlp6", loss)
    small_path = find_ckpt(sweep_dir, "mlp6", loss, suffix=f"_h{width}")
    base, _ = load_scaled_model(base_path, n_basis, device)
    small, _ = load_scaled_model(small_path, n_basis, device)
    print(f"  {loss:<8} baseline <- {os.path.basename(base_path)}")
    print(f"  {loss:<8} h{width:<7} <- {os.path.basename(small_path)}")
    return base, small


# ── figure 1: DEM curves ──────────────────────────────────────────────────────

@torch.no_grad()
def plot_curves(args, models, B_t, logT, device, out_dir):
    bp = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    en = BlockReader(args.enet_root, "test", patch_size=args.patch_size)

    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(bp), size=min(args.n_blocks, len(bp)), replace=False)
    picked = pick_pixels(bp, block_ids, args.pixels_per_panel, seed=args.seed)
    if not picked:
        raise RuntimeError("no block yielded enough valid pixels")

    paths, meta = [], []
    for b, ii, jj, bright in picked:
        obs, err, tol, dem_bp = bp.read_block(b)
        _, _, _, dem_en = en.read_block(b)
        px = bp.gather(obs, err, tol, dem_bp, ii, jj)
        px_en = en.gather(obs, err, tol, dem_en, ii, jj)
        patch = px["patch"].to(device)

        pred = {k: ((m(patch) @ B_t.T)[:, :N_AIA_BINS]).cpu().numpy()
                for k, m in models.items()}

        n = len(ii)
        fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n), squeeze=False)
        for r in range(n):
            for c, (loss, ref_px, label) in enumerate(
                    (("barrier", px, "BP"), ("enet", px_en, "ENet"))):
                ax = axes[r][c]
                ref = ref_px["dem"][r].numpy()
                ax.plot(logT, ref, label=f"{label} (solver)", **STYLE["ref"])
                ax.plot(logT, pred[(loss, "base")][r], label="mlp6 1.43M",
                        **STYLE["base"])
                ax.plot(logT, pred[(loss, "small")][r],
                        label=f"mlp6 {args.width_label}", **STYLE["small"])
                ax.set_title(f"block {b} px ({ii[r]},{jj[r]})  tol={int(px['tol'][r])}  "
                             f"sum(obs)={bright[r]:.3g}  "
                             f"{label} peaks={count_peaks(ref)[0]}", fontsize=7.5)
                ax.set_xlabel("log T", fontsize=7)
                ax.set_ylabel("DEM", fontsize=7)
                ax.tick_params(labelsize=6)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.grid(alpha=0.18, lw=0.5)
                if r == 0:
                    ax.legend(fontsize=6.5, frameon=False)
            meta.append({"block": int(b), "i": int(ii[r]), "j": int(jj[r]),
                         "sum_obs": float(bright[r])})

        fig.suptitle(f"Held-out test pixels -- solver vs 1.43M baseline vs "
                     f"{args.width_label} production model", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        p = os.path.join(out_dir, f"final_dem_curves_block{b:05d}.png")
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        paths.append(p)
        print(f"  wrote {p}")
    return paths, meta


# ── figure 2: clean interior-bimodal examples ─────────────────────────────────

def _two_peak_shape(curve, prominence, m, args):
    """(separation, weaker/taller ratio, weaker peak height) or None.

    Exactly two peaks required when args.exact_two: with 6 AIA channels the
    data cannot resolve three independent temperature components, so a 3+ peak
    solution is a sparsity artifact rather than a physical claim.
    """
    T = len(curve)
    n, where = count_peaks(curve, prominence)
    mid = where[(where >= m) & (where <= T - 1 - m)]
    if len(mid) < 2 or (args.exact_two and len(mid) != 2):
        return None
    sep = int(mid.max() - mid.min())
    if not (args.min_sep <= sep <= args.max_sep):
        return None
    h = curve[mid]
    return sep, float(h.min() / max(h.max(), 1e-12)), float(h.min())


@torch.no_grad()
def find_clean_bimodal(reader, block_ids, args, model=None, B_t=None,
                       device=None, chunk=4096):
    """Pixels showing two genuine, comparable temperature components.

    Filters, each of which replaced one that selected the wrong pixels:

      * Peaks inside [edge_margin, T-1-edge_margin]. Excluding only the
        outermost bin still admits peaks one step in, riding the same loss of
        AIA sensitivity.
      * Separation bounded ABOVE as well as below. With 18 bins, "15 apart" is
        bin 1 and bin 16 -- end-to-end, the boundary case again. Ranking by
        widest separation actively selected for it.
      * Weaker peak at least `--min_ratio` of the taller, which is what "two
        components" means rather than one peak plus a ripple.
      * Brightness from the upper tail. A mid quantile drew pixels at sum(obs)
        ~35 where the DEM is ~0.1 and BP's own perturbed solves scatter freely.

    With `model` given, keeps only pixels where the NETWORK is also bimodal.
    Without it the ranking finds BP's strongest double peaks, which are
    overwhelmingly hot-second-peak cases the network misses -- a true picture of
    the 29.85% recall, but it shows the model failing at every panel. Both
    groups are real and the honest figure needs each labelled.
    """
    hits = []
    for b in block_ids:
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        curves = np.ascontiguousarray(dem[:, rows, cols].T)
        _, npk_in = count_peaks_batch(curves, args.prominence, interior=True)
        bright = obs[:, rows, cols].sum(axis=0)
        lo, hi = np.quantile(bright, [args.bright_lo, args.bright_hi])
        m = args.edge_margin

        cand = []
        for k in np.nonzero(npk_in >= 2)[0]:
            if not (lo <= bright[k] <= hi):
                continue
            shape = _two_peak_shape(curves[k], args.prominence, m, args)
            if shape is None or shape[1] < args.min_ratio:
                continue
            cand.append((int(k), shape))
        if not cand:
            continue

        if model is not None:
            idx = np.array([k for k, _ in cand])
            px = reader.gather(obs, err, tol, None, rows[idx], cols[idx])
            patch = px["patch"]
            keep = np.zeros(len(idx), dtype=bool)
            for s in range(0, len(idx), chunk):
                x = model(patch[s:s + chunk].to(device))
                pred = (x @ B_t.T)[:, :N_AIA_BINS].cpu().numpy()
                for t in range(pred.shape[0]):
                    sh = _two_peak_shape(pred[t], args.prominence, m, args)
                    keep[s + t] = sh is not None and sh[1] >= args.nn_min_ratio
            cand = [c for c, kp in zip(cand, keep) if kp]

        for k, (sep, ratio, weak_h) in cand:
            hits.append((int(b), int(rows[k]), int(cols[k]), float(bright[k]),
                         sep, ratio, weak_h))
    return hits


@torch.no_grad()
def _build_records(reader, hits, models, B_t, D64, B_np, rng, args, device, group):
    recs = []
    for (b, i, j, bright, sep, ratio, weak_h) in sorted(hits, key=lambda h: h[0]):
        obs_a, err_a, tol_a, dem_a = reader.read_block(b)
        obs = obs_a[:, i, j].astype(np.float64)
        err = err_a[:, i, j].astype(np.float64)
        x_bp, _ = solve_bp(D64, obs, err)
        if x_bp is None:
            continue

        perturbed, n_still = [], 0
        for _ in range(args.n_perturb):
            x_p, _ = solve_bp(D64, obs + rng.normal(0, 1, size=obs.shape) * err, err)
            if x_p is None:
                continue
            d = np.maximum(B_np @ x_p, 0)
            perturbed.append(d)
            if count_peaks(d, args.prominence)[0] >= 2:
                n_still += 1

        px = reader.gather(obs_a, err_a, tol_a, dem_a, [i], [j])
        patch = px["patch"].to(device)
        pred = {k: ((m(patch) @ B_t.T)[:, :N_AIA_BINS]).cpu().numpy()[0]
                for k, m in models.items()}
        recs.append({"block": b, "i": i, "j": j, "bright": bright, "sep": sep,
                     "ratio": ratio, "group": group,
                     "stab": n_still / max(len(perturbed), 1),
                     "dem_bp": np.maximum(B_np @ x_bp, 0),
                     "perturbed": perturbed, "pred": pred})
    return recs


def plot_bimodal(args, models, D_t, B_t, logT, device, out_dir):
    """Two groups, labelled: pixels the network reproduces, and pixels it misses.

    Showing only one group misleads in opposite directions -- agreements alone
    hide the 29.85% recall, misses alone hide the 80.75% precision. Both are
    real and the figure has to carry both.
    """
    reader = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(reader), size=min(args.n_blocks_bimodal, len(reader)),
                           replace=False)

    crit = (f"peaks in bins [{args.edge_margin},{N_AIA_BINS-1-args.edge_margin}], "
            f"sep {args.min_sep}-{args.max_sep}, weaker >= {args.min_ratio:.2f} "
            f"of taller"
            + (", exactly 2 peaks" if args.exact_two else ""))

    agree = find_clean_bimodal(reader, block_ids, args,
                               model=models[("barrier", "small")],
                               B_t=B_t, device=device)
    allhits = find_clean_bimodal(reader, block_ids, args)
    agree_key = {(h[0], h[1], h[2]) for h in agree}
    miss = [h for h in allhits if (h[0], h[1], h[2]) not in agree_key]
    print(f"  {crit}")
    print(f"  {len(agree)} where the {args.width_label} network is ALSO bimodal, "
          f"{len(miss)} where it is not")
    if not agree and not miss:
        print("  none found; skipping bimodal figure")
        return [], []

    # Tallest weaker peak first -- visually unambiguous double peaks, and for
    # the agreement group the ones that best show the model reproducing both.
    n_each = max(args.n_plot // 2, 1)
    picks = ([( "agrees", h) for h in sorted(agree, key=lambda h: -h[6])[:n_each]]
             + [("misses", h) for h in sorted(miss, key=lambda h: -h[6])[:n_each]])

    D64 = D_t.cpu().numpy().astype(np.float64)
    B_np = B_t.cpu().numpy()
    recs = []
    for group in ("agrees", "misses"):
        hs = [h for g, h in picks if g == group]
        if hs:
            recs += _build_records(reader, hs, models, B_t, D64, B_np, rng,
                                   args, device, group)
    if not recs:
        return [], []
    recs.sort(key=lambda r: (r["group"] != "agrees",))

    fig, axes = plt.subplots(len(recs), 1, figsize=(9.5, 3.0 * len(recs)),
                             squeeze=False)
    for ax, r in zip(axes[:, 0], recs):
        for d in r["perturbed"]:
            ax.plot(logT, d, color=C_PERTURB, lw=0.6, alpha=0.5, zorder=1)
        ax.plot(logT, r["dem_bp"], label=f"BP (stability {r['stab']:.2f})",
                **STYLE["ref"])
        ax.plot(logT, r["pred"][("barrier", "base")], label="mlp6 1.43M",
                **STYLE["base"])
        ax.plot(logT, r["pred"][("barrier", "small")],
                label=f"mlp6 {args.width_label}", **STYLE["small"])
        tag = ("NETWORK REPRODUCES BOTH PEAKS" if r["group"] == "agrees"
               else "NETWORK MISSES THE SECOND PEAK")
        ax.set_title(f"[{tag}]  block {r['block']} px ({r['i']},{r['j']})  "
                     f"sum(obs)={r['bright']:.3g}  sep {r['sep']} bins  "
                     f"weaker/taller {r['ratio']:.2f}  "
                     f"gray: {len(r['perturbed'])} perturbed BP solves",
                     fontsize=8)
        ax.set_xlabel("log T", fontsize=7)
        ax.set_ylabel("DEM", fontsize=7)
        ax.tick_params(labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.18, lw=0.5)
        ax.legend(fontsize=7, frameon=False, ncol=3)

    fig.suptitle("Bimodal BP pixels: where the network reproduces both "
                 "components (top) and where it misses the hot one (bottom)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(out_dir, "final_bimodal.png")
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    print(f"  wrote {p}")
    return [p], [{k: v for k, v in r.items()
                  if k not in ("perturbed", "dem_bp", "pred")} for r in recs]


def main():
    args = parse_args()
    args.width_label = f"{args.width_params} (h{args.width})"
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    models = {}
    for loss in ("barrier", "enet"):
        base, small = load_pair(args.base_dir, args.sweep_dir, loss, args.width,
                                n_basis, device)
        models[(loss, "base")] = base
        models[(loss, "small")] = small

    os.makedirs(args.out_dir, exist_ok=True)
    print("\nDEM curves:")
    curve_paths, curve_meta = plot_curves(args, models, B_t, logT, device, args.out_dir)
    print("\nBimodal examples:")
    bim_paths, bim_meta = plot_bimodal(args, models, D_t, B_t, logT, device,
                                       args.out_dir)

    with open(os.path.join(args.out_dir, "plot_final_summary.json"), "w") as f:
        json.dump({"width": args.width, "curves": curve_meta,
                   "bimodal": bim_meta,
                   "figures": curve_paths + bim_paths}, f, indent=2)
    print(f"\nSummary -> {os.path.join(args.out_dir, 'plot_final_summary.json')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--enet_root", required=True)
    p.add_argument("--sweep_dir", default="output/experiments/sweep")
    p.add_argument("--base_dir", default="output/experiments")
    p.add_argument("--out_dir", default="output/experiments/final")
    p.add_argument("--width", type=int, default=232)
    p.add_argument("--width_params", default="176k")
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--pixels_per_panel", type=int, default=6)
    p.add_argument("--n_blocks_bimodal", type=int, default=20)
    p.add_argument("--n_plot", type=int, default=8)
    p.add_argument("--n_perturb", type=int, default=30)
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--min_sep", type=int, default=3,
                   help="minimum bins between the two peaks")
    p.add_argument("--max_sep", type=int, default=8,
                   help="maximum bins apart; above this the pair is end-to-end, "
                        "i.e. the boundary case in disguise")
    p.add_argument("--edge_margin", type=int, default=2,
                   help="peaks must sit in bins [margin, T-1-margin]")
    p.add_argument("--min_ratio", type=float, default=0.25,
                   help="weaker peak as a fraction of the taller one")
    p.add_argument("--nn_min_ratio", type=float, default=0.15,
                   help="same test on the network's curve; looser because the "
                        "network's second peak is systematically shallower")
    p.add_argument("--exact_two", action="store_true", default=True,
                   help="require exactly 2 peaks: 6 AIA channels cannot resolve "
                        "3 independent temperature components")
    p.add_argument("--allow_three_plus", dest="exact_two", action="store_false")
    p.add_argument("--bright_lo", type=float, default=0.90)
    p.add_argument("--bright_hi", type=float, default=0.999)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
