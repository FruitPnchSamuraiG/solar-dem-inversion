"""
Evaluate the four scaled checkpoints (array 15092854) on the untouched TEST
split, and plot per-pixel DEM curves against the solver.

Two outputs:

  1. Aggregate metrics on the 153-timestamp test split -- the first numbers from
     data no run has ever touched (training used 917 timestamps, model selection
     used the 153-timestamp val split).

  2. Per-pixel DEM curves. Each row is one pixel, shown twice: left against BP's
     own solution with the two barrier-loss networks, right against ENet's
     solution with the two enet-loss networks. Same pixel both sides -- the BP
     and ENet stagings share their AIA arrays, which is asserted, not assumed.

Run from the project root, on Torch, with the venv active:

    python3 experiments/eval_scaled.py \
        --bp_root   $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --enet_root $SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS \
        --ckpt_dir  output/experiments
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS
from src.scaled_eval import (BlockReader, assert_same_observations, count_peaks,
                             describe_ckpt, load_scaled_model)
from experiments.train_neural_field import pick_device
from experiments.train_scaled import evaluate, load_operators, make_loss_fn

# Fixed categorical order, never cycled: the solver is ink, then mlp6, then cnn.
# Blue/orange is the safest CVD pair (validated: worst-case protan dE 23.3), and
# each series also carries its own linestyle so identity never rests on hue.
C_REF, C_MLP6, C_CNN = "#1a1a1a", "#1F6FB2", "#E8762C"
STYLE = {
    "ref":  dict(color=C_REF,  lw=2.2, ls="-",  zorder=10),
    "mlp6": dict(color=C_MLP6, lw=1.8, ls="--", zorder=9),
    "cnn":  dict(color=C_CNN,  lw=1.8, ls="-.", zorder=8),
}

RUNS = [("mlp6", "barrier"), ("cnn", "barrier"),
        ("mlp6", "enet"), ("cnn", "enet")]


def find_ckpt(ckpt_dir, variant, loss):
    """Checkpoints are named by train_scaled's tag; accept the usual spellings."""
    for name in (f"scaled_{variant}_{loss}.pt", f"{variant}_{loss}.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    hits = [f for f in os.listdir(ckpt_dir)
            if f.endswith(".pt") and variant in f and loss in f]
    if len(hits) == 1:
        return os.path.join(ckpt_dir, hits[0])
    raise FileNotFoundError(
        f"no unique checkpoint for {variant}/{loss} in {ckpt_dir} (found {hits})")


# ── part 1: aggregate metrics on the test split ───────────────────────────────

def run_test_metrics(args, models, ckpts, D_t, B_t, device):
    print(f"\n{'='*78}\nTEST SPLIT -- 153 timestamps, never used for training or "
          f"model selection\n{'='*78}")
    results = {}
    for (variant, loss) in RUNS:
        root = args.bp_root if loss == "barrier" else args.enet_root
        _, loader = make_loader(root, "test", batch_blocks=args.batch_blocks,
                                num_workers=args.num_workers, shuffle=False,
                                with_labels=True,
                                pixels_per_block=args.pixels_per_block,
                                max_blocks=args.max_blocks)
        loss_fn = make_loss_fn(Namespace(**ckpts[(variant, loss)]["args"]))
        out = evaluate(models[(variant, loss)], loader, D_t, B_t, loss_fn,
                       device, N_AIA_BINS)
        results[f"{variant}_{loss}"] = out
        t = out["1"]
        print(f"{variant:>5} / {loss:<7}  test_loss={out['loss']:.4f}  "
              f"sp_coef={t['sp_coef']:.2f} (hist: BP 1.79)  "
              f"sp_dem nn/ref={t['sp_dem']:.2f}/{t['sp_ref']:.2f}  "
              f"mae_aia={t['mae_aia']:.3f}  n(tol=1)={t['n']:,}")
    return results


# ── part 1b: where the barrier loss actually lives ────────────────────────────

def barrier_terms(x, D_t, lb, ub, a_l1=1.0, mu=1.0):
    """Per-pixel barrier components, matching barrier_loss_batch term for term
    (which returns only the batch mean, so it cannot show a tail)."""
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8
    Dx = x @ D_t.T
    l1 = a_l1 * x.abs().sum(dim=1)
    below = mu * (torch.relu(lb - Dx) ** 2 / sigma2).sum(dim=1)
    above = mu * (torch.relu(Dx - ub) ** 2 / sigma2).sum(dim=1)
    return l1, below, above


@torch.no_grad()
def run_loss_tail(args, models, ckpts, D_t, device):
    """Decompose the barrier objective per pixel.

    Motivation: on the test split mlp6/barrier scores 44.32 against cnn's 11.27,
    while sp_coef and mae_aia agree to the third decimal. The barrier is a sum of
    squared constraint violations divided by sigma^2, so it is heavy-tailed by
    construction and a mean says nothing about whether the gap is a broad
    regression or a handful of catastrophic pixels. This separates the two.
    """
    print(f"\n{'='*78}\nBARRIER LOSS TAIL -- per-pixel decomposition\n{'='*78}")
    results = {}
    for (variant, loss) in RUNS:
        if loss != "barrier":
            continue                      # the enet runs minimise a different objective
        a = Namespace(**ckpts[(variant, loss)]["args"])
        _, loader = make_loader(args.bp_root, "test", batch_blocks=args.batch_blocks,
                                num_workers=args.num_workers, shuffle=False,
                                with_labels=True,
                                pixels_per_block=args.pixels_per_block,
                                max_blocks=args.max_blocks)
        tot, l1s, lbs, ubs, brights = [], [], [], [], []
        for batch in loader:
            patch, obs, lb, ub, dem, tol = (t.to(device) for t in flatten_blocks(batch))
            x = models[(variant, loss)](patch)
            l1, below, above = barrier_terms(x, D_t, lb, ub,
                                             a_l1=getattr(a, "alpha_l1", 1.0),
                                             mu=getattr(a, "mu", 1.0))
            tot.append((l1 + below + above).cpu().numpy())
            l1s.append(l1.cpu().numpy())
            lbs.append(below.cpu().numpy())
            ubs.append(above.cpu().numpy())
            brights.append(obs.sum(dim=1).cpu().numpy())

        tot = np.concatenate(tot)
        l1s, lbs, ubs = (np.concatenate(v) for v in (l1s, lbs, ubs))
        bright = np.concatenate(brights)
        order = np.argsort(tot)
        share = lambda frac: float(tot[order][-max(int(len(tot) * frac), 1):].sum()
                                   / max(tot.sum(), 1e-12))
        q = {f"p{p}": float(np.percentile(tot, p))
             for p in (50, 90, 99, 99.9, 99.99)}

        print(f"\n{variant} / barrier   mean={tot.mean():.3f}")
        print(f"  median={q['p50']:.3f}  p90={q['p90']:.3f}  p99={q['p99']:.3f}  "
              f"p99.9={q['p99.9']:.2f}  p99.99={q['p99.99']:.2f}  "
              f"max={tot.max():.4g}")
        print(f"  share of total loss from the worst 1%: {100*share(0.01):.1f}%   "
              f"worst 0.1%: {100*share(0.001):.1f}%")
        print(f"  mean by term:  l1={l1s.mean():.3f}  "
              f"below-lb={lbs.mean():.3f}  above-ub={ubs.mean():.3f}")
        # A tail concentrated in the bright decile is a targeted failure (flare
        # cores); one spread across deciles is a broad regression.
        edges = np.quantile(bright, np.linspace(0, 1, 11))
        print("  mean loss by brightness decile:")
        for k in range(10):
            sel = (bright >= edges[k]) & (bright <= edges[k + 1])
            if sel.sum():
                print(f"    d{k+1:>2} [{edges[k]:>9.3g},{edges[k+1]:>9.3g}]: "
                      f"mean={tot[sel].mean():>9.3f}  p99={np.percentile(tot[sel], 99):>10.3f}")

        results[f"{variant}_barrier"] = dict(
            q, mean=float(tot.mean()), max=float(tot.max()),
            share_worst_1pct=share(0.01), share_worst_0p1pct=share(0.001),
            mean_l1=float(l1s.mean()), mean_below=float(lbs.mean()),
            mean_above=float(ubs.mean()))
    return results


# ── part 2: per-pixel DEM curves ──────────────────────────────────────────────

def pick_pixels(reader, block_ids, n_per_block, seed=0):
    """Stratify by total brightness so the panel spans quiet sun to flare core.

    A uniform random draw is dominated by faint on-disk pixels, which all four
    models fit easily; the interesting disagreements are in the bright tail.
    """
    rng = np.random.default_rng(seed)
    picked = []
    for b in block_ids:
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) < n_per_block:
            continue
        bright = obs[:, rows, cols].sum(axis=0)
        order = np.argsort(bright)
        # even spread over the brightness rank, jittered so repeated runs on the
        # same block do not always return the identical pixel
        qs = np.linspace(0.05, 0.995, n_per_block)
        idx = np.clip((qs * len(order)).astype(int)
                      + rng.integers(-2, 3, size=n_per_block), 0, len(order) - 1)
        sel = order[idx]
        picked.append((b, rows[sel], cols[sel], bright[sel]))
    return picked


@torch.no_grad()
def plot_curves(args, models, D_t, B_t, logT, device, out_dir):
    bp = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    en = BlockReader(args.enet_root, "test", patch_size=args.patch_size)

    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(bp), size=min(args.n_blocks, len(bp)),
                           replace=False)
    picked = pick_pixels(bp, block_ids, args.pixels_per_panel, seed=args.seed)
    if not picked:
        raise RuntimeError("no block yielded enough valid pixels")

    rows_meta, figs = [], []
    for b, ii, jj, bright in picked:
        obs, err, tol, dem_bp = bp.read_block(b)
        _, _, _, dem_en = en.read_block(b)
        px_bp = bp.gather(obs, err, tol, dem_bp, ii, jj)
        px_en = en.gather(obs, err, tol, dem_en, ii, jj)

        patch = px_bp["patch"].to(device)
        preds = {}
        for (variant, loss) in RUNS:
            x = models[(variant, loss)](patch)
            preds[(variant, loss)] = ((x @ B_t.T)[:, :N_AIA_BINS].cpu().numpy(),
                                      x.cpu().numpy())

        n = len(ii)
        fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n), squeeze=False)
        for r in range(n):
            for c, (loss, px, label) in enumerate(
                    (("barrier", px_bp, "BP"), ("enet", px_en, "ENet"))):
                ax = axes[r][c]
                ref = px["dem"][r].numpy()
                ax.plot(logT, ref, label=f"{label} (solver)", **STYLE["ref"])
                for variant in ("mlp6", "cnn"):
                    ax.plot(logT, preds[(variant, loss)][0][r],
                            label=variant, **STYLE[variant])

                npk_ref, _ = count_peaks(ref)
                ax.set_title(
                    f"block {b} px ({ii[r]},{jj[r]})  "
                    f"tol={int(px['tol'][r])}  sum(obs)={bright[r]:.3g}  "
                    f"{label} peaks={npk_ref}", fontsize=7.5)
                ax.set_xlabel("log T", fontsize=7)
                ax.set_ylabel("DEM", fontsize=7)
                ax.tick_params(labelsize=6)
                # recessive frame: the curves are the content
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.grid(alpha=0.18, lw=0.5)
                if r == 0:
                    ax.legend(fontsize=6.5, frameon=False)

            rows_meta.append({
                "block": int(b), "i": int(ii[r]), "j": int(jj[r]),
                "tol": int(px_bp["tol"][r]), "sum_obs": float(bright[r]),
                "peaks_bp": int(count_peaks(px_bp["dem"][r].numpy())[0]),
                "peaks_enet": int(count_peaks(px_en["dem"][r].numpy())[0]),
                **{f"peaks_{v}_{l}": int(count_peaks(preds[(v, l)][0][r])[0])
                   for (v, l) in RUNS},
            })

        fig.suptitle(f"Held-out test pixels -- solver vs networks (block {b})",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = os.path.join(out_dir, f"dem_curves_block{b:05d}.png")
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        figs.append(path)
        print(f"  wrote {path}")

    return figs, rows_meta


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    assert_same_observations(args.bp_root, args.enet_root, "test",
                             n_check=args.n_align_check)

    D_t, B_t, n_basis, logT = load_operators(device)
    print(f"D: {tuple(D_t.shape)}   basis: {n_basis}   temps: {len(logT)}")

    models, ckpts = {}, {}
    for key in RUNS:
        path = find_ckpt(args.ckpt_dir, *key)
        m, c = load_scaled_model(path, n_basis, device)
        models[key], ckpts[key] = m, c
        print(f"  loaded {describe_ckpt(c)}   <- {os.path.basename(path)}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    if not args.skip_metrics:
        summary["test_metrics"] = run_test_metrics(args, models, ckpts,
                                                   D_t, B_t, device)
    if not args.skip_tail:
        summary["loss_tail"] = run_loss_tail(args, models, ckpts, D_t, device)

    if args.skip_curves:
        with open(os.path.join(out_dir, "eval_scaled_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    print(f"\n{'='*78}\nPER-PIXEL DEM CURVES\n{'='*78}")
    figs, rows_meta = plot_curves(args, models, D_t, B_t, logT, device, out_dir)
    summary["curve_pixels"] = rows_meta
    summary["figures"] = figs

    with open(os.path.join(out_dir, "eval_scaled_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {os.path.join(out_dir, 'eval_scaled_summary.json')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--enet_root", required=True)
    p.add_argument("--ckpt_dir", default="output/experiments")
    p.add_argument("--out_dir", default="output/experiments/eval_scaled")
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--batch_blocks", type=int, default=16)
    p.add_argument("--pixels_per_block", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_blocks", type=int, default=None,
                   help="cap test blocks for the aggregate metrics (debug)")
    p.add_argument("--n_blocks", type=int, default=3,
                   help="how many test blocks to draw curve pixels from")
    p.add_argument("--pixels_per_panel", type=int, default=6)
    p.add_argument("--n_align_check", type=int, default=4)
    p.add_argument("--skip_metrics", action="store_true")
    p.add_argument("--skip_tail", action="store_true")
    p.add_argument("--skip_curves", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
