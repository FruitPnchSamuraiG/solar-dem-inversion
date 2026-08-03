"""
Decide the production width: every sweep checkpoint on the untouched TEST split,
scored on the metrics that actually settle the question.

The sweep's own logs report VALIDATION numbers, and validation is the split that
chose each run's saved epoch -- mildly optimistic by construction. This rescores
all 16 checkpoints (8 widths x 2 losses) on the 153-timestamp test split, and
adds the two things the training logs cannot show:

  * Loss PERCENTILES, not the mean. On 2026-08-02 a single pixel out of
    5,013,504 was 94% of mlp6/barrier's mean loss and inverted the ranking
    against cnn. The mean is not a usable statistic for this objective.

  * Bimodality precision/recall against BP's interior-bimodal pixels. The
    aggregate metrics are averages over millions of easy pixels and will not
    reveal a small model losing the hard multi-thermal ones. At 1.43M,
    mlp6/barrier hit 80.75% precision against a 14.17% base rate; if that
    collapses at a width where sp_coef and mae_aia still look fine, the
    aggregate numbers were hiding the regression.

One pass per checkpoint for metrics, then a single census pass that scores every
model on the same pixels (blocks are decompressed once and shared, so 16 models
cost barely more than one).

    python3 experiments/eval_sweep.py \
        --bp_root   $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --enet_root $SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS \
        --ckpt_dir  output/experiments/sweep
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from argparse import Namespace

import numpy as np
import torch

from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS
from src.scaled_eval import BlockReader, load_scaled_model
from experiments.train_neural_field import effective_sparsity, pick_device
from experiments.train_scaled import load_operators
from experiments.eval_scaled import barrier_terms, find_ckpt
from experiments.bimodal_scaled import _network_peaks, count_peaks_batch

WIDTHS = [680, 480, 336, 232, 160, 108, 72, 48]
PARAMS = {680: "1.43M", 480: "722k", 336: "360k", 232: "176k",
          160: "87k", 108: "42k", 72: "20k", 48: "10k"}
LOSSES = ["barrier", "enet"]


@torch.no_grad()
def test_metrics(model, ckpt, loader, D_t, B_t, device):
    """One pass: sparsity, both MAEs, and the barrier-loss distribution.

    Percentiles rather than a mean, per the 2026-08-02 finding. The barrier
    terms are computed for every run including enet ones -- for an enet model it
    is not its own objective, so it is reported only as a comparable diagnostic,
    never as that model's training loss.
    """
    a = Namespace(**ckpt["args"])
    a_l1 = getattr(a, "alpha_l1", 1.0)
    mu = getattr(a, "mu", 1.0)

    acc = {k: 0.0 for k in ("sp_coef", "sp_dem", "sp_ref", "mae_dem", "mae_aia")}
    n = 0
    losses = []

    for batch in loader:
        patch, obs, lb, ub, dem, tol = (t.to(device) for t in flatten_blocks(batch))
        sel = tol == 1                      # tight-tolerance pixels, as in training
        if not bool(sel.any()):
            continue
        x = model(patch)
        pred = (x @ B_t.T)[:, :N_AIA_BINS]

        vals = {
            "sp_coef": effective_sparsity(x),
            "sp_dem": effective_sparsity(pred),
            "sp_ref": effective_sparsity(dem),
            "mae_dem": (pred - dem).abs().mean(dim=1),
            "mae_aia": (x @ D_t.T - obs).abs().mean(dim=1),
        }
        k = int(sel.sum())
        n += k
        for key in acc:
            acc[key] += float(vals[key][sel].sum())

        l1, below, above = barrier_terms(x, D_t, lb, ub, a_l1=a_l1, mu=mu)
        losses.append((l1 + below + above)[sel].cpu().numpy())

    losses = np.concatenate(losses) if losses else np.zeros(1)
    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["n"] = n
    out["loss_mean"] = float(losses.mean())
    for p in (50, 90, 99, 99.9):
        out[f"loss_p{p}"] = float(np.percentile(losses, p))
    out["loss_max"] = float(losses.max())
    return out


@torch.no_grad()
def bimodal_census(reader, block_ids, models, B_t, device, prominence):
    """Interior-bimodal cross-tab for every model, on one shared set of pixels.

    Interior-only: the boundary bins sit at the ends of AIA's temperature
    response, and a curve merely rising into one registers as a peak there.
    """
    conf = {k: np.zeros(4, dtype=np.int64) for k in models}
    n_tot = n_bi = 0

    for b in block_ids:
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        curves = np.ascontiguousarray(dem[:, rows, cols].T)
        _, ref_in = count_peaks_batch(curves, prominence, interior=True)
        ref_bi = ref_in >= 2
        n_tot += len(rows)
        n_bi += int(ref_bi.sum())

        peaks = _network_peaks(reader, obs, err, tol, rows, cols, models,
                               B_t, device, N_AIA_BINS, prominence)
        for key, (_, pk_in) in peaks.items():
            nn_bi = pk_in >= 2
            conf[key] += np.array([
                int((nn_bi & ref_bi).sum()), int((nn_bi & ~ref_bi).sum()),
                int((~nn_bi & ref_bi).sum()), int((~nn_bi & ~ref_bi).sum())])

    base = n_bi / max(n_tot, 1)
    out = {}
    for key, (tp, fp, fn, tn) in conf.items():
        out[key] = {"recall": tp / max(tp + fn, 1),
                    "precision": tp / max(tp + fp, 1),
                    "rate": (tp + fp) / max(tp + fp + fn + tn, 1)}
    return out, base, n_tot


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    print(f"D: {tuple(D_t.shape)}   basis: {n_basis}   temps: {len(logT)}")

    widths = args.widths or WIDTHS
    models, ckpts = {}, {}
    for w in widths:
        for loss in LOSSES:
            try:
                path = find_ckpt(args.ckpt_dir, "mlp6", loss, suffix=f"_h{w}")
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue
            m, c = load_scaled_model(path, n_basis, device)
            models[(w, loss)] = m
            ckpts[(w, loss)] = c
    print(f"Loaded {len(models)} checkpoints\n")

    # ── test-split metrics ────────────────────────────────────────────────────
    print("=" * 100)
    print("TEST SPLIT -- 153 timestamps, never used for training or epoch selection")
    print("=" * 100)
    results = {}
    for loss in LOSSES:
        root = args.bp_root if loss == "barrier" else args.enet_root
        print(f"\n{loss.upper()}  (reference: BP sp_coef 1.79)")
        print(f"  {'hidden':>7} {'params':>7} {'sp_coef':>8} {'mae_dem':>8} "
              f"{'mae_aia':>8} {'p50':>7} {'p99':>8} {'p99.9':>9} {'mean':>10}")
        for w in widths:
            if (w, loss) not in models:
                continue
            _, loader = make_loader(root, "test", batch_blocks=args.batch_blocks,
                                    num_workers=args.num_workers, shuffle=False,
                                    with_labels=True,
                                    pixels_per_block=args.pixels_per_block,
                                    max_blocks=args.max_blocks)
            r = test_metrics(models[(w, loss)], ckpts[(w, loss)], loader,
                             D_t, B_t, device)
            results[f"h{w}_{loss}"] = r
            print(f"  {w:>7} {PARAMS.get(w, '?'):>7} {r['sp_coef']:>8.3f} "
                  f"{r['mae_dem']:>8.4f} {r['mae_aia']:>8.3f} "
                  f"{r['loss_p50']:>7.3f} {r['loss_p99']:>8.3f} "
                  f"{r['loss_p99.9']:>9.2f} {r['loss_mean']:>10.4g}")
        print("  (mae_dem compares within a loss only -- BP's spiky DEMs are "
              "harder to match than\n   ENet's smooth ones; mae_aia is the "
              "cross-loss-comparable number. Rank on percentiles,\n   not the "
              "mean: one pixel in 5M carried 94% of it on 2026-08-02.)")

    # ── bimodality, the capacity-sensitive metric ─────────────────────────────
    print("\n" + "=" * 100)
    print("INTERIOR BIMODALITY vs BP -- the metric aggregate averages would hide")
    print("=" * 100)
    reader = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(reader), size=min(args.n_blocks, len(reader)),
                           replace=False)
    bim, base, n_scanned = bimodal_census(reader, block_ids, models, B_t,
                                          device, args.prominence)
    print(f"\nScanned {n_scanned:,} pixels; BP interior-bimodal base rate "
          f"{100*base:.2f}%  (1.43M baseline: 80.75% precision, 29.85% recall)")
    for loss in LOSSES:
        print(f"\n{loss.upper()}")
        print(f"  {'hidden':>7} {'params':>7} {'rate':>7} {'recall':>8} {'precision':>10}")
        for w in widths:
            if (w, loss) not in bim:
                continue
            d = bim[(w, loss)]
            results[f"h{w}_{loss}"]["bimodal"] = d
            print(f"  {w:>7} {PARAMS.get(w, '?'):>7} {100*d['rate']:>6.2f}% "
                  f"{100*d['recall']:>7.2f}% {100*d['precision']:>9.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "eval_sweep_summary.json")
    with open(path, "w") as f:
        json.dump({"base_rate_interior_bimodal": base,
                   "n_census_pixels": n_scanned, "runs": results}, f, indent=2)
    print(f"\nSummary -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--enet_root", required=True)
    p.add_argument("--ckpt_dir", default="output/experiments/sweep")
    p.add_argument("--out_dir", default="output/experiments/eval_sweep")
    p.add_argument("--widths", type=int, nargs="+", default=None)
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--batch_blocks", type=int, default=16)
    p.add_argument("--pixels_per_block", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_blocks", type=int, default=None,
                   help="cap test blocks per metrics pass (16 passes, so this "
                        "is the main runtime knob)")
    p.add_argument("--n_blocks", type=int, default=20,
                   help="test blocks for the bimodal census")
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
