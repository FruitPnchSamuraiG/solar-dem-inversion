"""
Why does the barrier loss have a 10^8 tail?

On the test split mlp6/barrier scores mean 44.32 against cnn's 11.27, while
being *better* than cnn at the median, p90, p99, p99.9 and p99.99. The whole
reversal lives beyond the 99.99th percentile -- a few hundred pixels out of
5,013,504, with a maximum of 2.09e8.

ANSWER (2026-08-02, full test split). Not the normalisation. The suspected
mechanism was a near-zero sigma in the denominator; the worst pixel has a
perfectly healthy one:

    211A  obs=0.062  err=0.611  band=[-0.793, 0.917]  Dx=9023

The observation carries no signal, the error bar is ten times larger, and the
network over-predicts by 9000x. Flooring sigma changes the mean by 0.07%.

*A single pixel is 94% of mlp6's mean loss* (2.09e8 of a 2.22e8 total over
5,013,504 pixels), which is the whole 44.32-vs-11.27 gap against cnn -- mlp6 is
better at the median, p90, p99, p99.9 and p99.99. Report percentiles, never the
mean, for this objective.

Nor is it a training problem: at ~1 pathological pixel per epoch with gradient
clipping already bounding its step, the four trained models are unaffected.

The `lb < 0` criterion is *not* a usable filter either -- it catches 89% of
pixels, because AIA's faint channels are genuinely noise-dominated (94A has a
median of ~0.76 DN against a read-noise floor of the same size). The narrow
population worth excluding is near-zero readings in the *bright* channels
(171/193/211A), which is the deconvolution positivity clamp.

Three outputs:

  1. The worst N pixels, with per-channel obs, err, err/obs, the fitted Dx, the
     band it missed, and which channel carries the penalty.
  2. The distribution of err/obs over the whole split, per channel -- is a near
     zero error bar a handful of pixels or a population?
  3. A counterfactual: the mean and percentiles of the same loss with sigma
     floored at various fractions of the observation. If a floor collapses the
     mean onto the median while barely moving the percentiles, the tail is an
     artifact of the normalisation and the floor is the fix.

Run from the project root on Torch:

    python3 experiments/diag_loss_outliers.py \
        --bp_root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --ckpt_dir output/experiments
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from argparse import Namespace

import numpy as np
import torch

from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS
from src.scaled_eval import describe_ckpt, load_scaled_model
from experiments.eval_scaled import find_ckpt
from experiments.train_neural_field import pick_device
from experiments.train_scaled import load_operators

CHANNELS = ["94A", "131A", "171A", "193A", "211A", "335A"]
# Candidate floors, as a fraction of the pixel's own observation: sigma is
# forced to at least frac * obs in each channel. 0.0 is the current behaviour.
FLOORS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)


def per_pixel_loss(x, D_t, obs, lb, ub, a_l1, mu, floor=0.0):
    """Barrier loss per pixel, optionally with a relative floor under sigma.

    floor=0.0 reproduces barrier_loss_batch exactly (which returns only the
    batch mean, so it cannot expose a tail).
    """
    sigma = (ub - lb) / 2
    if floor > 0:
        sigma = torch.maximum(sigma, floor * obs.abs())
    sigma2 = sigma ** 2 + 1e-8
    Dx = x @ D_t.T
    per_ch = mu * ((torch.relu(lb - Dx) ** 2 + torch.relu(Dx - ub) ** 2) / sigma2)
    return a_l1 * x.abs().sum(dim=1) + per_ch.sum(dim=1), per_ch, Dx


@torch.no_grad()
def scan(model, loader, D_t, device, a_l1, mu, topn):
    """One pass: keep the worst `topn` pixels, plus global error statistics."""
    keep = None                    # dict of numpy arrays for the worst pixels
    err_ratio_sum = np.zeros(6)    # mean err/obs per channel
    err_ratio_min = np.full(6, np.inf)
    n_seen = 0
    tiny = {t: 0 for t in (1e-4, 1e-3, 1e-2)}   # pixels with min err/obs below t
    floor_sums = {f: 0.0 for f in FLOORS}
    floor_all = {f: [] for f in FLOORS}
    # Pixels whose band straddles zero: lb = obs - tolfac*err < 0 in some channel,
    # so the constraint degenerates to "emit less than ub" and carries no lower
    # information, while still able to generate unbounded loss upward.
    neg = {"n": 0, "sum": 0.0, "vals": []}
    pos = {"n": 0, "sum": 0.0, "vals": []}

    for batch in loader:
        patch, obs, lb, ub, dem, tol = (t.to(device) for t in flatten_blocks(batch))
        x = model(patch)

        for f in FLOORS:
            lo, per_ch, Dx = per_pixel_loss(x, D_t, obs, lb, ub, a_l1, mu, floor=f)
            floor_sums[f] += float(lo.sum())
            # subsample for percentiles: 5M x 6 floats per floor would not fit
            floor_all[f].append(lo[::37].cpu().numpy())
            if f == 0.0:
                loss0, per_ch0, Dx0 = lo, per_ch, Dx

        good = (lb > 0).all(dim=1)
        for tgt, sel in ((neg, ~good), (pos, good)):
            tgt["n"] += int(sel.sum())
            if sel.any():
                v = loss0[sel]
                tgt["sum"] += float(v.sum())
                tgt["vals"].append(v[::7].cpu().numpy())

        err = (ub - lb) / 2 / 1.4                      # back out the staged error
        ratio = (err / obs.clamp(min=1e-12)).cpu().numpy()
        err_ratio_sum += ratio.sum(axis=0)
        err_ratio_min = np.minimum(err_ratio_min, ratio.min(axis=0))
        rmin = ratio.min(axis=1)
        for t in tiny:
            tiny[t] += int((rmin < t).sum())
        n_seen += obs.shape[0]

        # worst pixels in this batch, merged into the running set
        k = min(topn, loss0.numel())
        idx = torch.topk(loss0, k).indices
        cand = {
            "loss": loss0[idx].cpu().numpy(),
            "obs": obs[idx].cpu().numpy(),
            "err": err[idx].cpu().numpy(),
            "Dx": Dx0[idx].cpu().numpy(),
            "lb": lb[idx].cpu().numpy(),
            "ub": ub[idx].cpu().numpy(),
            "per_ch": per_ch0[idx].cpu().numpy(),
            "tol": tol[idx].cpu().numpy(),
        }
        if keep is None:
            keep = cand
        else:
            keep = {k2: np.concatenate([keep[k2], cand[k2]]) for k2 in keep}
        order = np.argsort(-keep["loss"])[:topn]
        keep = {k2: v[order] for k2, v in keep.items()}

    def _grp(g):
        v = np.concatenate(g["vals"]) if g["vals"] else np.zeros(1)
        return {"n": g["n"], "frac": g["n"] / max(n_seen, 1),
                "mean": g["sum"] / max(g["n"], 1),
                "share_of_total": g["sum"] / max(neg["sum"] + pos["sum"], 1e-12),
                "pct": np.percentile(v, [50, 90, 99, 99.9, 100]).tolist()}

    stats = {
        "n": n_seen,
        "neg_band": _grp(neg),
        "pos_band": _grp(pos),
        "err_ratio_mean": (err_ratio_sum / max(n_seen, 1)).tolist(),
        "err_ratio_min": err_ratio_min.tolist(),
        "tiny": {str(t): c for t, c in tiny.items()},
        "floor_mean": {str(f): floor_sums[f] / max(n_seen, 1) for f in FLOORS},
        "floor_pct": {str(f): np.percentile(np.concatenate(v),
                                            [50, 90, 99, 99.9, 100]).tolist()
                      for f, v in floor_all.items()},
    }
    return keep, stats


def report(name, keep, stats, args):
    print(f"\n{'='*78}\n{name}\n{'='*78}")

    print(f"\nWorst {min(args.show, len(keep['loss']))} pixels "
          f"(of {stats['n']:,} scanned):")
    print(f"  {'loss':>12}  {'tol':>3}  {'ch':>5}  {'obs':>10}  {'err':>10}  "
          f"{'err/obs':>9}  {'Dx':>10}  {'band':>21}  {'this ch':>12}")
    for r in range(min(args.show, len(keep["loss"]))):
        c = int(np.argmax(keep["per_ch"][r]))       # the channel doing the damage
        print(f"  {keep['loss'][r]:>12.4g}  {int(keep['tol'][r]):>3}  "
              f"{CHANNELS[c]:>5}  {keep['obs'][r][c]:>10.4g}  "
              f"{keep['err'][r][c]:>10.4g}  "
              f"{keep['err'][r][c]/max(keep['obs'][r][c],1e-12):>9.2e}  "
              f"{keep['Dx'][r][c]:>10.4g}  "
              f"[{keep['lb'][r][c]:>9.3g},{keep['ub'][r][c]:>9.3g}]  "
              f"{keep['per_ch'][r][c]:>12.4g}")

    print("\nerr/obs across the split (a small ratio is a tight band, "
          "hence a large 1/sigma^2):")
    print(f"  {'channel':>8}  {'mean':>10}  {'min':>10}")
    for c, ch in enumerate(CHANNELS):
        print(f"  {ch:>8}  {stats['err_ratio_mean'][c]:>10.4g}  "
              f"{stats['err_ratio_min'][c]:>10.3e}")
    print("  pixels whose *smallest* err/obs falls below:")
    for t, c in stats["tiny"].items():
        print(f"    {float(t):>8.0e}: {c:>10,}  ({100*c/max(stats['n'],1):.4f}%)")

    print("\nPixels whose tolerance band straddles zero (lb < 0 in some channel):")
    print(f"  {'group':>12}  {'n':>10}  {'frac':>7}  {'mean':>12}  "
          f"{'of total':>9}  {'median':>8}  {'p99':>9}  {'max':>12}")
    for label, key in (("lb < 0", "neg_band"), ("lb > 0", "pos_band")):
        g = stats[key]
        print(f"  {label:>12}  {g['n']:>10,}  {100*g['frac']:>6.2f}%  "
              f"{g['mean']:>12.4g}  {100*g['share_of_total']:>8.2f}%  "
              f"{g['pct'][0]:>8.4f}  {g['pct'][2]:>9.3f}  {g['pct'][4]:>12.4g}")
    print("  lb = obs - 1.4*err < 0 means the observation is below its own noise: "
          "the\n  constraint degenerates to 'stay under ub' and carries no lower "
          "information,\n  while still able to generate unbounded loss upward. "
          "These are the\n  deconvolution positivity-clamp pixels MIN_OBS=1e-3 was "
          "meant to exclude.")

    print("\nCounterfactual -- sigma floored at frac * obs:")
    print(f"  {'floor':>7}  {'mean':>12}  {'median':>9}  {'p90':>8}  {'p99':>9}  "
          f"{'p99.9':>10}  {'max':>12}")
    for f in FLOORS:
        m = stats["floor_mean"][str(f)]
        p = stats["floor_pct"][str(f)]
        print(f"  {f:>7.3f}  {m:>12.4g}  {p[0]:>9.4f}  {p[1]:>8.3f}  "
              f"{p[2]:>9.3f}  {p[3]:>10.3f}  {p[4]:>12.4g}")
    print("  A floor that collapses the mean toward the median while leaving the "
          "percentiles\n  almost unchanged means the tail was normalisation, not "
          "a fitting failure.")


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    out = {}
    for variant in args.variants:
        path = find_ckpt(args.ckpt_dir, variant, "barrier", suffix=args.ckpt_suffix)
        model, ckpt = load_scaled_model(path, n_basis, device)
        print(f"\nloaded {describe_ckpt(ckpt)}")
        a = Namespace(**ckpt["args"])

        _, loader = make_loader(args.bp_root, args.phase,
                                batch_blocks=args.batch_blocks,
                                num_workers=args.num_workers, shuffle=False,
                                with_labels=True,
                                pixels_per_block=args.pixels_per_block,
                                max_blocks=args.max_blocks)
        keep, stats = scan(model, loader, D_t, device,
                           getattr(a, "alpha_l1", 1.0), getattr(a, "mu", 1.0),
                           args.topn)
        report(f"{variant} / barrier -- {args.phase} split", keep, stats, args)
        out[variant] = {"stats": stats,
                        "worst": {k: v.tolist() for k, v in keep.items()}}

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "loss_outliers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--ckpt_dir", default="output/experiments")
    p.add_argument("--out_dir", default="output/experiments/eval_scaled")
    p.add_argument("--ckpt_suffix", default="",
                   help="select a sweep width, e.g. _h160")
    p.add_argument("--phase", default="test")
    p.add_argument("--variants", nargs="+", default=["mlp6", "cnn"])
    p.add_argument("--batch_blocks", type=int, default=16)
    p.add_argument("--pixels_per_block", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--topn", type=int, default=200)
    p.add_argument("--show", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    main()
