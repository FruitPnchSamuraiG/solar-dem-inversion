"""
Multimodality vs model size -- graded by peak quality, not by a binary test.

`eval_sweep.py` asks "did the network also find >=2 interior peaks", which is a
binary shape test and a forgiving one: a broad smooth curve with two local maxima
passes it while matching BP's two sharp spikes poorly. Every bimodal pixel counts
the same whether its second peak is a barely-resolved shoulder or a fully
separated component half the height of the first.

The suspicion this script exists to test is sharper, and it is the one the final
figures suggest by eye: the networks reproduce the WEAK, smooth, barely-bimodal
curves and miss the STRONG, well-separated, high-second-peak ones. If true, the
aggregate 29.85% recall is optimistic in a specific and reportable way -- it is
carried by the easy end of the distribution, and performance runs BACKWARD in
peak quality. If false, recall is uniform in quality and the deficit is simply a
threshold effect.

So every BP-multimodal pixel is described by three quantities before anything is
scored against it:

  * n_modes      2 vs 3+, kept apart. Reproducing more than one mode is the goal;
                 reproducing three is strictly harder and gets its own line.
  * separation   bins between the outermost interior peaks. Adjacent peaks are
                 nearly a single broad hump; well-separated ones are two genuinely
                 distinct thermal components.
  * ratio        weaker interior peak height / stronger. Near 1 means two equal
                 components -- unmistakable and the hardest to excuse missing;
                 near 0 means a shoulder on a dominant peak.

and every metric is then reported per quality tier AND per parameter count, so
the deficit can be read as a surface rather than a single number. x-axis is
always parameters, spanning the full 10k -> 11.4M range once job_sweep_big lands.

Note that `mae_dem` here carries the caveat established by bp_self_consistency.py
on 2026-08-03: at bimodal pixels BP's own re-solve scatter is 0.60 against our
0.59 deviation, so a raw 3x penalty vs unimodal pixels is close to the label's own
noise floor. Grading by quality is what can still separate a real capacity limit
from that floor -- if the deficit were purely label noise it should not track peak
separation or height ratio at all.

    python3 experiments/eval_multimodal.py \
        --bp_root  $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --ckpt_dir output/experiments/sweep \
        --n_blocks 60
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
from experiments.eval_scaled import find_ckpt
from experiments.bimodal_scaled import count_peaks_batch
from experiments.eval_sweep import _network_peaks_and_pred

# Downward sweep (15185224) then upward (job_sweep_big). One axis, one curve.
WIDTHS = [48, 72, 108, 160, 232, 336, 480, 680, 960, 1360, 1920]


def peak_descriptors(curve, prominence):
    """(n_interior, separation, ratio) for one curve, or None if not multimodal.

    Interior only: the first and last logT bins sit at the ends of AIA's
    temperature response, and the zero-padding in count_peaks makes a curve that
    merely rises into one register a peak there. 11.2% of raw bimodal detections
    on the test split were exactly that.
    """
    T = len(curve)
    _, where = count_peaks(curve, prominence)
    mid = where[(where >= 1) & (where <= T - 2)]
    if len(mid) < 2:
        return None
    h = np.asarray(curve)[mid]
    return len(mid), int(mid.max() - mid.min()), float(h.min() / max(h.max(), 1e-12))


@torch.no_grad()
def scan(reader, block_ids, models, B_t, device, args, logT):
    """One pass over shared pixels; per-pixel records for BP-multimodal ones.

    Descriptors depend only on BP's curve, so they are computed once per pixel
    and shared across all checkpoints -- the cost is scipy's per-curve find_peaks
    on ~15% of pixels, not on all of them.
    """
    hot = np.asarray(logT[:N_AIA_BINS]) >= args.hot_logT
    keys = list(models)

    uni = {k: [0.0, 0] for k in keys}          # mae sum, count -- the baseline
    conf = {k: np.zeros(4, dtype=np.int64) for k in keys}   # tp, fp, fn, tn
    rec = {"n_modes": [], "sep": [], "ratio": [],
           **{f"nn_modes__{k[0]}_{k[1]}": [] for k in keys},
           **{f"mae__{k[0]}_{k[1]}": [] for k in keys},
           **{f"mae_hot__{k[0]}_{k[1]}": [] for k in keys}}
    n_tot = 0

    for n_done, b in enumerate(block_ids, 1):
        obs, err, tol, dem = reader.read_block(b)
        valid = reader.valid_mask(obs, err, tol)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            continue
        curves = np.ascontiguousarray(dem[:, rows, cols].T)
        _, ref_in = count_peaks_batch(curves, args.prominence, interior=True)
        ref_multi = ref_in >= 2
        n_tot += len(rows)

        peaks, preds = _network_peaks_and_pred(reader, obs, err, tol, rows, cols,
                                               models, B_t, device, args.prominence)
        for k in keys:
            nn_multi = peaks[k] >= 2
            conf[k] += np.array([
                int((nn_multi & ref_multi).sum()), int((nn_multi & ~ref_multi).sum()),
                int((~nn_multi & ref_multi).sum()), int((~nn_multi & ~ref_multi).sum())])
            d = np.abs(preds[k] - curves).mean(axis=1)
            s = ~ref_multi
            uni[k][0] += float(d[s].sum())
            uni[k][1] += int(s.sum())

        for i in np.nonzero(ref_multi)[0]:
            desc = peak_descriptors(curves[i], args.prominence)
            if desc is None:                   # batch/scalar counters disagreed
                continue
            n_modes, sep, ratio = desc
            rec["n_modes"].append(n_modes)
            rec["sep"].append(sep)
            rec["ratio"].append(ratio)
            for k in keys:
                name = f"{k[0]}_{k[1]}"
                rec[f"nn_modes__{name}"].append(int(peaks[k][i]))
                dd = np.abs(preds[k][i] - curves[i])
                rec[f"mae__{name}"].append(float(dd.mean()))
                rec[f"mae_hot__{name}"].append(float(dd[hot].mean()))

        if n_done % 10 == 0:
            print(f"  {n_done}/{len(block_ids)} blocks, "
                  f"{len(rec['n_modes']):,} multimodal pixels")

    rec = {k: np.asarray(v) for k, v in rec.items()}
    return rec, uni, conf, n_tot


def tiers(vals, n=4, ascending=True):
    """Quantile tiers with labels, robust to heavy ties.

    Separation is integer-valued over a short range, so requested quartiles
    frequently collapse; np.unique on the edges keeps the tiering honest rather
    than silently emitting empty bins.
    """
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n + 1)))
    if len(edges) < 3:
        return [(f"all", np.ones(len(vals), dtype=bool))]
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (vals >= lo) & (vals <= hi) if i == len(edges) - 2 else \
              (vals >= lo) & (vals < hi)
        if sel.sum() >= 20:
            out.append((f"{lo:g}-{hi:g}", sel))
    return out if ascending else out[::-1]


def report(rec, uni, conf, n_tot, keys, params, args):
    n_multi = len(rec["n_modes"])
    print(f"\nScanned {n_tot:,} pixels; {n_multi:,} BP interior-multimodal "
          f"({100*n_multi/max(n_tot,1):.2f}%)")
    if n_multi == 0:
        return {}
    print(f"  modes: 2 -> {int((rec['n_modes']==2).sum()):,}   "
          f"3+ -> {int((rec['n_modes']>=3).sum()):,}")
    print(f"  separation bins  p25/p50/p75: "
          f"{np.percentile(rec['sep'],[25,50,75])}")
    print(f"  weak/strong ratio p25/p50/p75: "
          f"{np.round(np.percentile(rec['ratio'],[25,50,75]),3)}")

    sep_t = tiers(rec["sep"], args.n_tiers)
    rat_t = tiers(rec["ratio"], args.n_tiers)
    out = {}

    print("\n" + "=" * 104)
    print("DETECTION AND ERROR vs SIZE, GRADED BY PEAK QUALITY")
    print("=" * 104)
    print(f"\n{'params':>9} {'hidden':>7} | {'prec':>6} {'recall':>7} | "
          f"{'rec 2':>6} {'rec 3+':>7} | "
          + " ".join(f"{'sep '+l:>9}" for l, _ in sep_t) + " | "
          + " ".join(f"{'rat '+l:>10}" for l, _ in rat_t))
    for k in keys:
        name = f"{k[0]}_{k[1]}"
        tp, fp, fn, tn = conf[k]
        nn = rec[f"nn_modes__{name}"]
        hit = nn >= 2
        row = {"params": params[k], "hidden": k[0],
               "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
               "rate": (tp + fp) / max(n_tot, 1),
               "recall_2": float(hit[rec["n_modes"] == 2].mean())
                           if (rec["n_modes"] == 2).any() else float("nan"),
               "recall_3p": float(hit[rec["n_modes"] >= 3].mean())
                            if (rec["n_modes"] >= 3).any() else float("nan"),
               "mae_uni": uni[k][0] / max(uni[k][1], 1),
               "mae_multi": float(rec[f"mae__{name}"].mean()),
               "by_sep": {l: {"recall": float(hit[s].mean()),
                              "mae": float(rec[f"mae__{name}"][s].mean()),
                              "n": int(s.sum())} for l, s in sep_t},
               "by_ratio": {l: {"recall": float(hit[s].mean()),
                                "mae": float(rec[f"mae__{name}"][s].mean()),
                                "n": int(s.sum())} for l, s in rat_t}}
        row["penalty"] = row["mae_multi"] / max(row["mae_uni"], 1e-12)
        out[name] = row
        print(f"{params[k]:>9,} {k[0]:>7} | {100*row['precision']:>5.1f}% "
              f"{100*row['recall']:>6.1f}% | {100*row['recall_2']:>5.1f}% "
              f"{100*row['recall_3p']:>6.1f}% | "
              + " ".join(f"{100*row['by_sep'][l]['recall']:>8.1f}%" for l, _ in sep_t)
              + " | "
              + " ".join(f"{100*row['by_ratio'][l]['recall']:>9.1f}%" for l, _ in rat_t))

    print("\n  'sep' tiers are bins between outermost interior peaks; 'rat' is "
          "weaker/stronger peak\n  height. If recall FALLS to the right in either "
          "block, the networks are catching the\n  easy, barely-bimodal end and "
          "missing the unmistakable ones -- which is the claim\n  the aggregate "
          "recall figure hides.")

    print(f"\n{'params':>9} {'hidden':>7} | {'mae uni':>8} {'mae multi':>10} "
          f"{'penalty':>8} | " + " ".join(f"{'sep '+l:>9}" for l, _ in sep_t))
    for k in keys:
        r = out[f"{k[0]}_{k[1]}"]
        print(f"{r['params']:>9,} {k[0]:>7} | {r['mae_uni']:>8.4f} "
              f"{r['mae_multi']:>10.4f} {r['penalty']:>7.2f}x | "
              + " ".join(f"{r['by_sep'][l]['mae']:>9.4f}" for l, _ in sep_t))
    print("\n  penalty is mae_multi/mae_uni. bp_self_consistency (2026-08-03) put "
          "BP's own re-solve\n  scatter at bimodal pixels at 0.60 against our 0.59 "
          "deviation, so a penalty near 3x is\n  roughly the label's noise floor. "
          "A penalty FLAT in params says capacity is not the\n  bottleneck; a "
          "penalty that rises with separation says something real is being missed.")
    return out


def plot(summary, sep_labels, rat_labels, out_dir):
    runs = sorted(summary.values(), key=lambda r: r["params"])
    if len(runs) < 2:
        print("  [skip plots] need >=2 checkpoints")
        return
    p = [r["params"] for r in runs]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Multimodality vs model size, graded by peak quality "
                 "(BP track, test split)", fontsize=13)

    a = ax[0, 0]
    a.plot(p, [100 * r["precision"] for r in runs], "o-", label="precision")
    a.plot(p, [100 * r["recall"] for r in runs], "s-", label="recall")
    a.plot(p, [100 * r["recall_2"] for r in runs], "^--", label="recall (2 modes)")
    a.plot(p, [100 * r["recall_3p"] for r in runs], "v--", label="recall (3+ modes)")
    a.set_title("detection")
    a.set_ylabel("%")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(p, [r["mae_uni"] for r in runs], "o-", label="unimodal")
    a.plot(p, [r["mae_multi"] for r in runs], "s-", label="multimodal")
    a2 = a.twinx()
    a2.plot(p, [r["penalty"] for r in runs], "k:", label="penalty")
    a2.set_ylabel("multi / uni")
    a.set_title("mae_dem")
    a.set_ylabel("DEM units")
    a.legend(fontsize=8)

    for a, key, labels, what in ((ax[1, 0], "by_sep", sep_labels, "peak separation (bins)"),
                                 (ax[1, 1], "by_ratio", rat_labels, "weak/strong peak ratio")):
        for lab in labels:
            y = [100 * r[key][lab]["recall"] for r in runs if lab in r[key]]
            xs = [r["params"] for r in runs if lab in r[key]]
            a.plot(xs, y, "o-", label=lab)
        a.set_title(f"recall by {what}")
        a.set_ylabel("recall %")
        a.legend(fontsize=8, title=what.split(" (")[0])

    for a in ax.ravel():
        a.set_xscale("log")
        a.set_xlabel("parameters")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_vs_size.png")
    fig.savefig(path, dpi=140)
    print(f"Figure -> {path}")


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")

    D_t, B_t, n_basis, logT = load_operators(device)
    models, params = {}, {}
    for w in (args.widths or WIDTHS):
        try:
            path = find_ckpt(args.ckpt_dir, "mlp6", args.loss, suffix=f"_h{w}")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        m, _ = load_scaled_model(path, n_basis, device)
        models[(w, args.loss)] = m
        params[(w, args.loss)] = sum(q.numel() for q in m.parameters())
    if not models:
        raise SystemExit("no checkpoints found")
    keys = sorted(models, key=lambda k: k[0])
    print("Loaded: " + ", ".join(f"h{k[0]}={params[k]:,}" for k in keys))

    reader = BlockReader(args.bp_root, "test", patch_size=args.patch_size)
    rng = np.random.default_rng(args.seed)
    block_ids = rng.choice(len(reader), size=min(args.n_blocks, len(reader)),
                           replace=False)
    rec, uni, conf, n_tot = scan(reader, block_ids, models, B_t, device, args, logT)
    summary = report(rec, uni, conf, n_tot, keys, params, args)

    os.makedirs(args.out_dir, exist_ok=True)
    if summary:
        sep_l = [l for l, _ in tiers(rec["sep"], args.n_tiers)]
        rat_l = [l for l, _ in tiers(rec["ratio"], args.n_tiers)]
        plot(summary, sep_l, rat_l, args.out_dir)
    path = os.path.join(args.out_dir, "eval_multimodal.json")
    with open(path, "w") as f:
        json.dump({"n_pixels": n_tot, "n_multimodal": len(rec["n_modes"]),
                   "loss": args.loss, "runs": summary}, f, indent=2)
    print(f"Summary -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_root", required=True)
    p.add_argument("--ckpt_dir", default="output/experiments/sweep")
    p.add_argument("--out_dir", default="output/experiments/multimodal")
    p.add_argument("--loss", default="barrier", choices=["barrier", "enet"])
    p.add_argument("--widths", type=int, nargs="+", default=None)
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--n_blocks", type=int, default=60,
                   help="more than eval_sweep's 20: tiering splits the "
                        "multimodal pixels four ways and each tier needs to hold")
    p.add_argument("--n_tiers", type=int, default=4)
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--hot_logT", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
