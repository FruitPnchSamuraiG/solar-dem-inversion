"""
The June per-pixel MLP against the patch CNN and BP, on the same held-out pixels.

The original 213k MLP's checkpoint was not kept, so it is retrained first with
its original settings (train_unsupervised.py: 4 timestamps, crop
1800,1800,256,256, hidden 256, raw intensities, barrier loss, 30 epochs). The
patch CNN is the ablation's `ablation_cnn_barrier.pt` (crop 1800,1800,128,128,
which is the top-left quarter of the MLP's crop, so every pixel here was in
both training sets' images). Pixels come from the ablation's 20% held-out split.

Writes a slide figure (three pixels of the X2.1 flare image) and, over all
held-out pixels of the four images, how often each method's DEM oscillates:
the share of curves with 3+ local maxima above 10% of the curve's peak.

    uv run python experiments/plot_first_mlp_vs_patch_cnn.py \
        --data_dirs data/20110906_2217 data/20120603_0000 data/20131113_0908 data/20140910_1731
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fullBP import getBasis, solveLP
from experiments.train_ablations import load_ablation_model
from experiments.train_neural_field import AIAPatchDataset
from experiments.train_neural_field_amortized import split_dataset
from experiments.train_unsupervised import DEMNet

COLORS = {"BP": "#3b3a37", "First MLP (213k)": "#d6452a", "Patch CNN (1.5M)": "#2a78d6"}


def local_maxima(dem, frac=0.1):
    """Interior or edge bins higher than both neighbours and above frac * max."""
    d = np.pad(dem, 1, constant_values=0.0)
    peak = (d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] > frac * dem.max())
    return int(peak.sum())


def bp_dem(obs, ub, D, B, tolfac):
    err = (ub - obs) / tolfac
    for t in (1.4, 2.0, 2.8, 5.0):          # BP's relaxation schedule
        x = solveLP((D, obs, obs - t * err, obs + t * err, None))
        if x is not None:
            return np.maximum(B @ x, 0)
    return None


def main(args):
    torch.set_num_threads(args.threads)
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * 1e26).astype(np.float64)
    B = getBasis(R, logT, alphas=[0.0, 0.1, 0.2])
    D = R @ B
    n_basis = B.shape[1]

    mlp = DEMNet(n_basis=n_basis, hidden=args.hidden)
    mlp.load_state_dict(torch.load(args.mlp, map_location="cpu", weights_only=False)["model"])
    mlp.eval()
    cnn, _ = load_ablation_model(args.cnn, n_basis, torch.device("cpu"))
    cnn.eval()

    def predict(patch, obs):
        with torch.no_grad():
            return {"First MLP (213k)": np.maximum(B @ mlp(obs[None]).double().numpy()[0], 0),
                    "Patch CNN (1.5M)": np.maximum(B @ cnn(patch[None]).double().numpy()[0], 0)}

    stats, examples = {}, None
    rng = np.random.default_rng(args.seed)
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        ds = AIAPatchDataset(data_dir, args.crop, patch_size=9, tolfac=args.tolfac)
        _, val = split_dataset(ds, args.val_frac, args.seed)   # the ablation's held-out pixels
        pick = rng.choice(len(val), size=min(args.n_stats, len(val)), replace=False)
        counts = {k: [] for k in COLORS}
        rows = []
        for i in pick:
            patch, obs, lb, ub = val[int(i)]
            curves = predict(patch, obs)
            ref = bp_dem(obs.double().numpy(), ub.double().numpy(), D, B, args.tolfac)
            if ref is None or ref.max() <= 0:
                continue
            curves["BP"] = ref
            for k, c in curves.items():
                counts[k].append(local_maxima(c))
            rows.append((obs[2].item(), curves))
        stats[tag] = {k: {"n": len(v), "mean_maxima": float(np.mean(v)),
                          "pct_3plus": float(100 * np.mean(np.array(v) >= 3))}
                      for k, v in counts.items()}
        print(tag, json.dumps(stats[tag]))
        if tag == args.example_tag:
            examples = rows

    # slide figure: the first n_examples sampled pixels above the median 171 A brightness
    med = np.median([b for b, _ in examples])
    chosen = [c for b, c in examples if b >= med][:args.n_examples]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(4.4, 2.0 * len(chosen)), sharex=True)
    for ax, curves in zip(np.atleast_1d(axes), chosen):
        for k in ("BP", "First MLP (213k)", "Patch CNN (1.5M)"):
            ax.plot(logT, curves[k], color=COLORS[k], lw=2.2 if k == "BP" else 1.6,
                    ls="-" if k != "Patch CNN (1.5M)" else "--", label=k)
        ax.set_ylabel("DEM", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("log T", fontsize=8)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "first_mlp_vs_patch_cnn.png"), dpi=200)
    with open(os.path.join(args.out_dir, "first_mlp_vs_patch_cnn.json"), "w") as f:
        json.dump({"stats": stats, "settings": vars(args)}, f, indent=1)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--crop", default="1800,1800,128,128")
    p.add_argument("--mlp", default="output/experiments/barrier_nn_model.pt")
    p.add_argument("--cnn", default="output/experiments/ablation_cnn_barrier.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_stats", type=int, default=400, help="held-out pixels per image")
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--example_tag", default="20110906_2217")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out_dir", default="output/experiments/first_mlp_repro")
    main(p.parse_args())
