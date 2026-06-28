"""
Compare amortized neural field val results across mask_prob values.
Loads saved checkpoints and runs val evaluation for each, printing a
combined table so you can compare without re-training.

Run from project root:
    uv run python experiments/compare_mask_runs.py \
        --data_dirs ./data/20110906_2217 ./data/20120603_0000 \
                    ./data/20131113_0908 ./data/20140910_1731 \
        --crop 1800,1800,128,128
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from fullBP import getBasis
from experiments.train_neural_field import (
    AIAPatchDataset, PatchDEMNet, pick_device, bp_sparsity_reference,
)
from experiments.train_neural_field_amortized import split_dataset, evaluate_val

ACTIVITY = {
    "20110906_2217": "X2.1 flare",
    "20120603_0000": "Quiet sun",
    "20131113_0908": "Moderate",
    "20140910_1731": "X1.6 flare",
}


def main(args):
    device = pick_device()

    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData["R"], RData["logT"]
    R = (R * scale).astype(np.float32)
    B = getBasis(R.astype(np.float64), logT, alphas=[0.0, 0.1, 0.2]).astype(np.float32)
    D = (R @ B).astype(np.float32)

    # build val subsets (same seed/frac as training so same held-out pixels)
    val_subsets, bp_refs = {}, {}
    for data_dir in args.data_dirs:
        tag = os.path.basename(data_dir.rstrip("/"))
        ds = AIAPatchDataset(data_dir, args.crop, patch_size=args.patch_size, tolfac=args.tolfac)
        _, val_ds = split_dataset(ds, args.val_frac, args.seed)
        val_subsets[tag] = val_ds
        bp_refs[tag] = bp_sparsity_reference(ds, D, n_pixels=args.bp_compare, seed=args.seed)

    # checkpoints to compare
    ckpt_entries = [
        ("no mask", "output/experiments/neural_field_amortized_4ts.pt"),
        ("mask 0.1", "output/experiments/neural_field_amortized_4ts_mask0.1.pt"),
        ("mask 0.3", "output/experiments/neural_field_amortized_4ts_mask0.3.pt"),
        ("mask 0.5", "output/experiments/neural_field_amortized_4ts_mask0.5.pt"),
        ("mask 0.7", "output/experiments/neural_field_amortized_4ts_mask0.7.pt"),
    ]

    tags = list(val_subsets.keys())

    # header
    col = 12
    print(f"\n{'Timestamp':<22} {'Activity':<12} {'BP sp':>6}  " +
          "  ".join(f"{label:>{col}}" for label, _ in ckpt_entries))
    print("-" * (22 + 12 + 8 + (col + 2) * len(ckpt_entries)))

    # per-timestamp rows
    all_results = {}
    for label, ckpt_path in ckpt_entries:
        if not os.path.exists(ckpt_path):
            all_results[label] = None
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = PatchDEMNet(n_basis=D.shape[1], patch_size=ckpt["patch_size"],
                            channels=ckpt["channels"]).to(device)
        model.load_state_dict(ckpt["model"])
        all_results[label] = evaluate_val(model, val_subsets, D, device, seed=args.seed)

    for tag in tags:
        bp_sp = bp_refs[tag]
        row = f"{tag:<22} {ACTIVITY.get(tag,''):<12} {bp_sp:>6.2f}  "
        for label, _ in ckpt_entries:
            res = all_results.get(label)
            if res is None:
                row += f"{'missing':>{col}}  "
            else:
                nn_sp = res[tag]["nn_sparsity"]
                row += f"{nn_sp:>{col}.2f}  "
        print(row)

    print(f"\n(lower sparsity = sparser = closer to BP's L1 behavior)")
    print(f"\nMAE table:")
    print(f"\n{'Timestamp':<22} {'Activity':<12} " +
          "  ".join(f"{label:>{col}}" for label, _ in ckpt_entries))
    print("-" * (22 + 12 + 2 + (col + 2) * len(ckpt_entries)))
    for tag in tags:
        row = f"{tag:<22} {ACTIVITY.get(tag,''):<12}  "
        for label, _ in ckpt_entries:
            res = all_results.get(label)
            if res is None:
                row += f"{'missing':>{col}}  "
            else:
                mae = res[tag]["nn_mae"]
                row += f"{mae:>{col}.4f}  "
        print(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs",  nargs="+", required=True)
    p.add_argument("--crop",       default="1800,1800,128,128")
    p.add_argument("--patch_size", type=int,   default=9)
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--tolfac",     type=float, default=1.4)
    p.add_argument("--bp_compare", type=int,   default=100)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
