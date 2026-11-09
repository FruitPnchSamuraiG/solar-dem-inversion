#!/usr/bin/env python3
"""Can a neighbor check find the bad readings before the network sees them?

The self-check flags pixels whose DEM predicts far more light than AIA observed
(above 1e3 DN/s and 100x the reading). Most are lone spikes: one channel bright and
the others at the deconvolution floor. Plasma that is bright in one channel is
usually bright in the neighboring pixels and in other channels too, so a reading far
above its eight neighbors in one channel alone points to a detector event (a cosmic
ray or a hot pixel).

For every pixel of a staged split (solved or not) this compares each channel with
the median of its eight neighbors on the full 4096 x 4096 AIA grid, runs the model
and its self-check, and counts, for a grid of rules, how many self-check pixels a
rule catches and how many quiet, bright and flare-core pixels it catches by mistake.
A rule marks channel c of a pixel when its reading is at least K times the neighbor
median (floored at 0.1 DN/s) and at least Z noise sigmas above it; "lone" rules also
require that no other channel is above twice its own neighbor median. Pixels on a
block's first row or column (neighbors outside the stored block) are left out.

    python3 experiments/spike_check.py --root $ROOT --phase val --model <ckpt> --output out.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.full_validation import BRIGHT, label_free  # noqa: E402
from experiments.train_scaled import load_operators  # noqa: E402
from src.zarr_data import MIN_OBS  # noqa: E402

CHANNELS = ("94", "131", "171", "193", "211", "335")
KS = (3, 5, 10, 20, 50)
ZS = (3, 5, 10)
FLOOR = 0.1                              # DN/s, floor on the neighbor median
GROUPS = ("all", "flagged", "quiet", "bright", "flare_core", "unsolved", "relaxed", "clamped")
N_EXAMPLES = 40
CUT = 4                                  # cutout half-width on the 4096 grid


class Blocks(torch.utils.data.Dataset):
    """One staged block at full AIA resolution with its neighbor medians, sampled on the DEM grid."""

    def __init__(self, root, phase, max_blocks):
        self.root, self.phase = root, phase
        self.n = zarr.open(f"{root}/{phase}_x.zarr", mode="r").shape[-1]
        if max_blocks:
            self.n = min(self.n, max_blocks)
        self.z = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.z is None:
            self.z = {s: zarr.open(f"{self.root}/{self.phase}_{s}.zarr", mode="r") for s in "xem"}
        x = np.asarray(self.z["x"][:, :, :, i], dtype=np.float32)            # [6, 256, 256]
        e = np.asarray(self.z["e"][:, ::2, ::2, i], dtype=np.float32)
        tol = np.asarray(self.z["m"][:, :, i])
        pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        H, W = x.shape[1:]
        shifts = [pad[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                  for dy, dx in itertools.product((-1, 0, 1), repeat=2) if (dy, dx) != (0, 0)]
        med = np.median(np.stack(shifts), axis=0)[:, ::2, ::2]
        return x, x[:, ::2, ::2].copy(), e, med, tol, i


def rule_hits(ratio, excess, k, z, lone):
    """Per pixel: does the rule mark any channel? ratio, excess: [N, 6]."""
    spike = (ratio >= k) & (excess >= z)
    hit = spike.any(1)
    if lone:
        elevated = (ratio >= 2).sum(1)
        hit &= (spike.sum(1) == 1) & (elevated == 1)
    return hit


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--phase", default="val")
    p.add_argument("--model", required=True)
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_t, B_t, n_basis, _ = load_operators(device)
    predict = label_free(args.model, n_basis, device)
    thr = np.asarray(BRIGHT, np.float32)
    rules = [(k, z, lone) for k in KS for z in ZS for lone in (True, False)]
    counts = {g: np.zeros(len(rules), np.int64) for g in GROUPS}
    totals = {g: 0 for g in GROUPS}
    flagged_rows, examples = [], {"flagged": [], "flare_core_hit": [], "bright_hit": []}
    ref_rule = rules.index((10, 5, True))
    rng = np.random.default_rng(args.seed)

    data = Blocks(args.root, args.phase, args.max_blocks)
    loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=args.workers, prefetch_factor=2)
    t0 = time.time()
    print(f"{args.phase}: {len(data):,} blocks, device {device}", flush=True)
    for n_done, (x_full, obs, err, med, tol, block) in enumerate(loader):
        obs, err, med, tol = (np.asarray(a) for a in (obs, err, med, tol))
        o = obs.reshape(6, -1).T
        s = err.reshape(6, -1).T
        m = med.reshape(6, -1).T
        t = tol.reshape(-1)
        edge = np.zeros((128, 128), bool)
        edge[0, :] = edge[:, 0] = True
        ok = np.isfinite(o).all(1) & np.isfinite(s).all(1) & ~edge.reshape(-1)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        o, s, m, t = o[idx], s[idx], m[idx], t[idx]
        w = predict(torch.as_tensor(o, device=device)).double()
        rec = (w @ D_t.double().T).cpu().numpy()
        flagged = ((rec > 1e3) & (rec > 100 * np.maximum(o, 1))).any(1)
        ratio = o / np.maximum(m, FLOOR)
        excess = (o - m) / np.maximum(s, 1e-6)
        inten = (o / thr).max(1)
        groups = {"all": np.ones(len(o), bool), "flagged": flagged, "quiet": inten < 1,
                  "bright": (inten >= 1) & (inten < 32), "flare_core": inten >= 32,
                  "unsolved": t == 0, "relaxed": t > 1, "clamped": (o <= MIN_OBS).any(1)}
        hits = np.stack([rule_hits(ratio, excess, k, z, lone) for k, z, lone in rules], 1)   # [N, rules]
        for g, mask in groups.items():
            totals[g] += int(mask.sum())
            counts[g] += hits[mask].sum(0)
        b = int(block)
        rows, cols = np.divmod(idx, 128)
        for j in np.flatnonzero(flagged):
            flagged_rows.append({"block": b, "i": int(rows[j]), "j": int(cols[j]), "tol": int(t[j]),
                                 "obs": o[j].round(4).tolist(), "neighbor_median": m[j].round(4).tolist(),
                                 "sigma": s[j].round(4).tolist(), "max_recon": float(rec[j].max()),
                                 "ref_rule_hit": bool(hits[j, ref_rule])})
        xf = np.asarray(x_full)
        for key, sel in (("flagged", flagged), ("flare_core_hit", groups["flare_core"] & hits[:, ref_rule]),
                         ("bright_hit", groups["bright"] & hits[:, ref_rule])):
            for j in np.flatnonzero(sel):
                if len(examples[key]) < N_EXAMPLES or rng.random() < 0.02:
                    r4, c4 = 2 * rows[j], 2 * cols[j]
                    pad = np.pad(xf, ((0, 0), (CUT, CUT), (CUT, CUT)), mode="edge")
                    cut = pad[:, r4:r4 + 2 * CUT + 1, c4:c4 + 2 * CUT + 1]
                    item = {"block": b, "i": int(rows[j]), "j": int(cols[j]), "cut": cut.astype(np.float32)}
                    if len(examples[key]) < N_EXAMPLES:
                        examples[key].append(item)
                    else:
                        examples[key][rng.integers(N_EXAMPLES)] = item
        if n_done % 500 == 0:
            print(f"  {n_done:,}/{len(data):,} blocks, {totals['all']:,} px, flagged {totals['flagged']}, "
                  f"{time.time() - t0:.0f} s", flush=True)

    table = []
    for r, (k, z, lone) in enumerate(rules):
        row = {"K": k, "Z": z, "lone": lone,
               "catches_flagged_pct": 100 * counts["flagged"][r] / max(totals["flagged"], 1)}
        for g in GROUPS:
            row[f"hits_{g}"] = int(counts[g][r])
            row[f"rate_{g}_per_million"] = 1e6 * counts[g][r] / max(totals[g], 1)
        table.append(row)
    out = {"settings": vars(args), "floor_dn_s": FLOOR, "totals": totals, "rules": table,
           "flagged_pixels": flagged_rows, "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.splitext(args.output)[0] + "_examples.npz",
                        **{f"{key}_{n}": item["cut"] for key, items in examples.items()
                           for n, item in enumerate(items)},
                        meta=json.dumps({key: [{k: v for k, v in it.items() if k != "cut"} for it in items]
                                         for key, items in examples.items()}))
    print(f"done: {totals['all']:,} px, {totals['flagged']} flagged, {time.time() - t0:.0f} s")
    best = sorted(table, key=lambda r: (-r["catches_flagged_pct"], r["hits_all"]))[:8]
    for r in best:
        print(f"K={r['K']:>3} Z={r['Z']:>2} lone={r['lone']!s:5}  catches {r['catches_flagged_pct']:5.1f}% of flagged; "
              f"hits {r['hits_all']:,} px ({r['rate_all_per_million']:.1f}/M), flare cores {r['hits_flare_core']}, "
              f"bright {r['hits_bright']}")


if __name__ == "__main__":
    main()
