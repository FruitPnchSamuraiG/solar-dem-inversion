"""
Why did the scaled runs collapse? Characterise the staged AIA errors.

The four runs of array 15088220 died in epoch 1: mean loss 1.2e10, then a
bit-identical 339.2775 for eleven epochs with sp_coef=0. A frozen loss means
zero gradient, i.e. the Softplus output head underflowed to exactly 0.

barrier_loss_batch divides by sigma2 = ((ub-lb)/2)^2 + 1e-8, so a channel whose
staged error is zero contributes obs^2/1e-8 -- at 171A's median 145 DN that is
2e12 from a single pixel. This script tests whether such pixels exist, since
ZarrPatchBlockDataset._valid_mask only requires err to be *finite*, not positive.

    python3 experiments/diag_errors.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zarr_data import MIN_OBS

WAVE = [94, 131, 171, 193, 211, 335]
TOLFAC = 1.4


def main(root, phase, n_blocks, stride):
    import zarr

    X = zarr.open(os.path.join(root, f'{phase}_x.zarr'), mode='r')
    E = zarr.open(os.path.join(root, f'{phase}_e.zarr'), mode='r')
    M = zarr.open(os.path.join(root, f'{phase}_m.zarr'), mode='r')

    n = min(n_blocks, X.shape[-1])
    obs_l, err_l, tol_l = [], [], []
    for idx in range(n):
        o = np.asarray(X[:, :, :, idx], dtype=np.float32)[:, ::stride, ::stride]
        e = np.asarray(E[:, :, :, idx], dtype=np.float32)[:, ::stride, ::stride]
        t = np.asarray(M[:, :, idx])
        obs_l.append(o.reshape(6, -1))
        err_l.append(e.reshape(6, -1))
        tol_l.append(t.reshape(-1))

    obs = np.concatenate(obs_l, axis=1)          # [6, N]
    err = np.concatenate(err_l, axis=1)          # [6, N]
    tol = np.concatenate(tol_l)                  # [N]
    print(f"{n} blocks, {obs.shape[1]:,} pixels\n")

    # The mask the dataloader actually applies today.
    cur = np.all(np.isfinite(obs) & np.isfinite(err) & (obs > MIN_OBS), axis=0) \
        & np.isin(tol, (1, 3, 5))
    print(f"pixels passing the CURRENT mask: {cur.mean():.4f}\n")

    print("per-channel, over pixels passing the current mask:")
    print(f"{'chan':>6} {'med obs':>10} {'med err':>10} {'err<=0':>9} "
          f"{'err<1e-6':>9} {'med s/o':>9} {'p99 s/o':>10}")
    for c in range(6):
        o, e = obs[c][cur], err[c][cur]
        ratio = e / np.maximum(o, 1e-30)
        print(f"{WAVE[c]:>6} {np.median(o):>10.3f} {np.median(e):>10.3f} "
              f"{(e <= 0).mean():>9.5f} {(e < 1e-6).mean():>9.5f} "
              f"{np.median(ratio):>9.3f} {np.percentile(ratio, 99):>10.3e}")

    # The quantity that actually blew up: relu(lb)^2 / sigma2 at x = 0, which is
    # the very first thing the loss sees from an untrained (near-zero) network.
    o, e = obs[:, cur], err[:, cur]
    lb = o - TOLFAC * e
    sigma2 = (TOLFAC * e) ** 2 + 1e-8
    term = np.maximum(lb, 0) ** 2 / sigma2
    per_pix = term.sum(axis=0)
    print(f"\nbarrier_lb at x=0 (the epoch-1 starting point):")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  p{q:<5} {np.percentile(per_pix, q):>14.4e}")
    print(f"  mean   {per_pix.mean():>14.4e}   <- compare to the logged 1.2e10")

    # How much of the damage comes from a vanishing denominator alone?
    bad = (e <= 0).any(axis=0)
    print(f"\npixels with a non-positive error in ANY channel: {bad.mean():.5f}")
    if bad.any():
        print(f"  their mean barrier_lb: {per_pix[bad].mean():.4e}")
        print(f"  everyone else's:       {per_pix[~bad].mean():.4e}")

    # Candidate fix: require a strictly positive error in every channel.
    keep = ~bad
    print(f"\nwith err>0 required, mean barrier_lb: {per_pix[keep].mean():.4e} "
          f"(keeps {keep.mean():.4f} of masked pixels)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--phase', default='val')
    p.add_argument('--n_blocks', type=int, default=40)
    p.add_argument('--stride', type=int, default=2)
    a = p.parse_args()
    main(a.root, a.phase, a.n_blocks, a.stride)
