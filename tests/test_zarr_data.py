"""
Alignment test for the zarr patch dataset, run against a synthetic zarr.

The failure this guards against is silent: if the patch is centred on the wrong
AIA pixel, the network is shown one bit of Sun and scored against the DEM of
another. Training loss looks perfectly healthy either way, so it has to be
checked directly. Each synthetic AIA pixel is stamped with its own coordinate
(r*1000 + c + 1), which turns any misalignment into a visibly wrong number.

    uv run python tests/test_zarr_data.py
"""

import os
import sys
import tempfile

import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zarr_data import ZarrPatchBlockDataset

A, D, N = 64, 32, 3          # AIA block, DEM block, number of blocks
STRIDE = A // D


def build_synthetic(root):
    rows = np.arange(A)[:, None] * 1000
    cols = np.arange(A)[None, :] + 1
    obs = np.broadcast_to((rows + cols)[None, :, :, None],
                          (6, A, A, N)).astype(np.float32)
    err = np.full((6, A, A, N), 1.0, np.float32)

    tol = np.ones((D, D, N), np.uint8)
    tol[0, 0, :] = 0                                  # one never-solved pixel
    y = np.random.default_rng(0).random((26, D, D, N)).astype(np.float32)
    y[:, 0, 0, :] = np.nan                            # its DEM is NaN, as in real data

    for name, arr in (('x', obs), ('e', err), ('y', y), ('m', tol)):
        z = zarr.open(os.path.join(root, f'train_{name}.zarr'), mode='w',
                      shape=arr.shape, dtype=arr.dtype)
        z[:] = arr


def main():
    with tempfile.TemporaryDirectory() as root:
        build_synthetic(root)
        ds = ZarrPatchBlockDataset(root, 'train', pixels_per_block=200,
                                   with_labels=True, seed=1)
        patch, obs, lb, ub, dem, tol = ds[0]
        K = ds.patch_size

        # 1. the patch centre is the pixel the DEM was actually solved from
        assert torch.equal(patch[:, :, K // 2, K // 2], obs), "patch centre != centre obs"

        # 2. that pixel is stride-aligned, i.e. DEM (i,j) <- AIA (2i,2j)
        v = int(obs[0, 0].item())
        r_aia, c_aia = v // 1000, v % 1000 - 1
        i, j = r_aia // STRIDE, c_aia // STRIDE
        assert (r_aia, c_aia) == (STRIDE * i, STRIDE * j), \
            f"centre AIA pixel ({r_aia},{c_aia}) is not stride-aligned"

        # 3. neighbours step by `stride` on the AIA grid, clamped at the edges
        row = patch[0, 0, K // 2, :].numpy()
        expected = [(STRIDE * i) * 1000 + STRIDE * min(max(j + k - K // 2, 0), D - 1) + 1
                    for k in range(K)]
        assert np.allclose(row, expected), f"\n got {row}\n exp {expected}"

        # 4. the tolLevel mask excludes unsolved pixels, so no NaN label gets through
        assert torch.isfinite(dem).all(), "NaN label survived the tolLevel mask"
        assert 0 not in set(tol.tolist()), "an unsolved pixel was sampled"

        # 5. the feasibility band the unsupervised losses need is non-degenerate
        assert torch.all(ub > lb), "degenerate tolerance band"

        print(f"patch {tuple(patch.shape)}  centre row {row[:5]} ...")
        print("all alignment checks passed")

        check_resampling(root)


def check_resampling(root):
    """The default loader repeats each block's pixels every epoch; resample=True
    draws fresh ones, through persistent workers, and keeps epoch 1 identical."""
    from src.zarr_data import EpochResampler, make_loader

    ds = ZarrPatchBlockDataset(root, 'train', pixels_per_block=40, seed=3)
    same = [ds[b][1][:, 0].numpy() for b in range(N)]
    again = [ds[b][1][:, 0].numpy() for b in range(N)]
    assert all(np.array_equal(a, b) for a, b in zip(same, again)), "default draw not repeatable"

    sampler = EpochResampler(N, seed=3)
    epochs = [list(sampler) for _ in range(3)]
    for e, order in enumerate(epochs):
        assert sorted(i % N for i in order) == list(range(N)), "a block was skipped or repeated"
        assert all(i // N == e for i in order), "epoch offset wrong"

    # draw 0 is the original per-block draw; later draws differ from it
    first = [ds[i][1][:, 0].numpy() for i in range(N)]
    second = [ds[N + i][1][:, 0].numpy() for i in range(N)]
    assert all(np.array_equal(a, b) for a, b in zip(first, same)), "draw 0 changed"
    assert not any(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(first, second)), \
        "draw 1 repeated draw 0"

    # through a real DataLoader with persistent workers, epochs differ
    for workers in (0, 2):
        _, loader = make_loader(root, 'train', batch_blocks=1, num_workers=workers,
                                resample=True, pixels_per_block=40, seed=3)
        e1 = sorted(tuple(np.sort(o[0, :, 0].numpy())) for _, o, _, _ in loader)
        e2 = sorted(tuple(np.sort(o[0, :, 0].numpy())) for _, o, _, _ in loader)
        assert e1 == sorted(tuple(np.sort(a)) for a in same), f"epoch 1 != default draw ({workers} workers)"
        assert set(e1).isdisjoint(e2), f"epoch 2 repeated epoch 1 ({workers} workers)"
        _, fixed = make_loader(root, 'train', batch_blocks=1, num_workers=workers,
                               shuffle=True, pixels_per_block=40, seed=3)
        f1 = sorted(tuple(np.sort(o[0, :, 0].numpy())) for _, o, _, _ in fixed)
        f2 = sorted(tuple(np.sort(o[0, :, 0].numpy())) for _, o, _, _ in fixed)
        assert f1 == f2, "default loader should repeat its pixels"
    print("resampling checks passed: fresh pixels each epoch, epoch 1 unchanged, "
          "default loader unchanged")


if __name__ == '__main__':
    main()
