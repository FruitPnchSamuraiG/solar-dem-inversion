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


if __name__ == '__main__':
    main()
