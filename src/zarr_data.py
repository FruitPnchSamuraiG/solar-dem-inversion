"""
Zarr-backed patch dataset for the scaled (1,223-timestamp) DEM training runs.

Reads the four arrays written by dataset/stage_hofdeconv_full.py:

    {phase}_x.zarr  [6, 256, 256, N]   AIA observations, full 4096 grid
    {phase}_e.zarr  [6, 256, 256, N]   AIA per-pixel uncertainties
    {phase}_y.zarr  [26, 128, 128, N]  solver DEM, native (decimated) grid
    {phase}_m.zarr  [128, 128, N]      tolLevel: 0 unsolved, 1 tight, 3/5 relaxed

Two things about this data drive the design:

1. AIA and DEM are on different grids. fullBP.py:979 decimates by plain
   subsampling (`AIACube[:, ::2, ::2]`), so DEM pixel (i,j) was solved from AIA
   pixel (2i,2j) exactly -- no averaging. To give the patch CNN the same
   physical footprint as the validated 128x128-crop runs, the 9x9 neighborhood
   is sampled at stride 2, i.e. from a 17x17 region of the AIA grid.

2. A chunk is one whole block (chunks=(..., 1)), so reading a single pixel costs
   the same as reading all 16,384 DEM pixels of its block. A dataset item is
   therefore one *block*, from which `pixels_per_block` pixels are sampled; the
   training loop flattens the leading two dimensions. Indexing pixels
   individually would re-decompress the same chunk thousands of times.

Sample tuple matches AIAPatchDataset so the existing training loops are
unchanged: (patch, obs, lb, ub), plus (dem, tol) when labels are requested.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Labels are evaluation-only (training is unsupervised), but keep the same bin
# convention: for AIA-only, fullBP stacks 8 rows of zeros onto the 18 real bins
# (fullBP.py:1013-1019). Scoring on all 26 would credit the model for 8
# guaranteed zeros and break the Hoyer sparsity denominator.
N_AIA_BINS = 18
N_STORED_BINS = 26


class ZarrPatchBlockDataset(Dataset):
    """One item = one staged block, subsampled to `pixels_per_block` pixels.

    Returns (patch [P,6,K,K], obs [P,6], lb [P,6], ub [P,6]); with
    `with_labels=True`, also (dem [P,n_bins], tol [P]).
    """

    def __init__(self, root, phase, patch_size=9, stride=2, tolfac=1.4,
                 pixels_per_block=512, n_bins=N_AIA_BINS, with_labels=False,
                 tol_levels=(1, 3, 5), max_blocks=None, seed=0):
        import zarr

        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.stride = stride
        self.tolfac = tolfac
        self.pixels_per_block = pixels_per_block
        self.n_bins = n_bins
        self.with_labels = with_labels
        self.tol_levels = tuple(tol_levels)
        self.seed = seed

        def _open(suffix):
            path = os.path.join(root, f'{phase}_{suffix}.zarr')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing {path}. Restage with dataset/stage_hofdeconv_full.py "
                    f"(the '_e' errors array was added in d3c58d7).")
            return zarr.open(path, mode='r')

        self.X = _open('x')
        self.E = _open('e')
        self.M = _open('m')
        self.Y = _open('y') if with_labels else None

        self.n_blocks = self.X.shape[-1]
        if max_blocks is not None:
            self.n_blocks = min(self.n_blocks, max_blocks)

        self.aia_block = self.X.shape[1]
        self.dem_block = self.M.shape[0]
        assert self.aia_block == self.dem_block * stride, (
            f"stride {stride} does not relate AIA block {self.aia_block} to "
            f"DEM block {self.dem_block}")

        # Padding on the AIA grid so every DEM pixel has a full stride-2 patch.
        self.pad = (patch_size // 2) * stride

        print(f"[{phase}] {self.n_blocks:,} blocks x {pixels_per_block} px "
              f"= {self.n_blocks * pixels_per_block:,} samples/epoch "
              f"(AIA {self.aia_block}, DEM {self.dem_block}, patch {patch_size}@stride{stride})")

    def __len__(self):
        return self.n_blocks

    def _valid_mask(self, obs_c, err_c, tol):
        """Pixels usable for training: solver-solved and finite positive AIA."""
        finite = np.all(np.isfinite(obs_c) & np.isfinite(err_c) & (obs_c > 0), axis=0)
        solved = np.isin(tol, self.tol_levels)
        return finite & solved

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 32))

        obs = np.asarray(self.X[:, :, :, idx], dtype=np.float32)   # [6, A, A]
        err = np.asarray(self.E[:, :, :, idx], dtype=np.float32)   # [6, A, A]
        tol = np.asarray(self.M[:, :, idx])                        # [D, D]

        s = self.stride
        obs_c = obs[:, ::s, ::s]   # AIA value at the exact pixel each DEM came from
        err_c = err[:, ::s, ::s]

        valid = self._valid_mask(obs_c, err_c, tol)
        rows, cols = np.nonzero(valid)

        P = self.pixels_per_block
        if len(rows) == 0:
            # Fully masked block (possible far off-limb). Emit zeros with tol=0 so
            # the caller's mask drops them rather than the batch shape changing.
            patch = torch.zeros(P, 6, self.patch_size, self.patch_size)
            zc = torch.zeros(P, 6)
            out = (patch, zc, zc, zc)
            if self.with_labels:
                out = out + (torch.zeros(P, self.n_bins), torch.zeros(P, dtype=torch.uint8))
            return out

        pick = rng.choice(len(rows), size=P, replace=len(rows) < P)
        i = rows[pick].astype(np.int64)
        j = cols[pick].astype(np.int64)

        # Patch on the padded AIA grid. DEM (i,j) sits at padded AIA (s*i+pad, s*j+pad);
        # offsets s*(-k..k) then span s*i .. s*i+2*pad inclusive -- always in bounds.
        obs_pad = np.pad(obs, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                         mode='edge')
        off = s * np.arange(self.patch_size, dtype=np.int64)       # [K]
        rr = (s * i)[:, None] + off                                # [P, K]
        cc = (s * j)[:, None] + off                                # [P, K]
        patch = obs_pad[:, rr[:, :, None], cc[:, None, :]]         # [6, P, K, K]
        patch = np.ascontiguousarray(patch.transpose(1, 0, 2, 3))  # [P, 6, K, K]

        center_obs = np.ascontiguousarray(obs_c[:, i, j].T)        # [P, 6]
        center_err = np.ascontiguousarray(err_c[:, i, j].T)        # [P, 6]
        lb = center_obs - self.tolfac * center_err
        ub = center_obs + self.tolfac * center_err

        out = (torch.from_numpy(patch), torch.from_numpy(center_obs),
               torch.from_numpy(lb), torch.from_numpy(ub))

        if self.with_labels:
            dem = np.asarray(self.Y[:self.n_bins, :, :, idx], dtype=np.float32)
            dem = np.ascontiguousarray(dem[:, i, j].T)             # [P, n_bins]
            out = out + (torch.from_numpy(dem),
                         torch.from_numpy(tol[i, j].astype(np.uint8)))
        return out


def flatten_blocks(batch):
    """Collapse the [n_blocks, pixels_per_block, ...] batch into per-pixel rows.

    DataLoader stacks block items, so every tensor arrives with two leading
    dimensions; the models and losses all expect a flat [B, ...] batch.
    """
    return tuple(t.flatten(0, 1) for t in batch)


def make_loader(root, phase, batch_blocks=8, num_workers=4, shuffle=True, **kwargs):
    from torch.utils.data import DataLoader

    ds = ZarrPatchBlockDataset(root, phase, **kwargs)
    loader = DataLoader(ds, batch_size=batch_blocks, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0,
                        drop_last=shuffle)
    return ds, loader


def _self_test(root, phase):
    """Sanity-check shapes, alignment and mask semantics against the staged zarr."""
    ds = ZarrPatchBlockDataset(root, phase, pixels_per_block=64, with_labels=True)
    patch, obs, lb, ub, dem, tol = ds[0]
    print(f"patch {tuple(patch.shape)}  obs {tuple(obs.shape)}  "
          f"dem {tuple(dem.shape)}  tol {tuple(tol.shape)}")

    K = ds.patch_size
    centre = patch[:, :, K // 2, K // 2]
    max_dev = (centre - obs).abs().max().item()
    print(f"patch centre == centre obs: max deviation {max_dev:.3e}")
    assert max_dev == 0, "patch centre is not the pixel the DEM was solved from"

    assert torch.all(ub > lb), "degenerate tolerance band"
    assert torch.isfinite(dem).all(), "NaN label survived the tolLevel mask"
    print(f"tol levels present: {sorted(set(tol.tolist()))}")
    print(f"obs range [{obs.min():.3e}, {obs.max():.3e}]  "
          f"band width/obs median {((ub - lb) / 2 / obs.clamp(min=1e-12)).median():.3f}")
    print("self-test OK")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='e.g. $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS')
    p.add_argument('--phase', default='val')
    _a = p.parse_args()
    _self_test(_a.root, _a.phase)
