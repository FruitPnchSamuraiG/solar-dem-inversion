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
   pixel (2i,2j) exactly -- no averaging, and 3 of every 4 stored AIA pixels
   were never seen by the solver. So the first thing this dataset does is apply
   the same subsampling: after that AIA and DEM share one grid and patches are
   plain contiguous slices, exactly as in the validated 128x128-crop runs.
   (Equivalently, a stride-2 patch on the 4096 grid -- same nine pixels, but
   the subsample-first form has far fewer indices to get wrong.)

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

# Minimum per-channel observation for a pixel to be trainable.
#
# ~4.4% of pixels carry a hard zero in 171A or 193A -- the two *brightest*
# channels (medians 131 and 145 DN). Faintness cannot produce that; it is the
# positivity clamp applied after Hofmeister deconvolution, which can drive
# deconvolved values negative and then floors them at zero. BP still returns a
# feasible DEM for such pixels (mostly at tolLevel 1), so the mask alone does
# not catch them -- the label is simply fitted to a detector artifact.
#
# 1e-3 sits three orders of magnitude below the faintest channel's median
# (94A, 0.76) so it cannot clip real signal, and removes ~5% of pixels.
MIN_OBS = 1e-3

# Band half-width for a channel that carries no constraint (clamp_mode="mask"):
# relu(lb - Dx) and relu(Dx - ub) vanish and ((ub - lb) / 2)**2 stays finite in
# float32, so both losses drop the channel without a separate weight argument.
UNCONSTRAINED = 1e15


class ZarrPatchBlockDataset(Dataset):
    """One item = one staged block, subsampled to `pixels_per_block` pixels.

    Returns (patch [P,6,K,K], obs [P,6], lb [P,6], ub [P,6]); with
    `relax_band=True` also half [P,6] (the loss's normalising half-width), and
    with `with_labels=True` then (dem [P,n_bins], tol [P]).
    """

    def __init__(self, root, phase, patch_size=9, stride=2, tolfac=1.4,
                 pixels_per_block=512, n_bins=N_AIA_BINS, with_labels=False,
                 tol_levels=(1, 3, 5), min_obs=MIN_OBS, max_blocks=None, seed=0,
                 bright_gamma=0.0, thresholds=None, bright_cap=1000.0,
                 clamp_mode="exclude", relax_band=False):
        import zarr

        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.stride = stride
        self.tolfac = tolfac
        # Channels at or below min_obs are the deconvolution clamp (an artefact
        # in that channel, ~40% of all pixels have one, nearly all faint).
        # "exclude" drops the whole pixel (every run before 2026-10-29); "mask"
        # keeps the pixel and frees only the clamped channels from the band, so
        # the network sees such inputs in training instead of extrapolating;
        # "keep" keeps the pixel and the clamped channel's own band around zero
        # (half-width 1.4 sigma at the error floor), as the solver does: a zeroed
        # channel then admits at most that much light.
        if clamp_mode not in ("exclude", "mask", "keep"):
            raise ValueError(f"clamp_mode must be exclude, mask or keep, got {clamp_mode!r}")
        self.clamp_mode = clamp_mode
        # The solver widens the band to 3x and then 5x where no DEM fits 1.4
        # sigma (fullBP.py:515) and records the level in tolLevel; relax_band
        # applies the same band here instead of keeping 1.4 sigma everywhere.
        self.relax_band = relax_band
        self.pixels_per_block = pixels_per_block
        self.n_bins = n_bins
        self.with_labels = with_labels
        self.tol_levels = tuple(tol_levels)
        self.min_obs = min_obs
        self.seed = seed
        # Brightness-weighted pixel draw (0 = uniform, the default). Flare cores
        # (>= 32x the Bright cutoff) are ~1 in 10,000 pixels, so a uniform draw
        # shows the network almost none of them. With gamma > 0 a pixel is drawn
        # with probability proportional to clip(I, 1, cap)**gamma, where I is the
        # max over channels of observed / Bright threshold: Quiet pixels keep
        # weight 1 and a 100x flare core gets 100x the chance (gamma = 1).
        self.bright_gamma = float(bright_gamma)
        self.bright_cap = float(bright_cap)
        self.thresholds = None
        if self.bright_gamma > 0:
            if thresholds is None:
                raise ValueError("bright_gamma > 0 needs the six Bright thresholds")
            self.thresholds = np.asarray(thresholds, dtype=np.float32).reshape(6, 1)

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

        # Padding on the (subsampled) shared grid so edge pixels get a full patch.
        self.pad = patch_size // 2

        print(f"[{phase}] {self.n_blocks:,} blocks x {pixels_per_block} px "
              f"= {self.n_blocks * pixels_per_block:,} samples/epoch "
              f"(AIA {self.aia_block}, DEM {self.dem_block}, patch {patch_size}@stride{stride})")

    def __len__(self):
        return self.n_blocks

    def _valid_mask(self, obs_c, err_c, tol):
        """Pixels usable for training: solver-solved, finite, and (clamp_mode
        "exclude") above the deconvolution-clamp floor in every channel."""
        ok = np.isfinite(obs_c) & np.isfinite(err_c)
        if self.clamp_mode == "exclude":
            ok &= obs_c > self.min_obs
        if self.clamp_mode == "keep":
            # registration padding at the image border carries reading 0 and error 0: a
            # band of zero width (the other modes drop or free those channels already)
            ok &= err_c > 0
        solved = np.isin(tol, self.tol_levels)
        return np.all(ok, axis=0) & solved

    def __getitem__(self, idx):
        # An index past n_blocks is a later draw of the same block (EpochResampler):
        # draw 0 keeps the original per-block seed, so default loaders are unchanged.
        draw, idx = divmod(idx, self.n_blocks)
        if draw == 0:
            rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 32))
        else:
            rng = np.random.default_rng((self.seed, draw, idx))

        obs = np.asarray(self.X[:, :, :, idx], dtype=np.float32)   # [6, A, A]
        err = np.asarray(self.E[:, :, :, idx], dtype=np.float32)   # [6, A, A]
        tol = np.asarray(self.M[:, :, idx])                        # [D, D]

        # Same subsampling fullBP applied, so obs_c[:, i, j] is precisely the
        # observation the solver used to produce the DEM at (i, j).
        s = self.stride
        obs_c = obs[:, ::s, ::s]
        err_c = err[:, ::s, ::s]

        valid = self._valid_mask(obs_c, err_c, tol)
        rows, cols = np.nonzero(valid)

        P = self.pixels_per_block
        if len(rows) == 0:
            # Fully masked block (possible far off-limb). Emit zeros with tol=0 so
            # the caller's mask drops them rather than the batch shape changing.
            patch = torch.zeros(P, 6, self.patch_size, self.patch_size)
            zc = torch.zeros(P, 6)
            out = (patch, zc, zc, zc) + ((zc,) if self.relax_band else ())
            if self.with_labels:
                out = out + (torch.zeros(P, self.n_bins), torch.zeros(P, dtype=torch.uint8))
            return out

        if self.bright_gamma > 0:
            intensity = (obs_c[:, rows, cols] / self.thresholds).max(axis=0)
            w = np.clip(intensity, 1.0, self.bright_cap).astype(np.float64) ** self.bright_gamma
            pick = rng.choice(len(rows), size=P, replace=True, p=w / w.sum())
        else:
            pick = rng.choice(len(rows), size=P, replace=len(rows) < P)
        i = rows[pick].astype(np.int64)
        j = cols[pick].astype(np.int64)

        # Contiguous patch on the shared grid. DEM (i,j) sits at padded (i+pad, j+pad),
        # so offsets 0..K-1 from i span i-pad .. i+pad -- always in bounds.
        obs_pad = np.pad(obs_c, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                         mode='edge')
        off = np.arange(self.patch_size, dtype=np.int64)           # [K]
        rr = i[:, None] + off                                      # [P, K]
        cc = j[:, None] + off                                      # [P, K]
        patch = obs_pad[:, rr[:, :, None], cc[:, None, :]]         # [6, P, K, K]
        patch = np.ascontiguousarray(patch.transpose(1, 0, 2, 3))  # [P, 6, K, K]

        center_obs = np.ascontiguousarray(obs_c[:, i, j].T)        # [P, 6]
        center_err = np.ascontiguousarray(err_c[:, i, j].T)        # [P, 6]
        half = self.tolfac * center_err
        wide = half
        if self.relax_band:
            wide = half * np.maximum(tol[i, j], 1).astype(np.float32)[:, None]
        lb = center_obs - wide
        ub = center_obs + wide
        if self.clamp_mode == "mask":
            clamped = center_obs <= self.min_obs
            lb[clamped], ub[clamped] = -UNCONSTRAINED, UNCONSTRAINED
            half = np.where(clamped, UNCONSTRAINED, half).astype(np.float32)

        out = (torch.from_numpy(patch), torch.from_numpy(center_obs),
               torch.from_numpy(lb), torch.from_numpy(ub))
        if self.relax_band:
            # The losses measure the excess in units of the unrelaxed 1.4 sigma;
            # dividing by the widened half-width instead would soften the penalty
            # 9x or 25x at relaxed pixels (experiments/relaxed_band_screen.py).
            out = out + (torch.from_numpy(np.ascontiguousarray(half)),)

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


class EpochResampler(torch.utils.data.Sampler):
    """Shuffle blocks, and offset epoch e's indices by e * n_blocks so every block
    yields fresh pixels each epoch.

    Without this, each block's pixel draw is seeded by the block alone, so every
    epoch sees the same 512 pixels per block. A set_epoch() on the dataset would
    not reach persistent workers; the index does, because the sampler runs in the
    main process.
    """

    def __init__(self, n_blocks, seed=0):
        self.n_blocks, self.seed, self.epoch = n_blocks, seed, 0

    def __len__(self):
        return self.n_blocks

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(self.n_blocks, generator=g) + self.epoch * self.n_blocks
        self.epoch += 1
        return iter(order.tolist())


def make_loader(root, phase, batch_blocks=8, num_workers=4, shuffle=True,
                resample=False, **kwargs):
    """resample=True draws fresh pixels from every block each epoch (implies shuffle)."""
    from torch.utils.data import DataLoader

    ds = ZarrPatchBlockDataset(root, phase, **kwargs)
    sampler = EpochResampler(len(ds), kwargs.get('seed', 0)) if resample else None
    loader = DataLoader(ds, batch_size=batch_blocks, shuffle=shuffle and sampler is None,
                        sampler=sampler, num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0,
                        drop_last=shuffle or resample)
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


class CachedPixelLoader:
    """Training batches from a per-pixel cache (experiments/build_pixel_cache.py).

    Draws as ZarrPatchBlockDataset + EpochResampler do with resample=True: every epoch the
    blocks in a fresh random order, pixels_per_block fresh pixels from each (with replacement
    here; the block loader draws without replacement when a block has enough pixels, which
    differs in about 2% duplicate draws), batch_blocks blocks per batch, the last partial
    batch dropped. The bands follow ZarrPatchBlockDataset.__getitem__ for the cache's
    clamp_mode. Only the centre pixel is real: the patch is zero except at its centre, which
    is all the centre-pixel MLP reads. The whole cache sits in RAM, so a batch is one gather.
    Yields flat, device-resident tuples (patch, obs, lb, ub[, half]) as flatten_blocks does.
    """

    def __init__(self, cache_dir, batch_blocks, pixels_per_block, tolfac=1.4, clamp_mode="keep",
                 relax_band=False, min_obs=MIN_OBS, patch_size=9, seed=0, device="cpu", max_blocks=None):
        import json
        with open(os.path.join(cache_dir, "meta.json")) as fh:
            meta = json.load(fh)
        if meta["clamp_mode"] != clamp_mode:
            raise ValueError(f"cache built for clamp_mode {meta['clamp_mode']}, not {clamp_mode}")
        self.obs = torch.from_numpy(np.load(os.path.join(cache_dir, "obs.npy")))
        self.err = torch.from_numpy(np.load(os.path.join(cache_dir, "err.npy")))
        self.tol = torch.from_numpy(np.load(os.path.join(cache_dir, "tol.npy")))
        offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        n_blocks = len(offsets) - 1 if max_blocks is None else min(len(offsets) - 1, max_blocks)
        counts = offsets[1:n_blocks + 1] - offsets[:n_blocks]
        self.n_blocks = n_blocks
        self.blocks = torch.from_numpy(np.flatnonzero(counts > 0))
        self.offsets = torch.from_numpy(offsets[:n_blocks]).long()
        self.counts = torch.from_numpy(counts).long()
        self.bb, self.ppb = batch_blocks, pixels_per_block
        self.tolfac, self.clamp_mode, self.relax_band = tolfac, clamp_mode, relax_band
        self.min_obs, self.k = min_obs, patch_size
        self.seed, self.epoch, self.device = seed, 0, torch.device(device)
        print(f"[cache] {len(self.obs):,} pixels in {n_blocks:,} blocks ({int((counts == 0).sum())} empty) from "
              f"{cache_dir}; {len(self)} batches of {batch_blocks} x {pixels_per_block}")

    def __len__(self):
        return len(self.blocks) // self.bb         # as DataLoader(drop_last=True) over the blocks

    def batch(self, obs, err, tol):
        """Bands for a batch of pixels, as ZarrPatchBlockDataset.__getitem__ builds them."""
        half = self.tolfac * err
        wide = half
        if self.relax_band:
            wide = half * torch.clamp(tol.float(), min=1.0)[:, None]
        lb, ub = obs - wide, obs + wide
        if self.clamp_mode == "mask":
            clamped = obs <= self.min_obs
            lb = torch.where(clamped, torch.full_like(lb, -UNCONSTRAINED), lb)
            ub = torch.where(clamped, torch.full_like(ub, UNCONSTRAINED), ub)
            half = torch.where(clamped, torch.full_like(half, UNCONSTRAINED), half)
        patch = torch.zeros(len(obs), 6, self.k, self.k, device=obs.device)
        patch[:, :, self.k // 2, self.k // 2] = obs
        out = (patch, obs, lb, ub)
        return out + ((half,) if self.relax_band else ())

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        self.epoch += 1
        order = self.blocks[torch.randperm(len(self.blocks), generator=g)]
        n_batches = len(self)
        for k in range(n_batches):
            bl = order[k * self.bb:(k + 1) * self.bb]
            if len(bl) == 0:
                break
            start = self.offsets[bl].repeat_interleave(self.ppb)
            cnt = self.counts[bl].repeat_interleave(self.ppb)
            idx = start + (torch.rand(len(start), generator=g) * cnt).long().clamp(max=cnt - 1)
            obs = self.obs[idx].to(self.device, non_blocking=True)
            err = self.err[idx].to(self.device, non_blocking=True)
            tol = self.tol[idx].to(self.device, non_blocking=True)
            yield self.batch(obs, err, tol)
