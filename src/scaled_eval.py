"""
Shared loading helpers for evaluating the scaled (array 15092854) checkpoints.

Two things the training code does not need but every evaluation does:

  1. Rebuilding a scaled checkpoint. train_scaled.py saves the *bare* variant's
     state_dict, not the NormalizedInput wrapper, plus the fields describing the
     representation those weights expect (`input_transform`, `softplus_floor`).
     Loading without re-applying the wrapper feeds raw DN to weights trained on
     log1p input and produces silent nonsense, so `load_scaled_model` is the only
     supported way in.

  2. Addressing *specific* pixels. ZarrPatchBlockDataset deliberately returns a
     random subsample per block, which is right for training and useless for a
     comparison that must hit the same pixel with four different models. The
     extractors below take explicit (i, j) indices on the shared grid.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.zarr_data import N_AIA_BINS, MIN_OBS
from experiments.train_ablations import build_model


# ── checkpoints ───────────────────────────────────────────────────────────────

def load_scaled_model(ckpt_path, n_basis, device):
    """Rebuild a train_scaled.py checkpoint, wrapper and all.

    Returns (model, ckpt). The model is in eval mode and expects raw physical-DN
    patches: the log1p compression lives inside NormalizedInput, so callers pass
    the same units the loss and metrics use.
    """
    from experiments.train_scaled import (ClampedSoftplus, NormalizedInput,
                                          harden_softplus)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    core = build_model(ckpt["variant"], n_basis, ckpt["patch_size"],
                       ckpt["channels"], perm=ckpt.get("perm"))
    # The saved weights come from a model whose Softplus was already replaced, so
    # rebuild that structure before load_state_dict (it is parameter-free, but
    # skipping it would silently restore the module that died in 15088220).
    harden_softplus(core, ckpt.get("softplus_floor", -20.0))
    core.load_state_dict(ckpt["model"])
    model = NormalizedInput(core, ckpt.get("input_transform", "log1p")).to(device)
    model.eval()
    return model, ckpt


def describe_ckpt(ckpt):
    return (f"{ckpt['variant']:>5} / {ckpt['loss']:<7} "
            f"epoch {ckpt.get('epoch', '?')}  "
            f"input={ckpt.get('input_transform', 'none')}")


# ── pixel-level extraction ────────────────────────────────────────────────────

class BlockReader:
    """Random access to one staged split, addressed by (block, i, j).

    (i, j) are indices on the *shared* grid -- the AIA grid after the stride-2
    subsample fullBP applied, which is the DEM grid. So (i, j) indexes the DEM
    label and its originating AIA observation simultaneously; see zarr_data for
    why that alignment is a plain slice rather than an offset calculation.
    """

    def __init__(self, root, phase, patch_size=9, stride=2, tolfac=1.4,
                 n_bins=N_AIA_BINS, min_obs=MIN_OBS, with_labels=True):
        import zarr

        self.root, self.phase = root, phase
        self.patch_size, self.stride, self.tolfac = patch_size, stride, tolfac
        self.n_bins, self.min_obs = n_bins, min_obs
        self.pad = patch_size // 2

        def _open(suffix):
            path = os.path.join(root, f"{phase}_{suffix}.zarr")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing {path}")
            return zarr.open(path, mode="r")

        self.X = _open("x")
        self.E = _open("e")
        self.M = _open("m")
        self.Y = _open("y") if with_labels else None
        self.n_blocks = self.X.shape[-1]

    def __len__(self):
        return self.n_blocks

    def read_block(self, idx):
        """Decompress one block once. Returns arrays on the shared grid."""
        s = self.stride
        obs = np.asarray(self.X[:, :, :, idx], dtype=np.float32)[:, ::s, ::s]
        err = np.asarray(self.E[:, :, :, idx], dtype=np.float32)[:, ::s, ::s]
        tol = np.asarray(self.M[:, :, idx])
        dem = None
        if self.Y is not None:
            dem = np.asarray(self.Y[:self.n_bins, :, :, idx], dtype=np.float32)
        return obs, err, tol, dem

    def valid_mask(self, obs, err, tol, tol_levels=(1, 3, 5)):
        """Same admissibility rule the dataloader trains on, so evaluation and
        training see one definition of a usable pixel."""
        finite = np.all(np.isfinite(obs) & np.isfinite(err) &
                        (obs > self.min_obs), axis=0)
        return finite & np.isin(tol, tol_levels)

    def gather(self, obs, err, tol, dem, i, j):
        """Per-pixel tensors for explicit index arrays i, j."""
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        p = self.pad
        obs_pad = np.pad(obs, ((0, 0), (p, p), (p, p)), mode="edge")
        off = np.arange(self.patch_size, dtype=np.int64)
        rr, cc = i[:, None] + off, j[:, None] + off
        patch = obs_pad[:, rr[:, :, None], cc[:, None, :]]           # [6,P,K,K]
        patch = np.ascontiguousarray(patch.transpose(1, 0, 2, 3))    # [P,6,K,K]

        c_obs = np.ascontiguousarray(obs[:, i, j].T)                 # [P,6]
        c_err = np.ascontiguousarray(err[:, i, j].T)
        out = {
            "patch": torch.from_numpy(patch),
            "obs": torch.from_numpy(c_obs),
            "err": torch.from_numpy(c_err),
            "lb": torch.from_numpy(c_obs - self.tolfac * c_err),
            "ub": torch.from_numpy(c_obs + self.tolfac * c_err),
            "tol": torch.from_numpy(tol[i, j].astype(np.uint8)),
            "i": i, "j": j,
        }
        if dem is not None:
            out["dem"] = torch.from_numpy(np.ascontiguousarray(dem[:, i, j].T))
        return out


def assert_same_observations(root_a, root_b, phase, n_check=4, seed=0):
    """The BP and ENet stagings must share their AIA arrays block-for-block.

    Both were staged from the same 1,223 files with the same split, so block k
    should be the same pixels in both and only `_y` should differ. The whole
    cross-loss comparison depends on that, and it is one assertion -- so assert
    it rather than assume it.
    """
    import zarr

    rng = np.random.default_rng(seed)
    A = zarr.open(os.path.join(root_a, f"{phase}_x.zarr"), mode="r")
    B = zarr.open(os.path.join(root_b, f"{phase}_x.zarr"), mode="r")
    if A.shape != B.shape:
        raise AssertionError(f"staging shape mismatch: {A.shape} vs {B.shape}")

    for k in rng.choice(A.shape[-1], size=min(n_check, A.shape[-1]), replace=False):
        a = np.asarray(A[:, :, :, int(k)], dtype=np.float32)
        b = np.asarray(B[:, :, :, int(k)], dtype=np.float32)
        dev = np.nanmax(np.abs(a - b))
        if not (dev == 0):
            raise AssertionError(
                f"block {k}: BP and ENet stagings disagree on AIA "
                f"(max deviation {dev:.3e}). Pixel-for-pixel comparison across "
                f"the two losses is not valid.")
    print(f"[check] {n_check} blocks: BP and ENet stagings share AIA exactly "
          f"(max deviation 0.000e+00)")


# ── DEM curve shape ───────────────────────────────────────────────────────────

def count_peaks(dem, prominence_frac=0.15):
    """Peaks in a DEM curve, prominence relative to the curve max.

    Zero-padded at both ends so a maximum in the first or last logT bin counts.
    Identical to the 2026-07-11 diagnostic so the prevalence numbers here are
    directly comparable to the 5.1-9.1% measured then on the 4 crop timestamps.
    """
    dem = np.asarray(dem, dtype=np.float64)
    if dem.size == 0 or not np.isfinite(dem).all() or dem.max() <= 0:
        return 0, np.array([], dtype=int)
    from scipy.signal import find_peaks
    padded = np.concatenate([[0.0], dem, [0.0]])
    peaks, _ = find_peaks(padded, prominence=prominence_frac * dem.max())
    return len(peaks), peaks - 1
