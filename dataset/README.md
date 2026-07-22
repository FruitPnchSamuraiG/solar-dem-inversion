# dataset

Scripts to build the training dataset for the DEM network.

Pipeline:

    1. invert     — run LP inversions on AIA+Hofmeister data
    2. stage      — collect inversion .npz files into training zarr

---

## step 1 — LP inversions

Generates one `.npz` per timestamp × noise realization and submits as a SLURM array. AIA-only (no XRT).

```bash
# preview job list without submitting
python3 submit_bp_aia_hofdeconv_full.py --dry_run

# submit
python3 submit_bp_aia_hofdeconv_full.py
```

Input: raw AIA FITS in `/scratch/.../data/xrtSource/` (Hofmeister deconvolution applied during inversion)  
Output: `.npz` files in `/scratch/.../data/bp_AIA_hofdeconv_full/`  
SLURM worker: `inv_bp_aia_hofdeconv_full.sh`  
Resources: 16 CPU, 64 GB, 2h per job. Auto-submits next batch when done (chained arrays).

### running fullBP.py directly

Each SLURM task calls `fullBP.py` with arguments like:

```bash
python3 fullBP.py <src_dir> <out.npz> \
    --deconvolve hofmeister \
    --errorfn full --notrunc --extendto8 \
    --decimate 2 --zerochill --fitfn lp --parallel 16 \
    --noisy <0..4> \
    --pointing_file aia_pointing_master_2014-2016.ecsv
```

Don't worry about most of these — the submit script fills them in automatically.
The ones that matter: `--noisy` selects the noise realization (0 = clean, 1–4 = noisy draws);
`--deconvolve hofmeister` applies Hofmeister BID PSF deconvolution before inversion (GPU required);
`--decimate 2` halves the spatial resolution of the DEM output (4096→2048) to save memory and disk.

---

## step 2 — stage to zarr

Collects the inversion `.npz` files and stages them into a zarr dataset for training.
Uses the same train/val/test splits as `lp_AIA_notrunc_DS`.

```bash
# submit SLURM job
CONFIG=bp_AIA_hofdeconv_full sbatch stage_hofdeconv_full.sh

# or run directly (needs 64 workers, 256 GB RAM)
launchfast python3 stage_hofdeconv_full.py --config bp_AIA_hofdeconv_full
```

Input: `.npz` files from step 1  
Output: zarr dataset at `/scratch/.../data/bp_AIA_hofdeconv_full_DS/`  
Resources: 64 CPU, 256 GB, 12h

---

## expected dataset sizes

| split | timestamps | patches (×64/disk) |
|---|---|---|
| train | 917  | ~59k  |
| val   | 153  | ~10k  |
| test  | 153  | ~10k  |

5 noise realizations per timestamp (indices 0–4); index 0 is the clean inversion.
