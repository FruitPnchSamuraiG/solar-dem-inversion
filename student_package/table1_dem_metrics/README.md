# Table 1 — DEM MSE / Rel. Err. / W1

`compute_paper_table_metrics.py` evaluates one (model checkpoint, reference
DEM solver) pair over a full test split and reports, stratified by
Full/Bright/Quiet pixels:

- **DEM MSE** — mean squared error, predicted vs. reference DEM, per
  (pixel, temperature bin).
- **DEM Rel. Err. (%)** — mean `|pred - gt| / (|gt| + 0.1)` per (pixel,
  temperature bin). The `0.1` floor avoids blowing up on near-zero GT.
- **W1 (dex)** — 1-D Wasserstein distance between the predicted and
  reference DEM curves at each pixel, treated as distributions over the
  18-bin logT grid, then averaged.

This is a trimmed version of the lab's `scripts/compute_paper_table_metrics.py`:
the original also computes several alternative relative-error definitions
(symmetric, global-EM-normalized, etc.) used only for internal sanity
checks during development — those were stripped out here since only the
three metrics above are cited in the paper table.

"Bright" pixels: any AIA channel is at or above that channel's own top-5%
intensity threshold, computed from the same test split (not borrowed from
elsewhere). Everything else is "Quiet". "Full" = all valid pixels.

## Usage

```bash
python3 compute_paper_table_metrics.py \
    --model results/models/<run>/model_best.pth \
    --data /path/to/<reference_solver>_AIA_hofdeconv_full_DS \
    --rdata RData.npz \
    --variant methodbp --reference "Basis Pursuit (BP)" \
    --output table_comparison_bp.json
```

Required inputs:
- `--model`: a `model_best.pth`-style checkpoint saved by `src/train.py`
  (needs `model_name`, `args`, `model_state_dict` keys).
- `--data`: a directory with `test_x.zarr` (AIA, `[6,H,W,N]`) and
  `test_y.zarr` (reference-solver DEM ground truth, `[18+,h,w,N]` — `h,w`
  may be lower resolution than `H,W`; the script upsamples with NaN-fill,
  matching `zarrDataset`'s handling of "compact" Y).
- `--rdata`: `RData.npz` with the `logT` temperature grid (18 bins).

Optional: `--thresholds_json` to reuse a precomputed
`{"test": [thr_94, thr_131, ...]}` file instead of recomputing the top-5%
AIA thresholds from scratch (this pass over the full test set otherwise
adds several minutes).

## SLURM

`table_comparison_metrics.sh` is a GPU job wrapper:

```bash
sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp,REFERENCE="Basis Pursuit (BP)" \
    table_comparison_metrics.sh
```

Edit the `#SBATCH` account/mail lines and the `REPO_DIR`/overlay/container
paths at the top before running — they're copied from the original lab
cluster config.
