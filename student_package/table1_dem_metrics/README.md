# Table 1 — DEM MSE / EM Rel. Err. / W1

`compute_paper_table_metrics.py` evaluates one (model checkpoint, reference
DEM solver) pair over a test split, stratified by Full/Bright/Quiet pixels:

- **DEM MSE** — mean squared error, predicted vs. reference DEM.
- **EM Rel. Err. (%)** — mean per-pixel relative error in total emission
  measure: `|sum(pred) - sum(gt)| / (|sum(gt)| + 0.1)`.
- **W1 (dex)** — 1-D Wasserstein distance between predicted and reference
  DEM curves over the 18-bin logT grid.

The JSON also retains the mean per-bin relative error as
`dem_bin_rel_err_pct` for diagnosis; that is not the Table-1 EM metric.

"Bright" = any AIA channel at/above its own top-5% intensity threshold
(computed from the same split). "Quiet" = everything else. "Full" = all
valid pixels.

## Usage

```bash
python3 compute_paper_table_metrics.py \
    --model results/models/<run>/model_best.pth \
    --data /path/to/<reference_solver>_AIA_hofdeconv_full_DS \
    --rdata RData.npz \
    --variant methodbp --reference "Basis Pursuit (BP)" \
    --output table_comparison_bp.json
```

- `--model`: checkpoint with `model_name`, `args`, `model_state_dict` keys.
- `--data`: dir with `test_x.zarr` (AIA) and `test_y.zarr` (reference DEM,
  upsampled with NaN-fill if stored at lower resolution).
- `--rdata`: `RData.npz` with the `logT` grid.
- `--thresholds_json` (optional): precomputed `{"test": [...]}` thresholds
  to skip the AIA threshold pass.

## SLURM

```bash
sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp,REFERENCE="Basis Pursuit (BP)" \
    table_comparison_metrics.sh
```

Edit the `#SBATCH`/`REPO_DIR`/overlay/container paths at the top first.
