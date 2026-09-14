# Corrected ENet visualizer

The original local full-disk ENet job list omitted `--fitlinearalpha`, using
the default alpha=1. Both its solver reference images and its model predictions
therefore need replacement for the alpha=0.001 comparison.

From the Torch repository root:

```bash
mkdir -p logs/visuals
sbatch --partition=cpu_short experiments/job_visuals_matched_cpu.sbatch
```

One CPU allocation (16 cores, 64 GB, four-hour limit) sequentially processes
the nine dates in `visualizer_dates.txt`. It uses `dataset/fullBP.py`, the
existing Hofmeister PSFs and pointing cache, and Samuel's generation options:
alpha=0.001, l1_ratio=0.5, positive/no-intercept ElasticNet, clean input,
Hofmeister deconvolution, full errors, no basis truncation, stride two, extended
temperature storage, and `--zerochill`. Only the clean solve is needed for
each displayed disk; noisy test targets are not regenerated.

Then it exports the trained checkpoint under
`output/experiments/matched_enet_alpha0p001/` and renders both solver and model
images. CPU inference is supported by the exporter. Full runtime for this
combined job has not yet been measured.

Everything is written beneath
`$SCRATCH/dem/visuals/matched_enet_alpha0p001/`, preserving the old assets.
Completed reference files are reused on retry; predictions and renders are
regenerated. The preview is staged only after every date succeeds. BP assets
are linked from the existing visualizer.

Logs: `logs/visuals/matched_cpu_JOBID.log` and `.err`.
Preview: `$SCRATCH/dem/visuals/matched_enet_alpha0p001/preview`.
Use this new preview directory when serving or deploying. Deployment must
dereference the asset symlinks (for example `rsync --copy-links`). Do not run
the old sequential export job for this update: it selects the historical
ENet checkpoint and references.

## Add the supervised predictions

After the matched preview exists, submit from the Torch repository root:

```bash
mkdir -p logs/visuals
sbatch --partition=cpu_short experiments/job_visuals_supervised_cpu.sbatch
```

One CPU job exports both supplied supervised checkpoints for all nine dates
(18 exports), using the saved AIA grids from each track's existing viewer
exports. It does not rerun either solver. The exporter checks the pointwise
architecture before batching pixels and uses the same regression path as the
supervised evaluator. The renderer and its colour scales are shared with the
existing solver and label-free images. Only the first 18 temperature bins are
displayed. This shows deterministic DEM predictions, not the uncertainty head.

New assets are `assets/bp_supervised/DATE` and `assets/enet_supervised/DATE`
under the matched visualizer root above. The job restages the preview after
all exports complete. Logs are `logs/visuals/supervised_cpu_JOBID.log` and `.err`.
The viewer keeps the reference solver on the left; the Prediction dropdown on
the right offers that track's label-free MLP6 or supervised model. It lists
only dates rendered for both selections and preserves the prediction type
when switching solver tracks, when available. Existing URLs remain supported.

After checking completion, repeat the preview download and public-site upload
with rsync. The public site does not update automatically when a Torch job
finishes. Keep symlink dereferencing enabled for the download (`rsync -rLt`).
