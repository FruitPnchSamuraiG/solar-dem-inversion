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
