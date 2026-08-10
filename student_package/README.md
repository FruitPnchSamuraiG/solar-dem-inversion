# DEM package

Standalone copies of three pieces of the `demdemo` pipeline:

- `table1_dem_metrics/` — DEM MSE, Rel. Err. %, W1 vs. a reference solver.
- `fig_dem_jpdf/` — predicted-vs-reference DEM joint-density figure.
- `visuals_pipeline/` — predict + render per-pixel/per-region DEM visuals
  for a flare timestamp, plus a static model-vs-classical-solver compare
  viewer.

## Environment

Runs inside the Apptainer/Singularity container with the `dem_new` conda
env — no system Python. Needs:
- the same overlay/container image referenced in the `.sh` scripts, and
- the `src/` package (models, losses, data loaders) from the main repo —
  the Python scripts import from it via a relative `sys.path.append(...)`,
  so keep this folder inside (or alongside) a checkout of the main repo.

## Notes

- Absolute paths (`/scratch/vp2435/...`, mail address) in a few scripts/SLURM
  wrappers are inherited from the original environment — marked with `NOTE:`
  comments where they need adjusting.
- `dumpVisuals.py`'s confidence-interval overlay path does
  `sys.path.append('../src')` relative to cwd, not script location — run it
  from the main repo's `scripts/` dir if that import fails.

See each subfolder's README for usage.
