# DEM student handoff package

Self-contained copies of three pieces of the `demdemo` pipeline, pulled out
of the main repo so they can be handed to a new student without them having
to find their way through the whole codebase first.

- `table1_dem_metrics/` — computes the paper's Table 1 (DEM MSE, Rel. Err.
  %, W1) for one model vs its reference solver.
- `fig_dem_jpdf/` — computes and plots the joint-density (predicted vs.
  reference DEM) figure.
- `visuals_pipeline/` — predicts + renders per-pixel/per-region DEM visuals
  for a flare timestamp and serves them from a small static webapp,
  including the model-vs-classical-method comparison viewers.

## Environment

All of this runs inside the lab's Apptainer/Singularity container with the
`dem_new` conda env — there is no system Python. See the top-level
`demdemo/CLAUDE.md` in the main repo for the `launch`/`launchfast` helper
functions and the `dem` persistent-instance trick. If you're setting up a
fresh account, you'll need:
- access to the same overlay (`overlay-25GB-500K.ext3`) and container image
  referenced in the `.sh` scripts below, or your own equivalent,
- the `src/` package (models, losses, data loaders) from the main `demdemo`
  repo — the Python scripts here import from it via a relative
  `sys.path.append(...)`, so keep this folder inside (or alongside) a
  checkout of the main repo.

## Known rough edges (inherited from the lab codebase, not cleaned up here)

- Several scripts/SLURM wrappers have absolute paths baked in
  (`/scratch/vp2435/...`, `vp2435@nyu.edu`) from the original author's
  account. Search-and-replace your own netid/paths before running —
  each `.sh` has a `NOTE:` comment at the top marking what to change.
- `dumpVisuals.py` has one code path (confidence-interval overlays, only
  triggered when a prediction includes CI outputs) that does
  `sys.path.append('../src')` relative to the *current working directory*,
  not the script location. If you hit an `ImportError: utils` there, `cd`
  into the main repo's `scripts/` dir first, or fix the import to be
  file-relative.

See each subfolder's own README for usage.
