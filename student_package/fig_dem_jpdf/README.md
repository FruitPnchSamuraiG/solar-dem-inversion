# fig:dem_jpdf — predicted vs. reference DEM joint density

Two-step pipeline: a GPU pass pools every (temperature bin, valid pixel)
pair over the full test split into a log-log 2D histogram of predicted vs.
reference-solver DEM value, then a plotting script renders it as a 1x2
panel figure (one method per panel, e.g. DEMNet-vs-BP left,
DEMNet-vs-ElasticNet right).

## Step 1 — build the cache

```bash
python3 compute_dem_jpdf_cache.py \
    --model results/models/<run>/model_best.pth \
    --data /path/to/<reference_solver>_AIA_hofdeconv_full_DS \
    --variant methodbp \
    --output cache/dem_jpdf_methodbp.npz
```

Same model/data conventions as `../table1_dem_metrics/`. Masks to a
positivity floor (`1e-1` in native/scaled DEM units) before taking log10 on
both axes, since the joint density is only meaningful where both predicted
and reference DEM have real signal. Also reports pooled R^2 as a sanity
number.

Run it once per (model, reference) pair you want a panel for — e.g. once
for the BP model against BP ground truth, once for the ElasticNet model
against ElasticNet ground truth. `dem_jpdf_hof.py` expects the outputs at
`cache/dem_jpdf_methodbp.npz` and `cache/dem_jpdf_methoden.npz`; edit the
`CACHES` list at the top of that script if you want different
variant names or a different number of panels.

### SLURM

```bash
sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp dem_jpdf_cache.sh
```

Edit the `#SBATCH` account/mail lines and `REPO_DIR`/overlay/container paths
first.

## Step 2 — plot

```bash
python3 dem_jpdf_hof.py
```

Reads `cache/dem_jpdf_*.npz`, writes `plots/dem_jpdf_hof.{png,pdf}`.
Requires `matplotconfig.py` (included here) for the serif/Computer-Modern
plot style — no external assets needed beyond a normal matplotlib install.
