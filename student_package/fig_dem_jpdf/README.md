# fig:dem_jpdf — predicted vs. reference DEM joint density

GPU pass pools every (temperature bin, valid pixel) pair over a test split
into a log-log 2D histogram of predicted vs. reference-solver DEM value;
plotting script renders a 1x2 panel (one method per panel).

## Step 1 — build the cache

```bash
python3 compute_dem_jpdf_cache.py \
    --model results/models/<run>/model_best.pth \
    --data /path/to/<reference_solver>_AIA_hofdeconv_full_DS \
    --variant methodbp \
    --output cache/dem_jpdf_methodbp.npz
```

Same model/data conventions as `../table1_dem_metrics/`. Masks to a
positivity floor (`1e-1`) before log10 on both axes; also reports pooled
R^2. Run once per (model, reference) pair — `dem_jpdf_hof.py` expects
`cache/dem_jpdf_methodbp.npz` and `cache/dem_jpdf_methoden.npz` (edit the
`CACHES` list at the top of that script for different names/panel counts).

```bash
sbatch --export=MODEL=<ckpt>,DATA=<ds_dir>,VARIANT=methodbp dem_jpdf_cache.sh
```

## Step 2 — plot

```bash
python3 dem_jpdf_hof.py
```

Reads `cache/dem_jpdf_*.npz`, writes `plots/dem_jpdf_hof.{png,pdf}`.
Needs `matplotconfig.py` (included) for plot styling.
