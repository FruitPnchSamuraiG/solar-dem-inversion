# Flare DEM visualization pipeline (prediction -> PNGs -> webapp)

Given a flare timestamp's AIA FITS data, this pipeline runs a model to
predict a DEM cube, renders it (per-bin DEM maps, resynthesized AIA,
joint-PDFs, region-of-interest DEM curves, etc.) as a folder of PNGs, and
serves the result from a small static HTML/JS side-by-side comparison
viewer (model vs. classical solver, e.g. neural-net prediction vs. BP/QP/
ElasticNet on the same timestamp).

## Pipeline stages

```
AIA FITS (data/flares/<timestamp>/)
  → predict_classification.py         → preds/<model_name>/dem_<timestamp>.npz
  → createTestVisuals.py               (wraps dumpVisuals.py per .npz file)
  → dumpVisuals.py                     → results/vis/<model_name>/<timestamp>/*.png
  → webapp/*.html + models.json        (browse in a static web page)
```

### 1. Predict — `predict_classification.py`

Loads a checkpoint, runs it over a folder of raw AIA FITS
(`--mode raw_aia`), writes `dem_<timestamp>.npz` (DEM cube + classification
outputs). Needs `src/data.py`, `src/model.py`, `src/utils.py` from the main
repo on `PYTHONPATH` (handled by the `sys.path.append(...)` at the top —
keep this folder inside/alongside a `demdemo` checkout).

### 2. Visualize a single .npz — `createTestVisuals.py` → `dumpVisuals.py`

```bash
python3 createTestVisuals.py \
    --input_folder preds/<model_name>/dem_<timestamp>.npz \
    --run_name <model_name>
```

`createTestVisuals.py` is a thin wrapper: it infers the timestamp from the
filename, looks for a comparison/reference DEM file (`--dem_folder`, so a
classical-solver result can be overlaid), detects whether classification
(uncertainty) outputs are present, and calls `dumpVisuals.py` with the
right flags (`--dem --aia_resynth --dem_jpdfs --dem_pixels --regions_jpdfs
--roi_dems`).

`dumpVisuals.py` (~1100 lines) is the actual rendering code — per-bin DEM
maps, resynthesized-vs-observed AIA, region-of-interest pixel/joint-PDF
plots, boundary detection for active regions, etc. Run `python3
dumpVisuals.py --help` for the full flag list; most of the time you don't
call it directly, `createTestVisuals.py` does.

Output lands in `results/vis/<model_name>/<timestamp>/`:
`dem_0.png`…`dem_25.png`, `mean_logt.png`, `aia_0_synth.png`…`aia_5_synth.png`,
`roi_pixels/`, `roi_jpdfs/`, etc.

### 3. One-shot single-timestamp driver — `predictVisualFlare.py`

Combines steps 1-2 for one (model, timestamp) pair — this is what the
SLURM array jobs below call. Also handles the special case of a classical
solver (`--model_name lp|qp|enet|...`) where there's nothing to predict,
only visuals to build from an already-solved `.npz`.

```bash
python3 predictVisualFlare.py \
    --model_name <ModelClass-LossName_run_..._timestamp> \
    --input_path xrtData_lp_full \
    --timestamp 20170910_1548 \
    [--skip_existing] [--just_predict]
```

Model-architecture heuristics (bin spacing, ReLU, AIA-only, Hofmeister
deconvolution) are auto-detected from substrings in the model directory
name — see the comments in the script if you add a model whose name
doesn't match the existing conventions.

`predictAndVisualFlares.py` is an older, "quick and dirty" batch driver
(loops over every timestamp in `data/flares/`, hardcoded model list) —
superseded by `predictVisualFlare.py` + the array jobs below, kept here
only because it may still be a useful reference for looping without SLURM.

### 4. Batch over models x timestamps — SLURM array jobs

`slurm/array_flare_predict.sh` (predict only) and
`slurm/array_flare_visualize.sh` (predict + visualize) both take a models
list and a timestamps list and fan out over every combination as array
tasks:

```bash
sbatch --array=1-N slurm/array_flare_visualize.sh <models_file> <timestamps_file>
# N = (number of models) * (number of timestamps)
```

Example lists included: `slurm/flare_timestamps_one.txt` (single flare),
`slurm/flare_models_bp_deconv.txt`, `slurm/flare_models_corrected_bp_aia.txt`.

Edit the `#SBATCH` account/mail lines and `REPO_DIR`/overlay/container
paths at the top of each `.sh` first (marked with `NOTE:` comments).

## Webapp viewer (`webapp/`)

Static HTML/JS, no build step — open directly or serve with any static
file server. Only one viewer is included here (the lab's full webapp has
several others — single-model browsers, a regression-vs-classification-head
comparison, a one-off frozen before/after page — trimmed since this
package's scope is just "model vs. classical solver"):

- **`compare.html`/`.js`** — side-by-side comparison of two runs (e.g. a
  neural-net model's `results/vis/<model_name>/` output vs. a classical
  solver's, such as `results/vis/lp/` or `results/vis/enet/` — see
  `predictVisualFlare.py` above for how those classical-solver folders get
  populated) for the same timestamp. Dropdowns pick "Model 1"/"Model 2"
  from `models.json`; view mode switches between metrics, per-bin DEM maps,
  resynthesized AIA, and joint-PDFs.

`models.json.example` shows the expected format (a flat list of
`"<model_name>/<timestamp>"` strings, populated from `results/vis/`).
Regenerate your own with `generate_models_json.py` after running the
pipeline above, and rename the output to `models.json` next to
`compare.html`.
