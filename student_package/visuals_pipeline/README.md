# Flare DEM visualization pipeline (prediction -> PNGs -> webapp)

Predicts a DEM cube for a flare timestamp, renders per-bin DEM maps /
resynthesized AIA / joint-PDFs / ROI curves as PNGs, and serves them from a
static model-vs-classical-solver compare viewer.

```
AIA FITS (data/flares/<timestamp>/)
  → predict_classification.py         → preds/<model_name>/dem_<timestamp>.npz
  → createTestVisuals.py               (wraps dumpVisuals.py per .npz file)
  → dumpVisuals.py                     → results/vis/<model_name>/<timestamp>/*.png
  → webapp/compare.html + models.json
```

### 1. Predict — `predict_classification.py`

Loads a checkpoint, runs it over raw AIA FITS (`--mode raw_aia`), writes
`dem_<timestamp>.npz`. Imports `src/data.py`/`model.py`/`utils.py` from the
main repo via a relative `sys.path.append(...)`.

### 2. Visualize — `createTestVisuals.py` → `dumpVisuals.py`

```bash
python3 createTestVisuals.py \
    --input_folder preds/<model_name>/dem_<timestamp>.npz \
    --run_name <model_name>
```

`createTestVisuals.py` infers the timestamp from the filename, finds a
comparison/reference DEM file (`--dem_folder`), and calls `dumpVisuals.py`
with `--dem --aia_resynth --dem_jpdfs --dem_pixels --regions_jpdfs
--roi_dems`. `dumpVisuals.py` (~1100 lines) is the actual rendering code;
usually called through the wrapper rather than directly.

Output: `results/vis/<model_name>/<timestamp>/` — `dem_0.png`…`dem_25.png`,
`mean_logt.png`, `aia_0_synth.png`…`aia_5_synth.png`, `roi_pixels/`,
`roi_jpdfs/`, etc.

### 3. Single-timestamp driver — `predictVisualFlare.py`

Combines steps 1-2 for one (model, timestamp) pair — used by the SLURM
array jobs below. Handles classical solvers (`--model_name lp|qp|enet|...`)
by skipping prediction and only building visuals from an already-solved
`.npz`.

```bash
python3 predictVisualFlare.py \
    --model_name <ModelClass-LossName_run_..._timestamp> \
    --input_path xrtData_lp_full \
    --timestamp 20170910_1548 \
    [--skip_existing] [--just_predict]
```

Model heuristics (bin spacing, ReLU, AIA-only, Hofmeister deconvolution)
are auto-detected from substrings in the model directory name.

`predictAndVisualFlares.py` is an older batch driver, superseded by
`predictVisualFlare.py` + the array jobs below.

### 4. Batch over models x timestamps — SLURM array jobs

```bash
sbatch --array=1-N slurm/array_flare_visualize.sh <models_file> <timestamps_file>
# N = (number of models) * (number of timestamps)
```

Example lists: `slurm/flare_timestamps_one.txt`,
`slurm/flare_models_bp_deconv.txt`, `slurm/flare_models_corrected_bp_aia.txt`.

Edit `#SBATCH`/`REPO_DIR`/overlay/container paths at the top of each `.sh`
first.

## Webapp viewer (`webapp/`)

Static HTML/JS, no build step.

- **`compare.html`/`.js`** — side-by-side comparison of two runs (e.g. a
  model's `results/vis/<model_name>/` output vs. a classical solver's, such
  as `results/vis/lp/` or `results/vis/enet/`) for the same timestamp.
  Dropdowns pick "Model 1"/"Model 2" from `models.json`; view mode switches
  between metrics, per-bin DEM maps, resynthesized AIA, and joint-PDFs.

`models.json.example` shows the expected format (flat list of
`"<model_name>/<timestamp>"` strings). Regenerate with
`generate_models_json.py` after running the pipeline, rename to
`models.json` next to `compare.html`.
