#!/bin/bash
#SBATCH --job-name=dem_vis_match
#SBATCH --output=logs/visuals/matched_cpu_%j.log
#SBATCH --error=logs/visuals/matched_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_general
#SBATCH --mail-type=FAIL

# Nine clean ENet references, corrected MLP predictions, and a separate preview.
# Submit from the repository root. No GPU or dependency on test evaluation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the DEM repository root}"
source env.sh
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SOURCE_ROOT=/projects/rps/dff6142/fouheylab/solar_dem/xrtSource
OLD_VIS="$SCRATCH/dem/visuals"
export VIS_ROOT="$OLD_VIS/matched_enet_alpha0p001"
MODEL=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt
POINTING=dataset/aia_pointing_master_2014-2016.ecsv
mapfile -t DATES < <(grep -v '^#' experiments/visualizer_dates.txt | sed '/^$/d')
test "${#DATES[@]}" -gt 0
test -r "$MODEL"
test -r "$POINTING"
test -r RData.npz
for wl in 94 131 171 193 211 335; do
  test -r "dataset/hofmeister_psf/psf_aia_${wl}.fits"
done
for stamp in "${DATES[@]}"; do
  [[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]
  if [ ! -d "$SOURCE_ROOT/$stamp" ]; then
    echo "Missing source directory: $SOURCE_ROOT/$stamp" >&2
    exit 1
  fi
done
mkdir -p "$VIS_ROOT/references" "$VIS_ROOT/exports" "$VIS_ROOT/assets"
echo "Model: $MODEL"
echo "Reference settings: alpha=0.001 l1_ratio=0.5 noisy=0 zerochill; ${#DATES[@]} dates"

for stamp in "${DATES[@]}"; do
  echo "Processing $stamp"
  reference="$VIS_ROOT/references/$stamp.npz"
  if [ ! -f "$reference" ]; then
    # Publish only completed files; an interrupted solve is never reused.
    work_dir=$(mktemp -d "$VIS_ROOT/references/.solve-${stamp}-XXXXXX")
    python3 -u dataset/fullBP.py "$SOURCE_ROOT/$stamp" "$work_dir/reference.npz" \
      --noisy 0 --deconvolve hofmeister --errorfn full --notrunc --extendto8 \
      --decimate 2 --zerochill --fitfn elasticnet \
      --fitlinearalpha 0.001 --fitlinearl1ratio 0.5 \
      --parallel "$SLURM_CPUS_PER_TASK" --pointing_file "$POINTING"
    test -s "$work_dir/reference.npz"
    mv "$work_dir/reference.npz" "$reference"
    rmdir "$work_dir"
  fi
  export_file="$VIS_ROOT/exports/dem_$stamp.npz"
  OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    python3 -u experiments/export_mlp6_full_disk.py \
      --model "$MODEL" --input_npz "$reference" --output "$export_file"
  python3 -u experiments/render_full_disk_visuals.py \
    --input "$export_file" --source reference \
    --output "$VIS_ROOT/assets/enet_solver/$stamp"
  python3 -u experiments/render_full_disk_visuals.py \
    --input "$export_file" --source model \
    --output "$VIS_ROOT/assets/enet_mlp6_h232/$stamp"
done

# Reuse BP images; only the ENet track needed a new objective.
for run in bp_solver bp_mlp6_h232; do
  test -d "$OLD_VIS/assets/$run"
  if [ ! -e "$VIS_ROOT/assets/$run" ]; then
    ln -s "$OLD_VIS/assets/$run" "$VIS_ROOT/assets/$run"
  fi
done
bash experiments/stage_visualizer_preview.sh
echo "Completed matched ENet visuals and preview: $VIS_ROOT/preview"
