#!/bin/bash
#SBATCH --job-name=dem_vis_sup
#SBATCH --output=logs/visuals/supervised_cpu_%j.log
#SBATCH --error=logs/visuals/supervised_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_general
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the DEM repository root}"
source env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export VIS_ROOT="$SCRATCH/dem/visuals/matched_enet_alpha0p001"
mapfile -t DATES < <(grep -v '^#' experiments/visualizer_dates.txt | sed '/^$/d')
test "${#DATES[@]}" -gt 0

# Reuse each track's existing, aligned AIA input. No new solver runs.
for track in bp enet; do
  if [ "$track" = bp ]; then
    model=eval_and_enet_specs/checkpoints/model_best_bp.pth
    input_dir="$SCRATCH/dem/visuals/exports/bp_mlp6_h232"
  else
    model=eval_and_enet_specs/checkpoints/model_best_en.pth
    input_dir="$VIS_ROOT/exports"
  fi
  test -r "$model"
  for stamp in "${DATES[@]}"; do
    [[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]
    test -r "$input_dir/dem_$stamp.npz"
  done
done

for track in bp enet; do
  if [ "$track" = bp ]; then
    model=eval_and_enet_specs/checkpoints/model_best_bp.pth
    input_dir="$SCRATCH/dem/visuals/exports/bp_mlp6_h232"
  else
    model=eval_and_enet_specs/checkpoints/model_best_en.pth
    input_dir="$VIS_ROOT/exports"
  fi
  for stamp in "${DATES[@]}"; do
    echo "Supervised $track: $stamp"
    output="$VIS_ROOT/exports/${track}_supervised/dem_$stamp.npz"
    python3 -u experiments/export_supervised_full_disk.py \
      --model "$model" --input "$input_dir/dem_$stamp.npz" \
      --output "$output" --device cpu
    python3 -u experiments/render_full_disk_visuals.py \
      --input "$output" --source model \
      --output "$VIS_ROOT/assets/${track}_supervised/$stamp"
  done
done
bash experiments/stage_visualizer_preview.sh
echo "Completed supervised viewer additions: $VIS_ROOT/preview"
