#!/bin/bash
#SBATCH --job-name=dem_aia_pair
#SBATCH --output=logs/eval/aia_pair_%j.log
#SBATCH --error=logs/eval/aia_pair_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
case "${TRACK:?Set TRACK=bp or TRACK=enet}" in
  bp)
    label_free=output/experiments/sweep/scaled_mlp6_barrier_h232.pt
    supervised=eval_and_enet_specs/checkpoints/model_best_bp.pth ;;
  enet)
    label_free=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt
    supervised=eval_and_enet_specs/checkpoints/model_best_en.pth ;;
  *) exit 2 ;;
esac
extra=()
if [[ -n "${MAX_BLOCKS:-}" ]]; then extra+=(--max-blocks "$MAX_BLOCKS"); fi
python3 -u experiments/eval_aia_shared_test.py \
  --track "$TRACK" \
  --data "/projects/rps/dff6142/fouheylab/solar_dem/${TRACK}_AIA_hofdeconv_full_DS" \
  --label-free "$label_free" --supervised "$supervised" --device cpu \
  --output "output/experiments/paper_eval/${TRACK}_aia_${OUTPUT_TAG:-shared_test}.json" \
  "${extra[@]}"
