#!/bin/bash
#SBATCH --job-name=dem_eval_gpu
#SBATCH --output=logs/eval/eval_gpu_%j.log
#SBATCH --error=logs/eval/eval_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:59:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# Full shared-test bright-failure diagnostic, then the AIA fit, for one large
# label-free checkpoint, with the label-free model on the GPU. The ~10M models
# cost ~25x the 360k ones to predict, too slow for the CPU jobs. Reading the
# zarr stays CPU-parallel. The supervised model still runs on the CPU.
#   sbatch --export=ALL,TRACK=bp,MODEL_PATH=...,TAG=... experiments/job_eval_followup_gpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export DEM_EVAL_DEVICE=cuda
WORKERS=8
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp) SUP=eval_and_enet_specs/checkpoints/model_best_bp.pth ;;
  enet) SUP=eval_and_enet_specs/checkpoints/model_best_en.pth ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
[ -f "${MODEL_PATH:?}" ] || { echo "missing checkpoint: $MODEL_PATH" >&2; exit 3; }
DATA=/projects/rps/dff6142/fouheylab/solar_dem/${TRACK}_AIA_hofdeconv_full_DS
python3 -u experiments/diagnose_bright_failures.py --model "$MODEL_PATH" --data "$DATA" \
  --reference "$TRACK" --workers "$WORKERS" --threads-per-worker "$THREADS" \
  --output "output/experiments/diagnostics/${TRACK}_bright_failure_${TAG:?}.json"
python3 -u experiments/aia_fit_by_brightness.py --label-free "$MODEL_PATH" --supervised "$SUP" \
  --reference "$TRACK" --data "$DATA" --workers "$WORKERS" --threads-per-worker "$THREADS" \
  --samples 60 --output "output/experiments/diagnostics/${TRACK}_aia_fit_${TAG}.json"
