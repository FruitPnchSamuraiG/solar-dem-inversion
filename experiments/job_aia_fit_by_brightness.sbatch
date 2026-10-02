#!/bin/bash
#SBATCH --job-name=dem_aia_fit
#SBATCH --output=logs/eval/aia_fit_%j.log
#SBATCH --error=logs/eval/aia_fit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=02:30:00
#SBATCH --account=torch_pr_41_general

# AIA reconstruction by brightness band for label-free, supervised and solver
# DEMs, plus a seeded sample of example pixels. Evaluation only, CPU.
# Smoke test: add MAX_BLOCKS=640,TAG=smoke to --export.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
WORKERS=${WORKERS:-8}
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp) LF=output/experiments/sweep/scaled_mlp6_barrier_h232.pt
      SUP=eval_and_enet_specs/checkpoints/model_best_bp.pth ;;
  enet) LF=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt
        SUP=eval_and_enet_specs/checkpoints/model_best_en.pth ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac
# LF_MODEL overrides the label-free checkpoint (e.g. a new training run); use with TAG.
LF=${LF_MODEL:-$LF}
[ -f "$LF" ] || { echo "missing checkpoint: $LF" >&2; exit 3; }
args=(--label-free "$LF" --supervised "$SUP" --reference "$TRACK"
      --data "/projects/rps/dff6142/fouheylab/solar_dem/${TRACK}_AIA_hofdeconv_full_DS"
      --output "output/experiments/diagnostics/${TRACK}_aia_fit${TAG:+_$TAG}.json"
      --workers "$WORKERS" --threads-per-worker "$THREADS" --samples 60)
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
python3 -u experiments/aia_fit_by_brightness.py "${args[@]}"
