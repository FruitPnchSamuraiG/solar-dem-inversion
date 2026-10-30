#!/bin/bash
#SBATCH --job-name=dem_blowup
#SBATCH --output=logs/eval/blowup_%j.log
#SBATCH --error=logs/eval/blowup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=01:59:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_cilvr

# Blow-up census (experiments/blowup_census.py): every model of one track over
# the full shared test set on one GPU. The worst pixels per model, their overlap
# between models, what they are, and their distance to the training data.
#   sbatch --export=ALL,TRACK=bp experiments/job_blowup_census.sbatch
#   sbatch --export=ALL,TRACK=enet --account=torch_pr_41_courant experiments/job_blowup_census.sbatch
# Smoke test: add MAX_BLOCKS=640,TRAIN_BLOCKS=100,TAG=smoke. Block ranges: START
# and STOP write a part file (one job per range); MERGE=1 combines the parts.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export DEM_EVAL_DEVICE=cuda
WORKERS=${WORKERS:-8}
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
TRACK=${TRACK:?set TRACK=bp or TRACK=enet}
args=(--track "$TRACK"
      --output "output/experiments/diagnostics/${TRACK}_blowup_census${TAG:+_$TAG}.json"
      --workers "$WORKERS" --threads-per-worker "$THREADS"
      --train-blocks "${TRAIN_BLOCKS:-3000}")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
if [ -n "${START:-}" ]; then args+=(--start "$START" --stop "${STOP:?set STOP with START}" --part); fi
if [ -n "${MERGE:-}" ]; then args+=(--merge); fi
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a OV <<< "$MODELS_OVERRIDE"; args+=(--models "${OV[@]}"); fi
python3 -u experiments/blowup_census.py "${args[@]}"
