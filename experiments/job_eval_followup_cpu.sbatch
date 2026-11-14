#!/bin/bash
#SBATCH --job-name=dem_eval_cpu
#SBATCH --output=logs/eval/eval_cpu_%j.log
#SBATCH --error=logs/eval/eval_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_general

# Full shared-test evaluation of one label-free checkpoint, on the CPU: the bright-failure
# diagnostic, then the AIA fit by brightness (the two CPU job scripts, one after the other).
# Reading the zarr dominates this work, so on a GPU node the GPU would mostly wait, under the
# cluster's GPU-utilization limits; here it runs after the training job, which holds its GPU
# only while it trains (experiments/submit_followup.sh one() chains it with afterok).
#   sbatch --dependency=afterok:<train job> --export=ALL,TRACK=bp,MODEL_PATH=...,TAG=... \
#          experiments/job_eval_followup_cpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
mkdir -p logs/eval output/experiments/diagnostics
[ -f "${MODEL_PATH:?}" ] || { echo "missing checkpoint: $MODEL_PATH" >&2; exit 3; }
: "${TRACK:?set TRACK=bp or TRACK=enet}" "${TAG:?}"
MODEL_PATH="$MODEL_PATH" TAG="$TAG" bash experiments/job_diagnose_bright_failures.sbatch
LF_MODEL="$MODEL_PATH" TAG="$TAG" bash experiments/job_aia_fit_by_brightness.sbatch
