#!/bin/bash
#SBATCH --job-name=dem_final_layer
#SBATCH --output=logs/eval/final_layer_%j.log
#SBATCH --error=logs/eval/final_layer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_41_cilvr

# experiments/final_layer.py on our staged validation split: the output layer
# (Linear 336 -> 54) of the final label-free models, its singular vectors, the
# activity and usage of the 54 basis outputs against the solver's own
# decomposition, and the rank of the DEMs. Evaluation only, CPU.
# Smoke test: PER_GROUP=300 CORE_CAP=200 SOLVER_PIXELS=300 MAX_BLOCKS=40 TAG=smoke.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODELS=(bp=output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt
        enet=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
        bp_old=output/experiments/resample/scaled_mlp6_barrier_h336_resample.pt)
# MODELS_OVERRIDE="label=ckpt label=ckpt" analyses other checkpoints.
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
args=(--root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS" --phase "${PHASE:-val}"
      --models "${MODELS[@]}" --workers "${SLURM_CPUS_PER_TASK:-16}"
      --per-group "${PER_GROUP:-100000}" --core-cap "${CORE_CAP:-1000000}"
      --solver-pixels "${SOLVER_PIXELS:-20000}" --subsample "${SUBSAMPLE:-5000}"
      --output "output/experiments/diagnostics/final_layer${TAG:+_$TAG}.json")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
python3 -u experiments/final_layer.py "${args[@]}"
