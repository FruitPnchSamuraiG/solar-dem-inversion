#!/bin/bash
#SBATCH --job-name=dem_ctoynet
#SBATCH --output=logs/scaled/ctoynet_%j.log
#SBATCH --error=logs/scaled/ctoynet_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_41_general

# experiments/constrained_toy_net.py: one network trained on the pixel set written by
# job_constrained_toy.sbatch, under one unconstrained stand-in for the solver's linear
# program. FORM=quad|quad_ramp|price|huber|al|alnet, NAME names the output, DATA is the
# pixel-set stem (default toy), ARGS passes further options (spaces, no commas).
# Runs on a GPU when the job has one, otherwise on the CPUs it was given.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/constrained
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
read -r -a extra <<< "${ARGS:-}"
python3 -u experiments/constrained_toy_net.py \
  --data "output/experiments/constrained/${DATA:-toy}" \
  --form "${FORM:?set FORM}" \
  --output "output/experiments/constrained/net_${NAME:?set NAME}.json" "${extra[@]}"
