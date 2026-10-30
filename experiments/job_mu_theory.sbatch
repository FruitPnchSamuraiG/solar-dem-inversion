#!/bin/bash
#SBATCH --job-name=dem_mu_theory
#SBATCH --output=logs/eval/mu_theory_%j.log
#SBATCH --error=logs/eval/mu_theory_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_41_cilvr

# experiments/mu_theory.py: the band weight from penalty-method theory, checked
# per pixel on the validation population of the mu screen (BP track; needs the
# per-pixel errors, so our staged split). CPU only. PHASE=val|test; TAG adds a
# suffix; MAX_BLOCKS and PER_GROUP shrink it for a smoke test; MODELS lists
# label=checkpoint pairs scored the same way (empty string for none).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PHASE=${PHASE:-val}
MODELS=${MODELS-adopted=output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt}
args=(--root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS" --phase "$PHASE"
      --per-group "${PER_GROUP:-1000}" --core-cap "${CORE_CAP:-6000}" --workers "${WORKERS:-16}"
      --output "output/experiments/diagnostics/bp_mu_theory_${PHASE}${TAG:+_$TAG}.json")
if [ -n "${MAX_BLOCKS:-}" ]; then args+=(--max-blocks "$MAX_BLOCKS"); fi
if [ -n "$MODELS" ]; then read -r -a m <<< "$MODELS"; args+=(--models "${m[@]}"); fi
python3 -u experiments/mu_theory.py "${args[@]}"
