#!/bin/bash
#SBATCH --job-name=dem_relaxed_band
#SBATCH --output=logs/eval/relaxed_band_%j.log
#SBATCH --error=logs/eval/relaxed_band_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_41_cilvr

# experiments/relaxed_band_screen.py on the arrays of the band-weight theory check
# (job_mu_theory.sbatch): re-solves the best answer at the pixels where the solver
# relaxed its band, with the solver's own band, and recomputes the mu screen. CPU only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIAG=output/experiments/diagnostics
python3 -u experiments/relaxed_band_screen.py --npz "$DIAG/bp_mu_theory_val.npz" \
  --json "$DIAG/bp_mu_theory_val.json" --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --output "$DIAG/bp_relaxed_band_screen.json"
