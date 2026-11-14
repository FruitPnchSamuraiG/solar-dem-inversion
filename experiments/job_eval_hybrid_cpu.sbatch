#!/bin/bash
#SBATCH --job-name=dem_eval_hybrid
#SBATCH --output=logs/eval/eval_hybrid_%j.log
#SBATCH --error=logs/eval/eval_hybrid_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_general

# The hybrid (network everywhere, exact solve of each pixel's own band problem at REFINE x the
# Bright threshold or above; src/refine.py) for one BP checkpoint, on the CPU:
#   STEP=pixels  every pixel of our staged test split (full_validation.py --refine), with the
#                pixels the solver fitted at the tight band and the rest reported separately
#   STEP=shared  the shared test set (bright-failure diagnostic and AIA fit), noise rebuilt from
#                the readings with the fitted calibration ratios (src/aia_noise.py)
#   sbatch --dependency=afterok:<train> --export=ALL,STEP=pixels,MODEL_PATH=...,TAG=... experiments/job_eval_hybrid_cpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
[ -f "${MODEL_PATH:?}" ] || { echo "missing checkpoint: $MODEL_PATH" >&2; exit 3; }
REFINE=${REFINE:-3.2}
case "${STEP:?set STEP=pixels or STEP=shared}" in
  pixels)
    python3 -u experiments/full_validation.py --root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS" --phase test \
      --loss barrier --mu 100 --models "full=$MODEL_PATH" --labels --refine "$REFINE" --workers 4 \
      --output "output/experiments/diagnostics/bp_full_validation_${TAG:?}_test.json" ;;
  shared)
    export DEM_REFINE=$REFINE DEM_REFINE_NPZ=$SCRATCH/dem/data/lp_AIA_hofdeconv_full
    export DEM_REFINE_RATIO=${DEM_REFINE_RATIO:-output/experiments/diagnostics/shared_noise_fit.json}
    MODEL_PATH="$MODEL_PATH" TAG="${TAG:?}_refine$REFINE" TRACK=bp bash experiments/job_diagnose_bright_failures.sbatch
    LF_MODEL="$MODEL_PATH" TAG="${TAG}_refine$REFINE" TRACK=bp bash experiments/job_aia_fit_by_brightness.sbatch ;;
  *) echo "STEP must be pixels or shared" >&2; exit 2 ;;
esac
