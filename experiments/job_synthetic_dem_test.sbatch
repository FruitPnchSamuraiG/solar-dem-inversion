#!/bin/bash
#SBATCH --job-name=dem_synth
#SBATCH --output=logs/eval/synth_%j.log
#SBATCH --error=logs/eval/synth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL

# experiments/synthetic_dem_test.py: synthetic DEMs with a known answer, inverted by both
# solvers, our final models and the supervised models (1.05M and 10.13M), on the CPU.
#   sbatch experiments/job_synthetic_dem_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/diagnostics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -u experiments/synthetic_dem_test.py --root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS" \
  --models bp_ours=output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt \
           enet_ours=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt \
  --supervised bp_sup10m=paper_model/checkpoints/BP_5_Wide_10.1M.pth enet_sup10m=paper_model/checkpoints/EN_2_Wide_10.1M.pth \
               bp_sup1m=eval_and_enet_specs/checkpoints/model_best_bp.pth enet_sup1m=eval_and_enet_specs/checkpoints/model_best_en.pth \
  --n "${N:-4000}" --workers "$SLURM_CPUS_PER_TASK" \
  --output "output/experiments/diagnostics/synthetic_dem_test${TAG:+_$TAG}.json"
