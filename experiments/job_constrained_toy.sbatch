#!/bin/bash
#SBATCH --job-name=dem_ctoy
#SBATCH --output=logs/eval/constrained_toy_%j.log
#SBATCH --error=logs/eval/constrained_toy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --partition=cpu_short
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_41_general

# experiments/constrained_toy.py: the solver's linear program per pixel against its
# unconstrained stand-ins (exact penalty, augmented Lagrangian), no network. Writes the
# pixel sets the network toy trains on. CPU only. TAG names the output; N_TRAIN,
# TRAIN_BLOCKS, N_HOLD, VAL_BLOCKS, PER_GROUP and CORE_CAP shrink it for a smoke test.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/eval output/experiments/constrained
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
args=(--root "$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS"
      --n-train "${N_TRAIN:-2000000}" --train-blocks "${TRAIN_BLOCKS:-3000}"
      --n-hold "${N_HOLD:-300000}" --per-group "${PER_GROUP:-1000}" --core-cap "${CORE_CAP:-3000}"
      --workers "${WORKERS:-22}"
      --output "output/experiments/constrained/toy${TAG:+_$TAG}.json")
if [ -n "${VAL_BLOCKS:-}" ]; then args+=(--val-blocks "$VAL_BLOCKS"); fi
python3 -u experiments/constrained_toy.py "${args[@]}"
