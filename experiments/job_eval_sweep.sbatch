#!/bin/bash
#SBATCH --job-name=dem_evalsweep
#SBATCH --output=logs/eval/evalsweep_%j.log
#SBATCH --error=logs/eval/evalsweep_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# 16 metrics passes over the test split plus one shared census pass. The
# 4-pass eval job (15179492) took 3:43, so 16 passes is ~15 min; an hour is
# generous and stays under the 2 h GPU-utilisation threshold.
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced
#SBATCH --mail-user=hsr3649@nyu.edu
#
#   mkdir -p logs/eval && sbatch experiments/job_eval_sweep.sbatch
#
# Scores all 16 sweep checkpoints on the untouched test split. The sweep's own
# logs are validation numbers -- validation chose each run's saved epoch, so
# those are mildly optimistic and cannot settle the production width.

set -e

if [ -n "$SLURM_SCRIPT" ]; then
  EXP_DIR="$(cd "$(dirname "$SLURM_SCRIPT")" && pwd)"
elif [ -d "$SLURM_SUBMIT_DIR/experiments" ]; then
  EXP_DIR="$SLURM_SUBMIT_DIR/experiments"
else
  EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_DIR="$(dirname "$EXP_DIR")"
echo "REPO_DIR=$REPO_DIR"

DATA_ROOT=${DEM_DATA_ROOT:-$SCRATCH/dem/data}

cd "$REPO_DIR"
source env.sh
echo "### git HEAD: $(git rev-parse --short HEAD)   (pull before submitting!)"

python3 -u experiments/eval_sweep.py \
  --bp_root "$DATA_ROOT/lp_AIA_hofdeconv_full_DS" \
  --enet_root "$DATA_ROOT/elasticnet_AIA_hofdeconv_full_DS" \
  --ckpt_dir "${CKPT_DIR:-output/experiments/sweep}" \
  --out_dir "${OUT_DIR:-output/experiments/eval_sweep}" \
  --n_blocks "${N_BLOCKS:-20}" \
  $EXTRA_ARGS
