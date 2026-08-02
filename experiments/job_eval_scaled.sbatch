#!/bin/bash
#SBATCH --job-name=dem_eval
#SBATCH --output=logs/eval/eval_%j.log
#SBATCH --error=logs/eval/eval_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Both scripts together are well under an hour: the test split is ~9.8k blocks
# (~30 s at the measured 170 s/58,688 blocks) and the LP re-solves in the
# bimodal study are ~0.14 ms each. Kept under 2 h like the training jobs.
#SBATCH --time=00:45:00
#SBATCH --mail-type=FAIL
#
# --account and --mail-user differ per user; pass them on the sbatch line:
#   mkdir -p logs/eval
#   sbatch --account=<acct> --mail-user=<you>@nyu.edu experiments/job_eval_scaled.sbatch
#
# No --partition: Torch's job-submit plugin routes to whatever pool is free.

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
BP_ROOT="$DATA_ROOT/lp_AIA_hofdeconv_full_DS"
ENET_ROOT="$DATA_ROOT/elasticnet_AIA_hofdeconv_full_DS"

cd "$REPO_DIR"
source env.sh

echo "### git HEAD: $(git rev-parse --short HEAD)   (pull before submitting!)"

python3 -u experiments/eval_scaled.py \
  --bp_root "$BP_ROOT" --enet_root "$ENET_ROOT" \
  --ckpt_dir "${CKPT_DIR:-output/experiments}" \
  --out_dir "${OUT_DIR:-output/experiments/eval_scaled}" \
  --n_blocks "${N_BLOCKS:-3}" --pixels_per_panel "${N_PIXELS:-6}" \
  $EVAL_ARGS

python3 -u experiments/bimodal_scaled.py \
  --bp_root "$BP_ROOT" --enet_root "$ENET_ROOT" \
  --ckpt_dir "${CKPT_DIR:-output/experiments}" \
  --out_dir "${OUT_DIR:-output/experiments/bimodal_scaled}" \
  --n_blocks "${BIMODAL_BLOCKS:-20}" --n_study "${N_STUDY:-60}" \
  $BIMODAL_ARGS
