#!/bin/bash
#SBATCH --job-name=dem_scaled
#SBATCH --output=logs/scaled/scaled_%A_%a.log
#SBATCH --error=logs/scaled/scaled_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=FAIL
#
# --account and --mail-user differ per user; pass them on the sbatch line:
#   mkdir -p logs/scaled
#   sbatch --account=<acct> --mail-user=<you>@nyu.edu experiments/job_scaled.sbatch
#
# The four runs are {mlp6, cnn} x {barrier, enet}. All are AIA-only: the
# solver distinction is in the loss, not the inputs. Tasks 0/1 read the BP
# staging and 2/3 the ENet staging only so each run's *labels* come from the
# solver it is emulating -- training itself never touches them.

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

case $SLURM_ARRAY_TASK_ID in
  0) VARIANT=mlp6; LOSS=barrier; DS=lp_AIA_hofdeconv_full_DS ;;
  1) VARIANT=cnn;  LOSS=barrier; DS=lp_AIA_hofdeconv_full_DS ;;
  2) VARIANT=mlp6; LOSS=enet;    DS=elasticnet_AIA_hofdeconv_full_DS ;;
  3) VARIANT=cnn;  LOSS=enet;    DS=elasticnet_AIA_hofdeconv_full_DS ;;
  *) echo "unexpected array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "run: variant=$VARIANT loss=$LOSS data=$DATA_ROOT/$DS"

cd "$REPO_DIR"
source env.sh

python3 -u experiments/train_scaled.py \
  --root "$DATA_ROOT/$DS" \
  --variant "$VARIANT" \
  --loss "$LOSS" \
  --epochs "${EPOCHS:-12}" \
  --num_workers 8 \
  $EXTRA_ARGS
