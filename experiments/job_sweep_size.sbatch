#!/bin/bash
#SBATCH --job-name=dem_sweep
#SBATCH --output=logs/sweep/sweep_%A_%a.log
#SBATCH --error=logs/sweep/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# Same 40 epochs as array 15092854, deliberately: the sweep is only meaningful
# against that baseline if nothing else changed. Wall time does not fall with
# width -- this workload is data-loading bound at ~170 s/epoch regardless of
# model size -- so every task costs the same ~1.5 h the full model did.
#SBATCH --time=01:45:00
# %4 caps concurrency: the account's group GPU quota is what put earlier jobs
# in (QOSGrpGRES), and 16 at once would just queue behind itself.
#SBATCH --array=0-15%4
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced
#SBATCH --mail-user=hsr3649@nyu.edu
#
#   mkdir -p logs/sweep && sbatch experiments/job_sweep_size.sbatch
#
# Width sweep on mlp6, the architecture the test-split evaluation settled on
# (better than cnn at every percentile of the objective, better bimodality
# recall at equal precision, sparsity closer to BP's 1.79).
#
# Eight widths, each roughly halving the parameter count, x both losses:
#
#   hidden  680  480  336  232  160  108   72   48
#   params 1.43M 722k 360k 176k  87k  42k  20k  10k
#
# 680 is the baseline already trained, rerun here so the whole curve comes from
# one submission under identical conditions rather than being stitched together.

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

WIDTHS=(680 480 336 232 160 108 72 48)
IDX=$SLURM_ARRAY_TASK_ID
HIDDEN=${WIDTHS[$((IDX % 8))]}

if [ $((IDX / 8)) -eq 0 ]; then
  LOSS=barrier; DS=lp_AIA_hofdeconv_full_DS
else
  LOSS=enet;    DS=elasticnet_AIA_hofdeconv_full_DS
fi

echo "run: mlp6 hidden=$HIDDEN loss=$LOSS data=$DATA_ROOT/$DS"

cd "$REPO_DIR"
source env.sh
echo "### git HEAD: $(git rev-parse --short HEAD)   (pull before submitting!)"

python3 -u experiments/train_scaled.py \
  --root "$DATA_ROOT/$DS" \
  --variant mlp6 \
  --hidden "$HIDDEN" \
  --loss "$LOSS" \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks "${BATCH_BLOCKS:-16}" \
  --num_workers 10 \
  --out_dir "${OUT_DIR:-output/experiments/sweep}" \
  $EXTRA_ARGS
