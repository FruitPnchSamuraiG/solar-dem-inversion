#!/bin/bash
#SBATCH --job-name=dem_resample
#SBATCH --output=logs/scaled/resample_%j.log
#SBATCH --error=logs/scaled/resample_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=02:15:00
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# The production h232 model with fresh pixels every epoch (--resample_pixels).
# Every earlier scaled run drew the same 512 pixels per block each epoch, so it
# trained on one fixed 30M-pixel sample (~0.8% of the training pixels) 40 times.
# Everything else matches the production runs exactly, warmup included: BP is
# sweep task 15185224_3 (500 warmup steps, the sweep's first submission), ENet
# the matched alpha=0.001 retrain 17385814 (3000). The _resample tag keeps both
# production checkpoints intact.
#
#   sbatch --export=ALL,TRACK=bp experiments/job_train_resample.sbatch
#   (or experiments/submit_resample.sh, which also queues the evaluations)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/scaled output/experiments/resample
echo "### git HEAD: $(git rev-parse --short HEAD)"

case "${TRACK:?set TRACK=bp or TRACK=enet}" in
  bp)   ROOT=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS
        LOSS_ARGS=(--loss barrier); WARMUP=500 ;;
  enet) ROOT=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS
        LOSS_ARGS=(--loss enet --enet_C -1 --enet_alpha 0.001 --enet_lam 0.5); WARMUP=3000 ;;
  *) echo "TRACK must be bp or enet" >&2; exit 2 ;;
esac

python3 -u experiments/train_scaled.py \
  --root "$ROOT" \
  --variant mlp6 \
  --hidden 232 \
  "${LOSS_ARGS[@]}" \
  --epochs 40 \
  --batch_blocks 16 \
  --num_workers 10 \
  --warmup_steps "$WARMUP" \
  --resample_pixels \
  --out_dir output/experiments/resample
