#!/bin/bash
#SBATCH --job-name=dem_sweep_rs
#SBATCH --output=logs/sweep/sweep_rs_%A_%a.log
#SBATCH --error=logs/sweep/sweep_rs_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|h100|h200
#SBATCH --time=02:15:00
#SBATCH --array=0-15%4
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced

# The width sweep and the CNN-vs-MLP comparison, redone with fresh pixels every
# epoch (--resample_pixels), now the headline training setup. h232 for both
# tracks is already trained (19071403, 19071406), so it is not repeated.
#   tasks 0-6   mlp6 barrier  h680 h480 h336 h160 h108 h72 h48   (warmup 500)
#   tasks 7-13  mlp6 enet     same widths, alpha=0.001            (warmup 3000)
#   tasks 14-15 patch cnn     barrier, enet
# Warmups match each track's h232 resample run. h680 is trained without
# --hidden (680 is the default), so it and the CNN carry the plain _resample
# suffix that eval_scaled.py looks for. Submit with submit_sweep_resample.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
mkdir -p logs/sweep output/experiments/resample
echo "### git HEAD: $(git rev-parse --short HEAD)"

WIDTHS=(680 480 336 160 108 72 48)
IDX=$SLURM_ARRAY_TASK_ID
BP=$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS
EN=$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS
ENET_ARGS=(--loss enet --enet_C -1 --enet_alpha 0.001 --enet_lam 0.5)
if (( IDX < 7 )); then
  W=${WIDTHS[$IDX]}; ARGS=(--variant mlp6 --root "$BP" --loss barrier --warmup_steps 500)
elif (( IDX < 14 )); then
  W=${WIDTHS[$((IDX - 7))]}; ARGS=(--variant mlp6 --root "$EN" "${ENET_ARGS[@]}" --warmup_steps 3000)
elif (( IDX == 14 )); then
  W=; ARGS=(--variant cnn --root "$BP" --loss barrier --warmup_steps 500)
else
  W=; ARGS=(--variant cnn --root "$EN" "${ENET_ARGS[@]}" --warmup_steps 3000)
fi
if [ -n "$W" ] && [ "$W" != 680 ]; then ARGS+=(--hidden "$W"); fi
echo "task $IDX: ${ARGS[*]}"

python3 -u experiments/train_scaled.py "${ARGS[@]}" \
  --epochs 40 --batch_blocks 16 --num_workers 10 --resample_pixels \
  --out_dir output/experiments/resample
