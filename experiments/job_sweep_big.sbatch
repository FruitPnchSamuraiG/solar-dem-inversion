#!/bin/bash
#SBATCH --job-name=dem_big
#SBATCH --output=logs/sweep/big_%A_%a.log
#SBATCH --error=logs/sweep/big_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# 40 epochs, same as arrays 15092854 and 15185224. The whole value of this run is
# that it extends one curve rather than starting a new one, so nothing else moves.
# Wall time is data-loading bound (~170 s/epoch) and barely responds to width, but
# 11M params do add real GPU work, hence 1:59 rather than 1:45 -- still under the
# 2 h GPU-utilisation policy threshold.
#SBATCH --time=01:59:00
#SBATCH --array=0-2%3
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_41_tandon_advanced
#SBATCH --mail-user=hsr3649@nyu.edu
#
#   mkdir -p logs/sweep && sbatch experiments/job_sweep_big.sbatch
#
# Does MORE capacity buy multi-thermal fidelity? (2026-08-07 meeting, item 1.)
#
# job_sweep_size.sbatch swept DOWNWARD from the 1.43M baseline and found 176k
# loses nothing. This sweeps UPWARD from the same baseline, roughly doubling
# parameters each step, so the two arrays plot as a single curve from 10k to 11M:
#
#   hidden   960   1360   1920
#   params  2.85M  5.72M  11.4M
#
# BARRIER ONLY, deliberately. The question is bimodality, and that is a BP-track
# question: the ENet-trained networks sit at 35-40% bimodal precision against a
# 14.17% base rate because the L2 term smooths adjacent peaks together by
# construction, so scaling them tests nothing about multi-thermal structure. The
# ENet size question was already answered downward (h232 was the best of the
# entire sweep on both mae_dem and p99). Flip the array to 0-5%3 and restore the
# IDX/3 loss branch if we later want the ENet arm for a size figure.
#
# PREDICTION, recorded before the run so the result is interpretable either way.
# Bimodal recall across the downward sweep was 29.46 / 27.74 / 27.14 / 26.33% at
# 680 / 480 / 336 / 232 -- already flattening at the top, ~1.7 points per doubling.
# Straight-line extrapolation puts 11.4M near 32-34%. Meanwhile the mae_dem
# penalty at bimodal pixels was FLAT at 2.89-3.11x across that entire range, and
# bp_self_consistency.py showed that penalty is ~the label's own noise floor
# (ratio 0.98x). So the expected outcome is: recall creeps up a few points,
# curve-level agreement does not move at all.
#
# That is a positive result, not a null one. A deterministic map from 6 numbers
# cannot reproduce BP's noise-driven tie-breaking between near-degenerate optima
# at any width -- if capacity saturates here, the amortization ceiling stops being
# an argument and becomes a measured number, and the case for a distribution
# head rests on evidence instead of assumption.

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

WIDTHS=(960 1360 1920)
HIDDEN=${WIDTHS[$SLURM_ARRAY_TASK_ID]}
LOSS=barrier
DS=lp_AIA_hofdeconv_full_DS

echo "run: mlp6 hidden=$HIDDEN loss=$LOSS data=$DATA_ROOT/$DS"

cd "$REPO_DIR"
source env.sh
echo "### git HEAD: $(git rev-parse --short HEAD)   (pull before submitting!)"

# --warmup_steps 3000 is not optional. train_scaled's default of 500 killed
# cnn+enet in 15091457 and mlp6 h480 enet in 15185224_9, both as an identically
# zero prediction at init. Wider models have a larger pre-activation range at
# init, so this arm is the most exposed of any run so far.
python3 -u experiments/train_scaled.py \
  --root "$DATA_ROOT/$DS" \
  --variant mlp6 \
  --hidden "$HIDDEN" \
  --loss "$LOSS" \
  --epochs "${EPOCHS:-40}" \
  --batch_blocks "${BATCH_BLOCKS:-16}" \
  --num_workers 10 \
  --warmup_steps "${WARMUP_STEPS:-3000}" \
  --out_dir "${OUT_DIR:-output/experiments/sweep}" \
  $EXTRA_ARGS
