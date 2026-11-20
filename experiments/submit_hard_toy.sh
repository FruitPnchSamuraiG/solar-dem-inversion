#!/bin/bash
# Hard-pixel toy: ~10k bright pixels BP solves at the tight band (experiments/hard_toy_set.py writes
# toy_hard_*), the 1.5M network of the final model trained on only those under the method of
# multipliers, one setting changed per run from the reference (mu 10, Adam lr 3e-4, a multiplier
# round every 20 epochs). Every run records 12 pixels' multipliers, band gaps and DEMs each epoch
# (--track) for figures. CPU jobs on cpu_short.
#   bash experiments/submit_hard_toy.sh            # all runs
#   bash experiments/submit_hard_toy.sh ref mu1    # some
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--hidden 680 --batch 512 --epochs ${EPOCHS:-600} --al-warm 20 --track 12 --score-every 5"
declare -A RUN=(
  [ref]="al|--mu 10 --mu-max 10 --lr 3e-4 --outer 20"
  [mu1]="al|--mu 1 --mu-max 1 --lr 3e-4 --outer 20"
  [mu100]="al|--mu 100 --mu-max 100 --lr 3e-4 --outer 20"
  [lr1e-4]="al|--mu 10 --mu-max 10 --lr 1e-4 --outer 20"
  [lr1e-3]="al|--mu 10 --mu-max 10 --lr 1e-3 --outer 20"
  [round5]="al|--mu 10 --mu-max 10 --lr 3e-4 --outer 5"
  [round50]="al|--mu 10 --mu-max 10 --lr 3e-4 --outer 50"
  [nofines]="quad|--mu 10 --lr 3e-4"
)
picks=("$@")
if [ ${#picks[@]} -eq 0 ]; then picks=(ref mu1 mu100 lr1e-4 lr1e-3 round5 round50 nofines); fi
for name in "${picks[@]}"; do
  spec=${RUN[$name]:?unknown run $name}
  id=$(sbatch --parsable --job-name="hard_${name}" --account=torch_pr_41_general --cpus-per-task="${CPUS:-4}" \
       --mem="${MEM:-12G}" --time="${TIME:-06:00:00}" \
       --export=ALL,FORM="${spec%%|*}",NAME="hard_${name}",DATA=toy_hard,ARGS="$COMMON ${spec#*|}" \
       experiments/job_constrained_toy_net.sbatch)
  echo "hard_${name} $id"
done
