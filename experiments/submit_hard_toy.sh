#!/bin/bash
# Toy runs on a small pixel set (experiments/hard_toy_set.py writes them): the 1.5M network of the
# final model trained on only those pixels, one setting changed per run from the reference (method of
# multipliers, mu 10, Adam lr 3e-4, a multiplier round every 20 epochs). Every run records 12 pixels'
# multipliers, band gaps and DEMs each epoch (--track). CPU jobs on cpu_short.
#   bash experiments/submit_hard_toy.sh                          # rate runs on the hard set (toy_hard)
#   DATA=toy_mix bash experiments/submit_hard_toy.sh             # the same on 10k hard + 10k easy
#   DATA=toy_mixall bash experiments/submit_hard_toy.sh bpflags tight trunc1 trunc3 trunc10
# The last line: every pixel including those BP widened or could not solve; BP's flags as in the final
# model (bpflags) against no flags at all, plain (tight) or with the penalty capped at 1 / 3 / 10
# half-widths (truncN). AFTER=<jobid> waits for that job. Outputs net_<set>_<run>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-toy_hard}
SET=${DATA#toy_}
COMMON="--hidden 680 --batch 512 --epochs ${EPOCHS:-600} --al-warm 20 --track 12 --score-every 5"
REF="--mu 10 --mu-max 10 --lr 3e-4 --outer 20"
declare -A RUN=(
  [ref]="al|$REF"
  [mu1]="al|--mu 1 --mu-max 1 --lr 3e-4 --outer 20"
  [mu100]="al|--mu 100 --mu-max 100 --lr 3e-4 --outer 20"
  [lr1e-4]="al|--mu 10 --mu-max 10 --lr 1e-4 --outer 20"
  [lr1e-3]="al|--mu 10 --mu-max 10 --lr 1e-3 --outer 20"
  [round5]="al|--mu 10 --mu-max 10 --lr 3e-4 --outer 5"
  [round50]="al|--mu 10 --mu-max 10 --lr 3e-4 --outer 50"
  [nofines]="quad|--mu 10 --lr 3e-4"
  [bpflags]="al|$REF --pixels all --relax-bp"
  [tight]="al|$REF --pixels all"
  [trunc1]="al|$REF --pixels all --trunc 1"
  [trunc3]="al|$REF --pixels all --trunc 3"
  [trunc10]="al|$REF --pixels all --trunc 10"
)
picks=("$@")
if [ ${#picks[@]} -eq 0 ]; then picks=(ref mu1 mu100 lr1e-4 lr1e-3 round5 round50 nofines); fi
dep=()
if [ -n "${AFTER:-}" ]; then dep=(--dependency=afterok:$AFTER); fi
for name in "${picks[@]}"; do
  spec=${RUN[$name]:?unknown run $name}
  id=$(sbatch --parsable "${dep[@]}" --job-name="${SET}_${name}" --account=torch_pr_41_general --cpus-per-task="${CPUS:-4}" \
       --mem="${MEM:-12G}" --time="${TIME:-06:00:00}" \
       --export=ALL,FORM="${spec%%|*}",NAME="${SET}_${name}",DATA="$DATA",ARGS="$COMMON ${spec#*|}" \
       experiments/job_constrained_toy_net.sbatch)
  echo "${SET}_${name} $id"
done
