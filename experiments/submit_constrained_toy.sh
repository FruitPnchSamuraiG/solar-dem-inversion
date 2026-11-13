#!/bin/bash
# Network toy: one run per unconstrained stand-in for the solver's linear program, all on
# the same fixed pixel set (experiments/job_constrained_toy.sbatch writes it first).
#   bash experiments/submit_constrained_toy.sh                      # six forms from scratch
#   MODE=warm bash experiments/submit_constrained_toy.sh quad100 ramp al alnet
#   MODE=warm PIXELS=all bash experiments/submit_constrained_toy.sh quad100 ramp al alnet
# MODE=warm starts every run from the trained quadratic-penalty model INIT (the method of
# multipliers' first iteration) at the weight it was trained with, 100. PIXELS=all trains
# on infeasible pixels too. CPUS and EPOCHS apply to every run; cpu_short allows 32 CPUs
# and 120 GB per user. Outputs are net_<run>[_warm][all].json.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${MODE:-scratch}
PIXELS=${PIXELS:-feasible}
INIT=${INIT:-output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt}
declare -A RUN=(
  [quad100]="quad|--mu 100"
  [ramp]="quad_ramp|--mu 1 --mu-end 10000"
  [price]="price|--eps 1 --eps-end 0.001"
  [huber]="huber|--kappa 64 --eps 1 --eps-end 0.001"
  [al]="al|--mu 100 --al-warm 20 --outer 10"
  [alnet]="alnet|--mu 100 --al-warm 20 --outer 10"
  [alfix]="al|--mu 100 --mu-max 100 --al-warm 20 --outer 10"
  # No mu chosen: the method of multipliers with the published primal-dual-learning rule (Park &
  # Van Hentenryck 2023: mu x10 when the violation measure fails to fall below 0.8 of its last value),
  # started from a small mu, stopping once 99% of pixels are within 0.1 half-widths of their band.
  [sched0.1]="al|--mu 0.1 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10"
  [sched1]="al|--mu 1 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10"
  [sched10]="al|--mu 10 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10"
  [sched1tight]="al|--mu 1 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.01 --al-warm 10 --outer 10"
  # The same rule made network-aware: a round ends when the training loss has settled (fell
  # under 2% over 10 epochs), and mu stops rising once a 10x raise cut the measure by under 2x.
  [settle0.1]="al|--mu 0.1 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10 --outer-mode settled --floor-gain 2"
  [settle1]="al|--mu 1 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10 --outer-mode settled --floor-gain 2"
  [settle10]="al|--mu 10 --mu-growth 10 --tau 0.8 --mu-max 1e6 --measure p99 --stop-viol 0.1 --al-warm 10 --outer 10 --outer-mode settled --floor-gain 2"
  # The method of multipliers with mu held fixed (no schedule): for convex problems it converges
  # for any fixed mu > 0, mu sets only the speed (Rockafellar 1976). Same answer from 1, 10, 100?
  [mm1]="al|--mu 1 --mu-max 1 --measure p99 --al-warm 10 --outer 10 --outer-mode settled"
  [mm10]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled"
  [mm100]="al|--mu 100 --mu-max 100 --measure p99 --al-warm 10 --outer 10 --outer-mode settled"
  # Training quality on the method as it stands (multipliers, mu 10, settled rounds), one change each:
  [tq_lr1e-3]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --lr 1e-3"
  [tq_lr1e-4]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --lr 1e-4"
  [tq_noclip]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --clip 0"
  [tq_muon]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer muon --muon-lr 0.02"
  [tq_muonrms]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer muon --muon-lr 3e-4 --muon-adjust match_rms_adamw"
  [tq_adamw]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer adamw"
  [tq_nadam]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer nadam"
  [tq_radam]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer radam"
  [tq_adafactor]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer adafactor"
  [tq_lion]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --optimizer lion"
  [tq_batch16k]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --batch 16384 --lr 6e-4"
  [tq_ema]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --ema 0.999"
  [tq_w680]="al|--mu 10 --mu-max 10 --measure p99 --al-warm 10 --outer 10 --outer-mode settled --hidden 680"
)
SUFFIX=""
if [ "$MODE" = warm ]; then
  CPUS=${CPUS:-8}
  COMMON="--epochs ${EPOCHS:-150} --pixels $PIXELS --init-from $INIT --lr ${LR:-1e-4} ${EXTRA:-}"
  RUN[ramp]="quad_ramp|--mu 100 --mu-end 10000"
  # the trained model already sits near violation q x 0.1 half-widths on its dear channels
  RUN[price]="price|--eps 0.1 --eps-end 0.001"
  RUN[huber]="huber|--kappa 64 --eps 0.1 --eps-end 0.001"
  RUN[al]="al|--mu 100 --al-warm 10 --outer 10"
  RUN[alnet]="alnet|--mu 100 --al-warm 10 --outer 10"
  # multipliers only: the weight stays at the trained 100 (no growth rule)
  RUN[alfix]="al|--mu 100 --mu-max 100 --al-warm 10 --outer 10"
  SUFFIX="_warm"
else
  CPUS=${CPUS:-5}
  COMMON="--epochs ${EPOCHS:-200} --pixels $PIXELS ${EXTRA:-}"
fi
if [ "$PIXELS" = all ]; then SUFFIX="${SUFFIX:-_}all"; fi
SUFFIX="$SUFFIX${TAG:-}"
picks=("$@")
if [ ${#picks[@]} -eq 0 ]; then picks=(quad100 ramp price huber al alnet); fi
for name in "${picks[@]}"; do
  spec=${RUN[$name]:?unknown run $name}
  form=${spec%%|*}
  args=${spec#*|}
  id=$(sbatch --parsable --job-name="ctn_${name}${SUFFIX}" --cpus-per-task="$CPUS" --mem="${MEM:-20G}" --time="${TIME:-03:00:00}" \
       --export=ALL,FORM="$form",NAME="${name}${SUFFIX}",DATA="${DATA:-toy}",ARGS="$COMMON $args" \
       experiments/job_constrained_toy_net.sbatch)
  echo "${name}${SUFFIX} $id"
done
