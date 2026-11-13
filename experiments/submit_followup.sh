#!/bin/bash
# Follow-up runs from the 2026-10-06 meeting, each changing one thing on the
# square-root + Fourier, 336-wide, fresh-pixel recipe (both tracks):
#   bg1    oversample bright pixels: draw probability ~ clip(I, 1, 1000)
#          (I = max over channels of observed / Bright cutoff)
#   ep120  train three times longer (120 epochs, cosine over all 120)
#   10m    ~10M parameters (1800 wide)
# Each run is a chain of resumable <2 h GPU segments, then the full shared-test
# bright-failure diagnostic and AIA fit (on the GPU for the 10M models).
# Baselines for comparison: input_encoding/scaled_mlp6_{barrier,enet}_h336_sqrt_ff12_resample.pt.
#
#   bash experiments/submit_followup.sh bp_bg1 enet_bg1   # the named runs, in order
set -euo pipefail
mkdir -p logs/scaled logs/eval
OUT=output/experiments/followup
JOBS=logs/followup_jobs.txt

run() {  # name track segments eval(cpu|gpu) checkpoint-tag env...
  local name=$1 track=$2 segs=$3 ev=$4 tag=$5; shift 5
  local env="ALL,TRACK=$track"
  for kv in "$@"; do env+=",$kv"; done
  # AFTER=<job id>: the first segment waits for that job (to order chains).
  local prev="${AFTER:-}" ids=()
  for _ in $(seq 1 "$segs"); do
    # ACCOUNT spreads chains over the accounts we hold (each has its own GPU quota).
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --job-name="fu_$name" \
           ${ACCOUNT:+--account="$ACCOUNT"} \
           --export="$env" experiments/job_train_followup.sbatch)
    ids+=("$prev")
  done
  local ck=$OUT/$tag.pt evals
  if [ "$ev" = gpu ]; then
    evals=$(sbatch --parsable --dependency=afterok:$prev --job-name="fe_$name" ${ACCOUNT:+--account="$ACCOUNT"} \
            --export=ALL,TRACK=$track,MODEL_PATH=$ck,TAG=fu_$name experiments/job_eval_followup_gpu.sbatch)
  else
    local d a
    d=$(sbatch --parsable --dependency=afterok:$prev --job-name="fd_$name" ${ACCOUNT:+--account="$ACCOUNT"} \
        --export=ALL,TRACK=$track,MODEL_PATH=$ck,TAG=fu_$name experiments/job_diagnose_bright_failures.sbatch)
    a=$(sbatch --parsable --dependency=afterok:$prev --job-name="fa_$name" ${ACCOUNT:+--account="$ACCOUNT"} \
        --export=ALL,TRACK=$track,LF_MODEL=$ck,TAG=fu_$name experiments/job_aia_fit_by_brightness.sbatch)
    evals="$d,$a"
  fi
  echo "$name train ${ids[*]} eval $evals ckpt $ck" | tee -a "$JOBS"
}

sel=("$@")
if [ ${#sel[@]} -eq 0 ]; then echo "name the runs to submit (see the pick lines below)" >&2; exit 2; fi
pick() { local n=$1; for s in "${sel[@]}"; do [ "$s" = "$n" ] && return 0; done; return 1; }

echo "### $(date -Iseconds) git $(git rev-parse --short HEAD)" >> "$JOBS"
B=scaled_mlp6_barrier; E=scaled_mlp6_enet
pick bp_bg1     && run bp_bg1     bp   2 cpu "${B}_h336_sqrt_ff12_resample_bg1"   BRIGHT_GAMMA=1
pick enet_bg1   && run enet_bg1   enet 2 cpu "${E}_h336_sqrt_ff12_resample_bg1"   BRIGHT_GAMMA=1
pick bp_ep120   && run bp_ep120   bp   5 cpu "${B}_h336_sqrt_ff12_resample_ep120" EPOCHS=120 TAG_SUFFIX=_ep120
pick enet_ep120 && run enet_ep120 enet 5 cpu "${E}_h336_sqrt_ff12_resample_ep120" EPOCHS=120 TAG_SUFFIX=_ep120
pick bp_10m     && run bp_10m     bp   4 gpu "${B}_h1800_sqrt_ff12_resample"      HIDDEN=1800 TF32=1
pick enet_10m   && run enet_10m   enet 4 gpu "${E}_h1800_sqrt_ff12_resample"      HIDDEN=1800 TF32=1
# Added after experiments/objective_gap.py showed the network sits at its own
# objective's optimum and that optimum under-fits bright pixels: a stiffer band.
pick bp_mu10    && run bp_mu10    bp   2 cpu "${B}_h336_sqrt_ff12_resample_mu10"  MU=10 TAG_SUFFIX=_mu10
pick bp_mu100   && run bp_mu100   bp   2 cpu "${B}_h336_sqrt_ff12_resample_mu100" MU=100 TAG_SUFFIX=_mu100
# Chosen on the validation split (objective_gap.py --phase val): mu30 with the
# L1 weight / sqrt(brightness) gives the lowest Quiet error and near-best overall.
pick bp_mu30_ibr05 && run bp_mu30_ibr05 bp 2 cpu "${B}_h336_sqrt_ff12_resample_mu30_ibr0.5" MU=30 L1_BRIGHT_POWER=0.5 TAG_SUFFIX=_mu30_ibr0.5
# The meeting's training-side suggestions repeated on the stiff band (mu 100):
# on the original loss the per-pixel optimum already bounds them, so the open
# question is whether the network can follow the stiffer loss at flare cores.
pick bp_mu100_bg1   && run bp_mu100_bg1   bp 2 cpu "${B}_h336_sqrt_ff12_resample_bg1_mu100" MU=100 BRIGHT_GAMMA=1 TAG_SUFFIX=_mu100
pick bp_mu100_ep120 && run bp_mu100_ep120 bp 5 cpu "${B}_h336_sqrt_ff12_resample_mu100_ep120" MU=100 EPOCHS=120 TAG_SUFFIX=_mu100_ep120
pick bp_mu100_10m   && run bp_mu100_10m   bp 4 gpu "${B}_h1800_sqrt_ff12_resample_mu100" MU=100 HIDDEN=1800 TF32=1 TAG_SUFFIX=_mu100
# One encoding for both tracks: does log1p still tie sqrt + Fourier on the stiff band?
pick bp_log1p_mu100 && run bp_log1p_mu100 bp 2 cpu "${B}_h336_resample_mu100" MU=100 INPUT=log1p FOURIER=0 TAG_SUFFIX=_mu100
# Gradient clipping (norm 1) fires on every step (norms 20-500 on mu 100): does it
# blunt the rare flare-core gradients?
pick bp_mu100_noclip && run bp_mu100_noclip bp 2 cpu "${B}_h336_sqrt_ff12_resample_mu100_noclip" MU=100 CLIP=1e12 TAG_SUFFIX=_mu100_noclip
# 2026-10-13, overnight before the meeting. Band x100 trained 120 epochs was
# still improving at flare cores (network 47% above its loss's best answer
# there), so: train longer two ways, and the two untested knobs on this loss.
# Segments of 1.85 h hold ~42 epochs (158 s each): 120 epochs = 3 segments,
# 240 = 6; one spare each, which exits at once when training is complete.
# Queue waits dominate, so each chain goes to a different account.
#   ep120_cont  120 more epochs from the 120-epoch checkpoint (fresh schedule)
#   ep240       240 epochs from scratch, one cosine schedule
#   h680        1.43M parameters (between 409k and the 10M that blew up)
#   mu300       the next band weight up (best answer 1.80 vs 2.32 overall)
pick bp_mu100_ep120_cont && ACCOUNT=torch_pr_41_tandon_advanced run bp_mu100_ep120_cont bp 4 cpu "${B}_h336_sqrt_ff12_resample_mu100_ep120_cont" MU=100 EPOCHS=120 MAX_HOURS=1.85 TAG_SUFFIX=_mu100_ep120_cont "INIT_FROM=$OUT/${B}_h336_sqrt_ff12_resample_mu100_ep120.pt"
pick bp_mu100_ep240      && ACCOUNT=torch_pr_718_cilvr run bp_mu100_ep240 bp 7 cpu "${B}_h336_sqrt_ff12_resample_mu100_ep240" MU=100 EPOCHS=240 MAX_HOURS=1.85 TAG_SUFFIX=_mu100_ep240
pick bp_mu100_h680_ep120 && ACCOUNT=torch_pr_718_cds run bp_mu100_h680_ep120 bp 4 cpu "${B}_h680_sqrt_ff12_resample_mu100_ep120" MU=100 HIDDEN=680 EPOCHS=120 MAX_HOURS=1.85 TAG_SUFFIX=_mu100_ep120
pick bp_mu300_ep120      && ACCOUNT=torch_pr_41_tandon_advanced run bp_mu300_ep120 bp 4 cpu "${B}_h336_sqrt_ff12_resample_mu300_ep120" MU=300 EPOCHS=120 MAX_HOURS=1.85 TAG_SUFFIX=_mu300_ep120
# 2026-10-29, after the 21 Oct meeting. The worst pixels of the large models and the
# low end of the JPDFs are deconvolution-clamp pixels (a channel at the 1e-3 floor),
# which every run so far excluded from training, so the network never saw them.
# CLAMP_MODE=mask keeps them and frees only the clamped channels from the band.
# Each run repeats an existing one with that single change:
#   clampmask         = bp_mu100 (336 wide, 40 epochs)          production size
#   h680_clampmask    = bp_mu100_h680_ep120 (blew up, DEM MSE 897)
#   enet_clampmask    = the ElasticNet final (336 wide, 40 epochs)
#   enet_10m_clampmask = enet_10m (blew up, DEM MSE 2,062)
pick bp_clampmask       && ACCOUNT=torch_pr_41_cilvr   run bp_clampmask bp 2 cpu "${B}_h336_sqrt_ff12_resample_clampmask_mu100" MU=100 CLAMP_MODE=mask TAG_SUFFIX=_mu100
pick bp_h680_clampmask  && ACCOUNT=torch_pr_41_courant run bp_h680_clampmask bp 5 cpu "${B}_h680_sqrt_ff12_resample_clampmask_mu100_ep120" MU=100 HIDDEN=680 EPOCHS=120 MAX_HOURS=1.85 CLAMP_MODE=mask TAG_SUFFIX=_mu100_ep120
pick enet_clampmask     && ACCOUNT=torch_pr_718_cds    run enet_clampmask enet 2 cpu "${E}_h336_sqrt_ff12_resample_clampmask" CLAMP_MODE=mask
pick enet_10m_clampmask && ACCOUNT=torch_pr_41_general run enet_10m_clampmask enet 5 gpu "${E}_h1800_sqrt_ff12_resample_clampmask" HIDDEN=1800 TF32=1 CLAMP_MODE=mask
# Repeats of the 1.5M runs with and without clamp pixels: is the fall from DEM MSE 897 to 1.08
# the masking or chance? One flare-core input held 96.7% of the 897.
C="MU=100 HIDDEN=680 EPOCHS=120 MAX_HOURS=1.85"
pick bp_h680_clampmask_s1 && ACCOUNT=torch_pr_41_cilvr   run bp_h680_clampmask_s1 bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampmask_mu100_ep120_s1" $C CLAMP_MODE=mask SEED=1 TAG_SUFFIX=_mu100_ep120_s1
pick bp_h680_clampmask_s2 && ACCOUNT=torch_pr_41_cilvr   run bp_h680_clampmask_s2 bp 5 cpu "${B}_h680_sqrt_ff12_resample_clampmask_mu100_ep120_s2" $C CLAMP_MODE=mask SEED=2 TAG_SUFFIX=_mu100_ep120_s2
pick bp_h680_s1           && ACCOUNT=torch_pr_41_cilvr   run bp_h680_s1 bp 4 cpu "${B}_h680_sqrt_ff12_resample_mu100_ep120_s1" $C SEED=1 TAG_SUFFIX=_mu100_ep120_s1
pick bp_h680_s2           && ACCOUNT=torch_pr_41_cilvr   run bp_h680_s2 bp 5 cpu "${B}_h680_sqrt_ff12_resample_mu100_ep120_s2" $C SEED=2 TAG_SUFFIX=_mu100_ep120_s2
# The solver widens its band where no DEM fits 1.4 sigma; our loss did not, and on
# validation those ~8% of pixels carry most of the stiff band's cost
# (experiments/relaxed_band_screen.py: with the solver's own band, excess still in
# noise units, the best answer improves monotonically with mu). mu = 300 on the
# fixed band trained behind mu = 100; this repeats it with the relaxed band.
pick bp_mu300_relax     && ACCOUNT=torch_pr_718_cilvr  run bp_mu300_relax bp 4 cpu "${B}_h336_sqrt_ff12_resample_relax_mu300_ep120" MU=300 EPOCHS=120 MAX_HOURS=1.85 RELAX_BAND=1 TAG_SUFFIX=_mu300_ep120
# 2026-11-09, after the supervised paper models arrived (10.13M parameters, 5.33M of them on
# the path that predicts the DEM; the rest is the uncertainty head). All on the final recipe
# (clamped channels masked, mu = 100, 120 epochs), changing one thing each. Accounts by queue estimate
# on 9 Nov (sbatch --test-only): the pr_718 and tandon accounts were a week out.
#   h966 / h1299 / h1803   3.0M / 5.33M / 10.13M parameters (does size keep helping?)
#   mu30 / mu300           the band weight at the final size
#   relax                  the solver's widened band at the final size
# Epochs take ~150 s at every width (data-bound), so 120 epochs fit 3 segments plus a spare.
F="EPOCHS=120 MAX_HOURS=1.85 CLAMP_MODE=mask"
pick bp_h966_cm   && ACCOUNT=torch_pr_41_cilvr            run bp_h966_cm   bp 4 gpu "${B}_h966_sqrt_ff12_resample_clampmask_mu100_ep120"  $F MU=100 HIDDEN=966  TF32=1 TAG_SUFFIX=_mu100_ep120
pick bp_h1299_cm  && ACCOUNT=torch_pr_41_courant          run bp_h1299_cm  bp 4 gpu "${B}_h1299_sqrt_ff12_resample_clampmask_mu100_ep120" $F MU=100 HIDDEN=1299 TF32=1 TAG_SUFFIX=_mu100_ep120
pick bp_h1803_cm  && ACCOUNT=torch_pr_41_courant          run bp_h1803_cm  bp 4 gpu "${B}_h1803_sqrt_ff12_resample_clampmask_mu100_ep120" $F MU=100 HIDDEN=1803 TF32=1 TAG_SUFFIX=_mu100_ep120
pick bp_h680_cm_mu30  && ACCOUNT=torch_pr_41_general run bp_h680_cm_mu30  bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampmask_mu30_ep120"  $F MU=30  HIDDEN=680 TAG_SUFFIX=_mu30_ep120
pick bp_h680_cm_mu300 && ACCOUNT=torch_pr_41_cilvr         run bp_h680_cm_mu300 bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampmask_mu300_ep120" $F MU=300 HIDDEN=680 TAG_SUFFIX=_mu300_ep120
pick bp_h680_cm_relax && ACCOUNT=torch_pr_41_general       run bp_h680_cm_relax bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampmask_relax_mu100_ep120" $F MU=100 HIDDEN=680 RELAX_BAND=1 TAG_SUFFIX=_mu100_ep120
# 2026-11-10: is the 3M gain at the tallest peaks real (two more seeds), and did the 5.3M and 10M
# runs fail because of the learning rate tuned on small models (peak 1e-4 instead of 3e-4)?
pick bp_h966_cm_s1  && ACCOUNT=torch_pr_41_cilvr   run bp_h966_cm_s1  bp 4 gpu "${B}_h966_sqrt_ff12_resample_clampmask_mu100_ep120_s1" $F MU=100 HIDDEN=966 TF32=1 SEED=1 TAG_SUFFIX=_mu100_ep120_s1
pick bp_h966_cm_s2  && ACCOUNT=torch_pr_41_general run bp_h966_cm_s2  bp 4 gpu "${B}_h966_sqrt_ff12_resample_clampmask_mu100_ep120_s2" $F MU=100 HIDDEN=966 TF32=1 SEED=2 TAG_SUFFIX=_mu100_ep120_s2
pick bp_h1299_cm_lr1 && ACCOUNT=torch_pr_41_courant run bp_h1299_cm_lr1 bp 4 gpu "${B}_h1299_sqrt_ff12_resample_clampmask_mu100_ep120_lr1e-4" $F MU=100 HIDDEN=1299 TF32=1 LR=1e-4 TAG_SUFFIX=_mu100_ep120_lr1e-4
pick bp_h1803_cm_lr1 && ACCOUNT=torch_pr_41_courant run bp_h1803_cm_lr1 bp 4 gpu "${B}_h1803_sqrt_ff12_resample_clampmask_mu100_ep120_lr1e-4" $F MU=100 HIDDEN=1803 TF32=1 LR=1e-4 TAG_SUFFIX=_mu100_ep120_lr1e-4
# 2026-11-12 advisor meeting: a zeroed channel keeps its band around zero (as the solver does)
# instead of being freed; and the band weight set per pixel and channel from the input with a
# tolerance lowered during training (src/losses.py:band_loss_batch). Final recipe otherwise.
FK="EPOCHS=120 MAX_HOURS=1.85 CLAMP_MODE=keep"
pick bp_h680_keep       && ACCOUNT=${ACCOUNT:-torch_pr_41_courant} run bp_h680_keep       bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_mu100_ep120"     $FK MU=100 HIDDEN=680 TAG_SUFFIX=_mu100_ep120
pick bp_h680_keep_price && ACCOUNT=${ACCOUNT:-torch_pr_41_courant} run bp_h680_keep_price bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_price_ep120"     $FK BAND_FORM=price HIDDEN=680 TAG_SUFFIX=_ep120
pick bp_h680_keep_huber && ACCOUNT=${ACCOUNT:-torch_pr_41_courant} run bp_h680_keep_huber bp 4 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_huber_k64_ep120" $FK BAND_FORM=huber HIDDEN=680 TAG_SUFFIX=_ep120
# Warm start from the final model, 60 more epochs with zeroed channels keeping their band: today's
# loss against the method of multipliers (copies every 20 epochs), the full-scale version of the
# constrained toy's warm-start pair (results/plots/26_constrained_theory_20261112).
FINAL=output/experiments/followup/${B}_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt
FW="EPOCHS=60 MAX_HOURS=1.85 CLAMP_MODE=keep MU=100 HIDDEN=680 LR=1e-4 INIT_FROM=$FINAL"
pick bp_h680_keep_warm && ACCOUNT=${ACCOUNT:-torch_pr_41_general} run bp_h680_keep_warm bp 3 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_mu100_warm_ep60" $FW TAG_SUFFIX=_mu100_warm_ep60
pick bp_h680_keep_al   && ACCOUNT=${ACCOUNT:-torch_pr_41_general} run bp_h680_keep_al   bp 3 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_al100_warm_ep60"   $FW AL_MU=100 AL_EVERY=20 AL_MAX=3 TAG_SUFFIX=_warm_ep60
# No mu chosen (advisor, 12 Nov): the method of multipliers from mu = 1 with the textbook rule
# (mu x10 when the violation measure fails to fall below 0.8 of its last value), a multiplier copy
# every 10 epochs, stop once 99% of solver-feasible validation pixels are within 0.1 half-widths,
# multipliers capped at 256 x the price scale (elastic at infeasible pixels). Zero band kept.
FS="EPOCHS=120 MAX_HOURS=1.85 CLAMP_MODE=keep HIDDEN=680"
pick bp_h680_keep_sched && ACCOUNT=${ACCOUNT:-torch_pr_41_general} run bp_h680_keep_sched bp 6 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_al1_sched_ep120" $FS AL_MU=1 AL_RULE=adaptive AL_EVERY=10 AL_MAX=8 AL_STOP=0.1 AL_CAP=256 TAG_SUFFIX=_ep120
# The network-aware version (constrained toy, 12 Nov): a round ends when the training loss has
# settled (fell under 2% over 10 epochs), and mu stops rising once a 10x raise cut the violation
# measure by under 2x (the raise is undone). From mu = 1, no mu chosen; 200 epochs for the rounds.
pick bp_h680_keep_settle && ACCOUNT=${ACCOUNT:-torch_pr_41_general} run bp_h680_keep_settle bp 8 cpu "${B}_h680_sqrt_ff12_resample_clampkeep_al1_sched_settled_ep200" EPOCHS=200 MAX_HOURS=1.85 CLAMP_MODE=keep HIDDEN=680 AL_MU=1 AL_RULE=adaptive AL_OUTER=settled AL_FLOOR=2 AL_EVERY=10 AL_MAX=12 AL_STOP=0.1 AL_CAP=256 TAG_SUFFIX=_settled_ep200
true
