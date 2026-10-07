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
  local prev="" ids=()
  for _ in $(seq 1 "$segs"); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --job-name="fu_$name" \
           --export="$env" experiments/job_train_followup.sbatch)
    ids+=("$prev")
  done
  local ck=$OUT/$tag.pt evals
  if [ "$ev" = gpu ]; then
    evals=$(sbatch --parsable --dependency=afterok:$prev --job-name="fe_$name" \
            --export=ALL,TRACK=$track,MODEL_PATH=$ck,TAG=fu_$name experiments/job_eval_followup_gpu.sbatch)
  else
    local d a
    d=$(sbatch --parsable --dependency=afterok:$prev --job-name="fd_$name" \
        --export=ALL,TRACK=$track,MODEL_PATH=$ck,TAG=fu_$name experiments/job_diagnose_bright_failures.sbatch)
    a=$(sbatch --parsable --dependency=afterok:$prev --job-name="fa_$name" \
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
true
