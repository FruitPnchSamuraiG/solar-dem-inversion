#!/bin/bash
# Retrain both production models with fresh pixels every epoch, then run the
# bright-failure diagnostic and the AIA fit on each once its training succeeds.
# Together those two reproduce the full results table (DEM MSE, EM error, W1,
# AIA MAE/MSE) and the flare-core analysis. Run from the repo root on Torch:
#
#   bash experiments/submit_resample.sh
set -euo pipefail
mkdir -p logs/scaled logs/eval
for TRACK in bp enet; do
  case $TRACK in
    bp)   CKPT=output/experiments/resample/scaled_mlp6_barrier_h232_resample.pt ;;
    enet) CKPT=output/experiments/resample/scaled_mlp6_enet_h232_resample.pt ;;
  esac
  train=$(sbatch --parsable --export=ALL,TRACK=$TRACK experiments/job_train_resample.sbatch)
  diag=$(sbatch --parsable --dependency=afterok:$train \
         --export=ALL,TRACK=$TRACK,MODEL_PATH=$CKPT,TAG=resample \
         experiments/job_diagnose_bright_failures.sbatch)
  aia=$(sbatch --parsable --dependency=afterok:$train \
        --export=ALL,TRACK=$TRACK,LF_MODEL=$CKPT,TAG=resample \
        experiments/job_aia_fit_by_brightness.sbatch)
  echo "$TRACK: train $train -> diagnostic $diag, aia fit $aia"
done
