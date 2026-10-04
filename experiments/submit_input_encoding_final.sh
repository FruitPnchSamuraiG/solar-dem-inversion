#!/bin/bash
# sqrt + Fourier-feature input on the final model (h680, fresh pixels), both
# tracks, each with its baseline's warmup; the bright-failure diagnostic and
# AIA fit follow each run (TAG=sqrtff12_resample_h680).
#
#   bash experiments/submit_input_encoding_final.sh
set -euo pipefail
mkdir -p logs/scaled logs/eval
for TRACK in bp enet; do
  case $TRACK in
    bp)   L=barrier; WU=500 ;;
    enet) L=enet;    WU=3000 ;;
  esac
  CK=output/experiments/input_encoding/scaled_mlp6_${L}_h680_sqrt_ff12_resample.pt
  t=$(sbatch --parsable --export=ALL,TRACK=$TRACK,HIDDEN=680,RESAMPLE=1,WARMUP=$WU \
      experiments/job_train_input_encoding.sbatch)
  d=$(sbatch --parsable --dependency=afterok:$t \
      --export=ALL,TRACK=$TRACK,MODEL_PATH=$CK,TAG=sqrtff12_resample_h680 \
      experiments/job_diagnose_bright_failures.sbatch)
  a=$(sbatch --parsable --dependency=afterok:$t \
      --export=ALL,TRACK=$TRACK,LF_MODEL=$CK,TAG=sqrtff12_resample_h680 \
      experiments/job_aia_fit_by_brightness.sbatch)
  echo "$TRACK: train $t -> diagnostic $d, aia fit $a"
done
