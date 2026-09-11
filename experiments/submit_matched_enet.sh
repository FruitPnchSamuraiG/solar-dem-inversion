#!/bin/bash
# Submit the alpha=0.001 ENet retrain and its full shared-test evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_JOB=$(sbatch --parsable experiments/job_train_enet_alpha0p001.sbatch)
MODEL=output/experiments/matched_enet_alpha0p001/scaled_mlp6_enet_h232.pt
EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB" \
  --export="ALL,TRACK=enet,OUTPUT_TAG=alpha0p001,MODEL_OVERRIDE=$MODEL" \
  experiments/job_eval_full_paper_test.sbatch)

echo "ENet alpha=0.001 training job: $TRAIN_JOB"
echo "Dependent full shared-test evaluation job: $EVAL_JOB"
