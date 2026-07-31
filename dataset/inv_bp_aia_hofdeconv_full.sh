#!/bin/bash
#SBATCH --job-name=inv_hof_%a
#SBATCH --output=logs/inv_hof/inv_hof_%A_%a.log
#SBATCH --error=logs/inv_hof/inv_hof_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_41_general
#SBATCH --mail-user=vp2435@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --comment="preemption=yes;requeue=true"

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get job list file from environment variable or use default
JOB_LIST_FILE=${JOB_LIST_FILE:-$DATASET_DIR/jobs_bp_aia_hofdeconv_full.txt}

# read job parameters from job list file
job_params=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $JOB_LIST_FILE)

# extract output file from job parameters (second argument)
output_file=$(echo $job_params | awk '{print $2}')

# skip if already done
if [ -f "$output_file" ]; then
  echo "output $output_file already exists, skipping $SLURM_ARRAY_TASK_ID"
  exit 0
fi

module purge

REPO_DIR="$(dirname "$DATASET_DIR")"
OVERLAY=${OVERLAY:-/scratch/$USER/workspace/dem/overlay-25GB-500K.ext3}

# Two supported environments: the singularity overlay (vp2435's setup) or the uv venv
# at the repo root (hsr3649's). Pick whichever this user actually has.
if [ -f "$OVERLAY" ]; then
  echo "environment: singularity overlay $OVERLAY"
  run_job() {
    singularity exec \
      --overlay "$OVERLAY":ro \
      $CONTAINER_IMAGES_FOLDER/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
      /bin/bash -c "source /ext3/env.sh; export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; conda activate dem_new; \
      cd $DATASET_DIR && python3 -u fullBP.py $job_params"
  }
elif [ -f "$REPO_DIR/env.sh" ]; then
  echo "environment: uv venv via $REPO_DIR/env.sh"
  run_job() {
    ( cd "$REPO_DIR" && source env.sh && python3 -u dataset/fullBP.py $job_params )
  }
else
  echo "ERROR: no usable environment (no overlay at $OVERLAY, no $REPO_DIR/env.sh)"
  exit 1
fi

max_attempts=3
attempt=1
while [ $attempt -le $max_attempts ]; do
  echo "processing job $SLURM_ARRAY_TASK_ID: $job_params (attempt $attempt/$max_attempts)"
  run_job && break
  echo "attempt $attempt failed; retrying in 60s..."
  sleep 60
  attempt=$((attempt+1))
done

if [ $attempt -gt $max_attempts ]; then
  echo "ERROR: job $SLURM_ARRAY_TASK_ID failed after $max_attempts attempts."
  exit 1
fi

# chained batch submission — BATCH_SIZE is preserved across batches
BATCH_SIZE=${BATCH_SIZE:-1000}
if [ "$SLURM_ARRAY_TASK_ID" -eq "${BATCH_END:-999999}" ] && [ "${NEXT_START:--1}" -ge 0 ]; then
  next_end=$((NEXT_START + BATCH_SIZE - 1))
  if [ "$next_end" -ge "$TOTAL_JOBS" ]; then
    next_end=$((TOTAL_JOBS - 1))
  fi
  batch_num=$((NEXT_START / BATCH_SIZE))
  echo "launching next batch: tasks $NEXT_START-$next_end"
  next_next_start=$((next_end + 1))
  if [ "$next_next_start" -ge "$TOTAL_JOBS" ]; then
    next_next_start=-1
  fi
  sbatch --array=$NEXT_START-$next_end \
    --job-name=${BASE_JOB_NAME}_b${batch_num} \
    --export=JOB_LIST_FILE=$JOB_LIST_FILE,BATCH_END=$next_end,NEXT_START=$next_next_start,BATCH_SIZE=$BATCH_SIZE,SLURM_SCRIPT=$SLURM_SCRIPT,TOTAL_JOBS=$TOTAL_JOBS,BASE_JOB_NAME=$BASE_JOB_NAME \
    $SLURM_SCRIPT
fi
