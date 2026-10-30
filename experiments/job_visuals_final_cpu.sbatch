#!/bin/bash
#SBATCH --job-name=dem_vis_final
#SBATCH --output=logs/visuals/final_cpu_%j.log
#SBATCH --error=logs/visuals/final_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_41_general
#SBATCH --mail-type=FAIL

# Final label-free models for the public viewer: one 360k MLP per track, both
# with the square-root + 12 Fourier frequency input. BP (2026-10-13) is the
# stiff-band model (barrier weight 100) trained 120 epochs and continued for 120
# more; ENet (2026-10-04) is unchanged. Inputs are the same full-disk solver
# files the earlier viewer exports used, so the solver and supervised images are
# reused unchanged. The run name bp_mlp6_h336 is kept so the viewer needs no change.
# BP_MODEL, BP_RUN, TRACKS and NO_STAGE=1 render a candidate BP model alongside.
#   mkdir -p logs/visuals
#   sbatch experiments/job_visuals_final_cpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the DEM repository root}"
source env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export VIS_ROOT="$SCRATCH/dem/visuals/matched_enet_alpha0p001"
mapfile -t DATES < <(grep -v '^#' experiments/visualizer_dates.txt | sed '/^$/d')
test "${#DATES[@]}" -gt 0

setup_track() {
  if [ "$1" = bp ]; then
    MODEL=${BP_MODEL:-output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt}
    INPUT_DIR="$SCRATCH/dem/data/lp_AIA_hofdeconv_full"
    RUN=${BP_RUN:-bp_mlp6_h336}
  else
    MODEL=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
    INPUT_DIR="$VIS_ROOT/references"
    RUN=enet_mlp6_h336_sqrtff
  fi
}

# Check every input before spending time on the first export.
for track in ${TRACKS:-bp enet}; do
  setup_track "$track"
  test -r "$MODEL"
  for stamp in "${DATES[@]}"; do
    [[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]
    test -r "$INPUT_DIR/$stamp.npz"
  done
done

for track in ${TRACKS:-bp enet}; do
  setup_track "$track"
  echo "Model ($track): $MODEL"
  for stamp in "${DATES[@]}"; do
    echo "Label-free $track: $stamp"
    export_file="$VIS_ROOT/exports/$RUN/dem_$stamp.npz"
    python3 -u experiments/export_mlp6_full_disk.py \
      --model "$MODEL" --input_npz "$INPUT_DIR/$stamp.npz" --output "$export_file"
    # Sanity check against the solver on this disk (finite pixels only).
    python3 - "$export_file" <<'EOF'
import sys
import numpy as np
with np.load(sys.argv[1]) as d:
    dem, ref = d["dem"], d["reference_dem"]
ok = np.isfinite(ref).all(0) & np.isfinite(dem).all(0)
mse = float(np.mean((dem[:, ok] - ref[:, ok]) ** 2))
em = float(dem[:, ok].sum() / ref[:, ok].sum())
print(f"  check: {ok.sum():,} pixels, DEM MSE vs solver {mse:.4g}, EM ratio {em:.3f}")
EOF
    python3 -u experiments/render_full_disk_visuals.py \
      --input "$export_file" --source model \
      --output "$VIS_ROOT/assets/$RUN/$stamp"
  done
done
# NO_STAGE=1 renders assets only, leaving the preview as it is (a candidate run).
if [ "${NO_STAGE:-0}" != 1 ]; then bash experiments/stage_visualizer_preview.sh; fi
echo "Completed final-model viewer assets under $VIS_ROOT/assets"
