#!/bin/bash
#SBATCH --job-name=dem_xclass
#SBATCH --output=logs/eval/xclass_%j.log
#SBATCH --error=logs/eval/xclass_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL

# One date outside every split, end to end: the X2.1 flare of 6 September 2011, three
# years before our 2014-2015 data and larger than any flare in the test window. Runs
# both solvers exactly as the reference runs (Hofmeister deconvolution; BP with its
# retries, Elastic Net alpha 0.001 without), exports our final models and the
# supervised ones on the same grid, scores them against each solver and renders the
# viewer images. STAMP and SRC choose another date (with a pointing table for it).
#   sbatch experiments/job_xclass_date.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/dem}"
source env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1
STAMP=${STAMP:-20110906_2217}
SRC=${SRC:-$SCRATCH/dem/data/$STAMP}
POINTING=${POINTING:-dataset/aia_pointing_20110906.ecsv}
OUT=$SCRATCH/dem/xclass/$STAMP
mkdir -p "$OUT/exports" "$OUT/assets" output/experiments/diagnostics logs/eval
BP_LF=output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt
EN_LF=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
COMMON=(--noisy 0 --deconvolve hofmeister --errorfn full --notrunc --extendto8 --decimate 2
        --parallel "$SLURM_CPUS_PER_TASK" --pointing_file "$POINTING")
[ -s "$OUT/bp_solver.npz" ] || python3 -u dataset/fullBP.py "$SRC" "$OUT/bp_solver.npz" "${COMMON[@]}" --fitfn lp
[ -s "$OUT/enet_solver.npz" ] || python3 -u dataset/fullBP.py "$SRC" "$OUT/enet_solver.npz" "${COMMON[@]}" \
  --zerochill --fitfn elasticnet --fitlinearalpha 0.001 --fitlinearl1ratio 0.5

for TRACK in bp enet; do
  LF=$BP_LF; SUP10=paper_model/checkpoints/BP_5_Wide_10.1M.pth; SUP1=eval_and_enet_specs/checkpoints/model_best_bp.pth
  if [ $TRACK = enet ]; then LF=$EN_LF; SUP10=paper_model/checkpoints/EN_2_Wide_10.1M.pth; SUP1=eval_and_enet_specs/checkpoints/model_best_en.pth; fi
  E=$OUT/exports/$TRACK
  python3 -u experiments/export_mlp6_full_disk.py --model "$LF" --input_npz "$OUT/${TRACK}_solver.npz" --output "${E}_ours.npz"
  python3 -u experiments/export_supervised_full_disk.py --model "$SUP10" --input "${E}_ours.npz" --output "${E}_sup10m.npz"
  python3 -u experiments/export_supervised_full_disk.py --model "$SUP1" --input "${E}_ours.npz" --output "${E}_sup1m.npz"
  python3 -u experiments/date_metrics.py --solver "$OUT/${TRACK}_solver.npz" \
    --exports ours="${E}_ours.npz" sup10m="${E}_sup10m.npz" sup1m="${E}_sup1m.npz" \
    --output "output/experiments/diagnostics/${TRACK}_date_metrics_${STAMP}.json"
  python3 -u experiments/render_full_disk_visuals.py --input "${E}_ours.npz" --source reference --output "$OUT/assets/${TRACK}_solver"
  for M in ours sup10m sup1m; do
    python3 -u experiments/render_full_disk_visuals.py --input "${E}_${M}.npz" --source model --output "$OUT/assets/${TRACK}_$M"
  done
done
echo "done: $OUT"
