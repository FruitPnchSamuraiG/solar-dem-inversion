#!/bin/bash
# Create a Torch-local static preview of solver, label-free and supervised full-disk assets.
# Serve with the command printed at the end, then tunnel port 8000 over SSH.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIS_ROOT="${VIS_ROOT:-${SCRATCH:?SCRATCH must be set}/dem/visuals}"
PREVIEW="$VIS_ROOT/preview"
WEBAPP="$PREVIEW/webapp"
RESULTS="$PREVIEW/results/test"

# results/test holds only symlinks into assets/, so clearing it drops runs that
# are no longer shown (e.g. the superseded h232 models) without touching images.
rm -rf "$RESULTS"
mkdir -p "$WEBAPP" "$RESULTS"
cp "$REPO_DIR/student_package/visuals_pipeline/webapp/compare.html" "$WEBAPP/compare.html"
cp "$REPO_DIR/student_package/visuals_pipeline/webapp/compare.js" "$WEBAPP/compare.js"

entries=()
RUNS=(bp_solver bp_mlp6_h336 bp_supervised enet_solver enet_mlp6_h336_sqrtff enet_supervised)
# Only the dates in visualizer_dates.txt are shown, whatever else is rendered.
mapfile -t DATES < <(grep -v '^#' "$REPO_DIR/experiments/visualizer_dates.txt" | sed '/^$/d')
for run in "${RUNS[@]}"; do
  for stamp in "${DATES[@]}"; do
    source="$VIS_ROOT/assets/$run/$stamp"
    if [ ! -d "$source" ]; then
      echo "not rendered, skipped: $run/$stamp" >&2
      continue
    fi
    name="$run/$stamp"
    target="$RESULTS/$name"
    mkdir -p "$(dirname "$target")"
    ln -sfn "$source" "$target"
    entries+=("$name")
  done
done
if [ "${#entries[@]}" -eq 0 ]; then
  echo "no rendered assets found under $VIS_ROOT/assets" >&2
  exit 1
fi

{
  echo '['
  for i in "${!entries[@]}"; do
    comma=','; [ "$i" -eq $((${#entries[@]} - 1)) ] && comma=''
    printf '  "%s"%s\n' "${entries[$i]}" "$comma"
  done
  echo ']'
} > "$WEBAPP/models.json"

cat <<EOF
Preview staged at: $PREVIEW

On Torch, serve it:
  cd "$PREVIEW" && python3 -m http.server 8000 --bind 127.0.0.1

On your local machine, in another terminal:
  ssh -N -L 8000:127.0.0.1:8000 torch

Then open (replace TIMESTAMP with any rendered date):
  http://localhost:8000/webapp/compare.html?solver=bp_solver&date=TIMESTAMP&mode=dems
EOF
