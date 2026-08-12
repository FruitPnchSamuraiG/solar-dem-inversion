#!/bin/bash
# Create a Torch-local static preview of solver-vs-mlp6 full-disk assets.
# Serve with the command printed at the end, then tunnel port 8000 over SSH.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIS_ROOT="${VIS_ROOT:-${SCRATCH:?SCRATCH must be set}/dem/visuals}"
STAMP="${TIMESTAMP:-20150923_180356}"
PREVIEW="$VIS_ROOT/preview"
WEBAPP="$PREVIEW/webapp"
RESULTS="$PREVIEW/results/test"

mkdir -p "$WEBAPP" "$RESULTS"
cp "$REPO_DIR/student_package/visuals_pipeline/webapp/compare.html" "$WEBAPP/compare.html"
cp "$REPO_DIR/student_package/visuals_pipeline/webapp/compare.js" "$WEBAPP/compare.js"

entries=()
for track in bp enet; do
  for run in solver mlp6_h232; do
    source="$VIS_ROOT/assets/${track}_${run}/$STAMP"
    name="${track}_${run}/${STAMP}"
    target="$RESULTS/$name"
    if [ ! -d "$source" ]; then
      echo "missing assets: $source" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$target")"
    ln -sfn "$source" "$target"
    entries+=("$name")
  done
done

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

Then open:
  http://localhost:8000/webapp/compare.html?model1=bp_solver/${STAMP}&model2=bp_mlp6_h232/${STAMP}&mode=dems
EOF
