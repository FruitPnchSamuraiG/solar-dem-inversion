#!/bin/bash
# Copy every finished follow-up evaluation from Torch into the results folder,
# rebuild runs.json for the runs that have results, and print the summary table.
#   bash experiments/collect_followup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/plots/21_followup_20261006
for pat in "*_fu_*.json" "*_fu_*.txt" "bp_objective_gap_val*.json" "bp_objective_gap_val*.txt"; do
  scp -q "torch:projects/dem/output/experiments/diagnostics/$pat" "$R/" 2>/dev/null || true
done
scp -q "torch:projects/dem/output/experiments/followup/*_history.json" "$R/" 2>/dev/null || true
python3 - <<'PY'
import json, os
R = "results/plots/21_followup_20261006"
base = json.load(open(f"{R}/runs.json"))
labels = {"bp_bg1": "oversample bright", "bp_ep120": "120 epochs", "bp_10m": "10M parameters",
          "bp_mu100": "band x100", "bp_mu30_ibr05": "band x30, L1/sqrt(I)",
          "bp_mu100_bg1": "band x100 + oversample", "bp_mu100_ep120": "band x100, 120 epochs",
          "bp_mu100_10m": "band x100, 10M", "bp_log1p_mu100": "band x100, log1p input",
          "bp_mu100_noclip": "band x100, no clip",
          "enet_bg1": "oversample bright", "enet_ep120": "120 epochs", "enet_10m": "10M parameters"}
for track in ("bp", "enet"):
    keep = [r for r in base[track] if not r.get("followup")]
    for name, label in labels.items():
        if not name.startswith(track + "_"):
            continue
        d, a = f"{R}/{track}_bright_failure_fu_{name}.json", f"{R}/{track}_aia_fit_fu_{name}.json"
        if os.path.exists(d):
            keep.append({"label": label, "diag": d, "aia": a if os.path.exists(a) else None,
                         "followup": True})
    base[track] = keep
json.dump(base, open(f"{R}/runs.json", "w"), indent=1)
PY
python3 experiments/summarize_followup.py "$R/runs.json" "$R/summary.json"
