#!/usr/bin/env python3
"""Production label-free model vs a retrained variant vs supervised, one track.

Reads diagnose_bright_failures.py and aia_fit_by_brightness.py outputs:
  production  results/plots/13_bright_diagnostic_20260928/{track}_{bright_failure,aia_fit}.json
  supervised  results/plots/13_bright_diagnostic_20260928/{track}_bright_failure_supervised.json
  new run     <dir>/{track}_{bright_failure,aia_fit}_{tag}.json

    uv run python experiments/compare_input_encoding.py <dir> [tag=sqrtff12] [track=bp]
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "results/plots/13_bright_diagnostic_20260928")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    new_dir = sys.argv[1] if len(sys.argv) > 1 else BASE
    tag = sys.argv[2] if len(sys.argv) > 2 else "sqrtff12"
    track = sys.argv[3] if len(sys.argv) > 3 else "bp"
    prod = load(os.path.join(BASE, f"{track}_bright_failure.json"))
    sup = load(os.path.join(BASE, f"{track}_bright_failure_supervised.json"))
    new = load(os.path.join(new_dir, f"{track}_bright_failure_{tag}.json"))
    prod_aia = load(os.path.join(BASE, f"{track}_aia_fit.json"))["results"]
    new_aia = load(os.path.join(new_dir, f"{track}_aia_fit_{tag}.json"))["results"]

    def g(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d

    rows = []
    for label, path in [
            ("DEM MSE, all pixels (website)", ("population", "full", "dem_mse")),
            ("EM error %, all pixels", ("population", "full", "em_abs_rel_err_pct")),
            ("W1 (dex), all pixels", ("population", "full", "w1_dex")),
            ("DEM MSE, Bright", ("population", "bright", "dem_mse")),
            ("DEM MSE, Quiet", ("population", "quiet", "dem_mse")),
            ("Median pixel error, all", ("population", "full", "sse_p50")),
            ("Median pixel error, Bright", ("population", "bright", "sse_p50")),
            ("Relative error, 1-1.8x", ("by_intensity_over_threshold", "1-1.8x", "relative_sse")),
            ("Relative error, 3.2-10x", ("by_intensity_over_threshold", "3.2-10x", "relative_sse")),
            ("Relative error, 10-32x", ("by_intensity_over_threshold", "10-32x", "relative_sse")),
            ("Relative error, 32x+", ("by_intensity_over_threshold", ">=32x", "relative_sse")),
            ("Flare-core emission / solver", ("by_intensity_over_threshold", ">=32x", "em_ratio_pred_over_ref")),
            ("Flare-core peak / solver", ("by_intensity_over_threshold", ">=32x", "peak_ratio_pred_over_ref")),
            ("Flare-core multi-peaked %", ("by_intensity_over_threshold", ">=32x", "model_multimodal_pct")),
            ("Relative error, 0.03-0.1x (faint)", ("by_intensity_over_threshold", "0.03-0.1x", "relative_sse"))]:
        rows.append((label, g(prod, *path), g(new, *path), g(sup, *path)))
    for ch in ("94", "131", "335"):
        path = ("label_free", "by_band", ">=32x", "channels", ch, "recon_over_obs")
        rows.append((f"Flare-core {ch} A fit (recon/obs)", g(prod_aia, *path), g(new_aia, *path),
                     g(prod_aia, "supervised", "by_band", ">=32x", "channels", ch, "recon_over_obs")))
    for key, label in (("mae", "AIA MAE, all pixels"), ("mse", "AIA MSE, all pixels")):
        rows.append((label, g(prod_aia, "label_free", "pooled_full", key), g(new_aia, "label_free", "pooled_full", key),
                     g(prod_aia, "supervised", "pooled_full", key)))

    fmt = lambda v: "-" if v is None else f"{v:.4g}"
    print(f"{track.upper() + ' track':<34}{'production':>12}{tag:>12}{'supervised':>12}")
    print("-" * 70)
    for label, a, b, c in rows:
        print(f"{label:<34}{fmt(a):>12}{fmt(b):>12}{fmt(c):>12}")
    print(f"\nsolver flare-core 94 A fit: "
          f"{fmt(g(prod_aia, 'solver', 'by_band', '>=32x', 'channels', '94', 'recon_over_obs'))}")


if __name__ == "__main__":
    main()
