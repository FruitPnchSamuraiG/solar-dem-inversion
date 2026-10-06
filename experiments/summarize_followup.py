#!/usr/bin/env python3
"""One row per model from its full shared-test evaluation files.

Reads each model's bright-failure diagnostic JSON (DEM metrics by brightness)
and AIA-fit JSON (reconstruction by brightness), and prints the headline
metrics side by side: DEM MSE and MAE, median per-pixel error, EM error, W1,
AIA MAE and MSE, the Bright/Quiet DEM MSE split, and the flare-core (>=32x)
emission ratio and 94 A fit.

    python3 experiments/summarize_followup.py results/plots/21_followup_20261006/runs.json

runs.json: {"bp": [{"label": ..., "diag": path, "aia": path}, ...], "enet": [...]}
A row may give "supervised": true to read the supervised model's columns from
an AIA-fit file instead of the label-free ones.
"""
import json
import sys


def row(spec):
    d = json.load(open(spec["diag"]))
    pop = d["population"]
    full, bright, quiet = pop["full"], pop["bright"], pop["quiet"]
    core = d["by_intensity_over_threshold"][">=32x"]
    out = {"label": spec["label"],
           "dem_mse": full["dem_mse"],
           "dem_mae": sum(full["mae_by_logt"]) / len(full["mae_by_logt"]),
           "median_sse": full["sse_p50"],
           "em_err_pct": full["em_abs_rel_err_pct"],
           "w1": full["w1_dex"],
           "bright_mse": bright["dem_mse"],
           "quiet_mse": quiet["dem_mse"],
           "core_mse": core["dem_mse"],
           "core_em": core["em_ratio_pred_over_ref"],
           "core_mse_share_pct": 100 * core["dem_mse"] * core["n_pixels"] / (full["dem_mse"] * full["n_pixels"])}
    if spec.get("aia"):
        a = json.load(open(spec["aia"]))["results"]
        key = "supervised" if spec.get("supervised") else "label_free"
        out["aia_mae"] = a[key]["pooled_full"]["mae"]
        out["aia_mse"] = a[key]["pooled_full"]["mse"]
        out["core_94"] = a[key]["by_band"][">=32x"]["channels"]["94"]["recon_over_obs"]
    return out


COLS = [("dem_mse", "DEM MSE", "{:.3f}"), ("dem_mae", "DEM MAE", "{:.3f}"),
        ("median_sse", "median", "{:.4f}"), ("em_err_pct", "EM err %", "{:.1f}"),
        ("w1", "W1", "{:.3f}"), ("aia_mae", "AIA MAE", "{:.2f}"), ("aia_mse", "AIA MSE", "{:.0f}"),
        ("bright_mse", "Bright MSE", "{:.2f}"), ("quiet_mse", "Quiet MSE", "{:.4f}"),
        ("core_em", "core EM", "{:.2f}"), ("core_94", "core 94A", "{:.2f}")]


def main():
    spec = json.load(open(sys.argv[1]))
    result = {}
    for track, runs in spec.items():
        rows = [row(r) for r in runs]
        result[track] = rows
        print(f"\n== {track}")
        print(f"{'model':34s}" + "".join(f"{h:>11s}" for _, h, _ in COLS))
        for r in rows:
            cells = []
            for k, _, fmt in COLS:
                v = r.get(k)
                cells.append(f"{fmt.format(v):>11s}" if v is not None else f"{'-':>11s}")
            print(f"{r['label']:34s}" + "".join(cells))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
