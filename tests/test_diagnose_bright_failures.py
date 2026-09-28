"""
Checks for experiments/diagnose_bright_failures.py on a synthetic shared-test set.

1. The vectorised interior peak counter agrees with the project's scipy-based
   count_peaks on BP-like sparse, smooth, noisy and degenerate curves.
2. Full/Bright/Quiet pixel counts and DEM MSE reconcile exactly with
   eval_full_paper_test.py, which produced the published numbers. That script
   builds real 9x9 patches, so this also checks the centre-only fast path.
3. One worker and several workers give identical results, including repeated
   inputs whose five targets straddle a worker boundary.
4. The solver-scatter split is an exact identity: mean target SSE equals
   bias + scatter.
5. Samuel's supervised checkpoint gives the same Full/Bright/Quiet counts and
   MSE as his published Table-1 script, compute_paper_table_metrics.py.

    uv run python tests/test_diagnose_bright_failures.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.diagnose_bright_failures import interior_peak_counts, N_BINS
from src.scaled_eval import count_peaks

THRESHOLDS = [3.1212, 16.9986, 448.8948, 498.3444, 190.7388, 10.3326]
UNIQUE, TARGETS = 6, 5
N_BLOCKS = UNIQUE * TARGETS


def test_peaks():
    rng = np.random.default_rng(0)
    curves = []
    for _ in range(3000):                                   # BP-like sparse spikes
        c = np.zeros(N_BINS)
        k = rng.integers(1, 5)
        c[rng.choice(N_BINS, k, replace=False)] = rng.lognormal(0, 1.5, k)
        curves.append(c)
    t = np.arange(N_BINS)
    for _ in range(3000):                                   # smooth mixtures
        c = sum(rng.uniform(0.1, 3) * np.exp(-0.5 * ((t - rng.uniform(-2, 20)) / rng.uniform(0.5, 4)) ** 2)
                for _ in range(rng.integers(1, 4)))
        curves.append(c)
    curves += list(rng.normal(0, 1, (1000, N_BINS)))        # noise, negatives
    curves += [np.zeros(N_BINS), -np.ones(N_BINS), np.ones(N_BINS)]
    curves = np.array(curves)
    fast = interior_peak_counts(curves)
    ref = np.array([int(((w >= 1) & (w <= N_BINS - 2)).sum()) for w in
                    (count_peaks(c)[1] for c in curves)])
    bad = np.flatnonzero(fast != ref)
    assert bad.size == 0, f"{bad.size} peak-count mismatches, e.g. {curves[bad[0]]} fast={fast[bad[0]]} ref={ref[bad[0]]}"
    print(f"[ok] peak counts match scipy on {len(curves)} curves "
          f"({(ref >= 2).mean():.1%} multimodal)")


def build(root):
    rng = np.random.default_rng(1)
    x = zarr.open(os.path.join(root, "test_x.zarr"), mode="w", shape=(6, 256, 256, N_BLOCKS),
                  chunks=(6, 256, 256, 1), dtype="f4")
    y = zarr.open(os.path.join(root, "test_y.zarr"), mode="w", shape=(26, 128, 128, N_BLOCKS),
                  chunks=(26, 128, 128, 1), dtype="f4")
    thr = np.array(THRESHOLDS)[:, None, None]
    t = np.arange(N_BINS)[:, None, None]
    for u in range(UNIQUE):
        obs = (thr * 10 ** rng.normal(-0.6, 0.5, (6, 256, 256))).astype("f4")
        centre = rng.uniform(3, 14, (1, 128, 128))
        base = 10 ** rng.normal(0, 1, (1, 128, 128)) * np.exp(-0.5 * ((t - centre) / 1.5) ** 2)
        spike = rng.random((1, 128, 128)) < 0.3
        base = base + spike * 3 * np.exp(-0.5 * ((t - centre - 5) / 0.5) ** 2)
        for k in range(TARGETS):
            # Target-major layout: the five copies of an input are UNIQUE blocks
            # apart, so with two workers some groups straddle the boundary.
            b = k * UNIQUE + u
            dem = np.zeros((26, 128, 128), "f4")
            dem[:N_BINS] = np.clip(base * (1 + 0.3 * rng.normal(size=base.shape)), 0, None)
            dem[:, rng.random((128, 128)) < 0.01] = np.nan
            x[:, :, :, b] = obs
            y[:, :, :, b] = dem


def make_ckpt(path):
    from experiments.train_ablations import build_model
    from experiments.train_scaled import harden_softplus
    torch.manual_seed(0)
    core = build_model("mlp6", 54, 9, 6, hidden=32)
    harden_softplus(core, -20.0)
    torch.save({"model": core.state_dict(), "variant": "mlp6", "loss": "barrier", "patch_size": 9,
                "stride": 2, "channels": 6, "hidden": 32, "input_transform": "log1p",
                "softplus_floor": -20.0, "epoch": 0}, path)


def run(root, ckpt, out, workers):
    """Run the diagnostic; returns its JSON output."""
    subprocess.run([sys.executable, "experiments/diagnose_bright_failures.py", "--model", ckpt,
                    "--data", root, "--reference", "bp", "--output", out, "--workers", str(workers),
                    "--threads-per-worker", "1", "--max-pending", "64"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(out) as f:
        return json.load(f)


SUPERVISED = os.path.join(ROOT, "eval_and_enet_specs/checkpoints/model_best_bp.pth")


def published_supervised(root):
    out = os.path.join(root, "table1.json")
    thr = os.path.join(root, "thr.json")
    with open(thr, "w") as f:
        json.dump({"test": THRESHOLDS}, f)
    subprocess.run([sys.executable, "student_package/table1_dem_metrics/compute_paper_table_metrics.py",
                    "--model", SUPERVISED, "--data", root, "--variant", "sup", "--reference", "bp",
                    "--thresholds_json", thr, "--device", "cpu", "--batch_size", "2", "--output", out],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(out) as f:
        rows = json.load(f)["rows"]
    names = {"Full": "full", "Bright": "bright", "Quiet": "quiet"}
    return {names[r["pixel_type"]]: (r["n_pixels"], r["dem_mse"]) for r in rows}


def published(root, ckpt):
    """Full/Bright/Quiet MSE exactly as eval_full_paper_test.py computes them."""
    from experiments.eval_full_paper_test import add_metrics, empty_acc, predict_block
    from experiments.train_scaled import load_operators
    from src.scaled_eval import load_scaled_model
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        _, basis_t, n_basis, logt = load_operators(torch.device("cpu"))
    finally:
        os.chdir(cwd)
    model, _ = load_scaled_model(ckpt, n_basis, torch.device("cpu"))
    x = zarr.open(os.path.join(root, "test_x.zarr"), mode="r")
    y = zarr.open(os.path.join(root, "test_y.zarr"), mode="r")
    thr = np.asarray(THRESHOLDS, np.float32)
    acc = {k: empty_acc() for k in ("full", "bright", "quiet")}
    for i in range(N_BLOCKS):
        pred, obs = predict_block(model, basis_t, x[:, :, :, i], 8192)
        truth = np.asarray(y[:N_BINS, :, :, i], dtype=np.float32)
        valid = np.isfinite(truth).all(axis=0) & np.isfinite(pred).all(axis=0)
        bright = (obs >= thr[:, None, None]).any(axis=0) & valid
        for key, mask in (("full", valid), ("bright", bright), ("quiet", valid & ~bright)):
            add_metrics(acc[key], pred, truth, mask, logt[:N_BINS], i)
    return {k: (a["n_px"], a["sq"] / a["n_bins"]) for k, a in acc.items()}


def main():
    test_peaks()
    with tempfile.TemporaryDirectory() as root:
        build(root)
        ckpt = os.path.join(root, "model.pt")
        make_ckpt(ckpt)
        one = run(root, ckpt, os.path.join(root, "one.json"), 1)
        two = run(root, ckpt, os.path.join(root, "two.json"), 2)

        ref = published(root, ckpt)
        for key, (n_px, mse) in ref.items():
            row = one["population"][key]
            assert row["n_pixels"] == n_px, (key, row["n_pixels"], n_px)
            assert np.isclose(row["dem_mse"], mse, rtol=1e-9), (key, row["dem_mse"], mse)
        print(f"[ok] Full/Bright/Quiet counts and MSE match eval_full_paper_test: "
              + ", ".join(f"{k} n={n:,} mse={m:.5g}" for k, (n, m) in ref.items()))

        for part in ("population", "by_intensity_over_threshold", "by_dem_shape",
                     "by_trigger_pattern_exact", "worst_tail_composition", "solver_scatter"):
            fa = np.array(list(_flat(one[part])), float)
            fb = np.array(list(_flat(two[part])), float)
            assert fa.shape == fb.shape and np.allclose(fa, fb, rtol=1e-9, equal_nan=True), part
        sc1, sc2 = one["solver_scatter"], two["solver_scatter"]
        assert sc1["groups_by_targets"] == {"5": UNIQUE}, sc1["groups_by_targets"]
        assert sc2["groups_by_targets"] == {"5": UNIQUE}, sc2["groups_by_targets"]
        assert np.isclose(sc1["population"]["full"]["model_mse"], sc2["population"]["full"]["model_mse"], rtol=1e-9)
        print("[ok] 1 worker == 2 workers, including 5-target groups split across the boundary")

        full = sc1["population"]["full"]
        assert full["identity_check_rel"] < 1e-9, full
        # Pixels valid in all five targets are a subset, so model_mse here is
        # close to, not equal to, the population MSE.
        print(f"[ok] scatter identity exact (rel {full['identity_check_rel']:.1e}); "
              f"model {full['model_mse']:.4g} = bias {full['model_bias_mse']:.4g} "
              f"+ scatter {full['solver_scatter_mse']:.4g}")
        pop = one["population"]["full"]
        assert abs(pop["along_reference_share_pct"] + pop["orthogonal_share_pct"] - 100) < 1e-6
        print("[ok] along + orthogonal SSE shares sum to 100%")
        if os.path.exists(SUPERVISED):
            sup = run(root, SUPERVISED, os.path.join(root, "sup.json"), 2)
            ref = published_supervised(root)
            for key, (n_px, mse) in ref.items():
                row = sup["population"][key]
                assert row["n_pixels"] == n_px, (key, row["n_pixels"], n_px)
                assert np.isclose(row["dem_mse"], mse, rtol=1e-5), (key, row["dem_mse"], mse)
            print("[ok] supervised checkpoint matches compute_paper_table_metrics: "
                  + ", ".join(f"{k} n={n:,} mse={m:.5g}" for k, (n, m) in ref.items()))
            cmp = subprocess.run([sys.executable, "experiments/diagnose_bright_failures.py", "--compare",
                                  os.path.join(root, "one.json"), os.path.join(root, "sup.json")],
                                 cwd=ROOT, capture_output=True, text=True, check=True).stdout
            print(cmp[:1200])
        else:
            print("[skip] supervised checkpoint not present")
        print(subprocess.run([sys.executable, "experiments/diagnose_bright_failures.py", "--summarize",
                              os.path.join(root, "one.json")], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout[:1500])


def _flat(obj):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from _flat(obj[k])
    elif isinstance(obj, list):
        for v in obj:
            yield from _flat(v)
    elif isinstance(obj, (int, float)) or obj is None:
        yield np.nan if obj is None else obj


if __name__ == "__main__":
    main()
