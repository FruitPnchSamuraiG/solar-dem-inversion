#!/usr/bin/env python3
"""What does the final layer look like?  (advisor question, 2026-10-21)

The label-free network ends in Linear(336 -> 54) followed by Softplus: the 54
non-negative weights of the DEM basis (18 spikes, 18 Gaussians of 0.1 dex, 18
of 0.2 dex, each centred on logT 5.5 .. 7.2, shape-major), and DEM = B w.
This looks at that layer from both sides, for each checkpoint given:

  weights   the 54 x 336 output matrix W and its bias: singular values,
            effective rank (participation ratio, components for 90 / 99% of
            the energy), the 54 outputs and the 336 hidden units sorted by the
            leading singular vectors ("sort by eigenvector");
  activity  on a sample of validation pixels from every brightness band (all
            flare cores): covariance and correlation of the 54 outputs, their
            eigenvectors and the outputs sorted along them (overall and per
            band); which outputs the network uses (mass by shape and by
            temperature, how often each is active) against the solver's own
            fewest-weights decomposition of its DEM on the same pixels; the
            PCA of the 18-bin DEMs (network vs solver, raw and unit-sum
            shapes); the PCA of the 336 pre-output activations.

Everything is an association on a sample; nothing here changes a model.
Evaluation only, CPU.  Runs on our staged split (collect() needs the errors and
tolLevel to pick valid pixels), from the project root (RData.npz).

    python3 experiments/final_layer.py --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS \
        --models bp=output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt \
        --output output/experiments/diagnostics/final_layer.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from experiments.diagnose_bright_failures import INTENSITY_LABELS, N_BINS
from experiments.objective_gap import (CORE, G, N_GROUPS, collect, init_worker, min_l1_x)

SHAPES = ("spike", "g0.1", "g0.2")
LOGT = np.round(5.5 + 0.1 * np.arange(N_BINS), 1)
LABELS = [f"{s}:{t:.1f}" for s in SHAPES for t in LOGT]       # the 54 outputs, shape-major
DEFAULT_MODELS = [
    "bp=output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt",
    "enet=output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt",
    "bp_old=output/experiments/resample/scaled_mlp6_barrier_h336_resample.pt",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="staged split with _x/_e/_y/_m zarrs")
    p.add_argument("--phase", default="val")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="label=checkpoint ...")
    p.add_argument("--thresholds", default="eval_and_enet_specs/aia_thresholds.json")
    p.add_argument("--per-group", type=int, default=100000, help="pixels kept per brightness band")
    p.add_argument("--core-cap", type=int, default=1000000, help="max flare-core pixels (default: all)")
    p.add_argument("--solver-pixels", type=int, default=20000,
                   help="pixels (stratified by band) given the solver's fewest-weights decomposition")
    p.add_argument("--subsample", type=int, default=5000, help="per-pixel rows kept in the npz")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="JSON path; .txt and .npz use the same stem")
    return p.parse_args()


# ── linear algebra helpers ────────────────────────────────────────────────────

def spectrum_summary(values):
    """Participation ratio and the components for 90 / 99% of a non-negative spectrum."""
    v = np.asarray(values, np.float64)
    v = np.clip(v, 0, None)
    tot = v.sum()
    if tot <= 0:
        return {"participation_ratio": 0.0, "n90": 0, "n99": 0, "total": 0.0}
    c = np.cumsum(np.sort(v)[::-1]) / tot
    return {"participation_ratio": float(tot ** 2 / np.square(v).sum()),
            "n90": int(np.searchsorted(c, 0.90) + 1),
            "n99": int(np.searchsorted(c, 0.99) + 1),
            "total": float(tot)}


def fix_sign(vec):
    """Make the largest-magnitude loading positive so orderings are reproducible."""
    i = int(np.argmax(np.abs(vec)))
    return vec * (1.0 if vec[i] >= 0 else -1.0)


def eig_desc(mat):
    """Symmetric eigen-decomposition, descending."""
    w, v = np.linalg.eigh(mat)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def covariance_and_correlation(x):
    """x [N, k] -> (mean, cov, corr, std); dead columns get zero correlation."""
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    alive = std > 1e-12 * max(std.max(), 1e-300)
    inv = np.where(alive, 1.0 / np.where(alive, std, 1.0), 0.0)
    corr = cov * inv[:, None] * inv[None, :]
    corr[np.arange(len(std)), np.arange(len(std))] = np.where(alive, 1.0, 0.0)
    return mean, cov, corr, std


def pca_curve(x):
    """Centred PCA of x [N, k]: eigenvalues (descending) and cumulative share."""
    _, cov, _, _ = covariance_and_correlation(x)
    w, _ = eig_desc(cov)
    w = np.clip(w, 0, None)
    tot = w.sum()
    cum = np.cumsum(w) / tot if tot > 0 else np.zeros_like(w)
    return w, cum


def unit_sum(x):
    s = x.sum(1, keepdims=True)
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), 0.0)


# ── usage of the 54 outputs ───────────────────────────────────────────────────

def usage(x, group):
    """Mass by shape and by temperature, how often each output is active (> 1% of
    the pixel's largest weight), overall and per brightness band."""
    x = np.asarray(x, np.float64)
    out = {"overall": _usage_rows(x), "by_band": {}}
    for g, lab in enumerate(INTENSITY_LABELS):
        idx = np.flatnonzero(group == g)
        if idx.size:
            out["by_band"][lab] = _usage_rows(x[idx])
    return out


def _usage_rows(x):
    tot = x.sum()
    by_shape = x.reshape(len(x), len(SHAPES), N_BINS).sum((0, 2))
    by_temp = x.reshape(len(x), len(SHAPES), N_BINS).sum((0, 1))
    peak = x.max(1, keepdims=True)
    active = (x > 0.01 * np.maximum(peak, 1e-300)) & (peak > 0)
    n_active = active.sum(1)
    return {"n": int(len(x)),
            "mass_share_by_shape": {s: float(by_shape[i] / tot) if tot > 0 else 0.0
                                    for i, s in enumerate(SHAPES)},
            "mass_share_by_logt": {f"{t:.1f}": float(by_temp[i] / tot) if tot > 0 else 0.0
                                   for i, t in enumerate(LOGT)},
            "mean_weight": x.mean(0).tolist(),
            "active_fraction": active.mean(0).tolist(),
            "active_outputs_per_pixel": {"mean": float(n_active.mean()),
                                         "median": float(np.median(n_active)),
                                         "p90": float(np.percentile(n_active, 90))}}


# ── the network side ──────────────────────────────────────────────────────────

def find_head(core):
    """(last Linear, the module feeding it) inside the variant's Sequential."""
    import torch.nn as nn
    seq = core.net
    lin = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    last = lin[-1]
    return seq[last], seq[last - 1], last


def run_model(path, obs, batch=8192):
    """-> dict with W, b, the 54 weights, the 336 pre-output activations and ckpt."""
    import torch
    from src.scaled_eval import load_scaled_model
    n_basis = G["D"].shape[1]
    model, ckpt = load_scaled_model(path, n_basis, torch.device("cpu"))
    core = model.inner
    head, feeder, _ = find_head(core)
    captured = []
    handle = feeder.register_forward_hook(lambda m, i, o: captured.append(o.detach().numpy().copy()))
    k = ckpt["patch_size"]
    c = k // 2
    ws = []
    with torch.no_grad():
        for s in range(0, len(obs), batch):
            o = obs[s:s + batch].astype(np.float32)
            patch = np.zeros((len(o), 6, k, k), np.float32)
            patch[:, :, c, c] = o
            ws.append(model(torch.from_numpy(patch)).numpy())
    handle.remove()
    return {"W": head.weight.detach().numpy().astype(np.float64),
            "b": head.bias.detach().numpy().astype(np.float64),
            "w": np.concatenate(ws).astype(np.float64),
            "h": np.concatenate(captured).astype(np.float64),
            "ckpt": {key: ckpt.get(key) for key in ("variant", "loss", "hidden", "epoch",
                                                     "input_transform", "fourier_freqs", "mu")}}


def analyse_weights(W, b):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    energy = s ** 2
    u1, u2 = fix_sign(U[:, 0]), fix_sign(U[:, 1])
    v1 = fix_sign(Vt[0])
    out = {"shape": list(W.shape),
           "singular_values": s.tolist(),
           "energy": spectrum_summary(energy),
           "row_norms": np.linalg.norm(W, axis=1).tolist(),
           "bias": b.tolist(),
           "outputs_by_u1": np.argsort(-u1).tolist(),
           "outputs_by_u2": np.argsort(-u2).tolist(),
           "u1": u1.tolist(), "u2": u2.tolist(),
           "hidden_by_v1": np.argsort(-v1).tolist()}
    return out, (U, s, Vt)


def analyse_outputs(w, group):
    mean, cov, corr, std = covariance_and_correlation(w)
    ev_cov, vec_cov = eig_desc(cov)
    ev_corr, vec_corr = eig_desc(corr)
    e1, e2, e3 = (fix_sign(vec_corr[:, i]) for i in range(3))
    c1 = fix_sign(vec_cov[:, 0])
    # scale-free view: each pixel's weights as shares of its total
    _, cov_n, _, _ = covariance_and_correlation(unit_sum(w))
    ev_n, vec_n = eig_desc(cov_n)
    n1 = fix_sign(vec_n[:, 0])
    res = {"mean": mean.tolist(), "std": std.tolist(),
           "dead_outputs": [LABELS[i] for i in np.flatnonzero(std <= 1e-12 * max(std.max(), 1e-300))],
           "cov_eigenvalues": np.clip(ev_cov, 0, None).tolist(),
           "cov_spectrum": spectrum_summary(ev_cov),
           "corr_eigenvalues": np.clip(ev_corr, 0, None).tolist(),
           "corr_spectrum": spectrum_summary(ev_corr),
           "unit_sum_cov_spectrum": spectrum_summary(ev_n),
           "corr_e1": e1.tolist(), "corr_e2": e2.tolist(), "corr_e3": e3.tolist(),
           "outputs_by_corr_e1": np.argsort(-e1).tolist(),
           "outputs_by_corr_e2": np.argsort(-e2).tolist(),
           "outputs_by_cov_e1": np.argsort(-c1).tolist(),
           "outputs_by_unit_sum_e1": np.argsort(-n1).tolist(),
           "by_band": {}}
    per_band_e1 = np.zeros((N_GROUPS, w.shape[1]))
    for g, lab in enumerate(INTENSITY_LABELS):
        idx = np.flatnonzero(group == g)
        if idx.size < 10:
            continue
        _, cov_g, corr_g, _ = covariance_and_correlation(w[idx])
        evg, vg = eig_desc(corr_g)
        evc, _ = eig_desc(cov_g)
        g1 = fix_sign(vg[:, 0])
        per_band_e1[g] = g1
        res["by_band"][lab] = {"n": int(idx.size),
                               "corr_spectrum": spectrum_summary(evg),
                               "cov_spectrum": spectrum_summary(evc),
                               "outputs_by_corr_e1": np.argsort(-g1).tolist()}
    mats = {"cov": cov, "corr": corr, "corr_vectors": vec_corr, "cov_vectors": vec_cov,
            "unit_sum_cov": cov_n, "unit_sum_vectors": vec_n, "band_e1": per_band_e1}
    return res, mats


def analyse_hidden(h):
    mean, cov, corr, std = covariance_and_correlation(h)
    ev, vec = eig_desc(cov)
    evc, _ = eig_desc(corr)
    p1 = fix_sign(vec[:, 0])
    res = {"n_units": int(h.shape[1]),
           "cov_spectrum": spectrum_summary(ev),
           "corr_spectrum": spectrum_summary(evc),
           "dead_units": int(np.sum(std <= 1e-12 * max(std.max(), 1e-300))),
           "mean_abs_activation": float(np.abs(h).mean()),
           "hidden_by_cov_e1": np.argsort(-p1).tolist()}
    return res, {"hidden_cov_vectors": vec[:, :16], "hidden_cov_eigenvalues": np.clip(ev, 0, None)}


def analyse_dems(dem_net, dem_sol, group):
    out = {}
    for name, d in (("network", dem_net), ("solver", dem_sol)):
        raw_w, raw_c = pca_curve(d)
        shp_w, shp_c = pca_curve(unit_sum(d))
        out[name] = {"raw": spectrum_summary(raw_w), "raw_cumulative": raw_c.tolist(),
                     "shape": spectrum_summary(shp_w), "shape_cumulative": shp_c.tolist(),
                     "mean_dem_by_band": {lab: d[group == g].mean(0).tolist()
                                          for g, lab in enumerate(INTENSITY_LABELS)
                                          if np.any(group == g)}}
    return out


def solver_chunk(dems):
    return min_l1_x(dems)


# ── rendering ─────────────────────────────────────────────────────────────────

def order_text(order, k=18):
    return " ".join(LABELS[i] for i in order[:k])


def render(result):
    L = [f"final layer: {result['n_pixels']:,} validation pixels, per band "
         f"{result['n_per_band']}; solver decomposition on {result['n_solver_pixels']:,}",
         f"outputs are shape:logT, shape-major (spike, g0.1, g0.2); runtime {result['runtime_s']}s", ""]
    for label, m in result["models"].items():
        W, o, hdn, d, u, us = (m["weights"], m["outputs"], m["hidden"], m["dems"], m["usage"],
                               m["usage_solver_pixels"])
        L.append(f"== {label}: {m['ckpt']}")
        L.append(f"  W {W['shape']}  participation ratio {W['energy']['participation_ratio']:.1f}, "
                 f"{W['energy']['n90']} / {W['energy']['n99']} components for 90 / 99% of the energy; "
                 f"top singular values {', '.join(f'{v:.2f}' for v in W['singular_values'][:6])}")
        L.append(f"  outputs by W's first left singular vector: {order_text(W['outputs_by_u1'])} ...")
        L.append(f"  outputs by W's second:                     {order_text(W['outputs_by_u2'])} ...")
        L.append(f"  output correlation: participation ratio {o['corr_spectrum']['participation_ratio']:.1f}, "
                 f"{o['corr_spectrum']['n90']} / {o['corr_spectrum']['n99']} for 90 / 99%; "
                 f"covariance {o['cov_spectrum']['participation_ratio']:.1f} ({o['cov_spectrum']['n90']} / "
                 f"{o['cov_spectrum']['n99']}); unit-sum shares {o['unit_sum_cov_spectrum']['participation_ratio']:.1f}"
                 f" ({o['unit_sum_cov_spectrum']['n90']} / {o['unit_sum_cov_spectrum']['n99']}); "
                 f"dead outputs {len(o['dead_outputs'])}")
        L.append(f"  outputs by correlation eigenvector 1: {order_text(o['outputs_by_corr_e1'])} ...")
        L.append(f"  outputs by correlation eigenvector 2: {order_text(o['outputs_by_corr_e2'])} ...")
        L.append(f"  outputs by unit-sum eigenvector 1:    {order_text(o['outputs_by_unit_sum_e1'])} ...")
        for lab, bb in o["by_band"].items():
            L.append(f"    {lab:>9}: n {bb['n']:>7,}  corr PR {bb['corr_spectrum']['participation_ratio']:5.1f} "
                     f"(n90 {bb['corr_spectrum']['n90']:2d})  e1 order {order_text(bb['outputs_by_corr_e1'], 10)} ...")
        L.append(f"  hidden ({hdn['n_units']}): covariance PR {hdn['cov_spectrum']['participation_ratio']:.1f}, "
                 f"{hdn['cov_spectrum']['n90']} / {hdn['cov_spectrum']['n99']} for 90 / 99%; "
                 f"correlation PR {hdn['corr_spectrum']['participation_ratio']:.1f}; dead units {hdn['dead_units']}")
        sh_n = u["overall"]["mass_share_by_shape"]
        sh_s = us["solver"]["overall"]["mass_share_by_shape"]
        sh_m = us["network"]["overall"]["mass_share_by_shape"]
        L.append("  mass share by shape (network, all pixels | network, solver pixels | solver): "
                 + ", ".join(f"{s} {sh_n[s]:.2f} | {sh_m[s]:.2f} | {sh_s[s]:.2f}" for s in SHAPES))
        act = u["overall"]["active_outputs_per_pixel"]
        act_s = us["solver"]["overall"]["active_outputs_per_pixel"]
        L.append(f"  active outputs per pixel (> 1% of the pixel's largest): network mean {act['mean']:.1f}, "
                 f"median {act['median']:.0f}, p90 {act['p90']:.0f}; solver mean {act_s['mean']:.1f}, "
                 f"median {act_s['median']:.0f}, p90 {act_s['p90']:.0f}")
        tn = u["overall"]["mass_share_by_logt"]
        ts = us["solver"]["overall"]["mass_share_by_logt"]
        L.append("  mass share by logT, network: " + " ".join(f"{k}:{v:.3f}" for k, v in tn.items()))
        L.append("  mass share by logT, solver:  " + " ".join(f"{k}:{v:.3f}" for k, v in ts.items()))
        for lab in INTENSITY_LABELS:
            if lab in u["by_band"] and lab in us["solver"]["by_band"]:
                a, b_ = u["by_band"][lab]["mass_share_by_shape"], us["solver"]["by_band"][lab]["mass_share_by_shape"]
                L.append(f"    {lab:>9}: network spike {a['spike']:.2f} g0.1 {a['g0.1']:.2f} g0.2 {a['g0.2']:.2f}"
                         f"   solver spike {b_['spike']:.2f} g0.1 {b_['g0.1']:.2f} g0.2 {b_['g0.2']:.2f}"
                         f"   active/pixel {u['by_band'][lab]['active_outputs_per_pixel']['mean']:.1f} vs "
                         f"{us['solver']['by_band'][lab]['active_outputs_per_pixel']['mean']:.1f}")
        for name in ("network", "solver"):
            r, s_ = d[name]["raw"], d[name]["shape"]
            L.append(f"  DEM PCA, {name:>7}: raw {r['n90']} / {r['n99']} components for 90 / 99% (PR {r['participation_ratio']:.1f}); "
                     f"unit-sum shapes {s_['n90']} / {s_['n99']} (PR {s_['participation_ratio']:.1f})")
        L.append("")
    return "\n".join(L)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import zarr
    t0 = time.time()
    with open(args.thresholds) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload.get("test", payload), np.float64)
    nb = zarr.open(os.path.join(args.root, f"{args.phase}_x.zarr"), mode="r").shape[-1]
    if args.max_blocks:
        nb = min(nb, args.max_blocks)
    cfg = {"root": os.path.abspath(args.root), "phase": args.phase, "loss": "barrier",
           "thresholds": thresholds.tolist(), "seed": args.seed, "tolfac": 1.4,
           "alpha_l1": 1.0, "mu": 1.0, "enet_alpha": 0.001, "enet_lam": 0.5, "l1_power": 0.0}
    # Same sampling as objective_gap.py: aim at --per-group pixels per band
    # (population shares from the final diagnostic), keep every flare core.
    share = np.array([3.85, 10.51, 33.02, 42.50, 6.53, 2.35, 1.15, 0.08, 0.0098]) / 100
    est_valid = nb * 128 * 128 * 0.85
    keep = np.minimum(1.0, args.per_group / (share * est_valid) * 1.5)
    keep[CORE] = 1.0
    cfg["keep"] = keep.tolist()
    chunk = math.ceil(nb / args.workers)
    jobs = [(cfg, s, min(s + chunk, nb), i) for i, s in enumerate(range(0, nb, chunk))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts_ = pool.map(collect, jobs)
    pop = sum(p for _, p in parts_)
    data = {k: np.concatenate([d[k] for d, _ in parts_ if d[k] is not None])
            for k in parts_[0][0]}
    rng = np.random.default_rng(args.seed)
    take = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(data["group"] == g)
        cap = args.core_cap if g == CORE else args.per_group
        take.append(rng.choice(idx, size=min(cap, idx.size), replace=False) if idx.size else idx)
    take = np.sort(np.concatenate(take))
    data = {k: v[take] for k, v in data.items()}
    obs = data["obs"].astype(np.float32)
    dem_sol = np.maximum(data["dem"].astype(np.float64), 0.0)
    group = data["group"]
    n_per_band = np.bincount(group, minlength=N_GROUPS).tolist()
    print(f"collected {len(obs):,} pixels from {nb:,} blocks in {time.time() - t0:.0f}s; "
          f"per band {n_per_band}", flush=True)

    init_worker(cfg)
    B = G["B"]

    # Stratified subset given the solver's fewest-weights decomposition (an LP per pixel).
    per = max(1, args.solver_pixels // N_GROUPS)
    sol_idx = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(group == g)
        sol_idx.append(rng.choice(idx, size=min(per, idx.size), replace=False) if idx.size else idx)
    sol_idx = np.sort(np.concatenate(sol_idx))
    t1 = time.time()
    n_chunks = max(1, args.workers * 4)
    bounds = np.linspace(0, len(sol_idx), n_chunks + 1).astype(int)
    chunks = [dem_sol[sol_idx[a:b]] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        x_sol = np.concatenate(pool.map(solver_chunk, chunks))
    print(f"solver decompositions for {len(sol_idx):,} pixels in {time.time() - t1:.0f}s", flush=True)
    usage_sol = usage(x_sol, group[sol_idx])

    # A fixed per-pixel subsample for the npz (scatter figures), stratified by band.
    per = max(1, args.subsample // N_GROUPS)
    sub_idx = []
    for g in range(N_GROUPS):
        idx = np.flatnonzero(group == g)
        sub_idx.append(rng.choice(idx, size=min(per, idx.size), replace=False) if idx.size else idx)
    sub_idx = np.sort(np.concatenate(sub_idx))

    result = {"purpose": "the output layer of the label-free MLP: weights, activity, usage, DEM rank",
              "root": cfg["root"], "phase": args.phase, "seed": args.seed,
              "n_pixels": int(len(obs)), "n_per_band": dict(zip(INTENSITY_LABELS, n_per_band)),
              "population_per_band": dict(zip(INTENSITY_LABELS, pop.tolist())),
              "n_solver_pixels": int(len(sol_idx)), "labels": LABELS, "logt": LOGT.tolist(),
              "shapes": list(SHAPES), "models": {}}
    arrays = {"group": group.astype(np.int8), "sub_idx": sub_idx, "sol_idx": sol_idx,
              "sub_obs": obs[sub_idx], "sub_group": group[sub_idx].astype(np.int8),
              "sub_tol": data["tol"][sub_idx], "sub_dem_solver": dem_sol[sub_idx].astype(np.float32),
              "sol_x_solver": x_sol.astype(np.float32), "sol_group": group[sol_idx].astype(np.int8)}
    for spec in args.models:
        label, path = spec.split("=", 1)
        t2 = time.time()
        r = run_model(path, obs)
        w, h = r["w"], r["h"]
        dem_net = w @ B.T
        weights, (U, s, Vt) = analyse_weights(r["W"], r["b"])
        outputs, out_mats = analyse_outputs(w, group)
        hidden, hid_mats = analyse_hidden(h)
        dems = analyse_dems(dem_net, dem_sol, group)
        us_net = usage(w[sol_idx], group[sol_idx])
        m = {"path": path, "ckpt": r["ckpt"], "weights": weights, "outputs": outputs, "hidden": hidden,
             "dems": dems, "usage": usage(w, group),
             "usage_solver_pixels": {"network": us_net, "solver": usage_sol},
             "dem_mse_vs_solver": float(np.square(dem_net - dem_sol).mean()),
             "runtime_s": round(time.time() - t2, 1)}
        result["models"][label] = m
        # Matrices for the figures: W in basis order, W reordered by the data-driven
        # eigenvectors (rows by output correlation e1, columns by hidden covariance e1)
        # and by its own singular vectors.
        W = r["W"]
        row_e1 = np.asarray(outputs["outputs_by_corr_e1"])
        col_e1 = np.asarray(hidden["hidden_by_cov_e1"])
        arrays.update({
            f"{label}__W": W.astype(np.float32), f"{label}__b": r["b"].astype(np.float32),
            f"{label}__U": U.astype(np.float32), f"{label}__s": s.astype(np.float32),
            f"{label}__V": Vt.T.astype(np.float32),
            f"{label}__W_sorted_data": W[np.ix_(row_e1, col_e1)].astype(np.float32),
            f"{label}__W_sorted_svd": W[np.ix_(np.asarray(weights["outputs_by_u1"]),
                                              np.asarray(weights["hidden_by_v1"]))].astype(np.float32),
            f"{label}__out_cov": out_mats["cov"].astype(np.float32),
            f"{label}__out_corr": out_mats["corr"].astype(np.float32),
            f"{label}__out_corr_sorted": out_mats["corr"][np.ix_(row_e1, row_e1)].astype(np.float32),
            f"{label}__out_corr_vectors": out_mats["corr_vectors"].astype(np.float32),
            f"{label}__out_cov_vectors": out_mats["cov_vectors"].astype(np.float32),
            f"{label}__out_unit_sum_vectors": out_mats["unit_sum_vectors"].astype(np.float32),
            f"{label}__band_e1": out_mats["band_e1"].astype(np.float32),
            f"{label}__hidden_cov_vectors": hid_mats["hidden_cov_vectors"].astype(np.float32),
            f"{label}__hidden_cov_eigenvalues": hid_mats["hidden_cov_eigenvalues"].astype(np.float32),
            f"{label}__sub_w": w[sub_idx].astype(np.float32),
            f"{label}__sub_h": h[sub_idx].astype(np.float32),
            f"{label}__sub_dem": dem_net[sub_idx].astype(np.float32),
            f"{label}__sol_x_network": w[sol_idx].astype(np.float32),
        })
        print(f"{label}: {m['runtime_s']}s, W participation ratio "
              f"{weights['energy']['participation_ratio']:.1f}", flush=True)

    result["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    stem = os.path.splitext(args.output)[0]
    np.savez_compressed(stem + ".npz", **arrays)
    text = render(result)
    with open(stem + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.output} (+ .txt, .npz) in {result['runtime_s']}s")


if __name__ == "__main__":
    main()
