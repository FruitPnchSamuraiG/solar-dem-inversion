"""
One-pixel experiment: test BarrierLoss with and without the fit term.

Reproduces the PhD student's OnePixel.ipynb, but:
  - uses our own flare data (no pre-solved .npz needed)
  - tests alpha_fit=0 (his setup, fails) vs alpha_fit>0 (our fix)
  - compares against LP solver ground truth

Run from project root:
    uv run python experiments/one_pixel.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from types import SimpleNamespace

from fullBP import getBasis, solveLP, _minimumDNs
from src.losses import barrier_loss_batch


# ── data loading ─────────────────────────────────────────────────────────────

def load_one_pixel(data_dir, crop_y=1950, crop_x=1950):
    """
    Load AIA data, calibrate it the same way fullBP.py does,
    and return a single pixel's observations + errors.
    """
    import sunpy.map
    import astropy.io.fits as fits
    import astropy.units as u
    from aiapy.calibrate import register, update_pointing, degradation
    from aiapy.calibrate.utils import get_correction_table, get_pointing_table

    wavelengths = [94, 131, 171, 193, 211, 335]
    getByWL = lambda p, wl: [fn for fn in os.listdir(p) if fn.endswith("%d.image_lev1.fits" % wl)][0]
    AIAFiles = [os.path.join(data_dir, getByWL(data_dir, wl)) for wl in wavelengths]

    AIAMaps = [sunpy.map.Map(fn) for fn in AIAFiles]
    AIAFits = [fits.open(fn) for fn in AIAFiles]
    exposures = [f[1].header["EXPTIME"] for f in AIAFits]

    correction_table = get_correction_table("aia_corr.csv")
    degradationFactors = [degradation(wl * u.angstrom, AIAMaps[i].date,
                          correction_table=correction_table)
                          for i, wl in enumerate(wavelengths)]
    scaleFactor = np.array([(exposures[i] * degradationFactors[i]).to_value()[0]
                             for i in range(len(exposures))])

    pointing_table = get_pointing_table("JSOC", time_range=(
        AIAMaps[0].date - 12 * u.h, AIAMaps[0].date + 12 * u.h))
    AIAMaps = [register(update_pointing(m, pointing_table=pointing_table)) for m in AIAMaps]

    # pad maps to 4096x4096 (register() can produce slightly smaller images)
    def pad4096(arr):
        if arr.shape[0] != 4096:
            diff = 4096 - arr.shape[0]
            arr = np.pad(arr, diff // 2)
        return arr

    AIACube = np.stack([pad4096(m.data) for m in AIAMaps], axis=0)       # [6, H, W]
    AIAErrors = np.stack([pad4096(np.maximum(m.data, _minimumDNs) ** 0.5)
                          for m in AIAMaps], axis=0)

    AIACube = AIACube / scaleFactor[:, None, None]
    AIAErrors = AIAErrors / scaleFactor[:, None, None]

    obs = AIACube[:, crop_y, crop_x].astype(np.float64)    # [6]
    err = AIAErrors[:, crop_y, crop_x].astype(np.float64)  # [6]
    return obs, err, wavelengths


def load_response():
    scale = 10 ** 26
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']
    R = (R * scale).astype(np.float64)
    B = getBasis(R, logT, alphas=[0.0, 0.1, 0.2])
    D = R @ B  # [6, n_basis]
    return R, B, D, logT


# ── solvers ───────────────────────────────────────────────────────────────────

def solve_lp(obs, err, D, B):
    for tolfac in [1.4, 3.0, 5.0]:
        lb = obs - tolfac * err
        ub = obs + tolfac * err
        t = (D, obs, lb, ub, None)
        x_basis = solveLP(t)
        if x_basis is not None:
            return x_basis, B @ x_basis, tolfac
    return None, None, None


def solve_barrier_direct(obs, err, D, B, tolfac=1.4, alpha_fit=0.0, mu=1.0, steps=5000):
    """Optimize x directly (not a NN) — reproduces his L-BFGS experiment."""
    obs_t = torch.tensor(obs, dtype=torch.float64)
    D_t = torch.tensor(D, dtype=torch.float64)
    lb = torch.tensor(obs - tolfac * err, dtype=torch.float64).unsqueeze(0)
    ub = torch.tensor(obs + tolfac * err, dtype=torch.float64).unsqueeze(0)

    # pseudoinverse init
    D_pinv = torch.linalg.pinv(D_t)
    x0 = torch.relu(obs_t @ D_pinv.T).unsqueeze(0)
    x = nn.Parameter(x0.clone())

    opt = torch.optim.LBFGS([x], lr=1e-3, max_iter=50,
                             line_search_fn='strong_wolfe',
                             tolerance_change=1e-12, tolerance_grad=1e-32)

    losses = []
    def closure():
        opt.zero_grad()
        x_pos = torch.relu(x)
        loss = barrier_loss_batch(x_pos, D_t, obs_t.unsqueeze(0), lb, ub,
                                  a_l1=1.0, a_l2=0.0, mu=mu, alpha=alpha_fit)
        loss.backward()
        losses.append(loss.item())
        return loss

    for _ in range(steps):
        opt.step(closure)

    x_star = torch.relu(x).detach().numpy()[0]
    dem = B @ x_star
    return x_star, dem, losses


def train_nn(obs, err, D, B, tolfac=1.4, alpha_fit=0.0, mu=1.0, steps=5000, lr=1e-3):
    """Train a small MLP with BarrierLoss — the core experiment."""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,6,1,1]
    D_t = torch.tensor(D, dtype=torch.float32)
    obs_flat = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 6]
    lb = torch.tensor(obs - tolfac * err, dtype=torch.float32).unsqueeze(0)
    ub = torch.tensor(obs + tolfac * err, dtype=torch.float32).unsqueeze(0)

    n_basis = D.shape[1]
    model = nn.Sequential(
        nn.Conv2d(6, 256, 1), nn.SiLU(),
        nn.Conv2d(256, 256, 1), nn.SiLU(),
        nn.Conv2d(256, 128, 1), nn.SiLU(),
        nn.Conv2d(128, n_basis, 1), nn.Softplus(),  # softplus keeps output > 0 smoothly
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for step in range(steps):
        opt.zero_grad()
        out = model(obs_t)  # [1, n_basis, 1, 1]
        x = out.permute(0, 2, 3, 1).reshape(-1, n_basis)  # [1, n_basis]
        loss = barrier_loss_batch(x, D_t, obs_flat, lb, ub,
                                  a_l1=1.0, a_l2=0.0, mu=mu, alpha=alpha_fit)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 1000 == 0:
            print(f"  step {step:5d}  loss={loss.item():.4f}")

    with torch.no_grad():
        out = model(obs_t)
        x_star = out.permute(0, 2, 3, 1).reshape(-1, n_basis).numpy()[0]
    dem = B @ x_star
    return x_star, dem, losses


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "./data/20170910_1548"
    os.makedirs("output/experiments", exist_ok=True)

    print("Loading AIA data...")
    obs, err, wavelengths = load_one_pixel(DATA_DIR, crop_y=2048, crop_x=2048)
    print(f"obs: {np.round(obs, 2)}")
    print(f"err: {np.round(err, 2)}")

    R, B, D, logT = load_response()

    # ── 1. LP solver ground truth ─────────────────────────────────────────────
    print("\n[1] LP solver (ground truth)...")
    x_lp, dem_lp, tolfac_used = solve_lp(obs, err, D, B)
    assert x_lp is not None, "LP solver failed on all tolerances — pick a different pixel"
    print(f"  L1={np.sum(np.abs(x_lp)):.3f}  (tolfac={tolfac_used})")
    tolfac = tolfac_used

    # ── 2. Direct optimization with BarrierLoss (reproduces his L-BFGS) ──────
    print("\n[2] Direct optimization, alpha_fit=0 (his setup)...")
    _, dem_direct, _ = solve_barrier_direct(obs, err, D, B, tolfac=tolfac, alpha_fit=0.0)

    print("\n[3] Direct optimization, alpha_fit=0.1 (with fit term)...")
    _, dem_direct_fit, _ = solve_barrier_direct(obs, err, D, B, tolfac=tolfac, alpha_fit=0.1)

    # ── 3. NN with alpha_fit=0 (his failing experiment) ───────────────────────
    print("\n[4] NN, alpha_fit=0 (his setup, expected to fail)...")
    _, dem_nn_nf, losses_nf = train_nn(obs, err, D, B, tolfac=tolfac, alpha_fit=0.0, steps=5000)

    # ── 4. NN with alpha_fit=0.1 (our fix) ───────────────────────────────────
    print("\n[5] NN, alpha_fit=0.1 (our fix)...")
    _, dem_nn_fix, losses_fix = train_nn(obs, err, D, B, tolfac=tolfac, alpha_fit=0.1, steps=5000)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # DEM curves
    ax = axes[0]
    ax.plot(logT, dem_lp, 'k-', linewidth=2, label='LP solver (ground truth)')
    ax.plot(logT, dem_direct, 'b--', label='Direct optim, no fit term')
    ax.plot(logT, dem_direct_fit, 'b-', label='Direct optim, fit term')
    ax.plot(logT, dem_nn_nf, 'r--', label='NN, no fit term (his setup)')
    ax.plot(logT, dem_nn_fix, 'r-', label='NN, fit term (our fix)')
    ax.set_xlabel("log T")
    ax.set_ylabel("DEM")
    ax.set_title("DEM curves — one pixel, X8.2 flare")
    ax.legend(fontsize=8)

    # Loss curves
    ax = axes[1]
    ax.plot(losses_nf, 'r--', label='NN, no fit term', alpha=0.7)
    ax.plot(losses_fix, 'r-', label='NN, fit term', alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/experiments/one_pixel.png", dpi=150)
    print("\nSaved to output/experiments/one_pixel.png")

    # ── resynthesis check ─────────────────────────────────────────────────────
    print("\nResynthesis MAE (lower = better reconstruction of AIA observations):")
    for name, dem in [("LP solver", dem_lp), ("NN no fit", dem_nn_nf), ("NN with fit", dem_nn_fix)]:
        resynth = R @ np.maximum(dem, 0)
        mae = np.mean(np.abs(resynth - obs))
        print(f"  {name:20s}  MAE={mae:.4f}")
