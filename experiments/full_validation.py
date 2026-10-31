#!/usr/bin/env python3
"""Score models on every pixel of a staged split, clamped and relaxed pixels included.

The validation loss logged in training covers four images and, for runs that drop
clamped pixels, none of the inputs that fail. This pass covers all 153 images with
one objective for every model: the training loss with clamped channels freed. That
needs no solver DEM. With --labels it also scores the staged solver DEM, which holds
the solver's relaxed solutions, so it covers the pixels the shared test set leaves out.

    python3 experiments/full_validation.py --root $ROOT --phase val --loss barrier --mu 100 \
        --models final=<ckpt> old=<ckpt> --supervised <pth> --labels --output out.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.train_scaled import load_operators  # noqa: E402
from src.scaled_eval import load_scaled_model  # noqa: E402
from src.zarr_data import MIN_OBS  # noqa: E402

N_BINS = 18
BRIGHT = (3.1212, 16.9986, 448.8948, 498.3444, 190.7388, 10.3326)     # eval_and_enet_specs/README.md
GROUPS = ("all", "unclamped", "clamped", "feasible", "relaxed", "quiet", "bright", "flare_core")
LO, HI, STEP = -10.0, 16.0, 0.01
N_HIST = int((HI - LO) / STEP)


class Blocks(torch.utils.data.Dataset):
    """One staged block on the DEM grid: obs and err [6, D, D], tol [D, D], dem [18, D, D]."""

    def __init__(self, root, phase, labels, max_blocks):
        self.root, self.phase, self.labels = root, phase, labels
        self.n = zarr.open(f"{root}/{phase}_x.zarr", mode="r").shape[-1]
        if max_blocks:
            self.n = min(self.n, max_blocks)
        self.open = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.open is None:
            self.open = {s: zarr.open(f"{self.root}/{self.phase}_{s}.zarr", mode="r") for s in "xemy"}
        z = self.open
        out = [np.asarray(z["x"][:, ::2, ::2, i], dtype=np.float32),
               np.asarray(z["e"][:, ::2, ::2, i], dtype=np.float32),
               np.asarray(z["m"][:, :, i])]
        out.append(np.asarray(z["y"][:N_BINS, :, :, i], dtype=np.float32) if self.labels else np.zeros(0))
        return out


class Tally:
    """Running totals per group, on the device, so a pass stays a few minutes per model."""

    def __init__(self, device):
        self.d = {g: {k: torch.zeros(1, dtype=torch.float64, device=device) for k in
                      ("n", "loss", "sse", "ae", "em_pred", "em_ref", "aia_se", "wild", "n_dem")}
                  for g in GROUPS}
        self.h_loss = {g: torch.zeros(N_HIST, dtype=torch.int64, device=device) for g in GROUPS}
        self.h_sse = {g: torch.zeros(N_HIST, dtype=torch.int64, device=device) for g in GROUPS}
        self.max_loss = {g: 0.0 for g in GROUPS}

    @staticmethod
    def index(v):
        return ((torch.log10(v.clamp_min(1e-30)) - LO) / STEP).long().clamp(0, N_HIST - 1)

    def add(self, masks, loss, sse, ae, em_pred, em_ref, aia_se, wild, has_dem):
        for g, m in masks.items():
            t = self.d[g]
            t["n"] += m.sum()
            t["aia_se"] += aia_se[m].sum()
            t["wild"] += wild[m].sum()
            if loss is not None:
                t["loss"] += loss[m].sum()
                self.h_loss[g] += torch.bincount(self.index(loss[m]), minlength=N_HIST)
                if m.any():
                    self.max_loss[g] = max(self.max_loss[g], float(loss[m].max()))
            if sse is not None:
                k = m & has_dem
                t["n_dem"] += k.sum()
                t["sse"] += sse[k].sum()
                t["ae"] += ae[k].sum()
                t["em_pred"] += em_pred[k].sum()
                t["em_ref"] += em_ref[k].sum()
                self.h_sse[g] += torch.bincount(self.index(sse[k]), minlength=N_HIST)

    def result(self):
        def pct(hist, q):
            c = hist.cumsum(0).double()
            if c[-1] == 0:
                return None
            return float(10 ** (LO + STEP * (torch.searchsorted(c, q * c[-1]).item() + 0.5)))

        out = {}
        for g in GROUPS:
            t = {k: float(v) for k, v in self.d[g].items()}
            n, nd = max(t["n"], 1), max(t["n_dem"], 1)
            r = {"n": int(t["n"]), "wild": int(t["wild"]), "aia_mse": t["aia_se"] / (6 * n)}
            if self.h_loss[g].sum() > 0:
                r["loss"] = {"mean": t["loss"] / n, "max": self.max_loss[g],
                             **{f"p{q}": pct(self.h_loss[g], q / 100) for q in (50, 90, 99, 99.9, 99.99)}}
            if t["n_dem"] > 0:
                r["dem"] = {"n": int(t["n_dem"]), "mse": t["sse"] / (N_BINS * nd), "mae": t["ae"] / (N_BINS * nd),
                            "em_ratio": t["em_pred"] / max(t["em_ref"], 1e-30),
                            **{f"sse_p{q}": pct(self.h_sse[g], q / 100) for q in (50, 90, 99, 99.9)}}
            out[g] = r
        return out


def label_free(path, n_basis, device):
    model, ckpt = load_scaled_model(path, n_basis, device)
    if ckpt["variant"] != "mlp6":
        raise ValueError(f"only mlp6 is supported, got {ckpt['variant']}")
    k, c = ckpt["patch_size"], ckpt["patch_size"] // 2

    def predict(obs):
        patch = torch.zeros(len(obs), 6, k, k, device=device)
        patch[:, :, c, c] = obs
        return model(patch)
    return predict


def supervised(path, device):
    from student_package.table1_dem_metrics.compute_paper_table_metrics import load_model, predict_regression_only
    model, _ = load_model(path, device)

    def predict(obs):
        return predict_regression_only(model, obs.T[None, :, None, :])[0, :N_BINS, 0, :].T
    return predict


@torch.no_grad()
def score(predict, is_sup, loader, D, B, R, args, device):
    tally = Tally(device)
    thr = torch.tensor(BRIGHT, device=device)
    for obs, err, tol, dem in loader:
        obs, err, tol = (torch.as_tensor(a, device=device) for a in (obs, err, tol))
        ok = (torch.isfinite(obs).all(0) & torch.isfinite(err).all(0) & (tol > 0)).flatten()
        if not ok.any():
            continue
        o = obs.reshape(6, -1).T[ok]
        h = args.tolfac * err.reshape(6, -1).T[ok]
        t = tol.flatten()[ok]
        free = o <= MIN_OBS
        o64, h64 = o.double(), h.double()
        if is_sup:
            x = predict(o.clamp_min(0)).double().clamp_min(0)
            rec = x @ R.T
            loss = None
        else:
            w = predict(o).double()
            x = w @ B.T
            rec = w @ D.T
            if args.loss == "barrier":
                v = (torch.abs(rec - o64) - h64).clamp_min(0)
                loss = w.sum(1) + args.mu * ((v / h64) ** 2).masked_fill(free, 0).sum(1)
            else:
                fit = (((rec - o64) / h64) ** 2).masked_fill(free, 0).sum(1) / (2 * args.enet_C)
                loss = fit + args.enet_alpha * args.enet_lam * w.sum(1) \
                    + 0.5 * args.enet_alpha * (1 - args.enet_lam) * (w ** 2).sum(1)
        inten = (o / thr).amax(1)
        clamped = free.any(1)
        masks = {"all": torch.ones_like(clamped), "unclamped": ~clamped, "clamped": clamped,
                 "feasible": t == 1, "relaxed": t > 1, "quiet": inten < 1,
                 "bright": (inten >= 1) & (inten < 32), "flare_core": inten >= 32}
        wild = ((rec > 1e3) & (rec > 100 * o64.clamp_min(1))).any(1)
        aia_se = ((rec - o64) ** 2).sum(1)
        sse = ae = pe = re = has = None
        if args.labels:
            y = torch.as_tensor(dem, device=device).reshape(N_BINS, -1).T[ok].double()
            has = torch.isfinite(y).all(1)
            y = torch.nan_to_num(y)
            sse = ((x - y) ** 2).sum(1)
            ae = (x - y).abs().sum(1)
            pe, re = x.sum(1), y.sum(1)
        tally.add(masks, loss, sse, ae, pe, re, aia_se, wild, has)
    return tally.result()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--phase", default="val")
    p.add_argument("--loss", choices=("barrier", "enet"), default="barrier")
    p.add_argument("--mu", type=float, default=100.0)
    p.add_argument("--tolfac", type=float, default=1.4)
    p.add_argument("--enet_alpha", type=float, default=0.001)
    p.add_argument("--enet_lam", type=float, default=0.5)
    p.add_argument("--enet_C", type=float, default=6.0)
    p.add_argument("--models", nargs="+", default=[], help="label=checkpoint")
    p.add_argument("--supervised", default=None)
    p.add_argument("--labels", action="store_true", help="also score the staged solver DEM")
    p.add_argument("--max_blocks", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - args.workers))
    D_t, B_t, n_basis, _ = load_operators(device)
    D, B = D_t.double(), B_t.double()       # [6, 54] and [18, 54]
    R = torch.as_tensor(np.load("RData.npz")["R"] * 10 ** 26, device=device).double()
    data = Blocks(args.root, args.phase, args.labels, args.max_blocks)
    print(f"{args.phase}: {len(data):,} blocks, device {device}", flush=True)

    out = {"settings": vars(args), "models": {}}
    runs = [m.split("=", 1) for m in args.models]
    if args.supervised:
        runs.append(["supervised", args.supervised])
    for label, path in runs:
        t0 = time.time()
        is_sup = label == "supervised"
        predict = supervised(path, device) if is_sup else label_free(path, n_basis, device)
        loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=args.workers, prefetch_factor=4)
        out["models"][label] = score(predict, is_sup, loader, D, B, R, args, device)
        a = out["models"][label]["all"]
        print(f"{label}: {a['n']:,} px, loss {a.get('loss')}, DEM {a.get('dem')}, wild {a['wild']}, "
              f"{time.time() - t0:.0f} s", flush=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
