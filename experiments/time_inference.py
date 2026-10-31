"""Time the per-pixel network over one full DEM grid (2048 x 2048 pixels).

The paper compares this with the solver's minutes per image. The time of a
forward pass does not depend on the weights, so a checkpoint is optional.

    python experiments/time_inference.py --device cpu
    python experiments/time_inference.py --device cuda --ckpt <final BP checkpoint>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ablations import CenterMLP  # noqa: E402
from train_scaled import NormalizedInput  # noqa: E402

N_PIXELS = 2048 * 2048


def build(hidden, fourier_freqs, device, ckpt=None):
    core = CenterMLP(n_basis=54, patch_size=1, hidden=hidden, fourier_freqs=fourier_freqs)
    model = NormalizedInput(core, "sqrt").to(device).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        core.load_state_dict(state["model"], strict=False)
    return model


@torch.inference_mode()
def one_pass(model, obs, B, batch, device):
    """All pixels in batches: 54 weights, then the 18-bin DEM."""
    dems = 0.0
    for i in range(0, len(obs), batch):
        x = obs[i:i + batch].to(device, non_blocking=True)[:, :, None, None]
        w = model(x)
        dem = w @ B
        dems += float(dem[0, 0])   # keeps the work from being optimised away
    return dems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, nargs="+", default=[680, 336])
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n_pixels", type=int, default=N_PIXELS, help="timed sample; the result scales to one image")
    p.add_argument("--ckpt", default=None, help="optional, width-680 checkpoint")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(0)
    # Log-uniform intensities over the data's range; values do not change the time.
    obs = torch.from_numpy(10 ** rng.uniform(-3, 4, size=(args.n_pixels, 6))).float()
    B = torch.from_numpy(rng.random((54, 18))).float().to(device)

    out = {"device": args.device, "n_pixels": args.n_pixels, "batch": args.batch,
           "threads": torch.get_num_threads(), "runs": {}}
    if device.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(device)
    for hidden in args.hidden:
        model = build(hidden, 12, device, args.ckpt if hidden == 680 else None)
        n_params = sum(q.numel() for q in model.parameters())
        one_pass(model, obs[:args.batch], B, args.batch, device)   # warm start
        times = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_pass(model, obs, B, args.batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        rate = args.n_pixels / float(np.median(times))
        out["runs"][f"h{hidden}"] = {"params": n_params, "seconds": times, "pixels_per_s": rate,
                                     "seconds_per_image": N_PIXELS / rate}
        print(f"h{hidden}: {n_params:,} params, {N_PIXELS / rate:.1f} s per image, "
              f"{rate / 1e6:.3f} Mpix/s on {args.device}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
