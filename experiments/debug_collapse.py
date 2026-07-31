"""
Watch the first N steps of a scaled run to see how the network dies.

Array 15088220 logged a mean epoch-1 loss of 1.2e10 and then a bit-identical
339.2775 for eleven epochs. Block sampling is deterministic, so an identical
epoch mean means the weights stopped moving entirely -- the Softplus head is
saturated into the denormal range where its gradient is exactly 0.

experiments/diag_errors.py ruled out the data: no non-positive errors, and the
worst pixel in the set can only contribute 9.9e3 to the barrier at x=0. So the
1.2e10 has to come from the *initial* forward pass. This logs the quantities
that would show that -- input magnitude, |Dx| vs ub, the head pre-activation,
and the grad norm before clipping -- for the first few hundred steps.

    python3 experiments/debug_collapse.py \
        --root $SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS --steps 200
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import barrier_loss_batch
from src.zarr_data import make_loader, flatten_blocks, N_AIA_BINS
from experiments.train_scaled import load_operators
from experiments.train_neural_field import pick_device
from experiments.train_ablations import build_model, VARIANTS


def main(a):
    device = pick_device()
    D_t, B_t, n_basis, logT = load_operators(device)
    print(f"D: {tuple(D_t.shape)}  |D| min/med/max "
          f"{D_t.abs().min():.3e}/{D_t.abs().median():.3e}/{D_t.abs().max():.3e}")

    _, loader = make_loader(a.root, 'train', batch_blocks=a.batch_blocks,
                            num_workers=4, shuffle=True, with_labels=False,
                            max_blocks=a.max_blocks, seed=42,
                            pixels_per_block=a.pixels_per_block)

    model = build_model(a.variant, n_basis, 9, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # The last Linear before Softplus: its pre-activation is what underflows.
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    pre = {}
    head.register_forward_hook(lambda m, i, o: pre.__setitem__('z', o.detach()))

    print(f"\n{'step':>5} {'loss':>12} {'patch_max':>10} {'|Dx|max':>10} "
          f"{'ub_max':>9} {'z_min':>9} {'z_max':>9} {'x_min':>10} {'gnorm':>11}")

    step = 0
    for batch in loader:
        patch, obs, lb, ub = (t.to(device) for t in flatten_blocks(batch))
        opt.zero_grad()
        x = model(patch)
        loss = barrier_loss_batch(x, D_t, obs, lb, ub,
                                  a_l1=a.alpha_l1, a_l2=0.0, mu=a.mu)
        loss.backward()
        # Read the true gradient magnitude *before* clipping hides it.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()

        if step < 20 or step % a.every == 0:
            Dx = (x @ D_t.T).detach()
            z = pre['z']
            print(f"{step:>5} {loss.item():>12.4e} {patch.max():>10.3e} "
                  f"{Dx.abs().max():>10.3e} {ub.max():>9.3e} "
                  f"{z.min():>9.2f} {z.max():>9.2f} {x.min():>10.3e} "
                  f"{gnorm:>11.4e}")
        step += 1
        if step >= a.steps:
            break

    z = pre['z']
    dead = (z < -20).float().mean().item()
    print(f"\nfinal head pre-activation: min {z.min():.2f}  max {z.max():.2f}  "
          f"frac < -20 (Softplus effectively dead): {dead:.4f}")
    print("A z_min diving past about -90 is the collapse; float32 Softplus "
          "underflows there and its gradient becomes exactly 0.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--variant', choices=VARIANTS, default='mlp6')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--every', type=int, default=10)
    p.add_argument('--batch_blocks', type=int, default=8)
    p.add_argument('--pixels_per_block', type=int, default=512)
    p.add_argument('--max_blocks', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha_l1', type=float, default=1.0)
    p.add_argument('--mu', type=float, default=1.0)
    main(p.parse_args())
