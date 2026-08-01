# From Samuel's solver to four trained networks — the complete record

**Date**: 2026-07-31 (single day, ~14 h wall)
**Outcome**: four networks trained and converged on 1,223 timestamps; first result on
entirely unseen images.

This is the step-by-step account: what we started from, what broke, what we fixed, what we
merely assumed, and how each conclusion was actually established. Read top to bottom.

---

## Step 0 — The starting point

Samuel's `dataset/fullBP.py` is a working per-pixel DEM solver. Given a calibrated AIA
observation `o` and its uncertainty `sigma`, it finds a DEM `x >= 0` satisfying

```
lb = o - 1.4*sigma   <=   Rx   <=   o + 1.4*sigma = ub
```

**This band is the single most important object in the project.** Everything downstream —
the NaN problem, the loss function, the training collapse — is about it. The solver does
not demand an exact fit; it demands a fit *within noise*, and among all DEMs inside that
band, Basis Pursuit picks the **sparsest** one. Elastic Net solves a related regularised
problem instead.

`R` is the temperature response matrix: how strongly each AIA channel responds to plasma at
each temperature.

Our job was to scale this from the 4 timestamps / 128x128 crops the networks had been
validated on, to the full 2-year, 1,223-timestamp dataset at full disk.

---

## Step 1 — A framing decision that removed most of the difficulty

Training is **unsupervised**: the network minimises the barrier (BP-like) or ENet objective
*directly against the observations*, and never reads the solver's answer.

Consequences, all good:

- Solver labels are **evaluation-only** — we score the network against them afterwards.
- Storage drops from ~4.6 TB to ~2 TB.
- No need to generate noisy realisations (those exist only to train an uncertainty head).

The four runs are `{mlp6, cnn} x {barrier, enet}`. **All four are AIA-only** — the BP/ENet
distinction lives entirely in the loss, not in the inputs. This confuses people; say it
explicitly.

---

## Step 2 — Four bugs in the job scripts, each fatal to all 1,223 jobs

Found by reading, before launching anything:

1. Singularity overlay path hardcoded to one that only Samuel has.
2. `DATASET_DIR` derived from `BASH_SOURCE`, which under `sbatch` resolves to the SLURM
   **spool copy**, not the repo. Fixed with a `SLURM_SCRIPT` -> `SLURM_SUBMIT_DIR` ->
   `BASH_SOURCE` fallback chain (`3479d8e`).
3. `--account` / `--mail-user` hardcoded, so every collaborator carries a local edit that
   blocks `git pull`. Now passed on the sbatch line (`95edc08`).
4. `DATA_ROOT` in the staging script pointing at Samuel's scratch.

**Lesson**: these cost minutes to find by reading and would have cost hours to find by
running.

---

## Step 3 — The `--zerochill` finding (the most important scientific result of the day)

**Symptom**: BP returned NaN for **10.2%** of pixels — and disproportionately the good ones:
13.4% on-disk, **17.7% in the brightest decile**. It was discarding active regions and
flares, precisely the scientifically valuable pixels.

**Two hypotheses falsified first** (this matters — we did not guess):

- *Deconvolution positivity clamp causing it*: `P(NaN | zeroed pixel)` = 10.5% vs 9.6%
  baseline. No relationship.
- *Off-limb low signal*: predicts a higher rate off-disk. Measured rate is **higher
  on-disk**. Backwards.

**The actual mechanism, and it is a band effect.** Photon noise gives `sigma ~ sqrt(counts)`.
So as a pixel brightens, `sigma` grows like `sqrt(o)` while `o` grows like `o`, and the
*relative* band `1.4*sigma/o` **shrinks**. Past some brightness it is narrower than the
response matrix `R`'s own accuracy — and then **no DEM exists inside it**, so the LP
correctly reports infeasible. This is model mismatch at high signal-to-noise, not bad data.

`--zerochill` disabled BP's retry-at-wider-tolerance schedule. Dropping the flag:
**10.2% -> 0.02% NaN, for +90 s/job.**

**This is where `tolLevel` comes from.** Each pixel now records the band it was solved at:

| tolLevel | meaning | share of final data |
|---|---|---|
| 1 | tight band | 89.78% |
| 3 | relaxed 3x | 10.04% |
| 5 | relaxed 5x | 0.16% |
| 0 | never solved (DEM is NaN) | 0.02% |

It is both a validity mask **and** a label-quality signal — a level-3 label satisfied a much
weaker constraint, so the network is scored separately on level 1 vs 3/5.

**Still forced for AIA+XRT** by the assert at `fullBP.py:691` ("XRT errors are currently
iffy"). Worth asking Samuel whether that is still true, since it is the exact flag that was
destroying 10% of pixels.

---

## Step 4 — Two supposed blockers resolved from the code, not from Samuel

**"AIA is 4096x4096, DEM is 2048x2048 — how do they correspond?"**
`fullBP.py:979` is `AIACube[:, ::2, ::2]` — plain subsampling, no averaging. So **DEM pixel
(i,j) came from AIA pixel (2i,2j) exactly**, and 3 of every 4 stored AIA pixels were never
seen by the solver.

**"18 output bins or 26?"**
`fullBP.py:1013-1019` stacks 8 rows of zeros onto 18 real bins for AIA-only. A 26-bin head
would spend capacity on guaranteed zeros and corrupt the Hoyer sparsity denominator.
**18 for AIA-only**, 26 only when XRT is present. Confirmed `temps: 18` at runtime.

---

## Step 5 — Label generation

Two arrays on Torch, 1,223 timestamps each, AIA-only, Hofmeister-deconvolved, clean.

**Result: 1,223/1,223 files per solver, zero errors, 888 GB each (1.74 TB total).**
BP 9.5 min/job, ENet 5.25 min/job (the LP is ~2x slower than ENet).

**Hofmeister deconvolution validated end to end** by channel-wise change: 94A 32%, 131A 29%,
335A 35% versus 171/193/211A 8-12%. Largest corrections on the faintest channels is the
correct scattered-light signature — so the deconvolution is genuinely doing something, not a
no-op.

---

## Step 6 — Staging to Zarr

Repacked the 1,223 `.npz` files into chunked arrays: `_x` (AIA), `_e` (errors), `_y` (DEM
labels), `_m` (tolLevel). ~16-25 min per solver.

**`_e` was a hard blocker.** The first staging carried observations and labels but not
errors — and without `sigma` you cannot build `lb`/`ub`, so the unsupervised losses cannot
be computed at all. Fixed in `d3c58d7` before launch.

**Split: 917 train / 153 val / 153 test — by file, i.e. by timestamp.** This matters
enormously for interpreting the final numbers: validation is on **entirely unseen images**,
unlike the 2026-06-27 amortized run which held out *pixels of seen images*. The test split
remains untouched.

**Scheduling lesson**: 32 CPU / 180 GB jobs sat in `(Priority)` indefinitely while
16 CPU / 64 GB jobs started instantly. `--partition=cpu_short` must be requested explicitly
(the account only holds QOS `normal`).

---

## Step 7 — The dataloader (`src/zarr_data.py`)

**A dataset item is one block, not one pixel.** Zarr chunks are `(..., 1)`, so reading a
single pixel decompresses its entire 256x256 block. Each item therefore samples 512 pixels
from one block and the training loop flattens. Per-pixel indexing would re-decompress the
same chunk thousands of times per epoch.

**Alignment was verified, not assumed.** The loader subsamples AIA by 2 *first*; after that
AIA and DEM share one grid and patches are plain contiguous slices — the same nine pixels as
a stride-2 patch on the 4096 grid, with far less indexing to get wrong.
`tests/test_zarr_data.py` stamps each synthetic AIA pixel with its own coordinate so a
misaligned patch fails loudly rather than training happily on the wrong pixel. Real staged
data passes at `max deviation 0.000e+00`.

---

## Step 8 — The deconvolution positivity clamp zeroes the *brightest* channels

**~4.4% of pixels carry a hard zero in 171A or 193A** — the two brightest channels, medians
131 and 145 DN. Faintness cannot produce that; the positivity clamp applied after Hofmeister
deconvolution can (deconvolution can drive values negative, which are then floored at zero).

**And `tolLevel` does not catch them**: BP returns a feasible DEM anyway, 1,574 at tolLevel 1
versus only 195 relaxed in a 40k sample. So the label is quietly fitted to a detector
artifact.

Fixed with `MIN_OBS = 1e-3` — three decades below the faintest channel's median (94A, 0.76),
so it cannot clip real signal — costing ~5% of pixels (`866991d`).

Note this is *related to but distinct from* the falsified NaN hypothesis in Step 3: the
clamp does not cause NaNs, it zeroes bright channels.

---

## Step 9 — Two mismatches caught before running rather than after

**ENet constants.** Generation used `--fitlinearalpha 1 --fitlinearl1ratio 0.5`, but the
training default was `lam=0.9, C=1`. `fullBP.solveElasticNet` is
`1/(2N)||Dx-y||^2 + a*l*||x||_1 + a(1-l)*0.5*||x||^2` and scales `D,y` by
`tol = meas - lb` (= our sigma), so the loss *form* already matched — only the constants
were wrong. Now `alpha=1, lam=0.5, C=n_obs` (`685c33c`).

**Sparsity metric space.** Every historical number — ablation BP 1.79, cnn 1.70, mlp6 1.89 —
is Hoyer sparsity on the **54 basis coefficients**, and "MAE" meant **AIA resynthesis
error**. The new script measured both on the 18-bin DEM, making `ref 5.32` look alarming
when it is simply a different quantity. Now reports both, labelled (`a987566`).

---

## Step 10 — Training attempt 1: total collapse

Array `15088220`. All four runs **completed with exit 0** and produced **worthless all-zero
networks**.

**The signature**: mean epoch-1 loss `1.18e10`, then a **bit-identical** `339.2775` for
eleven epochs with `sp_coef=0.00`.

**Why bit-identical is the key clue**: block sampling is deterministic (`rng` seeded per
block index), so epochs differ only in *ordering*. An identical epoch mean therefore proves
gradients are **exactly zero**, not merely small. That single observation distinguishes a
dead network from a slowly-converging one, and it is worth remembering.

---

## Step 11 — Two more wrong hypotheses, both falsified by measurement

Before changing any code (`experiments/diag_errors.py`, 655k pixels):

1. **"`x=0` is genuinely feasible and L1-optimal."** If `lb <= 0`, the zero DEM sits inside
   the band and L1 drives it to exactly zero. **Falsified**: `barrier_lb` at `x=0` has mean
   408 and maximum **9.9e3** across 655k pixels. The data cannot produce 1e10. (Also, the
   converged 339.28 is *below* 408.12, so the network sits just under true zero, not at it.)
2. **"Some staged errors are zero, blowing up the barrier's `sigma^2` denominator."**
   **Falsified**: *zero* pixels have a non-positive error in any channel, and per-channel
   `sigma/obs` is physical:

   | channel | 94A | 131A | 171A | 193A | 211A | 335A |
   |---|---|---|---|---|---|---|
   | median sigma/obs | 1.12 | 0.44 | 0.074 | 0.057 | 0.082 | 0.601 |

   The faint channels genuinely are noise-dominated; the bright ones are not, and they are
   what pins `lb > 0`.

---

## Step 12 — The real cause, in two halves

`experiments/debug_collapse.py` instrumented the first 200 steps.

**Half 1 — input scale.** At step 0, before any weight update:

| step | loss | patch_max | \|Dx\|max | ub_max | z_min | grad norm |
|---|---|---|---|---|---|---|
| 0 | 5.3e13 | 8.8e3 | 5.4e4 | 7.9e3 | **-139.8** | 2.0e13 |
| 60 | 2.1e2 | 1.5e3 | 9.7e1 | 1.5e3 | -13244 | 3.3e4 |

The architectures were validated on 128x128 **on-disk crops** spanning ~1 decade of
brightness. Full disk spans ~6 (1e-3 off-limb to ~1e4 flare cores). In float32,
`softplus(z)` underflows below `z ~ -90` where its gradient is **exactly 0** — and `z_min`
is already -139.8 on the first forward pass. Gradient clipping does not help, because Adam
discards gradient magnitude anyway. By step 60, **98.15% of outputs were dead**.

**Fix**: `log1p` on the network input only — loss, `lb`/`ub`, `D` and every metric stay in
physical DN, so results remain comparable to the crop runs (`d5f8ae5`).

**Half 2 — output scale. This one I missed, and the next run collapsed again.**

Array `15091457` had `log1p` and still died at `1.6e11`. The reason: `softplus(0) = 0.693`,
so at default initialisation **all 54 basis coefficients start near 0.7**. And:

```
sum|D| per channel = [13.2, 117.6, 2541.4, 1478.7, 415.0, 33.5]
```

So `Dx` for 171A starts at `0.7 x 2541 ~ 1780` against a typical `ub ~ 155` — **a 10x
overshoot before the network has seen anything**. The barrier squares the excess and divides
by `sigma^2`, which is small for the faint channels.

**Fix**: initialise the output head with `bias = -3` and weights scaled by 0.01, so the
initial prediction is `softplus(-3) ~ 0.049` and `|Dx|` max becomes **123** against
observations of 141-175 — inside the band (`8155c24`). Verified numerically for both
variants *before* submitting.

**Honest note**: this arithmetic takes thirty seconds and should have been done before the
`log1p` submission, not after. Fixing input conditioning without checking output scale cost
one full run.

---

## Step 13 — Everything that was changed, in one list

| Change | Commit | Why |
|---|---|---|
| Per-pixel `tolLevel` saved with every DEM | `b565cbd` | label-quality signal + validity mask |
| Drop `--zerochill` | (flag) | 10.2% -> 0.02% NaN |
| Job script portability (4 bugs) | `3479d8e`, `95edc08` | would have killed all 1,223 jobs |
| Carry `AIAErrors` into zarr | `d3c58d7` | unsupervised losses need `lb`/`ub` |
| Zarr dataloader + training script | `44291e2` | block-not-pixel sampling |
| Subsample AIA first + alignment test | `e8e7107` | fewer indices to get wrong; fails loudly |
| ENet constants match the solver | `685c33c` | `alpha=1, lam=0.5, C=n_obs` |
| `MIN_OBS = 1e-3` | `866991d` | deconvolution clamp zeroes bright channels |
| Report sparsity in both spaces | `a987566` | comparability with all historical numbers |
| `log1p` input, warmup, clamped Softplus | `d5f8ae5` | input-scale collapse |
| In-band output-head init | `8155c24` | output-scale collapse |
| Collapse guard (relative, not exact) | `a034bb0` | fires at epoch 1 instead of never |

**Four submissions were needed**: `15088220` and `15090865` ran pre-fix code (the repo was
not pulled on Torch before submitting — a real and easily repeated mistake); `15091457` had
`log1p` but not the init fix; `15091734` got 3/4, with cnn+enet still dying at init.
Raising warmup 500 -> 3000 rescued it, and `15092854` ran all four with identical settings.

---

## Step 14 — The result

Array `15092854`, 40 epochs, 3,000-step warmup, ~1.5 h each. All `COMPLETED`, all converged
(`val_loss` flat to 4 dp over the final 4 epochs).

| variant | loss | sp_coef | sp_dem nn/ref | val_loss | mae_aia |
|---------|------|---------|---------------|----------|---------|
| **mlp6** | barrier | **1.90** | 5.16/5.37 | **2.145** | 4.881 |
| cnn | barrier | 1.96 | 5.10/5.37 | 2.241 | 4.838 |
| **mlp6** | enet | 3.68 | 5.24/5.36 | **1.850** | 4.613 |
| cnn | enet | 3.61 | 5.14/5.36 | 1.858 | 4.745 |

BP reference: **1.79** (on the 54 basis coefficients).

**Reading it:**

- **Sparsity is in the right place.** 1.90 and 1.96 against BP's 1.79 — the networks find
  genuinely BP-like sparse solutions on the full dataset, not merely *some* solution that
  fits. `sp_dem` 5.10-5.16 against the solver's own 5.37 says the same in DEM-bin space.
- **The patch CNN's edge is absent.** mlp6 is closer to BP and wins the actual training
  objective under both losses. This is the **fourth** independent look (ablation in-sample,
  LOO held-out, the 30-epoch run, now converged at scale) and the third finding no CNN
  advantage. **mlp6 — 1.43M parameters, single pixel — is the production architecture.**
- **These are 153 unseen timestamps**, not held-out pixels of seen images. Strongest
  evidence the project has produced on this question.
- **ENet behaves exactly as the 2026-07-02 ablation predicted**: best fit, least sparse.
  The L2-smoothing tradeoff by design, not a defect.
- **The units question is resolved.** Losses at 2.1-2.8 versus the crop runs' 1.43 — same
  ballpark. The earlier 345k was the broken initialisation, not a unit mismatch between the
  staged `AIACube` and `processIndAIAData`.

---

## What is verified vs what is assumed

**Verified by measurement**: AIA/DEM alignment; 18 bins; the NaN mechanism and its fix;
tolLevel distribution; Hofmeister deconvolution; the deconvolution clamp; that the staged
errors are clean; both halves of the collapse; convergence of all four runs.

**Assumed / open**:

- **ENet hyperparameters** — these runs used the generation defaults (`alpha=1, lam=0.5,
  C=n_obs`). If Samuel validated different values, the ENet labels need regenerating
  (~2.5 h + 40 min restage) and runs 2/3 repeating. **Highest-priority question for him.**
- Whether `fullBP.py:691`'s "XRT errors are currently iffy" still holds — it forces
  `--zerochill` for AIA+XRT, the exact flag that was destroying 10% of pixels.
- Whether AIA+XRT is in this round, and how many timestamps have usable XRT coverage.
- Whether the uncertainty head stays in scope given unsupervised-only training (it is the
  only reason to generate noisy realisations).

---

## Operational lessons worth keeping

- **`git pull` on the cluster before every submission.** Two of four submissions re-ran old
  code and reproduced an already-fixed failure exactly.
- **A small smoke test cannot catch a scale-dependent failure.** The 64-block / 2-epoch
  shakeout showed healthy training (345k -> 195k, sp 8.25) because it never drew the bright
  pixels that detonate the barrier.
- **Check the first forward pass numerically before submitting.** `|Dx|` vs `ub` at
  initialisation would have caught both halves of the collapse in seconds.
- **A bit-identical loss across epochs means exactly-zero gradients**, given deterministic
  sampling. Fastest possible diagnosis of a dead network.
- **Keep all variants on identical settings** when the run exists to compare architectures.
  A per-run hyperparameter rescue invalidates the comparison.
- **An epoch is ~170 s** over all 58,688 blocks (the 46 min estimated from the shakeout was
  startup-dominated). Keep GPU jobs **under 2 h** — beyond that they fall under the
  utilisation policy, and this workload is data-loading bound.
- **Torch has a job-submit routing plugin**: submit with no `--partition` and it places you
  across whatever pools are free. Better than hand-picking.

---

## Next steps

1. Evaluate on the untouched 153-timestamp **test** split.
2. Per-pixel DEM curve plots versus BP for the paper; copy into
   `results/plots/10_scaled_20260731/`.
3. Settle the ENet hyperparameters with Samuel; regenerate if needed.
4. Then: AIA+XRT scope, and the uncertainty head decision.
