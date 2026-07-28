# Scaling plan — agreed with Samuel, 2026-07-28

Follows `results/meeting_20260728_scaling_brief.md` (pipeline validation + blockers).
This is the post-meeting version: what was decided and what to do next.

---

## 1. Decisions from the meeting

1. **NaN policy: mask, don't interpolate.** Infeasible-LP pixels (measured 6.6% of a
   full disk) stay NaN in the saved `.npz`. Carry a boolean validity mask into the Zarr
   and exclude those pixels from the loss. Interpolating would teach the NN to imitate
   `nnInterpNaN` rather than the solver.
2. **Scale in images, not pixels per image.** Diversity comes from more timestamps, not
   from more pixels off the same disk. `--decimate 2` (2048x2048) is fine.
3. **Model should be as small as possible.** Not a hard constraint, but parameter count
   has to be justified — efficiency is part of the claim, and the architecture has to
   make sense rather than just being big. Consistent with our LOO result (the patch
   CNN's in-sample edge over `mlp6` disappeared on unseen images).
4. **Extensive ablations**, specifically: does the required model size grow with the
   dataset? That is a capacity sweep crossed with dataset size, not a single run.

---

## 2. Experiment matrix

**Unsupervised only — no label-fitting runs.** Four runs, two architectures x two
differentiable losses:

|  | `barrier` loss | `enet` loss |
|---|---|---|
| `mlp6` | run | run |
| `cnn`  | run | run |

This matches everything done since June: the 2026-06-19 unsupervised runs, the 2026-07-02
ablation and the 2026-07-11 LOO study all minimise the loss directly rather than fitting
solver labels.

Each run is evaluated against its corresponding classical solver (barrier-loss runs vs
the BP/LP solver, enet-loss runs vs the ElasticNet solver) on held-out **timestamps**, not
held-out pixels of seen images.

Note the two distinct meanings of "BP"/"ENet", which are easy to conflate:

- **As solvers** — `solveLP` and `solveElasticNet` in `dataset/fullBP.py`, selected with
  `--fitfn lp` / `--fitfn elasticnet`. These produce ground-truth DEM labels. Now used
  only for *evaluation*.
- **As losses** — the differentiable `barrier` and `enet` objectives from the 2026-07-02
  ablation, which train the NN directly with no labels at all. These are what we train.

Note the two distinct meanings of "BP"/"ENet" here, which are easy to conflate:

- **As solvers** — `solveLP` and `solveElasticNet` in `dataset/fullBP.py`, selected with
  `--fitfn lp` / `--fitfn elasticnet`. These produce ground-truth DEM labels.
- **As losses** — the differentiable `barrier` and `enet` objectives from the 2026-07-02
  ablation, which train the NN directly with no labels at all.

### Capacity sweep (the "do we need a bigger model" question)

On the two best cells, sweep parameter count (roughly 0.2M / 0.5M / 1.5M / 5M) across
dataset sizes (roughly 50 / 200 / 600 / 1223 timestamps). The deliverable plot is
params on x, quality on y, one curve per dataset size. If the curves flatten at the same
place regardless of dataset size, the small model wins and we say so.

---

## 2b. Consequences of dropping supervised runs

Two things follow, and both should be settled before launching generation.

### Labels are now evaluation-only, so we need far fewer

Nothing trains on solver labels. They are used to score the trained models (sparsity vs
BP, Wasserstein vs BP) on held-out timestamps. So we need labels for **val + test (306
timestamps)**, plus a small train slice (~100) for in-sample sanity checks — not all 917
train timestamps.

That is roughly a 4x cut in both generation compute and storage: order 500 GB rather than
2 TB, which matters on a filesystem already at 97%. Revise the job list accordingly rather
than submitting the full 6,115.

### The stored noise realisations lose their purpose

The 5 realisations per timestamp (index 0 clean, 1-4 noisy at `--noisescale 0.5`) exist so
the network can learn to reproduce the *distribution* of solver outputs under photon
noise — the uncertainty head, one of the three claimed improvements over the original
DeepEM. That is inherently supervised: it fits the spread of BP's answers across noise
draws.

With unsupervised losses we would perturb the input on the fly and never read the stored
realisations. So going unsupervised-only either drops the uncertainty contribution, or
makes it a separate later supervised head trained on a small labelled subset. **This is a
decision to make deliberately with Samuel, not one that should fall out of the
training-loss choice.** If the uncertainty head stays in scope, keep the noisy
realisations for the val/test/subset timestamps we do generate.

---

## 3. ENet generation — open cost question

Running the ENet half doubles generation. On the reduced evaluation-only job list
(section 2b) that is roughly 800 jobs rather than 400, which is comfortable — the cost
concern largely goes away once we stop generating labels for the 917 train timestamps.

- **Measure first**: time a full-disk smoke run with `--fitfn elasticnet`. `sklearn`'s
  coordinate descent is typically much cheaper than the LP, so this may be nearly free —
  or it may not be. Decide after measuring.
- **Ask Samuel which hyperparameters he used.** Defaults in the script are
  `--fitlinearalpha 1` and `--fitlinearl1ratio 0.5`; ENet solutions are very sensitive to
  both, and the labels are meaningless if they do not match what he validated.
- Fallback if it is expensive: generate ENet labels for a subset (e.g. the 153-timestamp
  val split plus a matched 153 from train) to establish the comparison, and only scale it
  out if the ENet-target models look competitive.

---

## 4. Hofmeister PSFs

Files are several GB each, so they were not shared over Slack. Source:
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DYT4ZL`

Download on the **login node** (compute nodes have no internet), inside tmux, onto
scratch. Check the total size against the 5 TB quota before starting, since the labels
themselves are 2-4 TB.

```bash
cd $SCRATCH/dem/data && mkdir -p hofmeister_psf && cd hofmeister_psf
curl -L -O -J "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/DYT4ZL"
# then expose the six per-wavelength files where the code looks for them:
#   dataset/hofmeister_psf/psf_aia_{94,131,171,193,211,335}.fits
```

---

## 5. Ordered next steps

1. **Time `--fitfn elasticnet`** on the same full-disk smoke timestamp. Cheapest action,
   decides section 3.
2. **Download the PSFs** to scratch (login node, tmux). Then re-run the smoke test with
   `--deconvolve hofmeister` on a GPU node to confirm that path works end to end — it is
   the only major code path still unexercised.
3. **Fix the two script defects** before any large submission:
   - `submit_bp_aia_hofdeconv_full.py`: `SRC_DIR` ->
     `/projects/rps/dff6142/fouheylab/solar_dem/xrtSource`
   - `fullBP.py:790`: cache the SSW error table to a local file like the pointing and
     correction tables already are. As written it downloads on every job, which fails on
     internet-less compute nodes and would rate-limit at 6,115 jobs.
4. **Add NaN masking to `stage_hofdeconv_full.py`** (it currently has none) so the Zarr
   carries a validity mask alongside `Y`. Must land before any Zarr is built.
5. **Launch BP generation**, then staging.
6. Write the Zarr dataloader — blocked on confirming the AIA(4096) / DEM(2048) alignment
   convention and the 18-vs-26 bin decision (see brief sections 5.1, 5.2).

---

## 6. Still to confirm with Samuel

- ENet hyperparameters (`fitlinearalpha`, `fitlinearl1ratio`).
- 18-bin vs 26-bin output head for the AIA-only models (8 of the 26 bins are exactly
  zero by construction when XRT is absent).
- AIA/DEM resolution alignment convention for the dataloader.
- Whether the existing `xrtData_lp_full` / `lp_AIA_notrunc_noisy_full` outputs are
  superseded by the Hofmeister re-run or still usable.
- Who runs the generation jobs, and whose scratch they land on.
- **Whether the uncertainty head stays in scope** given unsupervised-only training
  (section 2b). This determines whether we keep generating noisy realisations at all.
