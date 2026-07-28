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

Two architectures x four training targets:

|  | supervised, BP labels | supervised, ENet labels | barrier loss (unsupervised) | enet loss (unsupervised) |
|---|---|---|---|---|
| `mlp6` | run | run | run | run |
| `cnn`  | run | run | run | run |

Each cell is evaluated against its corresponding classical solver (BP-target runs vs the
BP/LP solver, ENet-target runs vs the ElasticNet solver), on held-out **timestamps**, not
held-out pixels of seen images.

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

## 3. ENet generation — open cost question

Running the ENet half doubles generation: 12,230 jobs and roughly 4 TB rather than 2 TB,
on a scratch filesystem already at 97%.

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
