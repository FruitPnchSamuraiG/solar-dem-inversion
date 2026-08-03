# From four trained networks to a production model — the complete record

**Dates**: 2026-08-02 to 2026-08-03
**Outcome**: the four scaled models validated on genuinely unseen data; two headline
metrics found to be artifacts and corrected; bimodality established as real, common and
learnable; a 176k-parameter model (8x smaller) chosen as production.

Continues `scaling_journey_20260731.md`, which ends at Step 14 with four converged
networks. Same format: what we did, what broke, what we fixed, and how each conclusion was
actually established.

---

## Step 15 — What was still missing after the four runs converged

The 2026-07-31 result (mlp6 `sp_coef` 1.90 vs cnn 1.96, BP reference 1.79) came from the
**validation** split. Validation also chose which epoch's checkpoint to keep, so those
numbers are mildly optimistic by construction. Two things had never been done:

1. **Test-split evaluation** — 153 timestamps held out separately, never touched by
   training *or* checkpoint selection.
2. **A bimodal census against the scaled models.** The only prior bimodal result
   (2026-07-11) used 4 small crops and the pre-scaling patch CNN, and concluded bimodality
   was rare (5-9%) and "mostly noise-driven degeneracy". Nothing about that carries over.

New code: `src/scaled_eval.py`, `experiments/eval_scaled.py`,
`experiments/bimodal_scaled.py`, `experiments/job_eval_scaled.sbatch`.

---

## Step 16 — One thing asserted rather than assumed

The per-pixel figures show the same pixel twice, BP's answer on the left and ENet's on the
right. That only means anything if block *k*, row *i*, column *j* is the **same pixel** in
both stagings. Both were staged from the same 1,223 files with the same split, so it should
hold — but if the two staging jobs had ordered their file lists differently, the figure
would silently compare two unrelated pixels and still render perfectly.

`assert_same_observations` opens random blocks from both `_x` arrays and requires
`max|a-b| == 0`. It passed (`max deviation 0.000e+00`). One assertion, and it is the
difference between the figure being evidence and being decoration.

---

## Step 17 — First run: a one-line plumbing bug

`make_loader` returns `(dataset, loader)`. Binding the pair to `loader` meant iterating a
2-tuple, so `evaluate()` received the Dataset itself as its first "batch". Died in 24 s
with `AttributeError: 'tuple' object has no attribute 'flatten'`. Fixed in `8b7f86c`.

Worth noting what had already passed before it died: the alignment assertion, all four
checkpoints rebuilding with `input=log1p`, and the operators (`D: (6, 54)`, 54 basis,
18 temps). The failure was in the first code that touched data.

---

## Step 18 — The test split confirms the models generalise

| run | sp_coef test/val | mae_aia test/val |
|---|---|---|
| mlp6 barrier | 1.91 / 1.90 | 4.875 / 4.881 |
| cnn barrier | 1.98 / 1.96 | 4.830 / 4.838 |
| mlp6 enet | 3.64 / 3.68 | 4.609 / 4.613 |
| cnn enet | 3.57 / 3.61 | 4.699 / 4.745 |

153 unseen timestamps, ~5.0M pixels, and nothing moves past the third decimal. **No
overfitting.** This is the number that makes everything downstream trustworthy.

---

## Step 19 — But `test_loss` said mlp6 was 4x worse, contradicting validation

mlp6/barrier scored **44.32** against cnn's **11.27**, where validation had mlp6 *ahead*
(2.145 vs 2.241). A 20x jump in the objective while `sp_coef` and `mae_aia` agree to the
third decimal cannot be a real regression — a model that got worse would move the other
metrics too.

**Hypothesis 1 (wrong): a near-zero error bar in the denominator.** The barrier is
`sum relu(violation)^2 / sigma^2` with `sigma = 1.4*err`, so a tiny staged error would
divide by nearly nothing. The July `diag_errors.py` had checked errors were never
*non-positive*; it never asked whether some were merely *tiny*.

`experiments/diag_loss_outliers.py` dumps the worst pixels. The worst one:

```
211A  obs=0.062  err=0.611  band=[-0.793, 0.917]  Dx=9023
```

The denominator is perfectly healthy. The **numerator** is a 9,000x over-prediction at a
pixel carrying no signal. Flooring sigma at 0.5-10% of the observation moves the mean by
**0.07%**. Hypothesis dead.

**Hypothesis 2 (also wrong): filter out pixels whose band straddles zero.** When
`obs < 1.4*err` the lower bound goes negative and the constraint degenerates to "emit less
than ub" — no lower information, but unbounded loss upward. Sharp criterion, and it catches
**89% of all pixels**. AIA's faint channels are genuinely noise-dominated: 94A has a median
of ~0.76 DN against a read-noise floor of the same size (`err` is *exactly* 0.6103 in 211A
and 0.8749 in 193A for many pixels). Not a usable filter.

**The actual answer.** mlp6's total loss over 5,013,504 pixels is 2.22e8. The single worst
pixel is **2.09e8 — 94.0% of it.** One pixel in five million decided the ranking. On
percentiles mlp6 is better *everywhere*:

| | median | p90 | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|
| mlp6 | **0.984** | **4.61** | **17.6** | **49.2** | **148** |
| cnn | 1.007 | 4.72 | 18.4 | 57.2 | 255 |

**Conclusions**: report percentiles, never the mean, for this objective. And **no retrain
needed** — at ~1 pathological pixel per epoch with gradient clipping already bounding its
step, the trained models are unaffected.

---

## Step 20 — The bimodal census, and a suspicion that turned out to be wrong

Counting peaks in the **stored** BP solutions costs nothing — the solver already ran on
every pixel — so prevalence came from 190,337 held-out pixels instead of the earlier
1,000-per-timestamp scans.

**15.96% bimodal**, against 2026-07-11's 5.1-9.1%. Rising monotonically with brightness
(7.4% faintest decile -> 30.3% brightest) and concentrated at `tol=1` (16.87% vs 0.38% at
tol=3), so not an artifact of relaxed solver tolerances.

**The suspicion**: looking at the plotted curves, many "second peaks" sat at the extreme
ends of the temperature range (logT 5.5 or 7.2). `count_peaks` zero-pads both ends so a
curve merely *rising* into a boundary bin registers a peak there — and those are exactly
where AIA's six channels have almost no discriminating power, a known inversion artifact.

**Falsified.** Splitting interior from boundary peaks: only **11.2%** of the bimodal set is
boundary-only. **14.17% of all pixels have genuine interior two-peak structure.** The
suspicion was reasonable and the check was cheap; the result got *stronger*, not weaker.

Stability under photon noise also rose sharply: mean **0.68** and 20/60 stable, against
0.40-0.46 and 6/120 on the crops. The 2026-07-11 "mostly noise-driven degeneracy"
conclusion does not survive contact with full-disk data at scale.

---

## Step 21 — Can the networks *predict* bimodality?

Recall alone cannot answer this: a network that emitted two peaks everywhere would match BP
at every bimodal pixel and be worthless. So the census scores every network over **all**
190,337 pixels, giving false positives and hence precision.

Interior-only, base rate 14.17%:

| network | firing rate | recall | precision |
|---|---|---|---|
| mlp6_barrier | 5.24% | 29.85% | **80.75%** |
| cnn_barrier | 2.01% | 11.88% | 83.70% |
| mlp6_enet | 3.21% | 7.90% | 34.87% |
| cnn_enet | 2.22% | 6.19% | 39.54% |

**When mlp6 says "two peaks", BP agrees ~4 times in 5.** This contradicts the 2026-07-10
meeting's working assumption that amortised point-estimate training makes this impossible
(the argument being that a deterministic network must average across pixels with
near-identical AIA inputs but different BP optima). It does not. That weakens the case for
an immediate distribution-output-head redesign.

**The enet rows are not a deficiency and should not be read as one.** ElasticNet's L2 term
explicitly penalises multi-peaked solutions, so an enet-trained network correctly
reproducing its own solver *should* rarely predict two peaks. "Can it predict bimodality"
is only a fair question for the BP-trained pair.

At the noise-stable pixels, mlp6_barrier beats **BP's own barrier loss in 20/20** (mean
-12.9) — where it answers unimodally, the two solutions are degenerate and BP simply broke
the tie differently.

---

## Step 22 — A plot that illustrated the wrong claim

The figure was drawn from the raw `npk>=2` hit set — including the 11.2% boundary
artifacts — while the headline numbers are about interior bimodality. Spotted by eye from
the figure itself, confirmed in the code, fixed in `8d9dc5a` (phases 2/3 now select from
interior-only hits).

Regenerated. Even so the figure is *illustrative, not the evidence*: 8 examples chosen by
brightness and stability, of which two are textbook clean, several have their second peak
one bin in from the boundary, and one shows mlp6 visibly failing to resolve a real double
peak. The 80.75% over 190,337 pixels is the evidence; the plot is a sanity check on what
that number looks like.

---

## Step 23 — The width sweep

mlp6 at 8 widths (1.43M / 722k / 360k / 176k / 87k / 42k / 20k / 10k) x both losses,
40 epochs each to match the baseline exactly. Array `15185224`, capped `%4` because the
account's group GPU quota — not node availability — is the binding constraint.

Wall time does **not** fall with width: the workload is data-loading bound at ~170 s/epoch
regardless of model size, so every task costs the same ~1.5 h.

**Two failures, both understood.** Task 5 hit the wall clock at epoch 33/40 (rerun with
more time). Task 9 (h480 enet) died at epoch 1 with an identically-zero prediction — the
sweep script omitted `--warmup_steps 3000` and fell back to the default 500, **reproducing
the July `15091457` collapse exactly**. The collapse guard added on 2026-07-31 did its job:
raised immediately instead of logging 39 more silent dead epochs. Fixed in `ed7a78a`.
Tasks 0-8 and 10-15 ran under the old 500-step warmup and were unaffected — the instability
is width-specific, not systemic.

---

## Step 24 — The metric that decided it

All 16 checkpoints rescored on the test split (`experiments/eval_sweep.py`, job
`15217470`, 10 min). Barrier:

| hidden | params | sp_coef | mae_aia | p99 | bimod recall | bimod prec |
|---|---|---|---|---|---|---|
| 680 | 1.43M | **1.904** | 4.898 | 17.80 | 29.46% | 80.95% |
| 480 | 722k | 1.913 | 4.893 | 17.87 | 27.74% | 79.82% |
| 336 | 360k | 1.981 | 4.858 | 17.85 | 27.14% | 80.27% |
| **232** | **176k** | 2.099 | **4.792** | 17.90 | 26.33% | 79.01% |
| 160 | 87k | 2.200 | 4.823 | 18.64 | **8.10%** | 76.79% |
| 108 | 42k | 2.280 | 4.604 | 18.75 | 8.86% | **58.45%** |
| 72 | 20k | 2.308 | 4.499 | 19.07 | 7.57% | 55.08% |
| 48 | 10k | 1.494 | 7.814 | 57.1 | 21.37% | 23.84% |

**Between h232 and h160, bimodal recall crashes 26.33% -> 8.10%** — two thirds of the real
multi-thermal pixels lost — while `mae_aia` *improves* and `sp_coef` drifts by 0.1.
**Nothing in the aggregate metrics sees it.** They are averages over ~5M mostly-easy pixels,
and capacity buys the *hard* ones. Judging on `mae_aia` and `sp_coef` alone would have
shipped h108 or h72 and silently lost the scientifically interesting pixels.

h48's 12.71% firing rate at 23.84% precision is guessing near the base rate, not detection
— the same trap as its "improved" `sp_coef`. A metric moving in the good direction because
the model has stopped fitting is not an improvement.

ENet: healthy to h160, **best of the whole sweep at h232** (`mae_dem` 0.0077, p99 41.29,
both better than the 1.43M baseline), collapsing at h72 (`mae_aia` 8.85) and h48 (22.0).

---

## Step 25 — The decision

**h232 / 176k for both losses — 8x smaller, no meaningful loss.** Smallest width safe on
both: bimodality intact, better `mae_aia` than the baseline on both losses, and enet's best
numbers of the sweep.

**The one honest cost**: barrier `sp_coef` moves 1.904 -> 2.099, further from BP's 1.79
reference. If sparsity fidelity is the paper's headline claim, **h336 (360k, 4x smaller,
`sp_coef` 1.981)** is the conservative fallback and gives up almost nothing.

---

## What is still open

- **Hot-channel undershoot.** At bright pixels the networks resynthesize 94A/131A far too
  faint (obs 1491, band [1440,1540], `Dx` 330) and their DEM peaks sit ~0.1-0.15 low in
  logT. This is the 2026-07-11 flare-fold peak shift, unchanged by scale, and it is the
  one genuine model deficiency the whole evaluation surfaced. Not addressed.
- **ENet hyperparameters unconfirmed** — generation used `alpha=1, lam=0.5, C=n_obs`
  (the `--fitlinearalpha 1 --fitlinearl1ratio 0.5` defaults). If Samuel validated
  different values, the 1,223 ENet files need regenerating (~2.5 h + 40 min restage).
- **AIA+XRT** — not in this round; scope and usable XRT coverage still unknown.
- **The deconvolution positivity clamp**, narrowly. Near-zero readings in the *bright*
  channels (171/193/211A) are unphysical and `MIN_OBS = 1e-3` is too permissive to exclude
  them. Small population, not blocking, but it is the residue of the Step 19 investigation.
- **Uncertainty head** — still unbuilt, and now less motivated: the bimodality result says
  a point estimate already locates multi-thermal pixels at 80% precision.

---

## Step 26 — The metric was measuring the label, not the model

The sweep left one number unexplained: `mae_dem` is ~3x worse where BP is bimodal
(0.29 vs 0.10), and that penalty is **constant** from 1.43M parameters down to 20k
(2.89x-3.11x). Capacity is not the bottleneck. That leaves the amortization or the
target.

BP solves each pixel independently from a *noisy* observation, so it has its own
reproducibility floor. `experiments/bp_self_consistency.py` measures it: 80 unimodal
and 80 bimodal bright test pixels, 30 BP re-solves each under photon noise, all
quantities in DEM units.

| | BP self-scatter | mean \|BP\| | noise share | \|NN−BP\| | ratio |
|---|---|---|---|---|---|
| unimodal | 0.3097 | 0.8351 | 37.1% | 0.2135 | **0.69x** |
| bimodal | 0.6027 | 1.3963 | 43.2% | 0.5895 | **0.98x** |

**43% of BP's own signal at bimodal pixels is noise-driven.** Re-solving the same
pixel under one realistic noise draw moves BP's answer by 0.60; our prediction
differs from it by 0.59. At unimodal pixels the network is *closer to BP than BP is
to itself*.

So the 3x penalty is the label's irreproducibility, not the model's error — BP's
self-scatter also roughly doubles at bimodal pixels (0.31 -> 0.60), which is the
whole effect. The gray perturbation spaghetti in `final_bimodal.png` had been showing
this all along; it now has a number attached.

**Residual genuine gap**: hot bins at bimodal pixels, **1.17x**. That is the
94A/131A undershoot, and it is the only part of the bimodal deficit that is ours.
Much smaller than the raw 3x implied.

**h232 is equally inside the envelope** (0.72x / 0.98x vs 0.69x / 0.98x), so the 8x
reduction costs nothing on this measure either.

**A hypothesis of ours, falsified.** We expected the network — which minimises the
objective, not the label — to sit closer to the *mean* of BP's noise ensemble than to
any single solve, i.e. to be estimating the expectation. It does not: `|NN−ens|`
(1.10x) is worse than `|NN−BP|` (0.98x). It tracks the clean solve better.

**Flaw to fix**: the `enet_*` rows compare ENet-trained models against *BP's* labels,
which is not their target. Those rows are not meaningful as written.

---

## Step 27 — How the bimodality result must be stated

The 80.75% precision figure is real but narrower than it sounds. Precision measures
*co-occurrence of a binary shape property* — the network's curve has two local maxima
and so does BP's. It is not curve agreement, and `final_bimodal.png` shows panels
passing that test while the network draws a broad smooth blob against BP's sharp
spikes.

Supported: the network **flags** multi-thermal pixels at 79-81% precision against a
14.17% base rate. Useful as a detector.

Not supported: the network **reproduces** bimodal DEMs. It is ~3x worse there — though
per Step 26 that gap is within BP's own reproducibility, so it is not a model failure
either.

Two further caveats: recall is only ~30%, and on the *strongest* double peaks (weaker
peak >= 25% of the taller) agreement falls to ~14% (328 of 2,372), so the cases it
catches skew toward subtle ones.

**Also corrected**: the h160 "cliff" (recall 26.33% -> 8.10%) is substantially a
detection-threshold effect. `mae_dem` at bimodal pixels rises only 0.293 -> 0.332
(13%) across that step — h160's curves are nearly as close to BP, but smoothed just
enough that the second bump stops registering as a local maximum at 15% prominence.
h232 remains the better choice (better `sp_coef`, `mae_dem` and recall), but the cliff
is less dramatic than peak-counting made it look.

---

## The result, stated plainly

**Production model: mlp6 at hidden=232, 176,374 parameters** — 8.1x smaller than the
1.43M baseline, for both the BP and ENet tracks. On 153 timestamps never used for
training or checkpoint selection it matches the baseline on every metric that
survives scrutiny, and it sits inside BP's own reproducibility envelope.
