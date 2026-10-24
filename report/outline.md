# Report outline: label-free DEM inversion

Working title: **Label-free DEM inversion: training a per-pixel network on the solver's objective**

Status: outline, 2026-10-24. The report is paper-shaped (abstract to conclusions, with an
appendix), written as markdown sections under `report/sections/`, figures under
`report/figures/` (copied from `results/plots/`), and rendered by `report/build_report.py`
to one HTML page and a docx. House rules: no personal names (say "the supervised side",
"the supervised model", "the advisor"), no em dashes, every number traceable to a file
named in the section's source list, percentiles rather than means wherever the error is
heavy-tailed, and the honest notes kept (test-informed ElasticNet width and epochs, the
fixed-sample bug, inference speed unmeasured).

Workflow: Claude does the analysis and fixes the numbers; writer agents draft the sections
from this outline and the sources; Claude reviews every section against the sources.

## Claims the report makes

1. A 409k-parameter per-pixel MLP trained only on the solver's objective (no solver DEM in
   training) matches or beats a supervised emulator trained on the solver's DEMs: ahead on
   every ElasticNet measure, on four of six BP measures, and far better on the typical pixel
   (median pixel error 0.036 vs 0.112 on BP).
2. The one BP failure, flare cores, was the loss and not training: the quadratic exterior
   penalty with μ = 1 prices hot plasma out of the DEM. A stiffer band fixes it, and μ is
   not a free hyperparameter: penalty-method theory gives the band violation per pixel as
   λ_c tσ_c / (2μ), so μ follows from the response matrix, the noise and a tolerance.
3. Bigger is not better: the patch CNN loses to the per-pixel MLP at scale, and models above
   409k blow up on rare pixels that are out of the training distribution (flare cores and
   the deconvolution-clamp pixels).
4. The remaining error is concentrated: flare cores (1 in 10,000 pixels) hold 94% of the BP
   error for both models, and the low-end JPDF pixels are a data artefact (the clamp) that
   the supervised model learned to copy.

## Sections

### 0. Abstract
Claims 1 to 4 with the headline numbers. Sources: `STATE.md` (2026-10-14 block),
`slides/deck_outline.md` slide 2, `slides/talk_prep.md` (the 30-second version).

### 1. Introduction
DEM inversion, why it matters, the two solvers (BP, ElasticNet), the cost of running them at
full resolution, the supervised emulator approach and its dependence on solver labels, the
label-free alternative (amortised optimisation of the solver's objective). Contributions.
Sources: `CLAUDE.md` overview, `slides/deck_outline.md` slides 3 to 5.

### 2. Problem setting
The DEM (18 bins, logT 5.5 to 7.2), the AIA response R, the basis B (spike and two
Gaussians per bin, 54 weights; DEM = Bw, AIA = RBw), BP's LP with the noise band ±tσ
(t = 1.4, relaxed when infeasible), ElasticNet's objective with the supervised side's
settings (α 0.001, λ 0.5, C = 6), the noise model σ (aiapy estimate_error). Sources:
`CLAUDE.md` (Reference Solvers), `eval_and_enet_specs/README.md`, `fullBP.py:91`
(getBasis), `results/followup_recap_20261013.md` section 6 (the response table).

### 3. Method: amortised optimisation
3.1 Network: 6 intensities, square root then 12 Fourier frequencies (150 inputs), four
hidden layers of 336 with SiLU, 54 non-negative weights (softplus), 409k parameters, one
forward pass per pixel. Why the 54 weights and not the 18 bins.
3.2 Losses: BP as the quadratic exterior penalty (sparsity + μ × band penalty, the hinge
squared, per-channel normalisation by tσ); ElasticNet as the solver's objective exactly.
3.3 The band weight μ: the penalty-method result, the dual prices per channel from R
(94 Å 1.48, 335 Å 0.60, 131 Å 0.10, 211 Å 0.026, 193 Å 0.007, 171 Å 0.003 weight units
per DN/s), the violation formula, the μ needed for a tolerance, and the validation check
(fork A: `results/plots/23_meeting_followup_20261024/bp_mu_theory_val.*`). The per-pixel
weight as the principled generalisation; the exact L1 penalty as the alternative.
3.4 Training: 917 days, 64 random blocks per image, 512 fresh pixels per block per epoch
(30M per epoch), AdamW with cosine schedule and warm-up, gradient clipping, 40 epochs
(BP final: 120 + 120 warm-started), the clamp floor MIN_OBS and what it excludes.
3.5 Model selection on validation without labels: the training loss on the validation days;
the honest exceptions (ElasticNet width and epochs chosen with test evidence).
Sources: `experiments/train_scaled.py`, `experiments/train_ablations.py:72`,
`src/losses.py:169`, `STATE.md`, `slides/deck_outline.md` slides 4 to 7, 10,
`slides/talk_prep.md` (ElasticNet width and epochs answers).

### 4. Evaluation protocol
The shared test set (153 days, five solver targets per day, 802M pixel DEMs), the
Bright/Quiet thresholds, the metrics (DEM MAE/MSE, EM error, W1, AIA MAE/MSE, per-pixel
percentiles), why percentiles. Sources: `eval_and_enet_specs/README.md`,
`student_package/table1_dem_metrics/compute_paper_table_metrics.py`,
`slides/deck_outline.md` slide 7 and the scoring diagram slide.

### 5. Results
5.1 Headline table, both tracks, label-free vs supervised (BP: DEM MAE 0.074 / 0.100,
MSE 1.75 / 0.91, EM 11.0% / 14.3%, W1 0.068 / 0.133, AIA MAE 3.48 / 2.95, MSE 103 / 157;
ElasticNet: 0.151 / 0.188, 0.310 / 0.318, 12.2% / 18.0%, 0.103 / 0.121, 0.56 / 6.32,
161 / 598). 5.2 The typical pixel and the mean disagree: percentiles (median 0.036 vs 0.112,
p90 1.9 vs 1.9, p99 25 vs 27), error by brightness band, the share of error in flare cores.
5.3 AIA reconstruction and the JPDFs. 5.4 Architecture and size: CNN vs MLP at scale, the
width sweep, the 10M models. Sources: `STATE.md`, `results/plots/19_final_models_20261004/`,
`results/plots/22_adopted_bp_20261014/`, `results/plots/18_cnn_vs_mlp_resample_20261003/`,
`results/plots/17_sweep_resample_20261003/`, `slides/deck_outline.md` slides 8 to 14.

### 6. Flare cores: the failure and the fix
The failure (old model: 40% of the solver's emission, 23% of 94 Å), the suggestions that did
not help, the objective-gap diagnosis (the best answer under the loss drops the hot plasma;
the solver's DEM rated 2.4× worse), why (the price of hot plasma; the real price from the LP
duals, fork A), the μ screen, the stiff-band runs, flare cores now (93% emission, 94 Å 77%
vs the solver's 92%). Sources: `results/followup_recap_20261013.md`,
`results/plots/21_followup_20261006/`, `slides/deck_outline.md` slides 15 to 21.

### 7. Failure analysis
7.1 Which pixels blow up and why (fork C: `bp_blowup_census.*`, `enet_blowup_census.*`):
overlap across models, brightness, clamp incidence, distance to the training data,
extrapolation, the role of the input encoding.
7.2 The low-end JPDF pixels and the deconvolution clamp (fork B: `*_jpdf_pixels.*`): what
they are, what the solver and the supervised model do there, the headline metrics with and
without them, the cosmic-ray check (the high-end cluster and spatial isolation).
7.3 The solver's relaxed pixels (fork B mode 2: `*_jpdf_pixels_tol.*`): share, error, band
membership for the three sources.
Sources: fork outputs under `results/plots/23_meeting_followup_20261024/`, plus
`CLAUDE.md` 2026-07-31 (the clamp), 2026-08-02 (the tail), 2026-10-13 (the blow-ups).

### 8. What the network learned
The final layer (fork D: `final_layer.*`): effective rank, the output ordering by
eigenvector, shape and temperature usage against the solver's own decomposition, the DEM
manifold dimensionality. Multi-peaked DEMs: flags 15.5% of pixels at 53% precision and 60%
recall (supervised 41.1% / 24% / 72%; BP itself 13.5%); at flare cores extra peaks 25% vs
the solver's 18%. Sources: fork D outputs, `slides/deck_outline.md` slide 35,
`results/plots/22_adopted_bp_20261014/README.md`.

### 9. Limitations and open items
Inference speed unmeasured; the uncertainty head and XRT not addressed; μ per pixel and the
exact penalty untested; validation on every pixel (the ElasticNet width lesson); a third
BP continuation untested; the clamp pixels excluded from training (and the fix: train with
the clamped channel masked). Sources: `slides/talk_prep.md` (open items),
`slides/deck_outline.md` slides 22 to 26.

### 10. Conclusions
Claims 1 to 4 restated with what changed since the last meeting.

### Appendix
A. History: the first per-pixel MLP, the patch CNN, the ablation, leave-one-day-out, the
collapse and its fix (`CLAUDE.md` 2026-06 to 2026-07-31). B. ElasticNet detail and the
epochs trade (`results/followup_recap_20261013.md` section 10). C. The fixed-sample bug
(`CLAUDE.md` 2026-10-02). D. Reproducibility: checkpoints, data roots, job scripts
(`STATE.md`, `experiments/`). E. The full response table and the worked flare-core example
with the real prices (fork A).

## Figures and tables (planned)

- Fig 1 pipeline (from the deck's slide 4 diagram, redrawn).
- Fig 2 the band as a rule vs a price (deck slide 6).
- Fig 3 violation vs μ with the 1/μ law (fork A).
- Table 1 headline metrics; Table 2 percentiles; Fig 4 error by brightness band.
- Fig 5 CNN vs MLP and width sweep.
- Fig 6 flare-core curves before and after the fix (`results/plots/22_adopted_bp_20261014/`).
- Table 3 the μ screen (deck slide 19) with the theory column.
- Fig 7 blow-up pixels in input space (fork C). Fig 8 the JPDF cluster explained (fork B).
- Fig 9 the final layer sorted by eigenvector (fork D).
