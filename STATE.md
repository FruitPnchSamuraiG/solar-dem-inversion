# DEM project state

Last updated: 2026-10-03 (Claude: fresh-pixel sweep mostly in; h232 no longer the clear choice; CNN runs + ENet h72/h48 pending)

## 2026-10-03 night: Torch cleanup (DEM only)

- `/home` was at 95% of its 30k-file limit. `logs/inv_hof/` (4,894 label-generation
  logs) is now one archive, `logs/inv_hof_20260731.tar.gz`; `/home` dropped to
  ~20.2k files. `logs/visuals/matched_cpu_17559157.err` (136 MB of ENet
  convergence warnings) is gzipped. The stray `$SCRATCH/dem/runs/dem_loo_13320865.out`
  moved to `logs/loo/`; empty `$SCRATCH/dem/{runs,checkpoints}` removed.
- Deleted `$SCRATCH/dem/data/bp_smoke_test/` (3.6 GB, July smoke tests) and
  `$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full/` (886 GB of alpha=1 raw ENet
  labels, superseded by alpha=0.001; training uses the `_DS` zarr). Kept: BP raw
  labels (the website export reads them), both `_DS` zarrs, `visuals/`, the four
  small-data images, the Hofmeister PSFs.
- Still to do once no jobs are running: move `logs/` to `$SCRATCH/dem/logs` with
  a symlink, so job logs stop counting against the `/home` file limit.
- Not DEM, untouched: `$SCRATCH/outputs.zip`, `$SCRATCH/checkpoints/`, other projects.

## 2026-10-03 evening: fresh-pixel sweep results (BP complete, ENet partial)

**Plan agreed with Hriday (2026-10-03):** wait for the CNN runs (19108189_14/15)
and the CNN-vs-MLP test comparison (19108218); report it; Hriday picks the
final model (h232, h336, or the CNN if it clearly wins, beyond the 0.2-0.4%
run-to-run spread). Then, on that model: diagnostic + AIA fit (exist for every
mlp6 width), failure figures (`plot_bright_failures.py --plain`),
self-consistency, **and rerun sqrt + Fourier-feature input with fresh pixels**
(`job_train_input_encoding.sbatch` plus `--resample_pixels` and the final width).
Then update slides 17, 18, Parts 6-7 and the summary slide in one pass, and
STATE/CLAUDE. Optionally refresh `results/bright_failure_story_20260929.md`.

`results/plots/17_sweep_resample_20261003/sweep_table.txt` (diagnostic + AIA fit
per width on the full shared test set; val loss/sparsity from training).

- **No recall cliff any more.** BP multi-peak recall is 26% at 1.43M, 25% at
  176k, 22% at 87k and 42k, 17% at 20k; h48 collapsed (val loss 61, recall 0).
  The fixed-sample sweep's 26% -> 8% cliff at h160 does not reproduce.
- **Bigger is better on the typical pixel, both tracks:** median pixel error BP
  0.023 (1.43M) / 0.026 (360k) / 0.030 (176k) / 0.043 (87k) / 0.068 (20k);
  ENet 0.81 / 1.07 / 1.70 / 1.55 / 2.24. AIA MAE moves the other way (smaller
  models fit AIA slightly better), as in the fixed-sample sweep.
- **DEM MSE is erratic across widths** because flare cores dominate it: BP
  4.28-5.45 with flare-core emission ratio jumping 0.12-0.40 non-monotonically;
  ENet h680 is 36.2 (flare cores over-predicted, 1.15x) against 0.48 at h336.
- **h336 (360k) beats h232 on nearly every test metric on both tracks** (BP DEM
  MSE 4.28 vs 5.28, median 0.026 vs 0.030; ENet 0.48 vs 0.79, 1.07 vs 1.70)
  and has the best BP validation loss (2.148 vs 2.237). Production width is
  Hriday's call; slide 18's "176k is enough" no longer holds as stated.
- Self-consistency (job 19108236): at BP-multimodal pixels the h232 gap is
  0.99x BP's own re-solve scatter (h680 0.96x); unimodal 0.77x / 0.68x.
- ENet h72 (`19108189_12`) was cancelled by HPC for low GPU use at epoch 38/40
  (tiny model, slow node, past 2 h); its epoch-38 checkpoint is being
  evaluated (`19130153`, `19130154`). ENet h48 and both CNNs still running or
  queued; eval_scaled `19108218` runs after the array.

## 2026-10-03 DECISION: fresh-pixel models are the headline

Deck status: `slides/DEM_deck_v2.pptx` now has all 8 parts (31 slides), built
from `slides/deck_outline.md`. Parts 6-7 use the fresh-pixel h232 numbers and
redrawn figures (`results/plots/16_resample_20261002/fig*.png`, via
`plot_bright_failures.py --plain --lf_dir ... --tag resample`). Slides 17
(CNN vs MLP), 18 (width sweep) and the last point of 22 (self-consistency)
still show fixed-sample numbers, labelled, until the rerun lands; their data
sit in `CNN_VS_MLP`, `SWEEP` and the slide-22 text in `build_deck_v2.py`.

Hriday's call: the method we describe samples fresh pixels every epoch, so the
deck and paper report the fresh-pixel models (`output/experiments/resample/`),
even though they score worse. The fixed-sample results are mentioned briefly
where it matters (they scored better). This supersedes "production stays"
below. Production is now `scaled_mlp6_{barrier,enet}_h232_resample.pt`.

Everything measured on fixed-sample models is being redone (submitted
2026-10-03 03:30, `bash experiments/submit_sweep_resample.sh`):

- Array `19108189` (`job_sweep_resample.sbatch`, %4): mlp6 widths
  680/480/336/160/108/72/48 x {barrier (warmup 500), enet alpha=0.001 (warmup
  3000)} plus patch cnn x both. h232 is the existing resample pair.
  h680 and cnn are saved with the plain `_resample` suffix.
- Per mlp6 task, afterok: diagnostic + AIA fit, TAG=`resample_h<W>`, jobs
  `19108190`-`19108217` (task i -> 19108190+2i diag, +1 aia). The diagnostic's
  `by_dem_shape` cross-tab gives multi-peak recall/precision on the full test set.
- `19108218` (afterany array): `eval_scaled.py --ckpt_dir output/experiments/resample
  --ckpt_suffix _resample` -> CNN vs MLP on the test split,
  `output/experiments/eval_scaled_resample/`.
- `19108236` (afterok tasks 0, 7): `bp_self_consistency.py` on the fresh-pixel
  1.43M and h232 models -> `output/experiments/final_resample/`.

## 2026-10-03 RESULT: fresh pixels made both models worse; production stays

All six jobs completed (`19071403`-`19071408`). Outputs and side-by-side
tables in `results/plots/16_resample_20261002/` (`{bp,enet}_comparison.txt`,
from `experiments/compare_input_encoding.py <dir> resample <track>`).

| Metric | BP prod | BP resample | ENet prod | ENet resample |
|---|---:|---:|---:|---:|
| Validation loss (final) | 2.156 | 2.237 | 0.0582 | 0.0689 |
| DEM MSE, all pixels | 4.19 | 5.28 | 0.588 | 0.794 |
| Median pixel error | 0.0257 | 0.0295 | 1.41 | 1.70 |
| Relative error, 1-1.8x | 0.168 | 0.221 | 0.111 | 0.220 |
| Flare-core emission / solver | 0.41 | 0.15 | 0.72 | 0.73 |
| Flare-core multi-peaked % | 83 | 89 | 81 | 40 |
| Flare-core 94 A fit | 0.24 | 0.07 | 0.26 | 0.40 |
| AIA MAE / MSE | 4.32 / 82.6 | 4.35 / 93.3 | 0.67 / 193 | 0.72 / 414 |

**Not seed noise.** `train_scaled.py` seeds neither the weights nor the block
order, but identical configs already in the logs agree to 0.2-0.4% in final
validation loss (h680 barrier 2.1446 vs 2.1395; h680 ENet alpha=1 1.8501 vs
1.8567). Fresh pixels moved it +3.8% (BP) and +18% (ENet). ENet flare cores did
improve (multi-peaked 81% -> 40%, 94 A 0.26 -> 0.40), nothing else did.

Untested hypothesis: the objective is heavy-tailed (mean barrier loss is ~94%
one pixel). A fixed sample lets the model absorb its pathological pixels once;
fresh pixels bring new ones every epoch, and their clipped gradients dominate
the steps. **Decision: keep the production (fixed-sample) models; do not rerun
the width sweep.** The data slide should say each block contributes one fixed
random sample of 512 pixels.

## 2026-10-03: the June "noisy" per-pixel MLP did not reproduce

Job `19077429` retrained it with the unchanged June code
(`train_unsupervised.py`, 4 images, crop 1800,1800,256,256, hidden 256, raw
inputs, barrier loss, 30 epochs; final loss 2.03) and compared it with the
ablation patch CNN and BP on 400 held-out pixels per image
(`experiments/plot_first_mlp_vs_patch_cnn.py`, results in
`results/plots/15_first_mlp_repro_20261002/`). Its curves are single-peaked:
mean 1.00-1.03 local maxima, 0% with 3+, same as the CNN (BP 1.12-1.19). It is
less accurate than the CNN (sharper peak, hot side cut off near logT 6.5), but
not oscillating. The June note's "visibly noisy" claim has no surviving figure
and does not reproduce; slide 10's framing was revised.

Accuracy against BP on the same 400 held-out pixels per image (job `19095906`,
mean over the 4 images): DEM MAE first MLP 0.078, ablation centre-pixel MLP
(1.43M) 0.042, patch CNN 0.047; W1 0.055 / 0.043 / 0.048 dex. The centre-pixel
MLP is closest on 3 of 4 images (the CNN on the X1.6 flare), so the first MLP's
gap was capacity (or input scaling), not missing neighbours, and the CNN's
lower sparsity did not mean DEMs closer to BP. Slides 10-11 now say this.

## 2026-10-02: every scaled run trained on one fixed 30M-pixel sample

`ZarrPatchBlockDataset.__getitem__` seeded each block's 512-pixel draw by the
block index alone (`src/zarr_data.py`), so every epoch of every scaled run saw
the same pixels: 58,688 blocks x 512 = 30M pixels, about 0.8% of the ~3.8B
training pixels, 40 times over. Results remain valid (test matched validation
to the third decimal), but rare pixels are under-sampled: flare cores (~0.09%
of pixels) contribute a fixed ~27k. The capacity-ceiling result was measured
under the same limit (h960 validation loss was already rising slightly).

Fix (`7067355`): `--resample_pixels` uses `EpochResampler`, which offsets epoch
e's indices by e x n_blocks; the dataset decodes the draw from the index, so it
reaches persistent workers. Draw 0 keeps the old seed, so epoch 1 and every
default loader are unchanged (tested in `tests/test_zarr_data.py`).

**LAUNCHED** (`bash experiments/submit_resample.sh`): production h232 retrained
with fresh pixels and otherwise identical settings (BP warmup 500 as in sweep
task 15185224_3; ENet alpha=0.001 warmup 3000 as in 17385814).

| Track | Train | Diagnostic | AIA fit |
|---|---|---|---|
| BP | `19071403` | `19071404` | `19071405` |
| ENet | `19071406` | `19071407` | `19071408` |

Checkpoints: `output/experiments/resample/scaled_mlp6_{barrier,enet}_h232_resample.pt`.
Evaluations (afterok) write `output/experiments/diagnostics/{bp,enet}_{bright_failure,aia_fit}_resample.*`;
together they reproduce the full results table (DEM MSE, EM error, W1, AIA
MAE/MSE) and the flare-core analysis. If the gain is large, rerun the width
sweep with `--resample_pixels` (decision deferred to the results).

## 2026-10-02 meeting outcome

- **Priority: consolidate and present everything done so far to the advisor**
  (slide deck, built section by section). Experiments run in parallel only.
- **Suggestion 1 (pushed as first): lift the inputs to a higher-dimensional
  space to fix the flare-core tail.** Square-root the AIA intensities and apply
  Fourier-feature encoding, as the supervised model does:
  `sqrt(clamp(aia))` then `src/model.py:posEncode` with 12 frequencies
  (1.6 to 71 rad per sqrt-DN), plus the raw sqrt channel appended.
- **Suggestion 2 (lower priority): predict the 18 DEM bins directly** instead of
  54 basis coefficients, as the supervised model does. Less targeted: the
  solver uses the same basis and reproduces flare-core AIA within 8%, so the
  basis can represent the right answer.
- **Question raised: do we square-root the input? Checked: no.** Label-free uses
  `log1p` (`NormalizedInput`, `experiments/train_scaled.py`) feeding a plain
  6-input MLP, with no encoding. log1p compresses the bright end far more:
  a flare core (~1e4 DN) against a bright active region (~500 DN) is 9.2 vs 6.2
  under log1p but 100 vs 22 under sqrt.
- **Experiment LAUNCHED 2026-10-02 (one run, both changes together, as the
  supervised model does):** training job `19049208`
  (`experiments/job_train_input_encoding.sbatch`), h232 BP barrier with
  `--input_transform sqrt --fourier_freqs 12`, everything else unchanged
  (40 epochs, batch 16, warmup 3000). Checkpoint:
  `output/experiments/input_encoding/scaled_mlp6_barrier_h232_sqrt_ff12.pt`.
  Evaluations start automatically after it (afterany): `19049209` bright
  diagnostic and `19049210` AIA fit, both TAG=sqrtff12, writing
  `output/experiments/diagnostics/bp_{bright_failure,aia_fit}_sqrtff12.*`.
  Code: `FourierFeatures` in `experiments/train_ablations.py` (verified equal to
  `posEncode`), `sqrt` mode in `NormalizedInput`; defaults unchanged. Score with
  `diagnose_bright_failures.py` and `aia_fit_by_brightness.py`. Success: flare-core
  94 A fit up from 0.24 toward the solver's 0.92, flare-core relative DEM error
  down from 0.80, no loss on the faint half. Watch: the appended raw sqrt channel
  reaches ~100 at flare cores; keep warmup and the clamped softplus (July
  collapse came from unscaled inputs).

## 2026-10-02 result: sqrt + Fourier-feature input did NOT fix flare cores

Run `19049208` (h232 BP, `--input_transform sqrt --fourier_freqs 12`, 40 epochs,
completed in 1:48) scored by `19049209`/`19049210`; results and
`comparison.txt` in `results/plots/14_input_encoding_20261002/`.

| BP track | production (log1p) | sqrt + FF12 | supervised |
|---|---:|---:|---:|
| DEM MSE, all pixels | 4.19 | 5.07 | 0.906 |
| Median pixel error | 0.0257 | 0.0257 | 0.112 |
| Relative error, 3.2-10x | 0.168 | 0.134 | 0.028 |
| Relative error, 32x+ | 0.80 | 0.98 | 0.17 |
| Flare-core emission / solver | 0.41 | 0.19 | 0.92 |
| Flare-core 94 A fit | 0.24 | 0.09 | 0.85 |
| Flare-core 131 A fit | 0.76 | 0.20 | 0.90 |
| Flare-core multi-peaked % | 83 | 5 | 13 |
| AIA MAE / MSE | 4.32 / 82.6 | 4.40 / 94.9 | 2.95 / 156.7 |

- Flare cores got **worse**: less emission, hot channels fit even more poorly;
  cool channels (171/193/211 A) still fit to ~2%. Fewer spurious humps (5%),
  but the curve is now flat and low.
- Small gains elsewhere: 3.2-10x relative error 0.168 -> 0.134; validation
  sparsity 1.90 (BP 1.79) vs production 2.08. Validation loss slightly worse
  (2.223 vs 2.156). Medians unchanged.
- Reading: the input representation is not what limits flare cores. Untested
  hypothesis for the drop: Fourier features are periodic, so the rare, very large
  sqrt values of flare cores alias onto encodings of ordinary pixels, making
  extrapolation harder. Remaining candidates are training-side: rarity of flare
  cores, gradient clipping, loss weighting. Production model stays the log1p h232.

## Current status

The requested label-free DEM study, matched supervised comparison, full-test
evaluation, and public visualizer are complete. Before presentation, the one
active follow-up is a targeted **Bright-failure diagnostic**. It will identify
which AIA-channel threshold combinations define the difficult pixels, whether
their solver-reference DEMs are multi-peaked, where their error sits in logT,
and how much of the total DEM SSE each group carries. This is explanatory
analysis, not another model sweep.

Public viewer:
https://triborough.cs.nyu.edu/hsr3649/demdemo/webapp/compare.html

The live deployment was verified to contain the final full-test Results page on
2026-09-19.

## Final comparison

The production label-free model is the 176k-parameter, center-pixel MLP6. It
maps six AIA intensities to 54 nonnegative basis coefficients, then uses fixed
basis and AIA-response operators to produce an 18-bin DEM and reconstruct the
six observations. It is trained without BP/ENet DEM labels. Comparisons use the
supplied supervised checkpoints and the corresponding classical solver output.

The ENet label-free model was retrained with the matched objective
(`alpha=0.001`, `l1_ratio=0.5`). Evaluation uses the complete shared test set:
153 held-out timestamps and 48,960 blocks (64 spatial blocks and five solver
targets per timestamp). Each target contributes its own random 64 of 256
spatial blocks of the same clean AIA image (see the layout correction below).

Full-population results (lower is better):

| Track | Model | DEM MSE | EM error (%) | W1 (dex) | AIA MAE | AIA MSE |
|---|---|---:|---:|---:|---:|---:|
| BP | Supervised | 0.9060 | 14.34 | 0.1334 | 2.9461 | 156.6699 |
| BP | Label-free MLP6 | 4.1910 | 20.16 | 0.0845 | 4.3153 | 82.5570 |
| ENet | Supervised | 0.3181 | 18.00 | 0.1210 | 6.3214 | 597.8458 |
| ENet | Label-free MLP6 | 0.5875 | 13.45 | 0.1176 | 0.6692 | 193.3688 |

The website additionally reports Bright and Quiet subsets. DEM squared error is
heavy-tailed: the worst 1% contributes about 97.80% of label-free BP SSE and
66.75% of label-free ENet SSE. Median per-pixel DEM SSE (sum over 18 bins) is
0.02618 vs 0.11022 for label-free vs supervised BP, and 1.39589 vs 1.12525 for
label-free vs supervised ENet. These tail values are retained, not discarded.

For a meeting explanation: label-free ENet has better AIA MAE and MSE than the
supervised ENet model on the full test set; label-free BP has better AIA MSE
but worse AIA MAE than supervised BP. Label-free training fits the AIA
observations through a fixed response operator and regularized DEM, whereas
supervised training targets solver-produced DEM labels. This explains why the
two approaches need not rank the same on AIA reconstruction and DEM-label
agreement, but does not guarantee label-free AIA superiority on every metric.
The bright population is about 10.1% of BP pixels and 10.5% of ENet pixels,
yet accounts for about 98.8% and 75.7%, respectively, of the label-free DEM
SSE. The bright medians are also worse than the supervised medians, so the gap
is not solely a few extreme pixels. Do not claim systematic bright-region
underprediction without examining signed residuals.

## Bright-failure diagnostic RESULTS (2026-09-28)

Torch jobs `18733242` (BP, 14:28) and `18733245` (ENet, 17:42), CPU, all
48,960 blocks, both `COMPLETED 0:0`; smoke `18733239`. Outputs copied to
`results/plots/13_bright_diagnostic_20260928/{bp,enet}_bright_failure.{json,txt}`.
Full MSE (4.19 / 0.588), Bright share and worst-1% share reproduce the website.

1. **Squared error is dominated by flare cores** (see the supervised section:
   true of both models). Pixels at >=10x the Bright threshold
   (max over channels of observed/threshold) are 0.09% (BP) / 0.08% (ENet) of
   pixels but carry **94.0% / 59.6%** of all DEM squared error.
2. **Ordinary Bright pixels are fit relatively better than Quiet ones.** At
   1-10x threshold, error relative to reference energy is 0.13-0.17 (BP) and
   0.04-0.11 (ENet) versus 0.25 / 0.23 for Quiet.
3. **At the extreme the model under-predicts emission: range compression.**
   Predicted/reference emission and peak height stay about 0.75-0.9 up to a
   reference peak of ~100, then fall: at >=32x threshold BP emission 0.41, peak
   0.24; ENet 0.72, 0.62. BP's extreme error is 97% *along* the reference curve
   (a scale shortfall, not wrong temperatures). Mean logT shift is -0.17 dex
   (cooler), and the signed error is most negative at logT 6.5-6.9.
4. **Associated channel: 94 A.** Where 94 A is the most over-threshold channel
   (1.96% of pixels) BP carries 86.1% of SSE with emission ratio 0.69; 94 A
   leads 62% / 66% of the worst 0.1% pixels. Pixels triggering all six channels
   carry 93.9% / 59.7%. Which channels trigger is not discriminative; intensity is.
5. **Spurious peaks, not missed peaks.** Reference unimodal but model multi-
   peaked, at Bright pixels: 0.24% of pixels, 22.1% of BP SSE (ENet 0.50%,
   33.8%). Missing a BP second peak costs little: 1.52% of pixels, 1.1% of SSE.
6. **Not solver noise.** Solver scatter across repeated targets is 0.4% (BP) /
   5.7% (ENet) of the Bright model error; it is 24% / 37% of the Quiet error.
7. **General offset**: label-free BP emits ~15% less total EM than BP
   everywhere (ratio 0.845, Quiet 0.841); ENet 0.98. Unexplained; plausibly the
   L1 term settling lower inside the tolerance band. Hypothesis only.

Caveats: associations, not causes. Solver scatter comes from blocks where 2-5
targets overlap (64% of blocks), so it is biased low for pairs; doubling it does
not change point 6.

### Supervised through the same diagnostic (jobs 18734665 BP, 18734666 ENet)

Same pixels, mask and validity rule; totals reproduce the website (BP 0.906,
ENet 0.318). Side-by-side tables: `results/plots/13_bright_diagnostic_20260928/{bp,enet}_compare.txt`
(regenerate with `diagnose_bright_failures.py --compare LF.json SUP.json`).
This revises point 1: flare-core dominance of SSE is a property of the metric,
not of label-free training.

- **Both models put ~90% of BP SSE in the >=32x bin (0.01% of pixels)**: LF
  89.4%, supervised 90.0%. The headline BP MSE ratio 4.63x is essentially the
  ratio on that bin (4.59x). ENet: LF 51.1%, supervised 33.9%.
- **The label-free gap grows with brightness.** LF/supervised MSE by intensity,
  BP: 0.65 (<0.03x), 0.79, 0.95 (0.1-0.3x), 1.86 (0.3-1x), 2.26, 2.89, 6.0
  (3.2-10x), 18.4 (10-32x), 4.59 (>=32x). LF is *better* on the faint 47% of
  pixels. ENet: 1.00-1.27 at every level below 10x, 4.17 and 2.78 above.
- **Typical-pixel error favours LF on BP**: median per-pixel SSE Quiet 0.016 vs
  0.085; Bright 2.82 vs 1.78.
- **LF-specific flare-core failures**, absent from supervised: emission ratio
  at >=32x BP 0.41 vs 0.92 (ENet 0.72 vs 0.87); peak ratio 0.24 vs 0.91 (ENet
  0.62 vs 0.92); spurious multi-peaks, model multimodal 83% vs 13% (ENet 81% vs
  18%). Both models draw extra peaks on some Bright pixels where the solver has
  one (LF 0.24%, supervised 0.41% of pixels on BP), but they cost LF 22% of its
  error and supervised 0.9%, because LF's sit on flare-core DEMs. (The shape
  groups are each model's own, so per-group MSE ratios across models are not
  like-for-like.)
- Faint-DEM over-prediction: on BP only supervised over-predicts where the
  solver's peak is tiny (27x vs LF 2.1x at 0.001-0.0032); on ENet both do (96x
  vs 52x), suggesting the ENet solver drives faint DEMs toward zero.
- Readable write-up with figures: `results/bright_failure_story_20260929.md`.

### AIA fit at flare cores (jobs 18783954 BP, 18783957 ENet, 2026-09-29)

`experiments/aia_fit_by_brightness.py` projects label-free, supervised and
solver DEMs through the AIA response by brightness band; pooled MAE/MSE
reproduce the website exactly. At flare cores (>=32x) on BP the solver
reproduces all six channels within 8% (94 A 0.92), while label-free fits
171/193/211 A to 2% but reproduces only **0.24 of 94 A**, 0.76 of 131 A and
0.66 of 335 A (98% of pixels short by >10% in 94 A). **Label-free fails its own
objective there: a training problem, not an objective that permits the smaller
DEM.** The 94 A fit breaks above ~10x (0.81, 0.73, 0.38, 0.24 across the bright
bands). ENet is murkier: the ENet solver itself reaches only 0.55 of 94 A at
flare cores; label-free 0.26. Random example curves (seeded reservoir sample,
`*_aia_fit_examples.npz`) show label-free missing the hot peak near logT
6.8-6.9 that solver and supervised both have. Faint pixels: BP solver sits at
0.3-0.7 of observed 94/131 A (noise-limited) and label-free slightly lower,
consistent with the 15% emission deficit. Next (not started, needs a decision):
measure training loss / clipping on flare-core pixels, then oversample or
up-weight them.
- **94 A-dominant pixels** are where LF trails most on both tracks (5.3x BP,
  5.2x ENet). On ENet, LF *beats* supervised where 131/193/211/335 A dominate
  (0.73-0.93x) and at reference-multimodal / model-unimodal Bright pixels (0.31x).
- **Neither model is at the solver-noise floor on Bright pixels**: BP bias MSE
  57.5 (LF) and 14.7 (supervised) vs scatter 0.22. On Quiet, scatter is 24% of
  LF and 42% of supervised BP error.
- LF emits ~15% less EM than BP everywhere; supervised is ~1.0 on BP. On ENet
  the pattern reverses: LF 0.98, supervised 0.89.

**Test-set layout correction**: the five targets per timestamp are not five
copies of the same 64 blocks. Each target file contributes its own random 64 of
the 256 spatial blocks (smoke test: 2 timestamps gave 212/140/40/7 groups of
1/2/3/4 identical blocks, matching Binomial(5, 1/4)). Identical AIA blocks do
recur across targets, but a given block is seen by 1-5 solver targets.

## Active diagnostic: why Bright pixels are difficult

Samuel's Bright/Quiet mask is based on the six observed, preprocessed AIA
intensities, not on DEM bins: a valid pixel is Bright when *any* channel is at
or above its supplied top-5% threshold. Thus a pixel may be triggered by more
than one channel. The existing tables store only the union, so they cannot say
which channel(s) triggered an individual Bright pixel or whether that is related
to a high DEM error.

`experiments/diagnose_bright_failures.py` (rewritten 2026-09-28, CPU job
`experiments/job_diagnose_bright_failures.sbatch`) makes one pass over every
valid pixel of the shared test set, separately for BP and matched ENet. It
uses the same validity rule and Bright mask as `eval_full_paper_test.py`; the
synthetic test `tests/test_diagnose_bright_failures.py` reconciles its
Full/Bright/Quiet counts and MSE with that script exactly. It reports:

1. DEM error by exact trigger pattern, inclusive channel, number of channels,
   dominant channel (largest observed/threshold) and intensity over threshold;
2. relative SSE (group SSE / reference energy) to separate "Bright pixels are
   worse" from "Bright pixels simply have larger DEMs";
3. an exact per-pixel split of SSE into the part along the reference curve
   (scale) and the part orthogonal to it (temperature placement / spreading),
   plus emission ratio, peak-height ratio, EM-weighted width and logT shift;
4. a regression-to-the-mean check: predicted vs reference peak, emission and
   width in bins of reference peak height;
5. reference x model multimodality (interior peaks, 15% prominence,
   zero-padded; vectorised, verified equal to `count_peaks`) with error by
   shape, and composition of the worst 0.1% / 1% SSE tail;
6. solver scatter: blocks with bit-identical AIA (the five targets) are grouped,
   and model MSE splits exactly into model bias (distance to the mean solver
   DEM) + solver scatter (variance across targets) — the floor no deterministic
   predictor can beat.

Run on Torch (smoke first; the full jobs start only if it succeeds):

```bash
cd ~/projects/dem && git pull --ff-only && source env.sh && mkdir -p logs/eval
job=experiments/job_diagnose_bright_failures.sbatch
smoke=$(sbatch --parsable --export=ALL,TRACK=bp,MAX_BLOCKS=640,TAG=smoke "$job")
sbatch --dependency=afterok:$smoke --export=ALL,TRACK=bp "$job"
sbatch --dependency=afterok:$smoke --export=ALL,TRACK=enet "$job"
```

Outputs: `output/experiments/diagnostics/{bp,enet}_bright_failure.{json,txt}`.
The `.txt` is the readable summary. The "groups by number of targets" line
states whether repeated inputs were found; if they were not, the scatter
section is empty and everything else still holds.

This distinguishes evidence from interpretation. It can show an *association*
(for example, multi-channel bright pixels having a larger error), but it cannot
prove that a particular AIA channel physically caused the inversion ambiguity.
Nor should "multimodal" be equated automatically with true multi-temperature
plasma: solver DEM peaks may reflect non-identifiability/noise.

## Validated artifacts

- DEM results: `output/experiments/paper_eval/{bp_h232,enet_h232}_aggregate.json`
  on Torch, plus the paired supervised aggregate JSON files.
- Full-test AIA results:
  `output/experiments/paper_eval/{bp,enet}_aia_shared_test.json` on Torch.
- Full-test AIA jobs: BP `18025355` (1:12:18) and ENet `18025356`
  (1:06:29), both `COMPLETED 0:0`, all 48,960 blocks, and zero nonfinite-model
  exclusions.
- AIA evaluation protocol and final summary: `experiments/aia_shared_test.md`.
- Viewer code: `student_package/visuals_pipeline/webapp/compare.html` and
  `compare.js`; full-test results commit `ddbd5fc`. A subsequent local update
  adds median-pixel DEM MSE columns and paired bars; deploy it before relying
  on those displays during a meeting.
- Matched viewer assets on Torch:
  `/scratch/hsr3649/dem/visuals/matched_enet_alpha0p001/preview`.

## Interpretation and caveats

- The BP/ENet solutions are solver references, not directly measured true DEMs.
- DEM MSE, EM error, W1, and AIA reconstruction measure different properties;
  no single metric is the complete ranking.
- AIA MAE/MSE use a common finite-pixel mask within each track. AIA percentiles
  were not computed, so MAE and the Bright/Quiet split accompany tail-sensitive
  MSE.
- Samuel's older `lp` viewer and this viewer use the same `turbo`, square-root,
  fixed-range DEM rendering. Their colours differ because his extended 26-bin
  LP/XRT pipeline and this 18-bin Hofmeister-deconvolved AIA-only pipeline
  produce different DEM values and grids.

## Next meeting

Present the problem, label-free physical objective, MLP6 architecture, matched
BP/ENet protocol, the full result tables, tail analysis, and the three-column
interactive examples. Ask Samuel and David which claim should lead and whether
the next artifact should be a slide deck or a written research report before
starting more experiments.
