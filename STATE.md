# DEM project state

Last updated: 2026-09-28 (Claude: Bright diagnostic run; results below)

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

1. **The Bright gap is a flare-core gap.** Pixels at >=10x the Bright threshold
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
not change point 6. The supervised model was *not* run through this diagnostic,
so whether it also fails on flare cores is unknown. That is the one comparison
needed before claiming "label-free matches supervised outside flare cores".

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
