# DEM project state

Last updated: 2026-09-21

## Current status

The requested label-free DEM study, matched supervised comparison, full-test
evaluation, and public visualizer are complete. The next step is to present the
results to Samuel Pérez-Díaz and David Fouhey and use their feedback to choose
the next deliverable (slides, report, or follow-up experiments). No additional
compute job is currently required.

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
targets per timestamp). The five targets repeat the clean AIA input, so they are
evaluation weighting rather than independent observations.

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
