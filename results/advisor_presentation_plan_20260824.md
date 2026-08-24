# Advisor presentation — neural DEM inversion

## Purpose

Present the completed label-free neural DEM inversion work to Samuel's advisor
as a research briefing, not as a paper pitch. The presentation should make three
things easy to judge:

1. What was built and why it is scientifically useful.
2. Which conclusions are supported by the evaluations.
3. Which limitations and next experiments deserve advisor direction.

## Recommended format

Eight to ten slides plus a live full-disk viewer demo. Aim for 12--15 minutes,
then discussion.

## Slide sequence

1. **Problem and motivation**
   - Six AIA channels must infer an 18-bin DEM.
   - Classical BP/ENet inversion is a per-pixel optimization.
   - Goal: amortize the physical inverse objective without solver DEM labels.

2. **Method**
   - AIA observation -> per-pixel MLP -> non-negative DEM basis coefficients.
   - Forward response reconstructs AIA; BP or ENet objective is the training
     signal.
   - Explicitly state: solver DEMs are evaluation references, not targets.

3. **Data and honest evaluation**
   - Hofmeister-deconvolved AIA-only data.
   - Timestamp-disjoint 917/153/153 train/validation/test split.
   - Why held-out pixels from seen images would not be sufficient.

4. **Architecture decision: MLP versus patch CNN**
   - The 176k h232 MLP is the production model.
   - The patch CNN did not show a consistent benefit on unseen timestamps.
   - Practical implication: no spatial context is needed for the present
     AIA-only objective.

5. **Capacity sweep**
   - h232 is 8.1x smaller than the 1.43M baseline.
   - Explain the sparsity-versus-resynthesis trade-off rather than claiming a
     universal best size.

6. **Multi-thermal result**
   - BP MLP: high precision (~80%), limited recall (~30%) for interior
     bimodality.
   - Best on well-separated, similar-height peaks; misses shoulders.
   - Do not claim that every bimodal DEM is reproduced.

7. **Reference ambiguity / self-consistency**
   - BP perturbation re-solves: at bimodal pixels, NN--BP discrepancy is 0.98x
     BP's own noise-induced scatter.
   - Core interpretation: capacity alone cannot resolve a reference that is
     unstable under photon noise.

8. **Full-disk visual demonstration**
   - Use the zoomable viewer: BP or ENet solver left, h232 MLP right.
   - Show DEM maps, AIA resynthesis, and JPDFs.
   - Explain why nearly identical AIA resynthesis does not imply identical DEM:
     six measurements underdetermine 18 bins.

9. **What the full-disk AIA metrics say**
   - For 20150923_180356, BP MLP versus BP solver:
     MAE 4.63 versus 5.29; MSE 6006 versus 110.
   - Typical-pixel AIA fidelity is similar, while rare MLP residuals dominate
     squared error.
   - State that BP solver invalid pixels must be masked in any reportable metric.

10. **Limitations and decisions requested**
    - Hot-channel (94/131 Å) undershoot at bright pixels.
    - Current ENet mismatch is broader than a few outliers.
    - Ask which next direction has the highest scientific value: improved
      objective/calibration, targeted hot-channel treatment, uncertainty output,
      AIA+XRT, or a new physics target.

## Materials to prepare

- One clean method diagram.
- Existing MLP/CNN or width-sweep figure.
- Existing multimodal-quality or BP self-consistency figure.
- A compact advisor-facing results table, with all metric units labelled.
- Live viewer staged locally from Torch; preselect one quiet and one active date.

## Deliberate exclusions

- Workshop-paper framing, submission schedule, and authorship.
- Cluster setup and implementation chronology.
- A claim that the model beats supervised work.
- Unconfirmed ENet hyperparameters or Bright/Quiet table comparisons.
