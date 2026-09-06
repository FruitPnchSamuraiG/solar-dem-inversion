# Advisor presentation — neural DEM inversion

## Purpose

Present the complete label-free neural DEM inversion research story to Samuel's
advisor, not as a paper pitch. It should follow the work chronologically from
the first direct-objective experiments to the current full-disk viewer, making
clear what each experiment changed in the next decision. The presentation should
make four things easy to judge:

1. What was built and why it is scientifically useful.
2. Which conclusions are supported by the evaluations.
3. Which limitations and next experiments deserve advisor direction.
4. How the project reached its current design, including negative results,
   corrected interpretations, and engineering failures that materially changed
   the science.

## Recommended format

Eighteen to twenty-two slides plus a live full-disk viewer demo. Aim for
25--30 minutes, then discussion. The results log and CLAUDE.md are the source
of truth for the chronology; every major experiment receives at least one slide,
but low-level commands and repeated job failures are condensed.

## Slide sequence

1. **Title, people, and presentation map**
   - What will be covered: method evolution, evidence, current state, decisions.

2. **Problem and motivation**
   - Six AIA channels must infer an 18-bin DEM.
   - Classical BP/ENet inversion is a per-pixel optimization.
   - Goal: amortize the physical inverse objective without solver DEM labels.

3. **Reference objectives and starting point**
   - BP sparsity and ENet smoothness select different feasible DEMs.
   - Initial direct per-pixel loss comparison: barrier-style objectives best
     tracked BP structure; purely fit-oriented objectives did not.

4. **First label-free neural attempts**
   - Channel-input amortized networks trained directly on the objective.
   - They converged but produced noisy/oscillatory DEMs.
   - Decision: test whether patch context and capacity repair this.

5. **Neural-field progression on four timestamps**
   - Per-image patch model -> amortized patch model -> neighborhood masking.
   - What held-out pixels within seen images established, and why it was not yet
     a true generalization test.

6. **Initial architecture ablation**
   - Patch CNN, centre-pixel MLP, flat patch MLP, shuffled CNN, and ENet loss.
   - What appeared to favor spatial context before timestamp-held-out testing.

7. **Leave-one-timestamp-out correction**
   - The CNN advantage weakened/disappeared on unseen images.
   - Flare timestamp as the early generalization weakness.

8. **Early bimodality diagnostic and revised hypothesis**
   - Initial perturbation study and the question it raised.
   - Why larger-scale evaluation was needed before making a final claim.

9. **Scaling to the full timestamp dataset**
   - Hofmeister deconvolution, clean data generation, and timestamp split.
   - 917/153/153 train/validation/test setup.

10. **Numerical-collapse failure and repair**
    - Raw-DN dynamic range, dead softplus outputs, and why clean-looking logs
      were misleading.
    - Input log transform, warmup, and clamped output as the remedy.

11. **Scaled MLP versus CNN result**
    - Re-evaluation on timestamp-disjoint validation/test data.
    - Simpler per-pixel MLP selected as production architecture.

12. **Why percentiles replaced mean training loss**
    - One BP barrier-loss pixel dominated a mean and reversed a model ranking.
    - Evaluation convention: percentiles and task-relevant diagnostics.

13. **Production-model width sweep**
    - 10k to 11.2M parameters.
    - h232, 176k parameters, selected for resynthesis-oriented production use.
    - Sparsity-versus-resynthesis trade-off with larger BP models.

14. **Final multimodal census**
    - Prevalence, precision, recall, and peak-quality dependence.
    - High-precision detection is not the same as reproducing all bimodal curves.

15. **Capacity ceiling and BP self-consistency**
    - More capacity does not repair the multimodal gap.
    - Photon-noise re-solves: NN--BP discrepancy compared with BP's own scatter.

16. **Current full shared-test evaluation**
    - Full 153-timestamp data, mean metrics, and tail-aware interpretation.
    - Clearly distinguish verified results from protocol details still awaiting
      Samuel's confirmation.

17. **Full-disk visualizer demonstration**
    - Zoomable BP/ENet solver versus h232 MLP.
    - DEM maps, AIA resynthesis, JPDFs, and why equal AIA images do not imply
      equal DEMs.

18. **AIA resynthesis: typical fidelity versus extreme residuals**
    - Full-disk BP example: MAE 4.63 MLP versus 5.29 solver; MSE tail caveat.
    - Hot-channel undershoot and the need for residual views.

19. **Current consolidated state**
    - What is established, what is tentative, and what is currently blocked by
      missing external configuration/access details.

20. **Advisor decisions requested**
    - Scientific primary target, evaluation priority, and next experiment.
    - Hot-channel calibration, ENet, AIA+XRT, uncertainty, or another physical
      target.

## Method slide content

The central method slide should still be simple:
   - AIA observation -> per-pixel MLP -> non-negative DEM basis coefficients.
   - Forward response reconstructs AIA; BP or ENet objective is the training
     signal.
   - Explicitly state: solver DEMs are evaluation references, not targets.

## Materials to prepare

- One clean method diagram.
- Existing figures for each major stage, selected from the results log.
- MLP/CNN, width-sweep, multimodal-quality, and BP self-consistency figures.
- Compact result tables with all metric units labelled.
- Live viewer staged locally from Torch; preselect one quiet and one active date.

## Deliberate exclusions

- Workshop-paper framing, submission schedule, and authorship.
- Raw command logs and repeated copies of the same job failure.
- A claim that the model beats supervised work.
- Unconfirmed ENet hyperparameters or Bright/Quiet table comparisons presented
  as final conclusions.
