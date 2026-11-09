Follow-up runs of 9 Nov 2026 (Torch). All supervised predictions clipped at zero unless marked raw.

headline_vs_supervised_10m.txt  shared test set, same pixels: our final models against the supervised
                                paper models (10.13M) and the 1.05M checkpoint, raw and clipped
                                (experiments/summarize_followup.py headline_runs.json)
*_bright_failure_supervised_*   diagnose_bright_failures.py on the supervised models (sup10m, clamped = 1.05M)
*_aia_fit_sup10m / sup1m_clamped aia_fit_by_brightness.py, our final model and the supervised one
bp_full_validation_sup10m_*     full_validation.py, every pixel of our own val/test images, final vs 10M
                                (the relaxed-pixel median 4.217 is identical for both models: a pile-up of
                                identical errors at relaxed pixels, not a meaningful median)
bp_objective_gap_tallpeaks_*    objective_gap.py at mu 100 with a mu 300 variant: peak height at pixels whose
                                solver peak exceeds 3,200, for the network, the loss's optimum, the solver
                                and the 10M supervised model
synthetic_dem_test.*            synthetic_dem_test.py: known-answer DEMs (gauss, two, solver families)
bp_spike_check_*                spike_check.py: neighbor rules against the self-check, every pixel incl.
                                unsolved; *_examples.npz hold 9x9 cutouts; spike_examples_val.png draws them
*_date_metrics_20110906_2217    date_metrics.py on the X2.1 flare of 2011-09-06 (job_xclass_date.sbatch)
