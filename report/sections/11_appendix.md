## Appendix

### A. History

The June and early-July results come from four days, each a crop near disk centre (the X2.1 flare of 2011-09-06, quiet Sun on 2012-06-03, moderate activity on 2013-11-13 and the X1.6 flare of 2014-09-10). Every scaled run from 31 July to 2 October trained on one fixed sample of pixels (Appendix C).

**Which loss (June).** Before any network, five losses were optimised directly, pixel by pixel, on the four days. The fit-only losses (χ² plus smoothness, maximum entropy, Tikhonov) reconstruct AIA far more closely than BP (AIA MAE 0.40 to 0.44 against BP's 5.2) yet land far from its DEM (DEM MAE against BP 0.18 to 0.20), while BP's own terms, the band penalty with L1 sparsity, land on it (0.034, or 0.043 with an added fit term). Six channels underdetermine an 18-bin DEM, so the tie-breaker decides the shape, and a network meant to emulate a solver has to train on its objective.

**The first network (19 June).** A per-pixel MLP of width 256 (213k parameters) on the six raw intensities, trained on the band-penalty loss over about 221,000 pixels. Its curves were described at the time as noisy; retrained with the unchanged code on 2026-10-03, they are single-peaked (1.00 to 1.03 local maxima per curve, none with three or more, against BP's 1.12 to 1.19). Its real deficit was accuracy: on 400 held-out pixels per day its DEM MAE against BP is 0.078, against the patch CNN's 0.047 (W1 0.055 against 0.048 dex).

**The patch CNN (27 June).** A CNN on the 9 by 9 neighbourhood (about 1.5M parameters) predicting the same 54 weights, trained first on one crop (sparsity score on the 54 weights 1.30, BP 1.84), then amortised over the four days (1.47 to 1.86 on held-out pixels, BP 1.67 to 1.97), then with the neighbourhood zeroed on 10% of batches so that it also works on a single pixel. The held-out pixels came from the training days, so none of this was a test.

**The ablation (2 July) and leave-one-day-out (11 July).** Four variants of about 1.45M parameters on the same loss and days: the patch CNN, a centre-pixel MLP (1.43M), a flat-patch MLP and a CNN on a fixed random permutation of the patch, with sparsity scores (BP 1.79) of 1.70, 1.89, 1.95 and 1.88; the patch CNN on the ElasticNet loss scored 2.28. The centre-pixel MLP drew smooth curves and was the closest to BP (DEM MAE 0.042 against the CNN's 0.047), so the first MLP's gap was capacity or input scaling (the two were not separated), not missing neighbours. Leave-one-day-out then removed the CNN's in-sample edge (section 5.4). Its weak fold was the X1.6 flare day (peaks at logT about 6.1 against BP's 6.4), with only one other flare day in training.

**The first bimodal check (11 July).** On the crops 5.1% to 9.1% of the scanned BP DEMs were bimodal, and only 6 of 120 studied pixels stayed bimodal across at least 80% of 20 photon-noise perturbations, all on the flare days. The conclusion that bimodality was mostly noise did not survive the move to scale (2 August, below).

**Labels at scale (31 July).** BP and ElasticNet were run on 1,223 Hofmeister-deconvolved AIA images (2,446 cluster jobs, zero errors, 888 GB per solver), split in time into 917 training, 153 validation and 153 test images. Dropping BP's `--zerochill` flag, which disables its retry at a wider tolerance, cut the unsolved pixels from 10.2% to 0.02%: photon noise narrows the relative band as pixels brighten until the LP becomes infeasible, so the flag had discarded mostly bright pixels. About 4.4% of pixels carry a hard zero in 171 or 193 Å, the two brightest channels, which only the deconvolution's positivity clamp can produce, and BP fits a DEM to them anyway. Pixels with any channel at or below MIN_OBS = 10^-3 DN/s, three decades below the faintest channel's median (94 Å, 0.76 DN/s), were then dropped from training at an estimated cost of about 5% of pixels; the count over all six channels, made in October, is 40% (sections 3.4 and 7.1).

**The collapse and its fix (31 July).** The first four full-disk runs exited cleanly and predicted an identically zero DEM. The architectures had been validated on crops spanning about one decade of intensity; the full disk runs from the 10^-3 DN/s floor off the limb to about 10^4 DN/s in flare cores. Fed raw intensities, the untrained network predicted AIA about seven times above the band, the loss at step 0 was 5.3 × 10^13, and the output softplus started so far negative (pre-activation -139.8) that in float32 its gradient was exactly zero. By step 60, 98% of outputs were dead, and the runs then repeated a bit-identical epoch loss for eleven epochs. Gradient clipping cannot help, as Adam normalises the step size. The fix has four parts: log1p on the network input only, with the loss, the band and every metric left in DN/s; a linear learning-rate warm-up (500 steps, raised to 3,000 when the CNN on the ElasticNet loss still died at initialisation); a clamped softplus whose pre-activation is floored at -20, so its gradient never vanishes; and a guard that stops a run whose sparsity is zero or whose epoch loss repeats exactly. A 64-block smoke test could not have caught the failure, because it never drew a flare pixel.

**First converged scaled runs (31 July).** With the fix all four runs converged, and on the first four validation images the centre-pixel MLP beat the patch CNN on both training losses (validation loss 2.145 against 2.241 on BP, 1.850 against 1.858 on ElasticNet) and on sparsity (1.90 against 1.96, BP 1.79). These ElasticNet runs used α = 1, the label-generation default, not the supervised side's α = 0.001 (Appendix B).

**The test split and the mean (2 August).** On the 153 test images the sparsity score and AIA MAE reproduced validation to the third decimal. The mean per-pixel BP loss proved unusable: the single worst pixel held 94.0% of the MLP's mean, which put that mean at four times the CNN's although the MLP was better at every percentile, so the BP loss has been reported by percentiles since. A census of 190,337 held-out pixels found 15.96% of BP DEMs bimodal (14.17% with both peaks interior); the MLP flagged interior-bimodal pixels at 80.75% precision and 29.85% recall.

**The width sweep and the capacity ceiling (3 to 7 August).** MLPs from 10k to 11.2M parameters, 40 epochs each. The sweep chose width 232 (176k parameters): eight times smaller than 1.43M at nearly the same DEM MAE (0.1266 against 0.1219) and a lower AIA MAE (4.792 against 4.898), while at width 160 the recall of bimodal pixels fell from 26.3% to 8.1%. Above width 232 recall stayed at 26% to 30% up to 11.2M. Re-solving BP under photon noise at 80 single-peaked and 80 double-peaked bright pixels (30 re-solves each) put the network's distance from BP at 0.69 and 0.98 times BP's own re-solve scatter, so most of the double-peak deficit was the label's irreproducibility. With fresh pixels (Appendix C) the recall cliff did not reproduce and width 336 beat 232 on nearly every test measure, so 336 became the width of both tracks' models at the time (section 5.4; the BP final later moved to 680, section 3.5); for the old model the self-consistency ratios are 0.97 at double-peaked and 0.70 at single-peaked pixels.

### B. ElasticNet detail and the epochs trade

**Objective and settings.** The ElasticNet model trains on the solver's objective itself (section 3.2), with the supervised side's settings: scikit-learn's ElasticNet, non-negative, no intercept, α = 0.001, l1 ratio λ = 0.5 and C = 6, the number of channels. The same settings produced the supervised model's labels and every ElasticNet test target. The label-free ElasticNet runs of July and August used α = 1, the label-generation default; they are valid for their own objective but not comparable with the supervised side's solver, and no number in this report comes from them.

**Input, width and epochs.** The square root plus 12 Fourier frequencies input was chosen on validation (validation loss 0.0533 against 0.0575 for log1p) and lifted the 336-wide model past the supervised model on every measure (Table 5.2). The width and the number of epochs were settled with test evidence: with clamp pixels excluded, wider models blow up on a few rare pixels that the validation sample did not contain (sections 5.4 and 7.1). Keeping clamp pixels removes most of the 10M model's blow-up but costs the 409k model more than it gives (Table B.1), so the ElasticNet model excludes them.

Table B.1. ElasticNet follow-up runs on the shared test set; the first row repeats the ElasticNet model's numbers from Table 5.2.

| ElasticNet run | DEM MSE | DEM MAE | Median pixel error | EM error | W1 (dex) | AIA MAE | AIA MSE | Flare-core emission | Flare-core 94 Å |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ElasticNet model (40 epochs) | 0.310 | 0.151 | 0.85 | 12.2% | 0.103 | 0.56 | 161 | 99% | 53% |
| Oversample flare cores | 0.379 | 0.152 | 0.89 | 12.2% | 0.105 | 0.57 | 141 | 102% | 55% |
| 120 epochs | 0.362 | 0.144 | 0.78 | 11.9% | 0.100 | 0.52 | 188 | 102% | 55% |
| 10M parameters | 2,062 | 0.159 | 0.78 | 15.2% | 0.099 | 2.62 | 83M | 216% | 107% |
| Clamp pixels kept | 0.439 | 0.201 | 1.55 | 17.0% | 0.143 | 0.62 | 162 | 101% | 54% |
| 10M parameters, clamp pixels kept | 1.82 | 0.198 | 1.48 | 17.2% | 0.143 | 0.71 | 1.5 × 10⁵ | 104% | 56% |

**The epochs trade.** 120 epochs improves five measures (DEM MAE 0.151 to 0.144, median pixel error 0.85 to 0.78, EM error 12.2% to 11.9%, W1 0.103 to 0.100, AIA MAE 0.56 to 0.52) and worsens two (DEM MSE 0.310 to 0.362, AIA MSE 161 to 188). The 40-epoch model was kept: DEM MSE is the one headline measure on which the supervised model comes close (0.318), and at 120 epochs the label-free model would lose it. That choice was made on the test set. Oversampling worsens DEM MSE (0.379) and changes little else; the 10M model blows up on a few rare pixels, as on BP. Clamp pixels in training cost the 409k model on most measures; the 10M model with them matches that run on DEM MAE, the median and W1, but has DEM MSE 1.82 and AIA MSE 1.5 × 10⁵.

**Flare cores.** Flare cores hold 44.5% of the ElasticNet model's DEM squared error (the supervised model's: 33.9%), but this is not the BP failure. The ElasticNet loss is the solver's objective itself, not a relaxation of it, so its best answer is the solver's own answer and there is no band to price. At flare cores the model reproduces the channels as the ElasticNet solver does, including the solver's own shortfall on 94 Å, which comes from its regularisation trading fit for smoothness.

Table B.2. ElasticNet flare cores on the shared test set (75,832 pixel-targets).

| | ElasticNet model | Supervised | Solver |
| --- | --- | --- | --- |
| Emission vs solver | 99% | 87% | 100% |
| Peak height vs solver | 101% | 92% | 100% |
| 94 Å reproduced | 53% | 48% | 55% |
| 131 Å reproduced | 80% | 68% | 78% |
| 171 Å reproduced | 97% | 70% | 100% |
| Curves with extra peaks | 18% | 18% | 21% |

### C. The fixed-sample bug

**What happened.** Until 2026-10-02 every scaled run drew the same 512 pixels from each block in every epoch, because the draw was seeded by the block index alone (`src/zarr_data.py`). Training therefore saw one fixed sample of 58,688 blocks × 512 = 30M pixels, about 0.8% of the roughly 3.8 billion training pixels, 40 times over. The results stayed valid for what they measured (test matched validation to the third decimal), but rare pixels were under-sampled: those at 10 or more times the Bright cutoff, about 0.09% of pixels, were a fixed set of about 27,000.

**The fix.** `--resample_pixels` (commit `7067355`, `EpochResampler` in `src/zarr_data.py`) offsets each epoch's indices so that every epoch draws fresh pixels; the first epoch's draw is unchanged, so default loaders behave as before (`tests/test_zarr_data.py`).

**The first retrain came out worse.** The width-232 models retrained with fresh pixels and otherwise identical settings scored worse than the fixed-sample ones by more than run-to-run noise (identical configurations agree to 0.2% to 0.4% in validation loss).

Table C.1. Width 232, fixed sample against fresh pixels, shared test set.

| | BP, fixed | BP, fresh | ElasticNet, fixed | ElasticNet, fresh |
| --- | --- | --- | --- | --- |
| Validation loss | 2.156 | 2.237 | 0.0582 | 0.0689 |
| DEM MSE | 4.19 | 5.28 | 0.588 | 0.794 |
| Median pixel error | 0.026 | 0.030 | 1.41 | 1.70 |
| Flare-core emission | 41% | 15% | 72% | 73% |
| Flare-core 94 Å | 24% | 7% | 26% | 40% |
| Flare-core curves with extra peaks | 83% | 89% | 81% | 40% |

A plausible reason, untested: the objective is heavy-tailed, and a fixed sample lets the model absorb its pathological pixels once, while fresh pixels bring new ones every epoch, whose clipped gradients dominate the steps.

**What it changed.** The method trains on fresh pixels, so the report uses fresh-pixel models even though this first retrain scored worse. The width sweep and the CNN comparison were rerun with fresh pixels: width 336 beat 232 on nearly every test measure on both tracks (BP DEM MSE 4.28 against 5.28, median pixel error 0.026 against 0.030; ElasticNet DEM MSE 0.48 against 0.79), the recall cliff at width 160 did not reproduce, and the patch CNN still lost (section 5.4). Every scaled result dated 31 July to 28 September, including the September flare-core diagnosis, was measured on the fixed sample, and Appendix A labels it as history.

### D. Reproducibility

**Checkpoints.** On the cluster, under `output/experiments/` (not tracked in git):

- Final BP model: `output/experiments/followup/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120.pt`.
- ElasticNet model: `input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt`.
- 409k stiff-band model: `followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt`, warm-started from `followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120.pt`.
- Old model: `resample/scaled_mlp6_barrier_h336_resample.pt`.

The supervised models are in the repository, as supplied: `eval_and_enet_specs/checkpoints/model_best_bp.pth` and `model_best_en.pth`.

**Data.** Training and validation read the staged zarr stores `$SCRATCH/dem/data/lp_AIA_hofdeconv_full_DS/` (BP) and `$SCRATCH/dem/data/elasticnet_AIA_hofdeconv_full_DS/` (ElasticNet): per split, AIA in `{phase}_x`, the solver DEM in `{phase}_y` and the solver's tolerance level in `{phase}_m`. Training never reads `{phase}_y`; in the ElasticNet store it holds the older α = 1 labels and serves only validation diagnostics. Every test number comes from the supervised side's staged shared test set (48,960 blocks per track; its path is recorded in the `data` field of each diagnostic JSON), scored with the Bright thresholds in `eval_and_enet_specs/aia_thresholds.json` and the protocol in `eval_and_enet_specs/README.md`.

**Scripts** (names only). Label generation and staging: `dataset/inv_bp_aia_hofdeconv_full.sh`, `eval_and_enet_specs/submit_enet_aia_hofdeconv_full.py`, `dataset/stage_hofdeconv_full.sh`, `dataset/job_stage_hofdeconv.sbatch`. Training (`experiments/train_scaled.py`): `experiments/job_scaled.sbatch`, `job_train_resample.sbatch`, `job_sweep_resample.sbatch`, `job_train_input_encoding.sbatch`, `job_train_enet_alpha0p001.sbatch`, `job_train_followup.sbatch` (submitted by `submit_followup.sh`; the final BP model is its `bp_h680_clampmask` entry, width 680, μ = 100, 120 epochs, `CLAMP_MODE=mask`). Evaluation: `job_eval_full_paper_test.sbatch`, `job_eval_supervised_paper_test.sbatch`, `job_diagnose_bright_failures.sbatch`, `job_aia_fit_by_brightness.sbatch`, `job_eval_followup_gpu.sbatch`, `job_eval_scaled.sbatch`, `job_eval_sweep.sbatch`, `job_objective_gap.sbatch`. The October follow-ups of section 7 and section 8: `job_mu_theory.sbatch`, `job_blowup_census.sbatch`, `job_jpdf_pixels.sbatch`, `job_final_layer.sbatch`. Figures: `experiments/plot_bright_failures.py --plain`, with provenance in each `results/plots/` folder's README.

**Compute.** GPU jobs stay under the cluster's 2-hour limit; longer runs are chains of resumable segments. An epoch of the 336-wide model takes about 150 s on one GPU, so 120 epochs is about 5 hours and 240 about 10; the final BP model took a median 169 s per epoch and 6.6 hours in all. The solvers took 9.5 min (BP) and 5.25 min (ElasticNet) per image in our July label-generation jobs. The network's inference time per image has not been measured, so the report claims no speed-up over the solvers.

**Website.** The viewer (code in `student_package/visuals_pipeline/webapp/compare.html` and `compare.js`; images built by `experiments/job_visuals_final_cpu.sbatch`) is public at https://triborough.cs.nyu.edu/hsr3649/demdemo/webapp/compare.html. Its code now shows the final BP model as run `bp_mlp6_h680_clamp`, with the self-check of section 7.1 blanking pixels whose DEM overshoots the observation by more than 100 times; the ElasticNet run is unchanged (`enet_mlp6_h336_sqrtff`). From 2026-10-19 the site showed the 409k stiff-band model (312 BP images), and an audit that day found all 1,875 build files byte-identical on the live site and all 72 numbers in its Results view matching the result JSONs; that audit has not been repeated for the new run. The server's TLS certificate expired on 2026-09-16, so browsers show a security warning; the warning concerns the server, not the viewer. The server was unreachable from 21 to 23 October and back on 24 October; a full copy of the site (6.5 GB) is kept off the server.

Sources: A: `CLAUDE.md` (progress log 2026-06-09 to 2026-08-07: the loss comparison, the first MLP, the patch CNN steps, the ablation, leave-one-day-out, the bimodal check, label generation, the collapse, the first converged runs, the test split and census, the width sweep, self-consistency and the capacity ceiling), `slides/deck_outline.md` (slides 28 to 32: the loss-comparison numbers and the collapse), `results/paper_outline_20260807.md` (M5, M6), `results/plots/15_first_mlp_repro_20261002/first_mlp_vs_patch_cnn.json` (first-MLP rerun: maxima, DEM MAE and W1 per day, averaged), `results/plots/10_scaled_test_20260802/bimodal_scaled_summary.json` (190,337 pixels, 15.96%), `results/plots/19_final_models_20261004/bp_self_consistency_h336.json` (0.97 and 0.70), `STATE.md` (2026-10-03 entries). B: `eval_and_enet_specs/README.md`, `slides/talk_prep.md` (where the settings come from), `CLAUDE.md` (2026-09-11, 2026-10-04, 2026-10-15), `results/followup_recap_20261013.md` (Part 10), `results/plots/21_followup_20261006/summary.json` (Table B.1), `results/plots/23_meeting_followup_20261024/enet_bright_failure_fu_enet_clampmask.{json,txt}`, `enet_bright_failure_fu_enet_10m_clampmask.{json,txt}`, `enet_aia_fit_fu_enet_clampmask.txt` and `enet_aia_fit_fu_enet_10m_clampmask.txt` (the clamp rows of Table B.1), `results/plots/19_final_models_20261004/enet_bright_failure_final.json` and `enet_aia_fit_final.json`, `results/plots/13_bright_diagnostic_20260928/enet_bright_failure_supervised.json` (Table B.2), `slides/deck_outline.md` (slide 34). C: `CLAUDE.md` (2026-10-02, 2026-10-03), `STATE.md` (2026-10-02 and 2026-10-03 entries, including the fresh-pixel sweep), `results/plots/16_resample_20261002/bp_comparison.txt` and `enet_comparison.txt` (Table C.1). D: `STATE.md` (2026-10-14 block, 2026-10-04 checkpoints, the viewer notes), `CLAUDE.md` (2026-07-31, 2026-10-13, 2026-10-19, 2026-10-21), `experiments/submit_followup.sh` (the warm-start checkpoint, segment chaining, the clamp-mask runs), `results/plots/23_meeting_followup_20261024/bp_h232_h680_clampmask.json` (the final checkpoint's path), `results/plots/21_followup_20261006/scaled_mlp6_barrier_*_history.json` and `results/plots/23_meeting_followup_20261024/scaled_mlp6_barrier_h680_sqrt_ff12_resample_clampmask_mu100_ep120_history.json` (epoch times), `student_package/visuals_pipeline/webapp/compare.js` (the viewer's runs and the self-check note), the `experiments/` and `dataset/` directory listings.
