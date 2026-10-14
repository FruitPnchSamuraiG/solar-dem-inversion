Adopted BP model (2026-10-13): stiff band (barrier weight mu = 100), square root + 12 Fourier
frequencies input, width 336 (409k parameters), 120 epochs then 120 more continued from that
checkpoint. Checkpoint on Torch:
  output/experiments/followup/scaled_mlp6_barrier_h336_sqrt_ff12_resample_mu100_ep120_cont.pt
ElasticNet is unchanged (19_final_models_20261004, tag final).

Figures drawn by experiments/plot_bright_failures.py --plain, with --lf_dir pointing at a
folder holding (tag adopted):
  bp_bright_failure_adopted.json      = 21_followup_20261006/bp_bright_failure_fu_bp_mu100_ep120_cont.json
  bp_aia_fit_adopted.json             = 21_followup_20261006/bp_aia_fit_fu_bp_mu100_ep120_cont.json
  bp_aia_fit_adopted_examples.npz     = 21_followup_20261006/bp_aia_fit_fu_bp_mu100_ep120_cont_examples.npz
  enet_*_adopted.*                    = 19_final_models_20261004/enet_*_final.*
Supervised comes from 13_bright_diagnostic_20260928 as always. The deck (slides/build_deck_v2.py)
uses fig4_example_curves_bp.png and fig5_aia_fit_flare_cores.png from here.
