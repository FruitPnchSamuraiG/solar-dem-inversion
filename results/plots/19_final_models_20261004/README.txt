Final models (chosen 2026-10-04), fresh pixels every epoch, scored on the full shared test set:
  bp_*_final*    = scaled_mlp6_barrier_resample.pt   (h680, 1.43M params; sweep task 19108189_0)
  enet_*_final*  = scaled_mlp6_enet_h336_resample.pt (h336, 360k params; sweep task 19108189_9)
BP: best-or-tied validation loss (2.155; h336 2.148 is within run-to-run noise).
ENet: largest stable width; 722k and 1.43M blow up on a few bright pixels
(AIA MSE 41,849 and 908,611 against 650 here). Sources: 17_sweep_resample_20261003.
Figures: uv run python experiments/plot_bright_failures.py --plain --lf_dir <this dir> --tag final
