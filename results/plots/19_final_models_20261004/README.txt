Final models (chosen 2026-10-04), one 360k MLP per track, fresh pixels every epoch,
scored on the full shared test set:
  bp_*_final*    = scaled_mlp6_barrier_h336_resample.pt (h336, 360k params; sweep task 19108189_2)
  enet_*_final*  = scaled_mlp6_enet_h336_resample.pt    (h336, 360k params; sweep task 19108189_9)
BP: best validation loss of the sweep (2.148; 1.43M 2.155 ties within noise).
ENet: largest stable width; 722k and 1.43M blow up on a few bright pixels
(AIA MSE 41,849 and 908,611 against 650 here). Sources: 17_sweep_resample_20261003.
Figures: uv run python experiments/plot_bright_failures.py --plain --lf_dir <this dir> --tag final
