Final models (chosen 2026-10-04 by validation loss), one 360k MLP (4 hidden layers of 336) per
track, fresh pixels every epoch, scored on the full shared test set:
  bp_*_final*    = output/experiments/resample/scaled_mlp6_barrier_h336_resample.pt
                   log1p input (sweep task 19108189_2). sqrt+Fourier input tied on validation
                   (2.1472 vs 2.1478), so the simpler log1p stays.
  enet_*_final*  = output/experiments/input_encoding/scaled_mlp6_enet_h336_sqrt_ff12_resample.pt
                   sqrt + 12 Fourier frequencies input (job 19141672); validation 0.0533 vs
                   0.0575 for log1p.
Width: BP validation best at 360k (2.148; 1.43M ties at 2.155); ENet 722k and 1.43M blow up
on a few bright pixels (AIA MSE 41,849 and 908,611). Sources: 17_sweep_resample_20261003,
20_sqrtff_final_20261004. bp_self_consistency_h336.json is for the BP model.
Figures: uv run python experiments/plot_bright_failures.py --plain --lf_dir <this dir> --tag final
