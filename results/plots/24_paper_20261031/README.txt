Paper draft, 31 Oct 2026. Everything here is drawn from saved results; no model or solver was retrained.

fig_disk.png                     Full-disk maps, one held-out image, from the viewer images in ~/dem_preview (paper/make_figures.py)
fig_law.pdf                      The penalty law on validation pixels (paper/make_figures.py)
fig_curves.pdf                   Solver DEM against the loss's minimizer at mu 1 and 100 at random flare cores (paper/make_figures.py)
law_threshold_check.json         Per-pixel mu_0 from the LP against the saved minimizers (experiments/law_threshold_check.py)
drop_threshold_check.json        Per-channel mu_drop and the clipped law at 94 A (experiments/check_drop_threshold.py)
inference_time_laptop_cpu.json   Forward-pass timing on four laptop CPU threads (experiments/time_inference.py)
