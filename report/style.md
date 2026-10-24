# House style for the report

Audience: the supervised side (the advisor, the PhD student) and later a paper draft.
Read `report/outline.md` first; each section file covers one outline entry.

## Rules

- No personal names anywhere. Say "the supervised side", "the supervised model", "the
  supervised side's solver settings", "the advisor". The label-free side is "we" or "the
  label-free model".
- No em dashes. Use commas, colons, parentheses or a new sentence.
- Every number comes from a file. Each section ends with a `Sources` list naming the files
  (repo-relative paths) the numbers came from. If a number cannot be found in a file, leave
  a `[[TODO: source]]` marker rather than guessing. Never round in a way that changes the
  comparison (0.310 vs 0.318 stays three decimals).
- Percentiles, not means, wherever the quantity is heavy-tailed (the per-pixel BP loss,
  DEM SSE). Say "median pixel error" for the per-pixel SSE median.
- Vocabulary: "flare cores" = pixels at 32 times the Bright cutoff and above; "Bright" and
  "Quiet" as the supervised side defines them (any channel at or above its top-5%
  threshold); "the band" = the solver's noise band ±1.4σ; "the band weight" = μ; "the
  best answer under the loss" = the per-pixel optimum of the training objective; "the
  stiff band" = μ = 100; "the old model" = width 336, log1p input, μ = 1, 40 epochs;
  "the adopted BP model" = width 336, square root plus 12 Fourier frequencies, μ = 100,
  120 epochs then 120 more continued from that checkpoint; "the ElasticNet model" = the
  same width and input, 40 epochs, the solver's α = 0.001, λ = 0.5, C = 6.
- Units: DN/s for AIA intensities; emission measure units of 10^26 cm^-5 for DEM values
  and the 54 weights; dex for logT differences; MK for temperatures when speaking to a
  physicist (1 MK = logT 6.0, 8 MK = logT 6.9).
- Tables in GitHub markdown. Figures as `![caption](figures/<file>.png)`; name the source
  file under `results/plots/` in the caption's trailing parenthesis so the build script can
  copy it. Do not invent figures; reference only files that exist, or write
  `[[FIGURE: description]]` for one to be made.
- Prose: plain, specific, short paragraphs. State the finding, then the evidence, then the
  caveat. Prefer "the loss rated the solver's DEM 2.4 times worse than one without the hot
  plasma" to "our analysis revealed a significant discrepancy".
- Keep the honest notes: the ElasticNet width (336, chosen because larger widths blow up on
  test) and epochs (40, kept because 120 worsens DEM MSE on test) were decided with test
  evidence; the fixed-sample bug before 2026-10-02; inference speed is unmeasured; μ was
  chosen on validation and is now backed by theory.
- Section files: `report/sections/NN_slug.md`, starting with a level-2 heading
  (`## 5. Results`), subsections level 3. One blank line between blocks. No front matter.
