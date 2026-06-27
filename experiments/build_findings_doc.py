"""
Build a single Word document compiling all experiment findings so far,
most recent first, with explanations + plots inline.

Run from project root:
    uv run python experiments/build_findings_doc.py
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_PATH = "results/findings_log.docx"

doc = Document()

# Use the largest practical page size (22in, the Word/LibreOffice max) with
# small margins so tall stacked-pixel plots fit on a single page instead of
# being split mid-image across a page break — gives a continuous, scroll-like feel.
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(22)
section.top_margin = Inches(0.4)
section.bottom_margin = Inches(0.4)
section.left_margin = Inches(0.6)
section.right_margin = Inches(0.6)
USABLE_HEIGHT_IN = 22 - 0.4 - 0.4 - 1.0  # leave room for heading/caption text above/below


def safe_width(path, requested_width_in):
    """Shrink width if needed so the image height stays under one page."""
    from PIL import Image
    if not os.path.exists(path):
        return requested_width_in
    w_px, h_px = Image.open(path).size
    aspect = h_px / w_px
    max_w_for_height = USABLE_HEIGHT_IN / aspect
    return min(requested_width_in, max_w_for_height)

# ── styling helpers ────────────────────────────────────────────────────────

def add_title(text):
    h = doc.add_heading(text, level=0)
    return h

def add_date_heading(text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.keep_with_next = True
    return h

def add_subheading(text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.keep_with_next = True
    return h

def add_para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p

def add_finding(text):
    p = doc.add_paragraph()
    r = p.add_run("Finding: ")
    r.bold = True
    r2 = p.add_run(text)
    return p

def add_image(path, width_in=5.5, caption=None):
    if not os.path.exists(path):
        add_para(f"[missing: {path}]", italic=True)
        return
    width_in = safe_width(path, width_in)
    doc.add_picture(path, width=Inches(width_in))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)

def add_divider():
    doc.add_paragraph("─" * 60).alignment = WD_ALIGN_PARAGRAPH.CENTER


# ── doc header ───────────────────────────────────────────────────────────────

add_title("DEM Inversion — Differentiable Loss Experiments Log")
add_para(
    "Running log of experiments comparing differentiable physics-inspired DEM "
    "losses against the classical Basis Pursuit (BP) solver. Most recent results "
    "are listed first; older sets are kept below for reference."
)
doc.add_paragraph()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-27 — Neural field (patch-conditioned, per-image, step 1)
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-27 — Neural Field DEM (Patch-Conditioned, Per-Image): Step 1 Validation")

add_para(
    "Motivation: the previous amortized channel-input NN f(6 AIA channels) → DEM failed in two "
    "compounded ways: (1) many spatially distant pixels share similar 6-channel values but need "
    "different DEMs, so the NN averaged across them and produced oscillating, non-sparse curves; "
    "(2) the network had no spatial context to resolve per-pixel structure. To isolate whether "
    "spatial context alone fixes the sparsity problem — before adding cross-image amortization — "
    "we implemented a per-image patch-conditioned neural field: one model fit jointly over all "
    "pixels of a single AIA timestamp/crop."
)

add_subheading("Architecture: PatchDEMNet")
add_para(
    "Input: a 9×9 local patch of 6-channel AIA data centered on the target pixel, shape [6, 9, 9]. "
    "Edge pixels receive edge-replicated padding so every pixel always has a full 9×9 patch. "
    "The model is a small CNN: 3 convolutional layers (6→64→64→64 channels, 3×3 kernels, "
    "padding=1, SiLU activations) followed by a flatten and 2-layer MLP head "
    "(64×9×9 → 256 → 256 → 54 basis coefficients, Softplus output to enforce positivity). "
    "Total: ~300k parameters."
)

add_subheading("Training")
add_para(
    "One model is trained per image (per-image, not amortized across timestamps). "
    "All ~16,000 valid pixels in a 128×128 crop of timestamp 20110906_2217 are used as "
    "training samples. Each sample is one pixel's 9×9 patch plus its center pixel's AIA "
    "observations and noise bounds (lb = obs − 1.4σ, ub = obs + 1.4σ). "
    "The loss is barrier_loss_batch — the same differentiable physics constraint used in "
    "prior experiments — which simultaneously penalizes: (a) D@x falling outside [lb, ub] "
    "(the AIA observation must be explained within noise), and (b) large L1 norm on the "
    "basis coefficients (sparsity pressure, matching BP's L1-minimization objective). "
    "No BP labels are used — the loss is fully physics-constrained / unsupervised. "
    "Optimizer: Adam with CosineAnnealingLR, gradient clipping at 1.0, 30 epochs, batch size 512. "
    "The shared CNN weights are updated using gradients from all pixels simultaneously — "
    "this is the key spatial regularizer: the weights learn a representation that is consistent "
    "across neighboring pixels, preventing the per-pixel oscillation seen before."
)

add_subheading("Inference")
add_para(
    "At inference time: extract the 9×9 AIA patch centered on the target pixel, run one forward "
    "pass through the frozen model, multiply the 54 output coefficients by the basis matrix B "
    "to get the 18-bin DEM. No LP solver, no per-pixel gradient loop — a single matrix-multiply "
    "chain. This is the speedup over BP, which must solve a linear program per pixel."
)

add_subheading("Sparsity metric")
add_para(
    "To directly detect the failure mode from the previous NN (low barrier loss but non-sparse "
    "solutions), we track effective sparsity (Hoyer L1²/L2 ratio) per pixel during training and "
    "compare against BP's sparsity on the same pixels. Lower = sparser. BP's target value is "
    "computed by solving the real LP on a random 200-pixel subset."
)

add_finding(
    "Training converged cleanly: loss 1.43 at epoch 30. Final NN effective sparsity: 1.30 vs "
    "BP reference: 1.84. The NN is actually sparser than BP on average — a strong step-1 "
    "validation that patch context alone is sufficient to reach BP-like sparse solutions. "
    "Per-pixel DEM curves (below) show the NN tracking BP's single-peaked shape well on most "
    "pixels, with physically meaningful peak temperatures. One limitation: on pixels where BP "
    "finds a bimodal (two-peak) solution with disjoint sparse support (e.g., a low-T secondary "
    "bump near logT 5.5–5.75 plus a main peak near logT 6.25), the NN collapses to a single "
    "smooth peak and misses the secondary component — expected, as the aggregate barrier loss "
    "pushes toward unimodal compromise rather than committing to disjoint support. "
    "Important caveat: this is a per-image fit. The model trains and evaluates on the same "
    "image, so it is not yet generalizing to unseen timestamps. Step 2 (amortized across all 4 "
    "timestamps) is needed before claiming generalization."
)

add_subheading("Per-pixel DEM curves: neural field (cyan dotted) vs BP (black)")
add_image(
    "output/experiments/neural_field_20110906_2217_pixel_dems.png",
    width_in=5.5,
    caption="10 random pixels from 128×128 crop of 20110906_2217 — NN patch-conditioned neural field vs BP"
)

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-19 — NN vs direct optimization
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-19 — Barrier / Barrier-Fit as Neural Networks (NN vs. Direct Per-Pixel Optimization)")
add_para(
    "Trained two neural networks, f(6 AIA channels) → DEM (18 temperature bins via the "
    "basis matrix), by minimizing the barrier loss (and barrier_fit loss) directly as the "
    "training objective — no BP labels needed. Each NN was trained on ~221,000 pixels "
    "pooled across all 4 timestamps for 30 epochs. The trained NNs were then run on the same "
    "individual pixels used in the optimizer comparison (and on random pixels in each "
    "timestamp), and their predicted DEM curves were overlaid against BP and the direct "
    "per-pixel L-BFGS/Adam/SGD optimization curves."
)
add_finding(
    "Both NNs converge cleanly during training (loss plateaus around 4.0 for both barrier "
    "and barrier_fit after ~30 epochs). However, when their predictions are plotted per pixel "
    "against the direct per-pixel optimization, the NN curves are visibly noisy/oscillating — "
    "they do not reproduce the smooth, single-peaked DEM shape that BP and direct optimization "
    "agree on. This happens because the basis matrix used for sparse (L1) recovery contains "
    "non-smooth, overlapping basis functions; even small per-pixel coefficient errors from the "
    "amortized NN get amplified into large visual swings when reconstructed through that basis. "
    "In short: the NN learns a low-loss amortized approximation across many pixels, but for this "
    "ill-posed problem (6 equations, 18 unknowns) that approximation lands on a different, less "
    "stable combination of basis coefficients than the exact per-pixel optimum — a real limitation "
    "of replacing per-pixel optimization with a single forward pass, not just a tuning issue."
)

add_subheading("Training loss curves")
add_image("output/experiments/barrier_train_loss.png", width_in=5.0,
           caption="barrier NN training loss (30 epochs, pooled across 4 timestamps)")
add_image("output/experiments/barrier_fit_train_loss.png", width_in=5.0,
           caption="barrier_fit NN training loss (30 epochs, pooled across 4 timestamps)")

add_subheading("Optimizer comparison + NN overlay (6 pixels, timestamp 20110906_2217)")
add_para(
    "Solid lines = barrier loss; dashed lines = barrier_fit loss; colors = optimizer "
    "(blue=L-BFGS, orange=Adam, green=SGD); dotted cyan/pink = NN predictions for "
    "barrier / barrier_fit respectively. Black = BP reference."
)
add_image("output/experiments/optimizer_comparison.png", width_in=4.5,
           caption="optimizer_comparison.png — BP + 3 optimizers x 2 losses + 2 NNs, per pixel")

add_subheading("Per-pixel DEM curves with NN overlay (all 4 timestamps, 10 random pixels each)")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    add_image(f"output/experiments/pixel_dems/{tag}_pixel_dems.png", width_in=4.5,
               caption=f"{tag} — pixel DEM curves with NN overlay")

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-18 — Optimizer comparison
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-18 — Optimizer Comparison: L-BFGS vs. Adam vs. SGD")
add_para(
    "Before training NNs, tested whether the choice of optimizer matters for direct "
    "per-pixel optimization of the barrier and barrier_fit losses. Ran L-BFGS, Adam, and "
    "SGD on 6 diversely-bright pixels (timestamp 20110906_2217), 2000 steps each, and "
    "compared the resulting DEM curves and resynthesis MAE against BP."
)
add_finding(
    "Optimizer choice barely matters for L-BFGS vs Adam — both converge to nearly identical "
    "curves that closely track BP. SGD, however, fails to converge within the given step "
    "budget on several pixels, producing flat-zero or badly wrong-shaped spikes. Conclusion: "
    "L-BFGS/Adam are both safe choices for per-pixel optimization; avoid SGD here."
)
add_image("output/dem_results/optimizer_comparison.png", width_in=4.5,
           caption="optimizer_comparison.png — BP + 3 optimizers x 2 losses, per pixel (pre-NN)")

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-18 — Wasserstein distance + MAE table
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-18 — Wasserstein Distance vs. BP on Top 5% Brightest Pixels")
add_para(
    "A more physically meaningful comparison metric than raw per-bin MAE: "
    "treat each DEM (normalized to sum to 1) as a probability distribution over logT, and "
    "compute the Wasserstein (Earth Mover's) distance to BP's distribution. This accounts for "
    "mass shifted to the wrong temperature, not just magnitude differences, and is computed "
    "only on the top 5% brightest pixels (by 171Å channel) — the most physically interesting "
    "active-region/flare pixels, where quiet-sun averaging would otherwise hide differences."
)
add_finding(
    "Across all 4 timestamps and both metrics (MAE vs BP and Wasserstein on the brightest 5%), "
    "barrier loss is consistently closest to BP (MAE vs BP: 0.015–0.053; Wasserstein: 0.028–0.037), "
    "while chisq_smooth/entropy/tikhonov diverge much further from BP's DEM shape (Wasserstein: "
    "0.09–0.23) — even though those losses actually achieve LOWER raw MAE vs the observed AIA "
    "channels. This confirms BP is optimizing for sparsity, not best-fit reconstruction, and "
    "barrier (which also imposes an L1 sparsity penalty) is the best differentiable proxy for it."
)

with open("results/mae_table.txt") as f:
    table_text = f.read()
mono = doc.add_paragraph()
run = mono.add_run(table_text)
run.font.name = "Courier New"
run.font.size = Pt(8)

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-09 — Initial 5-loss comparison
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-09 — Multi-Loss Comparison: BP vs. 5 Differentiable Losses (Full Image, All Pixels)")
add_para(
    "First full-scale run: BP + 5 differentiable loss formulations (barrier, barrier_fit, "
    "chisq_smooth, entropy, tikhonov) solved independently per pixel via direct optimization, "
    "across all valid pixels in all 4 timestamps (256x256 crops, ~50-60k valid pixels each)."
)
add_finding(
    "barrier and barrier_fit visually and numerically track BP's spatial structure (mean logT "
    "maps) most closely; chisq_smooth/entropy/tikhonov achieve better raw AIA-channel "
    "reconstruction (lower resynthesis error) but produce smoother/different DEM shapes than BP, "
    "since they don't impose BP's L1 sparsity constraint. This motivated the later Wasserstein "
    "and optimizer-comparison work above."
)

add_subheading("Mean logT spatial maps (BP vs. all 5 losses, per timestamp)")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    add_image(f"results/run_all_losses/{tag}_mean_logt.png", width_in=6.0,
               caption=f"{tag} — mean logT per loss")

add_subheading("Per-pixel DEM curves (10 random pixels per timestamp, pre-NN)")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    add_image(f"results/pixel_dems/{tag}_pixel_dems.png", width_in=4.5,
               caption=f"{tag} — pixel DEM curves, all 5 losses vs BP")


doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
