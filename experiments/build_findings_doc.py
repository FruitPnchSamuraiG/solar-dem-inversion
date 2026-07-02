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
# 2026-07-02 — Ablation study: why does the patch CNN work?
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-07-02 — Ablation Study: Why Does the Patch CNN Work? (+ ElasticNet Loss)")

add_para(
    "Motivation: the patch CNN fixed the oscillating-curve failure of the earlier per-pixel "
    "channel-input MLP, but we had no verified explanation for WHY. Several confounds were "
    "possible: the CNN has ~7x the parameters of the old MLP (1.48M vs 213k), it sees a 9x9 "
    "neighborhood instead of 1 pixel, and it uses convolution. To separate these, four "
    "capacity-matched variants (~1.43-1.50M params each) were trained with identical data, "
    "loss (barrier), and budget (30 epochs, 4 timestamps amortized, 44,796 train pixels, "
    "80/20 held-out val split): "
    "cnn = the reference patch CNN; "
    "mlp6 = MLP on the 6 center-pixel channels only (capacity vs information); "
    "mlp_patch = MLP on the flattened 486-dim patch, no convolution (patch info vs convolution); "
    "cnn_shuffled = patch CNN with a fixed random permutation of the 81 spatial positions "
    "(neighborhood values vs geometry). "
    "Additionally, the reference CNN was trained with the ElasticNet loss "
    "(Athiray & Winebarger 2024) — naturally unconstrained and differentiable, "
    "fit + L1 + L2 with lam=0.9 (mostly-L1, BP-like) — as a loss-function comparison. "
    "Implemented in experiments/train_ablations.py; combined plots via "
    "experiments/plot_ablation_comparison.py."
)

add_subheading("Held-out validation results (sparsity: lower = sparser = closer to BP)")
abl_sp_table = (
    f"{'Timestamp':<16} {'BP':>5}  {'cnn':>6}  {'mlp6':>6}  {'mlp_patch':>9}  {'cnn_shuf':>8}  {'cnn(enet)':>9}\n"
    f"{'-'*68}\n"
    f"{'20110906_2217':<16} {'1.97':>5}  {'1.87':>6}  {'1.87':>6}  {'1.61':>9}  {'1.95':>8}  {'2.23':>9}\n"
    f"{'20120603_0000':<16} {'1.67':>5}  {'1.54':>6}  {'1.95':>6}  {'2.27':>9}  {'1.80':>8}  {'2.46':>9}\n"
    f"{'20131113_0908':<16} {'1.82':>5}  {'1.48':>6}  {'1.63':>6}  {'2.15':>9}  {'1.65':>8}  {'2.02':>9}\n"
    f"{'20140910_1731':<16} {'1.71':>5}  {'1.89':>6}  {'2.12':>6}  {'1.76':>9}  {'2.13':>8}  {'2.40':>9}\n"
    f"{'mean':<16} {'1.79':>5}  {'1.70':>6}  {'1.89':>6}  {'1.95':>9}  {'1.88':>8}  {'2.28':>9}\n"
)
mono = doc.add_paragraph()
run = mono.add_run(abl_sp_table)
run.font.name = "Courier New"
run.font.size = Pt(8)

add_para(
    "MAE vs the BP solution: cnn 3.00/2.93/4.00/4.83, mlp6 3.30/3.25/4.41/4.34, "
    "mlp_patch 2.83/2.54/3.94/4.70, cnn_shuffled 2.92/2.75/3.84/4.73, "
    "cnn(enet) 1.34/1.42/1.91/2.75 per timestamp. Note MAE here is measured against BP as "
    "ground truth (not against the observed AIA image)."
)

add_finding(
    "(1) The results are genuine and reproducible: the reference cnn's sparsity "
    "(1.87/1.54/1.48/1.89) matches the earlier independent amortized run "
    "(1.84/1.58/1.47/1.86) almost exactly. "
    "(2) The biggest surprise, which revises our earlier explanation: mlp6 — center-pixel "
    "input only, but 1.43M parameters — produces SMOOTH, SINGLE-PEAKED curves. No variant "
    "oscillates. The old 213k-parameter channel-input MLP's noisy curves were therefore "
    "driven substantially by model capacity (and possibly training budget), not solely by the "
    "missing spatial context we previously credited. "
    "(3) The patch + convolution still measurably helps: cnn is the only variant whose mean "
    "sparsity (1.70) is at/below the BP reference (1.79) — the others cluster at 1.88-1.95 — "
    "and cnn has the fewest per-pixel outliers in the curve plots, while mlp6 shows the most "
    "visible failure cases (peak shifted to the wrong temperature on several flare-timestamp "
    "pixels, e.g. 20140910_1731 pixels (91,70) and (8,1)). "
    "(4) cnn_shuffled is close to cnn but slightly worse on 3 of 4 timestamps: most of the "
    "patch benefit comes from the neighborhood VALUES/statistics, with spatial geometry "
    "adding a modest extra gain. "
    "(5) mlp_patch is erratic — best of all variants on the X2.1 flare (1.61) but worst on "
    "quiet sun (2.27) and moderate activity (2.15). The patch information is available to it, "
    "but without convolution's weight sharing it uses that information unreliably. "
    "Convolution's contribution is consistency, not raw capability. "
    "(6) ElasticNet loss trains cleanly and roughly halves MAE vs BP, yet its curves are "
    "visually the broadest and least sparse (mean sparsity 2.28) — systematically wider "
    "peaks with elevated low-T wings. This is expected: ENet's L2 component smooths solutions "
    "by design, and it doubles as a demonstration that MAE against BP is a misleading metric "
    "(broad curves score lower pointwise error against BP's sharp peaks than sharp-but-"
    "slightly-shifted ones do). ENet is a viable alternative target, but it approximates a "
    "different solver behavior, not a better approximation of BP. "
    "(7) Bimodal BP solutions are still missed by every variant — unchanged fundamental "
    "limitation. "
    "Bottom line for 'why does the patch CNN work': capacity matters more than we previously "
    "believed (a big center-pixel MLP already produces smooth curves); on top of that, the "
    "9x9 patch input gives a consistent additional gain in matching BP's sparsity, driven "
    "mostly by the neighborhood values themselves, with convolution providing reliability "
    "across solar conditions rather than being essential. "
    "Open follow-up: rerun the exact old 213k architecture (hidden=256) in this identical "
    "harness to cleanly separate capacity from the other differences (dataset pooling, "
    "hyperparameters) between the old and new training setups."
)

add_subheading("Per-pixel DEM curves: all 5 variants (dashed) vs BP (black), 10 held-out val pixels")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    act = {"20110906_2217": "X2.1 flare", "20120603_0000": "Quiet sun",
           "20131113_0908": "Moderate activity", "20140910_1731": "X1.6 flare"}[tag]
    add_image(
        f"results/plots/experiments/ablation_comparison_{tag}.png",
        width_in=5.5,
        caption=f"{tag} ({act}) — cnn / mlp6 / mlp_patch / cnn_shuffled / cnn(enet) vs BP"
    )

add_subheading("Training loss curves")
for variant, label in [("cnn_barrier", "cnn (barrier)"), ("mlp6_barrier", "mlp6 (barrier)"),
                        ("mlp_patch_barrier", "mlp_patch (barrier)"),
                        ("cnn_shuffled_barrier", "cnn_shuffled (barrier)"),
                        ("cnn_enet", "cnn (ElasticNet)")]:
    add_image(
        f"results/plots/experiments/ablation_{variant}_train_loss.png",
        width_in=4.5,
        caption=f"Ablation training loss — {label}, 30 epochs"
    )

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-27 — Neural field step 3: neighborhood masking
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-27 — Neural Field Step 3: Neighborhood Masking for Single-Pixel Robustness")

add_para(
    "Project constraint: the model must work even when only a single pixel is available "
    "(no surrounding neighborhood). Step 3 adds random neighborhood masking during training: "
    "with probability mask_prob, the entire 9x9 patch is zeroed out except the center pixel, "
    "forcing the network to learn a fallback when no spatial context is available. "
    "Four mask_prob values were trained and compared: 0.1, 0.3, 0.5, 0.7, plus the no-mask "
    "baseline from step 2. All use the same architecture (PatchDEMNet, amortized across 4 "
    "timestamps, 80/20 val split, same random seed)."
)

add_subheading("Sparsity comparison (lower = sparser = closer to BP)")
mask_sp_table = (
    f"{'Timestamp':<22} {'Activity':<14} {'BP':>5}  {'no mask':>8}  {'m=0.1':>7}  {'m=0.3':>7}  {'m=0.5':>7}  {'m=0.7':>7}\n"
    f"{'-'*85}\n"
    f"{'20110906_2217':<22} {'X2.1 flare':<14} {'1.97':>5}  {'1.84':>8}  {'1.89':>7}  {'1.90':>7}  {'2.12':>7}  {'2.14':>7}\n"
    f"{'20120603_0000':<22} {'Quiet sun':<14} {'1.67':>5}  {'1.58':>8}  {'1.55':>7}  {'1.69':>7}  {'2.15':>7}  {'1.81':>7}\n"
    f"{'20131113_0908':<22} {'Moderate':<14} {'1.82':>5}  {'1.47':>8}  {'1.48':>7}  {'1.62':>7}  {'2.15':>7}  {'1.90':>7}\n"
    f"{'20140910_1731':<22} {'X1.6 flare':<14} {'1.71':>5}  {'1.86':>8}  {'2.00':>7}  {'2.15':>7}  {'2.05':>7}  {'2.13':>7}\n"
)
mono = doc.add_paragraph()
run = mono.add_run(mask_sp_table)
run.font.name = "Courier New"
run.font.size = Pt(8)

add_finding(
    "mask_prob=0.1 is the best choice: stays closest to BP's sparsity across all timestamps "
    "while gaining single-pixel robustness. Higher mask values (0.5, 0.7) make solutions "
    "progressively less sparse — the network spreads probability across more temperature bins "
    "when forced to work without neighborhood context more often. "
    "Note: MAE (AIA channel reconstruction error) decreases with higher masking, but this is "
    "not a meaningful win — removing the sparsity constraint always lowers MAE trivially. "
    "BP sparsity similarity is the correct metric since BP is the ground truth we approximate. "
    "Why does high mask still work at all (unlike the old channel-input NN that also saw only "
    "one pixel and failed)? Because the masked network was trained with (1-mask_prob) full-patch "
    "batches — the CNN weights learned spatial structure from those samples. When a masked "
    "sample arrives, the network falls back on weights that already encode spatial knowledge. "
    "The old channel-input NN never had patch context, so its weights never learned spatial "
    "structure at all. Masking preserves spatial knowledge in the weights; it just teaches "
    "the network to also function without it."
)

add_subheading("Training loss curves — all mask values")
add_para(
    "All runs converge cleanly. Higher mask values start with higher initial loss "
    "(harder problem with less neighborhood context) but plateau equally well by epoch 5-7."
)
for mp in ["0.1", "0.3", "0.5", "0.7"]:
    add_image(
        f"results/plots/experiments/neural_field_amortized_4ts_mask{mp}_train_loss.png",
        width_in=5.0,
        caption=f"Training loss — mask_prob={mp}, 4 timestamps, 30 epochs"
    )

add_subheading("All mask variants vs BP per pixel — 4 timestamps")
add_para(
    "Each subplot shows BP (black solid) and all 5 NN variants (colored dashed) on the same "
    "axes. Key observations: (1) all variants cluster tightly — mask level barely changes "
    "curve shape; (2) no oscillation anywhere; (3) no mask / mask 0.1 closest to BP peak "
    "sharpness; (4) bimodal BP solutions missed by ALL variants — fundamental architecture "
    "limitation, not a masking issue."
)
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    act = {"20110906_2217": "X2.1 flare", "20120603_0000": "Quiet sun",
           "20131113_0908": "Moderate activity", "20140910_1731": "X1.6 flare"}[tag]
    add_image(
        f"results/plots/experiments/mask_comparison_{tag}.png",
        width_in=5.5,
        caption=f"{tag} ({act}) — all mask values vs BP, 10 val pixels"
    )

add_divider()


# ════════════════════════════════════════════════════════════════════════════
# 2026-06-27 — Amortized neural field (step 2, all 4 timestamps)
# ════════════════════════════════════════════════════════════════════════════

add_date_heading("2026-06-27 — Amortized Neural Field (Step 2): One Model, All 4 Timestamps")

add_para(
    "Step 1 validated that the patch-conditioned CNN architecture can reach BP-like sparse "
    "solutions on a single image. Step 2 asks: does it generalize? One PatchDEMNet is trained "
    "jointly over all 4 timestamps (~64k pixels total) using the same barrier loss. Each "
    "timestamp contributes ~80% of its pixels to training; the remaining 20% are held out as "
    "a validation set before training begins and are never seen by the model. Implemented in "
    "experiments/train_neural_field_amortized.py."
)

add_subheading("Validation results (held-out pixels, ~2800 per timestamp)")
add_para(
    "After 30 epochs, the model is evaluated on held-out val pixels per timestamp. "
    "NN sparsity and MAE are computed from inference on those pixels; BP sparsity reference "
    "is computed by solving the real LP on a 100-pixel subset of each timestamp."
)

val_table = (
    f"{'Timestamp':<22} {'Activity':<22} {'Val px':>7} {'NN sp':>7} {'BP sp':>7} {'NN MAE':>8}\n"
    f"{'-'*77}\n"
    f"{'20110906_2217':<22} {'X2.1 flare (AR 11283)':<22} {'2,829':>7} {'1.84':>7} {'1.97':>7} {'2.97':>8}\n"
    f"{'20120603_0000':<22} {'Quiet sun':<22} {'2,661':>7} {'1.58':>7} {'1.67':>7} {'2.83':>8}\n"
    f"{'20131113_0908':<22} {'Moderate activity':<22} {'2,850':>7} {'1.47':>7} {'1.82':>7} {'3.93':>8}\n"
    f"{'20140910_1731':<22} {'X1.6 flare (AR 12158)':<22} {'2,857':>7} {'1.86':>7} {'1.71':>7} {'4.71':>8}\n"
)
mono = doc.add_paragraph()
run = mono.add_run(val_table)
run.font.name = "Courier New"
run.font.size = Pt(8)

add_finding(
    "Amortization works: one model generalizes to held-out pixels across all 4 timestamps. "
    "Per-pixel DEM curves are always smooth and single-peaked at physically correct temperatures "
    "— no oscillation (the failure mode of the previous channel-input NN). "
    "Consistent systematic pattern: NN curves are slightly broader and smoother than BP's sharper "
    "sparse peaks — expected cost of sharing weights across 64k diverse pixels, as the model "
    "finds a solution that works on average rather than the exact sparse optimum per pixel. "
    "20131113_0908 shows the largest sparsity gap; 20140910_1731 (flare-active timestamp) is "
    "the hardest — BP itself produces noisier solutions there, so disagreements may reflect "
    "BP's own instability rather than NN failure. "
    "Important caveat: val pixels are from the same 4 images as training (different pixels, "
    "not different images). This is NOT a true test of generalization to new solar conditions — "
    "a leave-one-timestamp-out evaluation is the natural next step to assess true generalization."
)

add_subheading("Training loss curve (step 2, no mask)")
add_image(
    "results/plots/experiments/neural_field_amortized_4ts_train_loss.png",
    width_in=5.0,
    caption="Amortized neural field training loss — 4 timestamps, no masking, 30 epochs"
)

add_subheading("Per-pixel DEM curves: amortized neural field (cyan dotted) vs BP (black)")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    add_image(
        f"results/plots/experiments/neural_field_{tag}_pixel_dems.png",
        width_in=5.5,
        caption=f"{tag} — amortized neural field vs BP on 10 held-out val pixels"
    )

add_divider()


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
    "results/plots/experiments/neural_field_20110906_2217_pixel_dems.png",
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
add_image("results/plots/experiments/barrier_train_loss.png", width_in=5.0,
           caption="barrier NN training loss (30 epochs, pooled across 4 timestamps)")
add_image("results/plots/experiments/barrier_fit_train_loss.png", width_in=5.0,
           caption="barrier_fit NN training loss (30 epochs, pooled across 4 timestamps)")

add_subheading("Optimizer comparison + NN overlay (6 pixels, timestamp 20110906_2217)")
add_para(
    "Solid lines = barrier loss; dashed lines = barrier_fit loss; colors = optimizer "
    "(blue=L-BFGS, orange=Adam, green=SGD); dotted cyan/pink = NN predictions for "
    "barrier / barrier_fit respectively. Black = BP reference."
)
add_image("results/plots/experiments/optimizer_comparison.png", width_in=4.5,
           caption="optimizer_comparison.png — BP + 3 optimizers x 2 losses + 2 NNs, per pixel")

add_subheading("Per-pixel DEM curves with NN overlay (all 4 timestamps, 10 random pixels each)")
for tag in ["20110906_2217", "20120603_0000", "20131113_0908", "20140910_1731"]:
    add_image(f"results/plots/experiments/pixel_dems/{tag}_pixel_dems.png", width_in=4.5,
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
add_image("results/plots/dem_results/optimizer_comparison.png", width_in=4.5,
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
