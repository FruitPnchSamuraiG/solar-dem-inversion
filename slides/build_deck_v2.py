"""Advisor deck, built section by section from slides/deck_outline.md.

    uv run --with python-pptx --with lxml --with pillow --with numpy python slides/build_deck_v2.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib import (CARD, DARK, DEEP, FULL_W, GREY, fix_chart_axis_ids, IMG, L, LAV, LIGHT, MUTED, PURPLE, SLIDES, TOP,
                      WHITE, Deck, box, caption, card, line, picture, table, text)
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

deck = Deck()


# ── Helpers ──────────────────────────────────────────────────────────────────

def head_card(slide, x, y, w, h, head, body, num=None, size=11.5):
    """A card with a bold purple head line and a short body."""
    card(slide, x, y, w, h)
    label = f"{num}. {head}" if num else head
    text(slide, x + Inches(0.18), y + Inches(0.14), w - Inches(0.36), Inches(0.35), [label],
         size=13, bold=True, color=PURPLE)
    text(slide, x + Inches(0.18), y + Inches(0.52), w - Inches(0.36), h - Inches(0.6), [body],
         size=size, color=DARK)


def card_grid(slide, items, cols, top=TOP + Inches(0.1), h=Inches(1.5), gap=Inches(0.2), numbered=False,
              size=11.5):
    w = (FULL_W - gap * (cols - 1)) // cols
    for k, (head, body) in enumerate(items):
        r, c = divmod(k, cols)
        head_card(slide, L + c * (w + gap), top + r * (h + gap), w, h, head, body,
                  num=k + 1 if numbered else None, size=size)


# ── Section 1: problem and approach ─────────────────────────────────────────

# Slide 1: title only (the template's title slide).

# Summary, before Part 1.
s = deck.slide("In one slide",
               "The four things to remember. Numbers are from the final models (one MLP of width 336 per "
               "track, about 400k parameters, fresh pixels) on the shared test set: 153 days never seen in "
               "training. The BP model is the stiff-band model of Part 7.")
card_grid(s, [
    ("No labels needed", "A network trained on the solver's own objective reproduces its DEMs. Solver DEMs "
                         "are used only to evaluate."),
    ("Small is enough", "A 400k-parameter MLP on one pixel's 6 intensities: one forward pass per pixel, "
                        "no neighbourhood. Bigger models fail on rare pixels."),
    ("Ahead of supervised on most measures", "ElasticNet: better on every metric. BP: 4 of 6, "
                                                        "median pixel error 3× lower (0.036 vs 0.112); DEM MSE "
                                                        "still 1.9× higher."),
    ("Flare cores: found and fixed in the loss", "0.01% of pixels, ~90% of the BP error. The BP loss priced "
                                                 "hot plasma out; a stiffer band puts it back (emission "
                                                 "40% → 93%).")],
    cols=2, h=Inches(1.6), size=13.5)

# Slide 2: Part 1 divider.
deck.divider(1, "Problem and approach", "What we train, and on what")

# Slide 3: why label-free : two training loops.
s = deck.slide("Same physics, different training signal",
               "The audience knows the inverse problem. This slide only says what is different "
               "about our training signal. BP and ElasticNet solve one optimisation per pixel. A "
               "supervised network learns to reproduce the solver's DEMs, so it needs them as labels. "
               "Ours learns to minimise the solver's objective directly, so it never sees a solver DEM; "
               "we use those only to evaluate.")
text(s, L, TOP, FULL_W, Inches(0.4),
     ["Solvers (BP, ElasticNet) run one optimisation per pixel: accurate, but slow at full resolution."],
     size=12.5)
bw, bh, gap, lgap = Inches(1.15), Inches(0.7), Inches(0.28), Inches(0.62)
x0 = Inches(1.95)
rows = [(Inches(1.6), "Supervised", "learns the solver's DEMs",
         [("Observed AIA", "6 channels"), ("Model", ""), ("DEM", "18 bins")], ("Solver's DEM", "label")),
        (Inches(2.8), "Label-free (unsupervised)", "learns the solver's objective implicitly",
         [("Observed AIA", "6 channels"), ("Model", ""), ("DEM", "18 bins"),
          ("Predicted AIA", "via response R")], ("Observed AIA", "6 channels"))]
for y, name, sub, chain, target in rows:
    text(s, L, y + Inches(0.08), Inches(1.45), Inches(0.79),
         [[(name, {"bold": True, "color": PURPLE, "size": 12.5})], [(sub, {"size": 10, "color": MUTED})]],
         space_after=2)
    xs = [x0 + i * (bw + gap) for i in range(len(chain))]
    for x, (head, small) in zip(xs, chain):
        net = head == "Model"
        box(s, x, y, bw, bh, head, small, fill=PURPLE if net else CARD,
            head_color=WHITE if net else PURPLE, head_size=11, sub_size=9)
    for i in range(len(chain) - 1):
        line(s, xs[i] + bw, y + bh // 2, xs[i + 1], y + bh // 2)
    tx = xs[-1] + bw + lgap
    box(s, tx, y, bw, bh, target[0], target[1], fill=LIGHT, head_color=DARK, head_size=11, sub_size=9)
    line(s, xs[-1] + bw, y + bh // 2, tx, y + bh // 2, arrow=False, dashed=True)
    pill = box(s, xs[-1] + bw + Inches(0.08), y + bh // 2 - Inches(0.13), lgap - Inches(0.16), Inches(0.26),
               "loss", fill=LAV, head_size=9)
text(s, L, Inches(3.9), FULL_W, Inches(0.5),
     [[("We never train on the solver's DEMs; we use them only to evaluate.",
        {"bold": True, "color": PURPLE})]], size=12.5)

# Slide 4: the model : pipeline diagram.
s = deck.slide("The model: a small MLP feeding a fixed forward model",
               "The solvers never solve for the 18 bins directly: both solve for weights on a fixed basis, "
               "a spike and two Gaussians at each bin, and report DEM = B times x. We predict the same "
               "weights, so the network searches the same space of DEMs the solver does, and BP's "
               "sparsity criterion means the same thing for both. Only the MLP is learned. Per pixel: six "
               "intensities in, encoded as the supervised model does it (square root, then 12 Fourier "
               "frequencies: 150 inputs), four hidden layers of 336 with SiLU, 54 non-negative basis weights "
               "out; DEM = B times w over 18 bins, logT 5.5 to 7.2; predicted AIA = R times DEM. 409k "
               "parameters, one forward pass per pixel. The width is chosen in Part 5.")
heads = [("Observed AIA", "6 channels"), ("MLP", "409k params"), ("Basis weights", "54, same as solver"),
         ("DEM", "18 bins, logT 5.5–7.2"), ("Predicted AIA", "via response R")]
bw, bh, by = Inches(1.5), Inches(0.95), Inches(1.3)
bgap = (FULL_W - 5 * bw) // 4
xs = [L + i * (bw + bgap) for i in range(5)]
for x, (head, sub) in zip(xs, heads):
    box(s, x, by, bw, bh, head, sub, fill=PURPLE if head == "MLP" else CARD,
        head_color=WHITE if head == "MLP" else PURPLE, sub_color=WHITE if head == "MLP" else DARK,
        head_size=13, sub_size=10)
for i in range(4):
    line(s, xs[i] + bw, by + bh // 2, xs[i + 1], by + bh // 2)
ly = by + bh + Inches(0.5)
loss = card(s, xs[1], ly, xs[4] + bw - xs[1], Inches(0.6), fill=LAV)
tf = loss.text_frame
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Loss: the solver's objective, predicted against observed AIA"
r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = PURPLE
line(s, xs[4] + bw // 2, by + bh, xs[4] + bw // 2, ly)
line(s, xs[0] + bw // 2, by + bh, xs[0] + bw // 2, ly + Inches(0.3), arrow=False)
line(s, xs[0] + bw // 2, ly + Inches(0.3), xs[1], ly + Inches(0.3))
text(s, L, Inches(3.45), FULL_W, Inches(1.4),
     ["6 intensities in, encoded as √ then 12 Fourier frequencies (150 inputs; both tracks); 4 hidden layers "
      "of 336, SiLU; 54 non-negative basis weights out.",
      "BP and ElasticNet solve for the same 54 weights (at each of 18 bins, a spike and Gaussians "
      "0.1 and 0.2 dex wide), and their L1 term acts on those weights.",
      "DEM = B·w over 18 bins; predicted AIA = R·DEM. B and R are fixed; only the MLP is learned.",
      "409k parameters (width chosen in Part 5); one forward pass per pixel."],
     size=12, bullets=True, space_after=4)

# Slide 5: the loss.
def formula(*pieces):
    """Alternating normal text and subscript pieces, as one paragraph of runs."""
    return [(piece, {"sub": i % 2 == 1}) for i, piece in enumerate(pieces) if piece]


s = deck.slide("The loss is the solver's own objective",
               "BP track: the penalty is zero anywhere inside the noise band, o plus or minus 1.4 sigma, "
               "and quadratic outside, scaled by the band width. There is no fit term inside the band, so "
               "any DEM within noise is equally good, and the L1 term on the basis weights then picks the "
               "sparsest one: BP's own criterion. One difference from the solver: BP treats the band as a rule "
               "that cannot be broken; our loss treats it as a price, set by the band weight mu. The original "
               "loss had mu = 1; the final BP model uses mu = 100, chosen in Part 7, where that difference "
               "turns out to matter. ElasticNet track: the solver's ElasticNet objective with "
               "its settings, alpha 0.001, l1 ratio 0.5, C equal to the six channels, using the same noise "
               "scaling. In both, y-hat is R times B times w, and w is non-negative.")
cw = Inches(5.5)
for k, (head, eq, lines) in enumerate([
        ("BP track",
         formula("L = Σ", "k", " w", "k", "  +  μ Σ", "c", " [ max(0, |ŷ", "c", " − o", "c", "| − tσ", "c",
                 ") / tσ", "c", " ]²"),
         ["Zero inside the noise band (t = 1.4), quadratic outside; μ prices leaving it.",
          "L1 picks the sparsest DEM that fits: BP's criterion. Final model: μ = 100 (Part 7)."]),
        ("ElasticNet track",
         formula("L = (1/2C) Σ", "c", " [ (ŷ", "c", " − o", "c", ") / tσ", "c", " ]²  +  αλ Σ", "k",
                 " w", "k", "  +  ½α(1−λ) Σ", "k", " w", "k", "²"),
         ["The solver's settings: α = 0.001, λ = 0.5, C = 6."])]):
    y = TOP + k * Inches(1.75)
    card(s, L, y, cw, Inches(1.55))
    text(s, L + Inches(0.2), y + Inches(0.12), cw - Inches(0.4), Inches(0.3), [head],
         size=12.5, bold=True, color=PURPLE)
    text(s, L + Inches(0.2), y + Inches(0.45), cw - Inches(0.4), Inches(0.4), [eq],
         size=13, font="Cambria", color=DARK)
    text(s, L + Inches(0.2), y + Inches(0.88), cw - Inches(0.4), Inches(0.6), lines,
         size=10.5, bullets=True, space_after=2)
text(s, L, Inches(4.55), cw, Inches(0.3), ["In both: ŷ = R·B·w, with w ≥ 0."], size=11, color=MUTED)
picture(s, os.path.join(IMG, "band_penalty.png"), Inches(6.15), TOP + Inches(0.2), Inches(3.45), Inches(2.6))
caption(s, Inches(6.15), Inches(3.75), Inches(3.45), "The BP band penalty for one channel.")

# Slide 6: data : the split as a timeline, to scale.
s = deck.slide("Data: two years of full-disk AIA, split in time",
               "Every run before October seeded each block's pixel draw by the block alone, so it reused the "
               "same 30M pixels every epoch. All results from here on use fresh pixels every epoch, as "
               "described. The fixed-sample runs scored better: validation loss BP 2.156 against 2.237, "
               "ElasticNet 0.0582 against 0.0689, far beyond the 0.2 to 0.4% run-to-run spread; only "
               "ElasticNet's flare cores improved with fresh pixels. "
               "The split is chronological, so every test image is later than anything seen in training. "
               "Each epoch draws 512 pixels from each of 58,688 training blocks: 917 images times 64 "
               "blocks of 256 squared. The test set uses 64 random 128-squared blocks of each test image "
               "per solver target, five targets per image: one clean solve and four re-solves under "
               "simulated photon noise of the same observation. Pixels where deconvolution clamped a "
               "bright channel to zero, about 5%, are excluded.")
spans = [("Train", "917 images", 17.5, PURPLE, WHITE), ("Validation", "153", 3.2, LAV, PURPLE),
         ("Test", "153", 3.3, DEEP, WHITE)]
gap, ty, th = Inches(0.05), TOP + Inches(0.1), Inches(0.62)
total = sum(m for _, _, m, _, _ in spans)
x = L
for name, count, months, fill, color in spans:
    w = int((FULL_W - 2 * gap) * months / total)
    box(s, x, ty, w, th, name, count, fill=fill, head_color=color, sub_color=color, head_size=11.5, sub_size=9.5)
    x += w + gap
for label, frac in (("Jan 2014", 0), ("Jun 2015", 17.5 / total), ("Sep 2015", 20.7 / total), ("Dec 2015", 1)):
    lx = min(max(L + int(FULL_W * frac) - Inches(0.45), L), L + FULL_W - Inches(0.9))
    text(s, lx, ty + th + Inches(0.05), Inches(0.9), Inches(0.25), [label], size=9.5, color=MUTED,
         align=PP_ALIGN.LEFT if frac == 0 else (PP_ALIGN.RIGHT if frac == 1 else PP_ALIGN.CENTER))
text(s, L, Inches(2.2), FULL_W, Inches(2.6),
     ["SDO/AIA, 6 EUV channels, about two images a day in 2014–2015: 1,223 full-disk images.",
      "PSF-deconvolved (Hofmeister); a noise σ per pixel and channel from the AIA error model; "
      "DEMs on a 2048² grid.",
      "Our model trains on AIA and σ only: each training image is cut into 64 blocks of 256×256 per image. "
      "Each epoch draws 512 fresh random pixels from each of the 58,688 blocks (917 × 64); 40 epochs "
      "(the final BP model: 240, Part 7).",
      "Earlier runs reused one fixed ~30M-pixel sample (a sampler bug); at the old 176k size it scored better "
      "on validation (BP 4%, ElasticNet 18%).",
      "Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.",
      "Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves."],
     size=12.5, bullets=True, space_after=7)

# ── Section 2: finding a trainable objective ────────────────────────────────

# Slide 7: Part 2 divider.
deck.divider(2, "Finding a trainable objective", "Which loss gives BP's DEMs?")

# Slide 8: which loss.
s = deck.slide("Six channels underdetermine the DEM: the regulariser picks it",
               "Before any network, five losses optimised directly per pixel on 4 timestamps. Averages: "
               "AIA MAE against the observation, BP 5.2, barrier 5.0, barrier plus fit 2.6, fit-only "
               "0.40 to 0.44. DEM MAE against BP: barrier 0.034, barrier plus fit 0.043, fit-only 0.18 to "
               "0.20. W1 on the brightest 5% of pixels: 0.030 and 0.033 against 0.15 to 0.18. Six "
               "equations, eighteen unknowns: many DEMs fit within noise, so whatever breaks the tie "
               "decides the shape. L-BFGS and Adam reach the same curves; SGD often fails to converge.")
text(s, L, TOP, FULL_W, Inches(1.5),
     ["Before any network: five losses optimised directly, pixel by pixel, on 4 timestamps, each compared with BP.",
      "Fit-only losses (χ² + smoothness, max entropy, Tikhonov) reconstruct AIA ~12x better than BP, "
      "yet land ~5x further from its DEM.",
      "BP's own terms (noise band + L1) land on BP's DEM: MAE 0.03 vs 0.18–0.20."],
     size=12.5, bullets=True, space_after=5)
picture(s, os.path.join(IMG, "loss_pixels.png"), L, Inches(2.12), FULL_W, Inches(2.0))
caption(s, L, Inches(4.15), FULL_W, "Two example pixels; BP in black. Legend MAE is AIA reconstruction error.")
text(s, L, Inches(4.5), FULL_W, Inches(0.3),
     [[("So the network has to train on the solver's objective, not on fit.", {"bold": True, "color": PURPLE})]],
     size=12.5)


# ── Section 3: architecture on small data ───────────────────────────────────

# Slide 9: Part 3 divider.
deck.divider(3, "Architecture on small data", "Which network, on four images")

# Slide 10: first network, then the patch CNN.
s = deck.slide("A per-pixel MLP missed BP's shape; a patch CNN tracked it",
               "The June write-up called the first MLP's curves noisy and oscillating, but no figure of them "
               "survives and its checkpoint was not kept. Retrained with the unchanged June code, its curves are "
               "smooth: none has three or more peaks. The real difference is accuracy, on 400 held-out pixels "
               "per image: mean DEM error against BP 0.078 against the CNN's 0.047, W1 0.055 against 0.048 dex. "
               "It had even seen these pixels in training. It is the same architecture as today's production "
               "model, at width 256 with raw intensities.")
text(s, L, TOP - Inches(0.05), FULL_W, Inches(0.3),
     ["4 images: X2.1 flare (2011), quiet Sun (2012), moderate activity (2013), X1.6 flare (2014); "
      "small crops near disk centre."], size=10.5, color=MUTED)
text(s, L, TOP + Inches(0.4), Inches(5.4), Inches(3.3),
     [[("First try: ", {"bold": True, "color": PURPLE}),
       ("the 6 intensities of one pixel → MLP (213k params) → 54 weights, trained on the BP loss over "
        "~221k pixels. Smooth, single-peaked curves, but further from BP: DEM error 0.078 vs the CNN's 0.047.", {})],
      [("Patch CNN: ", {"bold": True, "color": PURPLE}),
       ("a 9×9 neighbourhood → CNN (~1.5M params) → 54 weights. Tracked BP's shape, with sparsity close "
        "to BP's on held-out pixels.", {})],
      "Trained on one image, then on all 4 jointly; zeroing the neighbourhood for 10% of batches "
      "kept it usable on a single pixel."],
     size=12.5, bullets=True, space_after=10)
picture(s, os.path.join(IMG, "first_mlp_vs_cnn.png"), Inches(6.0), TOP + Inches(0.3), Inches(3.6), Inches(3.25))
caption(s, Inches(6.0), Inches(4.6), Inches(3.6), "Held-out pixels, X2.1 flare image.")

# Slide 11: ablation : a dot plot of sparsity against BP.
s = deck.slide("Ablation: the gap was capacity, not missing context",
               "The shuffled-patch CNN applies a fixed random permutation to the 81 patch pixels. It was "
               "less sparse than the CNN on all 4 images, 1.88 against 1.70 on average, level with the "
               "centre-pixel MLP, so the spatial arrangement of the neighbours mattered. The "
               "flat-patch MLP was erratic across images: 1.61 on the X2.1 flare, 2.27 on quiet Sun. "
               "The centre-pixel MLP was closest to BP on 3 of 4 images; the CNN on the X1.6 flare. So the "
               "CNN's lower sparsity did not mean DEMs closer to BP. Capacity and input scaling were not "
               "separated: the 213k MLP took raw intensities, and at full scale MLPs on log1p inputs "
               "does well.")
lo, hi = 1.6, 2.0
px0, px1 = Inches(3.0), Inches(9.2)
X = lambda v: int(px0 + (px1 - px0) * (v - lo) / (hi - lo))
rows = [("Patch CNN", 1.70), ("Shuffled-patch CNN", 1.88), ("Centre-pixel MLP", 1.89), ("Flat-patch MLP", 1.95)]
y0, dy, dot = TOP + Inches(0.35), Inches(0.42), Inches(0.17)
axis_y = y0 + dy * len(rows)
for k, (name, v) in enumerate(rows):
    cy = y0 + dy * k + dy // 2
    text(s, L, cy - Inches(0.13), Inches(2.5), Inches(0.28), [name], size=12, bold=k == 0,
         color=PURPLE if k == 0 else DARK)
    line(s, px0, cy, px1, cy, color=LIGHT, width=0.75, arrow=False)
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, X(v) - dot // 2, cy - dot // 2, dot, dot)
    d.fill.solid(); d.fill.fore_color.rgb = PURPLE; d.line.fill.background(); d.shadow.inherit = False
    text(s, X(v) + Inches(0.13), cy - Inches(0.13), Inches(0.6), Inches(0.26), [f"{v:.2f}"], size=11,
         color=PURPLE, bold=True)
line(s, px0, axis_y, px1, axis_y, color=GREY, width=1, arrow=False)
for t in (1.6, 1.7, 1.8, 1.9, 2.0):
    text(s, X(t) - Inches(0.3), axis_y + Inches(0.04), Inches(0.6), Inches(0.25), [f"{t:.1f}"], size=10,
         color=MUTED, align=PP_ALIGN.CENTER)
line(s, X(1.79), y0 - Inches(0.1), X(1.79), axis_y, color=DARK, width=1.25, arrow=False, dashed=True)
text(s, X(1.79) - Inches(0.5), y0 - Inches(0.38), Inches(1.0), Inches(0.26), ["BP 1.79"], size=11, bold=True,
     align=PP_ALIGN.CENTER)
caption(s, px0, axis_y + Inches(0.28), px1 - px0,
        "Effective number of active basis weights, mean over 4 images (lower = sparser).")
text(s, L, Inches(3.55), FULL_W, Inches(1.2),
     ["Four variants at ~1.45M params, same BP loss, same 4 images.",
      "Size, not neighbours, closed the gap: DEM error vs BP 0.042 for the 1.43M centre-pixel MLP, "
      "0.047 for the patch CNN, 0.078 for the first MLP.",
      "The patch CNN was sparsest, but measured on held-out pixels of the training images."],
     size=12.5, bullets=True, space_after=5)

# Slide 12: leave one image out.
s = deck.slide("On unseen days, the patch CNN's edge disappeared",
               "Closer to BP's sparsity: the CNN on 2 images, the MLP on 1, one tie; the MLP is closer on "
               "average, 0.19 against 0.25, because of the X1.6 flare image. "
               "Held-out sparsity by fold, MLP against CNN, with BP in brackets: X2.1 flare 1.59 vs 1.64 "
               "(1.97); quiet Sun 1.78 vs 1.61 (1.67); moderate 1.83 vs 1.83 (1.82); X1.6 flare 1.96 vs "
               "2.31 (1.71). The ablation's in-sample edge, 1.70 vs 1.89, came from scoring pixels of "
               "images the model trained on. Four folds is a small sample; scaling is the real test.")
text(s, L, TOP, Inches(5.4), Inches(3.7),
     ["Leave one image out: train on 3, test on the 4th; patch CNN and centre-pixel MLP, 4 folds.",
      "Neither is consistently closer to BP's sparsity on the unseen image; the MLP fits AIA better on 3 of 4.",
      "Little overfitting (held-out within ±0.2 of in-sample on 3 folds), except the X1.6 flare: peaks too "
      "cool (logT ~6.1 vs BP ~6.4), with one other flare image in training.",
      [("Carry the simpler centre-pixel MLP forward, and get more flare data → scaling.",
        {"bold": True, "color": PURPLE})]],
     size=12.5, bullets=True, space_after=9)
picture(s, os.path.join(IMG, "loo_flare.png"), Inches(6.0), TOP - Inches(0.1), Inches(3.6), Inches(3.6))
caption(s, Inches(6.0), Inches(4.5), Inches(3.6), "Held-out X1.6 flare image: CNN and MLP (dashed) vs BP.")

# ── Shared bits for Parts 4-8 ───────────────────────────────────────────────

P19 = os.path.join(os.path.dirname(SLIDES), "results/plots/19_final_models_20261004")


def muted(slide, y, words, h=Inches(0.3)):
    text(slide, L, y, FULL_W, h, [words], size=10, color=MUTED, italic=True)


def stat(slide, x, y, w, h, big, small):
    c = card(slide, x, y, w, h)
    text(slide, x + Inches(0.15), y + Inches(0.12), w - Inches(0.3), Inches(0.6), [big],
         size=26, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
    text(slide, x + Inches(0.15), y + Inches(0.78), w - Inches(0.3), h - Inches(0.85), [small],
         size=10.5, color=DARK, align=PP_ALIGN.CENTER)


def bold_cells(shape, cells):
    for i, j in cells:
        for r in shape.table.cell(i, j).text_frame.paragraphs[0].runs:
            r.font.bold = True


# ── Section 4: scaling to the full dataset ──────────────────────────────────

deck.divider(4, "Scaling to the full dataset", "What broke at full scale, and how we score")

# Slide 14: the collapse.
s = deck.slide("The first full-disk runs collapsed to an all-zero DEM",
               "Three submissions died the same way before the fix; the cause was visible at step 0, before "
               "any weight update. Gradient clipping does not help, because Adam normalises the step size.")
text(s, L, TOP, Inches(5.7), Inches(3.7),
     ["Full-disk intensities span ~7 decades (10⁻³ off-limb to 10⁴ in flare cores); the small crops "
      "spanned about one.",
      "Fed raw, the untrained network predicted AIA ~7× outside the noise band: loss 5×10¹³ at step 0. "
      "The output softplus underflowed, its gradient became exactly zero, and every pixel predicted zero.",
      "Fix: a bounded input (log1p then, √ + Fourier now; loss and metrics stay in physical units), "
      "learning-rate warmup, a clamped softplus whose gradient cannot vanish, and a guard that stops a run "
      "whose loss freezes.",
      [("A 64-block smoke test could not catch it: it never drew a flare pixel.",
        {"bold": True, "color": PURPLE})]],
     size=12.5, bullets=True, space_after=9)
bx, bw2, bh2 = Inches(6.6), Inches(2.9), Inches(0.8)
for k, (head, sub) in enumerate([("Raw intensity", "10⁻³ … 10⁴"), ("log1p", "input only"),
                                  ("Network input", "0 … 9.2")]):
    y = TOP + Inches(0.2) + k * Inches(1.2)
    box(s, bx, y, bw2, bh2, head, sub, fill=PURPLE if k == 1 else CARD,
        head_color=WHITE if k == 1 else PURPLE, sub_color=WHITE if k == 1 else DARK,
        head_size=13, sub_size=11)
    if k:
        line(s, bx + bw2 // 2, y - Inches(0.4), bx + bw2 // 2, y)

# Slide 15: how we evaluate.
s = deck.slide("How we score at scale",
               "The Bright cutoffs and the test set come from the supervised side's protocol, so both models "
               "are scored on identical pixels. W1 treats each DEM as a distribution over logT.")
text(s, L, TOP, Inches(5.6), Inches(3.8),
     ["153 test days, never seen in training; each scored on random quarters of the image against 5 solver "
      "targets (the clean solve and 4 noisy re-solves).",
      "Bright = any channel above its top-5% cutoff (~10% of pixels); the rest is Quiet.",
      "DEM against the solver: MSE, total-emission error, W1 (temperature-shape distance). "
      "AIA reconstruction against the observation: MAE, MSE.",
      "We also report medians and percentiles: the error is extremely heavy-tailed. For our final BP model, "
      "the worst 1% of pixels hold 98% of the DEM error, and the flare cores (0.01% of pixels, 1 in 10,000) "
      "alone hold 94%."],
     size=12, bullets=True, space_after=9)
gx, gy, cell = Inches(6.35), TOP + Inches(0.05), Inches(0.17)
picked = set(np.random.default_rng(3).choice(256, 64, replace=False).tolist())
for k in range(256):
    r, c = divmod(k, 16)
    sq = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, gx + c * cell, gy + r * cell, cell, cell)
    sq.fill.solid(); sq.fill.fore_color.rgb = PURPLE if k in picked else CARD
    sq.line.color.rgb = WHITE; sq.line.width = Pt(0.5); sq.shadow.inherit = False
text(s, gx, gy + 16 * cell + Inches(0.08), 16 * cell, Inches(0.9),
     [[("One test day: ", {"bold": True, "color": PURPLE}),
       ("2048² DEM image = 256 blocks of 128². One target is scored on its own random 64 (purple). "
        "× 5 targets: the clean solve + 4 noisy re-solves.", {})]],
     size=9.5, color=DARK)

# ── Section 5: experiments at scale ─────────────────────────────────────────

deck.divider(5, "Experiments at scale", "CNN or MLP, and how big")

# Slide 17: CNN vs MLP at scale (fresh pixels, test split, eval job 19108218).
s = deck.slide("At scale, the centre-pixel MLP beat the patch CNN",
               "Both ~1.5M parameters, fresh pixels, same 917 training days; scored on the 153 test days. "
               "The MLP is better on the training objective at every percentile and in every brightness "
               "decile on BP (median 0.98 vs 1.01, p99 17.9 vs 18.8, p99.9 51 vs 57). The CNN reconstructs "
               "BP's AIA slightly better. This is the fifth comparison; none found a CNN advantage on unseen days.")
text(s, L, TOP, FULL_W, Inches(0.7),
     ["Same 917 training days and losses, matched at ~1.5M parameters each, fresh pixels; scored on the 153 "
      "test days. (The final model is smaller: next slide.)"],
     size=12.5, bullets=True)
CNN_VS_MLP = [["", "BP: MLP", "BP: CNN", "ENet: MLP", "ENet: CNN"],
              ["Training objective (test)", "2.07", "2.24", "0.057", "0.245"],
              ["BP objective: median / p99", "0.98 / 17.9", "1.01 / 18.8", "", ""],
              ["Sparsity (BP 1.79)", "1.90", "2.03", "5.53", "5.17"],
              ["AIA MAE", "4.93", "4.73", "0.54", "0.82"]]
tb = table(s, L, TOP + Inches(0.55), FULL_W, CNN_VS_MLP, col_w=[2.6, 1.3, 1.3, 1.2, 1.2], size=11.5,
           row_h=Inches(0.36))
bold_cells(tb, [(1, 1), (1, 3), (2, 1), (3, 1), (4, 2), (4, 3)])
text(s, L, Inches(3.55), FULL_W, Inches(0.4),
     [[("The simplest model scales: no neighbourhood needed.", {"bold": True, "color": PURPLE})]], size=13)

# Slide 19: width sweep, framed as the model choice.
SWEEP = {"params": ["20k", "42k", "87k", "176k", "360k", "722k", "1.43M"],
         "median": [0.068, 0.047, 0.043, 0.030, 0.026, 0.025, 0.023]}
CHOSEN = SWEEP["params"].index("360k")
s = deck.slide("Choosing the model: width 336, for both tracks",
               "Fresh-pixel sweep with the log1p input, every width scored on the full test set "
               "(results/plots/17_sweep_resample). "
               "We choose by validation loss, never by test numbers. BP validation: 2.148 at width 336 (360k, best), "
               "2.155 at 1.43M (a tie within run-to-run noise), 2.167 at 722k, 2.237 at 176k. ElasticNet "
               "validation is best at 1.43M (0.0528), but its 722k and 1.43M models predict a few bright pixels "
               "orders of magnitude too bright (AIA MSE 41,849 and 908,611 against 650 at 360k); validation "
               "scores a fixed 512 pixels per block and never saw them. 10k collapsed and is left off the chart. "
               "Multi-peak recall is 22-26% from 42k up; the fixed-sample cliff below 176k does not reproduce. "
               "With the old fixed sample nothing above 176k helped; with fresh pixels size pays off up to "
               "width 336. Input encoding: on the original BP loss, square root plus Fourier features tied "
               "log1p on validation (2.1472 against 2.1478); ElasticNet 0.0533 against 0.0575. On the stiff "
               "BP band of Part 7 the tie breaks: square root plus Fourier beats log1p (DEM MSE 2.37 against "
               "3.23), so both tracks use it, and the same width is 409k parameters with the 150 encoded "
               "inputs. Width 680 (1.43M) on the stiff band blew up on a few flare-core pixels, as the 10M "
               "models did on both tracks.")
cd = CategoryChartData()
cd.categories = SWEEP["params"]
cd.add_series("BP median pixel error", SWEEP["median"])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, L, TOP, Inches(5.3), Inches(3.3), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "BP median pixel error by model size, log1p sweep (lower is better)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11.5)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ser = ch.plots[0].series[0]
ser.format.line.color.rgb = PURPLE
ser.format.line.width = Pt(2)
ser.marker.style = XL_MARKER_STYLE.CIRCLE
ser.marker.size = 6
ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = PURPLE
ser.marker.format.line.color.rgb = PURPLE
pt = ser.points[CHOSEN]
pt.marker.style = XL_MARKER_STYLE.CIRCLE
pt.marker.size = 13
pt.marker.format.fill.solid(); pt.marker.format.fill.fore_color.rgb = PURPLE
pt.marker.format.line.color.rgb = PURPLE
pl = ch.plots[0]
pl.has_data_labels = True
pl.data_labels.number_format = '0.000'
pl.data_labels.number_format_is_linked = False
pl.data_labels.font.size = Pt(9)
pl.data_labels.position = XL_LABEL_POSITION.ABOVE
va = ch.value_axis
va.maximum_scale = 0.08; va.minimum_scale = 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = LIGHT
va.tick_labels.font.size = Pt(9)
ch.category_axis.tick_labels.font.size = Pt(9)
fix_chart_axis_ids(ch)
text(s, Inches(5.95), TOP + Inches(0.05), Inches(3.65), Inches(3.5),
     ["We choose by validation loss.",
      "BP: validation is best at width 336 (360k); 1.43M ties, but only nudges the typical pixel "
      "(0.026 → 0.023) at 4× the size, and on the stiff band (Part 7) it blows up on a few flare-core pixels.",
      "ElasticNet: from 722k up, a few bright pixels blow up by orders of magnitude; 336 is the largest "
      "stable width.",
      "Input, also by validation: √ + Fourier for ElasticNet (7% better) and, on the stiff band, for BP "
      "(DEM MSE 2.37 vs 3.23)."],
     size=11.5, bullets=True, space_after=7)
text(s, L, Inches(4.4), FULL_W, Inches(0.35),
     [[("Final: one MLP of width 336 (409k parameters) per track, √ + Fourier input; BP on the stiff band (Part 7).",
        {"bold": True, "color": PURPLE})]],
     size=13)

# ── Section 6: results ───────────────────────────────────────────────────────

deck.divider(6, "Results", "Against the solver and the supervised model")

# Slide 20: against the supervised model (final models; BP = the stiff-band model of Part 7).
s = deck.slide("Against supervised: all six on ElasticNet, four of six on BP",
               "Label-free trains on the observation through R; supervised trains on solver DEMs. The BP "
               "numbers are the stiff-band model (band weight 100, 120 + 120 epochs, Part 7); the old BP "
               "final had DEM MSE 4.28 and EM error 19.6%. Supervised keeps DEM MSE on BP (0.91 against "
               "1.75) and AIA MAE; we win DEM MAE, EM error, W1 and AIA MSE. On ElasticNet, with the "
               "square-root plus Fourier input, we win all six, DEM MAE and MSE included (MAE 0.151 vs "
               "0.188, MSE 0.310 vs 0.318). EM error is the total-emission error; W1 is in dex.")
RES = [["", "DEM MAE", "DEM MSE", "EM error", "W1", "AIA MAE", "AIA MSE"],
       ["BP, label-free", "0.074", "1.75", "11.0%", "0.068", "3.48", "103"],
       ["BP, supervised", "0.100", "0.91", "14.3%", "0.133", "2.95", "157"],
       ["ENet, label-free", "0.151", "0.31", "12.2%", "0.103", "0.56", "161"],
       ["ENet, supervised", "0.188", "0.32", "18.0%", "0.121", "6.32", "598"]]
tb = table(s, L, TOP + Inches(0.1), FULL_W, RES, col_w=[2.4, 1.15, 1.15, 1.15, 1.05, 1.15, 1.15], size=12,
           row_h=Inches(0.42), bold_first_col=True)
bold_cells(tb, [(1, 1), (2, 2), (1, 3), (1, 4), (2, 5), (1, 6)] + [(3, c) for c in range(1, 7)])
text(s, L, Inches(3.4), FULL_W, Inches(0.9),
     ["Bold = better. ElasticNet: we win all six, DEM MAE and MSE included. BP: we win DEM MAE, EM error, "
      "W1 and AIA MSE; supervised keeps DEM MSE (set by flare cores, Part 7) and AIA MAE."],
     size=12.5, bullets=True)
muted(s, Inches(4.3), "BP, label-free, is the stiff-band model of Part 7. The old BP final: DEM MSE 4.28, "
      "EM error 19.6%, W1 0.083, AIA MSE 83.")

# Slide 21: mean vs median.
s = deck.slide("The typical pixel and the average tell different stories",
               "Per-pixel error is the squared error summed over the 18 bins. BP percentiles, label-free against "
               "supervised: median 0.036 vs 0.112, p90 1.9 vs 1.9, p99 25 vs 27, p99.9 170 vs 162. The old BP "
               "final: median 0.026, p90 3.5, p99 71. ElasticNet: median 0.85 vs 1.12, p99 30 vs 35, p99.9 "
               "102 vs 178.")
stat(s, L, TOP + Inches(0.1), Inches(2.9), Inches(1.45), "0.036 vs 0.112",
     "BP median pixel error, ours vs supervised: ~3× better on the typical pixel")
stat(s, L, TOP + Inches(1.75), Inches(2.9), Inches(1.45), "98%",
     "of our BP error sits in the worst 1% of pixels (supervised 95%)")
text(s, Inches(3.65), TOP + Inches(0.1), Inches(5.95), Inches(3.4),
     ["On the typical BP pixel we are ~3× closer to the solver than the supervised model.",
      "Up to the 99th percentile we are level or ahead on BP (p90 1.9 vs 1.9, p99 25 vs 27); the gap opens "
      "in the last 0.1%: flare cores.",
      "ElasticNet: ours is better at the median, p99 and p99.9.",
      "The BP average is set by the worst 1% of pixels, for both models."],
     size=13, bullets=True, space_after=12)

# ── Section 7: flare cores, the failure and the fix ─────────────────────────

P22 = os.path.join(os.path.dirname(SLIDES), "results/plots/22_adopted_bp_20261014")

deck.divider(7, "Flare cores: the failure and the fix", "Why BP missed them, and what fixed it")

# Slide 23: the failure, in one slide (old BP model).
s = deck.slide("Flare cores, 1 pixel in 10,000, held 90% of the BP error",
               "The old BP final: log1p input, the original loss, 40 epochs. Pixels at 32 times the Bright "
               "cutoff or more are 0.01% of pixels and held 90% of its DEM error (and 90% of the supervised "
               "model's). At those pixels it held 40% of the solver's emission, peaked 23% as high and 0.18 "
               "dex too cool, and 83% of its curves had spurious extra peaks (solver 18%), 42% of its error. "
               "Yet it reproduced 171, 193 and 211 A to within a few percent there; only 94 A (23%) and "
               "131 A (76%) were missed. Solver noise (0.4% of the Bright error) and missed second peaks "
               "(1.2%) were ruled out. So we read it as a training problem, and the last meeting suggested "
               "training-side fixes: oversample flare cores, train longer, a 10M model.")
text(s, L, TOP, Inches(3.9), Inches(3.8),
     ["Old BP model: 0.01% of pixels, 90% of the DEM error (supervised: 90% too).",
      "There it held 40% of the solver's emission, peaked 23% as high, 0.2 dex too cool; 83% of its "
      "curves had extra peaks (solver 18%).",
      "It fit 171, 193 and 211 Å to a few percent, but reproduced only 23% of 94 Å: the hot plasma was "
      "missing.",
      [("Read as a training problem: oversample, train longer, go bigger.", {"bold": True, "color": PURPLE})]],
     size=12, bullets=True, space_after=8)
picture(s, os.path.join(P19, "fig4_example_curves_bp.png"), Inches(4.3), TOP - Inches(0.05),
        Inches(5.3), Inches(3.6))
caption(s, Inches(4.3), Inches(4.45), Inches(5.3),
        "Old BP model at random flare cores (top rows). Black solver, blue ours, orange supervised.")

# Slide 24: the suggestions on the original loss.
s = deck.slide("On the original loss, none of the suggestions helped",
               "All on the square root plus Fourier input, so the base is the sqrt plus Fourier BP model "
               "(DEM MSE 4.56, slightly worse than the log1p final's 4.28). Oversampling draws pixels in "
               "proportion to brightness, 8.5 times more flare cores. 120 epochs is three times the "
               "schedule. The 10M model is 1,800 wide. Away from flare cores the four score the same "
               "(DEM MAE 0.105 to 0.106, Quiet MSE 0.057 to 0.058); the differences sit at flare cores, "
               "and go the wrong way for oversampling and the 10M model. The solver reproduces 92% of 94 A "
               "at flare cores.")
SUGG = [["BP run, original loss", "DEM MSE", "Median", "AIA MSE", "Flare-core emission", "Flare-core 94 Å"],
        ["Base (√ + Fourier, 40 epochs)", "4.56", "0.023", "83", "35%", "15%"],
        ["Oversample flare cores (8.5× more)", "5.05", "0.025", "87", "24%", "9%"],
        ["120 epochs", "4.21", "0.022", "108", "37%", "16%"],
        ["10M parameters", "5.10", "0.022", "125", "20%", "8%"]]
tb = table(s, L, TOP + Inches(0.05), FULL_W, SUGG, col_w=[3.0, 1.1, 1.0, 1.1, 1.6, 1.5], size=11.5,
           row_h=Inches(0.4), bold_first_col=True)
text(s, L, Inches(3.35), FULL_W, Inches(1.4),
     ["Away from flare cores all four score the same (DEM MAE 0.105 to 0.106, Quiet MSE 0.057 to 0.058).",
      "At flare cores, oversampling and the 10M model make it worse; 120 epochs moves emission from 35% to "
      "37% (solver 94 Å: 92%).",
      [("Training harder does not help. The loss itself needs checking.", {"bold": True, "color": PURPLE})]],
     size=12, bullets=True, space_after=5)

# Slide 25: the cause.
s = deck.slide("The cause: the network already sits at its loss's best answer",
               "Forget the network: for one pixel, which 54 weights give the lowest possible loss? The loss "
               "is convex in the weights, so an optimizer finds that bottom; no network can beat it on that "
               "pixel. On 12,000 validation pixels (6,000 flare cores) we found each pixel's best answer, "
               "then scored four answers under the same loss: the network's output, the best answer, the "
               "solver's DEM and the supervised model's DEM (both rewritten as 54 weights). The network sits "
               "1 to 4% above its best answer at most brightness levels, 8% at flare cores, with the same DEM "
               "error as the best answer. And that best answer reproduces only 26% of 94 A. Look at the two "
               "terms: the solver keeps every channel in its band (penalty 6) but pays 1,119 in sparsity for "
               "the hot plasma; the best answer drops it, 220 plus a penalty of 243. Our loss rates the "
               "solver's DEM 2.4 times worse than a DEM without the hot plasma, so every training step "
               "moves away from the solver there. The supervised model scores worst of all under our loss "
               "yet is closest to the solver. This is the meeting's fifth point in numbers: the training "
               "objective and the evaluation disagree. It also explains the 15% emission shortfall at "
               "ordinary pixels: the best answer holds 82 to 86% of the solver's emission there.")
text(s, L, TOP - Inches(0.05), FULL_W, Inches(0.75),
     ["Per pixel, solve the loss directly (it is convex): the best answer no network can beat. 12,000 "
      "validation pixels, 6,000 of them flare cores. Our network sits 1 to 4% above it (8% at flare cores)."],
     size=12, bullets=True)
GAP = [["Flare cores, validation, original loss", "Loss", "Sparsity + band penalty", "94 Å reproduced",
        "Emission vs solver"],
       ["Old BP network", "565", "294 + 270", "28%", "38%"],
       ["Best answer under the loss", "463", "220 + 243", "26%", "30%"],
       ["Solver's DEM", "1,125", "1,119 + 6", "94%", "100%"],
       ["Supervised model's DEM", "1,878", "1,107 + 771", "91%", "95%"]]
tb = table(s, L, TOP + Inches(0.8), FULL_W, GAP, col_w=[3.0, 1.0, 2.0, 1.5, 1.5], size=11.5,
           row_h=Inches(0.38), bold_first_col=True, highlight=(2,))
text(s, L, Inches(3.85), FULL_W, Inches(1.0),
     ["A perfect network would land on the best answer, which still drops the hot plasma: 26% of 94 Å.",
      "Our loss rates the solver's DEM 2.4× worse than one without the hot plasma: every training step "
      "moves away from the solver there.",
      [("The loss, not training, was the limit.", {"bold": True, "color": PURPLE})]],
     size=12, bullets=True, space_after=4)

# Slide 26: why the loss drops hot plasma.
s = deck.slide("Why: hot plasma is expensive to include and cheap to leave out",
               "The response matrix says how bright each channel gets per unit of plasma (10^26 cm^-5 of "
               "emission measure; the 54 weights, the DEM and the sparsity term count in this unit). A unit "
               "of warm plasma (1 to 1.3 MK) lights 171, 193 and 211 A by tens to a hundred DN/s, so it has "
               "to be in the DEM anyway and its 94 A signal comes free. A unit of hot plasma (8 MK) lights no "
               "channel by more than 0.4 DN/s; only 94 and 131 A ask for it, so a bright 94 A reading needs "
               "hundreds of units. Worked example at a typical flare core: 94 A reads 164 DN/s with noise "
               "11.5, band 148 to 180, half width 16.1. Put the hot plasma in: 164 / 0.27 = 616 units, "
               "sparsity 616, penalty 0. Leave it out: 94 A predicts about 0, 148 below the band, penalty "
               "(148 / 16.1)^2 = 85. At mu = 1 leaving it out wins, 85 against 616. At mu = 100 the penalty "
               "is 8,500 and the plasma goes back in. The band is sound: the noise follows photon counting. "
               "Brighter pixels draw more of their 94 A from hot plasma, so the share dropped grows with "
               "brightness: the best answer reproduces 91% of 94 A at 1 to 1.8x the Bright level, 44% at "
               "10 to 32x, 26% at flare cores; the solver 100%, 88%, 94%. Raising mu reverses it: the best "
               "answer reproduces 81% of 94 A at flare cores with mu = 10 and 93% with mu = 100.")
text(s, L, TOP - Inches(0.05), FULL_W, Inches(0.7),
     ["Solver: the noise band is a rule. Our loss: the band is a price. Loss = sparsity (sum of the 54 "
      "weights) + μ × band penalty (distance outside the band, squared), with μ = 1 originally."],
     size=12, bullets=True)
RESP = [["DN/s per unit of plasma", "94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "335 Å"],
        ["Warm, 1 to 1.3 MK (logT 6.0–6.1)", "0.2", "0.5–1.2", "36–95", "29–55", "5–7", "0.4"],
        ["Hot, 8 MK (logT 6.9)", "0.27", "0.40", "0.05", "0.34", "0.08", "0.05"]]
tb = table(s, L, TOP + Inches(0.7), Inches(5.6), RESP, col_w=[2.5, 0.65, 0.85, 0.85, 0.85, 0.7, 0.65], size=10.5,
           row_h=Inches(0.36), bold_first_col=True)
EX = [["A flare core: 94 Å = 164 ± 11.5", "Sparsity", "Penalty (μ = 1)", "Total"],
      ["Hot plasma in: 164 ÷ 0.27 ≈ 616 units", "616", "0", "616"],
      ["Hot plasma out: 94 Å ≈ 0, 148 below the band", "0", "(148 ÷ 16.1)² = 85", "85"]]
tb = table(s, L, TOP + Inches(1.95), Inches(5.6), EX, col_w=[2.9, 0.8, 1.3, 0.7], size=10.5,
           row_h=Inches(0.36), bold_first_col=True, highlight=(2,))
text(s, L, Inches(4.12), Inches(5.6), Inches(0.7),
     ["Brighter pixels draw more of their 94 Å from hot plasma: the best answer reproduces 91% of 94 Å at "
      "1 to 1.8× the Bright level, 44% at 10 to 32×, 26% at flare cores (solver 100, 88, 94%)."],
     size=10, color=MUTED)
text(s, Inches(6.3), TOP + Inches(0.7), Inches(3.3), Inches(3.3),
     ["Warm plasma lights 171, 193, 211 Å by tens of DN/s: it has to be in the DEM, and its 94 Å comes free.",
      "Hot plasma lights no channel above 0.4 DN/s: a bright 94 Å reading needs hundreds of units, all paid "
      "for by the sparsity term.",
      "Leaving it out costs 85; putting it in costs 616. The best answer leaves it out.",
      [("At μ = 100 the penalty is 8,500: the plasma goes back in.", {"bold": True, "color": PURPLE})]],
     size=11, bullets=True, space_after=5)

# Slide 27: the fix, chosen on validation.
s = deck.slide("The fix: a stiffer band, chosen on validation",
               "The screen tests the loss, not the network: for each candidate we solved every validation "
               "pixel's best answer directly and scored it against the solver's DEMs. It answers: if a "
               "network learned this loss perfectly, how close to the solver would it get? DEM error weights "
               "each brightness level by its share of validation pixels. Stiffer helps bright pixels and "
               "costs faint ones: at faint pixels the band is mostly noise, and a stiff band makes the best "
               "answer chase it, so Quiet error is lowest near mu = 30 and rises beyond. mu = 100 gives most "
               "of the gain with Quiet error still below the supervised model's; 300 lowers overall error "
               "only from 2.32 to 1.80 and raises Quiet error to 0.048. A second candidate, mu = 30 with the "
               "sparsity term divided by the square root of brightness, screened as well (1.95, Quiet 0.032) "
               "but trained worse (validation 22.4 against 20.1). The validation sample holds more flare "
               "cores than the shared test set, so these values run higher; compare rows with each other. "
               "The band weight is a hyperparameter chosen with solver DEMs on validation, as any loss "
               "weight would be; training never sees them.")
SCREEN = [["Best answer under the loss, validation", "DEM error", "Quiet MSE", "Flare-core 94 Å",
           "Flare-core emission"],
          ["Original (μ = 1)", "36.9", "0.076", "26%", "30%"],
          ["μ = 10", "10.8", "0.037", "81%", "90%"],
          ["μ = 30", "5.10", "0.032", "89%", "97%"],
          ["μ = 100", "2.32", "0.041", "93%", "100%"],
          ["μ = 300", "1.80", "0.048", "94%", "101%"],
          ["Supervised model (a trained network)", "6.10", "0.054", "91%", "95%"]]
tb = table(s, L, TOP + Inches(0.05), FULL_W, SCREEN, col_w=[3.2, 1.2, 1.2, 1.5, 1.6], size=11.5,
           row_h=Inches(0.35), bold_first_col=True, highlight=(4,))
text(s, L, Inches(3.6), FULL_W, Inches(1.2),
     ["If a network learned the loss perfectly, how close to the solver would it get? Solved per pixel, "
      "no network, on the validation split.",
      "Stiffer helps bright pixels and costs faint ones: at faint pixels the band is mostly noise.",
      [("μ = 100: most of the gain, Quiet error still below the supervised model's.",
        {"bold": True, "color": PURPLE})]],
     size=12, bullets=True, space_after=4)

# Slide 28: trained on the stiff band.
s = deck.slide("On the stiff band, longer training is what helps",
               "All at width 336 with the square root plus Fourier input unless noted. The band alone "
               "halves DEM MSE (4.56 to 2.37) and lifts flare-core emission from 35% to 89%. Training is now "
               "the limit: at flare cores the network's loss sits 118% above its best answer after 40 epochs, "
               "against 8% on the original loss; after 120 epochs, 47%. So 120 epochs helps, and 120 more "
               "continued from that checkpoint with a fresh schedule helps again: DEM MSE 1.75, median 0.036, "
               "AIA MSE 103 (below the supervised model's 157), Quiet MSE 0.032. 240 epochs from scratch "
               "gets the same DEM gains but breaks the band on a few rare pixels: AIA MSE 591, and its band "
               "penalty at the validation flare cores is 130 times the 120-epoch model's (the continuation: "
               "13 times). Oversampling still hurts (DEM MSE 2.50). Bigger models blow up on a few "
               "flare-core pixels: 10M (DEM MSE 35.7, AIA MSE 1.7M) and 680 wide (DEM MSE 897, its worst 1% "
               "of pixels carry 99.996% of the error). log1p input is worse here (3.23), so one encoding "
               "serves both tracks. Removing the gradient clip fits AIA better but worsens every DEM "
               "measure. Validation is the DEM error against the solver on the validation split, which "
               "chose the model; it holds more flare cores than the shared test set, so it runs higher.")
STIFF = [["BP run, band ×100", "DEM MSE", "Median", "AIA MSE", "Quiet MSE", "Flare-core EM",
          "Flare-core 94 Å", "Validation"],
         ["Original loss, base (reference)", "4.56", "0.023", "83", "0.057", "35%", "15%", "39.7"],
         ["40 epochs", "2.37", "0.062", "44", "0.035", "89%", "68%", "20.1"],
         ["120 epochs", "2.00", "0.041", "44", "0.032", "92%", "75%", "17.8"],
         ["240 epochs from scratch", "1.74", "0.036", "591", "0.035", "91%", "73%", "14.0"],
         ["120 + 120 epochs, continued", "1.75", "0.036", "103", "0.032", "93%", "77%", "15.8"],
         ["+ oversample flare cores", "2.50", "0.071", "39", "0.038", "88%", "63%", "21.1"],
         ["10M parameters", "35.7", "0.041", "1.7M", "0.070", "92%", "74%", ""],
         ["680 wide (1.43M), 120 epochs", "897", "0.034", "21.7M", "12.5", "115%", "78%", "4,170"],
         ["Supervised model", "0.91", "0.112", "157", "0.034", "92%", "85%", "6.10"]]
tb = table(s, L, TOP - Inches(0.05), FULL_W, STIFF, col_w=[2.7, 0.9, 0.8, 0.9, 0.9, 1.2, 1.3, 1.0], size=10,
           row_h=Inches(0.29), bold_first_col=True, highlight=(5,))
text(s, L, Inches(3.95), FULL_W, Inches(0.9),
     ["The band alone halves DEM MSE; the network then sits far above its best answer (118% at flare cores), "
      "so training is the limit and longer training helps. 240 from scratch misfits rare pixels (AIA MSE 591); "
      "bigger models blow up.",
      [("Final BP model: band ×100, 120 epochs, then 120 more continued from that checkpoint.",
        {"bold": True, "color": PURPLE})]],
     size=11.5, bullets=True, space_after=3)

# Slide 29: flare cores now.
s = deck.slide("Flare cores now: 93% of the solver's emission, 77% of its 94 Å",
               "Shared test set, flare cores (32x and above), new model against the old final and the "
               "supervised model. Emission 40% to 93% (supervised 92%); 94 A reproduced 23% to 77% (solver "
               "92%, supervised 85%); 131 A 76% to 101%; peak height 23% to 66% of the solver's (supervised "
               "91%); mean temperature shift -0.18 dex to -0.01; W1 0.25 to 0.03 dex; curves with extra "
               "peaks 83% to 25% (solver 18%, supervised 13%), and their share of the BP error 42% to 3.5%. "
               "It tracks the solver's peak height to about 100 and still falls short on the tallest peaks: "
               "0.80 of the solver at 100 to 320, 0.40 above 3,000.")
NOW = [["BP flare cores, shared test", "Old BP", "New BP", "Supervised", "Solver"],
       ["Emission vs solver", "40%", "93%", "92%", "100%"],
       ["94 Å reproduced", "23%", "77%", "85%", "92%"],
       ["131 Å reproduced", "76%", "101%", "90%", "101%"],
       ["Peak height vs solver", "23%", "66%", "91%", "100%"],
       ["Temperature shift (dex)", "−0.18", "−0.01", "0.00", "0"],
       ["Curves with extra peaks", "83%", "25%", "13%", "18%"]]
tb = table(s, L, TOP, Inches(4.7), NOW, col_w=[2.2, 0.7, 0.7, 0.9, 0.7], size=11, row_h=Inches(0.36),
           bold_first_col=True)
text(s, L, Inches(3.7), Inches(4.7), Inches(1.15),
     ["Spurious extra peaks were 42% of the old error; 3.5% now.",
      "It tracks the solver's peak height to ~100; above 3,000 it reaches 40%."],
     size=11, bullets=True, space_after=4)
picture(s, os.path.join(P22, "fig4_example_curves_bp.png"), Inches(5.15), TOP - Inches(0.05),
        Inches(4.45), Inches(3.6))
caption(s, Inches(5.15), Inches(4.5), Inches(4.45),
        "New BP model at random flare cores (top rows), the same draw as two slides back.")

# Slide 30: what remains.
s = deck.slide("What remains: flare cores still set the BP DEM MSE",
               "Our error over the supervised model's by brightness: 0.55 on the faintest pixels, 0.98 at "
               "0.3 to 1x, 1.2 at 1 to 3x, 0.9 at 3 to 10x, 1.9 at 10 to 32x, 2.0 at flare cores (old model: "
               "0.64, 1.8, 2.2 to 2.6, 5, 18, 4.7). Flare cores hold 94% of our BP error (supervised 90%) and "
               "set the DEM MSE, 1.75 against 0.91. At flare cores we reproduce 77% of 94 A against the "
               "solver's 92%; the cool channels and 131 A are at 100%. Longer training keeps closing the gap "
               "(240 epochs from scratch cuts the flare-core error another 13%) but starts to misfit a few "
               "pixels' AIA; a third 120-epoch continuation is untested. Why the median pixel error rose from "
               "0.026 to 0.036 while Quiet MSE fell from 0.058 to 0.032 is open.")
text(s, L, TOP, Inches(4.5), Inches(3.9),
     ["Error vs supervised by brightness: 0.55 on the faintest pixels, 1.0 at 0.3–1×, 0.9 to 1.3 at 1–10×, "
      "1.9 at 10–32×, 2.0 at flare cores (old model: 0.64, 1.8, 2 to 5, 18, 4.7).",
      "Flare cores: 94% of our BP error, and the DEM MSE gap (1.75 vs 0.91). 94 Å is at 77% of observed "
      "against the solver's 92%; every other channel is at 100%.",
      "Longer training keeps closing it; 240 epochs from scratch starts to misfit a few pixels' AIA. A third "
      "120-epoch continuation is untested.",
      [("The BP loss was the limit; training is now the limit, and it still has room.",
        {"bold": True, "color": PURPLE})]],
     size=11.5, bullets=True, space_after=7)
picture(s, os.path.join(P22, "fig5_aia_fit_flare_cores_bp.png"), Inches(5.1), TOP - Inches(0.05),
        Inches(4.5), Inches(3.5))
caption(s, Inches(5.1), Inches(4.5), Inches(4.5), "New BP model: reconstructed over observed AIA at flare cores.")

# ── Section 8: lessons and next steps ───────────────────────────────────────

deck.divider(8, "Next steps", "What we learned, and what to decide")

# Slide 32: lessons.
s = deck.slide("What we learned")
card_grid(s, [
    ("Train on the objective", "The solver's own objective reproduces its DEMs without labels."),
    ("Check the loss first", "The BP network sat at its loss's optimum, and that optimum dropped the hot "
                            "plasma. No training change could fix that; a stiffer band did."),
    ("Simple scales", "A centre-pixel MLP; the patch CNN never won on unseen days."),
    ("Size has a ceiling", "Fresh pixels pay off up to width 336; 1.43M and 10M models blow up on a few "
                           "rare pixels, on both tracks."),
    ("Heavy tails", "Report medians and percentiles: a handful of pixels can own the mean, or blow up."),
    ("Input encoding", "√ + Fourier features for both tracks: lifts ElasticNet past the supervised model "
                       "and beats log1p on the stiff BP band.")],
    cols=3, h=Inches(1.55), numbered=True, size=12)

# Slide 33: next.
s = deck.slide("Next")
card_grid(s, [
    ("Keep training BP", "The stiff-band model still improves with epochs; a third 120-epoch continuation "
                         "is untested. Watch the AIA fit on rare pixels."),
    ("Validate on every pixel", "A sampled validation set missed the blow-ups; score all pixels."),
    ("Direct DEM output", "Predict the 18 bins instead of 54 basis weights (deferred)."),
    ("Uncertainty, AIA + XRT", "A distribution over DEMs from the noisy re-solves; XRT for the hot plasma.")],
    cols=2, h=Inches(1.6), size=13)

# Slide 34: for discussion.
s = deck.slide("For discussion")
text(s, L, TOP + Inches(0.1), FULL_W, Inches(3.7),
     ["Which metric should lead the paper? DEM MSE is set by 0.01% of pixels; up to the 99th percentile "
      "we are level with or ahead of the supervised model on BP.",
      "The band weight is chosen on validation with solver DEMs, as any loss weight would be. Does that "
      "sit well with the label-free claim?",
      "Train the BP model longer (a third continuation), or freeze here at 1.9× the supervised DEM MSE?",
      "Is AIA + XRT in scope for this paper?"],
     size=14, bullets=True, space_after=14)

out = deck.save(os.path.join(SLIDES, "DEM_deck_v2.pptx"))
print("saved", out, "slides:", len(deck.prs.slides))
