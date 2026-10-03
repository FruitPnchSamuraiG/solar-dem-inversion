"""Advisor deck, built section by section from slides/deck_outline.md.

    uv run --with python-pptx --with lxml --with pillow python slides/build_deck_v2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib import (CARD, DARK, DEEP, FULL_W, GREY, IMG, L, LAV, LIGHT, MUTED, PURPLE, SLIDES, TOP,
                      WHITE, Deck, box, caption, card, line, picture, table, text)
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

deck = Deck()


# ── Section 1: problem and approach ─────────────────────────────────────────

# Slide 1: title only (the template's title slide).

# Slide 2: Part 1 divider.
deck.divider(1, "Problem and approach")

# Slide 3: why label-free — two training loops.
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

# Slide 4: the model — pipeline diagram.
s = deck.slide("The model: a small MLP feeding a fixed forward model",
               "The solvers never solve for the 18 bins directly: both solve for weights on a fixed basis, "
               "a spike and two Gaussians at each bin, and report DEM = B times x. We predict the same "
               "weights, so the network searches the same space of DEMs the solver does, and BP's "
               "sparsity criterion means the same thing for both. Only the MLP is learned. Per pixel: six "
               "log-intensities in, four hidden layers of 232 with SiLU, 54 non-negative basis weights "
               "out; DEM = B times w over 18 bins, logT 5.5 to 7.2; predicted AIA = R times DEM. "
               "176k parameters, one forward pass per pixel.")
heads = [("Observed AIA", "6 channels"), ("MLP", "176k params"), ("Basis weights", "54, same as solver"),
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
     ["6 log-intensities in; 4 hidden layers of 232, SiLU; 54 non-negative basis weights out.",
      "BP and ElasticNet solve for the same 54 weights (at each of 18 bins, a spike and Gaussians "
      "0.1 and 0.2 dex wide), and their L1 term acts on those weights.",
      "DEM = B·w over 18 bins; predicted AIA = R·DEM. B and R are fixed; only the MLP is learned.",
      "176k parameters; one forward pass per pixel."],
     size=12, bullets=True, space_after=4)

# Slide 5: the loss.
def formula(*pieces):
    """Alternating normal text and subscript pieces, as one paragraph of runs."""
    return [(piece, {"sub": i % 2 == 1}) for i, piece in enumerate(pieces) if piece]


s = deck.slide("The loss is the solver's own objective",
               "BP track: the penalty is zero anywhere inside the noise band, o plus or minus 1.4 sigma, "
               "and quadratic outside, scaled by the band width. There is no fit term inside the band, so "
               "any DEM within noise is equally good, and the L1 term on the basis weights then picks the "
               "sparsest one: BP's own criterion. ElasticNet track: the solver's ElasticNet objective with "
               "its settings, alpha 0.001, l1 ratio 0.5, C equal to the six channels, using the same noise "
               "scaling. In both, y-hat is R times B times w, and w is non-negative.")
cw = Inches(5.5)
for k, (head, eq, lines) in enumerate([
        ("BP track",
         formula("L = Σ", "c", " [ max(0, |ŷ", "c", " − o", "c", "| − tσ", "c", ") / tσ", "c",
                 " ]²  +  Σ", "k", " w", "k", ""),
         ["Zero inside the noise band (t = 1.4), quadratic outside.",
          "Then the L1 term picks the sparsest DEM that fits: BP's criterion."]),
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

# Slide 6: data — the split as a timeline, to scale.
s = deck.slide("Data: two years of full-disk AIA, split in time",
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
      "Each epoch draws 512 random pixels from each of the 58,688 blocks(917*64), for 40 epochs.",
      "Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.",
      "Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves(only supervised)."],
     size=12.5, bullets=True, space_after=7)

# ── Section 2: finding a trainable objective ────────────────────────────────

# Slide 7: Part 2 divider.
deck.divider(2, "Finding a trainable objective")

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
deck.divider(3, "Architecture on small data")

# Slide 10: first network, then the patch CNN.
s = deck.slide("A per-pixel MLP drew noisy DEMs; a patch CNN fixed it",
               "All of Part 3 uses four images: two X-class flares, quiet Sun and moderate activity. "
               "The first MLP is the same architecture as today's production model, four hidden layers "
               "with SiLU and a softplus output, at width 256 with raw intensities as input. Its noise was "
               "put down at the time to the overlapping basis amplifying small weight errors, and to the "
               "problem's non-uniqueness. Its checkpoint was not kept, so there is no figure of it.")
text(s, L, TOP - Inches(0.05), FULL_W, Inches(0.3),
     ["4 images: X2.1 flare (2011-09-06), quiet Sun (2012-06-03), moderate activity (2013-11-13), "
      "X1.6 flare (2014-09-10); small crops near disk centre."], size=10.5, color=MUTED)
text(s, L, TOP + Inches(0.4), Inches(5.4), Inches(3.3),
     [[("First try: ", {"bold": True, "color": PURPLE}),
       ("the 6 intensities of one pixel → MLP (213k params) → 54 weights, trained on the BP loss "
        "over ~221k pixels. The loss converged, but per-pixel curves oscillated.", {})],
      [("Patch CNN: ", {"bold": True, "color": PURPLE}),
       ("a 9×9 neighbourhood → CNN (~1.5M params) → 54 weights. Smooth, single-peaked curves at "
        "BP's temperatures, sparsity close to BP's on held-out pixels.", {})],
      "Trained on one image, then on all 4 jointly; zeroing the neighbourhood for 10% of batches "
      "kept it usable on a single pixel."],
     size=12.5, bullets=True, space_after=10)
picture(s, os.path.join(IMG, "patch_cnn.png"), Inches(6.0), TOP - Inches(0.1), Inches(3.6), Inches(3.5))
caption(s, Inches(6.0), Inches(4.45), Inches(3.6), "Patch CNN (dotted) vs BP, X2.1 flare image.")

# Slide 11: ablation — a dot plot of sparsity against BP.
s = deck.slide("Ablation: the noise was capacity, not missing context",
               "The shuffled-patch CNN applies a fixed random permutation to the 81 patch pixels. It was "
               "less sparse than the CNN on all 4 images, 1.88 against 1.70 on average, level with the "
               "centre-pixel MLP, so the spatial arrangement of the neighbours mattered. The "
               "flat-patch MLP was erratic across images: 1.61 on the X2.1 flare, 2.27 on quiet Sun. "
               "Capacity and setup were never fully separated: the 213k MLP took raw intensities, and at "
               "full scale a 176k MLP on log1p inputs is smooth.")
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
      "A centre-pixel MLP this size was smooth too: the first MLP's noise was not about the neighbourhood.",
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

out = deck.save(os.path.join(SLIDES, "DEM_deck_v2.pptx"))
print("saved", out, "slides:", len(deck.prs.slides))
