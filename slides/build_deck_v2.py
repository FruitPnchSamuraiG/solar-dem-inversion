"""Advisor deck, built section by section from slides/deck_outline.md.

    uv run --with python-pptx --with lxml --with pillow python slides/build_deck_v2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib import (CARD, DARK, DEEP, FULL_W, IMG, L, LAV, LIGHT, MUTED, PURPLE, SLIDES, TOP,
                      WHITE, Deck, box, caption, card, line, picture, table, text)
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

# Slide 6: three routes, one test.
s = deck.slide("Three routes to a DEM, compared on the same test",
               "Data: 1,223 Hofmeister-deconvolved timestamps, split by day into 917 train, 153 "
               "validation and 153 test. The networks never see a test day in training. DEM metrics "
               "compare each network with the solver's DEM: MSE, total-emission error and W1. AIA "
               "reconstruction compares all three with the observation: MAE and MSE.")
text(s, L, TOP + Inches(0.1), FULL_W, Inches(3.3),
     [[("Solver (BP, ElasticNet): ", {"bold": True, "color": PURPLE}),
       ("optimises its objective for each pixel. This is the reference.", {})],
      [("Supervised network: ", {"bold": True, "color": PURPLE}),
       ("learns to predict the solver's DEMs, so it needs them as labels.", {})],
      [("Label-free network: ", {"bold": True, "color": PURPLE}),
       ("learns to minimise the solver's objective, with no labels.", {})],
      [("Same test for all: ", {"bold": True, "color": PURPLE}),
       ("153 days never seen in training, identical pixels. DEM metrics against the solver; "
        "AIA reconstruction against the observation.", {})]],
     size=14, bullets=True, space_after=14)


# Slide 7: data — the split as a timeline, to scale.
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
      "Our model trains on AIA and σ only: ~30M pixels per epoch, 40 epochs.",
      "Solver DEMs (BP, ElasticNet) for every image: labels for the supervised model, evaluation only for ours.",
      "Test: ~700M pixel DEMs per solver, the clean solve plus 4 noise re-solves."],
     size=12.5, bullets=True, space_after=7)

# ── Section 2: finding a trainable objective ────────────────────────────────

# Slide 8: Part 2 divider.
deck.divider(2, "Finding a trainable objective")

# Slide 9: which loss.
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

out = deck.save(os.path.join(SLIDES, "DEM_deck_v2.pptx"))
print("saved", out, "slides:", len(deck.prs.slides))
