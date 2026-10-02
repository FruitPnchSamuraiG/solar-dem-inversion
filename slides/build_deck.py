"""Build the advisor deck on the NYU template (slides/DEM.pptx).

    uv run --with python-pptx --with lxml --with pillow python slides/build_deck.py

Panel crops in slides/img/ were cut from results/plots/ figures by slides/crops.py.

Every content slide uses the template's BLANK layout (NYU logo + slide number),
a takeaway-sentence title, at most three short lines, and one visual. Details
live in the speaker notes.
"""
import copy
import os
import sys

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))  # slides/
IMG = os.path.join(W, "img")  # panel crops made by crops.py
P13 = os.path.join(REPO, "results/plots/13_bright_diagnostic_20260928")

PURPLE = RGBColor(0x57, 0x06, 0x8C)
DEEP = RGBColor(0x33, 0x06, 0x62)
DARK = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
CARD = RGBColor(0xF4, 0xF1, 0xF8)
LAV = RGBColor(0xE3, 0xDF, 0xE9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0xB9, 0xB4, 0xC2)
TEAL = RGBColor(0x00, 0x7E, 0x8A)

prs = Presentation(os.path.join(REPO, "slides/DEM.pptx"))
SW, SH = prs.slide_width, prs.slide_height
BLANK = next(l for l in prs.slide_layouts if l.name == "BLANK")

# Drop the template's empty "Page Title Here" slide; keep the title slide.
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids)[1:]:
    prs.part.drop_rel(sid.get(qn("r:id")))
    sld_ids.remove(sid)

SLDNUM = [sp for sp in BLANK.shapes._spTree.iterchildren(qn("p:sp"))
          if sp.find(".//" + qn("p:ph")) is not None
          and sp.find(".//" + qn("p:ph")).get("type") == "sldNum"]


# ── helpers ─────────────────────────────────────────────────────────────────

def new_slide(title, notes):
    s = prs.slides.add_slide(BLANK)
    for sp in SLDNUM:                       # python-pptx does not clone slide numbers
        s.shapes._spTree.append(copy.deepcopy(sp))
    text(s, Inches(0.29), Inches(0.24), Inches(9.3), Inches(0.62), [title],
         size=20, color=PURPLE, bold=True, anchor=MSO_ANCHOR.TOP)
    s.notes_slide.notes_text_frame.text = notes
    return s


def text(slide, x, y, w, h, paras, size=13, color=DARK, bold=False, bullets=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=6, italic=False):
    """paras: list of str or list of (str, {bold, color, size}) runs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for txt, opt in runs:
            r = p.add_run()
            r.text = txt
            f = r.font
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
        if bullets:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.2))))
            pPr.set("indent", str(-Emu(Inches(0.2))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                for el in pPr.findall(qn(tag)):
                    pPr.remove(el)
            clr = etree.SubElement(etree.SubElement(pPr, qn("a:buClr")), qn("a:srgbClr"))
            clr.set("val", str(color))
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
    return tb


def picture(slide, path, x, y, w, h, align="center"):
    """Fit the image inside the box, preserving aspect ratio."""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    px = x + (w - pw) // 2 if align == "center" else x
    py = y + (h - ph) // 2
    return slide.shapes.add_picture(path, px, py, pw, ph)


def card(slide, x, y, w, h, fill=CARD):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def stat(slide, x, y, w, number, label, h=Inches(1.15), num_size=26):
    card(slide, x, y, w, h)
    text(slide, x + Inches(0.15), y + Inches(0.1), w - Inches(0.3), Inches(0.5), [number],
         size=num_size, color=PURPLE, bold=True)
    text(slide, x + Inches(0.15), y + Inches(0.6), w - Inches(0.3), h - Inches(0.65), [label],
         size=10.5, color=DARK, space_after=0)


def table(slide, x, y, w, rows, col_w=None, size=10.5, highlight=(), row_h=Inches(0.3), text_cols=1):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, x, y, w, row_h * nr)
    t = shape.table
    tblPr = t._tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        tblPr.remove(style)
    if col_w:
        total = sum(col_w)
        for j, cw in enumerate(col_w):
            t.columns[j].width = int(w * cw / total)
    for i, row in enumerate(rows):
        t.rows[i].height = row_h
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = PURPLE if i == 0 else (LAV if i in highlight else WHITE)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j < text_cols else PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.bold = (i == 0) or (i in highlight and j == 0)
            r.font.color.rgb = WHITE if i == 0 else DARK
    # light horizontal rules only
    for i in range(nr):
        for j in range(nc):
            # Borders must precede the cell fill in a:tcPr (lnL, lnR, lnT, lnB,
            # then fill). PowerPoint tolerates the reverse; Google Slides rejects it.
            tcPr = t.cell(i, j)._tc.get_or_add_tcPr()
            for k, edge in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
                ln = etree.Element(qn(edge))
                tcPr.insert(k, ln)
                ln.set("w", "6350" if edge == "a:lnB" else "0")
                if edge == "a:lnB":
                    fill = etree.SubElement(ln, qn("a:solidFill"))
                    clr = etree.SubElement(fill, qn("a:srgbClr"))
                    clr.set("val", "D9D3E3")
                else:
                    etree.SubElement(ln, qn("a:noFill"))
    return shape


def arrow(slide, x1, y1, x2, y2, color=PURPLE):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), int(x2), int(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def caption(slide, x, y, w, words):
    text(slide, x, y, w, Inches(0.3), [words], size=9.5, color=MUTED, italic=True, space_after=0)


# Common geometry
L = Inches(0.4)
TOP = Inches(1.0)
FULL_W = Inches(9.2)
BOTTOM = Inches(4.85)

# ── 1. title ────────────────────────────────────────────────────────────────
title_slide = prs.slides[0]
text(title_slide, Inches(1.0), Inches(3.15), Inches(8.0), Inches(0.8),
     ["A neural DEM inversion trained on the solver's objective, not its labels",
      [("Progress update · October 2026", {"size": 12, "color": MUTED})]],
     size=15, color=DEEP, align=PP_ALIGN.CENTER, space_after=4)
title_slide.notes_slide.notes_text_frame.text = (
    "What we built since June, what each experiment taught us, how it compares with the "
    "supervised model, and the one failure mode that remains.")

# ── 2. outline ──────────────────────────────────────────────────────────────
s = new_slide("Outline", "Four parts. Most of the time goes to parts 3 and 4.")
items = [("1", "Approach", "Train on the solver's objective"),
         ("2", "What we tried", "Losses, architectures, scale, model size"),
         ("3", "How it compares", "Matched test against the supervised model"),
         ("4", "Where it fails", "Bright pixels, traced to flare cores")]
cw, gap = Inches(2.15), Inches(0.2)
for k, (n, head, sub) in enumerate(items):
    x = L + k * (cw + gap)
    card(s, x, Inches(1.6), cw, Inches(2.2))
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.2), Inches(1.85), Inches(0.5), Inches(0.5))
    circ.fill.solid(); circ.fill.fore_color.rgb = PURPLE; circ.line.fill.background()
    tf = circ.text_frame; tf.margin_left = tf.margin_right = 0
    r = tf.paragraphs[0].add_run(); r.text = n
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = WHITE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    text(s, x + Inches(0.2), Inches(2.55), cw - Inches(0.4), Inches(0.4), [head],
         size=15, color=PURPLE, bold=True)
    text(s, x + Inches(0.2), Inches(2.95), cw - Inches(0.4), Inches(0.8), [sub], size=11.5)

# ── 3. approach ─────────────────────────────────────────────────────────────
s = new_slide("Approach: learn the solver's objective, not its answers",
              "Same physics as the solvers. The network maps each pixel's six AIA intensities to 54 "
              "non-negative basis weights; the fixed basis gives an 18-bin DEM (logT 5.5 to 7.2); the "
              "AIA response projects it back to six intensities. The loss is the solver's own objective: "
              "for BP a barrier version of the tolerance band |RDx - o| <= t sigma plus an L1 term; for "
              "ENet the ElasticNet objective. Solver DEMs are never used in training, only to evaluate. "
              "Data: 1,223 Hofmeister-deconvolved timestamps, split by timestamp 917 / 153 / 153.")
boxes = [("Observed AIA", "6 channels"), ("MLP", "176k params, per pixel"),
         ("Basis weights", "54, all ≥ 0"), ("DEM", "18 bins, logT 5.5–7.2"),
         ("Predicted AIA", "via response R")]
bw, bh, by = Inches(1.5), Inches(0.95), Inches(1.35)
bgap = (FULL_W - 5 * bw) // 4
xs = [L + i * (bw + bgap) for i in range(5)]
for x, (head, sub) in zip(xs, boxes):
    b = card(s, x, by, bw, bh, fill=PURPLE if head == "MLP" else CARD)
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = head; r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = WHITE if head == "MLP" else PURPLE
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(10)
    r2.font.color.rgb = WHITE if head == "MLP" else DARK
for i in range(4):
    arrow(s, xs[i] + bw, by + bh // 2, xs[i + 1], by + bh // 2)
# loss loop under the boxes
ly = by + bh + Inches(0.55)
lb = card(s, xs[1], ly, xs[4] + bw - xs[1], Inches(0.62), fill=LAV)
tf = lb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Loss = the solver's objective on predicted vs observed AIA"
r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = PURPLE
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run(); r2.text = "BP: fit within the noise band + L1 sparsity   ·   ENet: ElasticNet"
r2.font.size = Pt(10); r2.font.color.rgb = DARK
arrow(s, xs[4] + bw // 2, by + bh, xs[4] + bw // 2, ly)
down = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, xs[0] + bw // 2, by + bh, xs[0] + bw // 2, ly + Inches(0.31))
down.line.color.rgb = PURPLE; down.line.width = Pt(1.5)
arrow(s, xs[0] + bw // 2, ly + Inches(0.31), xs[1], ly + Inches(0.31))
text(s, L, Inches(3.75), FULL_W, Inches(1.0),
     ["Solver DEMs are never used in training, only to evaluate.",
      "1,223 timestamps, split by day: 917 train, 153 validation, 153 test."],
     size=12.5, bullets=True)

# ── 4. losses ───────────────────────────────────────────────────────────────
s = new_slide("Barrier-style losses reproduce BP; fit-only losses do not",
              "June. Direct per-pixel optimisation of five differentiable losses on four timestamps, "
              "compared with BP. Barrier and barrier-fit track BP's temperature structure and were "
              "closest to BP on MAE and Wasserstein distance. Chi-square smooth, entropy and Tikhonov "
              "reconstruct AIA better but drift from BP's shape because they lack its sparsity term. "
              "L-BFGS and Adam reach the same curves; SGD often fails to converge.")
text(s, L, TOP, FULL_W, Inches(1.2),
     ["Optimised 5 differentiable losses directly per pixel, against BP.",
      "Fit-only losses reconstruct AIA better but lose BP's sparse shape.",
      [("Lesson: the sparsity term, not the AIA fit, makes a DEM BP-like.", {"bold": True, "color": PURPLE})]],
     bullets=True)
picture(s, os.path.join(REPO, "results/plots/01_multiloss_20260609/20140910_1731_mean_logt.png"),
        L, Inches(2.3), FULL_W, Inches(2.2))
caption(s, L, Inches(4.5), FULL_W,
        "Mean logT maps, X1.6 flare timestamp. Left to right: BP, barrier, barrier-fit, χ² smooth, entropy, Tikhonov.")

# ── 5. first networks ───────────────────────────────────────────────────────
s = new_slide("A 6-channel network fit the loss but drew noisy DEMs",
              "Mid-June. A per-pixel network from the six intensities, trained directly on the barrier "
              "loss over about 221k pixels from four timestamps. The loss converged, but individual "
              "curves oscillated compared with BP and with direct optimisation. Six numbers constrain "
              "eighteen bins, so small errors in the overlapping basis functions show up as wiggles. "
              "This is what pushed us toward giving the network spatial context.")
text(s, L, TOP, Inches(4.2), Inches(3.4),
     ["Trained on the barrier loss over ~221k pixels from 4 timestamps.",
      "The loss converged, yet per-pixel curves oscillated.",
      "Six numbers constrain 18 bins: small basis errors become wiggles.",
      [("Next question: does spatial context help?", {"bold": True, "color": PURPLE})]],
     bullets=True, space_after=10)
picture(s, os.path.join(IMG, "noisy_nn.png"), Inches(4.8), TOP, Inches(4.8), Inches(3.55))
caption(s, Inches(4.8), Inches(4.58), Inches(4.8), "BP (black) against the network and direct optimisation, X2.1 flare.")

# ── 6. patch CNN ────────────────────────────────────────────────────────────
s = new_slide("Seeing the neighbourhood gave smooth, BP-like curves",
              "Late June. The patch CNN sees each pixel's 9x9 neighbourhood. Built in three steps: one "
              "image, then amortised across four timestamps, then with neighbourhood masking (10% best). "
              "Curves became smooth and single-peaked at the right temperatures, with sparsity close to "
              "BP's. Caveat at the time: validation pixels came from the same four images, so this was "
              "not yet a test on unseen days.")
text(s, L, TOP, Inches(4.3), Inches(1.4),
     ["Patch CNN: each pixel sees its 9×9 neighbourhood.",
      "One image, then 4 timestamps, then neighbourhood masking.",
      [("Caveat: held-out pixels of seen images, not unseen days.", {"bold": True, "color": PURPLE})]],
     bullets=True)
table(s, L, Inches(2.65), Inches(4.3),
      [["Timestamp", "Network", "BP"], ["X2.1 flare", "1.84", "1.97"], ["Quiet Sun", "1.58", "1.67"],
       ["Moderate activity", "1.47", "1.82"], ["X1.6 flare", "1.86", "1.71"]],
      col_w=[2.2, 1, 1])
caption(s, L, Inches(4.2), Inches(4.3), "Sparsity of basis weights on held-out pixels; lower is sparser.")
picture(s, os.path.join(IMG, "patch_cnn.png"), Inches(5.0), TOP, Inches(4.6), Inches(3.55))
caption(s, Inches(5.0), Inches(4.58), Inches(4.6), "BP (black) and the patch CNN, X2.1 flare.")

# ── 7. ablation + LOO ───────────────────────────────────────────────────────
s = new_slide("On unseen days, the patch CNN's advantage disappeared",
              "July. Capacity-matched ablation (~1.5M params): a centre-pixel MLP also gave smooth curves, "
              "so the old noise was largely a capacity effect. In-sample the CNN looked best (1.70 vs MLP "
              "1.89, BP 1.79). Leave-one-timestamp-out reversed that: the MLP won 2 of 4 held-out days, "
              "including the hard X1.6 flare, tied one, lost the quiet Sun. Lesson: test on unseen days. "
              "The flare fold showed peak shifts to lower temperature, a first sign of the bright-pixel "
              "weakness we analyse later.")
text(s, L, TOP, Inches(4.3), Inches(1.6),
     ["Capacity-matched ablation: a centre-pixel MLP is smooth too.",
      "Leave one day out: MLP wins 2 of 4, ties 1, incl. the flare.",
      [("Lesson: in-sample scores flattered the CNN.", {"bold": True, "color": PURPLE})]],
     bullets=True)
table(s, L, Inches(2.65), Inches(4.3),
      [["Held-out day", "Patch CNN", "Centre MLP"], ["X2.1 flare", "1.64", "1.59"],
       ["Quiet Sun", "1.61", "1.78"], ["Moderate activity", "1.83", "1.83"], ["X1.6 flare", "2.31", "1.96"]],
      col_w=[2.0, 1.1, 1.1])
caption(s, L, Inches(4.2), Inches(4.3), "Held-out sparsity, lower is sparser; BP is about 1.8.")
picture(s, os.path.join(IMG, "loo_flare.png"), Inches(5.0), TOP, Inches(4.6), Inches(3.55))
caption(s, Inches(5.0), Inches(4.58), Inches(4.6), "X1.6 flare held out: BP (black), CNN and MLP (dashed).")

# ── 8. scaling ──────────────────────────────────────────────────────────────
s = new_slide("Scaling to full disk: one silent failure, one fix",
              "End of July. Generated BP and ENet labels for 1,223 deconvolved timestamps, for evaluation "
              "only. BP's --zerochill flag disabled its tolerance relaxation, leaving 10.2% of pixels "
              "unsolved, mostly bright active-region and flare pixels; removing it gave 0.02%. The first "
              "full-disk training runs then predicted identically zero: intensities span six decades, the "
              "first forward pass overshot the band by orders of magnitude, and the softplus output died "
              "(zero gradient). Fixes: log input, LR warmup, a softplus floor that keeps a gradient, a "
              "small output-layer init, and a guard that stops on collapse. A 64-block smoke test never "
              "drew the bright pixels that triggered it.")
cw2 = Inches(4.45)
for k, (num, head, body) in enumerate([
        ("10.2% → 0.02%", "Unsolved BP pixels",
         "Restoring BP's tolerance relaxation recovered the active-region and flare pixels it was dropping."),
        ("5 × 10¹³", "Loss at step 0 of the first full-disk run",
         "Six decades of input range killed the output layer; the networks predicted zeros. Fixed with a log input, warmup and a safe output init.")]):
    x = L + k * (cw2 + Inches(0.3))
    card(s, x, TOP, cw2, Inches(2.3))
    text(s, x + Inches(0.25), TOP + Inches(0.2), cw2 - Inches(0.5), Inches(0.6), [num],
         size=28, color=PURPLE, bold=True)
    text(s, x + Inches(0.25), TOP + Inches(0.85), cw2 - Inches(0.5), Inches(0.4), [head],
         size=13, bold=True)
    text(s, x + Inches(0.25), TOP + Inches(1.3), cw2 - Inches(0.5), Inches(1.3), [body], size=11.5)
text(s, L, Inches(3.6), FULL_W, Inches(0.5),
     [[("Lesson: a short smoke test never saw the bright pixels that broke training.", {"bold": True, "color": PURPLE})]],
     size=12.5)

# ── 9. MLP at scale ─────────────────────────────────────────────────────────
s = new_slide("At scale the simple MLP won, and the mean loss misled",
              "August 2. 153 unseen test timestamps, about 5M pixels. Test reproduced validation to the "
              "third decimal, so no overfitting. The centre-pixel MLP beat the patch CNN at every "
              "percentile of the training objective; the CNN's apparent win on the mean came from one "
              "pixel that was 94% of the MLP's mean. Since then we report percentiles and per-band results, "
              "never a bare mean. The MLP became the production architecture.")
text(s, L, TOP, Inches(4.0), Inches(3.3),
     ["153 unseen test days: test matched validation to 3 decimals.",
      "The MLP beat the CNN at every percentile of the objective.",
      "One pixel in 5 million was 94% of the MLP's mean loss.",
      [("Lesson: report percentiles, not means.", {"bold": True, "color": PURPLE})]],
     bullets=True, space_after=10)
table(s, Inches(4.75), Inches(1.15), Inches(4.85),
      [["BP objective, test split", "Centre MLP", "Patch CNN"],
       ["Sparsity (BP 1.79)", "1.91", "1.98"], ["Median loss", "0.98", "1.01"],
       ["99th percentile", "17.9", "18.8"], ["99.99th percentile", "148", "255"],
       ["Mean loss", "44.3", "11.3"]],
      col_w=[2.2, 1.1, 1.1], size=11, highlight=(5,), row_h=Inches(0.36))
caption(s, Inches(4.75), Inches(3.4), Inches(4.85), "Lower is better. The mean ranks them backwards.")

# ── 10. size sweep ──────────────────────────────────────────────────────────
s = new_slide("176k parameters suffice; more capacity doesn't help",
              "August 3 to 7. Swept the MLP from 10k to 11.2M parameters on both losses, scored on the "
              "test split. Averages barely move across sizes, but recall of BP's multi-peaked pixels "
              "collapses below 176k (26% to 8%), which is what decided the size. Above 176k nothing we "
              "report improves: recall peaks at 2.8M, precision stays near 81%, sparsity creeps toward BP "
              "while AIA fit slowly worsens. 176k became production: 8x smaller than the 1.43M model.")
cd = CategoryChartData()
cd.categories = ["87k", "176k", "360k", "722k", "1.4M", "2.8M", "5.6M", "11.2M"]
vals = [8.2, 26.4, 27.5, 28.4, 29.3, 29.9, 28.5, 28.7]
cd.add_series("Recall of BP multi-peaked pixels (%)", vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, L, TOP, Inches(5.4), Inches(3.5), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Recall of BP's multi-peaked pixels, by model size (%)"
tr = ch.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size = Pt(11); tr.font.bold = True; tr.font.color.rgb = DARK
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = '0.0'; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(10); dl.font.color.rgb = DARK
ser = plot.series[0]
for i in range(len(vals)):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = PURPLE if i == 1 else GREY
va = ch.value_axis
va.maximum_scale = 35; va.minimum_scale = 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE5, 0xE0)
va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = MUTED
va.format.line.fill.background()
ca = ch.category_axis
ca.tick_labels.font.size = Pt(10); ca.tick_labels.font.color.rgb = DARK
# python-pptx can write negative axis ids, which the schema forbids (unsignedInt)
# and Google Slides rejects. Remap every id consistently to a positive value.
_ids = {}
for el in ch._chartSpace.iter(qn("c:axId"), qn("c:crossAx")):
    v = el.get("val")
    if int(v) < 0:
        el.set("val", _ids.setdefault(v, str(500000001 + len(_ids))))
text(s, Inches(6.05), TOP + Inches(0.2), Inches(3.55), Inches(3.2),
     ["Swept 10k to 11.2M parameters.",
      "Below 176k the multi-peaked pixels are lost while averages barely move.",
      "Above it, nothing we report improves.",
      [("176k became production: 8× smaller.", {"bold": True, "color": PURPLE})]],
     bullets=True, space_after=10)

# ── 11. bimodality ──────────────────────────────────────────────────────────
s = new_slide("Multi-peaked DEMs: precise but partial, and mostly noise",
              "August. 14% of test pixels have two interior BP peaks (rising to 30% in the brightest "
              "decile). When the model draws two peaks it is right 80% of the time against a 14% base "
              "rate, but it recalls only about 30%; misses are mostly close, unequal peaks. Re-solving BP "
              "30 times under photon noise moves BP's own answer as much as our prediction differs from "
              "it (ratio 0.98 at bimodal pixels, 0.69 at unimodal ones). So most of the bimodal gap is the "
              "label's own irreproducibility. Caveat for claims: detecting peaks is not reproducing curves.")
sw = Inches(2.05)
for k, (num, lab) in enumerate([("14%", "of test pixels have two interior BP peaks"),
                                 ("80%", "precision when the model draws two peaks"),
                                 ("~30%", "of BP's double peaks recalled"),
                                 ("0.98×", "model–BP gap vs BP's own scatter under noise")]):
    stat(s, L + (k % 2) * (sw + Inches(0.15)), TOP + (k // 2) * Inches(1.3), sw, num, lab)
text(s, L, Inches(3.75), Inches(4.25), Inches(0.8),
     [[("Most of the bimodal gap is the label's own noise.", {"bold": True, "color": PURPLE})]],
     size=12.5)
picture(s, os.path.join(IMG, "bimodal.png"), Inches(4.9), TOP, Inches(4.7), Inches(3.55))
caption(s, Inches(4.9), Inches(4.58), Inches(4.7), "BP (black), its noisy re-solves (grey), networks (dashed).")

# ── 12. comparison ──────────────────────────────────────────────────────────
s = new_slide("Versus supervised: worse DEM MSE, better W1 and AIA MSE",
              "September. Matched comparison with the supervised BP and ENet models on the full shared "
              "test set: 153 held-out timestamps, the same pixels and the same Bright mask for both. The "
              "label-free ENet model was retrained with the matched objective (alpha = 0.001). Label-free "
              "has the better W1 and AIA MSE on both tracks and is better on everything except DEM MSE on "
              "ENet. The BP DEM MSE gap (4.19 vs 0.91) is what the rest of the talk explains. Live "
              "viewer: triborough.cs.nyu.edu/hsr3649/demdemo/webapp/compare.html")
table(s, L, TOP, FULL_W,
      [["Track", "Model", "DEM MSE", "EM error %", "W1 (dex)", "AIA MAE", "AIA MSE"],
       ["BP", "Supervised", "0.906", "14.3", "0.133", "2.95", "156.7"],
       ["BP", "Label-free", "4.191", "20.2", "0.085", "4.32", "82.6"],
       ["ENet", "Supervised", "0.318", "18.0", "0.121", "6.32", "597.8"],
       ["ENet", "Label-free", "0.588", "13.5", "0.118", "0.67", "193.4"]],
      col_w=[0.8, 1.3, 1, 1, 1, 1, 1], size=11.5, highlight=(2, 4), row_h=Inches(0.4), text_cols=2)
caption(s, L, Inches(3.05), FULL_W, "Lower is better. 153 held-out days, same pixels for both models.")
text(s, L, Inches(3.45), FULL_W, Inches(1.2),
     ["Label-free: lower W1 and AIA MSE on both tracks; on ENet, better on all but DEM MSE.",
      [("The BP DEM MSE gap is what the rest of this talk explains.", {"bold": True, "color": PURPLE})]],
     bullets=True)

# ── 13. mean vs median ──────────────────────────────────────────────────────
s = new_slide("The mean and the median tell opposite stories",
              "BP track. The website mean says label-free is 4.6x worse; the median pixel says it is "
              "about 4x better. Both are true because squared error lets a few huge errors dominate: the "
              "worst 1% of pixels hold 97.8% of label-free's error, and Bright pixels hold 98.8%. On the "
              "typical Bright pixel label-free is still worse (2.82 vs 1.78), so there is a real Bright "
              "weakness, but the mean exaggerates it enormously.")
table(s, L, TOP, Inches(5.3),
      [["BP track", "Label-free", "Supervised"],
       ["DEM MSE (website mean)", "4.19", "0.906"],
       ["Median pixel error", "0.026", "0.112"],
       ["Median, Quiet pixels", "0.016", "0.085"],
       ["Median, Bright pixels", "2.82", "1.78"],
       ["Error held by worst 1% of pixels", "97.8%", "95.2%"]],
      col_w=[2.6, 1.1, 1.1], size=11.5, row_h=Inches(0.42))
stat(s, Inches(6.0), TOP, Inches(3.6), "4.6× worse", "on the mean (website number)", h=Inches(1.25), num_size=24)
stat(s, Inches(6.0), TOP + Inches(1.45), Inches(3.6), "~4× better", "on the typical pixel (median)", h=Inches(1.25), num_size=24)
text(s, L, Inches(3.85), FULL_W, Inches(0.6),
     [[("A few pixels decide the mean. Which ones?", {"bold": True, "color": PURPLE})]], size=12.5)

# ── 14. flare cores ─────────────────────────────────────────────────────────
s = new_slide("Flare cores, 1 pixel in 10,000, hold ~90% of the error",
              "Pixels sorted by brightness: the largest of the six channel ratios, observed over that "
              "channel's top-5% cutoff (1x is the Bright line). The 69,283 pixels at 32x or more, 0.01% "
              "of 705M, hold 89.4% of label-free error and 90.0% of supervised error. Their DEMs are "
              "thousands of times larger than typical, and squared error grows with that square, so they "
              "dominate any squared-error score. The headline 4.63x gap equals the gap on this bucket "
              "(4.59x). These pixels are over the cutoff in all six channels, usually led by 94 A.")
picture(s, os.path.join(P13, "fig1_error_share_by_brightness.png"), L, TOP, Inches(5.5), Inches(3.75), align="left")
text(s, Inches(6.1), TOP + Inches(0.1), Inches(3.5), Inches(3.4),
     ["Brightness: largest channel over its top-5% cutoff.",
      "Same for both models: it's the metric meeting huge DEMs.",
      "The 4.6× headline gap is the gap on this bucket.",
      [("Is label-free also worse relative to DEM size?", {"bold": True, "color": PURPLE})]],
     bullets=True, space_after=10)

# ── 15. relative error ──────────────────────────────────────────────────────
s = new_slide("Relative to DEM size, the gap grows with brightness",
              "Relative error per band: total squared error over total squared solver DEM, so the size "
              "effect cancels. Supervised gets relatively better as pixels brighten; label-free improves "
              "to about 3x, then turns worse (0.46 at 10-32x, 0.80 at 32x+). On the faint half of the Sun "
              "label-free is better on BP. ENet is within 27% of supervised in every band below 10x.")
picture(s, os.path.join(P13, "fig2_relative_error_by_brightness.png"), L, TOP, FULL_W, Inches(3.3))
text(s, L, Inches(4.35), FULL_W, Inches(0.5),
     [[("Faint half of the Sun: label-free equal or better. ", {}),
       ("Flare cores: genuinely worse, not just bigger numbers.", {"bold": True, "color": PURPLE})]],
     size=12)

# ── 16. example curves ──────────────────────────────────────────────────────
s = new_slide("At flare cores, label-free misses the hot plasma",
              "Randomly drawn pixels from a seeded sample, not hand-picked. Top two rows are flare cores "
              "(32x and above), bottom row ordinary Bright pixels. Solver and supervised both have a tall "
              "hot peak near logT 6.8 to 6.9; label-free nearly misses it, with a small bump near 6.3 and "
              "a low tail near 7.0. At these pixels label-free predicts 41% of the solver's emission, a "
              "quarter of its peak height, 0.17 dex cooler, multi-peaked 83% of the time (solver 18%).")
picture(s, os.path.join(P13, "fig4_example_curves_bp.png"), L, TOP, Inches(6.2), Inches(3.75), align="left")
for k, (num, lab) in enumerate([("41%", "of the solver's emission"), ("0.24×", "of its peak height"),
                                 ("83%", "of curves split into humps")]):
    stat(s, Inches(6.85), TOP + k * Inches(1.25), Inches(2.75), num, lab, h=Inches(1.1), num_size=24)

# ── 17. ceiling ─────────────────────────────────────────────────────────────
s = new_slide("It tracks the solver up to peak ~100, then saturates",
              "Predicted peak over solver peak, by solver peak height; 1 is a perfect match. Label-free "
              "holds 0.8 to 0.9 up to a solver peak of about 100, then falls to 0.33, 0.21, 0.19 and "
              "0.11. Supervised stays near 0.9. So this is saturation at the top end, not regression to "
              "the mean. On BP, supervised has its own weak end: it over-predicts the faintest DEMs, which "
              "is why label-free wins on faint pixels. Those values are near zero, so they carry little error.")
picture(s, os.path.join(P13, "fig3_peak_height_calibration.png"), L, TOP, FULL_W, Inches(3.3))
text(s, L, Inches(4.35), FULL_W, Inches(0.5),
     [[("Not regression to the mean: a ceiling at the tallest DEMs.", {"bold": True, "color": PURPLE})]],
     size=12)

# ── 18. AIA check ───────────────────────────────────────────────────────────
s = new_slide("At flare cores label-free fails its own objective",
              "Each DEM projected through the AIA response and compared with the observation at flare "
              "cores. The BP solver reproduces all six channels within 8%, so a DEM that fits exists. "
              "Label-free fits 171, 193 and 211 A to 2% but reproduces only 24% of 94 A, 76% of 131 A and "
              "66% of 335 A; 98% of flare-core pixels fall short in 94 A by more than 10%. The loss asks "
              "for the right answer and training does not deliver it on these rare pixels: a training "
              "problem. ENet is murkier, since the ENet solver itself reaches only 55% of 94 A there.")
picture(s, os.path.join(P13, "fig5_aia_fit_flare_cores.png"), L, TOP, Inches(6.0), Inches(3.6), align="left")
text(s, Inches(6.6), TOP + Inches(0.1), Inches(3.0), Inches(3.4),
     ["BP solver: all six channels within 8%.",
      "Label-free: cool channels fit, 94 Å only 24%.",
      [("The objective asks for the right answer; training misses it.", {"bold": True, "color": PURPLE})]],
     bullets=True, space_after=10)

# ── 19. what it is not ──────────────────────────────────────────────────────
s = new_slide("Not solver noise, not missed double peaks",
              "Solver noise: where a block was re-solved 2 to 5 times, the spread of BP's answers is an "
              "unavoidable floor. It explains 0.4% of label-free's Bright error on BP (1.5% for "
              "supervised; 5.7% and 9.1% on ENet). Double peaks: missing BP's second peak costs about 1% of "
              "the error. Drawing extra peaks where BP has one costs label-free 22% of its BP error and "
              "supervised 0.9%, because label-free's extra peaks sit on the huge flare-core DEMs.")
for k, (num, head, body) in enumerate([
        ("0.4%", "of the Bright error is solver noise",
         "BP's own scatter across noisy re-solves explains almost none of it (supervised 1.5%)."),
        ("~1%", "of the error from missing BP's second peak",
         "Extra peaks at flare cores cost more: 22% of label-free's error, 0.9% of supervised's.")]):
    x = L + k * (cw2 + Inches(0.3))
    card(s, x, TOP, cw2, Inches(2.15))
    text(s, x + Inches(0.25), TOP + Inches(0.2), cw2 - Inches(0.5), Inches(0.6), [num],
         size=28, color=PURPLE, bold=True)
    text(s, x + Inches(0.25), TOP + Inches(0.85), cw2 - Inches(0.5), Inches(0.5), [head], size=13, bold=True)
    text(s, x + Inches(0.25), TOP + Inches(1.4), cw2 - Inches(0.5), Inches(1.1), [body], size=11.5)
text(s, L, Inches(3.45), FULL_W, Inches(0.6),
     [[("The Bright gap is real model error, concentrated in flare cores.", {"bold": True, "color": PURPLE})]],
     size=12.5)

# ── 20. learnings ───────────────────────────────────────────────────────────
s = new_slide("What we learned",
              "The five lessons, in the order we learned them.")
lessons = ["The sparsity term, not the AIA fit, makes a DEM BP-like.",
           "Test on unseen days: in-sample scores flattered the CNN.",
           "Means of squared errors hide everything: use percentiles and per-band results.",
           "176k parameters suffice; more capacity doesn't recover multi-peaked DEMs.",
           "The remaining gap is narrow and diagnosed: flare cores, a training problem."]
for k, lesson in enumerate(lessons):
    y = TOP + k * Inches(0.72)
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, L, y, Inches(0.46), Inches(0.46))
    circ.fill.solid(); circ.fill.fore_color.rgb = PURPLE; circ.line.fill.background()
    tf = circ.text_frame; tf.margin_left = tf.margin_right = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = str(k + 1)
    r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = WHITE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    text(s, L + Inches(0.7), y + Inches(0.06), Inches(8.4), Inches(0.5), [lesson], size=14,
         anchor=MSO_ANCHOR.MIDDLE)

# ── 21. next ────────────────────────────────────────────────────────────────
s = new_slide("Next",
              "Running now: the production BP model with the supervised model's input path, square root "
              "of the intensities plus Fourier features at 12 frequencies. Our model uses log1p, which "
              "squeezes a flare core (~1e4) and a bright active region (~500) to 9.2 vs 6.2; the square "
              "root keeps them at 100 vs 22. Success: flare-core 94 A fit rising from 0.24 toward 0.92 "
              "without losing the faint half. If it is not enough, show the model flare-core pixels more "
              "often or weight them more. Either way, report by brightness band plus the median.")
cols = [("Running now", ["Production model with the supervised input path: √intensity + Fourier features.",
                         "Target: flare-core 94 Å fit from 0.24 toward 0.92."]),
        ("Then", ["If needed, oversample or up-weight flare-core pixels.",
                  "Report by brightness band plus the median."]),
        ("For discussion", ["Should the headline stay plain MSE?",
                            "Do flare cores matter for the science use?"])]
cw3 = Inches(2.9)
for k, (head, lines) in enumerate(cols):
    x = L + k * (cw3 + Inches(0.25))
    card(s, x, TOP, cw3, Inches(2.5), fill=PURPLE if k == 0 else CARD)
    text(s, x + Inches(0.22), TOP + Inches(0.2), cw3 - Inches(0.44), Inches(0.45), [head],
         size=15, bold=True, color=WHITE if k == 0 else PURPLE)
    text(s, x + Inches(0.22), TOP + Inches(0.75), cw3 - Inches(0.44), Inches(2.4), lines,
         size=12, color=WHITE if k == 0 else DARK, bullets=True, space_after=10)

out = os.path.join(REPO, "slides/DEM_advisor_deck.pptx")
prs.save(out)
print("saved", out, "slides:", len(prs.slides))
