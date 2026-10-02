"""Turn tall stacks of per-pixel panels into landscape strips for slides."""
import numpy as np
from PIL import Image

def panels(path, min_gap=6, min_h=60):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    blank = (a.min(axis=2) > 245).mean(axis=1) > 0.995       # near-white rows
    segs, start = [], None
    for y, b in enumerate(blank):
        if not b and start is None:
            start = y
        elif b and start is not None:
            segs.append([start, y]); start = None
    if start is not None:
        segs.append([start, len(blank)])
    merged = []                                               # join title rows to their plot
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return im, [s for s in merged if s[1] - s[0] >= min_h]

def strip(path, picks, out, pad=12, trim_cols=True):
    im, segs = panels(path)
    crops = [im.crop((0, max(segs[i][0] - 4, 0), im.width, min(segs[i][1] + 4, im.height))) for i in picks]
    h = max(c.height for c in crops)
    canvas = Image.new("RGB", (sum(c.width for c in crops) + pad * (len(crops) - 1), h), "white")
    x = 0
    for c in crops:
        canvas.paste(c, (x, 0)); x += c.width + pad
    canvas.save(out)
    return len(segs), canvas.size

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        im, segs = panels(p)
        print(p.split("/")[-1], im.size, "panels:", len(segs), [s[1] - s[0] for s in segs][:12])
