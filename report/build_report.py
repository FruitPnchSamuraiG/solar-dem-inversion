#!/usr/bin/env python3
"""Build the written report from report/sections/*.md.

    uv run --with markdown python report/build_report.py

Produces report/report.html (one self-contained page, figures inlined) and
report/report.docx. Figures are referenced in the sections as
``![caption](figures/<name>.png)``; the file is looked up under report/figures/
first and then by basename anywhere under results/plots/, and copied into
report/figures/ so the report folder is self-contained.
"""
import base64
import glob
import html
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
SECTIONS = os.path.join(REPORT, "sections")
FIGURES = os.path.join(REPORT, "figures")
TITLE = "Label-free DEM inversion: training a per-pixel network on the solver's objective"

CSS = """
:root { --fg: #1b1b1f; --bg: #ffffff; --muted: #5a5a66; --rule: #d9d9e0; --accent: #57068c; --soft: #f4f1f8; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8e8ee; --bg: #15151a; --muted: #a5a5b3; --rule: #33333d; --accent: #c9a6e6; --soft: #22202a; } }
html { background: var(--bg); color: var(--fg); }
body { max-width: 52rem; margin: 0 auto; padding: 2rem 1rem 6rem; font: 16px/1.55 Georgia, 'Times New Roman', serif; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .25rem; }
h2 { font-size: 1.35rem; margin: 2.4rem 0 .6rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.1rem; margin: 1.6rem 0 .4rem; }
h4 { font-size: 1rem; margin: 1.2rem 0 .3rem; color: var(--muted); }
p { margin: 0 0 .9rem; }
.meta { color: var(--muted); margin-bottom: 2rem; font-family: system-ui, sans-serif; font-size: .9rem; }
table { border-collapse: collapse; margin: .8rem 0 1.2rem; font-family: system-ui, sans-serif; font-size: .86rem; max-width: 100%; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: .3rem .6rem; text-align: left; vertical-align: top; white-space: nowrap; }
th { border-bottom: 2px solid var(--accent); }
tr:hover td { background: var(--soft); }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto .3rem; }
figcaption, .caption { color: var(--muted); font-family: system-ui, sans-serif; font-size: .85rem; margin-bottom: 1.2rem; }
code { font-size: .9em; background: var(--soft); padding: 0 .25em; border-radius: 3px; }
pre code { display: block; padding: .8rem; overflow-x: auto; }
.sources { color: var(--muted); font-family: system-ui, sans-serif; font-size: .82rem; }
.toc { font-family: system-ui, sans-serif; font-size: .9rem; background: var(--soft); padding: 1rem 1.4rem; border-radius: 6px; }
.toc ul { margin: .2rem 0; padding-left: 1.2rem; }
blockquote { border-left: 3px solid var(--accent); margin: 1rem 0; padding: .2rem 1rem; color: var(--muted); }
@media print { body { font-size: 11pt; } h2 { page-break-after: avoid; } img { page-break-inside: avoid; } }
"""


def section_files():
    return sorted(glob.glob(os.path.join(SECTIONS, "*.md")))


def find_figure(name, alt=""):
    """The caption names the source folder (style rule); honour it, since several
    result folders hold figures with the same basename."""
    local = os.path.join(FIGURES, name)
    named = re.search(r"(results/plots/[^\s)]+/)" + re.escape(name), alt)
    if named:
        src = os.path.join(ROOT, named.group(1), name)
        if os.path.exists(src):
            os.makedirs(FIGURES, exist_ok=True)
            if not (os.path.exists(local) and os.path.getsize(local) == os.path.getsize(src)):
                shutil.copy(src, local)
            return local
    if os.path.exists(local):
        return local
    hits = sorted(glob.glob(os.path.join(ROOT, "results", "plots", "**", name), recursive=True))
    if hits:
        os.makedirs(FIGURES, exist_ok=True)
        shutil.copy(hits[-1], local)   # the latest folder wins when the caption is silent
        return local
    return None


def read_sections():
    text = []
    missing = []
    for path in section_files():
        with open(path) as f:
            body = f.read()

        def repl(m):
            alt, src = m.group(1), m.group(2)
            name = os.path.basename(src)
            found = find_figure(name, alt)
            if not found:
                missing.append((os.path.basename(path), name))
                return f"*[[missing figure: {name}]]*"
            return f"![{alt}](figures/{name})"
        body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, body)
        text.append(body.strip() + "\n")
    return "\n\n".join(text), missing


def to_html(md_text, meta):
    import markdown
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = md.convert(md_text)

    def inline(m):
        src = m.group(1)
        path = os.path.join(REPORT, src)
        if not os.path.exists(path):
            return m.group(0)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = "image/svg+xml" if ext == "svg" else f"image/{'jpeg' if ext == 'jpg' else ext}"
        return f'src="data:{mime};base64,{data}"'
    body = re.sub(r'src="(figures/[^"]+)"', inline, body)
    # figure captions: a paragraph holding only an image becomes a figure
    body = re.sub(r"<p>(<img [^>]*alt=\"([^\"]*)\"[^>]*>)</p>",
                  lambda m: f"<figure>{m.group(1)}<figcaption>{m.group(2)}</figcaption></figure>", body)
    body = re.sub(r"<p>Sources:", '<p class="sources">Sources:', body)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(TITLE)}</title><style>{CSS}</style></head>
<body><h1>{html.escape(TITLE)}</h1><div class="meta">{html.escape(meta)}</div>
<div class="toc">{md.toc}</div>
{body}
</body></html>
"""


# ── docx (a small markdown subset: headings, paragraphs, lists, tables, images) ──

def add_runs(par, text):
    """Bold (**x**), italics (*x*) and code (`x`) inline."""
    pos = 0
    for m in re.finditer(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if m.start() > pos:
            par.add_run(text[pos:m.start()])
        tok = m.group(0)
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
        else:
            par.add_run(tok[1:-1]).italic = True
        pos = m.end()
    if pos < len(text):
        par.add_run(text[pos:])


def to_docx(md_text, meta, out):
    from docx import Document
    from docx.shared import Inches, Pt
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Georgia"; style.font.size = Pt(10.5)
    doc.add_heading(TITLE, level=0)
    doc.add_paragraph(meta)
    lines = md_text.split("\n")
    i = 0
    para = []

    def flush():
        if para:
            add_runs(doc.add_paragraph(), " ".join(s.strip() for s in para))
            para.clear()
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            doc.add_heading(s.lstrip("#").strip(), level=min(level, 4)); i += 1; continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            flush()
            path = os.path.join(REPORT, m.group(2))
            if os.path.exists(path):
                doc.add_picture(path, width=Inches(6.0))
            cap = doc.add_paragraph(); add_runs(cap, m.group(1)); cap.runs and setattr(cap.runs[0].font, "size", Pt(9))
            i += 1; continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                t = doc.add_table(rows=len(rows), cols=ncol); t.style = "Light Grid Accent 1"
                for r, row in enumerate(rows):
                    for c in range(ncol):
                        cell = t.cell(r, c); cell.text = ""
                        add_runs(cell.paragraphs[0], row[c] if c < len(row) else "")
                        for run in cell.paragraphs[0].runs:
                            run.font.size = Pt(8.5)
                            if r == 0: run.bold = True
                doc.add_paragraph()
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if m:
            flush()
            kind = "List Number" if m.group(2)[0].isdigit() else "List Bullet"
            add_runs(doc.add_paragraph(style=kind), m.group(3)); i += 1; continue
        if s.startswith(">"):
            flush(); add_runs(doc.add_paragraph(style="Intense Quote"), s.lstrip("> ")); i += 1; continue
        if s.startswith("```"):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            p = doc.add_paragraph(); r = p.add_run("\n".join(code)); r.font.name = "Consolas"; r.font.size = Pt(8.5)
            i += 1; continue
        para.append(line); i += 1
    flush()
    doc.save(out)


def main():
    import datetime
    md_text, missing = read_sections()
    if not md_text.strip():
        sys.exit("no sections under report/sections/")
    meta = f"Built {datetime.date.today().isoformat()} from {len(section_files())} section files."
    html_out = os.path.join(REPORT, "report.html")
    with open(html_out, "w") as f:
        f.write(to_html(md_text, meta))
    docx_out = os.path.join(REPORT, "report.docx")
    try:
        to_docx(md_text, meta, docx_out)
    except Exception as e:  # python-docx is optional for the page
        print(f"docx skipped: {e}")
        docx_out = None
    words = len(re.findall(r"\w+", md_text))
    print(f"wrote {html_out}" + (f" and {docx_out}" if docx_out else "") + f" ({words} words)")
    for sec, name in missing:
        print(f"  missing figure in {sec}: {name}")
    todo = re.findall(r"\[\[[^\]]+\]\]", md_text)
    if todo:
        print(f"  {len(todo)} open markers: " + "; ".join(sorted(set(todo))[:12]))


if __name__ == "__main__":
    main()
