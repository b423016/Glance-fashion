"""Build the single submission PDF from design.md + approaches.md, with the
result boards embedded as base64 (so they render inside the PDF, not as broken
paths). HTML -> PDF via headless Chrome.

Run:  .venv/bin/python docs/build_pdf.py
"""
from __future__ import annotations

import base64
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

BOARDS = [
    ('"white top + black pants" - GLOBAL baseline (mixes in wrong-colour tops and black shorts)',
     "S4_white_top_black_pants__global.png"),
    ('"white top + black pants" - FULL pipeline (clean; P@8 0.75 to 1.00)',
     "S4_white_top_black_pants__full.png"),
    ('"red tie + white shirt" - FULL (the marquee compositional query; supplement gives it real positives)',
     "Q5_red_tie_white_shirt__full.png"),
    ('"business attire in an office" - FULL (scene-driven)',
     "Q2_office_business__full.png"),
]

CSS = """
@page { size: A4; margin: 15mm 13mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 11px; line-height: 1.45; color: #111; }
h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 15px; margin-top: 18px; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 12.5px; margin-top: 12px; }
p, li { orphans: 2; widows: 2; }
code { font-family: ui-monospace, Menlo, monospace; font-size: 10px; background:#f2f2f2; padding:0 2px; }
pre { background:#f6f6f6; padding:8px; border-radius:5px; font-size:9.5px; white-space:pre; overflow-x:auto; }
pre code { background:none; }
table { border-collapse: collapse; width: 100%; font-size: 10px; margin: 6px 0; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left; }
th { background: #f0f0f0; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; margin: 3px 0 10px; page-break-inside: avoid; }
em { color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 16px 0; }
"""


def build():
    design = (DOCS / "design.md").read_text()
    approaches = (DOCS / "approaches.md").read_text()

    board_md = "\n\n### Result boards - global (baseline) vs full (my pipeline)\n\n" + "\n\n".join(
        f"*{cap}*\n\n![]({ROOT / 'outputs' / fn})" for cap, fn in BOARDS)
    design = design.replace("## 5. Honest limitations", board_md + "\n\n## 5. Honest limitations", 1)

    full_md = ("# Multimodal Fashion & Context Retrieval - Write-up\n\n"
               "*Glance ML assignment - compositional text-to-image fashion retrieval that beats "
               "vanilla CLIP - GitHub: https://github.com/b423016/Glance-fashion*\n\n"
               + design + "\n\n---\n\n" + approaches)

    html_body = markdown.markdown(full_md, extensions=["tables", "fenced_code"])

    def inline(m):
        src = m.group(1)
        p = pathlib.Path(src)
        if not p.is_absolute():
            p = ROOT / src
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)

    html_body = re.sub(r'src="([^"]+)"', inline, html_body)
    head = f"<meta charset='utf-8'><style>{CSS}</style>"
    html = f"<!doctype html><html><head>{head}</head><body>{html_body}</body></html>"
    (DOCS / "writeup.html").write_text(html)
    print("wrote writeup.html", len(html), "bytes")


if __name__ == "__main__":
    build()
