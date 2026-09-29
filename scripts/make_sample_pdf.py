#!/usr/bin/env python3
"""Generate the sample PDF used by the acceptance run.

Two pages: a heading, two body paragraphs and a ruled 3x3 table, so the
pipeline's title detection, table extraction and multi-page assembly are all
exercised.

Usage: make_sample_pdf.py <output.pdf>
"""
from __future__ import annotations

import sys

import pymupdf


def build(path: str) -> None:
    doc = pymupdf.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "Freerag Acceptance Report", fontsize=22)
    page.insert_textbox(
        pymupdf.Rect(72, 120, 520, 190),
        "This is the first body paragraph. It is long enough to be treated as body "
        "text rather than a heading, and it is deliberately left without a closing "
        "marker so a following fragment may be appended to it.",
        fontsize=11,
    )
    page.insert_textbox(
        pymupdf.Rect(72, 200, 520, 250),
        "The second paragraph is separate and closes properly. It gives the parser "
        "more than one body block to order.",
        fontsize=11,
    )

    x0, y0, col_w, row_h, rows, cols = 72.0, 300.0, 120.0, 24.0, 3, 3
    for i in range(rows + 1):
        y = y0 + i * row_h
        page.draw_line((x0, y), (x0 + cols * col_w, y), width=0.8)
    for j in range(cols + 1):
        x = x0 + j * col_w
        page.draw_line((x, y0), (x, y0 + rows * row_h), width=0.8)
    for i in range(rows):
        for j in range(cols):
            page.insert_text((x0 + j * col_w + 6, y0 + i * row_h + 16), f"r{i}c{j}", fontsize=10)

    second = doc.new_page(width=595, height=842)
    second.insert_textbox(
        pymupdf.Rect(72, 90, 520, 160),
        "Page two carries a single paragraph of body text so multi-page assembly and "
        "page numbering can be verified end to end.",
        fontsize=11,
    )

    doc.save(path)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: make_sample_pdf.py <output.pdf>", file=sys.stderr)
        raise SystemExit(2)
    build(sys.argv[1])
    print(f"wrote {sys.argv[1]}")
