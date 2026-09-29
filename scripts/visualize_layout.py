#!/usr/bin/env python3
"""Visualise PP-DocLayout output on a PDF (docs/plan.md §5.1 step ②).

Each page is rendered twice — the page with the detected regions drawn on top,
and a blank canvas showing the regions alone — and both are embedded in a
self-contained HTML report with per-page counts and timings.

Drawing and rasterising use PyMuPDF's own API, so this adds no imaging
dependency beyond what the parser already needs.

Usage:
    python3 scripts/visualize_layout.py <pdf> [--pages 1-6] [--dpi 110] [--out DIR]
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sidecar"))

import pymupdf  # noqa: E402

from layout_onnx import BLOCK_COLOURS, LayoutDetector  # noqa: E402

GREY = (120, 120, 120)


def parse_pages(spec: str, total: int) -> List[int]:
    """Parse ``"1-5,8"`` into zero-based page indices, clamped to the document."""
    if not spec:
        return list(range(total))

    picked: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            low, high = int(start), int(end or start)
        else:
            low = high = int(chunk)
        for number in range(low, high + 1):
            if 1 <= number <= total and (number - 1) not in picked:
                picked.append(number - 1)
    return picked


def _rgb(block_type: str) -> Tuple[int, int, int]:
    return BLOCK_COLOURS.get(block_type, GREY)


def _colour(block_type: str) -> Tuple[float, float, float]:
    red, green, blue = _rgb(block_type)
    return (red / 255.0, green / 255.0, blue / 255.0)


def draw_detections(page: Any, detections: Sequence[Dict[str, Any]], fontsize: float = 7.0) -> None:
    """Draw one box + label per detection; coordinates are PDF points."""
    for detection in detections:
        x0, y0, x1, y1 = (float(v) for v in detection["bbox"])
        colour = _colour(detection["block_type"])
        page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=colour, width=1.2)

        label = "%s %.2f" % (detection["block_type"], detection["score"])
        width = len(label) * fontsize * 0.52
        height = fontsize + 3.0
        # Sit the tag inside the box's top-left, or just above it when the box is
        # too short to hold the text.
        tag_y0 = y0 if y0 + height <= y1 else y0 - height
        tag_y0 = max(0.0, min(tag_y0, page.rect.height - height))
        tag = pymupdf.Rect(x0, tag_y0, x0 + width, tag_y0 + height)

        page.draw_rect(tag, color=colour, fill=colour, width=0)
        page.insert_text(
            pymupdf.Point(tag.x0 + 1.5, tag.y1 - 2.2),
            label,
            fontsize=fontsize,
            fontname="helv",
            color=(1, 1, 1),
        )


def render_png(page: Any, dpi: int) -> bytes:
    """Rasterise a page to PNG bytes."""
    return page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def build_report(
    pdf_path: str,
    pages: List[int],
    results: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    dpi: int,
) -> str:
    totals: Dict[str, int] = {}
    for entry in results:
        for detection in entry["detections"]:
            totals[detection["block_type"]] = totals.get(detection["block_type"], 0) + 1

    legend = "".join(
        '<span class="chip"><i style="background:rgb(%d,%d,%d)"></i>%s</span>'
        % (_rgb(name) + (name,))
        for name in sorted(totals)
    )
    summary = "".join(
        "<tr><td><i class='dot' style='background:rgb(%d,%d,%d)'></i>%s</td><td>%d</td></tr>"
        % (_rgb(name) + (name, totals[name]))
        for name in sorted(totals)
    )

    sections = []
    for entry in results:
        counts: Dict[str, int] = {}
        rows = []
        for detection in entry["detections"]:
            counts[detection["block_type"]] = counts.get(detection["block_type"], 0) + 1
            x0, y0, x1, y1 = (round(float(v)) for v in detection["bbox"])
            rows.append(
                "<tr><td><i class='dot' style='background:rgb(%d,%d,%d)'></i>%s</td>"
                "<td>%.2f</td><td>%d, %d, %d, %d</td></tr>"
                % (_rgb(detection["block_type"])
                   + (detection["block_type"], detection["score"], x0, y0, x1, y1))
            )
        chips = " ".join("%s × %d" % (name, count) for name, count in sorted(counts.items()))
        sections.append(
            """
  <section>
    <h2>第 %(page)d 页 <small>%(count)d 个区域 · %(seconds).2fs · %(chips)s</small></h2>
    <div class="panes">
      <figure><figcaption>叠加在原页上</figcaption>
        <img src="%(overlay)s" alt="page %(page)d with boxes"></figure>
      <figure><figcaption>仅检测框（白底）</figcaption>
        <img src="%(boxes)s" alt="page %(page)d boxes only"></figure>
    </div>
    <details>
      <summary>区域明细</summary>
      <table>
        <thead><tr><th>类型</th><th>score</th><th>x0, y0, x1, y1 (pt)</th></tr></thead>
        <tbody>%(rows)s</tbody>
      </table>
    </details>
  </section>"""
            % {
                "page": entry["page"],
                "count": len(entry["detections"]),
                "seconds": entry["seconds"],
                "chips": chips,
                "overlay": data_uri(entry["overlay"]),
                "boxes": data_uri(entry["boxes"]),
                "rows": "".join(rows),
            }
        )

    return """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>PP-DocLayout 版面分析 · %(pdf)s</title>
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0 auto; max-width: 1560px; padding: 28px 24px 64px; color: #1c2330;
         background: #f6f7f9; }
  h1 { font-size: 21px; margin: 0 0 6px; }
  .meta { color: #5a6472; font-size: 13px; line-height: 1.95; margin-bottom: 12px; }
  .meta code { background: #e8ebf0; padding: 1px 5px; border-radius: 3px; }
  .chip { display: inline-flex; align-items: center; gap: 5px; margin-right: 14px;
          font-size: 12px; color: #3a4453; }
  .chip i { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  section { background: #fff; border: 1px solid #e2e6ec; border-radius: 9px;
            padding: 16px 18px; margin: 18px 0; }
  h2 { font-size: 15px; margin: 0 0 12px; }
  h2 small { color: #6b7484; font-weight: 400; margin-left: 8px; font-size: 12px; }
  .panes { display: flex; gap: 14px; align-items: flex-start; }
  .panes figure { margin: 0; flex: 1 1 50%%; min-width: 0; }
  .panes figcaption { font-size: 12px; color: #6b7484; margin-bottom: 5px; }
  .panes img { width: 100%%; border: 1px solid #dfe3e9; border-radius: 5px; display: block; }
  summary { cursor: pointer; font-size: 13px; color: #3355aa; margin-top: 12px; }
  table { border-collapse: collapse; width: 100%%; margin-top: 8px; font-size: 12px; }
  th, td { border-bottom: 1px solid #eceff3; padding: 4px 8px; text-align: left; }
  th { background: #f2f4f8; }
  .dot { width: 9px; height: 9px; border-radius: 50%%; display: inline-block; margin-right: 6px; }
  .totals { background: #fff; border: 1px solid #e2e6ec; border-radius: 9px; padding: 14px 18px; }
</style>
</head>
<body>
  <h1>PP-DocLayout 版面分析结果</h1>
  <div class="meta">
    文件：<code>%(pdf)s</code><br>
    模型：<code>%(model)s</code> · 推理后端：<code>%(provider)s</code><br>
    渲染：<code>%(dpi)d dpi</code> · 覆盖页：<code>%(pages)s</code>
  </div>
  <div class="meta">%(legend)s</div>
  <div class="totals">
    <strong>所选页面区域统计</strong>
    <table><thead><tr><th>类型</th><th>数量</th></tr></thead><tbody>%(summary)s</tbody></table>
  </div>
  %(sections)s
</body>
</html>
""" % {
        "pdf": os.path.basename(pdf_path),
        "model": model,
        "provider": provider,
        "dpi": dpi,
        "pages": ", ".join(str(p + 1) for p in pages),
        "legend": legend,
        "summary": summary,
        "sections": "".join(sections),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualise PP-DocLayout output on a PDF.")
    parser.add_argument("pdf")
    parser.add_argument("--pages", default="", help="e.g. 1-6 or 1,3,5 (default: all)")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--out", default="")
    parser.add_argument("--font-size", type=float, default=7.0)
    args = parser.parse_args()

    detector = LayoutDetector()
    if not detector.available:
        print("layout model not found: %s" % detector.path, file=sys.stderr)
        return 2

    document = pymupdf.open(args.pdf)
    pages = parse_pages(args.pages, len(document))

    out_dir = args.out or os.path.join(
        "artifacts", "layout", os.path.splitext(os.path.basename(args.pdf))[0]
    )
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for index in pages:
        page = document[index]

        started = time.time()
        detections = detector.detect_page(page, dpi=args.dpi)
        elapsed = time.time() - started

        draw_detections(page, detections, fontsize=args.font_size)
        overlay = render_png(page, args.dpi)

        # A blank page of the same size isolates the geometry from the content.
        blank_document = pymupdf.open()
        blank = blank_document.new_page(width=page.rect.width, height=page.rect.height)
        draw_detections(blank, detections, fontsize=args.font_size)
        boxes_only = render_png(blank, args.dpi)
        blank_document.close()

        for suffix, payload in (("", overlay), ("-boxes", boxes_only)):
            with open(os.path.join(out_dir, "page-%02d%s.png" % (index + 1, suffix)), "wb") as handle:
                handle.write(payload)

        results.append(
            {"page": index + 1, "seconds": elapsed, "detections": detections,
             "overlay": overlay, "boxes": boxes_only}
        )
        print("page %2d: %2d regions, %.2fs" % (index + 1, len(detections), elapsed))

    report = build_report(
        args.pdf,
        pages,
        results,
        provider=detector.provider_name,
        model=os.path.basename(detector.path),
        dpi=args.dpi,
    )
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    document.close()
    print("\nreport: %s" % os.path.abspath(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
