"""Layout block detection.

The production provider is PP-DocLayout-S (plan.md §5.1 step ②). This module
ships a PyMuPDF-based provider used whenever PP-DocLayout is unavailable, so the
pipeline runs end to end without the detector model — the same seam lets the
real detector drop in later without touching the pipeline.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Sequence, Tuple

from chunking import Block

#: A block whose glyphs are this much larger than the page median is a heading.
TITLE_SIZE_RATIO = 1.15
#: Headings longer than this are body text that happens to be large.
TITLE_MAX_CHARS = 80
#: Fraction of a text block's area that must sit inside a table to drop it.
TABLE_COVER_RATIO = 0.6


def _median_font_size(page_dict: Dict[str, Any]) -> float:
    sizes = [
        span.get("size", 0.0)
        for block in page_dict.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    sizes = [s for s in sizes if s > 0]
    return statistics.median(sizes) if sizes else 0.0


def _block_text(block: Dict[str, Any]) -> str:
    lines: List[str] = []
    for line in block.get("lines", []):
        text = "".join(span.get("text", "") for span in line.get("spans", []))
        if text.strip():
            lines.append(text.rstrip())
    return "\n".join(lines)


def _max_span_size(block: Dict[str, Any]) -> float:
    return max(
        (span.get("size", 0.0) for line in block.get("lines", []) for span in line.get("spans", [])),
        default=0.0,
    )


def _max_size_in_bbox(page_dict: Dict[str, Any], bbox: Sequence[float]) -> float:
    """Largest glyph size among the text blocks that lie inside one region.

    The layout detector says a region is a title; the font size says how big the
    letters in it actually were. Without this the ONNX path emits headings with
    no size at all, and every PDF parsed through it arrives at the tree builder
    flat — heading levels that the document plainly has, silently gone.
    """
    best = 0.0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = block.get("bbox")
        if not block_bbox:
            continue
        # Half the block's area inside the region counts as belonging to it: a
        # title region bounded mid-line by the detector is still the title's own
        # block, while a body paragraph merely touching the region is not.
        if _overlap_ratio(tuple(block_bbox), tuple(bbox)) < 0.5:
            continue
        size = _max_span_size(block)
        if size > best:
            best = size
    return best


def _overlap_ratio(bbox: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``bbox``'s area that lies inside ``outer``."""
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if width <= 0 or height <= 0:
        return 0.0
    ix = max(0.0, min(bbox[2], outer[2]) - max(bbox[0], outer[0]))
    iy = max(0.0, min(bbox[3], outer[3]) - max(bbox[1], outer[1]))
    return (ix * iy) / (width * height)


def _mostly_inside(bbox: Sequence[float], boxes: Sequence[Sequence[float]]) -> bool:
    return any(_overlap_ratio(bbox, box) >= TABLE_COVER_RATIO for box in boxes)


def table_to_markdown(table: Any) -> str:
    """Render an extracted table as a Markdown table (plan.md §5.2 output form)."""
    rows = table.extract()
    if not rows:
        return ""
    cleaned: List[List[str]] = [
        ["" if cell is None else " ".join(str(cell).split()) for cell in row] for row in rows
    ]
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    lines = [
        "| " + " | ".join(cleaned[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def text_in_bbox(page: Any, bbox: Sequence[float]) -> str:
    """Extract the PDF text layer inside a rectangle, grouped into lines.

    ``get_text("words")`` carries each word's ``(block_no, line_no, word_no)``,
    so lines are rebuilt from those indices rather than guessed from
    coordinates. A rectangle extending past the page is clipped by PyMuPDF, so an
    oversized detector box degrades to "the whole page" instead of raising.
    """
    try:
        words = page.get_text("words", clip=tuple(float(v) for v in bbox))
    except Exception:
        return ""
    if not words:
        return ""

    lines: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for word in words:
        # (x0, y0, x1, y1, text, block_no, line_no, word_no)
        text = word[4]
        if not str(text).strip():
            continue
        lines.setdefault((int(word[5]), int(word[6])), []).append((int(word[7]), str(text)))

    ordered = [" ".join(text for _, text in sorted(lines[key])) for key in sorted(lines)]
    return "\n".join(ordered).strip()


def _page_tables(page: Any) -> List[Tuple[Sequence[float], Any]]:
    """Extract every table on the page once — ``find_tables`` is not free."""
    try:
        return [(tuple(float(v) for v in table.bbox), table) for table in page.find_tables().tables]
    except Exception:
        return []


def _best_table_markdown(bbox: Sequence[float], tables: Sequence[Tuple[Sequence[float], Any]]) -> str:
    """Markdown for the extracted table that best overlaps a detected region."""
    best: Any = None
    best_ratio = 0.0
    for table_bbox, table in tables:
        ratio = _overlap_ratio(bbox, table_bbox)
        if ratio > best_ratio:
            best, best_ratio = table, ratio
    if best is None or best_ratio < TABLE_COVER_RATIO:
        return ""
    return table_to_markdown(best)


#: Sentinel so "not resolved yet" is distinguishable from "resolved to nothing".
_UNSET = object()
_TSR_DETECTOR: Any = _UNSET


def _tsr_detector() -> Any:
    """The process-wide TSR detector, or None when the model file is absent.

    Built once per process: loading the ONNX session costs far more than the
    ~50 ms per table the detector then takes.
    """
    global _TSR_DETECTOR
    if _TSR_DETECTOR is _UNSET:
        _TSR_DETECTOR = None
        try:
            from tsr_onnx import TSRDetector

            candidate = TSRDetector()
            if candidate.available:
                _TSR_DETECTOR = candidate
        except Exception:
            # A missing or unloadable model degrades to the PyMuPDF path rather
            # than failing the parse; the reply still names the provider.
            _TSR_DETECTOR = None
    return _TSR_DETECTOR


def _tsr_table_markdown(page: Any, bbox: Sequence[float], dpi: int) -> str:
    """Structure a table region with TSR, reading each cell from the text layer.

    Returns "" when TSR finds no usable grid, so the caller falls back. TSR is
    preferred over ``find_tables()`` because it *names* rows and columns instead
    of inferring them from ruling lines: measured on a real 38-page paper, ruling
    lines recovered 1 of 10 tables and flattened the rest into cell-text soup.

    Cell text still comes from the PDF text layer (never OCR), so a scanned table
    needs the OCR fallback regardless of how good the structure detection is.
    """
    import numpy as np
    import pymupdf

    from table_grid import render as render_grid

    detector = _tsr_detector()
    if detector is None:
        return ""

    rect = pymupdf.Rect(*(float(v) for v in bbox))
    if rect.is_empty:
        return ""

    pixmap = page.get_pixmap(dpi=dpi, clip=rect, colorspace=pymupdf.csRGB, alpha=False)
    crop = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    boxes = detector.detect(crop)
    if not boxes:
        return ""

    # TSR runs on the crop, so its boxes are crop pixels while the page speaks
    # points. The conversion lives in the reader the grid calls, applied exactly
    # once — doing it earlier would mean converting rectangles the grid then
    # compares against each other on a different scale.
    scale = dpi / 72.0

    def read_cell(cell: Tuple[float, float, float, float]) -> str:
        x0, top, x1, bottom = cell
        return text_in_bbox(
            page,
            (rect.x0 + x0 / scale, rect.y0 + top / scale,
             rect.x0 + x1 / scale, rect.y0 + bottom / scale),
        )

    return render_grid(boxes, read_cell)["markdown"]


def detect_blocks_layout(
    page: Any,
    page_num: int,
    source_file: str = "",
    detector: Any = None,
    dpi: int = 150,
) -> List[Block]:
    """PP-DocLayout provider: detect regions, then read each region's text.

    Text always comes from the PDF text layer, never OCR — the detector decides
    *where* things are, PyMuPDF decides *what they say* (plan.md §5.1: OCR is a
    scan-only fallback). Regions with no text layer (figures, images) are dropped
    here; the visualiser reads the detector directly and shows them anyway.

    Tables keep the PyMuPDF provider's Markdown rendering (§5.2): the detector
    says "this region is a table", but the row/column structure still has to come
    from ``find_tables``. Without this the ONNX path would silently downgrade a
    table to a flat run of cell text.
    """
    from layout_onnx import LayoutDetector  # local: keeps pymupdf/onnx optional

    detector = detector if detector is not None else LayoutDetector()
    tables = _page_tables(page)
    page_dict = page.get_text("dict")

    blocks: List[Block] = []
    for region in detector.detect_page(page, dpi=dpi):
        bbox = tuple(float(v) for v in region["bbox"])
        if region["block_type"] == "Table":
            # Ruling lines first, TSR second, plain text last.
            #
            # The order is by precision, not by sophistication: where a table is
            # drawn with ruling lines, find_tables() reads them exactly, and TSR
            # measurably *worse* — on the 3x3 fixture it reports phantom blank
            # bands between the real rows. TSR exists for the tables ruling lines
            # cannot see at all, which on a real paper is most of them.
            text = (
                _best_table_markdown(bbox, tables)
                or _tsr_table_markdown(page, bbox, dpi)
                or text_in_bbox(page, bbox)
            )
        else:
            text = text_in_bbox(page, bbox)
        if not text:
            continue
        blocks.append(
            Block(
                text=text,
                page_num=page_num,
                block_type=region["block_type"],
                bbox=bbox,
                font_size=round(_max_size_in_bbox(page_dict, bbox), 1),
                source_file=source_file,
            )
        )
    return blocks


def detect_blocks_pymupdf(page: Any, page_num: int, source_file: str = "") -> List[Block]:
    """Detect layout blocks on one page using PyMuPDF.

    Headings are inferred from font size; tables come from ``find_tables()`` and
    are rendered as Markdown so the row/column structure survives into the
    chunk text.
    """
    page_dict = page.get_text("dict")
    median = _median_font_size(page_dict)

    table_boxes: List[Tuple[float, ...]] = []
    table_blocks: List[Block] = []
    try:
        for table in page.find_tables().tables:
            text = table_to_markdown(table)
            if not text:
                continue
            bbox = tuple(float(v) for v in table.bbox)
            table_boxes.append(bbox)
            table_blocks.append(
                Block(
                    text=text,
                    page_num=page_num,
                    block_type="Table",
                    bbox=bbox,
                    source_file=source_file,
                )
            )
    except Exception:
        # find_tables is best-effort: a page it cannot parse is still read as text.
        table_boxes, table_blocks = [], []

    blocks: List[Block] = []
    for raw in page_dict.get("blocks", []):
        if raw.get("type") != 0:
            continue
        text = _block_text(raw)
        if not text.strip():
            continue
        bbox = tuple(float(v) for v in raw.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if _mostly_inside(bbox, table_boxes):
            continue  # the table block already carries this text

        size = _max_span_size(raw)
        block_type = "Text"
        if median > 0 and size >= median * TITLE_SIZE_RATIO and len(text.strip()) <= TITLE_MAX_CHARS:
            block_type = "Title"

        blocks.append(
            Block(
                text=text,
                page_num=page_num,
                block_type=block_type,
                bbox=bbox,
                font_size=round(size, 1),
                source_file=source_file,
            )
        )

    blocks.extend(table_blocks)
    return blocks
