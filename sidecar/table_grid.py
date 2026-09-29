"""Turn TSR structure boxes into a grid, then into Markdown.

Pure logic: no PDF or model dependency, and cell text arrives through an
injected callable. That keeps the arrangement rules — which are where the
mistakes live — directly unit-testable, the same way ``chunking.py`` is.

Why this exists: PyMuPDF's ``find_tables()`` only finds tables drawn with ruling
lines. Measured on a real 38-page paper, **1 of 10 tables** came back as
Markdown and the rest were flattened into cell-text soup. TSR names the rows and
columns explicitly, so the grid is an intersection rather than an inference.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

#: Box labels that describe a horizontal band.
ROW_LABELS = ("table row", "table projected row header")
#: Box label that describes a vertical band.
COLUMN_LABEL = "table column"
#: Box label marking the header band(s).
HEADER_LABEL = "table column header"

#: Two bands overlapping by more than this fraction of the smaller one are the
#: same band detected twice, not two adjacent bands.
BAND_MERGE_OVERLAP = 0.5


def _bands(boxes: Sequence[Dict[str, Any]], labels: Sequence[str], axis: str) -> List[Tuple[float, float]]:
    """Collect one axis's bands, merged and ordered.

    ``axis`` is ``"x"`` for columns (left/right) or ``"y"`` for rows (top/bottom).
    """
    lo, hi = (0, 2) if axis == "x" else (1, 3)
    raw = sorted(
        ((float(b["bbox"][lo]), float(b["bbox"][hi])) for b in boxes if b["label"] in labels),
        key=lambda band: band[0],
    )

    merged: List[List[float]] = []
    for start, end in raw:
        if merged and _overlap(merged[-1], (start, end)) > BAND_MERGE_OVERLAP:
            # Same band seen twice: widen rather than emit a duplicate row,
            # which would otherwise put a near-empty row in the output.
            merged[-1][0] = min(merged[-1][0], start)
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])
    return [(start, end) for start, end in merged]


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Intersection as a fraction of the smaller band."""
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection <= 0:
        return 0.0
    smaller = min(a[1] - a[0], b[1] - b[0])
    return intersection / smaller if smaller > 0 else 0.0


def row_bands(boxes: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Horizontal bands, top to bottom."""
    return _bands(boxes, ROW_LABELS, "y")


def column_bands(boxes: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Vertical bands, left to right, with overlaps split down the middle.

    TSR columns routinely overlap a few pixels. Left unhandled, one word would be
    clipped into two cells and appear twice, so each shared edge moves to the
    midpoint of the overlap.
    """
    bands = _bands(boxes, (COLUMN_LABEL,), "x")
    for index in range(len(bands) - 1):
        left_end = bands[index][1]
        right_start = bands[index + 1][0]
        if left_end > right_start:
            middle = (left_end + right_start) / 2.0
            bands[index] = (bands[index][0], middle)
            bands[index + 1] = (middle, bands[index + 1][1])
    return bands


def header_row_count(boxes: Sequence[Dict[str, Any]], rows: Sequence[Tuple[float, float]]) -> int:
    """How many leading rows the header band covers.

    Only leading rows count: a ``table column header`` further down would be a
    mid-table heading, and treating it as the Markdown header would silently
    promote a data row into the header line.
    """
    headers = [b for b in boxes if b["label"] == HEADER_LABEL]
    if not headers or not rows:
        return 0

    covered = 0
    for top, bottom in rows:
        overlapping = any(
            _overlap((float(h["bbox"][1]), float(h["bbox"][3])), (top, bottom)) > 0.5
            for h in headers
        )
        if not overlapping:
            break
        covered += 1
    return covered


def build_grid(
    boxes: Sequence[Dict[str, Any]],
    rows: Sequence[Tuple[float, float]],
    columns: Sequence[Tuple[float, float]],
    read_cell: Callable[[Tuple[float, float, float, float]], str],
) -> List[List[str]]:
    """Read one text value per row × column intersection.

    ``read_cell`` receives a rectangle on the **same axis** the bands use — the
    caller owns the mapping back to page coordinates.
    """
    grid: List[List[str]] = []
    for top, bottom in rows:
        row: List[str] = []
        for left, right in columns:
            row.append(collapse(read_cell((left, top, right, bottom))))
        grid.append(row)
    return grid


def collapse(text: str) -> str:
    """Flatten whitespace and escape pipes so a cell cannot break the table."""
    return " ".join(str(text).split()).replace("|", "\\|")


def to_markdown(grid: Sequence[Sequence[str]], header_rows: int) -> str:
    """Render the grid as a Markdown table.

    A multi-row header is folded into one line by joining each column's header
    cells, because Markdown has exactly one header row and dropping the extra
    ones would lose the distinctions that made them separate.
    """
    if not grid:
        return ""

    width = max(len(row) for row in grid)
    padded = [list(row) + [""] * (width - len(row)) for row in grid]

    header_rows = max(0, min(header_rows, len(padded)))
    if header_rows == 0:
        # No header band detected. A blank header keeps the shape valid Markdown
        # rather than promoting a data row and mislabelling it.
        header = [""] * width
        body = padded
    else:
        header = [
            " ".join(padded[r][c] for r in range(header_rows) if padded[r][c]).strip()
            for c in range(width)
        ]
        body = padded[header_rows:]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render(
    boxes: Sequence[Dict[str, Any]],
    read_cell: Callable[[Tuple[float, float, float, float]], str],
) -> Dict[str, Any]:
    """Full path: structure boxes plus a cell reader, out comes Markdown.

    Returns an empty ``markdown`` when the structure is too thin to be a table —
    a single band means TSR saw a line, not a grid, and emitting it would be worse
    than falling back to the region's plain text.
    """
    rows = row_bands(boxes)
    columns = column_bands(boxes)
    if len(rows) < 2 or len(columns) < 2:
        return {"markdown": "", "rows": len(rows), "columns": len(columns), "cells": 0}

    grid = build_grid(boxes, rows, columns, read_cell)
    cells = sum(1 for row in grid for cell in row if cell)
    return {
        "markdown": to_markdown(grid, header_row_count(boxes, rows)),
        "rows": len(rows),
        "columns": len(columns),
        "cells": cells,
    }
