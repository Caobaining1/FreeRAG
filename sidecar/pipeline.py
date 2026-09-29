"""Parse pipeline: PDF -> layout blocks -> chunks (plan.md §5.1).

    PDF
     -> 1. open / render (PyMuPDF opens the file; rendering only happens when a
           bitmap detector such as PP-DocLayout is configured)
     -> 2. layout detection (PP-DocLayout when available, else PyMuPDF)
     -> 3. content extraction per block (PyMuPDF text layer first)
     -> 4. semantic block assembly (page_num / block_type / bbox / parent_section)
     -> 5. chunking: one block = one chunk, short blocks folded in (§5.3/§5.3.1)
     -> 6. metadata -> return
"""
from __future__ import annotations

import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

from cache import ParseCache, cache_key, touch_marker
from chunking import DEFAULT_PROFILE, Block, consolidate_blocks, to_chunks
from layout import detect_blocks_layout, detect_blocks_pymupdf
from layout_onnx import LayoutDetector

#: PyMuPDF text blocks carry no reading order across columns; page order is the
#: best available proxy (TUIrag's pitfalls doc #3: trust the detector's order).
DEFAULT_DPI = 150

#: Layout provider selection: "onnx" forces PP-DocLayout, "pymupdf" forces the
#: fallback, "auto" uses PP-DocLayout when the model file is present.
DEFAULT_LAYOUT = "auto"


def make_detector(layout: str = DEFAULT_LAYOUT) -> Optional[LayoutDetector]:
    """Return a PP-DocLayout detector, or None to use the PyMuPDF provider."""
    if layout == "pymupdf":
        return None
    detector = LayoutDetector()
    if layout == "onnx" and not detector.available:
        raise ValueError(f"PP-DocLayout model not found: {detector.path}")
    return detector if detector.available else None


def _assign_sections(blocks: Sequence[Block]) -> List[Block]:
    """Attach each block to the most recent Title (plan.md §5.3: metadata, not merging)."""
    current = ""
    for block in blocks:
        if block.block_type == "Title":
            current = block.render()
            block.parent_section = ""
        else:
            block.parent_section = current
    return list(blocks)


def _mean_height(blocks: Sequence[Block]) -> float:
    """Median height of body-text blocks — the RAGFlow ``mean_height`` analogue."""
    heights = [b.height for b in blocks if b.block_type == "Text" and b.height > 0]
    return statistics.median(heights) if heights else 0.0


def parse_pdf(
    path: str,
    *,
    profile: str = DEFAULT_PROFILE,
    max_chars: Optional[int] = None,
    max_pages: int = 0,
    layout: str = DEFAULT_LAYOUT,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Parse one PDF into chunks. Raises FileNotFoundError / ValueError on bad input."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")

    cache = ParseCache() if use_cache else None
    key = ""
    if cache is not None:
        key = cache_key(
            path, profile=profile, max_chars=max_chars, max_pages=max_pages, layout=layout
        )
        hit = cache.get(key)
        if hit is not None:
            # Layout detection dominates a parse, so a hit is worth the whole
            # run. The stamp tells the caller which chunks were produced from
            # cache rather than from the document as it stands now.
            return touch_marker(hit)

    import pymupdf  # imported lazily so the module is importable without the wheel

    detector = make_detector(layout)
    source_file = os.path.basename(path)
    doc = pymupdf.open(path)
    try:
        if doc.needs_pass:
            raise ValueError("PDF is encrypted")

        total_pages = len(doc)
        page_limit = total_pages if max_pages <= 0 else min(total_pages, max_pages)

        blocks: List[Block] = []
        page_width = 0.0
        page_height = 0.0
        for index in range(page_limit):
            page = doc[index]
            page_width = max(page_width, float(page.rect.width))
            page_height = max(page_height, float(page.rect.height))
            if detector is not None:
                blocks.extend(detect_blocks_layout(page, index + 1, source_file, detector))
            else:
                blocks.extend(detect_blocks_pymupdf(page, index + 1, source_file=source_file))
    finally:
        doc.close()

    blocks = _assign_sections(blocks)
    mean_height = _mean_height(blocks)

    consolidated = consolidate_blocks(
        blocks,
        max_chars=max_chars,
        profile=profile,
        page_width=page_width,
        mean_height=mean_height,
        page_height=page_height,
    )
    chunks = to_chunks(consolidated)

    result = {
        "path": path,
        "source_file": source_file,
        "page_count": total_pages,
        "pages_parsed": page_limit,
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "layout_provider": "pp-doclayout" if detector is not None else "pymupdf",
        "profile": profile,
        "chunks": chunks,
    }
    if cache is not None:
        cache.put(key, result)
    return result
