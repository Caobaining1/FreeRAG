"""Parse pipeline: document -> layout blocks -> chunks (plan.md §5.1).

    document (PDF / DOCX / DOC / RTF / TXT / MD)
     -> 0. format dispatch: a PDF goes through layout detection, everything else
           through sidecar/documents.py (a reflowable file has no layout to
           detect, so the detector is skipped rather than run on nothing)
     -> 1. open / render (PyMuPDF opens the file; rendering only happens when a
           bitmap detector such as PP-DocLayout is configured)
     -> 2. layout detection (PP-DocLayout when available, else PyMuPDF)
     -> 3. content extraction per block (PyMuPDF text layer first)
     -> 4. semantic block assembly (page_num / block_type / bbox / parent_section)
     -> 5. chunking: filter noise, then merge blocks into chunks (§5.3/§5.3.1):
           TUIrag strategy B — title + body, buffered paragraphs, table/figure
           + caption — capped by a per-profile merge ceiling
     -> 6. metadata -> return
"""
from __future__ import annotations

import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

import documents
from cache import ParseCache, cache_key, touch_marker
from chunking import DEFAULT_PROFILE, Block, consolidate_blocks, to_chunks
from layout import detect_blocks_layout, detect_blocks_pymupdf
from layout_onnx import LayoutDetector
from vlm import VlmCaptioner, caption_figures

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


def _pdf_blocks(path: str, source_file: str, max_pages: int, layout: str) -> Dict[str, Any]:
    """Detect layout blocks page by page. PDFs only: needs a page and a text layer."""
    import pymupdf  # imported lazily so the module is importable without the wheel

    detector = make_detector(layout)
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

    return {
        "blocks": blocks,
        "page_count": total_pages,
        "pages_parsed": page_limit,
        "page_width": page_width,
        "page_height": page_height,
        "provider": "pp-doclayout" if detector is not None else "pymupdf",
    }


def _document_blocks(path: str, source_file: str) -> Dict[str, Any]:
    """Paragraphs from a text/Markdown/DOCX/DOC/RTF file.

    No pages and no boxes exist in these, so every block is page 1 with a zero
    bbox. That is not a placeholder: the chunker's geometry gate reads a
    zero-width box as "nothing to judge" and lets the run continue, which is
    exactly the behaviour a reflowable document wants.
    """
    paragraphs = documents.load_paragraphs(path)
    blocks = [
        Block(text=text, page_num=1, block_type=block_type, source_file=source_file)
        for block_type, text in paragraphs
    ]
    return {
        "blocks": blocks,
        # One page, because the file has no pages to number.
        "page_count": 1,
        "pages_parsed": 1,
        "page_width": 0.0,
        "page_height": 0.0,
        "provider": documents.suffix_of(path).lstrip('.') or "text",
    }


def parse_document(
    path: str,
    *,
    profile: str = DEFAULT_PROFILE,
    max_chars: Optional[int] = None,
    max_pages: int = 0,
    layout: str = DEFAULT_LAYOUT,
    vlm_model: str = "",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Parse one document into chunks, by format.

    ``vlm_model`` names a vision model on Ollama that describes Figure regions
    during the parse (parse-time only, see vlm.py). Empty disables it, which is
    what a machine without the model gets.

    Raises FileNotFoundError for a missing file and ValueError for a format that
    is not supported (the message names the suffix, so a caller can tell "we do
    not read this" from "this file is broken").
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    if not documents.is_supported(path):
        raise ValueError(f"unsupported document format: {documents.suffix_of(path) or path}")

    vlm_model = (vlm_model or "").strip()

    cache = ParseCache() if use_cache else None
    key = ""
    if cache is not None:
        key = cache_key(
            path, profile=profile, max_chars=max_chars, max_pages=max_pages, layout=layout,
            vlm=vlm_model,
        )
        hit = cache.get(key)
        if hit is not None:
            # Layout detection dominates a parse, so a hit is worth the whole
            # run. The stamp tells the caller which chunks were produced from
            # cache rather than from the document as it stands now.
            return touch_marker(hit)

    source_file = os.path.basename(path)
    if documents.suffix_of(path) in documents.PDF_SUFFIXES:
        loaded = _pdf_blocks(path, source_file, max_pages, layout)
    else:
        loaded = _document_blocks(path, source_file)

    blocks = _assign_sections(loaded["blocks"])
    mean_height = _mean_height(blocks)

    # Parse-time vision. Runs BEFORE consolidation so the description is part of
    # the Figure block when the chunker folds the caption onto it — otherwise
    # the two would be separate chunks and the figure would still be reachable
    # only through its caption. A no-op for non-PDF input (no Figure regions)
    # and for an empty model name.
    figures_described = 0
    if vlm_model:
        captioner = VlmCaptioner(vlm_model, os.environ.get("FREERAG_OLLAMA_URL", ""))
        figures_described = caption_figures(path, blocks, captioner)

    consolidated = consolidate_blocks(
        blocks,
        max_chars=max_chars,
        profile=profile,
        page_width=loaded["page_width"],
        mean_height=mean_height,
        page_height=loaded["page_height"],
    )
    chunks = to_chunks(consolidated)

    result = {
        "path": path,
        "source_file": source_file,
        "page_count": loaded["page_count"],
        "pages_parsed": loaded["pages_parsed"],
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "layout_provider": loaded["provider"],
        "profile": profile,
        "vlm_model": vlm_model,
        "figures_described": figures_described,
        "chunks": chunks,
    }
    if cache is not None:
        cache.put(key, result)
    return result
