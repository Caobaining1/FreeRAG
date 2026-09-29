"""Chunking and short-block consolidation.

Implements plan.md §5.3 (one layout block = one chunk) and §5.3.1 (short-block
merge / attach / discard). Pure logic: no PDF or model dependency, so it is
directly unit-testable.

Consolidation priority (per block, in order):

  1. exempt types (Title / Table / Figure / Equation) pass through untouched;
  2. a short Text block merges into the preceding mergeable block when the
     adjacency test holds (this is the actual merge);
  3. otherwise it is kept standalone with an ``attached_to`` link, unless it
     fails the usefulness test, in which case it is discarded.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --- tunables (plan.md §5.3.1, §5.5) -------------------------------------

#: A block shorter than this is "short" (plan.md §5.3.1).
SHORT_TEXT_CJK_CHARS = 40
SHORT_TEXT_LATIN_WORDS = 20

#: Max chunk size by language profile, derived from BGE-M3's 8192-token window
#: (plan.md §5.5). Used as a SPLIT threshold, not a target size.
MAX_CHARS_PROFILE = {"zh": 8000, "en": 30000, "mixed": 12000}
DEFAULT_PROFILE = "mixed"

#: Identical short texts seen at least this often are page headers/footers.
NOISE_REPEAT_THRESHOLD = 3

#: A latin fragment below this length with no CJK is dropped (TUIrag's rule).
MIN_USEFUL_CHARS = 10

#: x-overlap ratio below which two blocks are not considered stackable.
MIN_X_OVERLAP_RATIO = 0.3

#: Vertical gap (in mean line heights) above which stacking is rejected.
MAX_VERTICAL_GAP_RATIO = 1.5

#: Fraction of the page height treated as the header/footer band.
MARGIN_RATIO = 0.07

#: A margin block longer than this is content that merely sits high or low.
MARGIN_MAX_CHARS = 80

#: Types that are never dropped as margin noise — they are content wherever they
#: sit, so a Title in the top band is a real heading, not a running head.
CONTENT_TYPES = ("Title", "Table", "Figure", "Equation", "Caption")

#: Caption and Reference are exempt for the same reason as Table/Figure: they are
#: narrow by nature, so the usefulness test would discard them. The layout
#: detector names them explicitly, and a dropped caption loses the table's or
#: figure's meaning, so they must survive consolidation untouched.
EXEMPT_TYPES = ("Title", "Table", "Figure", "Equation", "Caption", "Reference")
MERGEABLE_TYPES = ("Text",)

#: Line-segment vertical slack when grouping words into lines.
_LINE_TOLERANCE = 3.0

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
#: Page numbers and decorative rulers: digits, bullets, dashes, spaces only.
PAGE_DECOR_RE = re.compile(r"^[0-9\s\u2022\u00b7\u2014\u2013_=*\-]+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b.!?\n])\s*")
_TERMINAL_PUNCT = "\u3002\uff01\uff1f.!?"


@dataclass
class Block:
    """One layout block on its way to becoming a chunk."""

    text: str
    page_num: int = 0
    block_type: str = "Text"
    bbox: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    font_size: float = 0.0
    parent_section: str = ""
    source_file: str = ""

    # Filled during consolidation.
    chunk_id: str = ""
    attached_to: str = ""
    merged_from: List[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return max(0.0, float(self.bbox[2]) - float(self.bbox[0]))

    @property
    def height(self) -> float:
        return max(0.0, float(self.bbox[3]) - float(self.bbox[1]))

    def render(self) -> str:
        return self.text.strip()

    def to_chunk(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "page_num": self.page_num,
            "block_type": self.block_type,
            "bbox": [round(float(v), 2) for v in self.bbox],
            "font_size": self.font_size,
            "parent_section": self.parent_section,
            "source_file": self.source_file,
        }
        if self.attached_to:
            meta["attached_to"] = self.attached_to
        if self.merged_from:
            meta["merged_from"] = list(self.merged_from)
        meta["chars"] = len(self.render())
        return {"chunk_id": self.chunk_id, "text": self.render(), "metadata": meta}


def normalize(text: str) -> str:
    """Collapse whitespace and case for noise detection."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def max_chars_for(profile: str = DEFAULT_PROFILE) -> int:
    """Chunk split threshold for a language profile."""
    return MAX_CHARS_PROFILE.get(profile, MAX_CHARS_PROFILE[DEFAULT_PROFILE])


def is_short(text: str) -> bool:
    """True when a block counts as short (plan.md §5.3.1)."""
    stripped = text.strip()
    if not stripped:
        return True
    if _CJK_RE.search(stripped):
        return len(stripped) < SHORT_TEXT_CJK_CHARS
    return len(stripped.split()) < SHORT_TEXT_LATIN_WORDS


def split_at_sentences(text: str, max_chars: int) -> List[str]:
    """Split text on sentence boundaries, never exceeding max_chars per part.

    A single sentence longer than max_chars is hard-sliced: losing content is
    worse than a mid-sentence cut (plan.md §5.3: "不截断").
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    current = ""
    for sentence in (s.strip() for s in _SENTENCE_SPLIT_RE.split(text)):
        if not sentence:
            continue
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts or [text]


def _join(a: str, b: str) -> str:
    """Concatenate two fragments with a space only between latin words."""
    if not a:
        return b
    if not b:
        return a
    if _CJK_RE.search(a[-1]) or _CJK_RE.search(b[0]):
        return a + b
    return a + " " + b


def _x_overlap_ratio(a: Block, b: Block) -> float:
    overlap = max(0.0, min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0]))
    smallest = min(a.width, b.width)
    if smallest <= 0:
        return 1.0 if overlap > 0 else 0.0
    return overlap / smallest


def _vertically_adjacent(prev: Block, cur: Block, mean_height: float) -> bool:
    """Same page (or the next one) and no big vertical gap between the blocks."""
    if cur.page_num == prev.page_num:
        gap = float(cur.bbox[1]) - float(prev.bbox[3])
        if gap <= 0:
            return True
        if mean_height <= 0:
            return True
        return gap <= mean_height * MAX_VERTICAL_GAP_RATIO
    if cur.page_num == prev.page_num + 1:
        return True
    return False


def _is_useful(block: Block, page_width: float, mean_height: float) -> bool:
    """RAGFlow's ``usefull()`` heuristic (pdf_parser.py:1550).

    Exempt types always pass; a Text block passes when it is wide enough or tall
    enough to be body content rather than a stray fragment.
    """
    if block.block_type in EXEMPT_TYPES:
        return True
    if page_width > 0 and block.width > page_width / 3.0:
        return True
    if mean_height > 0 and block.height > mean_height:
        return True
    return False


def can_merge(prev: Block, cur: Block, mean_height: float) -> bool:
    """Whether ``cur`` may be appended to ``prev``.

    Mirrors the RAGFlow gates plan.md §5.3.1 borrows: same section, vertical
    adjacency, enough horizontal overlap, and the previous block not closing a
    sentence.
    """
    if prev.block_type not in MERGEABLE_TYPES:
        return False
    if cur.block_type not in MERGEABLE_TYPES:
        return False
    if prev.parent_section != cur.parent_section:
        return False
    if not _vertically_adjacent(prev, cur, mean_height):
        return False
    if _x_overlap_ratio(prev, cur) < MIN_X_OVERLAP_RATIO:
        return False
    if prev.render() and prev.render()[-1] in _TERMINAL_PUNCT:
        return False
    return True


def _merge_into(prev: Block, cur: Block) -> None:
    """Append cur's text to prev, extending its box and provenance."""
    prev.text = _join(prev.render(), cur.render())
    x0 = min(prev.bbox[0], cur.bbox[0])
    y0 = min(prev.bbox[1], cur.bbox[1])
    x1 = max(prev.bbox[2], cur.bbox[2])
    y1 = max(prev.bbox[3], cur.bbox[3])
    prev.bbox = (x0, y0, x1, y1)
    if cur.chunk_id:
        prev.merged_from.append(cur.chunk_id)
    if cur.merged_from:
        prev.merged_from.extend(cur.merged_from)


def is_margin_noise(block: Block, page_height: float) -> bool:
    """True for a short block sitting in the page's header or footer band.

    The layout model's 10 classes have no Header/Footer class, so running heads
    and page numbers arrive labelled ``Reference`` (measured on a real paper: the
    arXiv sidebar and the journal footer both came back as Reference). Repeated
    text is already caught by the noise filter, but a page number differs on every
    page and slips through, so it needs a positional rule.

    ``Reference`` and ``Text`` are eligible; ``CONTENT_TYPES`` are not, because a
    real heading can legitimately sit at the top of a page.
    """
    if page_height <= 0 or block.block_type in CONTENT_TYPES:
        return False
    if len(block.render()) > MARGIN_MAX_CHARS:
        return False

    band = page_height * MARGIN_RATIO
    return float(block.bbox[3]) <= band or float(block.bbox[1]) >= page_height - band


def consolidate_blocks(
    blocks: Sequence[Block],
    *,
    max_chars: Optional[int] = None,
    profile: str = DEFAULT_PROFILE,
    page_width: float = 0.0,
    mean_height: float = 0.0,
    page_height: float = 0.0,
) -> List[Block]:
    """Apply plan.md §5.3 / §5.3.1: one block per chunk, short blocks folded in."""
    limit = max_chars if max_chars is not None else max_chars_for(profile)
    kept = [b for b in blocks if b.render()]

    # Page headers/footers repeat verbatim; identical short texts are noise.
    repeated = Counter(normalize(b.text) for b in kept if is_short(b.text))
    noise = {text for text, count in repeated.items() if count >= NOISE_REPEAT_THRESHOLD}

    out: List[Block] = []

    def append(block: Block) -> None:
        # The id is assigned on append so `attached_to` can reference a real id.
        block.chunk_id = f"c{len(out):05d}"
        out.append(block)

    for block in kept:
        text = block.render()

        if PAGE_DECOR_RE.match(text):
            continue
        if normalize(text) in noise:
            continue
        if is_margin_noise(block, page_height):
            continue
        if block.block_type in EXEMPT_TYPES:
            append(block)
            continue

        prev = out[-1] if out else None

        if is_short(text):
            if prev is not None and can_merge(prev, block, mean_height):
                _merge_into(prev, block)
                continue
            if len(text) < MIN_USEFUL_CHARS and not _CJK_RE.search(text) and not _is_useful(block, page_width, mean_height):
                continue
            if not _is_useful(block, page_width, mean_height) and prev is None:
                continue
            if prev is not None:
                block.attached_to = prev.chunk_id
            append(block)
            continue

        append(block)

    return _finalize(out, limit)


def _finalize(blocks: List[Block], max_chars: int) -> List[Block]:
    """Assign ids and split oversized blocks at sentence boundaries."""
    finalized: List[Block] = []
    for index, block in enumerate(blocks):
        base_id = block.chunk_id or f"c{index:05d}"
        parts = split_at_sentences(block.render(), max_chars)
        for part_index, part in enumerate(parts):
            copy = Block(
                text=part,
                page_num=block.page_num,
                block_type=block.block_type,
                bbox=block.bbox,
                font_size=block.font_size,
                parent_section=block.parent_section,
                source_file=block.source_file,
                attached_to=block.attached_to,
                merged_from=list(block.merged_from),
            )
            copy.chunk_id = base_id if len(parts) == 1 else f"{base_id}p{part_index}"
            finalized.append(copy)
    return finalized


def to_chunks(blocks: Sequence[Block]) -> List[Dict[str, Any]]:
    """Render consolidated blocks as the sidecar's chunk payload."""
    return [b.to_chunk() for b in blocks]
