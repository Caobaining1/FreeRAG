"""Chunking: layout blocks -> retrieval chunks.

Two stages, in order:

  1. **Filter** (plan.md §5.3.1): drop page decoration, repeated running heads
     and margin noise before anything else sees the blocks.
  2. **Merge** (TUIrag's strategy B, ``layout_chunker.py::_build_chunks_strategy_b``):
     fold blocks back into reading units —

       * a Title opens a Section and absorbs the body under it, so a heading is
         never indexed as a standalone chunk that out-scores the paragraph it
         announces (plan.md issue #2);
       * consecutive Text/Equation paragraphs buffer into one chunk up to a
         ceiling;
       * a Table/Figure takes its nearest following caption;
       * repeated header/footer captions are skipped without breaking a run.

     A final split at sentence boundaries keeps every chunk inside the hard
     threshold (BGE-M3's window, plan.md §5.5).

Pure logic: no PDF or model dependency, so it is directly unit-testable.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --- filter tunables (plan.md §5.3.1, §5.5) ------------------------------

#: A block shorter than this is "short" (plan.md §5.3.1).
SHORT_TEXT_CJK_CHARS = 40
SHORT_TEXT_LATIN_WORDS = 20

#: Max chunk size by language profile, derived from BGE-M3's 8192-token window
#: (plan.md §5.5). Used as a SPLIT threshold, not a target size.
MAX_CHARS_PROFILE = {"zh": 8000, "en": 30000, "mixed": 12000}
DEFAULT_PROFILE = "mixed"

#: Merge ceiling by language profile: how large a merged chunk may grow before
#: the merge closes it and starts another. Deliberately well below
#: MAX_CHARS_PROFILE, which is only the hard split threshold — but large enough
#: that a section fits in one chunk.
#:
#: Chosen by measuring six papers while raising it (chunks / body chunks /
#: sections that end up as a single body chunk, out of 111):
#:
#:   1200 → 618 / 500 / 28      4000 → 361 / 251 / 48
#:   2000 → 454 / 344 / 37      6000 → 334 / 225 / 49
#:   3000 → 386 / 276 / 45     12000 → 334 / 225 / 49
#:
#: Past ~4000 nothing more is bought: what still splits a section by then is a
#: table or figure interrupting it, or a page break, not the ceiling. In tokens
#: these are ~2000 (zh), ~1500 (mixed) and ~1000 (en) — a fifth of BGE-M3's
#: window at most, so a chunk still reads as one topic.
MERGE_MAX_PROFILE = {"zh": 2000, "en": 4000, "mixed": 3000}

#: Identical short texts seen at least this often are page headers/footers.
NOISE_REPEAT_THRESHOLD = 3

#: A latin fragment below this length with no CJK is dropped (TUIrag's rule).
MIN_USEFUL_CHARS = 10

#: Fraction of the page height treated as the header/footer band.
MARGIN_RATIO = 0.07

#: A margin block longer than this is content that merely sits high or low.
MARGIN_MAX_CHARS = 80

#: Types that are never dropped as margin noise — they are content wherever they
#: sit, so a Title in the top band is a real heading, not a running head.
#: Caption is in the set for the same reason: a dropped caption loses the table's
#: or figure's meaning.
CONTENT_TYPES = ("Title", "Table", "Figure", "Equation", "Caption")

#: The hierarchical merge walks the sequence: a Title absorbs BODY_TYPES under
#: it, and the paragraph buffer accumulates BODY_TYPES. Equations ride along with
#: the prose that introduces them rather than standing alone.
BODY_TYPES = ("Text", "Equation")

#: Fraction of the narrower block's width that must overlap for two blocks to
#: count as the same column.
MIN_X_OVERLAP_RATIO = 0.3

#: Slack (PDF points) for "starts below the previous block", so a detector's
#: slightly overlapping boxes still read as stacked.
STACK_SLACK = 6.0

#: A caption the detector failed to label: "Table 4: ...", "Fig. 3.", "表 2、...".
#:
#: The label must be followed by a separator, which is what separates a caption
#: from a sentence *about* a table: "Table 4: Results of…" is a caption, "Table 4
#: shows that…" is body text. Measured on a paper where the detector missed four
#: of its ten captions, the pattern matched 17 blocks — 13 already typed Caption
#: and exactly those 4, with no body sentence caught.
CAPTION_LABEL_RE = re.compile(
    r"^\s*(?:table|tab\.|figure|fig\.|chart|listing|algorithm|exhibit|表|图|图表)\s*"
    r"(?:\d+|[ivxlc]+|[一二三四五六七八九十]+)\s*[:：.\-–—、]",
    re.IGNORECASE,
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
#: Page numbers and decorative rulers: digits, bullets, dashes, spaces only.
PAGE_DECOR_RE = re.compile(r"^[0-9\s\u2022\u00b7\u2014\u2013_=*\-]+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b.!?\n])\s*")


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


def merge_max_for(profile: str = DEFAULT_PROFILE) -> int:
    """Merge ceiling for a language profile."""
    return MERGE_MAX_PROFILE.get(profile, MERGE_MAX_PROFILE[DEFAULT_PROFILE])


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


# --- merge helpers (TUIrag strategy B) -----------------------------------

def _union_bbox(blocks: Sequence[Block]) -> Sequence[float]:
    return (
        min(float(b.bbox[0]) for b in blocks),
        min(float(b.bbox[1]) for b in blocks),
        max(float(b.bbox[2]) for b in blocks),
        max(float(b.bbox[3]) for b in blocks),
    )


def _combine(group: List[Block], block_type: str) -> Block:
    """Fold a run of blocks into one, keeping the first block's provenance."""
    first = group[0]
    combined = Block(
        text="\n\n".join(b.render() for b in group),
        page_num=first.page_num,
        block_type=block_type,
        bbox=_union_bbox(group),
        font_size=first.font_size,
        parent_section=first.parent_section,
        source_file=first.source_file,
    )
    if block_type == "Section":
        # A Section is named by its own heading, which leads the group.
        combined.parent_section = first.render()
    combined.merged_from = [b.chunk_id for b in group[1:] if b.chunk_id]
    return combined


def _fits(group: Sequence[Block], candidate: Block, ceiling: int) -> bool:
    """Whether appending ``candidate`` keeps the run within the ceiling."""
    total = sum(len(b.render()) for b in group) + len(candidate.render())
    return total <= ceiling


def _stacked(prev: Block, cur: Block) -> bool:
    """Whether ``cur`` sits directly below ``prev`` in the same column.

    A chunk is drawn over the original page as one rectangle, so the blocks it
    folds together have to occupy one rectangle's worth of page. Reading order
    alone does not give that: a three-column author grid is read left to right
    across a row, and the last block of one column is followed by the first
    block of the next. Folding those pairs makes the chunk a rectangle covering
    the whole corner between them — a highlight over mostly empty page (measured
    across six papers: 42 of 272 merged chunks, 40 of them with a bounding box
    under 60% covered by the text in it).

    So a run only continues downward: the next block must be below the last one
    and share its column. Anything else — a block beside it, or one that jumps
    back up the page — ends the run and starts a new chunk.
    """
    if cur.page_num != prev.page_num:
        return False

    narrower = min(prev.width, cur.width)
    if narrower <= 0:  # a box with no width did not come from a real region
        return True
    overlap = min(prev.bbox[2], cur.bbox[2]) - max(prev.bbox[0], cur.bbox[0])
    if overlap / narrower < MIN_X_OVERLAP_RATIO:
        return False
    return float(cur.bbox[1]) >= float(prev.bbox[3]) - STACK_SLACK


def _flush_body(buffer: List[Block], out: List[Block], ceiling: int) -> None:
    """Emit a buffered run of paragraphs as one chunk (TUIrag ``_flush_text_buffer``)."""
    if not buffer:
        return
    combined = _combine(buffer, buffer[0].block_type)
    if len(combined.render()) > ceiling:
        # Only a single block over the ceiling reaches here: the buffer is filled
        # greedily, so a run of blocks never exceeds it. Split that block, not
        # the run, so a page-long paragraph is the only thing cut.
        for part in split_at_sentences(combined.render(), ceiling):
            piece = _combine(buffer, buffer[0].block_type)
            piece.text = part
            out.append(piece)
    else:
        out.append(combined)
    buffer.clear()


def _is_noise_caption(block: Block, noise_captions: set) -> bool:
    """A repeated running head/footer the detector labelled ``Caption``."""
    return block.block_type == "Caption" and normalize(block.text) in noise_captions


def _labeled_table_or_figure(blocks: Sequence[Block], index: int) -> Optional[Block]:
    """The table or figure the block at ``index`` sits with, if any.

    "Sits with" is the same stacking test the merge uses, so a caption on the far
    side of a column gap is not taken for a caption of that table.
    """
    block = blocks[index]
    if index > 0:
        previous = blocks[index - 1]
        if previous.block_type in ("Table", "Figure") and _stacked(previous, block):
            return previous
    if index + 1 < len(blocks):
        following = blocks[index + 1]
        if following.block_type in ("Table", "Figure") and _stacked(block, following):
            return following
    return None


def _promote_missed_captions(blocks: List[Block]) -> List[Block]:
    """Re-type a caption the detector handed back as body text.

    Measured on a real paper: the detector returned "Table 4: Results of ablation
    studies." as ``Text``, so the merge never attached it — it became a chunk of
    its own, and the next one ("Table 5: …") was absorbed into the paragraph
    below it instead. Both are captions of the table right above them.

    The text has to look like a caption AND sit with a table or figure: either
    alone is not enough, since "Table 4: …" in running text is not a caption, and
    a body block beside a table is not one either.
    """
    for index, block in enumerate(blocks):
        if block.block_type in ("Table", "Figure", "Caption"):
            continue
        if not CAPTION_LABEL_RE.match(block.render()):
            continue
        if _labeled_table_or_figure(blocks, index) is not None:
            block.block_type = "Caption"
    return blocks


def _absorb_caption(block: Block, caption: Block) -> Block:
    """Attach a table's or figure's caption, keeping the caption first.

    TUIrag prefixes the caption to both, and the caption is what a reader
    searches for ("Table 2: ..."), so it leads the chunk.
    """
    combined = Block(
        text=caption.render() + "\n" + block.render(),
        page_num=block.page_num,
        block_type=f"{block.block_type}WithCaption",
        bbox=_union_bbox([block, caption]),
        font_size=block.font_size,
        parent_section=block.parent_section,
        source_file=block.source_file,
    )
    combined.merged_from = list(block.merged_from) + ([caption.chunk_id] if caption.chunk_id else [])
    return combined


def _merge_hierarchy(blocks: Sequence[Block], ceiling: int, noise_captions: set) -> List[Block]:
    """Group blocks into retrieval chunks (TUIrag strategy B).

    Four rules, applied in reading order:

      * a Title opens a Section and absorbs the body under it;
      * consecutive Text/Equation paragraphs buffer into one chunk;
      * a Table/Figure takes its caption, on whichever side it sits;
      * repeated Captions are skipped without breaking a run.

    Every fold also has to pass ``_stacked``: a chunk is one rectangle on the
    page, so it may only collect blocks that occupy one rectangle's worth of it.
    """
    out: List[Block] = []
    buffer: List[Block] = []
    index = 0
    total = len(blocks)

    while index < total:
        block = blocks[index]

        if _is_noise_caption(block, noise_captions):
            index += 1
            continue

        if block.block_type == "Title":
            _flush_body(buffer, out, ceiling)
            group = [block]
            j = index + 1
            while j < total:
                nxt = blocks[j]
                if _is_noise_caption(nxt, noise_captions):
                    j += 1
                    continue
                if nxt.block_type in BODY_TYPES and _stacked(group[-1], nxt) and _fits(group, nxt, ceiling):
                    group.append(nxt)
                    j += 1
                    continue
                break
            out.append(_combine(group, "Section" if len(group) > 1 else "Title"))
            index = j
            continue

        if block.block_type in ("Table", "Figure"):
            _flush_body(buffer, out, ceiling)
            caption: Optional[Block] = None
            j = index + 1
            while j < total and _is_noise_caption(blocks[j], noise_captions):
                j += 1
            # The caption must be the next block that is not a running head, and
            # it must sit with the table or figure. A caption on the far side of
            # the column gap is somewhere else on the page, and joining it would
            # draw one box over both.
            if j < total and blocks[j].block_type == "Caption" and _stacked(block, blocks[j]):
                caption = blocks[j]
                j += 1
            out.append(_absorb_caption(block, caption) if caption is not None else block)
            index = j
            continue

        if block.block_type in BODY_TYPES:
            if buffer and not (_stacked(buffer[-1], block) and _fits(buffer, block, ceiling)):
                _flush_body(buffer, out, ceiling)
            buffer.append(block)
            index += 1
            continue

        if block.block_type == "Caption":
            # A caption sits above its table or figure as often as below it.
            # Taken forward-only it becomes a chunk of its own — a fragment
            # under 200 characters, which is the shape plan.md issue ② is about.
            _flush_body(buffer, out, ceiling)
            j = index + 1
            while j < total and _is_noise_caption(blocks[j], noise_captions):
                j += 1
            nxt = blocks[j] if j < total else None
            if nxt is not None and nxt.block_type in ("Table", "Figure") and _stacked(block, nxt):
                out.append(_absorb_caption(nxt, block))
                index = j + 1
                continue
            out.append(block)
            index += 1
            continue

        _flush_body(buffer, out, ceiling)
        out.append(block)
        index += 1

    _flush_body(buffer, out, ceiling)
    return out


def consolidate_blocks(
    blocks: Sequence[Block],
    *,
    max_chars: Optional[int] = None,
    profile: str = DEFAULT_PROFILE,
    merge_max: Optional[int] = None,
    page_width: float = 0.0,
    mean_height: float = 0.0,
    page_height: float = 0.0,
) -> List[Block]:
    """Filter then merge blocks into chunks (plan.md §5.3 / §5.3.1).

    ``max_chars`` is the hard split threshold (BGE-M3's window); ``merge_max`` is
    the smaller ceiling the hierarchical merge may grow a chunk to. ``page_width``
    and ``mean_height`` are accepted for compatibility with the filter rules.
    """
    limit = max_chars if max_chars is not None else max_chars_for(profile)
    ceiling = merge_max if merge_max is not None else merge_max_for(profile)

    kept = [b for b in blocks if b.render()]

    # Page headers/footers repeat verbatim; identical short texts are noise.
    # Content types are exempt, because a document's own title is also the
    # running head printed on every later page: identical text, opposite
    # meaning. Counting it would delete the heading the merge exists to attach
    # (measured on a real paper: the title was dropped and every other page's
    # Reference kept it). Repeated Captions are handled separately below.
    repeated = Counter(
        normalize(b.text)
        for b in kept
        if is_short(b.text) and b.block_type not in CONTENT_TYPES
    )
    noise = {text for text, count in repeated.items() if count >= NOISE_REPEAT_THRESHOLD}
    caption_counts = Counter(normalize(b.text) for b in kept if b.block_type == "Caption")
    noise_captions = {
        text for text, count in caption_counts.items() if count >= NOISE_REPEAT_THRESHOLD
    }

    cleaned: List[Block] = []
    for block in kept:
        text = block.render()
        if PAGE_DECOR_RE.match(text):
            continue
        # The drop mirrors how `noise` was built: a content block is never
        # discarded for repeating text, only the running heads that repeat it.
        if block.block_type not in CONTENT_TYPES and normalize(text) in noise:
            continue
        if is_margin_noise(block, page_height):
            continue
        cleaned.append(block)

    # After the filters, since a running head that happens to read like a caption
    # is noise rather than a caption; before the merge, so every caption rule
    # below sees the same block types.
    _promote_missed_captions(cleaned)

    # Provisional ids give the merge something to name its absorbed blocks by;
    # _finalize overwrites them with the emitted chunk ids.
    for position, block in enumerate(cleaned):
        block.chunk_id = f"b{position:05d}"

    merged = _merge_hierarchy(cleaned, ceiling, noise_captions)
    return _finalize(merged, limit)


def _finalize(blocks: List[Block], max_chars: int) -> List[Block]:
    """Drop bare fragments, assign chunk ids, split anything past the limit.

    TUIrag's last step discards any chunk under 10 characters; the same rule
    applies here, so a lone table number ("(1)") or a bare latin heading never
    reaches the index as a chunk that could out-score the text it labels. CJK is
    exempt because a handful of characters already carry meaning.
    """
    finalized: List[Block] = []
    for block in blocks:
        text = block.render()
        if len(text) < MIN_USEFUL_CHARS and not _CJK_RE.search(text):
            continue

        base_id = f"c{len(finalized):05d}"
        parts = split_at_sentences(text, max_chars)
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
