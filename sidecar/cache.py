"""Parse cache: skip re-parsing a document that has not changed.

Layout detection dominates a parse — measured 0.2–0.7 s/page with an accelerator
and up to 7.8 s/page without — and the kernel re-parses on every `index` call, so
the same file gets analysed twice for one user action. The cache makes the second
one free.

The key covers everything that can change the result: the file's identity (path,
size, mtime) and every option the pipeline reads. A cache that ignored an option
would return chunks built under different rules while looking like a hit, which
is worse than no cache at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

#: Cache location, overridable for tests and packaging.
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "parse"
)

#: Soft cap on the cache directory. On overflow the least recently modified
#: entries are dropped until it fits; a desktop app cannot grow without bound.
DEFAULT_LIMIT_BYTES = 256 * 1024 * 1024

#: Bumped whenever the pipeline's output format or semantics change, so stale
#: entries are never mistaken for current ones.
#:
#: 2: table regions now go through TSR before ruling-line extraction, so the
#:    same PDF yields different (structured) chunks. Version 1 entries describe
#:    the flattened text and must not be served.
#: 3: chunking now merges blocks (TUIrag strategy B: title + body, buffered
#:    paragraphs, table/figure + caption), so every document's chunk list differs
#:    from version 2 even when the options are identical.
#: 4: layout blocks come back in reading order (XY-cut, not a row-major sweep),
#:    so a two-column page's chunks are no longer interleaved across columns.
#: 5: a merged chunk now also carries `pieces`, the source rectangles it was
#:    folded from, which is what the chunk inspector draws instead of the
#:    bounding box that spans the whole corner between them.
#: 6: `pieces` is gone again — the merge no longer folds blocks that are not
#:    stacked in one column, so every chunk's bounding box is the region it
#:    actually occupies and one rectangle is the honest drawing.
#: 7: the merge ceiling is ~2.5x larger and a caption above its table/figure is
#:    absorbed too, so a section is no longer split into several chunks.
#: 8: a caption the detector left typed as body text ("Table 4: ...") is
#:    re-typed and attached, instead of becoming its own chunk or being absorbed
#:    into the paragraph below it.
#: 9: a detection sitting inside a larger one of the same type is dropped, so a
#:    region found twice no longer yields two blocks with the same text.
#: 10: a Figure region can now be described by a vision model, so the same PDF
#:    yields different Figure chunks when that is enabled. The model name is
#:    part of the key (two models are two different caption texts), and a version
#:    bump is needed regardless because a version-9 entry has no description.
#: 11: every block now carries the largest glyph size inside it, so headings
#:    parsed through PP-DocLayout keep their font size and the tree builder can
#:    rank heading levels. Version 11 entries have font_size 0 on every heading,
#:    which the tree builder reads as a document with no hierarchy at all.
CACHE_VERSION = "12"


def cache_dir() -> str:
    return os.environ.get("FREERAG_CACHE_DIR") or DEFAULT_CACHE_DIR


def cache_key(
    path: str,
    *,
    profile: str,
    max_chars: Optional[int],
    max_pages: int,
    layout: str,
    vlm: str = "",
) -> str:
    """Identity of one parse request.

    ``vlm`` is the vision model describing figures, or "" when that is off. It
    belongs in the key for the same reason every other option does: two models
    write different text into the chunk, so a hit across them would return
    chunks that were never produced under the rules now in force.
    """
    stat = os.stat(path)
    parts = (
        CACHE_VERSION,
        os.path.abspath(path),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        profile,
        str(max_chars if max_chars is not None else ""),
        str(max_pages),
        layout,
        vlm,
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class ParseCache:
    """A small on-disk cache of parse results."""

    def __init__(self, directory: str = "", limit_bytes: int = 0) -> None:
        self.directory = directory or cache_dir()
        self.limit_bytes = limit_bytes or DEFAULT_LIMIT_BYTES
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> str:
        return os.path.join(self.directory, "%s.json" % key)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        target = self.path_for(key)
        try:
            with open(target, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # A corrupt or half-written entry is a miss, not an error: the
            # pipeline simply re-parses.
            self.misses += 1
            return None

        # Touch so eviction treats a re-used entry as recent.
        try:
            os.utime(target, None)
        except OSError:
            pass

        self.hits += 1
        return payload

    def put(self, key: str, value: Dict[str, Any]) -> None:
        try:
            os.makedirs(self.directory, exist_ok=True)
            target = self.path_for(key)
            # Write to a temporary file first so a crash cannot leave a
            # half-written entry that later reads as corrupt.
            temporary = target + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False)
            os.replace(temporary, target)
        except OSError:
            return  # a cache that cannot write is a cache that does not exist

        self._evict()

    def _evict(self) -> None:
        """Drop least-recently-modified entries until the directory fits."""
        try:
            entries = []
            total = 0
            for name in os.listdir(self.directory):
                if not name.endswith(".json"):
                    continue
                full = os.path.join(self.directory, name)
                stat = os.stat(full)
                entries.append((stat.st_mtime, stat.st_size, full))
                total += stat.st_size
        except OSError:
            return

        if total <= self.limit_bytes:
            return

        entries.sort()  # oldest first
        for _, size, full in entries:
            if total <= self.limit_bytes:
                break
            try:
                os.remove(full)
                total -= size
            except OSError:
                continue

    def clear(self) -> int:
        """Remove every entry; returns how many were dropped."""
        removed = 0
        try:
            for name in os.listdir(self.directory):
                if not name.endswith(".json"):
                    continue
                try:
                    os.remove(os.path.join(self.directory, name))
                    removed += 1
                except OSError:
                    continue
        except OSError:
            return 0
        return removed

    def stats(self) -> Dict[str, Any]:
        return {"dir": self.directory, "hits": self.hits, "misses": self.misses, "limit_bytes": self.limit_bytes}


def touch_marker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a cached payload so a hit is visible in the response."""
    stamped = dict(payload)
    stamped["cached"] = True
    stamped["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return stamped
