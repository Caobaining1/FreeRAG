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
CACHE_VERSION = "2"


def cache_dir() -> str:
    return os.environ.get("FREERAG_CACHE_DIR") or DEFAULT_CACHE_DIR


def cache_key(
    path: str,
    *,
    profile: str,
    max_chars: Optional[int],
    max_pages: int,
    layout: str,
) -> str:
    """Identity of one parse request."""
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
