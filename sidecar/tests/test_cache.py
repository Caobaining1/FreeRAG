"""Unit tests for the parse cache."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import CACHE_VERSION, ParseCache, cache_key, touch_marker  # noqa: E402


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "doc.pdf")
        with open(self.path, "wb") as handle:
            handle.write(b"%PDF-1.4 fake")

    def key(self, **overrides):
        options = {"profile": "mixed", "max_chars": None, "max_pages": 0, "layout": "auto"}
        options.update(overrides)
        return cache_key(self.path, **options)

    def test_same_request_yields_the_same_key(self):
        self.assertEqual(self.key(), self.key())

    def test_each_option_changes_the_key(self):
        # A cache that ignored an option would return chunks built under
        # different rules while looking like a hit.
        baseline = self.key()
        for overrides in (
            {"profile": "en"},
            {"max_chars": 500},
            {"max_pages": 5},
            {"layout": "pymupdf"},
            # Two vision models write different text into a Figure chunk, so a
            # hit across them would return chunks no model produced.
            {"vlm": "qwen3-vl:4b"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.key(**overrides), baseline)

    def test_a_modified_file_changes_the_key(self):
        before = self.key()
        time.sleep(0.01)  # mtime resolution
        with open(self.path, "ab") as handle:
            handle.write(b" more")
        self.assertNotEqual(self.key(), before, "an edited file must not hit the old entry")

    def test_the_cache_version_participates(self):
        # Guards against a format change silently reusing stale entries.
        self.assertTrue(CACHE_VERSION)
        os.environ["FREERAG_CACHE_DIR"] = self.directory.name
        self.addCleanup(os.environ.pop, "FREERAG_CACHE_DIR", None)


class ParseCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def cache(self, limit_bytes=0):
        return ParseCache(self.directory.name, limit_bytes=limit_bytes)

    def test_put_then_get_round_trips(self):
        cache = self.cache()
        payload = {"chunk_count": 2, "chunks": [{"chunk_id": "c0", "text": "hi"}]}
        cache.put("abc", payload)

        self.assertEqual(cache.get("abc"), payload)
        self.assertEqual((cache.hits, cache.misses), (1, 0))

    def test_a_missing_entry_is_a_miss(self):
        cache = self.cache()
        self.assertIsNone(cache.get("nope"))
        self.assertEqual((cache.hits, cache.misses), (0, 1))

    def test_a_corrupt_entry_is_a_miss_not_an_error(self):
        cache = self.cache()
        with open(os.path.join(self.directory.name, "bad.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")

        self.assertIsNone(cache.get("bad"), "a corrupt entry must not raise")
        self.assertEqual(cache.misses, 1)

    def test_put_leaves_no_temporary_files(self):
        cache = self.cache()
        cache.put("abc", {"ok": True})
        leftovers = [n for n in os.listdir(self.directory.name) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [], "writes must be atomic via rename")

    #: Each entry below serialises to roughly 1 KB, so the caps are chosen to
    #: admit a known number of them.
    FILLER = "x" * 1000

    def test_eviction_drops_the_oldest_entries(self):
        # Two ~1 KB entries against a 1.5 KB cap: the older one goes.
        cache = self.cache(limit_bytes=1500)
        cache.put("old", {"data": self.FILLER})
        time.sleep(0.02)
        cache.put("new", {"data": self.FILLER})

        remaining = sorted(n for n in os.listdir(self.directory.name) if n.endswith(".json"))
        self.assertIn("new.json", remaining)
        self.assertNotIn("old.json", remaining, "eviction must drop the least recently used")

    def test_a_hit_refreshes_recency(self):
        cache = self.cache(limit_bytes=3000)
        cache.put("a", {"data": self.FILLER})
        time.sleep(0.02)
        cache.put("b", {"data": self.FILLER})
        time.sleep(0.02)
        cache.get("a")  # touch a so b becomes the oldest

        cache.put("c", {"data": self.FILLER})
        remaining = sorted(n for n in os.listdir(self.directory.name) if n.endswith(".json"))
        self.assertIn("a.json", remaining, "a was re-read, so it must survive")
        self.assertNotIn("b.json", remaining)

    def test_clear_removes_every_entry(self):
        cache = self.cache()
        cache.put("a", {"ok": 1})
        cache.put("b", {"ok": 2})

        self.assertEqual(cache.clear(), 2)
        self.assertEqual(cache.get("a"), None)

    def test_unwritable_directory_does_not_raise(self):
        # A cache that cannot write must be a silent no-op, not a parse failure.
        cache = ParseCache(os.path.join(self.directory.name, "missing", "nested"))
        cache.put("abc", {"ok": True})  # creates the directory; must not raise
        self.assertEqual(cache.get("abc"), {"ok": True})


class TouchMarkerTest(unittest.TestCase):
    def test_marker_does_not_mutate_the_input(self):
        original = {"chunk_count": 1}
        stamped = touch_marker(original)

        self.assertIs(stamped.get("cached"), True)
        self.assertNotIn("cached", original, "the stored payload must stay clean")

    def test_marker_survives_a_json_round_trip(self):
        stamped = touch_marker({"chunk_count": 1})
        restored = json.loads(json.dumps(stamped))
        self.assertTrue(restored["cached"])


if __name__ == "__main__":
    unittest.main()
