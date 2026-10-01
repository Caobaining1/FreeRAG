"""Integration test: a real PDF through the full parse pipeline (plan.md §5.1)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import parse_document  # noqa: E402

try:
    import pymupdf

    HAVE_PYMUPDF = True
except Exception:  # pragma: no cover - environment without the wheel
    HAVE_PYMUPDF = False


def _write_sample_pdf(path: str) -> None:
    """Two pages: a heading, body paragraphs and a ruled 3x3 table."""
    doc = pymupdf.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "Freerag Test Report", fontsize=22)
    page.insert_textbox(
        pymupdf.Rect(72, 120, 520, 190),
        "This is the first body paragraph. It is long enough to be treated as body text "
        "rather than a heading, and it ends without a closing marker so a following "
        "fragment could be appended to it.",
        fontsize=11,
    )
    page.insert_textbox(
        pymupdf.Rect(72, 200, 520, 250),
        "The second paragraph is separate and closes properly. It exists so the parser "
        "has more than one body block to order and section.",
        fontsize=11,
    )

    x0, y0, col_w, row_h, rows, cols = 72.0, 300.0, 120.0, 24.0, 3, 3
    for i in range(rows + 1):
        y = y0 + i * row_h
        page.draw_line((x0, y), (x0 + cols * col_w, y), width=0.8)
    for j in range(cols + 1):
        x = x0 + j * col_w
        page.draw_line((x, y0), (x, y0 + rows * row_h), width=0.8)
    for i in range(rows):
        for j in range(cols):
            page.insert_text((x0 + j * col_w + 6, y0 + i * row_h + 16), f"r{i}c{j}", fontsize=10)

    second = doc.new_page(width=595, height=842)
    second.insert_textbox(
        pymupdf.Rect(72, 90, 520, 160),
        "Page two carries a single paragraph of body text so that multi-page assembly "
        "and page numbering can be verified end to end.",
        fontsize=11,
    )

    doc.save(path)
    doc.close()


@unittest.skipUnless(HAVE_PYMUPDF, "pymupdf is not installed")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.pdf = os.path.join(cls._tmp.name, "report.pdf")
        _write_sample_pdf(cls.pdf)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_parses_pages_and_produces_chunks(self):
        result = parse_document(self.pdf)
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["pages_parsed"], 2)
        self.assertGreater(result["block_count"], 0)
        self.assertGreater(result["chunk_count"], 0)
        self.assertEqual(result["source_file"], "report.pdf")

    def test_heading_heads_a_section(self):
        result = parse_document(self.pdf)
        sections = [c for c in result["chunks"] if c["metadata"]["block_type"] == "Section"]
        self.assertTrue(sections, "expected the large-font heading to head a Section chunk")
        section = sections[0]
        self.assertIn("Freerag Test Report", section["text"])
        self.assertIn("first body paragraph", section["text"])
        self.assertEqual(section["metadata"]["parent_section"], "Freerag Test Report")

    def test_table_is_detected_and_rendered_as_markdown(self):
        result = parse_document(self.pdf)
        tables = [c for c in result["chunks"] if c["metadata"]["block_type"] == "Table"]
        self.assertTrue(tables, "expected the ruled table to be detected")
        text = tables[0]["text"]
        self.assertIn("| --- |", text)
        self.assertIn("r0c0", text)
        self.assertIn("r2c2", text)

    def test_table_cell_text_is_not_duplicated_as_body(self):
        result = parse_document(self.pdf)
        body = [c for c in result["chunks"] if c["metadata"]["block_type"] == "Text"]
        for chunk in body:
            self.assertNotIn("r0c0", chunk["text"])

    def test_chunks_carry_page_numbers(self):
        result = parse_document(self.pdf)
        pages = {c["metadata"]["page_num"] for c in result["chunks"]}
        self.assertEqual(pages, {1, 2})

    def test_max_pages_limits_work(self):
        result = parse_document(self.pdf, max_pages=1)
        self.assertEqual(result["pages_parsed"], 1)
        self.assertEqual({c["metadata"]["page_num"] for c in result["chunks"]}, {1})

    def test_chunk_ids_are_unique(self):
        result = parse_document(self.pdf)
        ids = [c["chunk_id"] for c in result["chunks"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_document(os.path.join(self._tmp.name, "missing.pdf"))


if __name__ == "__main__":
    unittest.main()
