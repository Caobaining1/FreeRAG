"""The render method: one page of a source document, as a PNG for the UI.

The desktop's chunk inspector draws a chunk's box over the page it was cut from,
and that box is measured in PDF POINTS. These tests pin the two properties that
make the overlay line up — the page size comes back in points, and page numbers
are 1-based — because getting either wrong draws every highlight in the wrong
place, which looks like bad index data rather than like a UI bug.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parse_server import INVALID_PARAMS, NOT_FOUND, RPCError, method_render  # noqa: E402

try:
    import pymupdf

    HAVE_PYMUPDF = True
except Exception:  # pragma: no cover - environment without the wheel
    HAVE_PYMUPDF = False


def _write_two_page_pdf(path: str) -> None:
    document = pymupdf.open()
    for index in (1, 2):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 90), "Page %d" % index, fontsize=22)
    document.save(path)
    document.close()


@unittest.skipUnless(HAVE_PYMUPDF, "pymupdf is not installed")
class RenderMethodTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = os.path.join(self.tmp.name, "two-pages.pdf")
        _write_two_page_pdf(self.pdf)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_renders_a_page_with_its_size_in_points(self) -> None:
        result = method_render({"path": self.pdf, "page": 2})

        # Points describe the PAGE, pixels describe the image, and it is points
        # the UI divides a bbox by — so both have to be reported and they have to
        # disagree.
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["pages"], 2)
        self.assertAlmostEqual(result["width_pt"], 595.0, places=1)
        self.assertAlmostEqual(result["height_pt"], 842.0, places=1)
        self.assertGreater(result["width_px"], result["width_pt"])
        self.assertGreater(result["height_px"], result["height_pt"])

        image = base64.b64decode(result["image"])
        self.assertTrue(image.startswith(b"\x89PNG"), "the payload must be a PNG")

    def test_page_numbers_are_one_based(self) -> None:
        # Page 1 has to be the FIRST page: chunk.page_num is 1-based, so an
        # off-by-one here would put every highlight on the wrong page.
        first = method_render({"path": self.pdf, "page": 1})
        second = method_render({"path": self.pdf, "page": 2})
        self.assertEqual(first["page"], 1)
        self.assertNotEqual(first["image"], second["image"], "pages must differ")

    def test_dpi_changes_the_raster_not_the_points(self) -> None:
        small = method_render({"path": self.pdf, "page": 1, "dpi": 72})
        large = method_render({"path": self.pdf, "page": 1, "dpi": 200})

        self.assertAlmostEqual(small["width_pt"], large["width_pt"], places=1)
        self.assertGreater(large["width_px"], small["width_px"])

    def test_rejects_a_page_outside_the_document(self) -> None:
        with self.assertRaises(RPCError) as caught:
            method_render({"path": self.pdf, "page": 3})
        self.assertEqual(caught.exception.code, NOT_FOUND)

    def test_reports_a_missing_file_as_not_found(self) -> None:
        # The file can move after it was indexed. That is a normal outcome, and
        # it has its own code so the UI can say so instead of showing a blank
        # frame.
        with self.assertRaises(RPCError) as caught:
            method_render({"path": os.path.join(self.tmp.name, "gone.pdf"), "page": 1})
        self.assertEqual(caught.exception.code, NOT_FOUND)

    def test_rejects_bad_arguments(self) -> None:
        for params in (
            "not an object",
            {"page": 1},
            {"path": self.pdf},
            {"path": self.pdf, "page": 0},
            {"path": self.pdf, "page": -1},
            {"path": self.pdf, "page": 1, "dpi": 5000},
            {"path": self.pdf, "page": 1, "dpi": 1},
        ):
            with self.assertRaises(RPCError) as caught:
                method_render(params)
            self.assertEqual(caught.exception.code, INVALID_PARAMS, params)


if __name__ == "__main__":
    unittest.main()
