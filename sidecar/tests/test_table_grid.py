"""Tests for TSR box -> grid -> Markdown arrangement.

Pure logic: no PDF and no model, and cell text arrives through an injected
reader. The arrangement rules are where the mistakes live, so they are checked
directly rather than through a parse.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from table_grid import (  # noqa: E402
    build_grid,
    column_bands,
    header_row_count,
    render,
    row_bands,
    to_markdown,
)


def box(label, x0, top, x1, bottom, score=0.5):
    return {"label": label, "score": score, "bbox": (x0, top, x1, bottom)}


def three_by_three():
    """A 3x3 grid: three columns, three rows, no header band."""
    return [
        box("table", 0, 0, 300, 90),
        box("table column", 0, 0, 100, 90),
        box("table column", 100, 0, 200, 90),
        box("table column", 200, 0, 300, 90),
        box("table row", 0, 0, 300, 30),
        box("table row", 0, 30, 300, 60),
        box("table row", 0, 60, 300, 90),
    ]


class RowBandTest(unittest.TestCase):
    def test_orders_rows_top_to_bottom(self):
        boxes = [box("table row", 0, 60, 300, 90), box("table row", 0, 0, 300, 30)]
        self.assertEqual(row_bands(boxes), [(0, 30), (60, 90)])

    def test_merges_a_band_detected_twice(self):
        # Two boxes over nearly the same band are one row, not two: emitting both
        # would put a near-empty row into the output.
        boxes = [box("table row", 0, 0, 300, 30), box("table row", 0, 2, 300, 29)]
        self.assertEqual(row_bands(boxes), [(0, 30)])

    def test_keeps_adjacent_bands_separate(self):
        boxes = [box("table row", 0, 0, 300, 30), box("table row", 0, 30, 300, 60)]
        self.assertEqual(len(row_bands(boxes)), 2)

    def test_projected_row_headers_are_bands_too(self):
        boxes = [box("table row", 0, 0, 300, 30), box("table projected row header", 0, 30, 300, 60)]
        self.assertEqual(len(row_bands(boxes)), 2)


class ColumnBandTest(unittest.TestCase):
    def test_splits_an_overlap_at_the_midpoint(self):
        # TSR columns routinely overlap a few pixels. Left alone, a word
        # straddling the seam is clipped into both cells and appears twice.
        boxes = [box("table column", 0, 0, 110, 90), box("table column", 90, 0, 200, 90)]
        self.assertEqual(column_bands(boxes), [(0.0, 100.0), (100.0, 200.0)])

    def test_leaves_disjoint_columns_alone(self):
        boxes = [box("table column", 0, 0, 100, 90), box("table column", 100, 0, 200, 90)]
        self.assertEqual(column_bands(boxes), [(0.0, 100.0), (100.0, 200.0)])


class HeaderTest(unittest.TestCase):
    def test_counts_the_leading_row_a_header_band_covers(self):
        boxes = three_by_three() + [box("table column header", 0, 0, 300, 30)]
        self.assertEqual(header_row_count(boxes, row_bands(boxes)), 1)

    def test_counts_a_wrapped_header_spanning_two_rows(self):
        boxes = three_by_three() + [box("table column header", 0, 0, 300, 60)]
        self.assertEqual(header_row_count(boxes, row_bands(boxes)), 2)

    def test_ignores_a_header_that_is_not_at_the_top(self):
        # A header band further down is a mid-table heading. Treating it as the
        # Markdown header would promote a data row into the header line.
        boxes = three_by_three() + [box("table column header", 0, 60, 300, 90)]
        self.assertEqual(header_row_count(boxes, row_bands(boxes)), 0)

    def test_zero_without_a_header_box(self):
        boxes = three_by_three()
        self.assertEqual(header_row_count(boxes, row_bands(boxes)), 0)


class BuildGridTest(unittest.TestCase):
    def reader(self, cells):
        return lambda cell: cells.get(cell, "")

    def test_one_cell_per_intersection(self):
        boxes = three_by_three()
        cells = {}
        for r, top in enumerate((0, 30, 60)):
            for c, left in enumerate((0, 100, 200)):
                cells[(left, top, left + 100, top + 30)] = "r%dc%d" % (r, c)

        grid = build_grid(boxes, row_bands(boxes), column_bands(boxes), self.reader(cells))

        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[0], ["r0c0", "r0c1", "r0c2"])
        self.assertEqual(grid[2][1], "r2c1")

    def test_collapses_whitespace_and_escapes_pipes(self):
        boxes = three_by_three()
        grid = build_grid(boxes, row_bands(boxes), column_bands(boxes), lambda cell: " a\n b | c ")
        self.assertEqual(grid[0][0], "a b \\| c")


class RenderTest(unittest.TestCase):
    def test_rejects_a_single_band(self):
        # One band is a line, not a grid. Emitting it would be worse than
        # falling back to the region's plain text.
        boxes = [
            box("table", 0, 0, 200, 30),
            box("table row", 0, 0, 200, 30),
            box("table column", 0, 0, 100, 30),
            box("table column", 100, 0, 200, 30),
        ]
        result = render(boxes, lambda cell: "x")
        self.assertEqual(result["markdown"], "")
        self.assertEqual(result["rows"], 1)

    def test_reports_the_shape(self):
        boxes = three_by_three()
        result = render(boxes, lambda cell: "v")
        self.assertEqual((result["rows"], result["columns"]), (3, 3))
        self.assertEqual(result["cells"], 9)


class MarkdownTest(unittest.TestCase):
    def test_blank_header_keeps_every_row_as_data(self):
        # With no header detected, a blank header line is valid Markdown, while
        # promoting row 1 would mislabel data as a heading.
        out = to_markdown([["a", "b"], ["c", "d"]], 0)
        self.assertEqual(out.splitlines(), [
            "|  |  |",
            "| --- | --- |",
            "| a | b |",
            "| c | d |",
        ])

    def test_folds_a_multi_row_header_into_one_line(self):
        # Markdown has exactly one header row; dropping the extra ones would
        # lose the distinctions that made them separate.
        out = to_markdown([["A", "B"], ["C", "D"], ["e", "f"]], 2)
        self.assertEqual(out.splitlines()[0], "| A C | B D |")
        self.assertEqual(out.splitlines()[-1], "| e | f |")

    def test_pads_ragged_rows(self):
        # An inconsistent column count makes most Markdown renderers drop the
        # whole table.
        out = to_markdown([["a", "b"], ["c"]], 0)
        for line in out.splitlines():
            self.assertEqual(line.count("|"), 3)

    def test_empty_grid_renders_nothing(self):
        self.assertEqual(to_markdown([], 0), "")


if __name__ == "__main__":
    unittest.main()
