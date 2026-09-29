"""Tests for the PP-DocLayout detector's pure logic.

Preprocessing and post-processing are tested directly — no model file and no
ONNX session, so these stay fast and run on any machine.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from layout_onnx import (  # noqa: E402
    CROSS_CLASS_COVER_RATIO,
    INPUT_SIZE,
    LABEL_TO_BLOCK_TYPE,
    MAX_BOXES,
    PAD_VALUE,
    YOLO_LABELS,
    _covered_ratio,
    letterbox,
    postprocess,
)


def make_output(rows):
    """Build a (1, 300, 6) model output from explicit detection rows."""
    out = np.zeros((1, MAX_BOXES, 6), dtype=np.float32)
    for index, row in enumerate(rows[:MAX_BOXES]):
        out[0, index] = row
    return out


#: A factor that leaves coordinates untouched (square source, no padding).
IDENTITY_FACTOR = (1.0, 1.0, 0.0, 0.0)


class LetterboxTest(unittest.TestCase):
    def test_landscape_image_pads_only_top_and_bottom(self):
        blob, (scale_x, scale_y, pad_x, pad_y) = letterbox(np.zeros((500, 1000, 3), dtype=np.uint8))

        self.assertEqual(blob.shape, (1, 3, INPUT_SIZE, INPUT_SIZE))
        # The long side (1000 wide) fills the canvas, so only height is padded.
        scale = INPUT_SIZE / 1000
        new_w, new_h = round(1000 * scale), round(500 * scale)
        self.assertEqual(new_w, INPUT_SIZE)
        self.assertEqual(pad_x, 0.0)
        self.assertEqual(pad_y, (INPUT_SIZE - new_h) / 2.0)
        self.assertAlmostEqual(scale_x, 1000 / new_w)
        self.assertAlmostEqual(scale_y, 500 / new_h)

    def test_portrait_image_pads_only_left_and_right(self):
        _, (scale_x, scale_y, pad_x, pad_y) = letterbox(np.zeros((1000, 500, 3), dtype=np.uint8))

        scale = INPUT_SIZE / 1000
        new_w, new_h = round(500 * scale), round(1000 * scale)
        self.assertEqual(new_h, INPUT_SIZE)
        self.assertEqual(pad_y, 0.0)
        self.assertEqual(pad_x, (INPUT_SIZE - new_w) / 2.0)
        self.assertAlmostEqual(scale_x, 500 / new_w)
        self.assertAlmostEqual(scale_y, 1000 / new_h)

    def test_padding_uses_114(self):
        blob, (_, _, pad_x, pad_y) = letterbox(np.full((100, 1000, 3), 255, dtype=np.uint8))

        # A 100-tall image is padded vertically; the top-left pixel is padding.
        self.assertEqual(pad_x, 0.0)
        self.assertGreater(pad_y, 0)
        self.assertAlmostEqual(float(blob[0, 0, 0, 0]) * 255.0, PAD_VALUE, delta=1.0)

    def test_channels_are_bgr(self):
        # Pure red RGB must land in the model's last channel (BGR order).
        red = np.zeros((1000, 1000, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        blob, _ = letterbox(red)

        middle = INPUT_SIZE // 2
        self.assertAlmostEqual(float(blob[0, 0, middle, middle]), 0.0, delta=0.01)
        self.assertAlmostEqual(float(blob[0, 2, middle, middle]), 1.0, delta=0.01)

    def test_values_are_normalised(self):
        blob, _ = letterbox(np.full((1000, 1000, 3), 255, dtype=np.uint8))
        self.assertLessEqual(float(blob.max()), 1.0)
        self.assertGreater(float(blob.min()), 0.9)


class CoveredRatioTest(unittest.TestCase):
    def test_identical_box_is_fully_covered(self):
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(_covered_ratio(box, [box]), 1.0)

    def test_disjoint_box_is_not_covered(self):
        self.assertAlmostEqual(_covered_ratio((0, 0, 10, 10), [(20, 20, 30, 30)]), 0.0)

    def test_half_overlap_is_counted_once(self):
        # The right half of the box is covered, exactly once despite two donors.
        ratio = _covered_ratio((0, 0, 10, 10), [(5, 0, 15, 10), (5, 0, 12, 10)])
        self.assertAlmostEqual(ratio, 0.5, places=4)

    def test_degenerate_box_counts_as_covered(self):
        self.assertEqual(_covered_ratio((0, 0, 0, 10), [(0, 0, 1, 1)]), 1.0)


class PostprocessTest(unittest.TestCase):
    def test_maps_coordinates_and_keeps_high_scores(self):
        out = make_output([(10, 20, 110, 220, 0.9, 1)])
        detections = postprocess(out, IDENTITY_FACTOR)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["block_type"], "Text")
        self.assertEqual(detections[0]["label"], "Text")
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=4)
        self.assertEqual(tuple(round(v) for v in detections[0]["bbox"]), (10, 20, 110, 220))

    def test_undoes_letterbox_geometry(self):
        # Model saw a half-scale canvas padded by 100; the source box is twice as
        # big and shifted back by the padding.
        detections = postprocess(make_output([(100, 100, 200, 200, 0.9, 5)]), (2.0, 2.0, 100.0, 100.0))
        self.assertEqual(tuple(round(v) for v in detections[0]["bbox"]), (0, 0, 200, 200))

    def test_drops_detections_at_or_below_threshold(self):
        out = make_output([(0, 0, 10, 10, 0.08, 1), (0, 0, 10, 10, 0.079, 1)])
        self.assertEqual(len(postprocess(out, IDENTITY_FACTOR)), 0)

    def test_suppresses_overlapping_boxes_within_a_class(self):
        out = make_output([
            (0, 0, 100, 100, 0.9, 1),
            (5, 5, 100, 100, 0.7, 1),  # IoU well above 0.45 -> suppressed
        ])
        detections = postprocess(out, IDENTITY_FACTOR)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=4)

    def test_drops_a_duplicate_across_classes(self):
        # The real failure mode: one region labelled twice, keeping the stronger.
        out = make_output([
            (10, 10, 200, 60, 0.60, 1),  # Text
            (10, 10, 200, 60, 0.11, 2),  # Reference, identical box
        ])
        detections = postprocess(out, IDENTITY_FACTOR)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["block_type"], "Text")
        self.assertGreater(CROSS_CLASS_COVER_RATIO, 0.11)

    def test_keeps_partially_overlapping_boxes(self):
        # A 30% overlap is normal for adjacent blocks and must survive.
        out = make_output([
            (0, 0, 100, 100, 0.9, 1),
            (70, 0, 170, 100, 0.8, 1),
        ])
        self.assertEqual(len(postprocess(out, IDENTITY_FACTOR)), 2)

    def test_keeps_boxes_that_only_touch(self):
        out = make_output([
            (0, 0, 100, 100, 0.9, 1),
            (100, 0, 200, 100, 0.8, 1),
        ])
        self.assertEqual(len(postprocess(out, IDENTITY_FACTOR)), 2)

    def test_ignores_out_of_range_class_ids(self):
        out = make_output([
            (0, 0, 10, 10, 0.9, 99),
            (20, 20, 30, 30, 0.9, -1),
        ])
        self.assertEqual(postprocess(out, IDENTITY_FACTOR), [])

    def test_orders_results_by_reading_position(self):
        out = make_output([
            (0, 500, 100, 600, 0.9, 1),  # lower on the page, emitted first
            (0, 10, 100, 50, 0.9, 1),  # higher
        ])
        detections = postprocess(out, IDENTITY_FACTOR)
        self.assertEqual([round(d["bbox"][1]) for d in detections], [10, 500])

    def test_clamps_reads_to_max_boxes(self):
        rows = [(0, i, 10, i + 5, 0.9, 1) for i in range(MAX_BOXES + 5)]
        self.assertLessEqual(len(postprocess(make_output(rows), IDENTITY_FACTOR)), MAX_BOXES)


class LabelMappingTest(unittest.TestCase):
    def test_every_label_maps_to_a_block_type(self):
        for label in YOLO_LABELS:
            self.assertIn(label.lower(), LABEL_TO_BLOCK_TYPE, "unmapped label: %s" % label)

    def test_caption_labels_collapse_onto_caption(self):
        self.assertEqual(LABEL_TO_BLOCK_TYPE["figure caption"], "Caption")
        self.assertEqual(LABEL_TO_BLOCK_TYPE["table caption"], "Caption")


if __name__ == "__main__":
    unittest.main()
