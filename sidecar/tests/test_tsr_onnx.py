"""Tests for the TSR detector's pure logic.

Preprocessing, decoding and alignment are exercised directly — no model file and
no ONNX session, so these stay fast and run anywhere.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tsr_onnx import (  # noqa: E402
    CANDIDATES,
    INPUT_SIZE,
    LABELS,
    align,
    blob_for,
    decode,
)


def box(label, x0, top, x1, bottom, score=0.5):
    return {"label": label, "score": score, "bbox": (x0, top, x1, bottom)}


def raw_output(rows):
    """Build a (1, 11, 8400) model output from explicit candidates.

    Each row is ``(cx, cy, w, h, {class_index: score})`` in the 640 canvas.
    """
    out = np.zeros((1, 11, CANDIDATES), dtype=np.float32)
    for anchor, (cx, cy, w, h, scores) in enumerate(rows):
        out[0, 0, anchor] = cx
        out[0, 1, anchor] = cy
        out[0, 2, anchor] = w
        out[0, 3, anchor] = h
        for class_index, score in scores.items():
            out[0, 4 + class_index, anchor] = score
    return out


class BlobTest(unittest.TestCase):
    def test_shape_and_normalisation(self):
        blob = blob_for(np.full((200, 400, 3), 255, dtype=np.uint8))
        self.assertEqual(blob.shape, (1, 3, INPUT_SIZE, INPUT_SIZE))
        self.assertAlmostEqual(float(blob.min()), 1.0)
        self.assertAlmostEqual(float(blob.max()), 1.0)

    def test_channels_are_reversed_to_bgr(self):
        # Solid red. After the RGB->BGR swap the first channel is blue, so it must
        # be zero and the third must be one. Getting this backwards would feed
        # the model swapped channels and quietly degrade every detection.
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255

        blob = blob_for(rgb)

        self.assertAlmostEqual(float(blob[0, 0].max()), 0.0)
        self.assertAlmostEqual(float(blob[0, 2].min()), 1.0)


class DecodeTest(unittest.TestCase):
    def test_scales_centre_boxes_back_to_crop_pixels(self):
        # The model works on a 640x640 canvas, so a 1280-wide crop means every
        # coordinate doubles on the way back.
        raw = raw_output([(100.0, 100.0, 50.0, 20.0, {0: 0.9})])

        boxes = decode(raw, 1280, 640, 0.2)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "table")
        x0, y0, x1, y1 = boxes[0]["bbox"]
        self.assertAlmostEqual(x0, 150.0)
        self.assertAlmostEqual(x1, 250.0)
        self.assertAlmostEqual(y0, 90.0)
        self.assertAlmostEqual(y1, 110.0)

    def test_drops_candidates_below_the_threshold(self):
        raw = raw_output([(100.0, 100.0, 50.0, 20.0, {0: 0.1})])
        self.assertEqual(decode(raw, 640, 640, 0.2), [])

    def test_drops_the_unused_seventh_class(self):
        # The model emits 7 score features but only 6 are real classes. Class 6
        # must not become a box with an out-of-range label.
        raw = raw_output([(100.0, 100.0, 50.0, 20.0, {6: 0.99})])
        self.assertEqual(decode(raw, 640, 640, 0.2), [])

    def test_picks_the_highest_scoring_class(self):
        raw = raw_output([(100.0, 100.0, 50.0, 20.0, {0: 0.3, 2: 0.8})])

        boxes = decode(raw, 640, 640, 0.2)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], LABELS[2])

    def test_nms_keeps_the_most_confident_of_overlapping_boxes(self):
        # Confidence has to sit where _nms reads it (index 4); parked elsewhere,
        # NMS suppresses by arbitrary order and the survivor is a coin flip.
        raw = raw_output([
            (100.0, 100.0, 50.0, 20.0, {0: 0.30}),
            (100.5, 100.0, 50.0, 20.0, {0: 0.95}),
        ])

        boxes = decode(raw, 640, 640, 0.2)

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["score"], 0.95)

    def test_keeps_boxes_of_different_classes_apart(self):
        # NMS is per class: a row and a column box occupying the same pixels are
        # both real, and suppressing one would lose a whole axis of the table.
        raw = raw_output([
            (100.0, 100.0, 50.0, 20.0, {1: 0.9}),
            (100.0, 100.0, 50.0, 20.0, {2: 0.9}),
        ])

        boxes = decode(raw, 640, 640, 0.2)

        self.assertEqual(len(boxes), 2)
        self.assertEqual({b["label"] for b in boxes}, {"table column", "table row"})

    def test_empty_output_yields_nothing(self):
        self.assertEqual(decode(np.zeros((1, 11, CANDIDATES), dtype=np.float32), 640, 640, 0.2), [])


class AlignTest(unittest.TestCase):
    def test_clips_row_edges_to_the_extremes(self):
        # With 4 or fewer boxes the bounds are the plain extremes, so an edge
        # already at the bound is left exactly as detected.
        boxes = [
            box("table row", 20, 0, 80, 10),
            box("table row", 0, 10, 100, 20),
            box("table row", 30, 20, 90, 30),
        ]

        align(boxes)

        self.assertEqual(boxes[0]["bbox"], (0, 0, 100, 10))
        self.assertEqual(boxes[1]["bbox"], (0, 10, 100, 20))
        self.assertEqual(boxes[2]["bbox"], (0, 20, 100, 30))

    def test_an_edge_inside_the_mean_is_left_alone(self):
        # Past four boxes the bound becomes the mean, and the pull is
        # one-sided: a box already inside the bound keeps its detected edge.
        boxes = [box("table row", 10 * i, 0, 10 * i + 5, 10) for i in range(5)]

        align(boxes)

        self.assertEqual(boxes[0]["bbox"][0], 0)    # inside the mean (20)
        self.assertEqual(boxes[4]["bbox"][0], 20)   # 40 was outside, clipped in

    def test_aligns_columns_vertically(self):
        boxes = [
            box("table column", 0, 0, 10, 100),
            box("table column", 90, 5, 100, 90),
        ]

        align(boxes)

        self.assertEqual(boxes[0]["bbox"], (0, 0, 10, 100))
        self.assertEqual(boxes[1]["bbox"], (90, 0, 100, 100))

    def test_ignores_boxes_that_are_neither_rows_nor_columns(self):
        boxes = [box("table spanning cell", 5, 5, 95, 95)]
        align(boxes)
        self.assertEqual(boxes[0]["bbox"], (5, 5, 95, 95))


if __name__ == "__main__":
    unittest.main()
