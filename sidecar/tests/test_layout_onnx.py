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
    _drop_contained,
    default_providers,
    letterbox,
    postprocess,
    reading_order,
    reset_session_cache,
    session_for,
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


class ContainedBoxTest(unittest.TestCase):
    """A box inside a larger box of its own type is one region found twice."""

    @staticmethod
    def block(x0, y0, x1, y1, name, score=0.9):
        return {"block_type": name, "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "score": score}

    def test_a_nested_box_of_the_same_type_is_dropped(self):
        # The measured case: the small box scores higher, so the coverage rule
        # claims it first and then keeps the large one too.
        outer = self.block(64, 199, 549, 250, "Text", score=0.90)
        inner = self.block(393, 199, 547, 261, "Text", score=0.93)
        kept = _drop_contained([inner, outer])

        self.assertEqual([d["bbox"] for d in kept], [outer["bbox"]])

    def test_a_nested_box_of_another_type_is_kept(self):
        # A caption inside the figure it labels is not a duplicate.
        figure = self.block(50, 50, 500, 400, "Figure")
        caption = self.block(60, 60, 480, 90, "Caption")
        self.assertEqual(len(_drop_contained([figure, caption])), 2)

    def test_a_partly_overlapping_box_is_kept(self):
        # Half-covered is two regions touching, not one inside the other.
        left = self.block(0, 0, 100, 100, "Text")
        right = self.block(50, 0, 150, 100, "Text")
        self.assertEqual(len(_drop_contained([left, right])), 2)


class ProviderChoiceTest(unittest.TestCase):
    """Which execution provider the models run on (FREERAG_ONNX_PROVIDERS)."""

    def setUp(self):
        self.saved = os.environ.get('FREERAG_ONNX_PROVIDERS')

    def tearDown(self):
        if self.saved is None:
            os.environ.pop('FREERAG_ONNX_PROVIDERS', None)
        else:
            os.environ['FREERAG_ONNX_PROVIDERS'] = self.saved

    def test_the_default_ends_on_cpu_so_inference_always_has_somewhere_to_run(self):
        """The default is "auto": whatever this machine can provide, CPU last.

        Asserted as "ends on CPU" rather than "is CPU" so the test holds on a
        machine with an accelerator and on one without — the same property the
        setting is for.
        """
        os.environ.pop('FREERAG_ONNX_PROVIDERS', None)
        providers = default_providers()
        self.assertTrue(providers)
        self.assertEqual(providers[-1], 'CPUExecutionProvider')

    def test_auto_puts_cpu_last_so_inference_still_has_somewhere_to_run(self):
        os.environ['FREERAG_ONNX_PROVIDERS'] = 'auto'
        providers = default_providers()
        self.assertEqual(providers[-1], 'CPUExecutionProvider')
        self.assertTrue(providers)

    def test_an_explicit_list_is_honoured_in_order(self):
        os.environ['FREERAG_ONNX_PROVIDERS'] = 'CPUExecutionProvider'
        self.assertEqual(default_providers(), ['CPUExecutionProvider'])

    def test_an_unavailable_provider_still_leaves_cpu(self):
        os.environ['FREERAG_ONNX_PROVIDERS'] = 'NoSuchExecutionProvider'
        self.assertEqual(default_providers(), ['CPUExecutionProvider'])

    def test_an_empty_setting_falls_back_to_cpu(self):
        os.environ['FREERAG_ONNX_PROVIDERS'] = '   '
        self.assertEqual(default_providers(), ['CPUExecutionProvider'])


class SessionCacheTest(unittest.TestCase):
    """A session is built once per process, not once per document.

    Regression guard for the 6.9s CoreML compile that used to be paid for every
    document (docs/plan.md §5.4): 60 corpus documents spent 414s of 505s
    re-compiling the same model. The build callable stands in for
    onnxruntime.InferenceSession so this runs without a model file.
    """

    def setUp(self):
        reset_session_cache()

    def tearDown(self):
        reset_session_cache()

    def test_one_configuration_is_built_once_and_shared(self):
        builds = []

        def build():
            builds.append(1)
            return object()

        first = session_for('layout.onnx', ['CPUExecutionProvider'], 1, build=build)
        again = session_for('layout.onnx', ['CPUExecutionProvider'], 1, build=build)

        self.assertEqual(len(builds), 1, 'the session was built more than once')
        self.assertIs(first, again)

    def test_a_different_configuration_gets_its_own_session(self):
        # Each of these changes what gets compiled, so sharing would be wrong.
        keys = [
            ('layout.onnx', ['CPUExecutionProvider'], 1),
            ('layout.onnx', ['CPUExecutionProvider'], 2),
            ('layout.onnx', ['CoreMLExecutionProvider', 'CPUExecutionProvider'], 1),
            ('table.onnx', ['CPUExecutionProvider'], 1),
        ]
        sessions = [session_for(path, providers, threads, build=object) for path, providers, threads in keys]
        self.assertEqual(len({id(s) for s in sessions}), len(keys))

    def test_the_provider_order_is_part_of_the_key(self):
        # [A, B] and [B, A] run inference on different hardware.
        one = session_for('m.onnx', ['CoreMLExecutionProvider', 'CPUExecutionProvider'], 1, build=object)
        two = session_for('m.onnx', ['CPUExecutionProvider', 'CoreMLExecutionProvider'], 1, build=object)
        self.assertIsNot(one, two)

    def test_concurrent_callers_build_it_once(self):
        # The lock is the whole point: an unlocked cache would compile the model
        # once per thread, which is exactly the waste this replaced.
        import threading
        import time

        builds = []

        def slow_build():
            time.sleep(0.05)
            builds.append(1)
            return object()

        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(session_for('m.onnx', ['CPUExecutionProvider'], 1, build=slow_build))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(builds), 1, 'built %d times across 8 threads' % len(builds))
        self.assertEqual(len({id(s) for s in results}), 1)


class ReadingOrderTest(unittest.TestCase):
    """XY-cut ordering, tested without a model (see layout_onnx.reading_order)."""

    @staticmethod
    def block(x0, y0, x1, y1, name):
        return {"block_type": name, "bbox": (float(x0), float(y0), float(x1), float(y1))}

    def names(self, detections):
        return [d["block_type"] for d in detections]

    def test_two_columns_are_read_column_by_column(self):
        # Left and right column each hold two stacked paragraphs. A (y, x) sort
        # would give L1, R1, L2, R2 — the interleaving that merged the columns.
        detections = [
            self.block(0, 0, 100, 50, "L1"),
            self.block(150, 0, 250, 50, "R1"),
            self.block(0, 70, 100, 120, "L2"),
            self.block(150, 70, 250, 120, "R2"),
        ]
        self.assertEqual(self.names(reading_order(detections)), ["L1", "L2", "R1", "R2"])

    def test_a_full_width_heading_stays_above_both_columns(self):
        # The heading spans the gutter, so no vertical cut is possible until it
        # has been split off — this is what keeps it first.
        detections = [
            self.block(0, 0, 250, 40, "H"),
            self.block(0, 60, 100, 110, "L1"),
            self.block(150, 60, 250, 110, "R1"),
            self.block(0, 130, 100, 180, "L2"),
            self.block(150, 130, 250, 180, "R2"),
        ]
        self.assertEqual(
            self.names(reading_order(detections)), ["H", "L1", "L2", "R1", "R2"]
        )

    def test_single_column_keeps_top_to_bottom_order(self):
        detections = [
            self.block(0, 200, 300, 260, "third"),
            self.block(0, 0, 300, 60, "first"),
            self.block(0, 100, 300, 160, "second"),
        ]
        self.assertEqual(
            self.names(reading_order(detections)), ["first", "second", "third"]
        )

    def test_a_sub_threshold_gap_is_not_a_column_break(self):
        # 2 pt apart is line spacing, not a gutter: the left block stays first.
        detections = [
            self.block(102, 0, 200, 50, "right"),
            self.block(0, 0, 100, 50, "left"),
        ]
        self.assertEqual(self.names(reading_order(detections)), ["left", "right"])

    def test_rows_separated_by_a_wide_band_read_top_to_bottom(self):
        detections = [
            self.block(400, 300, 500, 360, "high-right"),
            self.block(0, 0, 200, 100, "low-left"),
        ]
        self.assertEqual(
            self.names(reading_order(detections)), ["low-left", "high-right"]
        )


class LabelMappingTest(unittest.TestCase):
    def test_every_label_maps_to_a_block_type(self):
        for label in YOLO_LABELS:
            self.assertIn(label.lower(), LABEL_TO_BLOCK_TYPE, "unmapped label: %s" % label)

    def test_caption_labels_collapse_onto_caption(self):
        self.assertEqual(LABEL_TO_BLOCK_TYPE["figure caption"], "Caption")
        self.assertEqual(LABEL_TO_BLOCK_TYPE["table caption"], "Caption")


if __name__ == "__main__":
    unittest.main()
