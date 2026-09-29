"""PP-DocLayout-S layout detection through ONNX Runtime (plan.md §5.1 step ②).

This is a port of RAGFlow's YOLOv10 layout recognizer — the Go rewrite in
``internal/deepdoc/native/dla.go``, itself a port of
``deepdoc/vision/layout_recognizer.py::LayoutRecognizer4YOLOv10``:

  * letterbox the page to 1024x1024, filled with 114, channels in **BGR**;
  * the model emits 300 detections of ``[x0, y0, x1, y1, score, class_id]``;
  * drop anything under score 0.08, run per-class NMS at IoU 0.45;
  * map model coordinates back through the letterbox geometry.

Only detection lives here. Text extraction stays in ``layout.py`` so the ONNX and
PyMuPDF providers share one code path — that is what keeps the two swappable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: Model input side length and detection count (dla.go: dlaInputSize / dlaMaxBoxes).
INPUT_SIZE = 1024
MAX_BOXES = 300

#: Letterbox padding value, matching the YOLO family convention (dla.go).
PAD_VALUE = 114

#: Post-processing thresholds (dla.go: scoreThr / nms IoU).
SCORE_THRESHOLD = 0.08
NMS_IOU = 0.45

#: Cross-class coverage above which a box is dropped as a duplicate.
#:
#: RAGFlow runs NMS per class only, so the same region can survive in two classes
#: (measured on a real paper page: an identical box at score 0.60 as Text and
#: 0.11 as Reference) and a coarse low-score box can survive alongside the fine
#: boxes it spans. Chunking would then extract the same text twice, so we drop a
#: lower-scoring box once this much of its area is already claimed. Disjoint
#: blocks — the normal case — never approach this, so the rule only bites on
#: genuine duplicates.
CROSS_CLASS_COVER_RATIO = 0.4


def _covered_ratio(box: Sequence[float], others: List[Sequence[float]]) -> float:
    """Fraction of ``box``'s area covered by ``others`` (exact rectangle union).

    Sweeps vertical strips over the x-coordinates that matter, so overlapping
    neighbours are not double-counted.
    """
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    if x1 <= x0 or y1 <= y0:
        return 1.0

    cuts = {x0, x1}
    for other in others:
        cuts.add(min(max(float(other[0]), x0), x1))
        cuts.add(min(max(float(other[2]), x0), x1))
    xs = sorted(cuts)

    covered = 0.0
    for index in range(len(xs) - 1):
        left, right = xs[index], xs[index + 1]
        if right <= left:
            continue
        spans: List[Tuple[float, float]] = []
        for other in others:
            if float(other[0]) <= left and float(other[2]) >= right:
                top = max(float(other[1]), y0)
                bottom = min(float(other[3]), y1)
                if bottom > top:
                    spans.append((top, bottom))
        spans.sort()

        total = 0.0
        current: List[float] = []
        for top, bottom in spans:
            if not current:
                current = [top, bottom]
            elif top <= current[1]:
                current[1] = max(current[1], bottom)
            else:
                total += current[1] - current[0]
                current = [top, bottom]
        if current:
            total += current[1] - current[0]
        covered += total * (right - left)

    return covered / ((x1 - x0) * (y1 - y0))

#: The model's 10 output classes, in raw YOLO index order.
#:
#: Mirrors ``yoloDlaLabels`` in dla.go, including the duplicate entries at
#: indices 4/7/9 — they are deliberate and must stay element-for-element
#: identical, because the class id is a positional index into this list.
YOLO_LABELS = [
    "title",
    "Text",
    "Reference",
    "Figure",
    "Figure caption",
    "Table",
    "Table caption",
    "Table caption",
    "Equation",
    "Figure caption",
]

#: Model label -> our Block.block_type. Several labels collapse onto one type
#: because chunking only cares about the handling, not the detector's taxonomy.
LABEL_TO_BLOCK_TYPE = {
    "title": "Title",
    "text": "Text",
    "reference": "Reference",
    "figure": "Figure",
    "figure caption": "Caption",
    "table": "Table",
    "table caption": "Caption",
    "equation": "Equation",
}

#: Accelerator providers, in preference order. CPU is always appended last.
#:
#: Measured on an Apple M-series / 10 cores, one 1024x1024 page:
#:   CoreML 0.17s  |  CPU 1.9s (1 thread) .. 7.8s (default threads)
#: CoreML was verified to return byte-identical detections to the CPU provider,
#: so preferring it is a pure speed win when present.
ACCELERATOR_PROVIDERS = (
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
)

#: CPU intra-op threads when no accelerator exists.
#:
#: Counter-intuitive but measured: this model's ops do not parallelise, so extra
#: threads only add contention — 1/2/3/4 threads took 1.9/3.7/5.4/7.4 s per page.
#: Overridable with FREERAG_LAYOUT_THREADS for a machine that behaves otherwise.
DEFAULT_CPU_THREADS = 1


def default_providers() -> List[str]:
    """Available accelerators first, CPU last."""
    import onnxruntime

    available = set(onnxruntime.get_available_providers())
    ordered = [name for name in ACCELERATOR_PROVIDERS if name in available]
    ordered.append("CPUExecutionProvider")
    return ordered


#: Distinct colour per block type, shared by the visualiser.
BLOCK_COLOURS = {
    "Title": (220, 50, 50),
    "Text": (60, 120, 220),
    "Caption": (30, 170, 110),
    "Table": (230, 140, 20),
    "Figure": (150, 60, 200),
    "Equation": (20, 170, 200),
    "Reference": (140, 140, 140),
}

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deepdoc", "layout.onnx"
)


def model_path() -> str:
    """Where the layout model lives, overridable for tests and packaging."""
    return os.environ.get("FREERAG_LAYOUT_MODEL") or DEFAULT_MODEL


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return fallback


def _resize_bilinear(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bilinear resize to an exact size.

    cv2.INTER_LINEAR uses the half-pixel convention implemented here, so this
    reproduces RAGFlow's resize without pulling in OpenCV.
    """
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (new_w, new_h):
        return image

    ys = (np.arange(new_h, dtype=np.float64) + 0.5) * (src_h / new_h) - 0.5
    xs = (np.arange(new_w, dtype=np.float64) + 0.5) * (src_w / new_w) - 0.5
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)

    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    top = image[y0][:, x0] * (1.0 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1.0 - wx) + image[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.uint8)


def letterbox(rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Prepare one RGB page bitmap for the model.

    Returns the ``(1, 3, 1024, 1024)`` float blob and the
    ``(scale_x, scale_y, pad_x, pad_y)`` factor that maps model coordinates back
    to source pixels (dla.go: dlaGeom / dlaLetterbox / dlaScaleFactor).
    """
    src_h, src_w = rgb.shape[:2]
    scale = min(INPUT_SIZE / src_h, INPUT_SIZE / src_w)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_x = (INPUT_SIZE - new_w) / 2.0
    pad_y = (INPUT_SIZE - new_h) / 2.0

    resized = _resize_bilinear(rgb, new_w, new_h)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    # The truncation offset matches dlaLetterbox (round(dh - 0.1)); it keeps the
    # paste aligned with RAGFlow for odd padding amounts.
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    # The model was trained on BGR (dlaLetterbox channel assignment).
    bgr = canvas[:, :, ::-1]
    blob = bgr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

    factor = (src_w / new_w, src_h / new_h, pad_x, pad_y)
    return blob, factor


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[Sequence[float]], iou_threshold: float) -> List[int]:
    """Greedy non-maximum suppression; returns kept indices, best score first."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][4], reverse=True)
    keep: List[int] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if _iou(boxes[best], boxes[i]) <= iou_threshold]
    return keep


def postprocess(raw: np.ndarray, factor: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    """Turn the raw ``(1, 300, 6)`` output into labelled, de-duplicated boxes."""
    rows = np.asarray(raw).reshape(-1, 6)
    scale_x, scale_y, pad_x, pad_y = factor

    by_class: Dict[int, List[Sequence[float]]] = {}
    for row in rows[:MAX_BOXES]:
        score = float(row[4])
        if score <= SCORE_THRESHOLD:
            continue
        # Truncate toward zero: RAGFlow does astype(int), and rounding instead
        # shifts class channels in [2.5, 3.0) from class 2 to class 3.
        class_id = int(row[5])
        if class_id < 0 or class_id >= len(YOLO_LABELS):
            continue

        box = (
            (float(row[0]) - pad_x) * scale_x,
            (float(row[1]) - pad_y) * scale_y,
            (float(row[2]) - pad_x) * scale_x,
            (float(row[3]) - pad_y) * scale_y,
            score,
        )
        by_class.setdefault(class_id, []).append(box)

    detections: List[Dict[str, Any]] = []
    for class_id, boxes in by_class.items():
        label = YOLO_LABELS[class_id]
        block_type = LABEL_TO_BLOCK_TYPE.get(label.lower())
        if block_type is None:
            continue
        for index in _nms(boxes, NMS_IOU):
            box = boxes[index]
            detections.append(
                {
                    "block_type": block_type,
                    "label": label,
                    "bbox": (box[0], box[1], box[2], box[3]),
                    "score": round(box[4], 4),
                }
            )

    # Cross-class de-duplication: strongest first, so a weak duplicate loses.
    detections.sort(key=lambda d: d["score"], reverse=True)
    claimed: List[Sequence[float]] = []
    unique: List[Dict[str, Any]] = []
    for detection in detections:
        if _covered_ratio(detection["bbox"], claimed) >= CROSS_CLASS_COVER_RATIO:
            continue
        claimed.append(detection["bbox"])
        unique.append(detection)

    # Reading order: top to bottom, then left to right.
    unique.sort(key=lambda d: (round(d["bbox"][1], 1), d["bbox"][0]))
    return unique


class LayoutDetector:
    """A reusable PP-DocLayout session.

    Loading the model costs ~75MB of weights, so the session is created once and
    reused across pages; construction is lazy so importing this module is free.
    """

    def __init__(
        self,
        path: str = "",
        *,
        providers: Sequence[str] = (),
        score_threshold: float = SCORE_THRESHOLD,
        cpu_threads: int = 0,
    ) -> None:
        self.path = path or model_path()
        # Empty means "pick automatically" (accelerators first, CPU last).
        self.providers = list(providers)
        self.score_threshold = score_threshold
        self.cpu_threads = cpu_threads or _env_int("FREERAG_LAYOUT_THREADS", DEFAULT_CPU_THREADS)
        self._session: Any = None

    @property
    def available(self) -> bool:
        return os.path.isfile(self.path)

    @property
    def provider_name(self) -> str:
        """Which provider the loaded session actually runs on."""
        session = self._load()
        return ";".join(session.get_providers())

    def _load(self) -> Any:
        if self._session is None:
            import onnxruntime

            if not self.available:
                raise FileNotFoundError(f"layout model not found: {self.path}")

            providers = self.providers or default_providers()
            options = onnxruntime.SessionOptions()
            if providers[0] == "CPUExecutionProvider":
                options.intra_op_num_threads = self.cpu_threads
            self._session = onnxruntime.InferenceSession(
                self.path, sess_options=options, providers=providers
            )
        return self._session

    def detect_bitmap(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Detect layout regions in one RGB page bitmap."""
        session = self._load()
        blob, factor = letterbox(rgb)
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        return postprocess(outputs[0], factor)

    def detect_page(self, page: Any, dpi: int = 150) -> List[Dict[str, Any]]:
        """Detect layout regions on one PyMuPDF page.

        Coordinates are returned in **PDF points**, not render pixels, so they
        line up with the PyMuPDF provider's boxes and with the ``page.rect`` /
        ``get_text(clip=...)`` conventions the rest of the pipeline uses.
        """
        detections = self.detect_bitmap(render_page_rgb(page, dpi=dpi))
        to_points = 72.0 / dpi
        for detection in detections:
            x0, y0, x1, y1 = (float(v) for v in detection["bbox"])
            detection["bbox"] = (x0 * to_points, y0 * to_points, x1 * to_points, y1 * to_points)
        return detections


def render_page_rgb(page: Any, dpi: int = 150) -> np.ndarray:
    """Rasterise a PyMuPDF page to an RGB array (shared by the visualiser)."""
    import pymupdf

    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
