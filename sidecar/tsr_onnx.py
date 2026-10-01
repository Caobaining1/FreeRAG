"""Table structure recognition (PP-Det style ``TableStructureRecognizer``).

Ported from RAGFlow's Go implementation (``internal/deepdoc/native/tsr.go``)
rather than a Python one: the RAGFlow copy in this workspace is the Go rewrite
and ships no Python ``deepdoc``.

The detector answers one question — *where* are the table's rows, columns,
headers and spanning cells — and does not read text. Cell contents come from the
PDF text layer, exactly as layout blocks do (``sidecar/layout.py``), so a scanned
table still needs OCR.

Its input is a **cropped table region**, not a page, so coordinates come back in
crop pixel space and the caller maps them onto the page.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np

from layout_onnx import (
    DEFAULT_CPU_THREADS,
    _env_int,
    _nms,
    _resize_bilinear,
    default_providers,
    session_for,
)

#: Model input side length and candidate count (tsr.go: tsrInputSize / tsrCandidates).
INPUT_SIZE = 640
CANDIDATES = 8400

#: Class order is the model's own (tsr.go: tsrLabels). Note the model emits 11
#: features — 4 box coordinates plus 7 class scores — while only these 6 classes
#: are real, which is why class index 6 is discarded.
LABELS = (
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
)

#: Detection threshold and the per-class NMS IoU (tsr.go: scoreThr, nms(..., 0.2)).
SCORE_THRESHOLD = 0.2
NMS_IOU = 0.2

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deepdoc", "tsr.onnx"
)


def model_path() -> str:
    """Where the TSR model lives, overridable for tests and packaging."""
    return os.environ.get("FREERAG_TSR_MODEL") or DEFAULT_MODEL


def blob_for(rgb: np.ndarray) -> np.ndarray:
    """Build the ``(1, 3, 640, 640)`` input from a crop.

    A plain resize, not a letterbox: the model's scale factor is ``[W/640, H/640]``
    (tsr.go: tsrScaleFactor), which only holds when the crop fills the canvas.
    Channels are BGR, matching the layout model and the Go reference.
    """
    resized = _resize_bilinear(rgb, INPUT_SIZE, INPUT_SIZE)
    bgr = resized[:, :, ::-1]
    return bgr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0


def decode(raw: np.ndarray, width: int, height: int, score_threshold: float) -> List[Dict[str, Any]]:
    """Turn the raw ``(1, 11, 8400)`` output into labelled boxes in crop pixels.

    Vectorised rather than looped: 8400 candidates × 11 features is exactly the
    shape numpy is good at, and the per-candidate Python loop the Go port uses
    would dominate the runtime here.
    """
    features = np.asarray(raw).reshape(11, CANDIDATES)

    scores = features[4:11]
    best = scores.max(axis=0)
    best_class = scores.argmax(axis=0)
    # argmax keeps the first maximum, matching the Go loop's strict ">".
    keep = (best > score_threshold) & (best_class < len(LABELS))
    if not keep.any():
        return []

    scale_x = width / INPUT_SIZE
    scale_y = height / INPUT_SIZE
    cx = features[0][keep] * scale_x
    cy = features[1][keep] * scale_y
    half_w = features[2][keep] * scale_x * 0.5
    half_h = features[3][keep] * scale_y * 0.5
    confidence = best[keep]
    classes = best_class[keep]

    boxes: List[Dict[str, Any]] = []
    for class_index in range(len(LABELS)):
        selected = np.flatnonzero(classes == class_index)
        if selected.size == 0:
            continue
        # _nms reads the score from index 4 and sorts on it, so confidence has to
        # live there — parking it anywhere else makes NMS keep an arbitrary box
        # of each cluster instead of the most confident one.
        candidates = [
            (
                float(cx[i] - half_w[i]),
                float(cy[i] - half_h[i]),
                float(cx[i] + half_w[i]),
                float(cy[i] + half_h[i]),
                float(confidence[i]),
            )
            for i in selected
        ]
        for index in _nms(candidates, NMS_IOU):
            box = candidates[index]
            boxes.append({"label": LABELS[class_index], "score": box[4], "bbox": box[:4]})
    return boxes


def align(boxes: List[Dict[str, Any]]) -> None:
    """Pull rows/headers to the table's horizontal extremes and columns to its
    vertical ones, in place (tsr.go: alignTSR).

    The adjustment is one-sided: an edge is only pulled *in* to the bound, never
    pushed out, so a box that already fits is left exactly as detected.
    """
    rows = [b for b in boxes if "row" in b["label"] or "header" in b["label"]]
    columns = [b for b in boxes if b["label"] == "table column"]

    if rows:
        lefts = [b["bbox"][0] for b in rows]
        rights = [b["bbox"][2] for b in rows]
        left = min(lefts) if len(lefts) <= 4 else float(np.mean(lefts))
        right = max(rights) if len(rights) <= 4 else float(np.mean(rights))
        for box in rows:
            x0, y0, x1, y1 = box["bbox"]
            box["bbox"] = (min(x0, left) if x0 > left else x0, y0,
                           max(x1, right) if x1 < right else x1, y1)

    if columns:
        tops = [b["bbox"][1] for b in columns]
        bottoms = [b["bbox"][3] for b in columns]
        top = min(tops) if len(tops) <= 4 else float(np.median(tops))
        bottom = max(bottoms) if len(bottoms) <= 4 else float(np.median(bottoms))
        for box in columns:
            x0, y0, x1, y1 = box["bbox"]
            box["bbox"] = (x0, min(y0, top) if y0 > top else y0,
                           x1, max(y1, bottom) if y1 < bottom else y1)


class TSRDetector:
    """Runs the TSR model; the session is reused across tables."""

    def __init__(
        self,
        path: str = "",
        *,
        providers: Sequence[str] = (),
        score_threshold: float = SCORE_THRESHOLD,
        cpu_threads: int = 0,
    ) -> None:
        self.path = path or model_path()
        self.providers = list(providers)
        self.score_threshold = score_threshold
        self.cpu_threads = cpu_threads or _env_int("FREERAG_TSR_THREADS", DEFAULT_CPU_THREADS)
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
        """The shared session for this configuration (see layout_onnx.session_for).

        TSR brings its own builder rather than using make_session_options: it
        does not turn the memory pattern off the way the layout model does, and
        that flag was measured on the layout model only. The two models live in
        different files, so their registry entries cannot collide.
        """
        if self._session is None:
            if not self.available:
                raise FileNotFoundError(f"TSR model not found: {self.path}")
            providers = self.providers or default_providers()
            self._session = session_for(
                self.path, providers, self.cpu_threads, build=lambda: self._build(providers)
            )
        return self._session

    def _build(self, providers: Sequence[str]) -> Any:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if providers[0] == "CPUExecutionProvider":
            options.intra_op_num_threads = self.cpu_threads
        return onnxruntime.InferenceSession(
            self.path, sess_options=options, providers=list(providers)
        )

    def detect(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Detect structure in one cropped table bitmap; boxes in crop pixels."""
        session = self._load()
        height, width = rgb.shape[:2]
        outputs = session.run(None, {session.get_inputs()[0].name: blob_for(rgb)})
        boxes = decode(outputs[0], width, height, self.score_threshold)
        align(boxes)
        return boxes


def classes_of(boxes: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count boxes per label; for logging and tests."""
    counts: Dict[str, int] = {}
    for box in boxes:
        counts[box["label"]] = counts.get(box["label"], 0) + 1
    return counts
