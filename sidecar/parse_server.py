#!/usr/bin/env python3
"""freerag parse sidecar.

Speaks JSON-RPC 2.0 over stdio (NDJSON) with the Go kernel — the same protocol
the kernel speaks to the Electron shell (docs/plan.md §3).

Pipeline: see pipeline.py and docs/plan.md §5.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

# Allow `python3 sidecar/parse_server.py` from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documents import SUPPORTED_SUFFIXES  # noqa: E402
from pipeline import parse_document  # noqa: E402

PROTOCOL_VERSION = "2.0"

# JSON-RPC 2.0 standard error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
#: Application error: the requested document does not exist / cannot be opened.
NOT_FOUND = -32001

VERSION = "0.2.0"


class RPCError(Exception):
    """A JSON-RPC error raised by a handler."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _capabilities() -> Dict[str, Any]:
    """What this sidecar can actually do in the current environment."""
    caps: Dict[str, Any] = {
        "pymupdf": False,
        "pp_doclayout": False,
        "laya": False,
        "onnxruntime": False,
        "tsr": False,
        # The formats `parse` accepts, so a caller can tell "unsupported" from a
        # parse failure without guessing at suffixes.
        "formats": list(SUPPORTED_SUFFIXES),
    }
    try:
        import pymupdf  # noqa: F401

        caps["pymupdf"] = True
    except Exception:
        pass
    try:
        import onnxruntime  # noqa: F401

        caps["onnxruntime"] = True
        # Reported rather than inferred from the platform, because it is the
        # answer to "will this machine be accelerated" and it is not derivable:
        # the same macOS build resolves "auto" to CoreML here and to CPU on a
        # machine whose onnxruntime lacks the provider, silently.
        caps["onnx_providers"] = list(onnxruntime.get_available_providers())
    except Exception:
        pass

    from layout_onnx import LayoutDetector

    caps["pp_doclayout"] = LayoutDetector().available
    caps["laya"] = _decider().available
    # TSR is wired in a later phase (plan.md §5.2).
    return caps


#: The Laya session is expensive to build (~1.7 GB of weights), so it is cached
#: for the process lifetime and only built when a decision is first requested.
_DECIDERS_LOCK = threading.Lock()
_DECIDERS: Dict[str, Any] = {}


def _decider() -> Any:
    from laya import LayaDecider

    # Locked, and held across the construction: `serve` answers requests on a
    # thread pool, so two of them can ask for the decider before the first has
    # built it. Unlocked, that check builds the checkpoint twice.
    with _DECIDERS_LOCK:
        if "laya" not in _DECIDERS:
            _DECIDERS["laya"] = LayaDecider()
        return _DECIDERS["laya"]


def method_ping(_params: Any) -> Any:
    return {"pong": True}


def method_version(_params: Any) -> Any:
    return {"name": "freerag-parse-sidecar", "version": VERSION, "capabilities": _capabilities()}


def method_warmup(_params: Any) -> Any:
    """Build the deepdoc sessions now, and report what they landed on.

    Two jobs, and both are the reason this method exists:

    - The accelerated session is expensive to BUILD, not to use: measured here,
      CoreML compiles 657 of the layout model's 681 nodes and takes 7.5 s,
      against 0.05 s on CPU. Left alone that lands on whichever document is
      parsed first, inside a step the UI has already announced — a wait that
      reads as a hang rather than as a one-off. Building it here spends it at
      startup, where nothing is waiting.
    - It answers "what is this machine really running on" with the session's own
      report instead of the platform's guess. `capabilities.onnx_providers` says
      what the runtime offers; this says what the models actually got, which is
      the number that matters and the one to log.

    Serial by construction: one sidecar process, and the session registry builds
    under its own lock, so a second caller waits rather than compiling twice.
    """
    from layout_onnx import LayoutDetector, default_providers
    from tsr_onnx import TSRDetector

    result: Dict[str, Any] = {"providers": default_providers()}
    for name, detector in (("layout", LayoutDetector()), ("tsr", TSRDetector())):
        entry: Dict[str, Any] = {"available": detector.available}
        if detector.available:
            # `provider_name` builds the session as a side effect, which is the
            # work being paid for here; timing it is the whole measurement.
            started = time.perf_counter()
            try:
                entry["provider"] = detector.provider_name
                entry["seconds"] = round(time.perf_counter() - started, 2)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                entry["error"] = str(exc)
        result[name] = entry
    return result


def method_parse(params: Any) -> Any:
    if not isinstance(params, dict):
        raise RPCError(INVALID_PARAMS, "params must be an object")

    path = params.get("path")
    if not path or not isinstance(path, str):
        raise RPCError(INVALID_PARAMS, "params.path is required")

    profile = params.get("profile") or "mixed"
    max_chars = params.get("max_chars")
    max_pages = params.get("max_pages") or 0
    # Vision model for Figure regions, or "" to skip it. Parse-time only.
    vlm_model = params.get("vlm_model") or ""

    if max_chars is not None and (not isinstance(max_chars, int) or max_chars <= 0):
        raise RPCError(INVALID_PARAMS, "params.max_chars must be a positive integer")
    if not isinstance(max_pages, int) or max_pages < 0:
        raise RPCError(INVALID_PARAMS, "params.max_pages must be a non-negative integer")
    if not isinstance(profile, str):
        raise RPCError(INVALID_PARAMS, "params.profile must be a string")
    if not isinstance(vlm_model, str):
        raise RPCError(INVALID_PARAMS, "params.vlm_model must be a string")

    try:
        return parse_document(path, profile=profile, max_chars=max_chars, max_pages=max_pages,
                              vlm_model=vlm_model)
    except FileNotFoundError as exc:
        raise RPCError(NOT_FOUND, str(exc), {"path": path}) from exc
    except ValueError as exc:
        raise RPCError(INVALID_PARAMS, str(exc), {"path": path}) from exc
    except ImportError as exc:
        raise RPCError(INTERNAL_ERROR, f"missing dependency: {exc}", {"hint": "pip install pymupdf"}) from exc


def method_decide(params: Any) -> Any:
    """Answer one typed decision with Laya (docs/plan.md §6.6).

    Kept in this sidecar rather than a second process: it is the same
    onnxruntime and the same model directory, and the kernel already has a warm
    stdio channel to it.
    """
    if not isinstance(params, dict):
        raise RPCError(INVALID_PARAMS, "params must be an object")

    instructions = params.get("instructions")
    if not instructions or not isinstance(instructions, str):
        raise RPCError(INVALID_PARAMS, "params.instructions is required")

    criteria = params.get("criteria")
    if criteria is None or not isinstance(criteria, (dict, list)):
        raise RPCError(INVALID_PARAMS, "params.criteria must be an object or array")

    qtype = params.get("qtype") or "choice"
    if qtype not in ("choice", "score", "noul"):
        raise RPCError(INVALID_PARAMS, "params.qtype must be choice, score or noul")

    state = params.get("state") or ""
    if not isinstance(state, str):
        raise RPCError(INVALID_PARAMS, "params.state must be a string")

    decider = _decider()
    if not decider.available:
        raise RPCError(INTERNAL_ERROR, "Laya ONNX checkpoint not found", {"dir": decider.directory})

    try:
        return decider.decide(instructions, criteria, state=state, qtype=qtype)
    except ValueError as exc:
        raise RPCError(INVALID_PARAMS, str(exc)) from exc


def method_render(params: Any) -> Any:
    """Render one page of a source document to PNG.

    The desktop UI draws a chunk's bbox over the page that chunk came from, and
    that bbox is measured in PDF points (chunking.Block.to_chunk). Rendering the
    page HERE is what keeps that to a single coordinate system: PyMuPDF is
    already a dependency, the page's size in points comes back alongside the
    image, and the Electron side never has to open a PDF at all. The renderer
    scales by ``width_pt`` / ``height_pt``.

    ``page`` is 1-based, matching the ``page_num`` every chunk carries. An
    off-by-one here would draw every highlight on the wrong page, which reads as
    a data problem rather than as a UI one.
    """
    if not isinstance(params, dict):
        raise RPCError(INVALID_PARAMS, "params must be an object")

    path = params.get("path")
    if not path or not isinstance(path, str):
        raise RPCError(INVALID_PARAMS, "params.path is required")

    page_number = params.get("page")
    if not isinstance(page_number, int) or page_number < 1:
        raise RPCError(INVALID_PARAMS, "params.page must be a positive integer (1-based)")

    # 110 dpi is ~1.5x a typical screen rendering of a letter page: enough that
    # body text stays legible without shipping a megabyte per page over a pipe.
    dpi = params.get("dpi") or 110
    if not isinstance(dpi, int) or not 36 <= dpi <= 400:
        raise RPCError(INVALID_PARAMS, "params.dpi must be an integer in [36, 400]")

    # Optional crop, in the same PDF points the chunk bboxes use. The UI does not
    # pass it (it draws the bbox over a whole page); the figure captioner does,
    # and a crop is what keeps a vision model from being fed a whole page to look
    # at one chart.
    raw_clip = params.get("bbox")
    clip_values = None
    if raw_clip is not None:
        if (not isinstance(raw_clip, (list, tuple)) or len(raw_clip) != 4
                or not all(isinstance(v, (int, float)) for v in raw_clip)):
            raise RPCError(INVALID_PARAMS, "params.bbox must be [x0, y0, x1, y1] in points")
        clip_values = [float(v) for v in raw_clip]
        if clip_values[2] <= clip_values[0] or clip_values[3] <= clip_values[1]:
            raise RPCError(INVALID_PARAMS, "params.bbox is empty")

    # Longest-side cap in pixels. Image tokens cost prefill on the vision model
    # (measured: 82 tok/s, ~1/3 of its text prefill), so a page-sized crop is
    # mostly wasted work; the caller asks for what it needs.
    max_side = params.get("max_side") or 0
    if not isinstance(max_side, int) or max_side < 0:
        raise RPCError(INVALID_PARAMS, "params.max_side must be a non-negative integer")

    try:
        import pymupdf
    except ImportError as exc:
        raise RPCError(INTERNAL_ERROR, f"missing dependency: {exc}", {"hint": "pip install pymupdf"}) from exc

    try:
        document = pymupdf.open(path)
    except FileNotFoundError as exc:
        raise RPCError(NOT_FOUND, f"source file is gone: {path}", {"path": path}) from exc
    except Exception as exc:  # noqa: BLE001 - a damaged file must be reported, not crash
        raise RPCError(NOT_FOUND, f"cannot open {path}: {exc}", {"path": path}) from exc

    try:
        if page_number > document.page_count:
            raise RPCError(
                NOT_FOUND,
                f"page {page_number} is outside the document ({document.page_count} page(s))",
                {"path": path, "pages": document.page_count},
            )
        page = document.load_page(page_number - 1)
        rect = page.rect
        clip = None
        if clip_values is not None:
            clip = pymupdf.Rect(*clip_values) & rect
            if clip.is_empty:
                raise RPCError(
                    INVALID_PARAMS,
                    "params.bbox does not overlap the page",
                    {"path": path, "page": page_number},
                )

        # The cap is applied by LOWERING THE DPI, not by resizing afterwards:
        # rendering at 150 dpi and shrinking throws away the pixels it just paid
        # to produce, and the stored image is what the model sees either way.
        target = clip or rect
        effective_dpi = dpi
        if max_side:
            longest_pt = max(target.width, target.height)
            if longest_pt > 0:
                effective_dpi = max(24, min(dpi, int(max_side * 72.0 / longest_pt)))

        pixmap = page.get_pixmap(dpi=effective_dpi, alpha=False, clip=clip)
        return {
            "page": page_number,
            "pages": document.page_count,
            "dpi": effective_dpi,
            # Points are what bbox is measured in, so these two are what the
            # renderer divides by. The pixel sizes are informational.
            "width_pt": float(rect.width),
            "height_pt": float(rect.height),
            "width_px": int(pixmap.width),
            "height_px": int(pixmap.height),
            # PNG as base64: this channel is line-delimited JSON, so the bytes
            # have to survive as text.
            "image": base64.b64encode(pixmap.tobytes("png")).decode("ascii"),
        }
    finally:
        document.close()


# Method table: name -> handler. Add entries as phases land.
METHODS = {
    "ping": method_ping,
    "version": method_version,
    "warmup": method_warmup,
    "parse": method_parse,
    "decide": method_decide,
    "render": method_render,
}


def handle(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dispatch one request. Returns None for a notification."""
    req_id = request.get("id")
    is_notification = req_id is None

    version = request.get("jsonrpc")
    if version not in (None, PROTOCOL_VERSION):
        return _error(req_id, INVALID_REQUEST, "unsupported jsonrpc version: %s" % version)

    method = request.get("method")
    if not method or not isinstance(method, str):
        return _error(req_id, INVALID_REQUEST, "method is required")

    handler = METHODS.get(method)
    if handler is None:
        return _error(req_id, METHOD_NOT_FOUND, "method not found: %s" % method)

    try:
        result = handler(request.get("params"))
    except RPCError as exc:
        return None if is_notification else _error(req_id, exc.code, exc.message, exc.data)
    except Exception as exc:  # noqa: BLE001 - any handler failure becomes a JSON-RPC error
        return None if is_notification else _error(req_id, INTERNAL_ERROR, str(exc))

    if is_notification:
        return None
    return {"jsonrpc": PROTOCOL_VERSION, "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    resp: Dict[str, Any] = {"jsonrpc": PROTOCOL_VERSION, "error": err}
    if req_id is not None:
        resp["id"] = req_id
    return resp


def _thread_setting(name: str, fallback: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or fallback))
    except (TypeError, ValueError):
        return fallback


#: How many requests the sidecar works on at once, per lane.
#:
#: Two lanes, because the two kinds of work are limited by different things. A
#: `decide` is CPU-bound inside one ONNX session — measured 2.0s for a 236-token
#: state and 14.6s at about 2k — and two at once really do use two cores, so
#: parallelism buys throughput. A `parse` holds hundreds of MB of activations
#: and is bandwidth-bound (measured: four concurrent parses took four times the
#: wall time of one, for 836 MB each), so it stays serial. This is the explicit
#: inference budget RAGFlow keeps for its deepdoc stage too
#: (`deepdoc.inference_concurrency`, default 4), rather than "as many as the
#: pool happens to allow".
DECIDE_THREADS = _thread_setting("FREERAG_DECIDE_THREADS", 2)
PARSE_THREADS = _thread_setting("FREERAG_PARSE_THREADS", 1)

#: The methods that run on the serial lane.
_PARSE_LANE = frozenset({"parse", "render"})


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    """Read NDJSON JSON-RPC requests and answer them, up to N at a time.

    Requests are read on this thread and executed on a pool, so a long parse no
    longer blocks a decision behind it — which is the point: the kernel
    multiplexes by request id and does not need answers in order.

    Responses may therefore come back out of order. That is a property of the
    protocol rather than a coincidence — the kernel matches them by id
    (internal/sidecar/client.go's readLoop) — but the WRITE is still locked: two
    workers interleaving halves of a line would corrupt the stream, which no
    reader can recover from.
    """
    write_lock = threading.Lock()

    def emit(payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with write_lock:
            stdout.write(line)
            stdout.flush()

    def run(request: Dict[str, Any]) -> None:
        try:
            response = handle(request)
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the reader
            response = _error(request.get("id"), INTERNAL_ERROR, "handler crashed: %s" % exc)
        if response is not None:
            emit(response)

    parse_lane = ThreadPoolExecutor(max_workers=PARSE_THREADS, thread_name_prefix="freerag-parse")
    fast_lane = ThreadPoolExecutor(max_workers=DECIDE_THREADS, thread_name_prefix="freerag-decide")
    try:
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                emit(_error(None, PARSE_ERROR, "parse error: %s" % exc))
                continue
            if not isinstance(request, dict):
                emit(_error(None, INVALID_REQUEST, "request must be an object"))
                continue
            lane = parse_lane if request.get("method") in _PARSE_LANE else fast_lane
            lane.submit(run, request)
    finally:
        # Waiting here is the difference between a clean exit and silently
        # dropping the answers that were still in flight when stdin closed.
        parse_lane.shutdown(wait=True)
        fast_lane.shutdown(wait=True)


if __name__ == "__main__":
    serve()
