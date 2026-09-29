#!/usr/bin/env python3
"""freerag parse sidecar.

Speaks JSON-RPC 2.0 over stdio (NDJSON) with the Go kernel — the same protocol
the kernel speaks to the Electron shell (docs/plan.md §3).

Pipeline: see pipeline.py and docs/plan.md §5.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

# Allow `python3 sidecar/parse_server.py` from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import parse_pdf  # noqa: E402

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
    }
    try:
        import pymupdf  # noqa: F401

        caps["pymupdf"] = True
    except Exception:
        pass
    try:
        import onnxruntime  # noqa: F401

        caps["onnxruntime"] = True
    except Exception:
        pass

    from layout_onnx import LayoutDetector

    caps["pp_doclayout"] = LayoutDetector().available
    caps["laya"] = _decider().available
    # TSR is wired in a later phase (plan.md §5.2).
    return caps


#: The Laya session is expensive to build (~1.7 GB of weights), so it is cached
#: for the process lifetime and only built when a decision is first requested.
_DECIDERS: Dict[str, Any] = {}


def _decider() -> Any:
    from laya import LayaDecider

    if "laya" not in _DECIDERS:
        _DECIDERS["laya"] = LayaDecider()
    return _DECIDERS["laya"]


def method_ping(_params: Any) -> Any:
    return {"pong": True}


def method_version(_params: Any) -> Any:
    return {"name": "freerag-parse-sidecar", "version": VERSION, "capabilities": _capabilities()}


def method_parse(params: Any) -> Any:
    if not isinstance(params, dict):
        raise RPCError(INVALID_PARAMS, "params must be an object")

    path = params.get("path")
    if not path or not isinstance(path, str):
        raise RPCError(INVALID_PARAMS, "params.path is required")

    profile = params.get("profile") or "mixed"
    max_chars = params.get("max_chars")
    max_pages = params.get("max_pages") or 0

    if max_chars is not None and (not isinstance(max_chars, int) or max_chars <= 0):
        raise RPCError(INVALID_PARAMS, "params.max_chars must be a positive integer")
    if not isinstance(max_pages, int) or max_pages < 0:
        raise RPCError(INVALID_PARAMS, "params.max_pages must be a non-negative integer")
    if not isinstance(profile, str):
        raise RPCError(INVALID_PARAMS, "params.profile must be a string")

    try:
        return parse_pdf(path, profile=profile, max_chars=max_chars, max_pages=max_pages)
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


# Method table: name -> handler. Add entries as phases land.
METHODS = {
    "ping": method_ping,
    "version": method_version,
    "parse": method_parse,
    "decide": method_decide,
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


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    """Read NDJSON JSON-RPC requests and write responses until EOF."""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            stdout.write(json.dumps(_error(None, PARSE_ERROR, "parse error: %s" % exc)) + "\n")
            stdout.flush()
            continue
        if not isinstance(request, dict):
            stdout.write(json.dumps(_error(None, INVALID_REQUEST, "request must be an object")) + "\n")
            stdout.flush()
            continue
        response = handle(request)
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()


if __name__ == "__main__":
    serve()
