"""Shared helper for driving the Go kernel over stdio JSON-RPC from scripts.

One process per request, exactly as the desktop shell does. Kept in one place
because the ``FREERAG_DATA`` subtlety below is easy to get wrong twice.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

__all__ = ["rpc", "KernelError"]


class KernelError(RuntimeError):
    """The kernel returned a JSON-RPC error."""


def rpc(
    kernel: str,
    index_path: str,
    method: str,
    params: Any = None,
    *,
    timeout: int = 3600,
    quiet: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """Send one JSON-RPC request to a fresh kernel process.

    ``FREERAG_DATA`` is the index **file** path, not a directory: passing a
    directory makes ``Save`` fail behind a log warning while the reply still
    claims success.

    Returns ``(result, seconds)`` and raises ``KernelError`` on a JSON-RPC error.
    """
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params

    started = time.time()
    completed = subprocess.run(
        [kernel],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=dict(os.environ, FREERAG_DATA=index_path),
        timeout=timeout,
    )
    elapsed = time.time() - started

    if not quiet:
        for line in completed.stderr.splitlines():
            if "warning:" in line or "error" in line.lower():
                print("      kernel: %s" % line.strip())

    response: Optional[Dict[str, Any]] = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            response = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if response is None:
        raise RuntimeError(
            "no JSON-RPC response for %s\nstdout=%s\nstderr=%s"
            % (method, completed.stdout[-800:], completed.stderr[-800:])
        )
    if "error" in response:
        raise KernelError("%s: %s" % (method, response["error"]))

    return response.get("result") or {}, elapsed
