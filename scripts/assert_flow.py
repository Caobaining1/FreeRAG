#!/usr/bin/env python3
"""Validate the index -> search -> ask responses captured by the acceptance run.

Usage: assert_flow.py <index.json> <search.json> <ask.json>
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def load(path: str) -> Dict[str, Any]:
    """Read the JSON-RPC response out of a captured stream.

    A capture can hold several lines: the kernel reports progress as
    notifications on the same stream as the response. The response is the
    message carrying an id — a notification deliberately has none, which is what
    tells a client it is an event rather than something it forgot to await.
    """
    last: Dict[str, Any] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            last = message
            if message.get("id") is not None:
                return message
    return last


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: assert_flow.py <index.json> <search.json> <ask.json>", file=sys.stderr)
        return 2

    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    index = load(sys.argv[1])
    check("error" not in index, f"index error: {index.get('error')}")
    index_result = index.get("result") or {}
    check(index_result.get("added", 0) > 0, f"index added = {index_result.get('added')}")
    check(index_result.get("indexed_total", 0) > 0, f"indexed_total = {index_result.get('indexed_total')}")
    check(index_result.get("chunk_count", 0) > 0, "chunk_count should be > 0")

    search = load(sys.argv[2])
    check("error" not in search, f"search error: {search.get('error')}")
    search_result = search.get("result") or {}
    hits = search_result.get("hits") or []
    check(search_result.get("count", 0) > 0, f"search count = {search_result.get('count')}")
    check(len(hits) > 0, "search returned no hits")
    for hit in hits:
        chunk = hit.get("chunk") or {}
        check(bool(chunk.get("text")), "hit without text")
        check(bool(chunk.get("doc_id")), "hit without doc_id")
        check((hit.get("score") or 0) > 0, "hit with a non-positive score")

    ask = load(sys.argv[3])
    check("error" not in ask, f"ask error: {ask.get('error')}")
    ask_result = ask.get("result") or {}
    check(ask_result.get("mode") == "medium", f"mode = {ask_result.get('mode')}")
    check(ask_result.get("rounds", 0) >= 1, "ask ran no round")
    check(
        ask_result.get("verdict") in ("SUFFICIENT", "INSUFFICIENT", "UNKNOWN"),
        f"verdict = {ask_result.get('verdict')}",
    )
    check(len(ask_result.get("evidence") or []) > 0, "ask gathered no evidence")
    check(bool((ask_result.get("draft") or "").strip()), "ask produced no draft")
    check(len(ask_result.get("trace") or []) > 0, "ask produced no trace")

    if failures:
        print("flow FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "flow OK: "
        f"indexed={index_result.get('indexed_total')}, hits={search_result.get('count')}, "
        f"mode={ask_result.get('mode')}, rounds={ask_result.get('rounds')}, "
        f"verdict={ask_result.get('verdict')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
