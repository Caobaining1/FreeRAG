#!/usr/bin/env python3
"""Validate the retrieval tool surface captured by the acceptance run.

Usage: assert_tools.py <tools.json> <hybrid.json> <grep.json>
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

EXPECTED_TOOLS = ["hybrid_search", "grep_search", "list_chunks", "metadata_search"]


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: assert_tools.py <tools.json> <hybrid.json> <grep.json>", file=sys.stderr)
        return 2

    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    tools = load(sys.argv[1])
    check("error" not in tools, f"tools error: {tools.get('error')}")
    result = tools.get("result") or {}

    names = result.get("names") or []
    check(names == EXPECTED_TOOLS, f"tool surface = {names}, want {EXPECTED_TOOLS}")
    check(result.get("mode") == "medium", f"mode = {result.get('mode')}")

    specs = result.get("tools") or []
    check(len(specs) == len(EXPECTED_TOOLS), f"{len(specs)} tool specs, want {len(EXPECTED_TOOLS)}")
    for spec in specs:
        check(bool(spec.get("description")), f"{spec.get('name')} has no description")
        check(spec.get("parameters", {}).get("type") == "object",
              f"{spec.get('name')} parameters are not an object schema")

    for path, expected_tool in ((sys.argv[2], "hybrid_search"), (sys.argv[3], "grep_search")):
        payload = load(path)
        check("error" not in payload, f"{expected_tool} error: {payload.get('error')}")
        tool_result = payload.get("result") or {}
        check(tool_result.get("tool") == expected_tool, f"tool = {tool_result.get('tool')}")
        hits = tool_result.get("hits") or []
        check(len(hits) > 0, f"{expected_tool} returned no hits")
        for hit in hits:
            check(bool((hit.get("chunk") or {}).get("text")), f"{expected_tool} hit without text")

    if failures:
        print("tool surface FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    grep_hits = (load(sys.argv[3]).get("result") or {}).get("hits") or []
    check_line = grep_hits[0].get("line") if grep_hits else ""
    print(f"tool surface OK: {names}; grep matched line: {check_line!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
