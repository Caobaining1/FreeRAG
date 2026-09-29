#!/usr/bin/env python3
"""Validate a `parse` JSON-RPC response captured by the acceptance run.

Usage: assert_parse.py <response.json> [expected_pages]

Exits non-zero and prints every failed expectation.
"""
from __future__ import annotations

import json
import re
import sys

MIN_CHUNK_CHARS = 10
TABLE_MARKER = re.compile(r"^\|\s*---", re.MULTILINE)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: assert_parse.py <response.json> [expected_pages]", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)

    failures: list[str] = []

    check("error" not in payload, f"response carries an error: {payload.get('error')}", failures)
    result = payload.get("result") or {}

    check(result.get("page_count", 0) >= 1, "page_count should be >= 1", failures)
    if len(sys.argv) > 2:
        expected = int(sys.argv[2])
        check(
            result.get("page_count") == expected,
            f"page_count = {result.get('page_count')}, want {expected}",
            failures,
        )
        check(
            result.get("pages_parsed") == expected,
            f"pages_parsed = {result.get('pages_parsed')}, want {expected}",
            failures,
        )

    chunks = result.get("chunks") or []
    check(len(chunks) > 0, "no chunks were produced", failures)
    check(
        result.get("chunk_count") == len(chunks),
        f"chunk_count ({result.get('chunk_count')}) != len(chunks) ({len(chunks)})",
        failures,
    )

    ids = [c.get("chunk_id") for c in chunks]
    check(len(ids) == len(set(ids)), "chunk ids are not unique", failures)

    for chunk in chunks:
        cid = chunk.get("chunk_id")
        text = (chunk.get("text") or "").strip()
        meta = chunk.get("metadata") or {}
        check(bool(text), f"{cid}: empty text", failures)
        check(len(text) >= MIN_CHUNK_CHARS, f"{cid}: text shorter than {MIN_CHUNK_CHARS} chars", failures)
        for field in ("page_num", "block_type", "bbox", "source_file"):
            check(field in meta, f"{cid}: metadata missing {field}", failures)

    types = {c.get("metadata", {}).get("block_type") for c in chunks}
    check("Title" in types, f"no Title block detected (types={sorted(t for t in types if t)})", failures)
    check("Table" in types, f"no Table block detected (types={sorted(t for t in types if t)})", failures)

    tables = [c for c in chunks if c.get("metadata", {}).get("block_type") == "Table"]
    for table in tables:
        check(
            bool(TABLE_MARKER.search(table.get("text", ""))),
            f"{table.get('chunk_id')}: table is not Markdown",
            failures,
        )

    pages = sorted({c.get("metadata", {}).get("page_num") for c in chunks})
    check(pages == list(range(1, len(pages) + 1)), f"page numbers are not contiguous: {pages}", failures)

    titles = [c for c in chunks if c.get("metadata", {}).get("block_type") == "Title"]
    if titles:
        title_text = titles[0]["text"].strip()
        sectioned = [c for c in chunks if c.get("metadata", {}).get("parent_section") == title_text]
        check(bool(sectioned), "no chunk carries the title as parent_section", failures)

    if failures:
        print("parse response FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "parse response OK: "
        f"{result['page_count']} page(s), {result['block_count']} block(s), "
        f"{result['chunk_count']} chunk(s), types={sorted(t for t in types if t)}, pages={pages}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
