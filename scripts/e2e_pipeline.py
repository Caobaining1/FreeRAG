#!/usr/bin/env python3
"""End-to-end pipeline test: PDF -> parse -> index -> retrieve -> ask.

Drives the real Go kernel over stdio JSON-RPC, one process per request (the same
way the desktop shell does), against a real document. Reports timing and outcome
at every stage, and writes the raw responses for inspection.

Usage:
    python3 scripts/e2e_pipeline.py <pdf> [--question TEXT]... [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_QUESTIONS = [
    "What is RL-based agentic search?",
    "Along which three dimensions does the survey organize the field?",
    "How does GRPO differ from PPO?",
]

TOOL_CALLS = [
    ("hybrid_search", {"query": "agentic search reinforcement learning", "k": 5}),
    ("grep_search", {"pattern": "GRPO"}),
    ("grep_search", {"pattern": r"DAPO|DPO", "regex": True}),
    ("list_chunks", {"doc_id": "", "limit": 3}),
    ("metadata_search", {"block_type": "Table", "limit": 5}),
    ("metadata_search", {"block_type": "Figure", "limit": 5}),
]


def rpc(kernel: str, index_path: str, method: str, params: Any = None, timeout: int = 3600) -> Tuple[Dict[str, Any], float]:
    """Send one JSON-RPC request to a fresh kernel process.

    ``FREERAG_DATA`` is the index **file** path, not a directory — passing a
    directory makes ``Save`` fail silently behind a log warning.
    """
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params

    env = dict(os.environ, FREERAG_DATA=index_path)
    started = time.time()
    completed = subprocess.run(
        [kernel],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    elapsed = time.time() - started

    # Kernel warnings are the only signal for a failed persist / degraded start.
    for line in completed.stderr.splitlines():
        if "warning:" in line or "error" in line.lower():
            print("      kernel: %s" % line.strip())

    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line), elapsed
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "no JSON-RPC response for %s\nstdout=%s\nstderr=%s"
        % (method, completed.stdout[-800:], completed.stderr[-800:])
    )


def report(step: str, elapsed: float, detail: str) -> None:
    print("  %-26s %7.2fs   %s" % (step, elapsed, detail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--no-ask", action="store_true", help="skip the agentic Q&A stage")
    parser.add_argument("--kernel", default="bin/freerag")
    parser.add_argument("--out", default="artifacts/e2e")
    parser.add_argument("--keep", action="store_true", help="keep the index after the run")
    args = parser.parse_args()

    questions = args.question or DEFAULT_QUESTIONS
    pdf = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf):
        print("no such file: %s" % pdf, file=sys.stderr)
        return 2

    kernel = os.path.abspath(args.kernel)
    if not os.path.isfile(kernel):
        print("kernel not built: %s (run: go build -o bin/freerag ./cmd/freerag)" % kernel, file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    data_dir = tempfile.mkdtemp(prefix="freerag-e2e-")
    index_path = os.path.join(data_dir, "index.json")
    failures: List[str] = []

    print("\n=== 0. version ===")
    version, elapsed = rpc(kernel, index_path, "version")
    version_result = version.get("result") or {}
    embedding = version_result.get("embedding") or {}
    report("version", elapsed, "sidecar=%s checker=%s embedding=%s"
           % ("yes" if version_result.get("parse_sidecar") else "NO",
              version_result.get("checker"),
              "%s (%s dims)" % (embedding.get("provider"), embedding.get("dims"))
              if embedding.get("enabled") else "OFF"))
    if not version_result.get("parse_sidecar"):
        failures.append("the parse sidecar is unavailable")
    if not embedding.get("enabled"):
        print("      note: dense retrieval is off; hybrid_search will be keyword-only")

    print("\n=== 1. parse: %s ===" % os.path.basename(pdf))
    parsed, parse_seconds = rpc(kernel, index_path, "parse", {"path": pdf})
    if "error" in parsed:
        print("  parse FAILED: %s" % parsed["error"], file=sys.stderr)
        return 1
    parse_result = parsed["result"]
    types: Dict[str, int] = {}
    chars = 0
    for chunk in parse_result.get("chunks", []):
        meta = chunk.get("metadata") or {}
        types[meta.get("block_type", "?")] = types.get(meta.get("block_type", "?"), 0) + 1
        chars += len(chunk.get("text") or "")
    report("parse", parse_seconds,
           "%d pages, %d blocks, %d chunks, %d chars, provider=%s"
           % (parse_result.get("pages_parsed"), parse_result.get("block_count"),
              parse_result.get("chunk_count"), chars, parse_result.get("layout_provider")))
    print("      block types: %s" % dict(sorted(types.items())))
    if parse_result.get("chunk_count", 0) == 0:
        failures.append("parse produced no chunks")

    print("\n=== 2. index ===")
    indexed, index_seconds = rpc(kernel, index_path, "index", {"path": pdf})
    index_result = indexed.get("result") or {}
    report("index (parse + embed + store)", index_seconds,
           "added=%s, indexed_total=%s, embedded=%s, chunk_count=%s"
           % (index_result.get("added"), index_result.get("indexed_total"),
              index_result.get("embedded"), index_result.get("chunk_count")))
    if not index_result.get("added"):
        failures.append("index added nothing")
    if index_result.get("save_error"):
        failures.append("the index was not persisted: %s" % index_result["save_error"])
    if embedding.get("enabled") and not index_result.get("embedded"):
        failures.append("dense retrieval is on but no chunk carries a vector")

    print("\n=== 3. tool surface ===")
    surface, surface_seconds = rpc(kernel, index_path, "tools")
    names = (surface.get("result") or {}).get("names") or []
    report("tools", surface_seconds, "mode=%s tools=%s" % ((surface.get("result") or {}).get("mode"), names))

    print("\n=== 4. retrieval tools ===")
    for method, arguments in TOOL_CALLS:
        label = "%s(%s)" % (method, json.dumps(arguments, ensure_ascii=False))
        response, seconds = rpc(kernel, index_path, "tool", {"name": method, "arguments": arguments})
        raw = response.get("result") or {}
        hits = raw.get("hits") or []
        report(label[:26], seconds, "%d hit(s) — %s" % (len(hits), raw.get("note", "")))
        if method == "hybrid_search" and not hits:
            failures.append("hybrid_search returned no hits")

    print("\n=== 5. agentic Q&A (%d question(s)) ===" % (0 if args.no_ask else len(questions)))
    for question in ([] if args.no_ask else questions):
        response, seconds = rpc(kernel, index_path, "ask", {"question": question})
        if "error" in response:
            report(question[:26], seconds, "ERROR %s" % response["error"])
            failures.append("ask failed: %s" % question)
            continue
        answer = response["result"]
        draft = (answer.get("draft") or "").strip()
        report(question[:26], seconds,
               "rounds=%s verdict=%s evidence=%d draft=%d chars"
               % (answer.get("rounds"), answer.get("verdict"),
                  len(answer.get("evidence") or []), len(draft)))
        print("      Q: %s" % question)
        print("      A: %s" % draft.replace("\n", " ")[:300])
        print("      trace: %s" % " | ".join(answer.get("trace") or [])[:300])
        if not draft:
            failures.append("ask produced no draft: %s" % question)

    print("\n=== 6. persistence ===")
    if os.path.isfile(index_path):
        report("index.json", 0.0, "%.1f KB" % (os.path.getsize(index_path) / 1024.0))
    else:
        failures.append("index.json was not written")

    summary = {"parse": parse_result, "index": index_result, "questions": questions}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.keep:
        shutil.copytree(data_dir, os.path.join(args.out, "data"), dirs_exist_ok=True)
        print("\n  kept index at %s" % os.path.join(args.out, "data"))
    shutil.rmtree(data_dir, ignore_errors=True)

    print("\n=== summary ===")
    if failures:
        print("  FAILED (%d):" % len(failures))
        for failure in failures:
            print("   - %s" % failure)
        return 1
    print("  E2E PIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
