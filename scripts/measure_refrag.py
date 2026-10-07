#!/usr/bin/env python3
"""End-to-end A/B of the REFRAG-style evidence compression, on the real kernel.

What it measures, and why each number is here:

  evidence_chars  the rendered evidence block, before and after compression.
                  Deterministic — the same pool renders the same block — so it
                  is the one figure that needs no repetition to be trusted.
  gen_seconds     the answer-writing model call alone, parsed out of the trace
                  ("[Answer] written in 41.4s" on the simple path, "[Synthesize]
                  written in ..." on the complex path). This is where a prompt
                  change lands: the loop's retrieval and planning are untouched
                  by it, and they dominate the wall clock.
  ttft_seconds    first streamed fragment of the answer. Deliberately NOT the
                  headline number end to end: the answer is written once, after
                  the loop has settled, so the user's first character arrives at
                  the end of everything. The controlled measurement of the
                  effect lives in internal/agent/refrag_live_test.go, where the
                  retrieval is held fixed.
  fragments       how many `answer` notifications carried the text. Only
                  interesting with --coalesce-ms, whose whole claim is that
                  number.

Discipline borrowed from docs/performance.md, because each item was bought with
a wasted run there:

  - one arm at a time, alternating the order every repetition;
  - the generator is probed before the run starts (§13.27: a degraded run that
    "succeeds" is worse than one that fails);
  - answers at exactly the extractive cap mark the scorecard unusable, for the
    same reason;
  - the retrieval mode (hybrid vs keyword-only) is recorded per run, because a
    silent degrade to keyword-only changes the pools and therefore the prompt.

Usage:

    python3 scripts/measure_refrag.py --reps 2 --questions 3
    python3 scripts/measure_refrag.py --arms full,refrag,refrag+stream
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragas_eval import BENCH, Kernel, REPO  # noqa: E402  (needs the path above)

EXTRACTIVE_CAP = 6000  # agent.DefaultExtractiveAnswerChars

# The answer-writing call, timed by the loop itself. Both paths are matched:
# the simple one writes through Loop.writeAnswer, the complex one through the
# flow's synthesis.
RE_GEN = re.compile(r"\[(?:Answer|Synthesize)\] written in ([^()]+)")
GEN_TRACE_RE = re.compile(r"\[(?:Answer|Synthesize|Usage|Refrag|Stream)\]")
RE_ROUNDS = re.compile(r"\[SCA\] Round (\d+) verdict=(\w+)")
RE_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(h|ms|µs|ns|m|s)")
DURATION_SCALE = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}

ARMS: Dict[str, Dict[str, str]] = {
    "full": {},
    "refrag": {"FREERAG_REFRAG": "1"},
    "refrag+stream": {"FREERAG_REFRAG": "1", "FREERAG_STREAM_COALESCE_MS": "120"},
    "stream": {"FREERAG_STREAM_COALESCE_MS": "120"},
}


def parse_seconds(text: str) -> Optional[float]:
    """Seconds out of a Go duration, as the loop's own trace prints it.

    Round(100ms) formats as "41.4s", "800ms" — and "1m40.7s" once a call passes
    a minute. A parser that only knows plain seconds silently returns None for
    exactly the longest calls, which is the worst possible place to lose a
    measurement: a long answer is when a prompt-side change matters most.
    """
    match = RE_GEN.search(text)
    if not match:
        return None
    total = 0.0
    parts = RE_DURATION_PART.findall(match.group(1))
    if not parts:
        return None
    for value, unit in parts:
        total += float(value) * DURATION_SCALE[unit]
    return total


def generator_alive() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def load_questions(count: int) -> List[Dict[str, Any]]:
    """Answerable dev questions, in a fixed order.

    Null questions are excluded: their answer is a refusal, so their generation
    time says nothing about the prompt and they would dilute the mean.
    """
    rows = json.loads((REPO / "eval" / "dev.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("type") != "null_query" and r.get("reference_contexts")]
    return rows[:count]


def ask_once(kb_dir: Path, kb_id: str, question: str, env: Dict[str, str]) -> Dict[str, Any]:
    """Run one question in a kernel configured with `env`, timing the stream."""
    # Cleared first, then set: os.environ is process-wide and the previous arm
    # left its variables in it, so an arm that only ADDS would inherit them.
    for key in {key for arm_env in ARMS.values() for key in arm_env}:
        os.environ.pop(key, None)
    os.environ.update(env)

    kernel = Kernel(kb_dir)
    try:
        started = time.time()
        events: List[Dict[str, Any]] = []
        ttft = None
        fragments = 0
        first_chunk_at = None

        def on_progress(params: Dict[str, Any]) -> None:
            nonlocal ttft, fragments, first_chunk_at
            events.append(params)
            if params.get("delta"):
                fragments += 1
                if ttft is None:
                    ttft = time.time() - started
            if params.get("line") and first_chunk_at is None:
                first_chunk_at = time.time() - started

        result = kernel.call("ask", {"question": question, "kb": kb_id}, on_progress=on_progress)
        total = time.time() - started
    finally:
        kernel.close()

    trace = result.get("trace") or []
    # The flow's own steps — including the synthesis timing — are NOT in
    # Result.Trace: on the complex path they go to the notification channel
    # only (FlowDeps.OnStep). Parsing the result's trace alone silently finds no
    # generation time on exactly the path most questions here take, so both
    # sources are joined.
    lines = [str(event.get("line", "")) for event in events if event.get("line")]
    trace = [str(item) for item in trace]
    # Both sources, because the flow's step lines reach the notification channel
    # and the result's own trace separately — keeping only the first drops the
    # [Refrag] / [Usage] / [Stream] lines, which is exactly the instrumentation
    # this script exists to read.
    all_lines = lines + trace
    trace_text = "\n".join(all_lines)
    rounds = [int(m.group(1)) for m in RE_ROUNDS.finditer(trace_text)]
    notes = " ".join(str(e.get("note", "")) for e in events)

    return {
        "question": question,
        "env": env,
        "ttft_s": ttft,
        "total_s": total,
        "fragments": fragments,
        "answer_chars": len(result.get("answer") or ""),
        "gen_s": parse_seconds(trace_text),
        "rounds": max(rounds) if rounds else None,
        "verdict": result.get("verdict"),
        "evidence": len(result.get("evidence") or []),
        "prompt_stats": result.get("prompt_stats"),
        "keyword_only": "keyword only" in trace_text or "keyword only" in notes,
        "route": result.get("route"),
        # Only the lines the report reads, so a 12-run file stays inspectable
        # without carrying every tool call in it.
        "trace": [line for line in all_lines if GEN_TRACE_RE.search(line) or RE_ROUNDS.search(line)],
    }


def median(values: List[Optional[float]], fmt: str = "%.1f") -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    return fmt % statistics.median(present)


def usage_of(row: Dict[str, Any]) -> Dict[str, Any]:
    return ((row.get("prompt_stats") or {}).get("usage") or {})


def summarize(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n%-14s %3s %9s %8s %9s %7s %8s %7s %7s" % (
        "arm", "n", "ev_chars", "raw", "prompt_tok", "pre_s", "out_tok", "gen_s", "total_s"))
    print("-" * 92)
    for arm in sorted({row["arm"] for row in rows}):
        subset = [row for row in rows if row["arm"] == arm]
        print("%-14s %3d %9s %8s %9s %7s %8s %7s %7s" % (
            arm,
            len(subset),
            median([(r.get("prompt_stats") or {}).get("evidence_chars") for r in subset], "%d"),
            median([(r.get("prompt_stats") or {}).get("evidence_chars_raw") for r in subset], "%d"),
            median([usage_of(r).get("prompt_tokens") for r in subset], "%d"),
            # The generator's OWN prefill and generation timings. This is the
            # split the wall clock cannot give: a prompt-side change touches
            # prefill and a length change touches generation, and "elapsed"
            # mixes them into one number that moves for both reasons.
            median([usage_of(r).get("prompt_nanos", 0) / 1e9 or None for r in subset]),
            median([usage_of(r).get("output_tokens") for r in subset], "%d"),
            median([usage_of(r).get("output_nanos", 0) / 1e9 or None for r in subset]),
            median([r["total_s"] for r in subset]),
        ))

    # Is the run even comparable? The loop is not guaranteed to retrieve the
    # same pool twice: rewriting, plan-then-Laya tool choice and the fan-out are
    # each a decision, and the merge pool is their union. If an arm's own raw
    # evidence spans a wide range across repetitions, then an arm-versus-arm
    # difference in any latency figure is a difference between two pools rather
    # than between two prompts.
    print("\nsame-arm spread in raw evidence chars (the reproducibility control):")
    for arm in sorted({row["arm"] for row in rows}):
        per_question = {}
        for row in rows:
            if row["arm"] != arm:
                continue
            raw = (row.get("prompt_stats") or {}).get("evidence_chars_raw")
            if raw:
                per_question.setdefault(row["question_id"], []).append(raw)
        spreads = ["%s:%d-%d" % (qid, min(v), max(v))
                   for qid, v in sorted(per_question.items()) if len(v) > 1]
        print("  %-14s %s" % (arm, ", ".join(spreads) or "no repetition"))

    # The headline: the answer-writing call, which is where a prompt change
    # lands. Paired by question, because the questions differ in length and an
    # unpaired mean would compare the question mix rather than the change.
    for metric, label in (("prompt_nanos", "prefill"), ("output_nanos", "generation")):
        print("\n%s time:" % label)
        for arm in sorted({row["arm"] for row in rows} - {"full"}):
            pairs = []
            for row in rows:
                if row["arm"] != arm:
                    continue
                twin = next((r for r in rows
                             if r["arm"] == "full" and r["rep"] == row["rep"]
                             and r["question_id"] == row["question_id"]), None)
                if twin and usage_of(row).get(metric) and usage_of(twin).get(metric):
                    pairs.append((usage_of(twin)[metric] / 1e9, usage_of(row)[metric] / 1e9))
            if not pairs:
                print("  %-14s no paired sample" % arm)
                continue
            before = statistics.median(p[0] for p in pairs)
            after = statistics.median(p[1] for p in pairs)
            print("  %-14s %.1fs -> %.1fs (%+.1fs, %.2fx) over %d pair(s)"
                  % (arm, before, after, after - before,
                     before / after if after else float("inf"), len(pairs)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", default="eval/data-hybrid",
                        help="directory holding the eval knowledge base")
    parser.add_argument("--kb-id", default="", help="knowledge base id; default: kb_id.txt")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--arms", default="full,refrag")
    parser.add_argument("--out", default="eval/runs/refrag-streaming.jsonl")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="run even when the generator is unreachable")
    args = parser.parse_args()

    if not generator_alive() and not args.allow_degraded:
        print("the generator is not answering on 127.0.0.1:11434 — refusing to measure.\n"
              "A run without a generator returns the extractive draft, and its numbers\n"
              "look completely normal (docs/plan.md §13.27).",
              file=sys.stderr)
        return 2

    kb_dir = (REPO / args.kb).resolve()
    kb_id = args.kb_id or (kb_dir / "kb_id.txt").read_text(encoding="utf-8").strip()
    questions = load_questions(args.questions)
    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    for name in arm_names:
        if name not in ARMS:
            print("unknown arm %r; known: %s" % (name, ", ".join(sorted(ARMS))), file=sys.stderr)
            return 2

    print("kb=%s  questions=%d  reps=%d  arms=%s" % (kb_id, len(questions), args.reps, arm_names))

    rows: List[Dict[str, Any]] = []
    out = (REPO / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for rep in range(1, args.reps + 1):
            # Interleaved PER QUESTION, and reversed every repetition.
            #
            # Not per-arm blocks: the machine drifts (thermal, background load),
            # so three consecutive runs of one arm and then three of the other
            # attributes the drift to the arm. And not always in the same order,
            # or the arm that ran second inherits it. A first attempt used blocks
            # and its numbers were unreadable — see docs/performance.md §10.3.
            order = arm_names if rep % 2 else list(reversed(arm_names))
            for question in questions:
                for arm in order:
                    row = ask_once(kb_dir, kb_id, question["question"], ARMS[arm])
                    row["arm"] = arm
                    row["rep"] = rep
                    row["question_id"] = question["id"]
                    rows.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    print("  rep %d %-14s %-10s gen=%s total=%5.1fs ev=%s->%s frags=%d" % (
                        rep, arm, question["id"],
                        "%.1fs" % row["gen_s"] if row["gen_s"] else "-",
                        row["total_s"],
                        (row.get("prompt_stats") or {}).get("evidence_chars_raw", "-"),
                        (row.get("prompt_stats") or {}).get("evidence_chars", "-"),
                        row["fragments"],
                    ))

    summarize(rows)

    extractive = [r for r in rows if r["answer_chars"] == EXTRACTIVE_CAP
                  and not r["keyword_only"]]
    if extractive:
        print("\nWARNING: %d of %d answer(s) are exactly %d characters — the extractive "
              "draft the kernel returns without a generator (§13.27). This run's numbers "
              "are not about the prompt." % (len(extractive), len(rows), EXTRACTIVE_CAP))
    if any(r["keyword_only"] for r in rows):
        print("WARNING: at least one run was keyword-only (the query-side embedding "
              "failed); its pool came from a different retriever than the others.")
    print("\nwrote %s" % out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
