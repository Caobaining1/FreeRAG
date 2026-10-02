#!/usr/bin/env python3
"""Decide whether an unanswerable question was refused, by asking the judge. docs/plan.md §13.23.

The refusal guard is the one guard that decides against score (§13.2), and it was measured by a
REGEX over the answer text:

    refusals = sum(1 for r in null_rows if REFUSAL.search(r.get("answer", "")))

That measures wording, not behaviour. Checked against the 8 null questions of two real runs, it
got one outright wrong: an answer reading "The evidence provided does not explicitly mention
David Reeder's positions at specific companies" is a refusal, and the pattern does not match
"does not explicitly mention". With 8 samples and a lexical detector, the guard cannot tell a
refusal regression from a paraphrase — and it rejected a change worth +0.089 macro and +0.333
context_recall on that basis.

So ask the judge instead, which is what the metric is actually about: did the answer claim to
answer, or did it say the corpus does not cover this? Both counts are reported so the two
detectors can be compared rather than swapped blindly.

Usage:
    python3 scripts/refusal_check.py eval/runs/baseline.jsonl eval/runs/validate-grepfix.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ragas_eval import REFUSAL, load_env  # noqa: E402

INSTRUCTION = (
    "A user asked a question that the document collection CANNOT answer. An assistant replied. "
    "Decide which happened:\n"
    "- REFUSED: the reply says it does not have the information, or that the documents do not "
    "cover it, in any wording.\n"
    "- ANSWERED: the reply states a value, name, date or conclusion as if it were answering.\n"
    'Reply with JSON only: {"verdict": "REFUSED" or "ANSWERED", "why": "short reason"}'
)


def judge(answer: str) -> Optional[str]:
    base = os.environ.get("RAGAS_JUDGE_BASE", "").rstrip("/")
    body = {
        "model": os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4.1-flash"),
        "temperature": 0, "max_tokens": 2048, "reasoning_effort": "none",
        "messages": [{"role": "system", "content": INSTRUCTION},
                     {"role": "user", "content": f"REPLY:\n\n{answer[:4000]}"}],
    }
    request = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + (os.environ.get("JUDGE_API_KEY") or ""),
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"    judge failed: {exc}")
        return None
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    match = re.search(r'"verdict"\s*:\s*"(REFUSED|ANSWERED)"', text, re.I)
    return match.group(1).upper() if match else None


def check(path: Path) -> Dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nulls = [r for r in rows if r.get("type") == "null_query"]
    print(f"\n{path.name}: {len(nulls)} null question(s)")
    lexical = judge_based = 0
    for row in nulls:
        answer = row.get("answer") or ""
        by_regex = bool(REFUSAL.search(answer))
        verdict = judge(answer)
        lexical += by_regex
        judge_based += verdict == "REFUSED"
        agree = "  " if (verdict == "REFUSED") == by_regex else "<- disagree"
        print(f"  {row['id']:<8} regex={'拒' if by_regex else '答'} judge={verdict or '?':<9} {agree} "
              f"{answer[:70].replace(chr(10), ' ')}")
    return {"n": len(nulls), "lexical": lexical,
            "judge": judge_based,
            "rate_lexical": round(lexical / len(nulls), 4) if nulls else None,
            "rate_judge": round(judge_based / len(nulls), 4) if nulls else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    load_env(REPO / ".env")

    results = []
    for name in args.runs:
        results.append((name, check(REPO / name)))
    print(f"\n{'run':<34} {'regex':>8} {'judge':>8}")
    for name, result in results:
        print(f"{Path(name).name:<34} {result['rate_lexical']:>8} {result['rate_judge']:>8}")
    print("\nThe judge column is the one the guard should read: same answers, and the wording "
          "of a refusal is not its behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
