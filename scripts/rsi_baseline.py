#!/usr/bin/env python3
"""Reduce a run to the dev-loop subset's quality_macro — docs/plan.md §13.4/§13.5.

Why this exists: Phase D measures a candidate on `eval/dev-loop.json` (10 questions,
~28 min per iteration) but the shipped baseline scorecard is the FULL dev run (32
questions). Comparing the two compares sample sizes, not systems — a 10-question
macro drifts several points against a 24-question one, and the RSI loop would read
that drift as a gain or a loss and accept or revert a change on it.

So the comparison point has to be the SAME questions. This reduces any run to that
subset and writes a scorecard-shaped file, which is also how the noise band is
measured: reduce two runs of IDENTICAL code and take the difference.

Usage:
    python3 scripts/rsi_baseline.py --run eval/runs/baseline.jsonl \
        --scores eval/runs/baseline.scores.json --split eval/dev-loop.json \
        --out eval/runs/devloop-baseline.scorecard.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_run(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reduce_run(run_path: Path, scores_path: Path, split_path: Path) -> Dict:
    records = load_run(run_path)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))

    # The same filter the score stage applies, and for the same reason: null
    # questions are the refusal guard, not RAGAS samples. Getting this out of step
    # would silently pair each question with another question's scores.
    answerable = [r for r in records if r.get("type") != "null_query" and not r.get("error")]
    if len(answerable) != len(scores):
        raise SystemExit(
            f"{run_path.name} has {len(answerable)} answerable question(s) but {scores_path.name} "
            f"has {len(scores)} score row(s); re-score the run before reducing it")

    wanted = {row["id"] for row in json.loads(split_path.read_text(encoding="utf-8"))}
    pick = [(r, s) for r, s in zip(answerable, scores) if r["id"] in wanted]
    if not pick:
        raise SystemExit(f"none of {split_path.name}'s questions are in {run_path.name}")

    metrics: Dict[str, Optional[float]] = {}
    for metric in METRICS:
        values = [s.get(metric) for _, s in pick
                  if s.get(metric) is not None and s.get(metric) == s.get(metric)]
        metrics[metric] = round(statistics.mean(values), 4) if values else None

    usable = [v for v in metrics.values() if v is not None]
    null_rows = [r for r in records if r.get("type") == "null_query"]
    return {
        "run": run_path.name,
        "subset": split_path.name,
        "n_scored": len(pick),
        "n_null": len(null_rows),
        "metrics": metrics,
        "quality_macro": round(sum(usable) / len(usable), 4) if usable else None,
        "judge_nan_samples": {m: sum(1 for _, s in pick
                                     if s.get(m) is None or s.get(m) != s.get(m)) for m in METRICS},
        "refusal_rate_on_null": None,  # the refusal guard is computed from the parent scorecard
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--split", default="eval/dev-loop.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--refusal-from", default=None,
                        help="parent scorecard to carry the refusal rate from")
    args = parser.parse_args()

    reduced = reduce_run(REPO / args.run, REPO / args.scores, REPO / args.split)
    if args.refusal_from:
        parent = json.loads((REPO / args.refusal_from).read_text(encoding="utf-8"))
        reduced["refusal_rate_on_null"] = parent.get("refusal_rate_on_null")
    out = REPO / args.out
    out.write_text(json.dumps(reduced, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reduced, ensure_ascii=False, indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
