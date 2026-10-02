#!/usr/bin/env python3
"""Phase C: coordinate descent over the numeric knobs — docs/plan.md §13.4.

One iteration = change ONE knob, rebuild, re-measure `eval/dev-loop.json`,
compare against the same run's baseline, and keep or revert. Every attempt is
appended to `docs/rsi-ledger.md`, including the rejected ones: what a direction
costs is what tells Phase E which kinds of proposals are worth writing.

Why a knob table instead of "search the source for numbers": the genome has to be
DECLARED, because §13.2 also lists what must never be changed. A regex that walks
the tree would eventually edit a guard, a threshold inside the metric code, or
the embedding dimension — and the failure would look like an improvement.

What this does NOT do:
  - it does not touch prompts (that is Phase D: a model has to write those);
  - it does not edit anything outside KNobs, and refuses to run if a pattern does
    not match exactly once (a silent no-op would be reported as "no change").

Cost per iteration, measured: 10 questions x ~140 s = ~23 min of answering, plus
~5 min of judging with the hosted judge = ~28 min. A four-candidate sweep is
therefore about two hours, which is why the loop uses dev-loop (10) and not
dev (32).

Usage:
    python3 scripts/rsi_sweep.py --list
    python3 scripts/rsi_sweep.py --knob maxSearchQueries --value 2
    python3 scripts/rsi_sweep.py --sweep            # all candidates, in order
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "rsi-ledger.md"
BASELINE_SCORECARD = REPO / "eval" / "runs" / "baseline.scorecard.json"
GO = REPO / ".toolchain" / "go" / "bin" / "go"
PY = REPO / ".venv314" / "bin" / "python"
RAGAS_PY = REPO / ".venv-ragas" / "bin" / "python"


# The genome's numeric half. `pattern` must match exactly once: a rewrite of the
# surrounding code that makes it match twice, or not at all, is a refusal rather
# than a guess at what was meant.
KNobs: Dict[str, Dict[str, object]] = {
    "maxSearchQueries": {
        "file": "internal/agent/queries.go",
        "pattern": r"(?m)^const maxSearchQueries = (\d+)$",
        "current": 3, "candidates": [2, 4],
        "why": "queries per rewrite: fewer means a narrower first recall, more means "
               "another retrieval call per round",
    },
    "enumMaxRounds": {
        "file": "internal/agent/coverage.go",
        "pattern": r"(?m)^const enumMaxRounds = (\d+)$",
        "current": 2, "candidates": [3],
        "why": "rounds an enumeration is bounded to; raising it costs time where the "
               "run currently stops early",
    },
    "DefaultMaxStateChars": {
        "file": "internal/agent/checker.go",
        "pattern": r"(?m)^const DefaultMaxStateChars = (\d+)$",
        "current": 10000, "candidates": [6000, 14000],
        "why": "evidence the checker reads: less is cheaper and may drop a passage it "
               "needed, more costs seconds per decision",
    },
    "SCAMaxRounds": {
        "file": "internal/agent/loop.go",
        "pattern": r"(?m)^\t\tSCAMaxRounds: (\d+),$",
        "current": 3, "candidates": [2],
        "why": "rounds per simple question: the whole cost curve is linear in this",
    },
    "ActionMaxTurns": {
        "file": "internal/agent/loop.go",
        "pattern": r"(?m)^\t\tActionMaxTurns:   (\d+),$",
        "current": 12, "candidates": [8],
        "why": "calls per turn: fewer means the planner must choose, more means it can "
               "dump several searches into one round",
    },
    "defaultToolLimit": {
        "file": "internal/agent/tools.go",
        "pattern": r"(?m)^const defaultToolLimit = (\d+)$",
        "current": 6, "candidates": [10],
        "why": "passages per tool call: more per call, fewer calls",
    },
}


def read_scorecard(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fitness(scorecard: Dict) -> Optional[float]:
    """The scalar a candidate is judged by: quality, when it is measurable.

    Cost and the guards are NOT folded in here — they are checked separately
    (see `accepts`), because folding them into one number lets a large quality
    gain pay for a guard violation, which is exactly the trade §13.1 forbids.
    """
    return scorecard.get("quality_macro")


def accepts(candidate: Dict, baseline: Dict, noise: float) -> Tuple[bool, str]:
    """The acceptance rule, and the reason in words for the ledger.

    Three conditions, all of which must hold:
      1. the improvement clears the noise band (a 10-question subset cannot
         resolve less, so a smaller delta is not evidence);
      2. the refusal rate on unanswerable questions does not fall (the guard
         §13.2 forbids trading it for score);
      3. no new invariant violation appears.
    """
    before, after = fitness(baseline), fitness(candidate)
    if after is None or before is None:
        return False, "quality_macro 不可测（判分有 NaN，先看 judge_nan_samples）"
    if after < before + noise:
        return False, f"提升 {after - before:+.3f} 未超过噪声带 {noise:.3f}"
    refusal_before = baseline.get("refusal_rate_on_null")
    refusal_after = candidate.get("refusal_rate_on_null")
    if refusal_before is not None and refusal_after is not None and refusal_after < refusal_before:
        return False, f"拒答率下降 {refusal_before}→{refusal_after}（守卫项，不允许用来换分）"
    new_violations = set(candidate.get("violations") or []) - set(baseline.get("violations") or [])
    if new_violations:
        return False, f"新增不变量违规：{sorted(new_violations)[:2]}"
    return True, f"提升 {after - before:+.3f} ≥ 噪声带 {noise:.3f}，守卫无回归"


def apply_knob(name: str, value: int) -> Tuple[Path, str, str]:
    """Rewrite one knob, returning (file, original_text, patched_text)."""
    spec = KNobs[name]
    path = REPO / str(spec["file"])
    original = path.read_text(encoding="utf-8")
    pattern = str(spec["pattern"])
    matches = re.findall(pattern, original)
    if len(matches) != 1:
        raise SystemExit(f"refusing to edit {name}: pattern matched {len(matches)} time(s), want exactly 1")
    patched = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), str(value)), original, count=1)
    if patched == original:
        raise SystemExit(f"refusing to continue: {name} was not changed")
    path.write_text(patched, encoding="utf-8")
    return path, original, patched


def rebuild() -> None:
    result = subprocess.run([str(GO), "build", "-o", "bin/freerag", "./cmd/freerag"],
                            cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"rebuild failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def measure(name: str) -> Optional[Dict]:
    """Answer dev-loop, then score it. Returns the candidate's scorecard."""
    run = subprocess.run([str(PY), "-u", "scripts/ragas_eval.py", "run",
                          "--split", "eval/dev-loop.json", "--kb", "eval/data-hybrid",
                          "--name", name], cwd=REPO, capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout[-1500:]); print(run.stderr[-1500:])
        return None
    score = subprocess.run([str(RAGAS_PY), "-u", "scripts/ragas_eval.py", "score",
                            "--run", f"eval/runs/{name}.jsonl", "--with-answer-relevancy",
                            "--judge-workers", "4"], cwd=REPO, capture_output=True, text=True)
    if score.returncode != 0:
        print(score.stdout[-1500:]); print(score.stderr[-1500:])
    return read_scorecard(REPO / "eval" / "runs" / f"{name}.scorecard.json")


def append_ledger(rows: List[str]) -> None:
    header = ("# RSI 账本（docs/plan.md §13.6）\n\n"
              "每次提案一行；**被否决的同样入账**——\"哪些方向试过且无效\"是 Phase E 的输入。\n\n"
              "| 日期 | 对象 | 修改 | 前 | 后 | 成本 | 结果 | 理由 |\n"
              "| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |\n")
    if not LEDGER.exists():
        LEDGER.write_text(header, encoding="utf-8")
    with LEDGER.open("a", encoding="utf-8") as sink:
        sink.write("\n".join(rows) + "\n")


def run_one(name: str, value: int, baseline: Dict, noise: float) -> bool:
    spec = KNobs[name]
    started = time.time()
    path, original, _ = apply_knob(name, value)
    try:
        rebuild()
        candidate = measure(f"sweep-{name}-{value}")
        if candidate is None:
            append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` | → {value} | "
                           f"{fitness(baseline)} | 失败 | — | 否决 | 运行或判分未完成 |"])
            return False
        ok, reason = accepts(candidate, baseline, noise)
        append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` | → {value} | "
                       f"{fitness(baseline)} | {fitness(candidate)} | "
                       f"{(time.time() - started) / 60:.0f} min | "
                       f"{'**接受**' if ok else '否决'} | {reason} |"])
        if ok:
            # Accepted: the file keeps the patched value, and the new scorecard
            # becomes the comparison point for the next candidate.
            shutil.copy(REPO / "eval" / "runs" / f"sweep-{name}-{value}.scorecard.json",
                        BASELINE_SCORECARD)
            print(f"ACCEPTED {name}={value}: {reason}")
            return True
        path.write_text(original, encoding="utf-8")   # revert
        rebuild()
        print(f"rejected {name}={value}: {reason}")
        return False
    except BaseException:
        path.write_text(original, encoding="utf-8")
        rebuild()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knob")
    parser.add_argument("--value", type=int)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--noise", type=float, default=0.10,
                        help="minimum quality_macro gain to accept; 10 questions resolve ~0.10 (plan §13.5)")
    args = parser.parse_args()

    if args.list:
        print(f"{'knob':24s} {'current':>8s}  candidates   why")
        for name, spec in KNobs.items():
            print(f"{name:24s} {spec['current']:>8}  {spec['candidates']}   {spec['why'][:60]}")
        total = sum(len(spec["candidates"]) for spec in KNobs.values())
        print(f"\ncandidates: {total}  x ~28 min = ~{total * 28 / 60:.1f} h for a full sweep")
        print(f"acceptance: quality_macro gain >= {args.noise}, refusal must not fall, no new violations")
        return 0

    baseline = read_scorecard(BASELINE_SCORECARD)
    if baseline is None:
        print(f"no baseline at {BASELINE_SCORECARD.relative_to(REPO)} — Phase B must finish first")
        return 1
    print(f"baseline quality_macro = {fitness(baseline)}, refusal = {baseline.get('refusal_rate_on_null')}")

    if args.knob:
        if args.knob not in KNobs:
            print(f"unknown knob {args.knob}; --list shows the declared genome")
            return 1
        if args.value is None:
            print("--value is required with --knob")
            return 1
        return 0 if run_one(args.knob, args.value, baseline, args.noise) else 0

    if args.sweep:
        for name, spec in KNobs.items():
            for value in spec["candidates"]:
                current = read_scorecard(BASELINE_SCORECARD) or baseline
                run_one(name, value, current, args.noise)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
