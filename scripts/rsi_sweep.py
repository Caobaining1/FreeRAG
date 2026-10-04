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
# The dev-loop reduction, not the full-dev scorecard: a candidate is measured on
# dev-loop (10 questions, ~28 min), so the comparison point must be the same
# questions. See scripts/rsi_baseline.py.
BASELINE_SCORECARD = REPO / "eval" / "runs" / "devloop-baseline.scorecard.json"

# Knobs that look editable and are not. `defaultToolLimit` was swept, "improved"
# quality_macro by 0.079 and was ACCEPTED before anyone checked that it is the last
# fallback in Toolbox.limit(): every production Toolbox sets DefaultLimit (from
# Spec.SnippetsPerQuery, loop.go:754 and cmd/freerag/kb.go:132), so `return
# defaultToolLimit` is unreachable and the change was inert — the 0.079 was run-to-run
# noise, and accepting it taught the loop to prefer noise. A knob must be shown live
# before it is swept, and this table is where that is recorded.
INERT_KNOBS = {
    "defaultToolLimit":
        "shadowed by Toolbox.DefaultLimit, which every production construction sets "
        "from Spec.SnippetsPerQuery (loop.go:754, cmd/freerag/kb.go:132)",
}

# A refusal rate over fewer null questions than this is not comparable (dev-loop
# carries one, so its rate is 0.0 or 1.0). Below it the guard is reported as
# unmeasured rather than enforced.
MIN_NULL_FOR_GUARD = 4
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
        "current": 3, "candidates": [2, 4, 5],
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
        "current": 10000, "candidates": [4000, 6000, 14000, 20000],
        "why": "evidence the checker reads: less is cheaper and may drop a passage it "
               "needed, more costs seconds per decision",
    },
    "SCAMaxRounds": {
        "file": "internal/agent/loop.go",
        "pattern": r"(?m)^\t\tSCAMaxRounds: (\d+),$",
        "current": 3, "candidates": [2, 4],
        "why": "rounds per simple question: the whole cost curve is linear in this",
    },
    "ActionMaxTurns": {
        "file": "internal/agent/loop.go",
        "pattern": r"(?m)^\t\tActionMaxTurns:   (\d+),$",
        "current": 12, "candidates": [8, 16],
        "why": "calls per turn: fewer means the planner must choose, more means it can "
               "dump several searches into one round",
    },
    # The value that actually reaches Toolbox.DefaultLimit (loop.go:754). Note that
    # internal/agent/loop_test.go pins Medium().SnippetsPerQuery at 6 — a sweep of
    # this knob must update that expectation too, which the build step will not
    # catch on its own.
    "SnippetsPerQuery": {
        # 10 since the change shipped (see docs/rsi-ledger.md 2026-10-03); the candidate
        # values below are therefore DECREASES from the shipped value, not increases.
        "file": "internal/agent/loop.go", "current": 10,
        "pattern": r"SnippetsPerQuery:\s*(\d+)", "candidates": [4, 8],
        # TestMediumSpecMatchesPlan pins the shipped spec, and its job is to catch
        # accidental drift — not to forbid a measured change, which is what this loop
        # produces. So the sweep updates the pin with the value, leaving the test still
        # guarding every OTHER field and every unattempted edit.
        "also_update": [{"file": "internal/agent/loop_test.go",
                         "pattern": r"(spec\.SnippetsPerQuery != )(\d+)",
                         "replacement": r"\g<1>{value}"}],
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


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

def measurement_is_whole(candidate: Dict, baseline: Dict) -> Optional[str]:
    """None when both runs were scored on the same questions, else why not.

    A metric whose judge call failed is NaN, and NaN samples are DROPPED from that
    metric's mean — so a run with more failed judge calls has its macro computed over a
    smaller, easier sample set and can score HIGHER for that. Measured: a candidate had 3
    NaN context_precision samples against the baseline's 1 and the macro moved +0.054,
    which is not decidable from those two numbers. The floor is deliberately loose (one
    lost sample is tolerated) and any breach refuses the decision rather than shrinking it.
    """
    for card, side in ((candidate, "candidate"), (baseline, "baseline")):
        if card.get("usable") is False or (card.get("n_failed") or 0) > 0:
            return (f"measurement unusable: the {side} run had {card.get('n_failed')} question(s) "
                    f"that failed to run ({card.get('unusable_reason') or 'see the run log'}); "
                    f"re-run it before deciding")
    for name in METRIC_NAMES:
        before = (baseline.get("judge_nan_samples") or {}).get(name, 0)
        after = (candidate.get("judge_nan_samples") or {}).get(name, 0)
        if after > max(before, 1):
            return (f"measurement incomplete: {name} lost {after} sample(s) to judge failures "
                    f"against the baseline's {before}; re-score before deciding")
    return None


def accepts(candidate: Dict, baseline: Dict, noise: float) -> Tuple[bool, str]:
    """The acceptance rule, and the reason in words for the ledger.

    Three conditions, all of which must hold:
      1. the improvement clears the noise band (a 10-question subset cannot
         resolve less, so a smaller delta is not evidence);
      2. the refusal rate on unanswerable questions does not fall (the guard
         §13.2 forbids trading it for score);
      3. no new invariant violation appears.

    And a fourth that is not an opinion about the change but about the measurement: both
    runs must have been scored on the same questions. See measurement_is_whole.
    """
    incomplete = measurement_is_whole(candidate, baseline)
    if incomplete:
        return False, incomplete
    before, after = fitness(baseline), fitness(candidate)
    if after is None or before is None:
        return False, "quality_macro 不可测（判分有 NaN，先看 judge_nan_samples）"
    if after < before + noise:
        return False, f"提升 {after - before:+.3f} 未超过噪声带 {noise:.3f}"
    refusal_before = baseline.get("refusal_rate_on_null")
    refusal_after = candidate.get("refusal_rate_on_null")
    n_null = candidate.get("n_null") or 0
    if n_null >= MIN_NULL_FOR_GUARD and refusal_before is not None and refusal_after is not None \
            and refusal_after < refusal_before:
        return False, f"拒答率下降 {refusal_before}→{refusal_after}（守卫项，不允许用来换分）"
    guard = "" if n_null >= MIN_NULL_FOR_GUARD else f"（拒答守卫未测：只有 {n_null} 道 null 题）"
    new_violations = set(candidate.get("violations") or []) - set(baseline.get("violations") or [])
    if new_violations:
        return False, f"新增不变量违规：{sorted(new_violations)[:2]}"
    return True, f"提升 {after - before:+.3f} ≥ 噪声带 {noise:.3f}，守卫无回归{guard}"


# Every file this script has rewritten, with its original text, so a revert is exact.
# The main edit is not always the only file: a knob pinned by a regression test has to
# update the pin in the same change (see also_update), and reverting only the knob
# would leave a test asserting the REJECTED value — i.e. a rejected change that leaves
# the tree red.
_TOUCHED: List[Tuple[Path, str]] = []


def _write(path: Path, text: str) -> None:
    _TOUCHED.append((path, path.read_text(encoding="utf-8")))
    path.write_text(text, encoding="utf-8")


def restore() -> None:
    """Put every rewritten file back, newest first."""
    while _TOUCHED:
        path, original = _TOUCHED.pop()
        path.write_text(original, encoding="utf-8")


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
    _write(path, patched)
    for extra in spec.get("also_update") or []:
        extra_path = REPO / str(extra["file"])
        text = extra_path.read_text(encoding="utf-8")
        found = re.findall(str(extra["pattern"]), text)
        want = int(extra.get("expect", 1))
        if len(found) != want:
            raise SystemExit(
                f"refusing to edit {extra['file']}: its pin matched {len(found)} time(s), want {want}. "
                f"A swept knob that stops updating its pin leaves a failing test behind.")
        _write(extra_path, re.sub(str(extra["pattern"]),
                                  str(extra["replacement"]).replace("{value}", str(value)), text, count=want))
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
        restore()   # revert, including any test pin that was updated with it
        rebuild()
        print(f"rejected {name}={value}: {reason}")
        return False
    except BaseException:
        restore()
        rebuild()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knob")
    parser.add_argument("--value", type=int)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="actually apply the change, rebuild and re-measure. Without it "
                             "the command prints what it WOULD do and stops: a bare "
                             "`--knob X --value V` otherwise edits the source and starts a "
                             "~28-minute measurement, which is not what a flag named --knob "
                             "should do on its own (learned by doing exactly that).")
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

    if args.knob and args.knob in INERT_KNOBS:
        print(f"refusing: {args.knob} is not a knob")
        print(f"  {INERT_KNOBS[args.knob]}")
        return 1

    if args.knob:
        if args.knob not in KNobs:
            print(f"unknown knob {args.knob}; --list shows the declared genome")
            return 1
        if args.value is None:
            print("--value is required with --knob")
            return 1
        spec = KNobs[args.knob]
        print(f"plan: {args.knob} {spec['current']} -> {args.value} in {spec['file']}")
        print(f"      基线 {BASELINE_SCORECARD.name}: quality_macro={fitness(baseline)}, "
              f"拒答率={baseline.get('refusal_rate_on_null')}")
        print(f"      验收：提升 ≥ {args.noise}，拒答率不得下降，不得新增违规")
        print(f"      成本：约 28 分钟（10 题问答 + 判分）")
        if not args.execute:
            print("\n(dry plan — 加 --execute 才会真的改源码、重建并测量)")
            return 0
        return 0 if run_one(args.knob, args.value, baseline, args.noise) else 0

    if args.sweep:
        total = sum(len(spec["candidates"]) for spec in KNobs.values())
        print(f"plan: {total} candidate(s) over {len(KNobs)} knob(s), 约 {total * 28 / 60:.1f} 小时")
        if not args.execute:
            print("\n(dry plan — 加 --execute 才会真的执行)")
            return 0
        for name, spec in KNobs.items():
            for value in spec["candidates"]:
                current = read_scorecard(BASELINE_SCORECARD) or baseline
                run_one(name, value, current, args.noise)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
