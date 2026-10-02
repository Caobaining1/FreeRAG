#!/usr/bin/env python3
"""Run the improvement loop unattended until N changes are accepted. docs/plan.md §13.22.

Two stages per candidate, because §13.21 showed that one stage is not enough:

  screen   dev-loop (10 questions) at temperature 0, ~45 min. Cheap, and it only decides
           whether the candidate is worth the expensive stage. A screen pass is NOT an
           acceptance: the one change this loop accepted on a screen alone was rolled back
           after validation (-0.057 macro, and the refusal rate fell with it).
  decide   the FULL dev split (32 questions, 8 of them unanswerable) at the SHIPPED
           temperature 0.2, ~3h. This is the decision, and it is also the only place the
           refusal guard can be measured at all.

State rules, which is where an unattended loop gets dangerous:
  - the tree always equals the accepted state: a rejected candidate is reverted before the
    next one is applied, and an accepted one is committed at once;
  - the two comparison scorecards always describe the accepted state, and are rewritten
    from the candidate's own measurements when it is accepted;
  - every outcome goes to docs/rsi-ledger.md with the numbers and the reason, including the
    ones that failed to run.

Usage:
    python3 scripts/rsi_loop.py --accepts 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rsi_propose as P  # noqa: E402
import rsi_sweep as S  # noqa: E402

REPO = HERE.parent
RUNS = REPO / "eval" / "runs"
LEDGER = REPO / "docs" / "rsi-ledger.md"

# Screen stage: the dev-loop comparison point (temperature 0) and its threshold. The
# measured band there is 0.0271, so 0.05 is about two bands.
SCREEN_DELTA = 0.05

# Decide stage: the full-dev comparison point (temperature 0.2) and its threshold. Its band
# has NOT been measured (a second full run costs 3h), so this is deliberately conservative:
# a candidate has to beat the accepted state by more than the dev-loop band moved under
# validation (-0.057) before it is believed.
VALIDATE_DELTA = 0.05

# The temperature the product ships with. Measurements for a decision run at this value;
# screens run at 0 (scripts/ragas_eval.py sets that for every kernel it starts).
SHIPPED_TEMPERATURE = "0.2"


def log(line: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def append_ledger(rows: List[str]) -> None:
    with LEDGER.open("a", encoding="utf-8") as sink:
        sink.write("\n".join(rows) + "\n")


def scorecard(name: str) -> Optional[Dict]:
    path = RUNS / f"{name}.scorecard.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def measure_full_dev(name: str, fresh: bool) -> Optional[Dict]:
    """The deciding measurement: the full split at the shipped temperature."""
    path = RUNS / f"{name}.jsonl"
    env = dict(os.environ)
    env["FREERAG_GENERATION_TEMPERATURE"] = SHIPPED_TEMPERATURE
    if fresh or not path.exists():
        log(f"decide: running {name} on the full dev split at {SHIPPED_TEMPERATURE}")
        subprocess.run([str(P.PY), "-u", "scripts/ragas_eval.py", "run",
                        "--split", "eval/dev.json", "--kb", "eval/data-hybrid",
                        "--name", name], cwd=REPO, env=env, capture_output=True, text=True)
    if not path.exists():
        return None
    log(f"decide: judging {name}")
    subprocess.run([str(P.RAGAS_PY), "-u", "scripts/ragas_eval.py", "score",
                    "--run", f"eval/runs/{name}.jsonl", "--with-answer-relevancy",
                    "--judge-workers", "6"], cwd=REPO, env=env, capture_output=True, text=True)
    return scorecard(name)


def macro(card: Optional[Dict]) -> Optional[float]:
    return None if not card else card.get("quality_macro")


def decide(candidate: Dict, baseline: Dict, delta: float, stage: str) -> Tuple[bool, str]:
    """The deciding rule, shared by both stages so they cannot drift apart."""
    incomplete = S.measurement_is_whole(candidate, baseline)
    if incomplete:
        return False, incomplete
    before, after = macro(baseline), macro(candidate)
    if before is None or after is None:
        return False, "quality_macro 不可测"
    if after < before + delta:
        return False, f"{stage}: 提升 {after - before:+.4f} 未超过阈值 {delta:.3f}"
    # The refusal guard. Skipped only when the split cannot measure it, and said so.
    n_null = candidate.get("n_null") or 0
    rb, ra = baseline.get("refusal_rate_on_null"), candidate.get("refusal_rate_on_null")
    if n_null < S.MIN_NULL_FOR_GUARD:
        return True, f"{stage}: {after - before:+.4f} ≥ {delta:.3f}（拒答守卫未测：{n_null} 道 null 题）"
    if rb is not None and ra is not None and ra < rb:
        return False, f"{stage}: 拒答率 {rb}→{ra} 下降（§13.2 守卫，不允许换分）"
    return True, f"{stage}: {after - before:+.4f} ≥ {delta:.3f}，拒答 {rb}→{ra} 无回归"


def commit(message: str, paths: List[str]) -> None:
    subprocess.run(["git", "add"] + paths, cwd=REPO, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=REPO, capture_output=True, text=True)
    subprocess.run(["git", "push", "git@github.com:Caobaining1/FreeRAG.git", "main"],
                   cwd=REPO, capture_output=True, text=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def tree_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return out == ""


def run_knob(name: str, value: int, accepts: int) -> Optional[bool]:
    """One knob candidate. None if it could not be run."""
    if name in S.INERT_KNOBS:
        log(f"refusing {name}: inert ({S.INERT_KNOBS[name]})")
        return None
    spec = S.KNobs[name]
    log(f"candidate knob {name} {spec['current']} -> {value}")
    S._TOUCHED.clear()
    path, original, _ = S.apply_knob(name, value)
    try:
        S.rebuild()
    except SystemExit as exc:
        S.restore()
        log(f"rebuild failed, reverted: {exc}")
        return None
    screen = S.measure(f"screen-{name}-{value}")
    if screen is None:
        S.restore()
        return None
    ok, reason = decide(screen, S.read_scorecard(S.BASELINE_SCORECARD) or {}, SCREEN_DELTA, "筛选")
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` → {value} | 筛选 | "
                   f"{macro(S.read_scorecard(S.BASELINE_SCORECARD))} | {macro(screen)} | "
                   f"{'过' if ok else '止'} | {reason} |"])
    if not ok:
        log(f"screen rejected {name}={value}: {reason}")
        S.restore()
        S.rebuild()
        return False
    log(f"screen passed {name}={value}: {reason} — validating")
    validated = measure_full_dev(f"validate-{name}-{value}", fresh=True)
    full_baseline = S.read_scorecard(RUNS / "baseline.scorecard.json") or {}
    ok2, reason2 = decide(validated or {}, full_baseline, VALIDATE_DELTA, "判决")
    if validated is None:
        ok2, reason2 = False, "判决运行未完成"
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` → {value} | 判决 | "
                   f"{macro(full_baseline)} | {macro(validated)} | "
                   f"{'接受' if ok2 else '否决'} | {reason2} |"])
    if not ok2:
        log(f"decide rejected {name}={value}: {reason2}")
        S.restore()
        S.rebuild()
        return False
    # Accepted: the files stay changed, and the comparison points become this state.
    import shutil
    shutil.copy(RUNS / f"screen-{name}-{value}.scorecard.json", S.BASELINE_SCORECARD)
    shutil.copy(RUNS / f"validate-{name}-{value}.scorecard.json", RUNS / "baseline.scorecard.json")
    commit(f"Accept {name} {spec['current']} -> {value} (screen + full-dev validation)\n\n"
           f"screen {reason}\nfull dev {reason2}\n", [str(path.relative_to(REPO)),
                                                     "docs/rsi-ledger.md"])
    log(f"ACCEPTED {name}={value} ({accepts + 1})")
    return True


def run_prompt(name: str, accepts: int) -> Optional[bool]:
    """One prompt candidate, proposed by the model and then screened and validated."""
    path, whole, current = P.read_prompt(name)
    log(f"candidate prompt {name} ({len(current)} chars)")
    cases = P.worst_cases(6)
    # Same defaults as rsi_propose's own CLI, read from the same place, so the loop and the
    # hand-run path cannot drift apart.
    proposal = P.propose(name, current, cases,
                         os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4.1-flash"),
                         os.environ.get("RAGAS_JUDGE_BASE", "https://llm.goaichat.top/v1"))
    if proposal.get("no_edit"):
        append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` | — | — | — | 无提案 | "
                       f"{proposal.get('rationale', '')[:120]} |"])
        log(f"proposer declined: {proposal.get('rationale')}")
        return None
    new_prompt = proposal.get("new_prompt") or ""
    if not new_prompt or new_prompt == current or "`" in new_prompt:
        log("proposal unusable")
        return None
    path.write_text(whole.replace(f"`{current}`", f"`{new_prompt}`", 1), encoding="utf-8")
    try:
        S.rebuild()
    except SystemExit as exc:
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        log(f"rebuild failed, reverted: {exc}")
        return None
    screen = S.measure(f"screen-{name}")
    base = S.read_scorecard(S.BASELINE_SCORECARD) or {}
    ok, reason = decide(screen or {}, base, SCREEN_DELTA, "筛选") if screen else (False, "筛选未完成")
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` +{len(new_prompt) - len(current)} chars | 筛选 | "
                   f"{macro(base)} | {macro(screen)} | {'过' if ok else '止'} | {reason} |"])
    if not ok:
        log(f"screen rejected {name}: {reason}")
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        S.rebuild()
        return False
    validated = measure_full_dev(f"validate-{name}", fresh=True)
    full_baseline = S.read_scorecard(RUNS / "baseline.scorecard.json") or {}
    ok2, reason2 = decide(validated or {}, full_baseline, VALIDATE_DELTA, "判决")
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` +{len(new_prompt) - len(current)} chars | 判决 | "
                   f"{macro(full_baseline)} | {macro(validated)} | {'接受' if ok2 else '否决'} | {reason2} |"])
    if not ok2:
        log(f"decide rejected {name}: {reason2}")
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        S.rebuild()
        return False
    import shutil
    shutil.copy(RUNS / f"screen-{name}.scorecard.json", S.BASELINE_SCORECARD)
    shutil.copy(RUNS / f"validate-{name}.scorecard.json", RUNS / "baseline.scorecard.json")
    commit(f"Accept a {name} edit (+{len(new_prompt) - len(current)} chars)\n\n"
           f"screen {reason}\nfull dev {reason2}\n", [str(path.relative_to(REPO)),
                                                     "docs/rsi-ledger.md"])
    log(f"ACCEPTED {name} ({accepts + 1})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accepts", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=60)
    args = parser.parse_args()

    if not tree_is_clean():
        log("refusing to start: the working tree is not clean, so a revert would not be exact")
        return 1

    accepted = 0
    tried = 0
    for name, spec in S.KNobs.items():
        for value in spec["candidates"]:
            if accepted >= args.accepts or tried >= args.max_candidates:
                break
            tried += 1
            try:
                if run_knob(name, value, accepted) is True:
                    accepted += 1
            except Exception as exc:  # a broken candidate must not end the run
                append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` → {value} | 异常 | — | — | 跳过 | "
                               f"{type(exc).__name__}: {str(exc)[:100]} |"])
                log(f"candidate {name}={value} raised: {exc!r}")
                subprocess.run(["git", "checkout", "--", "."], cwd=REPO, capture_output=True)
        if accepted >= args.accepts or tried >= args.max_candidates:
            break

    for name in P.PROMPTS:
        if accepted >= args.accepts or tried >= args.max_candidates:
            break
        tried += 1
        try:
            if run_prompt(name, accepted) is True:
                accepted += 1
        except Exception as exc:
            append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` | 异常 | — | — | 跳过 | "
                           f"{type(exc).__name__}: {str(exc)[:100]} |"])
            log(f"candidate {name} raised: {exc!r}")
            subprocess.run(["git", "checkout", "--", "."], cwd=REPO, capture_output=True)

    log(f"stopping: {accepted} accepted, {tried} candidate(s) tried")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
