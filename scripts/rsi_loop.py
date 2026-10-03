#!/usr/bin/env python3
"""Run the improvement loop unattended over the whole declared menu. docs/plan.md §13.25.

One stage per candidate: the FULL dev split (32 questions, 8 of them unanswerable) at the
SHIPPED temperature 0.2, about 3h. That is the decision, and it is the only place the refusal
guard can be measured at all.

There used to be a cheap screen in front of it — the 10-question dev-loop at temperature 0,
45 min — and it was removed because it did not predict anything. Measured on the first seven
candidates: maxSearchQueries 3->4 scored +0.0867 on the screen and +0.0019 on the full split,
and SnippetsPerQuery 6->10 scored +0.060 then -0.057. A 0.085-0.117 swing on changes whose
real effect is under 0.05. The cause was mine: the screen ran at temperature 0 and the
decision at 0.2, which §13.18 had already recorded as two different distributions. A gate that
runs under different conditions than the decision it gates cannot predict it, and this one cost
45 minutes per candidate to be wrong in both directions — passing candidates that were then
rejected, and stopping ones that were never measured.

State rules, which is where an unattended loop gets dangerous:
  - the tree always equals the accepted state: a rejected candidate is reverted before the
    next one is applied, and an accepted one is committed at once;
  - the comparison scorecard always describes the accepted state, and is rewritten from an
    accepted candidate's own measurement;
  - every outcome goes to docs/rsi-ledger.md with the numbers and the reason, including the
    ones that failed to run.

Usage:
    python3 scripts/rsi_loop.py                  # the whole menu
    python3 scripts/rsi_loop.py --accepts 3       # stop early
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

# The decision threshold on the full dev split at the shipped temperature. Its own band has
# not been measured — a second full run costs 3h — so this is deliberately conservative: a
# candidate has to beat the accepted state by more than the screen's error turned out to be
# before it is believed. Worth revisiting with two full-dev runs of identical code.
VALIDATE_DELTA = 0.05

# The temperature the product ships with. The decision runs at this value; scripts/ragas_eval.py
# sets 0 for kernels it starts, which is right for measurement and wrong for a DECISION, since it
# is a different distribution than the one users get (see the module docstring).
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
        # --fresh is NOT optional here. `ragas_eval.py run` resumes by default, and a
        # decision must measure the code that is in the tree NOW: reusing a run from an
        # earlier attempt measures an earlier candidate. Measured: a decision "ran" the
        # full split in three minutes and then judged a stale file whose answers came from
        # a different build, which is how a candidate that had never been run got rejected.
        subprocess.run([str(P.PY), "-u", "scripts/ragas_eval.py", "run",
                        "--split", "eval/dev.json", "--kb", "eval/data-hybrid",
                        "--name", name, "--fresh"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if not path.exists():
        return None
    log(f"decide: judging {name}")
    # Four workers, not six: the provider rate-limits, and the failure mode is silent —
    # 17 to 19 of 24 samples came back NaN on one run at six, while the same settings on
    # another run at the same time were clean. The NaN guard refuses the decision, so this
    # costs a wasted 3h rather than a wrong verdict, but it still costs it.
    subprocess.run([str(P.RAGAS_PY), "-u", "scripts/ragas_eval.py", "score",
                    "--run", f"eval/runs/{name}.jsonl", "--with-answer-relevancy",
                    "--judge-workers", "4"], cwd=REPO, env=env, capture_output=True, text=True)
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
    """Do the files this loop may rewrite match their committed state?

    Only those files are checked. What has to be protected is that a revert is EXACT, and that
    only concerns the files the loop edits — which are also the only ones it commits, since
    `git add` is given these paths rather than `-A`. Checking the whole tree instead would let an
    untracked document being written elsewhere in the repo block a run that takes days.
    """
    watched = sorted({str(spec["file"]) for spec in S.KNobs.values()}
                     | {str(spec["file"]) for spec in P.PROMPTS.values()})
    out = subprocess.run(["git", "status", "--porcelain", "--"] + watched, cwd=REPO,
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
    candidate = measure_full_dev(f"validate-{name}-{value}", fresh=True)
    baseline = S.read_scorecard(RUNS / "baseline.scorecard.json") or {}
    ok, reason = decide(candidate or {}, baseline, VALIDATE_DELTA, "判决")
    if candidate is None:
        ok, reason = False, "判决运行未完成"
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` → {value} | "
                   f"{macro(baseline)} | {macro(candidate)} | "
                   f"{'接受' if ok else '否决'} | {reason} |"])
    if not ok:
        log(f"rejected {name}={value}: {reason}")
        S.restore()
        S.rebuild()
        return False
    # Accepted: the files stay changed, and the comparison point becomes this state.
    import shutil
    shutil.copy(RUNS / f"validate-{name}-{value}.scorecard.json", RUNS / "baseline.scorecard.json")
    commit(f"Accept {name} {spec['current']} -> {value} (full dev at the shipped temperature)\n\n"
           f"{reason}\n", [str(path.relative_to(REPO)), "docs/rsi-ledger.md"])
    log(f"ACCEPTED {name}={value} ({accepts + 1})")
    return True


def run_prompt(name: str, accepts: int) -> Optional[bool]:
    """One prompt candidate: proposed by the model, then decided on the full split."""
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
    validated = measure_full_dev(f"validate-{name}", fresh=True)
    full_baseline = S.read_scorecard(RUNS / "baseline.scorecard.json") or {}
    ok, reason = decide(validated or {}, full_baseline, VALIDATE_DELTA, "判决")
    append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{name}` +{len(new_prompt) - len(current)} chars | "
                   f"{macro(full_baseline)} | {macro(validated)} | {'接受' if ok else '否决'} | {reason} |"])
    if not ok:
        log(f"rejected {name}: {reason}")
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        S.rebuild()
        return False
    import shutil
    shutil.copy(RUNS / f"validate-{name}.scorecard.json", RUNS / "baseline.scorecard.json")
    commit(f"Accept a {name} edit (+{len(new_prompt) - len(current)} chars)\n\n{reason}\n",
           [str(path.relative_to(REPO)), "docs/rsi-ledger.md"])
    log(f"ACCEPTED {name} ({accepts + 1})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accepts", type=int, default=None,
                        help="stop early after this many acceptances (default: run the whole menu)")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="safety bound on attempts (default: the size of the declared menu)")
    args = parser.parse_args()

    if not tree_is_clean():
        log("refusing to start: the working tree is not clean, so a revert would not be exact")
        return 1

    # The goal is the declared menu, not a target number of acceptances.
    #
    # A target count is the wrong shape for an unattended run twice over: it keeps the loop
    # looking for material after the menu is exhausted, where there is none, and it applies
    # pressure to the one number that must never be pressured — the acceptance rule. The menu is
    # what the loop can actually vary; running it out is an honest end state, and so is
    # "0 accepted".
    menu_size = sum(len(spec["candidates"]) for spec in S.KNobs.values()) + len(P.PROMPTS)
    goal = args.accepts if args.accepts is not None else menu_size
    tried_goal = args.max_candidates if args.max_candidates is not None else menu_size
    log(f"goal: the whole menu ({menu_size} candidate(s))"
        + (f", stopping early after {args.accepts} acceptance(s)" if args.accepts is not None else ""))

    accepted = 0
    tried = 0
    for name, spec in S.KNobs.items():
        for value in spec["candidates"]:
            if accepted >= goal or tried >= tried_goal:
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
        if accepted >= goal or tried >= tried_goal:
            break

    for name in P.PROMPTS:
        if accepted >= goal or tried >= tried_goal:
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
