#!/usr/bin/env python3
"""Phase D: the system edits its own prompts — docs/plan.md §13.4.

This is the part that is actually self-improvement rather than tuning. One
iteration:

    1. read which questions the baseline scored worst (per-question scores, not
       just the mean);
    2. hand the failures and the CURRENT prompt text to a proposer model, and ask
       for the smallest edit that would address them, with a reason;
    3. apply it, rebuild, re-answer dev-loop, re-score;
    4. accept only if the gain clears the MEASURED noise band, the refusal rate
       does not fall and no invariant is violated; otherwise revert.

Two things are different from Phase C on purpose:

  - the proposer runs on a HOSTED model that is not the generator (`§13.5`): a
    model editing the prompt of another model whose output it then grades is the
    self-preference loop §13.1 warns about;
  - the revert is GIT-based (`git checkout -- <file>`), not a regex-undo of my own
    bookkeeping: prompt edits are multi-line string literals, and a partial revert
    would leave the source in a state neither the old nor the new prompt describes.

The genome is DECLARED, like Phase C's knobs, and for the same reason: §13.2 also
names what must never change (the guards, the evaluation, the tool surface). A
script that walked the tree with a regex would eventually edit one of those, and
the failure would look like an improvement.

Usage:
    python3 scripts/rsi_propose.py --list
    python3 scripts/rsi_propose.py --prompt queryRewriteSystemPrompt --dry-run
    python3 scripts/rsi_propose.py --prompt queryRewriteSystemPrompt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "rsi-ledger.md"
RUNS = REPO / "eval" / "runs"
GO = REPO / ".toolchain" / "go" / "bin" / "go"
PY = REPO / ".venv314" / "bin" / "python"
RAGAS_PY = REPO / ".venv-ragas" / "bin" / "python"

# Phase D's genome: the text half. Each entry names the Go file and the constant
# whose raw string literal is the prompt. Phase E will add "the prompt that
# writes these proposals" as an entry here — that is the recursion.
PROMPTS: Dict[str, Dict[str, str]] = {
    "queryRewriteSystemPrompt": {
        "file": "internal/agent/queries.go",
        "why": "turns the question into round 1's queries; feeds every retrieval",
    },
    "rewriteSystemPrompt": {
        "file": "internal/agent/loop.go",
        "why": "the gap-driven rewrite after an INSUFFICIENT verdict",
    },
    "toolPlanSystemPrompt": {
        "file": "internal/agent/loop.go",
        "why": "the planner's own rules (scope, batching, repeats)",
    },
    "decomposeSystemPrompt": {
        "file": "internal/agent/decompose.go",
        "why": "how the question is split into sub-questions",
    },
    "extractMembersSystemPrompt": {
        "file": "internal/agent/enumerate.go",
        "why": "how enumeration members are named",
    },
}

PROPOSER_SYSTEM = """You improve ONE prompt in a retrieval-augmented answering system.

You are given the current prompt and the cases where the system scored worst. Propose the SMALLEST
edit to that prompt that would plausibly fix those cases.

Rules:
- Change as little as possible. One clause or one rule, not a rewrite. A large edit cannot be
  attributed to any single cause when it is measured.
- Keep every existing rule that is not contradicted by the failures. Rules exist because of specific
  observed failures; deleting one to fix another usually trades one failure for the old one.
- Do not add facts about the corpus, do not name specific answers, and do not mention the evaluation.
- The prompt is a raw Go string literal: no backticks, no unescaped double quotes inside it.
- If the failures do NOT look like something this prompt can fix (they may be retrieval, parsing or
  scoring problems), say so instead of guessing: set "no_edit" to true and explain.

Reply with JSON only:
{"no_edit": false, "rationale": "one sentence", "new_prompt": "the complete new prompt text"}"""


# ---------------------------------------------------------------------------


def read_prompt(name: str) -> Tuple[Path, str, str]:
    """Return (file, whole file text, current prompt text)."""
    spec = PROMPTS[name]
    path = REPO / spec["file"]
    text = path.read_text(encoding="utf-8")
    pattern = r"(?ms)^const %s = `(.*?)`\s*$" % re.escape(name)
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise SystemExit(f"refusing to edit {name}: found {len(found)} literal(s), want exactly 1")
    return path, text, found[0]


def worst_cases(limit: int) -> List[Dict]:
    """The questions the baseline answered worst, with what the system produced."""
    records = [json.loads(line) for line in (RUNS / "baseline.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    scorecard = json.loads((RUNS / "baseline.scores.json").read_text(encoding="utf-8"))
    answerable = [r for r in records if r.get("type") != "null_query" and not r.get("error")]
    if len(answerable) != len(scorecard):
        raise SystemExit("baseline.jsonl and baseline.scores.json disagree; re-score before proposing")

    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

    def mean(row: Dict) -> Optional[float]:
        values = [row.get(m) for m in metrics if row.get(m) is not None and row.get(m) == row.get(m)]
        return sum(values) / len(values) if values else None

    pairs = [(r, s, mean(s)) for r, s in zip(answerable, scorecard)]
    pairs = [p for p in pairs if p[2] is not None]
    pairs.sort(key=lambda p: p[2])
    return [{"question": r["question"], "expected": r["ground_truth"],
             "answer": (r["answer"] or "")[:900], "score": round(m, 3),
             "scores": {k: s.get(k) for k in metrics}}
            for r, s, m in pairs[:limit]]


def propose(prompt_name: str, current: str, cases: List[Dict], model: str, base: str) -> Dict:
    """Ask the proposer for one edit. Plain HTTP, no SDK.

    Deliberately not langchain: this needs one chat completion, and importing an
    SDK would tie the RSI loop to the evaluation venv for no benefit. The judge
    side keeps ragas; the proposer side has no dependency at all.

    Also deliberately NOT the generator (`§13.5`): a model editing another model's
    prompt and then grading its output is the self-preference loop the plan rules out.
    """
    failing = "\n\n".join(
        f"[{i + 1}] question: {c['question'][:220]}\n"
        f"    expected: {str(c['expected'])[:120]}\n"
        f"    system answer: {c['answer'][:400]}\n"
        f"    per-metric: {c['scores']}"
        for i, c in enumerate(cases))
    body = {
        "model": model,
        "temperature": 0,
        # Measured on this endpoint (deepseek-v4.1-flash), same prompt, four configs:
        #   max_tokens=8192                      -> content EMPTY, finish_reason=stop
        #   max_tokens=32768                      -> 227 chars of JSON
        #   max_tokens=8192 + reasoning_effort=none -> 202 chars of JSON, reasoning 0
        #   max_tokens=8192 + thinking=false      -> JSON, but 2639 chars of reasoning
        # The first is the dangerous one: a reasoning model that spends the whole
        # budget before writing anything returns SUCCESS with no proposal, so a
        # silent no-op would be recorded as "the proposer declined".
        "max_tokens": 32768,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content":
                f"PROMPT NAME: {prompt_name}\n\nCURRENT PROMPT:\n\"\"\"\n{current}\n\"\"\"\n\n"
                f"CASES WHERE THE SYSTEM SCORED WORST ({len(cases)} of them):\n{failing}\n\n"
                f"Propose the smallest edit."},
        ],
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + (os.environ.get("JUDGE_API_KEY") or ""),
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"proposer call failed: HTTP {exc.code} {exc.read().decode('utf-8')[:300]}")
    message = (data.get("choices") or [{}])[0].get("message") or {}
    text = message.get("content") or ""
    if not text.strip():
        # Loud, not silent: a reasoning model that put everything in
        # `reasoning_content` has not proposed an edit, and treating that as
        # "no change" would record a proposal that never happened.
        raise SystemExit(f"proposer returned no content (finish_reason="
                         f"{(data.get('choices') or [{}])[0].get('finish_reason')!r})")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise SystemExit(f"proposer did not return JSON:\n{text[:500]}")
    return json.loads(match.group(0))


def measure(name: str) -> Optional[Dict]:
    run = subprocess.run([str(PY), "-u", "scripts/ragas_eval.py", "run",
                          "--split", "eval/dev-loop.json", "--kb", "eval/data-hybrid",
                          "--name", name], cwd=REPO, capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout[-1200:]); print(run.stderr[-1200:]); return None
    subprocess.run([str(RAGAS_PY), "-u", "scripts/ragas_eval.py", "score",
                    "--run", f"eval/runs/{name}.jsonl", "--with-answer-relevancy",
                    "--judge-workers", "4"], cwd=REPO, capture_output=True, text=True)
    path = RUNS / f"{name}.scorecard.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_once(prompt_name: str, args, case_limit: int) -> int:
    path, whole, current = read_prompt(prompt_name)
    print(f"prompt {prompt_name}: {len(current)} chars in {path.relative_to(REPO)}")

    cases = worst_cases(case_limit)
    print(f"worst {len(cases)} case(s): scores {[c['score'] for c in cases]}")

    proposal = propose(prompt_name, current, cases, args.proposer_model, args.proposer_base)
    if proposal.get("no_edit"):
        print(f"proposer declined: {proposal.get('rationale')}")
        append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{prompt_name}` | — | — | — | — | 无提案 | "
                       f"{proposal.get('rationale', '')[:120]} |"])
        return 0
    new_prompt = proposal.get("new_prompt", "")
    if not new_prompt or new_prompt == current:
        print("proposer returned no usable change")
        return 1
    if "`" in new_prompt:
        print("proposer used a backtick, which would break the Go literal; refusing")
        return 1

    print(f"\n--- rationale ---\n{proposal.get('rationale')}")
    diff = len(new_prompt) - len(current)
    print(f"--- size: {len(current)} -> {len(new_prompt)} chars ({diff:+d}) ---")
    if args.dry_run:
        print("--- new prompt (dry run) ---")
        print(new_prompt)
        return 0

    # Apply, measure, decide. The revert is `git checkout`, not a text undo.
    path.write_text(whole.replace(f"`{current}`", f"`{new_prompt}`", 1), encoding="utf-8")
    try:
        rebuild = subprocess.run([str(GO), "build", "-o", "bin/freerag", "./cmd/freerag"],
                                 cwd=REPO, capture_output=True, text=True)
        if rebuild.returncode != 0:
            print("rebuild failed; reverting")
            subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
            return 1
        started = time.time()
        candidate = measure(f"rsi-{prompt_name}")
        baseline = json.loads((RUNS / "baseline.scorecard.json").read_text(encoding="utf-8"))
        ok, reason = accepts(candidate, baseline, args.noise)
        append_ledger([f"| {time.strftime('%Y-%m-%d')} | `{prompt_name}` | "
                       f"{diff:+d} chars: {(proposal.get('rationale') or '')[:80]} | "
                       f"{quality(baseline)} | "
                       f"{quality(candidate) if candidate else '失败'} | "
                       f"{(time.time() - started) / 60:.0f} min | {'**接受**' if ok else '否决'} | {reason} |"])
        if ok:
            print(f"ACCEPTED: {reason}")
            return 0
        print(f"rejected: {reason}")
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        subprocess.run([str(GO), "build", "-o", "bin/freerag", "./cmd/freerag"], cwd=REPO)
        return 0
    except BaseException:
        subprocess.run(["git", "checkout", "--", str(path.relative_to(REPO))], cwd=REPO)
        raise


def quality(scorecard: Optional[Dict]) -> object:
    """The scalar a candidate is judged by. Both sides are dev-loop reductions."""
    if not scorecard:
        return "失败"
    return scorecard.get("quality_macro")


# A refusal rate measured on fewer null questions than this is not comparable: the
# dev-loop split carries ONE, so its rate is 0.0 or 1.0 and a candidate that refused
# it by luck would look like a regression (or a guard failure would look like a
# pass). Below the threshold the guard is reported as unmeasured rather than
# enforced, which is a documented weakening, not a silent one.
MIN_NULL_FOR_GUARD = 4


def accepts(candidate: Optional[Dict], baseline: Dict, noise: float) -> Tuple[bool, str]:
    if candidate is None:
        return False, "运行或判分未完成"
    before, after = quality(baseline), quality(candidate)
    if not isinstance(before, float) or not isinstance(after, float):
        return False, f"质量不可测（{before} → {after}）"
    if after < before + noise:
        return False, f"提升 {after - before:+.3f} 未超过实测噪声带 {noise:.3f}"
    rb, ra = baseline.get("refusal_rate_on_null"), candidate.get("refusal_rate_on_null")
    n_null = candidate.get("n_null") or 0
    if n_null >= MIN_NULL_FOR_GUARD and rb is not None and ra is not None and ra < rb:
        return False, f"拒答率 {rb}→{ra} 下降（守卫项）"
    guard = "" if n_null >= MIN_NULL_FOR_GUARD else f"（拒答守卫未测：只有 {n_null} 道 null 题）"
    new_violations = set(candidate.get("violations") or []) - set(baseline.get("violations") or [])
    if new_violations:
        return False, f"新增违规 {sorted(new_violations)[:2]}"
    return True, f"提升 {after - before:+.3f} ≥ 噪声带 {noise:.3f}，守卫无回归{guard}"


def append_ledger(rows: List[str]) -> None:
    header = ("# RSI 账本（docs/plan.md §13.6）\n\n"
              "每次提案一行；**被否决的同样入账**——\"哪些方向试过且无效\"是 Phase E 的输入。\n\n"
              "| 日期 | 对象 | 修改 | 前 | 后 | 成本 | 结果 | 理由 |\n"
              "| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |\n")
    if not LEDGER.exists():
        LEDGER.write_text(header, encoding="utf-8")
    with LEDGER.open("a", encoding="utf-8") as sink:
        sink.write("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", choices=sorted(PROMPTS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="propose and print, change nothing")
    parser.add_argument("--cases", type=int, default=6, help="how many worst questions to show the proposer")
    parser.add_argument("--noise", type=float, required=False, default=None,
                        help="acceptance threshold; REQUIRED in practice — pass the MEASURED band")
    parser.add_argument("--proposer-model", default=os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4.1-flash"))
    parser.add_argument("--proposer-base", default=os.environ.get("RAGAS_JUDGE_BASE", "https://llm.goaichat.top/v1"))
    args = parser.parse_args()

    # .env: the proposer key lives there (never in the repo).
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"'))

    if args.list:
        for name, spec in sorted(PROMPTS.items()):
            try:
                _, _, current = read_prompt(name)
                size = f"{len(current):5d} chars"
            except SystemExit as exc:
                size = str(exc)
            print(f"  {name:28s} {size}  {spec['why'][:56]}")
        print("\nacceptance: measured noise band, refusal must not fall, no new violations")
        print("revert: git checkout (not a text undo)")
        return 0

    if not args.prompt:
        parser.print_help()
        return 1
    if args.noise is None:
        print("refusing to run without --noise: pass the MEASURED run-to-run band, not a guess")
        return 1
    return run_once(args.prompt, args, args.cases)


if __name__ == "__main__":
    sys.exit(main())
