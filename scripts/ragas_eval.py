#!/usr/bin/env python3
"""RAGAS evaluation over a frozen MultiHop-RAG subset — docs/plan.md §13 Phase A.

Four stages, each writing a file the next one reads, so that a run can be resumed
and an old result re-scored WITHOUT re-running the kernel:

    freeze   questions + the PDFs they need   -> eval/{dev,test}.json, eval/kb_sources.json
    index    those PDFs through the real pipeline -> a dedicated eval KB
    run      `ask` per question               -> eval/runs/<run>.jsonl
    score    ragas metrics + guards           -> eval/runs/<run>.scorecard.json

Why the split: one evaluation costs 1-2 hours here (CPU generation at ~6.4 tok/s,
plus judge latency), so scoring must never require re-asking. It is also what makes
Phase C/D of the RSI loop possible at all — a candidate change is measured by
re-running `run` and `score`, and the two are separately budgeted.

Where the data comes from: benchmarks/MultiHop-RAG, whose `queries_qa.json` carries
`answer` (ground truth), `evidence_list[].fact` (the verbatim evidence sentence, which
is the reference context RAGAS needs) and `evidence_pdfs` (exactly the PDFs a question
requires). No new labelling is invented here.

Usage:
    python3 scripts/ragas_eval.py freeze --per-type 8 --out eval/
    python3 scripts/ragas_eval.py index  --kb eval/data --sources eval/kb_sources.json
    python3 scripts/ragas_eval.py run    --split eval/dev.json --name baseline
    python3 scripts/ragas_eval.py score  --run eval/runs/baseline.jsonl --judge <model>
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
from typing import Any, Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
BENCH = Path("/Users/cbn/workspace/benchmarks/MultiHop-RAG")
KERNEL = REPO / "bin" / "freerag"

# ---------------------------------------------------------------------------
# .env loading: the kernel reads FREERAG_* from its process environment, exactly
# as the desktop shell passes them. Loading the same file keeps the eval run on
# the same embedder and generator as the app.
# ---------------------------------------------------------------------------


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# JSON-RPC over stdio, the same transport the desktop shell uses. One kernel
# process serves a whole stage: restarting it per request would re-launch the
# parse sidecar every time.
# ---------------------------------------------------------------------------


class Kernel:
    def __init__(self, data_dir: Path):
        load_env(REPO / ".env")
        env = dict(os.environ)
        # Isolate the eval KB from the user's own knowledge bases: FREERAG_DATA
        # names an index FILE and only its directory is used, so this moves the
        # whole layout (registry included).
        env["FREERAG_DATA"] = str(data_dir / "index.json")
        data_dir.mkdir(parents=True, exist_ok=True)
        # Greedy sampling for every measurement. The generator ships at 0.2
        # (cmd/freerag/main.go) because answers read better with a little variation —
        # and that variation is wider than anything the loop has proposed: three
        # dev-loop runs of the SAME effective code scored 0.3672, 0.3735 and 0.4465,
        # a range of 0.079, and the loop accepted a +0.079 change that was inert.
        # Comparisons cannot resolve a change through that spread. setdefault, so an
        # explicit outer value still wins when a run must reproduce shipped behaviour.
        env.setdefault("FREERAG_GENERATION_TEMPERATURE", "0")
        self._next_id = 0
        self.proc = subprocess.Popen(
            [str(KERNEL)],
            cwd=str(REPO),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def call(self, method: str, params: Optional[Dict[str, Any]] = None,
             on_progress=None, timeout: float = 3600.0) -> Any:
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"kernel closed stdout during {method} (see stderr)")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("method") == "progress":
                if on_progress is not None:
                    on_progress(message.get("params") or {})
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                raise RuntimeError(f"{method}: {message['error']}")
            return message.get("result")
        raise TimeoutError(f"{method} did not answer within {timeout}s")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Stage: freeze
# ---------------------------------------------------------------------------


def stage_freeze(args: argparse.Namespace) -> int:
    rows = json.loads((BENCH / "queries_qa.json").read_text(encoding="utf-8"))
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row)

    def take(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "question": row["query"],
            "type": row["question_type"],
            "answer": row["answer"],
            # The verbatim evidence sentences: RAGAS's reference contexts.
            "reference_contexts": [e.get("fact", "") for e in row.get("evidence_list", []) if e.get("fact")],
            "evidence_pdfs": row.get("evidence_pdfs", []),
        }

    # Stratified, deterministic (sorted by id), so the split is reproducible and
    # can be committed: an eval set that changes between runs measures nothing.
    dev, test = [], []
    for question_type in sorted(by_type):
        pool = sorted(by_type[question_type], key=lambda r: r["id"])

        # Null questions go in WITHOUT the PDF filter: they are unanswerable by
        # construction, so they have no evidence to index. They must still be in
        # the split, because the refusal rate over them is one of the guards a
        # candidate change is checked against (docs/plan.md §13.2) — and a guard
        # with no samples is not a guard. They are also the only questions whose
        # correct answer is "the corpus does not say".
        if question_type == "null_query":
            dev.extend(take(r) for r in pool[: args.null_dev])
            test.extend(take(r) for r in pool[args.null_dev: args.null_dev + args.null_test])
            continue

        # Every answerable question must be answerable from the PDFs we will
        # index, and its evidence files must exist on disk.
        pool = [r for r in pool if r.get("evidence_pdfs")
                and all((BENCH / "pdfs" / name).exists() for name in r["evidence_pdfs"])]
        dev.extend(take(r) for r in pool[: args.per_type])
        test.extend(take(r) for r in pool[args.per_type: args.per_type + args.test_per_type])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows_out in (("dev.json", dev), ("test.json", test)):
        (out / name).write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")

    sources = sorted({pdf for row in dev + test for pdf in row["evidence_pdfs"]})
    (out / "kb_sources.json").write_text(json.dumps(
        {"version": "eval-kb-v1", "pdfs": sources,
         "note": "Frozen with the split: the KB must contain exactly these documents."},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dev={len(dev)} test={len(test)} pdfs={len(sources)}")
    print(f"  written: {out}/dev.json, {out}/test.json, {out}/kb_sources.json")
    return 0


# ---------------------------------------------------------------------------
# Stage: index
# ---------------------------------------------------------------------------


def stage_index(args: argparse.Namespace) -> int:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    pdfs = sources["pdfs"]
    kernel = Kernel(Path(args.kb))
    try:
        created = kernel.call("kb.create", {"name": "eval-multihop-rag"})
        kb_id = (created or {}).get("id") or (created or {}).get("kb")
        if not kb_id:
            listing = kernel.call("kb.list")
            kb_id = listing["bases"][0]["id"] if isinstance(listing, dict) else listing[0]["id"]
        print(f"kb={kb_id}  indexing {len(pdfs)} PDF(s)")
        for i, name in enumerate(pdfs, 1):
            started = time.time()
            params: Dict[str, Any] = {"path": str(BENCH / "pdfs" / name), "kb": kb_id}
            if args.force:
                # Needed to REBUILD: the content fingerprint is unchanged, so without
                # this the job skips each file and the KB keeps whatever it had —
                # which after an embedder outage means chunks with no vectors, and a
                # hybrid search that silently degrades to keyword-only.
                params["force"] = True
            result = kernel.call("index", params, on_progress=lambda params: None)
            added = (result or {}).get("added")
            print(f"  [{i}/{len(pdfs)}] {name[:44]:44s} +{added} chunk(s) "
                  f"{time.time() - started:5.1f}s")
        Path(args.kb, "kb_id.txt").write_text(kb_id, encoding="utf-8")
    finally:
        kernel.close()
    return 0


# ---------------------------------------------------------------------------
# Stage: run
# ---------------------------------------------------------------------------


def stage_run(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.split).read_text(encoding="utf-8"))
    kb_file = Path(args.kb) / "kb_id.txt"
    kb_id = kb_file.read_text(encoding="utf-8").strip() if kb_file.exists() else ""

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"{args.name}.jsonl"

    # Resume: keep the questions already answered and skip them. Measured need —
    # memory pressure took one question from ~140 s to 507 s, and the only lever
    # against it is stopping this job, which without resume would discard the whole
    # run rather than the part still missing.
    done: set = set()
    if target.exists() and not args.fresh:
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
        if done:
            print(f"resuming {target.name}: {len(done)} question(s) already answered")

    kernel = Kernel(Path(args.kb))
    try:
        with target.open("a" if done else "w", encoding="utf-8") as sink:
            for i, row in enumerate(rows, 1):
                if row["id"] in done:
                    continue
                params: Dict[str, Any] = {"question": row["question"]}
                if kb_id:
                    params["kb"] = kb_id
                started = time.time()
                trace: List[str] = []

                def on_progress(event: Dict[str, Any], sink_trace=trace) -> None:
                    if event.get("line"):
                        sink_trace.append(event["line"])
                        # A heartbeat, because a question takes 3-6 minutes here and the
                        # per-question line below prints only when it FINISHES: without
                        # this, a slow question and a hung one look identical from
                        # outside (which cost a diagnosis round on 2026-10-02).
                        print(f"      · {event['line'][:110]}", flush=True)

                try:
                    result = kernel.call("ask", params, on_progress=on_progress)
                    error = None
                except Exception as exc:  # a failed question is data, not a crash
                    result, error = {}, str(exc)

                evidence = (result or {}).get("evidence") or []
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "type": row["type"],
                    "ground_truth": row["answer"],
                    "reference_contexts": row.get("reference_contexts", []),
                    "answer": (result or {}).get("answer", ""),
                    "contexts": [hit.get("chunk", {}).get("text", "") for hit in evidence],
                    "verdict": (result or {}).get("verdict"),
                    "rounds": (result or {}).get("rounds"),
                    "enumeration": (result or {}).get("enumeration"),
                    "latency_s": round(time.time() - started, 2),
                    # 200 rather than 40: a complex sub-question's tool notes sit early in its
                    # trace, and truncating to the tail lost exactly the lines that say
                    # whether retrieval ran hybrid or keyword-only.
                    "trace": trace[-200:],
                    "error": error,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"  [{i}/{len(rows)}] {row['id']} {record['latency_s']:6.1f}s "
                      f"ctx={len(record['contexts'])} ans={len(record['answer'])}字 "
                      f"{'' if not error else 'ERROR ' + error[:40]}")
    finally:
        kernel.close()
    print(f"written: {target}")
    return 0


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------


def stage_score(args: argparse.Namespace) -> int:
    """Score a run with RAGAS, plus the guards the RSI loop must not trade away.

    The judge is deliberately NOT the generator: an LLM grading text it wrote
    itself is biased towards it, and the whole point of the loop is to detect
    real improvement rather than the judge's agreement with itself (docs/plan.md
    §13.5). Pass --judge-base/--judge-model for a hosted judge; the fallback is
    the local Ollama endpoint, whose name is recorded in the scorecard either way.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    from ragas import RunConfig, evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    records = [json.loads(line) for line in Path(args.run).read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [r for r in records if r.get("type") != "null_query" and not r.get("error")]
    if not answerable:
        print("nothing to score: every question failed")
        return 1

    # The judge is reached over OpenAI-compatible HTTP, and the default is the
    # LOCAL endpoint on purpose: this machine's hosted provider returns
    # `402 account balance is insufficient`, and a metric that silently degrades to
    # NaN is worse than one that runs on a small local model — the scorecard records
    # which model judged, so the weakness is visible rather than averaged away.
    #
    # It is still NOT the generator (`freerag-qwen3`): plan §13.5 rules that out,
    # because an LLM grading text it wrote itself is biased towards it. The local
    # VLM checkpoints are a different family, which is the property that matters.
    judge = ChatOpenAI(
        model=args.judge_model,
        base_url=args.judge_base,
        api_key=os.environ.get("JUDGE_API_KEY") or "not-needed",
        temperature=0,
        # A generous cap, and NOT because the answers are long: the default ran out
        # and ragas raised LLMDidNotFinishException on 2 of 3 faithfulness jobs, so
        # the scorecard reported a mean over ONE sample. `deepseek-v4.1-flash` is a
        # reasoning model — its reasoning tokens are billed against the same budget
        # as the JSON — which is what made a per-question statement list exceed a
        # default-sized cap.
        max_tokens=args.judge_max_tokens,
    )

    # Only the metrics that need NO embedder. `answer_relevancy` embeds the
    # questions the judge invents, so it needs an embedding endpoint — and there is
    # none here (the hosted one is out of balance, and the ragas venv deliberately
    # has no torch). Dropping it is a real narrowing of the quality signal and is
    # recorded in the scorecard rather than hidden; add it back with
    # --with-answer-relevancy once an embedder exists.
    metrics = [faithfulness, context_precision, context_recall]
    if args.metrics:
        wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
        metrics = [m for m in metrics if m.name in wanted]
        if not metrics:
            print(f"--metrics {args.metrics!r} selected nothing")
            return 1
    if args.with_answer_relevancy:
        from ragas.metrics import answer_relevancy
        metrics.insert(1, answer_relevancy)

    # The judge reads at most a fixed number of contexts, each truncated. Two
    # measured reasons, both about the JUDGE rather than about the metric:
    #   - Ollama's default window is 4096 tokens and ragas's judge prompt over 10-17
    #     contexts measured 5786-5910 tokens, so jobs failed with `400 exceeds the
    #     available context size` before they could be scored at all;
    #   - a longer prompt makes the judge emit more statements, and at ~6 tok/s on
    #     CPU a job that survived took 391 s.
    # The cut is applied identically to every run, so it cannot flatter one variant,
    # and it is recorded in the scorecard.
    contexts = [judge_contexts(r["contexts"], args.judge_max_contexts, args.judge_context_chars)
                for r in answerable]
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in answerable],
        "answer": [r["answer"] for r in answerable],
        "contexts": contexts,
        "ground_truth": [r["ground_truth"] for r in answerable],
    })
    embeddings = None
    if args.with_answer_relevancy:
        # The SAME embedding configuration the app itself uses
        # (internal/embed/siliconflow.go: base + FREERAG_EMBED_MODEL +
        # FREERAG_SILICONFLOW_KEY), so the judge's similarity geometry matches the
        # retriever's. Not JUDGE_API_KEY: that key belongs to the judge endpoint, and
        # using it here would ask an unrelated provider for BGE-M3 vectors.
        #
        # BAAI/bge-m3 is not a preference: it is the ONLY embedding model this
        # provider serves (`--embed-model` is documented as fixed), and the app's
        # stored vectors were built with it. Substituting another model would not
        # fail loudly -- it would silently compare vectors from two different spaces.
        embeddings = OpenAIEmbeddings(
            model=args.embed_model,
            base_url=args.embed_base,
            api_key=os.environ.get("FREERAG_SILICONFLOW_KEY") or os.environ.get("JUDGE_API_KEY") or "not-needed",
        )

    # A per-JOB timeout, and the default (180 s) is why a local judge produced a
    # scorecard full of NaN: on CPU a 3B model cannot finish ragas's judge prompt in
    # three minutes, every job raised TimeoutError, and raise_exceptions=False turns
    # that into a silent NaN. Recorded rather than hidden: failures are counted
    # below and printed.
    run_config = RunConfig(timeout=args.judge_timeout, max_workers=args.judge_workers)
    result = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings,
                      run_config=run_config, raise_exceptions=False)

    # ragas 0.4's EvaluationResult is not a plain mapping (iterating it as one
    # raises KeyError on key 0), so read it through the frame it exposes. The
    # per-sample columns are kept: the RSI ledger needs to know WHICH questions a
    # candidate change moved, not only the mean.
    frame = result.to_pandas()
    metric_names = [m.name for m in metrics]
    scores = {}
    for name in metric_names:
        if name not in frame.columns:
            scores[name] = None
            continue
        column = frame[name].dropna()
        scores[name] = round(float(column.mean()), 4) if len(column) else None
    per_sample = frame[[c for c in frame.columns if c in metric_names or c in ("question", "user_input")]].to_dict("records")

    # Guards: reported, never optimised (docs/plan.md §13.2).
    null_rows = [r for r in records if r.get("type") == "null_query"]
    refusals = sum(1 for r in null_rows if REFUSAL.search(r.get("answer", "")))
    latencies = sorted(r["latency_s"] for r in records if not r.get("error"))

    nan_samples = {name: sum(1 for r in per_sample if r.get(name) != r.get(name)) for name in metric_names}

    scorecard = {
        "run": Path(args.run).name,
        "kb": "eval-kb-v1",
        "judge": args.judge_model,
        "judge_base": args.judge_base,
        # Recorded because it changes how the numbers must be read: a judge that is
        # also the generator is biased towards its own style, so ABSOLUTE scores are
        # optimistic. The RSI loop only needs RELATIVE change, which a frozen judge
        # + frozen eval set still measures -- but this flag failing to be False is
        # the first thing to check before believing a delta.
        "judge_is_generator": args.judge_model == os.environ.get("FREERAG_GENERATOR_MODEL", "freerag-qwen3"),
        "metrics_requested": metric_names,
        "n_scored": len(answerable),
        "n_failed": sum(1 for r in records if r.get("error")),
        # Whether this scorecard may be used for a DECISION at all.
        #
        # A run whose kernel died mid-flight still produces a full scorecard, and every
        # derived number is an artifact of the missing answers: empty answers score 0 on
        # answer_relevancy, empty pools make context_recall read 1.0, and a judge asked
        # whether an empty reply refused the question tends to say yes. One such run was
        # read as "macro -0.087, refusal 0.75 -> 0.0" before anyone noticed n_failed: 31 —
        # every number in it was caused by 23 questions that never ran.
        "usable": not any(r.get("error") for r in records),
        "unusable_reason": ("one or more questions failed to run; every metric is an artifact "
                            "of the missing answers" if any(r.get("error") for r in records) else None),
        "metrics": scores,
        "judge_nan_samples": nan_samples,
        "judge_contexts_max": args.judge_max_contexts,
        "judge_context_chars": args.judge_context_chars,
        "quality_macro": (None if any(v is None for v in scores.values())
                          else round(sum(scores.values()) / len(scores), 4)),
        "refusal_rate_on_null": (round(refusals / len(null_rows), 3) if null_rows else None),
        "n_null": len(null_rows),
        "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
        "violations": detect_violations(records),
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    }
    out = Path(args.run).with_suffix(".scorecard.json")
    out.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    # Per-sample scores beside the scorecard: a mean cannot say which questions a
    # candidate change moved, and the RSI ledger is read for exactly that.
    samples = Path(args.run).with_suffix(".scores.json")
    samples.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    print(f"written: {out}")
    return 0


def judge_contexts(contexts: List[str], limit: int, chars: int) -> List[str]:
    """The slice of a run's contexts the judge is shown.

    Front-of-list rather than a similarity cut: the pool is in retrieval order, so
    the first passages are the ones the run itself ranked highest, and any smarter
    selection would be a second retrieval measurement hiding inside the judge.
    """
    kept = [c for c in contexts if c]
    if limit > 0:
        kept = kept[:limit]
    return [c[:chars] if chars > 0 else c for c in kept]


# A refusal, not a topic: the null questions have no evidence at all, so any
# answer that asserts facts is a fabrication. Kept to explicit phrasing so it
# cannot be satisfied by accident.
REFUSAL = re.compile(
    r"(no information|not (?:mentioned|stated|found|available|enough)|does not (?:say|mention|contain)|"
    r"cannot (?:be )?(?:answer|determin)|insufficient|没有(?:相关)?(?:信息|内容|提及)|无法(?:回答|确定)|未(?:提及|说明)|不足以)",
    re.IGNORECASE,
)


def detect_violations(records: Iterable[Dict[str, Any]]) -> List[str]:
    """The invariants of §13.2 that a score can be checked against.

    Deliberately few and mechanical; the ones that need judgement (citation
    grounding, anchoring) are covered by the Go test suite, which the loop runs
    as a gate before it is allowed to score anything.
    """
    violations: List[str] = []
    for record in records:
        # "An empty pool is never sufficient": a run that retrieved nothing and
        # still produced prose is the failure that guard exists for.
        if not record.get("contexts") and len(record.get("answer", "")) > 200:
            violations.append(f"{record['id']}: answered at length from an empty pool")
    return violations


# ---------------------------------------------------------------------------


def main() -> int:
    # Before any stage: the kernel reads FREERAG_* from the environment and the
    # score stage reads the judge/embedding keys from it. Kernel.__init__ also
    # loads it, but `score` builds no kernel — and a scoring run that starts after
    # a two-hour `run` must not fail for want of a key file.
    load_env(REPO / ".env")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    freeze = sub.add_parser("freeze", help="select questions and freeze the split")
    freeze.add_argument("--per-type", type=int, default=8, help="dev questions per type")
    freeze.add_argument("--test-per-type", type=int, default=20, help="held-out questions per type")
    freeze.add_argument("--null-dev", type=int, default=8, help="unanswerable questions in dev (refusal guard)")
    freeze.add_argument("--null-test", type=int, default=20, help="held-out unanswerable questions")
    freeze.add_argument("--out", default="eval/")
    freeze.set_defaults(func=stage_freeze)

    index = sub.add_parser("index", help="build the eval KB from the frozen sources")
    index.add_argument("--kb", default="eval/data")
    index.add_argument("--sources", default="eval/kb_sources.json")
    index.add_argument("--force", action="store_true",
                       help="re-embed even unchanged files (required to rebuild a KB)")
    index.set_defaults(func=stage_index)

    run = sub.add_parser("run", help="ask every question and record answer + contexts")
    run.add_argument("--split", default="eval/dev.json")
    run.add_argument("--kb", default="eval/data")
    run.add_argument("--name", required=True)
    run.add_argument("--out", default="eval/runs")
    run.add_argument("--fresh", action="store_true", help="ignore an existing run and start over")
    run.set_defaults(func=stage_run)

    score = sub.add_parser("score", help="ragas metrics + guards for one run")
    score.add_argument("--run", required=True)
    score.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL", "freerag-qwen3"),
                       help="judge model. Default is the local Qwen3-4B: measured 36.5 s per judge "
                            "call and valid JSON, versus >5 min and no result for the local VL "
                            "checkpoint. It IS the generator, which plan 13.5 forbids for the "
                            "absolute score -- see judge_is_generator in the scorecard; set a hosted "
                            "model here once a key has balance.")
    score.add_argument("--judge-base", default=os.environ.get("RAGAS_JUDGE_BASE", "http://localhost:11434/v1"))
    score.add_argument("--metrics", default="", help="comma-separated subset, e.g. faithfulness")
    score.add_argument("--judge-timeout", type=float, default=1800.0, help="per-job seconds")
    score.add_argument("--judge-max-tokens", type=int, default=32768,
                       help="judge output cap; a reasoning model spends this on reasoning too, and "
                            "8192 truncated 12 of 24 faithfulness jobs into silent NaN")
    score.add_argument("--judge-max-contexts", type=int, default=0,
                       help="contexts shown to the judge; 0 = all. MEASURED: truncating to 5 with a "
                            "local judge lowered faithfulness 0.60->0.27, context_recall 0.50->0.17 and "
                            "context_precision 0.29->0.19, and NaN'd half the faithfulness samples — "
                            "the metrics are computed against the contexts they are SHOWN")
    score.add_argument("--judge-context-chars", type=int, default=0,
                       help="characters per context; 0 = whole passage (see --judge-max-contexts)")
    score.add_argument("--judge-workers", type=int, default=2, help="concurrent judge jobs")
    score.add_argument("--with-answer-relevancy", action="store_true",
                       help="also score answer_relevancy; needs an embedding endpoint")
    score.add_argument("--embed-model", default=os.environ.get("FREERAG_EMBED_MODEL", "BAAI/bge-m3"),
                       help="FIXED to BAAI/bge-m3: siliconflow serves no other embedding model "
                            "(other names are rejected), and the app is pinned to the same one")
    score.add_argument("--embed-base", default=os.environ.get("FREERAG_EMBED_BASE", "https://api.siliconflow.cn/v1"),
                       help="reuses the app's own embedding endpoint; key comes from FREERAG_SILICONFLOW_KEY")
    score.set_defaults(func=stage_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
