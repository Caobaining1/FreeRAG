#!/usr/bin/env python3
"""Is the failing evidence in the corpus at all? — docs/plan.md §13.19.

Phase D declined twice on two different prompts, both times saying the failures are
retrieval-side rather than prompt-side. That claim is checkable, and the two ways it can
be true call for opposite actions:

  * the evidence IS in the corpus but was not retrieved -> the retrieval strategy is the
    lever, and the loop should be changing that rather than prompts;
  * the evidence is NOT in the corpus -> no change to this system can answer those
    questions, and the fitness function is scoring the loop against something
    unreachable. Iterating on anything else is then spending time on a target that
    cannot move.

Method: for each question, take the ground truth's evidence sentences and check how many
of their word 5-grams occur (a) anywhere in the indexed chunks, and (b) in the passages
the run actually retrieved. 5-grams rather than whole sentences because the benchmark's
`fact` is a rewritten sentence, not a verbatim quote: requiring the whole sentence would
report "not in the corpus" for text that is plainly there.

Usage:
    python3 scripts/evidence_check.py --run eval/runs/baseline.jsonl \
        --scores eval/runs/baseline.scores.json --kb eval/data-hybrid --worst 6
"""
from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
GRAM = 5

# Below this share of the evidence's 5-grams, the run retrieved essentially none of what
# the question needs. Shared with rsi_propose.py, which uses it to keep retrieval-failure
# cases out of the proposer's input: a prompt cannot fix them, and showing them made three
# Phase D iterations decline instead of editing.
RETRIEVAL_FLOOR = 0.15


def normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text)).strip()


def grams(text: str, n: int = GRAM) -> List[str]:
    words = normalize(text).split()
    return [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))] or [normalize(text)]


class Corpus:
    """All indexed chunk texts, concatenated, with offsets back to each chunk."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.chunks: List[Dict] = data.get("chunks") or []
        parts, self.starts, self.text = [], [], []
        for chunk in self.chunks:
            parts.append(normalize(chunk.get("text") or ""))
        self.text = "\u0000".join(parts)
        self.starts = []
        offset = 0
        for part in parts:
            self.starts.append(offset)
            offset += len(part) + 1

    def find_chunk(self, needle: str) -> Optional[int]:
        at = self.text.find(needle)
        if at < 0:
            return None
        return max(0, bisect_right(self.starts, at) - 1)

    def coverage(self, evidence: Sequence[str]) -> Tuple[float, Optional[int], Optional[str]]:
        """Fraction of the evidence's 5-grams present anywhere, plus a matching chunk."""
        found, chunk_index = 0, None
        for sentence in evidence:
            for gram in grams(sentence):
                index = self.find_chunk(gram)
                if index is not None:
                    found += 1
                    chunk_index = chunk_index if chunk_index is not None else index
        total = sum(len(grams(s)) for s in evidence) or 1
        matched = ""
        if chunk_index is not None:
            matched = (self.chunks[chunk_index].get("text") or "")[:110].replace("\n", " ")
        return found / total, chunk_index, matched


def text_coverage(evidence: Sequence[str], passages: Sequence[str]) -> float:
    hay = "\u0000".join(normalize(p) for p in passages)
    found = total = 0
    for sentence in evidence:
        for gram in grams(sentence):
            total += 1
            if gram in hay:
                found += 1
    return found / (total or 1)


def worst_cases(run: Path, scores: Path, limit: int) -> List[Dict]:
    records = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = json.loads(scores.read_text(encoding="utf-8"))
    answerable = [r for r in records if r.get("type") != "null_query" and not r.get("error")]
    if len(answerable) != len(table):
        raise SystemExit(f"{run.name}: {len(answerable)} answerable vs {len(table)} score rows")
    pairs = []
    for record, row in zip(answerable, table):
        values = [row.get(m) for m in METRICS if isinstance(row.get(m), float)]
        pairs.append((sum(values) / len(values) if values else 0.0, record, row))
    pairs.sort(key=lambda p: p[0])
    return [{"record": r, "score": s} for s, r, _ in pairs[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="eval/runs/baseline.jsonl")
    parser.add_argument("--scores", default="eval/runs/baseline.scores.json")
    parser.add_argument("--kb", default="eval/data-hybrid")
    parser.add_argument("--worst", type=int, default=6)
    parser.add_argument("--show", type=int, default=2, help="how many cases to print in full")
    args = parser.parse_args()

    index_files = list((REPO / args.kb / "kbs").glob("*/index.json"))
    if not index_files:
        raise SystemExit(f"no index under {args.kb}/kbs/*/index.json")
    corpus = Corpus(index_files[0])
    print(f"corpus: {len(corpus.chunks)} chunk(s) from {index_files[0].name}\n")

    rows = []
    for i, case in enumerate(worst_cases(REPO / args.run, REPO / args.scores, args.worst)):
        record = case["record"]
        evidence = [str(x) for x in (record.get("reference_contexts") or [])]
        retrieved = [str(x) for x in (record.get("contexts") or [])]
        in_corpus, chunk_index, sample = corpus.coverage(evidence)
        in_retrieved = text_coverage(evidence, retrieved)
        rows.append((i, case["score"], in_corpus, in_retrieved, record))
        if i < args.show:
            print(f"[{i + 1}] score={case['score']:.3f}  {record['question'][:110]}")
            print(f"    evidence: {evidence[0][:150] if evidence else '(none)'}")
            print(f"    in corpus {in_corpus:.0%}"
                  f"{'  e.g. ' + sample if sample else '  (no 5-gram found anywhere)'}")
            print(f"    in retrieved {in_retrieved:.0%} over {len(retrieved)} passage(s)")
            print(f"    retrieved sample: {(retrieved[0][:110].replace(chr(10), ' ') if retrieved else '(none)')}")
            print()

    print(f"{'#':>3} {'score':>6} {'in corpus':>10} {'in retrieved':>13}  verdict")
    for i, score, in_corpus, in_retrieved, _ in rows:
        if in_corpus < RETRIEVAL_FLOOR:
            verdict = "EVIDENCE ABSENT from the corpus — this system cannot answer it"
        elif in_retrieved < RETRIEVAL_FLOOR:
            verdict = "in the corpus, not retrieved — the retrieval strategy is the lever"
        else:
            verdict = "evidence retrieved — a generation/prompt problem"
        print(f"{i + 1:>3} {score:>6.3f} {in_corpus:>10.0%} {in_retrieved:>13.0%}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
