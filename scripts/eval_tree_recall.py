#!/usr/bin/env python3
"""Evidence recall of the retrieval channels, on the frozen MultiHop-RAG eval set.

`ragas_eval.py run` measures recall end to end: it asks the generator, and RAGAS's
`context_recall` then judges whether the retrieved contexts support the reference
facts. That is the number that matters for answers, but it costs a generator run
per question and it cannot say which CHANNEL lost the evidence — the agent loop
retrieves several times per question, and the tree channel is not even wired into
`ask`.

This script measures the half that is cheap to isolate: given a question and the
verbatim evidence sentences MultiHop-RAG labels for it, does channel X surface
that text? Same KB, same questions, same reference contexts as the RAGAS score;
no generator, no judge, so a sweep costs seconds instead of hours.

    python3 scripts/eval_tree_recall.py --build-tree -k 10
    python3 scripts/eval_tree_recall.py --split eval/test.json -k 20

Channels:

    keyword      `search`                — BM25 only, the pre-dense baseline
    hybrid       `hybrid_search`         — BM25 fused with BGE-M3 (what `ask` uses)
    tree         `tree.search`           — structure tree + BM25 safety net
    tree-only    `tree.search`           — no_fallback: what the tree earns alone

Matching is lexical on purpose: the reference facts are verbatim sentences from
the PDFs, so "the sentence is in the passage" is decidable without a judge. A
sentence split across a chunk boundary is still counted, because the check also
runs against the passages of one document joined in retrieval order.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ragas_eval import Kernel  # noqa: E402  (env + stdio JSON-RPC, one process)

REPO = HERE.parent

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "has", "was",
    "were", "are", "not", "but", "its", "their", "they", "which", "while",
    "into", "than", "then", "over", "about", "after", "before", "been", "other",
}

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    """Lowercase, one space per gap, punctuation dropped — the space in which a
    verbatim fact is looked for inside a passage."""
    return " ".join(WORD.findall((text or "").lower()))


def content_tokens(text: str) -> set:
    return {w for w in norm(text).split() if len(w) >= 4 and w not in STOPWORDS}


def fact_found(fact: str, haystacks: List[str]) -> bool:
    """True when one passage carries the fact.

    Exact containment first — these facts are verbatim — then a token coverage
    fallback for the sentence that a chunk boundary cut in half, which is a real
    case here and not a leniency: every channel is measured the same way.
    """
    needle = norm(fact)
    if not needle:
        return False
    for hay in haystacks:
        if needle in hay:
            return True
    tokens = content_tokens(fact)
    if not tokens:
        return False
    for hay in haystacks:
        if len(tokens & set(hay.split())) / len(tokens) >= 0.8:
            return True
    return False


def passages_of(hits: List[Dict[str, Any]]) -> List[str]:
    """Normalised text per hit, plus one joined text per document.

    The joined form is what lets a fact survive a chunk boundary: a passage is
    evidence whether or not the chunker put both halves in the same block.
    """
    per_doc: Dict[str, List[str]] = {}
    single: List[str] = []
    for hit in hits:
        chunk = hit.get("chunk") or {}
        text = norm(chunk.get("text") or "")
        if not text:
            continue
        single.append(text)
        per_doc.setdefault(str(chunk.get("doc_id") or ""), []).append(text)
    joined = [" ".join(parts) for parts in per_doc.values()]
    return single + joined


def basename(doc_id: str) -> str:
    return os.path.basename(str(doc_id or ""))


def run_channel(kernel: Kernel, channel: str, query: str, kb: str, k: int):
    """One retrieval, returned as (hits, seconds)."""
    started = time.time()
    if channel == "keyword":
        result = kernel.call("search", {"query": query, "limit": k, "kb": kb})
        hits = result.get("hits") or []
    elif channel == "hybrid":
        result = kernel.call("tool", {"name": "hybrid_search",
                                      "arguments": {"query": query, "k": k}, "kb": kb})
        hits = result.get("hits") or []
    elif channel == "tree":
        result = kernel.call("tree.search", {"query": query, "limit": k, "kb": kb})
        hits = result.get("hits") or []
    elif channel == "tree-only":
        result = kernel.call("tree.search", {"query": query, "limit": k, "kb": kb,
                                             "no_fallback": True})
        hits = result.get("hits") or []
    elif channel in ("tree-wide", "tree-wide-only"):
        # The same channel with the routing budget opened up: MaxDocs is how many
        # documents the router may descend into and Beam how many siblings survive
        # a level. The default (8/2) is sized for a desktop corpus; a multi-hop
        # question needs two or three documents in the same top-k, so this is the
        # variant that says whether a loss is a budget or a ranking.
        result = kernel.call("tree.search", {"query": query, "limit": k, "kb": kb,
                                             "max_docs": 20, "beam": 4,
                                             "no_fallback": channel.endswith("-only")})
        hits = result.get("hits") or []
    else:
        raise ValueError(channel)
    return hits, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="eval/dev.json")
    parser.add_argument("--kb", default="eval/data")
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--cuts", default="5,10,20",
                        help="recall is reported at each cut <= k, from one retrieval per channel")
    parser.add_argument("--channels", default="keyword,hybrid,tree,tree-only")
    parser.add_argument("--build-tree", action="store_true",
                        help="run tree.index first (needed once: eval/data predates the channel)")
    parser.add_argument("--out", default="eval/runs/tree-recall.json")
    args = parser.parse_args()

    rows = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    questions = [r for r in rows if r.get("reference_contexts")]
    skipped = len(rows) - len(questions)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    kb_dir = REPO / args.kb
    kb_id = ""
    kb_file = kb_dir / "kb_id.txt"
    if kb_file.exists():
        kb_id = kb_file.read_text(encoding="utf-8").strip()

    print("%s: %d question(s) with evidence (%d skipped: unanswerable)" %
          (args.split, len(questions), skipped))

    kernel = Kernel(kb_dir)
    try:
        version = kernel.call("version")
        embedding = version.get("embedding") or {}
        print("embedder: %s (%s dims) — %s" % (
            embedding.get("provider") or "none", embedding.get("dims"),
            "on" if embedding.get("enabled") else "OFF, hybrid is keyword-only"))

        if args.build_tree:
            built, seconds = None, time.time()
            built = kernel.call("tree.index", {"kb": kb_id} if kb_id else None)
            print("tree.index: %s doc(s) / %s chunk(s) in %.1fs  levels=%s" %
                  (built.get("built"), built.get("chunks"), time.time() - seconds,
                   built.get("levels")))

        status = kernel.call("tree.status", {"kb": kb_id} if kb_id else None)
        print("tree.status: %s document(s), levels_from=%s" %
              (status.get("documents"), status.get("levels")))

        cuts = sorted({c for c in (int(x) for x in args.cuts.split(",") if x.strip())
                       if 0 < c <= args.k})
        if not cuts:
            cuts = [args.k]

        per_channel: Dict[str, List[Dict[str, Any]]] = {c: [] for c in channels}
        for i, row in enumerate(questions, 1):
            gold_docs = {basename(p) for p in (row.get("evidence_pdfs") or [])}
            print("  [%d/%d] %s %s" % (i, len(questions), row["id"], row["type"]))
            for channel in channels:
                hits, seconds = run_channel(kernel, channel, row["question"], kb_id, args.k)
                facts = row["reference_contexts"]
                # One retrieval, several cut points: the pool is in rank order, so
                # recall@k is recall over its first k entries. Re-running per k would
                # cost the same answers several times over.
                found_at = {cut: sum(1 for f in facts
                                     if fact_found(f, passages_of(hits[:cut])))
                            for cut in cuts}
                retrieved_docs = {basename((hit.get("chunk") or {}).get("doc_id"))
                                  for hit in hits}
                gold_hit = len(gold_docs & retrieved_docs)
                sources = collections.Counter(str(hit.get("source") or "?")
                                              for hit in hits)
                per_channel[channel].append({
                    "id": row["id"],
                    "type": row["type"],
                    "facts": len(facts),
                    "facts_found_at": {str(c): v for c, v in found_at.items()},
                    "recall_at": {str(c): (v / len(facts) if facts else None)
                                  for c, v in found_at.items()},
                    "hits": len(hits),
                    "gold_docs": len(gold_docs),
                    "gold_docs_hit": gold_hit,
                    # Purity: the share of the pool that came from a gold document.
                    # Recall alone can be bought by returning more text, and this is
                    # the number that says whether it was.
                    "purity": (len([d for d in retrieved_docs if d in gold_docs]) /
                               len(retrieved_docs)) if retrieved_docs else 0.0,
                    # Which channel each hit came from — the only way to see a tree
                    # hit displacing a BM25 hit inside the same top-k.
                    "sources": dict(sources),
                    "seconds": round(seconds, 2),
                })
                print("      %-10s %s  gold docs %d/%d  purity %.2f  %d hits  %.2fs" % (
                    channel,
                    "  ".join("r@%d %.2f" % (c, found_at[c] / max(1, len(facts)))
                              for c in cuts),
                    gold_hit, len(gold_docs), per_channel[channel][-1]["purity"],
                    len(hits), seconds))
    finally:
        kernel.close()

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.fmean(values), 4) if values else None

    summary = {}
    for cut in cuts:
        key = str(cut)
        print("\nrecall@%d  %-10s %-9s %-9s %-9s %-8s %-8s" %
              (cut, "channel", "macro", "micro", "golddoc", "purity", "median_s"))
        print("-" * 68)
        for channel in channels:
            rows_c = per_channel[channel]
            recalls = [(r["recall_at"][key], r["type"]) for r in rows_c]
            micro = sum(r["facts_found_at"][key] for r in rows_c) / max(
                1, sum(r["facts"] for r in rows_c))
            by_type = collections.defaultdict(list)
            for value, kind in recalls:
                by_type[kind].append(value)
            summary.setdefault(channel, {})[key] = {
                "n": len(rows_c),
                "macro_recall": mean([v for v, _ in recalls]),
                "micro_recall": round(micro, 4),
                "macro_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
                "gold_doc_recall": mean([r["gold_docs_hit"] / r["gold_docs"]
                                         for r in rows_c if r["gold_docs"]]),
                "purity": mean([r["purity"] for r in rows_c]),
                "median_seconds": round(statistics.median([r["seconds"] for r in rows_c]), 3),
                "mean_hits": round(statistics.fmean([r["hits"] for r in rows_c]), 2),
                "questions_at_zero": sum(1 for r in rows_c if r["facts_found_at"][key] == 0),
                "sources": dict(sum((collections.Counter(r["sources"])
                                     for r in rows_c), collections.Counter())),
            }
            s = summary[channel][key]
            print("           %-10s %-9s %-9s %-9s %-8s %-8s" % (
                channel, s["macro_recall"], s["micro_recall"], s["gold_doc_recall"],
                s["purity"], s["median_seconds"]))
        print("-" * 68)

    for channel in channels:
        for cut in cuts:
            s = summary[channel][str(cut)]
            print("  %-10s @%-3d by type: %s   zero: %d   sources: %s" % (
                channel, cut,
                ", ".join("%s %.2f" % (t, v) for t, v in s["macro_by_type"].items()),
                s["questions_at_zero"], s["sources"]))

    payload = {
        "split": args.split,
        "kb": str(kb_dir),
        "k": args.k,
        "n_questions": len(questions),
        "channels": summary,
        "per_question": per_channel,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwritten: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
