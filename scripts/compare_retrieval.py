#!/usr/bin/env python3
"""Compare keyword-only and hybrid retrieval over the same index.

The ``search`` RPC is BM25-only; the ``hybrid_search`` tool fuses BM25 with
BGE-M3 vectors through reciprocal rank fusion. Running both over the same
queries shows what the dense half contributes — and, just as usefully, where it
changes nothing.

The probes are picked to separate the two rankers:

  * lexical overlap     — both should agree; a difference means fusion is noisy
  * paraphrase          — few shared terms, so BM25 is weak by construction
  * cross-lingual       — Chinese query, English corpus: BM25 cannot match at all
  * identifier          — a rare code; keyword search is strong here

Usage:
    python3 scripts/compare_retrieval.py <pdf> [--index FILE] [--build] [-k 5]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel_rpc import rpc  # noqa: E402

#: label -> query
PROBES = [
    ("lexical overlap", "GRPO value function advantage estimation"),
    ("paraphrase", "How does the survey categorise the different roles reinforcement learning plays?"),
    ("cross-lingual", "强化学习在智能搜索中有哪些作用？"),
    ("identifier", "DAPO"),
]


def ids_of(hits):
    return [hit["chunk"]["chunk_id"] for hit in hits]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--index", default="artifacts/e2e/compare-index.json")
    parser.add_argument("--build", action="store_true", help="(re)index the PDF first")
    parser.add_argument("--kernel", default="bin/freerag")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    kernel = os.path.abspath(args.kernel)
    index_path = os.path.abspath(args.index)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    version, _ = rpc(kernel, index_path, "version", quiet=False)
    embedding = version.get("embedding") or {}
    if not embedding.get("enabled"):
        print("dense retrieval is OFF; set FREERAG_EMBED_PROVIDER and a key", file=sys.stderr)
        return 2
    print("embedder: %s (%s dims)" % (embedding.get("provider"), embedding.get("dims")))

    if args.build or not os.path.isfile(index_path):
        result, seconds = rpc(kernel, index_path, "index", {"path": os.path.abspath(args.pdf)})
        print("indexed %s chunk(s), %s embedded, in %.1fs\n"
              % (result.get("indexed_total"), result.get("embedded"), seconds))

    print("%-16s %-9s %-9s %s" % ("probe", "bm25", "hybrid", "comparison"))
    print("-" * 92)

    total_only_keyword = 0
    total_only_hybrid = 0

    for label, query in PROBES:
        keyword, _ = rpc(kernel, index_path, "search", {"query": query, "limit": args.k})
        hybrid, _ = rpc(kernel, index_path, "tool",
                        {"name": "hybrid_search", "arguments": {"query": query, "k": args.k}})

        keyword_ids = ids_of(keyword.get("hits") or [])
        hybrid_hits = hybrid.get("hits") or []
        hybrid_ids = ids_of(hybrid_hits)

        only_keyword = [i for i in keyword_ids if i not in hybrid_ids]
        only_hybrid = [i for i in hybrid_ids if i not in keyword_ids]
        total_only_keyword += len(only_keyword)
        total_only_hybrid += len(only_hybrid)

        sources = {}
        for hit in hybrid_hits:
            for source in hit.get("sources") or []:
                sources[source] = sources.get(source, 0) + 1

        print("%-16s %-9d %-9d +%d from hybrid, +%d from bm25  %s"
              % (label, len(keyword_ids), len(hybrid_ids),
                 len(only_hybrid), len(only_keyword),
                 "sources=" + str(dict(sorted(sources.items())))))

    print("-" * 92)
    print("unique to hybrid: %d chunk(s); unique to bm25: %d chunk(s)"
          % (total_only_hybrid, total_only_keyword))

    # Show one cross-lingual result in full: BM25 should have nothing at all,
    # which is the clearest single demonstration that the dense half is load-bearing.
    label, query = PROBES[2]
    keyword, _ = rpc(kernel, index_path, "search", {"query": query, "limit": args.k})
    hybrid, _ = rpc(kernel, index_path, "tool",
                    {"name": "hybrid_search", "arguments": {"query": query, "k": args.k}})

    print("\n%s — %s" % (label, query))
    print("  bm25  : %d hit(s)" % len(keyword.get("hits") or []))
    for hit in (hybrid.get("hits") or [])[:3]:
        text = " ".join((hit["chunk"].get("text") or "").split())
        print("  hybrid: [%s] %.3f  p.%s  %s"
              % (",".join(hit.get("sources") or []), hit.get("score", 0),
                 hit["chunk"].get("page_num"), text[:110]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
