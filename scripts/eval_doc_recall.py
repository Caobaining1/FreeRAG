#!/usr/bin/env python3
"""Document-level recall: the metric that matches the fs channel.

`eval_tree_recall.py` measures passages, because the channels it compares return
passages. This one measures DOCUMENTS, because that is what `fs.search` returns:
a question's evidence lives in two or three PDFs, and the channel's job is to
name them. Two numbers, both cut at "how many documents you may keep":

    doc recall     the share of a question's evidence PDFs that are in the list
    fact recall    the share of its evidence sentences that those documents hold

The second is the honest consequence of the first: naming the right document is
only worth something if the sentence is in what you then read.

    python3 scripts/eval_doc_recall.py -k 5
    python3 scripts/eval_doc_recall.py --split eval/test.json -k 8

Channels:

    keyword-doc   the store's BM25, collapsed to documents (best passage wins)
    tree-doc      section-tree search, collapsed the same way
    fs            the directory tree, routed by Laya
    fs-keyword    the same tree, routed by keyword score (no model) — the
                  ablation that separates "the tree is wrong" from "the model is"
    fs-pure       the route with no keyword safety net
    fs-candN      the route with the options narrowed to the N strongest
                  children by keyword score before Laya is asked
    fs-candN-pure ... and without the safety net
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ragas_eval import Kernel  # noqa: E402
from eval_tree_recall import basename, fact_found, norm  # noqa: E402

REPO = HERE.parent


def run_channel(kernel: Kernel, channel: str, query: str, kb: str, k: int):
    """One retrieval, as a ranked list of document ids."""
    started = time.time()
    if channel == "keyword-doc":
        result = kernel.call("search", {"query": query, "limit": 200, "kb": kb})
        docs = docs_of_chunks(result.get("hits") or [])
    elif channel == "tree-doc":
        result = kernel.call("tree.search", {"query": query, "limit": 200, "kb": kb})
        docs = docs_of_chunks(result.get("hits") or [])
    elif channel == "fs" or channel.startswith("fs-"):
        result = kernel.call("fs.search", fs_arguments(channel, query, kb, k))
        docs = [hit["doc_id"] for hit in (result.get("hits") or [])]
    else:
        raise ValueError(channel)
    return docs, time.time() - started


FS_PARAMS = {"cand": "candidate_k", "br": "branches", "slots": "slots",
             "beam": "beam"}


def fs_arguments(channel: str, query: str, kb: str, k: int) -> Dict[str, Any]:
    """fs.search arguments for one channel name.

    fs, fs-keyword, fs-pure, fs-keyword-pure, and any combination with a
    sized knob: fs-cand3 (narrow the options), fs-br4 (open four leaves),
    fs-slots3 (split into three sub-questions).

    -kw (no model) and -pure (no safety net) are the two ablations every one of
    them needs. -kw separates "the mechanism works" from "the model works";
    -pure separates it from "the keyword net is carrying it". A number without
    either is measuring all three at once and can be read as evidence for any.
    """
    arguments: Dict[str, Any] = {"query": query, "limit": k, "kb": kb}
    if channel == "fs":
        return arguments
    for token in channel[len("fs-"):].split("-"):
        if not token:
            continue
        if token in ("keyword", "kw"):
            arguments["no_laya"] = True
        elif token == "pure":
            arguments["no_fallback"] = True
        elif token == "explain":
            # Output only: attaches the terms behind every score. It must not
            # change which documents come back — see the check in RUN.md §4.2.
            arguments["explain"] = True
        else:
            for prefix, param in FS_PARAMS.items():
                if token.startswith(prefix) and token[len(prefix):].isdigit():
                    arguments[param] = int(token[len(prefix):])
                    break
            else:
                raise ValueError("unknown fs channel: %s" % channel)
    return arguments


def docs_of_chunks(hits: List[Dict[str, Any]]) -> List[str]:
    """Collapse passage hits to the documents they came from, in rank order."""
    out: List[str] = []
    for hit in hits:
        doc = ((hit.get("chunk") or {}).get("doc_id")) or hit.get("doc_id")
        if doc and doc not in out:
            out.append(doc)
    return out


class DocText:
    """One document's normalised text, fetched once and kept.

    Reading a document is not retrieval — it is what happens AFTER a channel has
    named one — so it is cached rather than counted against any channel.
    """

    def __init__(self, kernel: Kernel, kb: str):
        self.kernel = kernel
        self.kb = kb
        self.cache: Dict[str, str] = {}

    def of(self, doc_id: str) -> str:
        if doc_id not in self.cache:
            result = self.kernel.call("chunks", {"kb": self.kb, "doc_id": doc_id,
                                                 "limit": 1000})
            self.cache[doc_id] = " ".join(
                norm(chunk.get("text") or "") for chunk in (result.get("chunks") or []))
        return self.cache[doc_id]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="eval/dev.json")
    parser.add_argument("--kb", default="eval/data-v12")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--cuts", default="3,5,8")
    parser.add_argument("--channels", default="keyword-doc,tree-doc,fs,fs-keyword,fs-pure")
    parser.add_argument("--build", action="store_true", help="run fs.index first")
    parser.add_argument("--out", default="eval/runs/doc-recall.json")
    args = parser.parse_args()

    rows = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    questions = [r for r in rows if r.get("evidence_pdfs")]
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    kb_dir = REPO / args.kb
    kb_id = (kb_dir / "kb_id.txt").read_text(encoding="utf-8").strip() \
        if (kb_dir / "kb_id.txt").exists() else ""

    print("%s: %d question(s) with evidence PDFs" % (args.split, len(questions)))

    kernel = Kernel(kb_dir)
    texts = DocText(kernel, kb_id)
    try:
        if args.build:
            started = time.time()
            built = kernel.call("fs.index", {"kb": kb_id} if kb_id else None)
            print("fs.index: %s document(s) -> %s node(s), from=%s, %.1fs" %
                  (built.get("documents"), built.get("nodes"), built.get("from"),
                   time.time() - started))
        status = kernel.call("fs.status", {"kb": kb_id} if kb_id else None)
        print("fs.status: %s doc(s), %s node(s), %s leaf/leaves, depth %s, from=%s, "
              "decider=%s" % (status.get("documents"), status.get("nodes"),
                              status.get("leaves"), status.get("depth"),
                              status.get("from"), status.get("decider")))

        cuts = sorted({c for c in (int(x) for x in args.cuts.split(",") if x.strip())
                       if 0 < c <= args.k}) or [args.k]
        per_channel: Dict[str, List[Dict[str, Any]]] = {c: [] for c in channels}

        for i, row in enumerate(questions, 1):
            gold = {basename(p) for p in row["evidence_pdfs"]}
            facts = row.get("reference_contexts") or []
            print("  [%d/%d] %s %s (%d evidence pdfs)" %
                  (i, len(questions), row["id"], row["type"], len(gold)))
            for channel in channels:
                docs, seconds = run_channel(kernel, channel, row["question"], kb_id, args.k)
                names = [basename(d) for d in docs]
                entry = {"id": row["id"], "type": row["type"], "gold": len(gold),
                         "docs": names, "seconds": round(seconds, 2),
                         "doc_recall_at": {}, "fact_recall_at": {}}
                for cut in cuts:
                    kept = names[:cut]
                    hit = len(gold & set(kept))
                    entry["doc_recall_at"][str(cut)] = hit / len(gold) if gold else None
                    if facts:
                        joined = " ".join(texts.of(d) for d in docs[:cut])
                        found = sum(1 for f in facts if fact_found(f, [joined]))
                        entry["fact_recall_at"][str(cut)] = found / len(facts)
                per_channel[channel].append(entry)
                print("      %-12s %s | %s" % (
                    channel,
                    "  ".join("d@%d %.2f" % (c, entry["doc_recall_at"][str(c)])
                              for c in cuts),
                    "  ".join("f@%d %.2f" % (c, entry["fact_recall_at"][str(c)])
                              for c in cuts if str(c) in entry["fact_recall_at"])))
    finally:
        kernel.close()

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.fmean(values), 4) if values else None

    summary = {}
    for cut in cuts:
        key = str(cut)
        print("\n@%d docs   %-12s %-10s %-10s %-9s %-8s" %
              (cut, "channel", "doc", "fact", "median_s", "full miss"))
        print("-" * 62)
        for channel in channels:
            rows_c = per_channel[channel]
            by_type = collections.defaultdict(list)
            for r in rows_c:
                by_type[r["type"]].append(r["doc_recall_at"][key])
            summary.setdefault(channel, {})[key] = {
                "doc_recall": mean([r["doc_recall_at"][key] for r in rows_c]),
                "fact_recall": mean([r["fact_recall_at"].get(key) for r in rows_c]),
                "doc_recall_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
                "questions_missing_all": sum(1 for r in rows_c
                                             if r["doc_recall_at"][key] == 0),
                "median_seconds": round(statistics.median(
                    [r["seconds"] for r in rows_c]), 3),
                # A question is fully answered only when EVERY evidence document
                # is named: multi-hop questions need all of them, and a channel
                # that finds one of two has not answered.
                "questions_complete": sum(1 for r in rows_c
                                          if r["doc_recall_at"][key] == 1),
            }
            s = summary[channel][key]
            print("           %-12s %-10s %-10s %-9s %-8s" % (
                channel, s["doc_recall"], s["fact_recall"], s["median_seconds"],
                s["questions_missing_all"]))
        print("-" * 62)

    for channel in channels:
        for cut in cuts:
            s = summary[channel][str(cut)]
            print("  %-12s @%-2d complete=%d  by type: %s" % (
                channel, cut, s["questions_complete"],
                ", ".join("%s %.2f" % (t, v) for t, v in s["doc_recall_by_type"].items())))

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "kb": str(kb_dir), "n_questions": len(questions),
        "channels": summary, "per_question": per_channel,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwritten: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
