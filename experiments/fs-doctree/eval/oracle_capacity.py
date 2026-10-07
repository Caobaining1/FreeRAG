#!/usr/bin/env python3
"""这棵树到底能装下多少答案——不看路由，只看结构。

它回答的是 go / no-go 问题：召回差，是树的簇切错了（换建树算法能救），
还是下降机制本身的天花板就在那（换算法也救不了）。

三个数字：

    in-some-leaf   证据文档是否至少存在于某个叶子里（树有没有把语料切丢）
    oracle-1       路由完美但只走一条路径（一个叶子）时的上限
    oracle-N       路由完美且最多同时进 max_leaves 个叶子时的上限

    python3 experiments/fs-doctree/eval/oracle_capacity.py
    python3 experiments/fs-doctree/eval/oracle_capacity.py --budget 8 --max-leaves 4

oracle-1 是关键：如果它低于 BM25 基线，那么任何"让模型选得更准"的工作都是在
一个天花板以下打转，不用做了。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Set

REPO = Path(__file__).resolve().parents[3]


def load_leaves(dirs_json: Path) -> Dict[str, List[str]]:
    data = json.loads(dirs_json.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    return {nid: (n.get("docs") or [])
            for nid, n in nodes.items() if not n.get("children")}


def best_coverage(gold: Set[str], leaf_docs: Dict[str, List[str]],
                  budget: int, max_leaves: int) -> int:
    """在 budget 篇文档内，最多进 max_leaves 个叶子，能覆盖多少 gold 文档。"""
    relevant = [(nid, len(gold & set(docs)))
                for nid, docs in leaf_docs.items() if gold & set(docs)]
    best = 0
    for size in range(1, min(max_leaves, len(relevant)) + 1):
        for combo in itertools.combinations(relevant, size):
            taken: Set[str] = set()
            for nid, _ in combo:
                taken |= set(leaf_docs[nid])
            if len(taken) > budget:
                continue
            best = max(best, len(gold & taken))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", default=None,
                        help="dirs.json 路径，默认在 --kb 下找")
    parser.add_argument("--kb", default="eval/data-v12")
    parser.add_argument("--split", default="eval/dev.json")
    parser.add_argument("--budget", type=int, default=8, help="召回文档数上限")
    parser.add_argument("--max-leaves", type=int, default=4,
                        help="oracle-N 允许同时进入的叶子数")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.dirs) if args.dirs else None
    if path is None:
        found = sorted(glob.glob(str(REPO / args.kb / "kbs" / "*" / "dirs.json")))
        if not found:
            raise SystemExit("no dirs.json under %s; run fs.index first" % args.kb)
        path = Path(found[0])
    leaf_docs = load_leaves(path)

    rows = [r for r in json.loads((REPO / args.split).read_text(encoding="utf-8"))
            if r.get("evidence_pdfs")]

    per_row = []
    for row in rows:
        gold = {os.path.basename(p) for p in row["evidence_pdfs"]}
        in_leaf = len([g for g in gold
                       if any(g in docs for docs in leaf_docs.values())])
        one = best_coverage(gold, leaf_docs, args.budget, 1)
        many = best_coverage(gold, leaf_docs, args.budget, args.max_leaves)
        per_row.append({
            "id": row["id"], "type": row["type"], "gold": len(gold),
            "in_some_leaf": in_leaf / len(gold) if gold else None,
            "oracle_1": one / len(gold) if gold else None,
            "oracle_n": many / len(gold) if gold else None,
            "leaves_holding_gold": len([1 for docs in leaf_docs.values()
                                        if gold & set(docs)]),
        })

    def macro(key: str) -> float:
        values = [r[key] for r in per_row if r[key] is not None]
        return round(statistics.fmean(values), 4)

    summary = {
        "split": args.split,
        "questions": len(per_row),
        "budget": args.budget,
        "max_leaves": args.max_leaves,
        "leaves": len(leaf_docs),
        "in_some_leaf": macro("in_some_leaf"),
        "oracle_1": macro("oracle_1"),
        "oracle_n": macro("oracle_n"),
        "mean_leaves_holding_gold": round(statistics.fmean(
            [r["leaves_holding_gold"] for r in per_row]), 2),
        "per_question": per_row,
    }

    print("tree      : %d leaves, budget %d docs, max %d leaves"
          % (summary["leaves"], args.budget, args.max_leaves))
    print("questions : %d (%s)" % (len(per_row), args.split))
    print()
    print("gold docs present in some leaf : %.4f" % summary["in_some_leaf"])
    print("ORACLE single path (1 leaf)    : %.4f" % summary["oracle_1"])
    print("ORACLE multi-branch (<=%d)      : %.4f"
          % (args.max_leaves, summary["oracle_n"]))
    print()
    print("evidence sits in %.2f different leaves per question (mean)"
          % summary["mean_leaves_holding_gold"])

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("\nwritten: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
