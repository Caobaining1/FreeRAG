#!/usr/bin/env python3
"""Run the bundle's own position ablation against the ONNX checkpoint.

    scripts/ablate_laya_position.py --model-dir models/laya-onnx --bucket 400

Why this and not a synthetic needle probe: the bundle states the criterion
outright (HANDOVER.md §验收): "`decay_pp` 是唯一能证明'32k 真的生效'的证据。只看准确率会自欺
—— 一个只学会计较'近处'的模型也能在域内测试集上刷出不错的准确率." Three points of
evidence, in the bundle's terms: macro accuracy >= 85%, `decay_pp <= 8`, and
Needle@32k >= 0.90. Accuracy is the one my earlier probe could not settle, and
position decay is the one it accidentally tripped.

decay_pp is the spread of accuracy across four evidence positions, 0.03 to 0.97
of the state (configs/laya32k.yaml). The relocation is the reference
implementation's: `_paragraphs`, `_guess_key_index` and `relocate_state` are
lifted from src/evaluate.py by AST and executed, NOT re-written — a second
implementation would only prove it agrees with itself, which is the same reason
scripts/golden_laya.py drives the production assembly.

The reference evaluator itself cannot run here: it imports torch, and this
virtualenv has none. Executing three pure string functions out of it is the
closest thing to running it that the machine allows.
"""
import argparse
import ast
import json
import os
import random
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BUNDLE = '/Users/cbn/workspace/laya-32k'
sys.path.insert(0, os.path.join(REPO, 'sidecar'))

from laya import LayaDecider  # noqa: E402

#: The bundle's own positions (configs/laya32k.yaml: position_fractions).
FRACTIONS = (0.03, 0.3, 0.6, 0.97)

#: The bundle's gate (src/evaluate.py:365, "位置衰减 <= 8pp").
DECAY_GATE_PP = 8.0


def load_reference_functions():
    """Exec the reference relocator out of src/evaluate.py, torch-free.

    Only these three are taken; the module around them pulls torch, a model and
    a tokenizer, so importing it is not an option on this machine.
    """
    path = os.path.join(BUNDLE, 'src', 'evaluate.py')
    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source)
    wanted = {'_paragraphs', '_guess_key_index', 'relocate_state'}
    namespace: Dict[str, object] = {'List': List}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    missing = wanted - set(namespace)
    if missing:
        raise SystemExit('could not lift %s from %s' % (sorted(missing), path))
    return namespace


def sample_rows(path, bucket, per_class, seed):
    positives, negatives = [], []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('qtype') != 'noul':
                continue
            if int(row['meta'].get('target_len') or 0) != bucket:
                continue
            if len(row['state'].split('\n\n')) < 2:
                continue  # a one-paragraph state cannot be relocated
            (positives if row['meta']['is_positive'] else negatives).append(row)
    random.Random(seed).shuffle(positives)
    random.Random(seed + 1).shuffle(negatives)
    sample = positives[:per_class] + negatives[:per_class]
    random.Random(seed + 2).shuffle(sample)
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--jsonl', default=os.path.join(BUNDLE, 'data', 'test.jsonl'))
    parser.add_argument('--bucket', type=int, default=400)
    parser.add_argument('--per-class', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    reference = load_reference_functions()
    relocate = reference['relocate_state']
    guess_key = reference['_guess_key_index']
    paragraphs = reference['_paragraphs']

    rows = sample_rows(args.jsonl, args.bucket, args.per_class, args.seed)
    if not rows:
        raise SystemExit('no relocatable rows for target_len=%d' % args.bucket)

    decider = LayaDecider(directory=args.model_dir)
    if not decider.available:
        raise SystemExit('no checkpoint in %s' % args.model_dir)
    decider._load()

    keyed = [(row, guess_key(paragraphs(row['state']), row)) for row in rows]
    print('%s  bucket=%d  rows=%d  state~%d chars  fractions=%s' % (
        os.path.basename(args.model_dir), args.bucket, len(rows),
        sum(len(r['state']) for r in rows) // len(rows), list(FRACTIONS)), flush=True)

    per_fraction = {}
    for fraction in FRACTIONS:
        correct = 0
        started = time.time()
        for row, key_index in keyed:
            state = relocate(row['state'], key_index, fraction)
            criteria = {option['key']: option['text'] for option in row['options']}
            result = decider.decide(row['instructions'], criteria, state=state, qtype='noul')
            expected = 'true' if row['answer_index'] == 1 else 'false'
            correct += int(result['choice'] == expected)
        accuracy = correct / len(keyed)
        per_fraction[fraction] = accuracy
        print('  key evidence at %3d%%  ->  acc %.3f  (n=%d, %.1fs)' % (
            int(fraction * 100), accuracy, len(keyed), time.time() - started), flush=True)

    values = list(per_fraction.values())
    decay_pp = (max(values) - min(values)) * 100
    print('\n  decay_pp = %.1fpp   gate <= %.1fpp  ->  %s' % (
        decay_pp, DECAY_GATE_PP, 'PASS' if decay_pp <= DECAY_GATE_PP else 'FAIL'), flush=True)
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump({'model_dir': os.path.abspath(args.model_dir), 'bucket': args.bucket,
                       'fractions': {str(k): v for k, v in per_fraction.items()},
                       'decay_pp': decay_pp, 'gate_pp': DECAY_GATE_PP,
                       'passed': decay_pp <= DECAY_GATE_PP}, handle, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
