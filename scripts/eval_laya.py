#!/usr/bin/env python3
"""Score a Laya checkpoint on the bundle's own labelled test set.

    scripts/eval_laya.py --model-dir models/laya-onnx --bucket 800 --per-class 8

The synthetic probes in compare_laya.py can show a window working or a cost
curve, but they cannot answer "is this checkpoint any good": their labels are
mine, and a single case flipping proves nothing. `laya-32k/data/test.jsonl` is
the instrument that can - 2,465 `noul` decisions with `answer_index` as ground
truth, balanced 1115 positive / 1350 negative, and bucketed by `meta.target_len`
so the long-context cases can be isolated.

Accuracy alone would be misleading here: saying `false` to everything scores 55%
on this set. So per-class accuracy (how often it says "enough" when the evidence
IS enough, and "not enough" when it is not) is what gets reported, along with
the answer distribution that shows a collapse.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'sidecar'))

from laya import LayaDecider  # noqa: E402

DEFAULT_JSONL = '/Users/cbn/workspace/laya-32k/data/test.jsonl'


def load_bucket(path, bucket, per_class, seed):
    """A balanced sample of one `target_len` bucket, deterministic in `seed`."""
    positives, negatives = [], []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('qtype') != 'noul':
                continue
            if int(row['meta'].get('target_len') or 0) != bucket:
                continue
            (positives if row['meta']['is_positive'] else negatives).append(row)
    random.Random(seed).shuffle(positives)
    random.Random(seed + 1).shuffle(negatives)
    sample = positives[:per_class] + negatives[:per_class]
    random.Random(seed + 2).shuffle(sample)
    return sample, len(positives), len(negatives)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--jsonl', default=DEFAULT_JSONL)
    parser.add_argument('--bucket', type=int, required=True, help='meta.target_len')
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    sample, available_pos, available_neg = load_bucket(
        args.jsonl, args.bucket, args.per_class, args.seed)
    if not sample:
        print('no rows for target_len=%d' % args.bucket, file=sys.stderr)
        return 2

    decider = LayaDecider(directory=args.model_dir)
    if not decider.available:
        print('no checkpoint in %s' % args.model_dir, file=sys.stderr)
        return 2
    started = time.time()
    decider._load()
    build_seconds = time.time() - started

    print('%s  bucket=%d  sample=%d (from %d pos / %d neg)  build=%.2fs' % (
        os.path.basename(args.model_dir), args.bucket, len(sample),
        available_pos, available_neg, build_seconds), flush=True)

    rows = []
    for row in sample:
        expected = 'true' if row['answer_index'] == 1 else 'false'
        criteria = {option['key']: option['text'] for option in row['options']}
        begin = time.time()
        try:
            result = decider.decide(row['instructions'], criteria,
                                    state=row['state'], qtype='noul')
            choice, probability = result['choice'], result['probability']
            error = ''
        except Exception as failure:  # noqa: BLE001
            choice, probability, error = '', 0.0, '%s: %s' % (type(failure).__name__, failure)
        elapsed = time.time() - begin
        rows.append({
            'id': row['id'],
            'is_positive': bool(row['meta']['is_positive']),
            'expected': expected,
            'choice': choice,
            'probability': probability,
            'state_chars': len(row['state']),
            'state_tokens': len(decider._encode(row['state'])),
            'seconds': round(elapsed, 1),
            'correct': choice == expected,
            'error': error,
        })
        print('  %-46s %s exp=%-5s got=%-5s p=%-6s %5d chars  %6.1fs' % (
            row['id'][:46], 'pos' if row['meta']['is_positive'] else 'neg',
            expected, choice or 'ERR', probability, len(row['state']), elapsed), flush=True)

    scored = [row for row in rows if not row['error']]
    positives = [row for row in scored if row['is_positive']]
    negatives = [row for row in scored if not row['is_positive']]
    says_true = [row for row in scored if row['choice'] == 'true']

    def rate(subset):
        if not subset:
            return float('nan')
        return sum(1 for row in subset if row['correct']) / len(subset)

    payload = {
        'model_dir': os.path.abspath(args.model_dir),
        'bucket': args.bucket,
        'accuracy': rate(scored),
        'accuracy_on_positives': rate(positives),
        'accuracy_on_negatives': rate(negatives),
        'says_true_rate': len(says_true) / len(scored) if scored else float('nan'),
        'median_seconds': sorted(row['seconds'] for row in scored)[len(scored) // 2] if scored else None,
        'errors': [row for row in rows if row['error']],
        'rows': rows,
    }
    print('\n%s bucket=%d  N=%d  accuracy=%.3f  (positives %.3f / negatives %.3f)  '
          'says_true=%.2f  median=%.1fs' % (
              os.path.basename(args.model_dir), args.bucket, len(scored), payload['accuracy'],
              payload['accuracy_on_positives'], payload['accuracy_on_negatives'],
              payload['says_true_rate'], payload['median_seconds'] or 0), flush=True)
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
