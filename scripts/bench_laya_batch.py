#!/usr/bin/env python3
"""Measure batched Laya decisions against one-at-a-time decisions.

    scripts/bench_laya_batch.py --bucket 400 --rows 8

The hypothesis, from the measurements that killed the parallel version: this
checkpoint is fp32 and about 1.2 GB, so every forward pass streams the whole
model, and the machine is bandwidth-bound rather than core-bound (1/2/4 sidecar
workers scored 16 decisions in 7.1/7.5/7.6 s; two processes were worse than one;
Ollama with two parallel slots was 0.82x). A batch streams the weights once for
all of its rows, so it should be the one form of concurrency this machine can
pay for.

Three things are reported per batch size, in this order, because a speed number
without the first two is worthless:

  1. agreement — every batched choice must equal the one-at-a-time choice. The
     rows are padded to a common width, so a mismatch means the padding changed
     what the model saw (attention mask or positions), which would make the
     whole idea wrong rather than merely slow.
  2. probability drift — the largest per-row difference in `probability`,
     reported so a "matches" that is really "close enough" is visible.
  3. speed — total wall time for the batch versus the same rows one at a time.

Cases come from the deployment bundle's labelled test set, so the states are the
size and shape real sufficiency checks carry.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'sidecar'))

from laya import LayaDecider  # noqa: E402

BUNDLE = '/Users/cbn/workspace/laya-32k'
SIZES = (1, 2, 4, 8)


def load_cases(path, bucket, rows, seed=20261001):
    """Real labelled noul cases, longest state first so a batch is not padded
    by one outlier — and so the padding cost is the visible worst case."""
    cases = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('qtype') != 'noul':
                continue
            if int(row['meta'].get('target_len') or 0) != bucket:
                continue
            cases.append({
                'instructions': row['instructions'],
                'criteria': {option['key']: option['text'] for option in row['options']},
                'state': row['state'],
                'qtype': 'noul',
            })
    cases.sort(key=lambda case: len(case['state']))
    return cases[:rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=os.path.join(REPO, 'models', 'laya-onnx'))
    parser.add_argument('--jsonl', default=os.path.join(BUNDLE, 'data', 'test.jsonl'))
    parser.add_argument('--bucket', type=int, default=400)
    parser.add_argument('--rows', type=int, default=8)
    args = parser.parse_args()

    cases = load_cases(args.jsonl, args.bucket, args.rows)
    if not cases:
        raise SystemExit('no cases for target_len=%d' % args.bucket)

    decider = LayaDecider(directory=args.model_dir)
    started = time.time()
    decider._load()
    print('model loaded in %.1fs; %d case(s), states %d..%d chars' % (
        time.time() - started, len(cases),
        len(cases[0]['state']), len(cases[-1]['state'])))

    # Warm-up, and it is not optional: onnxruntime builds its execution plan
    # lazily on the first Run, which cost 1.07s for one row here against about
    # 0.26s for the next. Timings taken without this are dominated by that
    # one-off, which is what made a first attempt at this measurement report
    # eight sequential rows as faster than two.
    for _ in range(3):
        decider.decide(cases[0]['instructions'], cases[0]['criteria'],
                       state=cases[0]['state'], qtype=cases[0]['qtype'])

    def time_alone(subset):
        started = time.time()
        answers = [decider.decide(case['instructions'], case['criteria'],
                                  state=case['state'], qtype=case['qtype'])
                   for case in subset]
        return time.time() - started, answers

    def time_together(subset):
        started = time.time()
        answers = decider.decide_many(subset)
        return time.time() - started, answers

    print('\n rows  tokens   one-at-a-time    batched     speed   per row  agree  max Δprob')
    for size in SIZES:
        if size > len(cases):
            break
        subset = cases[:size]

        # Alternated and repeated, keeping the fastest of each: the machine is
        # shared with the user's other work, and a minimum is the part of the
        # distribution that belongs to the code rather than to the neighbours.
        best_alone, best_together = float('inf'), float('inf')
        alone = together = None
        for _ in range(3):
            seconds, answers = time_together(subset)
            best_together = min(best_together, seconds)
            together = answers
            seconds, answers = time_alone(subset)
            best_alone = min(best_alone, seconds)
            alone = answers

        agree = all(a['choice'] == b['choice'] for a, b in zip(alone, together))
        drift = max(abs(a['probability'] - b['probability']) for a, b in zip(alone, together))
        tokens = sum(len(case['state']) for case in subset) // 4
        print(' %4d  ~%-6d  %6.2fs          %6.2fs    %5.2fx  %6.2fs   %-5s  %.4f' % (
            size, tokens, best_alone, best_together,
            (best_alone / best_together) if best_together else float('nan'),
            best_together / size, 'yes' if agree else 'NO', drift))
        if not agree:
            for index, (a, b) in enumerate(zip(alone, together)):
                if a['choice'] != b['choice']:
                    print('   row %d: alone=%s (%.4f) batched=%s (%.4f)' % (
                        index, a['choice'], a['probability'], b['choice'], b['probability']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
