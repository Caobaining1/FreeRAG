#!/usr/bin/env python3
"""Compare two Laya checkpoints on the decisions the loop actually makes.

    scripts/compare_laya.py --model-dir models/laya-onnx            --out /tmp/laya32k.json
    scripts/compare_laya.py --model-dir models/laya-onnx.old-512    --out /tmp/laya512.json

Run one model per process, not both: the two checkpoints together are ~3 GB and
this machine swaps, so a single process would be timing a pageout, not a model.

Two kinds of measurement, and the second is the point:

1. Rate — session build, per-decision wall time, and how many tokens the state
   needs against the tokens the checkpoint can hold. The token count comes from
   one tokenizer call on the state, deliberately: re-deriving the assembly would
   be a second implementation, and the contract (§六) is explicit that a second
   implementation only proves it agrees with itself.

2. Visibility — whether the model ANSWERS a question whose evidence sits at the
   head of a long state versus at its tail. That is the behavioural signature of
   truncation and needs no instrumentation: a checkpoint that keeps the head and
   drops the tail answers head-cases and fails tail-cases at the same size. The
   filler is real corpus text rather than a repeated sentence, because the
   contract warns that a shifted input distribution degrades this model silently.
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

#: The sentence that answers EVIDENCE_QUESTION, and the question it answers.
EVIDENCE = ('Sam Bankman-Fried founded FTX in 2019 and was convicted of fraud by '
            'a New York jury in November 2023.')
EVIDENCE_QUESTION = 'When was Sam Bankman-Fried convicted of fraud?'

#: Real corpus prose for the filler, so the state is not a repeated sentence.
FILLER_DIRS = (
    '/Users/cbn/workspace/datasets/multiformat100/txt',
)

#: State sizes in CHARACTERS: 1k fits the 512 checkpoint, 48k does not.
STATE_SIZES = (1000, 4000, 16000, 48000)


def filler_pool():
    for directory in FILLER_DIRS:
        if not os.path.isdir(directory):
            continue
        texts = []
        for name in sorted(os.listdir(directory))[:6]:
            try:
                with open(os.path.join(directory, name), encoding='utf-8', errors='replace') as handle:
                    body = ' '.join(handle.read().split())
            except OSError:
                continue
            if body:
                texts.append(body)
        if texts:
            return texts
    # Fallback for a machine without the corpus: still prose, still unrelated.
    return ['Unrelated passage: the survey reviews prior work on retrieval augmented generation.'] * 8


def filler_of(pool, size):
    """~``size`` characters of unrelated corpus prose."""
    pieces, total, index = [], 0, 0
    while total < size:
        piece = pool[index % len(pool)]
        index += 1
        pieces.append(piece)
        total += len(piece) + 2
    return '\n\n'.join(pieces)[:size]


def cases(pool):
    """(id, qtype, instructions, criteria, state, expectation) per decision."""
    built = []

    def noul(case_id, state, expect, question=EVIDENCE_QUESTION):
        built.append((case_id, 'noul',
                      'Does the provided evidence sufficiently answer the question: %s?' % question,
                      {'true': 'the evidence answers the question',
                       'false': 'the evidence does not answer the question'},
                      state, expect))

    def choice(case_id, instructions, criteria, state, expect):
        built.append((case_id, 'choice', instructions, criteria, state, expect))

    # Sanity: the bundle's own golden states, whose answers are unambiguous.
    noul('golden-zh-noul',
         '现在法定假日是元旦1天，春节3天，共计11天。\n\n无关段落：本综述回顾了检索增强生成的既有工作。',
         'true', '国家法定节假日共多少天')
    noul('golden-en-noul',
         'Sam Bankman-Fried founded FTX in 2019.\n\nUnrelated: the survey reviews prior work.',
         'true', 'Who is Sam Bankman-Fried?')
    choice('golden-en-choice',
           'Given the evidence, which option is correct?',
           {'A': 'first choice', 'B': 'second choice', 'C': 'third choice'},
           'Only the second option is supported by the evidence.',
           'B: second choice')

    # The visibility grid: same question, same evidence, differing only in
    # whether the evidence is reachable at the head or buried at the tail.
    for size in STATE_SIZES:
        body = filler_of(pool, size)
        noul('evidence-head-%dk' % (size // 1000), EVIDENCE + '\n\n' + body, 'true')
        noul('evidence-tail-%dk' % (size // 1000), body + '\n\n' + EVIDENCE, 'true')

    # A tool choice, which is what the loop calls every round: four options and
    # a short state. NOT scored: more than one of these tools is defensible for
    # this question, so a "correct" answer here would be my invention rather than
    # ground truth. It is here for the latency and for whether the two
    # checkpoints agree.
    choice('tool-choice-unscored',
           'Which retrieval tool is most likely to find what is still missing?',
           {'hybrid_search': 'finds passages that mean the same thing',
            'grep_search': 'finds exact strings such as codes',
            'list_chunks': 'reads a document back in order',
            'metadata_search': 'filters by metadata only'},
           'Question: what does the report say about the 2024 season?\n'
           'Missing terms: season',
           None)
    return built


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeat', type=int, default=3,
                        help='steady-state repeats for the timing of each case')
    args = parser.parse_args()

    decider = LayaDecider(directory=args.model_dir)
    if not decider.available:
        print('no checkpoint in %s' % args.model_dir, file=sys.stderr)
        return 2

    started = time.time()
    decider._load()  # the production loader; measured separately from inference
    build_seconds = time.time() - started

    config = decider._config
    pool = filler_pool()
    rows = []
    for case_id, qtype, instructions, criteria, state, expect in cases(pool):
        state_tokens = len(decider._encode(state))
        try:
            result = None
            started = time.time()
            for _ in range(max(1, args.repeat)):
                result = decider.decide(instructions, criteria, state=state, qtype=qtype)
            average_ms = (time.time() - started) * 1000 / max(1, args.repeat)
            rows.append({
                'case': case_id,
                'qtype': qtype,
                'state_chars': len(state),
                'state_tokens': state_tokens,
                'choice': result['choice'],
                'probability': result['probability'],
                'expect': expect,
                # Scored cases only: an unscored one has no ground truth, so
                # calling its answer a miss would be my label, not the model's
                # error.
                'correct': None if expect is None else result['choice'] == expect,
                'ms': round(average_ms, 1),
            })
            # Printed as each case finishes, not in a summary at the end: the
            # long-state cases can take minutes on a big checkpoint, and a run
            # whose progress is invisible cannot be told apart from a hung one.
            print('  %-24s %-7s %6d chars %6d tok  %-30s p=%-6s %7.1fms' % (
                case_id, qtype, len(state), state_tokens, result['choice'][:30],
                result['probability'], average_ms), flush=True)
        except Exception as error:  # noqa: BLE001 - a failure is a result here
            rows.append({'case': case_id, 'qtype': qtype, 'state_chars': len(state),
                         'state_tokens': state_tokens, 'error': '%s: %s' % (type(error).__name__, error),
                         'expect': expect, 'correct': None})

    payload = {
        'model_dir': os.path.abspath(args.model_dir),
        'max_len': config.get('max_len'),
        'head_max_len': config.get('head_max_len'),
        'session_build_s': round(build_seconds, 2),
        'config_keys': sorted(config.keys()),
        'rows': rows,
    }
    with open(args.out, 'w') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    scored = [row for row in rows if row.get('expect') is not None]
    print('%-9s max_len=%-6s build=%.2fs  scored=%d/%d  (unscored cases reported too)' % (
        os.path.basename(args.model_dir), config.get('max_len'), build_seconds,
        sum(1 for row in scored if row.get('correct')), len(scored)))
    for row in rows:
        if 'error' in row:
            print('  %-24s %-7s ERROR %s' % (row['case'], row['qtype'], row['error'][:70]))
            continue
        verdict = '--  ' if row['expect'] is None else ('ok  ' if row['correct'] else 'MISS')
        print('  %-24s %-7s %6d chars %6d tok  %-30s %s  %7.1fms' % (
            row['case'], row['qtype'], row['state_chars'], row['state_tokens'],
            row['choice'][:30], verdict, row['ms']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
