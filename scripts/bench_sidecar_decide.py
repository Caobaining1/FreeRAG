#!/usr/bin/env python3
"""Measure the sidecar's decide throughput at a given lane width.

    scripts/bench_sidecar_decide.py --threads 1 --calls 8
    scripts/bench_sidecar_decide.py --threads 2 --calls 8

This is the controlled half of the parallelism question. An end-to-end `ask`
cannot answer it: most of its wall time is one answer being generated at the
model's own rate (measured 6.4 tok/s, so a 1,200-character answer is about 45 s
on its own), and what the sub-loops do between the generations is a handful of
seconds either way. Driving the sidecar directly with N decisions isolates the
part that changed.

All calls are issued at once, the way two sub-loops issue them: the question is
whether the second one starts before the first finishes.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SIDECAR = os.path.join(REPO, 'sidecar', 'parse_server.py')
PYTHON = os.path.join(REPO, '.venv314', 'bin', 'python')

#: One paragraph of the size a real pool of passages contributes.
PARAGRAPH = ('The article reports that the Yankees rebuilt their roster for the 2024 season, '
             'signing two starting pitchers and promoting three infielders from the farm system. ')

#: How many paragraphs the state carries; --state-reps scales it. The default is
#: about 800 characters, near the low end of what a scoped pool looks like.
STATE_REPS = 6


def build_calls(count, reps):
    lines = []
    for index in range(count):
        lines.append(json.dumps({
            'jsonrpc': '2.0',
            'id': index + 1,
            'method': 'decide',
            'params': {
                'instructions': 'Does the provided evidence sufficiently answer the question: '
                                'What did the Yankees do for the 2024 season?',
                'criteria': {'true': 'the evidence answers the question',
                             'false': 'the evidence does not answer the question'},
                'state': PARAGRAPH * reps,
                'qtype': 'noul',
            },
        }))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, required=True, help='FREERAG_DECIDE_THREADS')
    parser.add_argument('--calls', type=int, default=8)
    parser.add_argument('--state-reps', type=int, default=STATE_REPS,
                        help='paragraphs in the state; scales the per-call compute')
    args = parser.parse_args()

    env = dict(os.environ, FREERAG_DECIDE_THREADS=str(args.threads))
    process = subprocess.Popen(
        [PYTHON, SIDECAR], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1,
    )

    started = time.time()
    for line in build_calls(args.calls, args.state_reps):
        process.stdin.write(line + '\n')
    process.stdin.flush()

    answers = {}
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get('id') is None:
            continue
        answers[message['id']] = message
        if len(answers) == args.calls:
            break
    elapsed = time.time() - started

    process.stdin.close()
    process.wait(timeout=60)

    failures = [m for m in answers.values() if 'error' in m]
    print('%d call(s), %d worker(s): %.1fs total, %.2fs per call, %d answered, %d failed' % (
        args.calls, args.threads, elapsed, elapsed / args.calls, len(answers), len(failures)))
    for message in failures[:3]:
        print('  error: %s' % json.dumps(message['error'], ensure_ascii=False)[:160])
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
