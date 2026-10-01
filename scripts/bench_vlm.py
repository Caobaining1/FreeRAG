#!/usr/bin/env python3
"""Benchmark a vision model on real figure crops from a knowledge base.

Two things are measured, because they answer different questions:

* speed — one image at a time, then N at a time. Everything else on this
  hardware has turned out to be memory-bandwidth bound (layout detection gets
  SLOWER with concurrency, 1/2/4 threads = 1.9/3.7/5.4 s per page), so whether a
  VLM can be run in parallel is not a given and has to be measured rather than
  assumed.
* quality — the actual prompt is used, guardrails included, so the output can be
  judged against what the figure really shows. The prompt forbids quoting
  numbers on purpose: a misread value that lands in the index is invisible
  afterwards (see docs/plan.md), while a misread value in a chat answer is not.

    scripts/bench_vlm.py --model qwen2.5vl:3b --dir /tmp/figs
    scripts/bench_vlm.py --model qwen2.5vl:3b --dir /tmp/figs --concurrency 4
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = 'http://127.0.0.1:11434'

#: The prompt this feature would actually ship with. It asks for the figure's
#: meaning rather than its words (the words are already in the text layer, see
#: the measurement in docs/plan.md), and it forbids the one output that would
#: poison the index: a confidently wrong number.
PROMPT = """你是文档图表理解助手。这张图来自一篇学术论文，图中文字可能是英文。

请用中文写 2-4 句描述，说明：
1. 这是什么类型的图（折线图/柱状图/示意图/流程图/截图等）；
2. 它展示了什么对比、趋势或结构；
3. 能得出的主要结论。

要求：
- 不要引用任何具体数值、分数、百分比或坐标轴刻度值——你读到的数字可能不准；
- 不要复述图注（caption）的文字；
- 只描述你能明确看出的内容，看不清就直说看不清；
- 直接给出描述，不要客套话。"""


def image_message(path):
    with open(path, 'rb') as handle:
        encoded = base64.b64encode(handle.read()).decode()
    return {'role': 'user', 'content': PROMPT, 'images': [encoded]}


def ask(model, path, timeout=600, keep_alive='5m'):
    payload = json.dumps({
        'model': model,
        'messages': [image_message(path)],
        'stream': False,
        'keep_alive': keep_alive,
        # Deterministic: two runs over the same figure must produce the same
        # caption, or the index is not reproducible.
        'options': {'temperature': 0, 'seed': 7},
    }).encode()
    request = urllib.request.Request(
        OLLAMA + '/api/chat', data=payload, headers={'Content-Type': 'application/json'}
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body.get('message', {}).get('content', '').strip(), time.time() - started


def unload(model):
    """Drop the model so the next load time is a real cold load."""
    payload = json.dumps({'model': model, 'messages': [], 'keep_alive': 0}).encode()
    request = urllib.request.Request(
        OLLAMA + '/api/chat', data=payload, headers={'Content-Type': 'application/json'}
    )
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except Exception:
        pass


def loaded_models():
    with urllib.request.urlopen(OLLAMA + '/api/ps', timeout=30) as response:
        return json.load(response).get('models', [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dir', default='/tmp/figs')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--out', default='')
    parser.add_argument('--cold', action='store_true', help='unload first, to time a real cold load')
    args = parser.parse_args()

    images = sorted(
        os.path.join(args.dir, name) for name in os.listdir(args.dir) if name.endswith('.png')
    )
    if not images:
        print('no images in %s' % args.dir, file=sys.stderr)
        return 1

    if args.cold:
        unload(args.model)

    print('model=%s images=%d concurrency=%d' % (args.model, len(images), args.concurrency))

    # The first call carries the load, so it is reported separately rather than
    # averaged into the per-image figure.
    first_text, first_seconds = ask(args.model, images[0])
    resident = loaded_models()
    size = resident[0].get('size_vram', 0) / 1e9 if resident else 0
    print('  load + first image: %.1fs   resident: %.2f GB' % (first_seconds, size))

    results = [{'image': os.path.basename(images[0]), 'seconds': first_seconds, 'text': first_text}]
    rest = images[1:]

    started = time.time()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outputs = list(pool.map(lambda p: ask(args.model, p), rest))
    else:
        outputs = [ask(args.model, p) for p in rest]
    elapsed = time.time() - started

    for path, (text, seconds) in zip(rest, outputs):
        results.append({'image': os.path.basename(path), 'seconds': seconds, 'text': text})

    if rest:
        per_image = sum(seconds for _t, seconds in outputs) / len(outputs)
        print('  %d image(s): %.1fs total, %.1fs per image, %.2f images/s'
              % (len(rest), elapsed, per_image, len(rest) / elapsed))

    path = args.out or '/tmp/bench_vlm_%s_c%d.json' % (args.model.replace(':', '_'), args.concurrency)
    with open(path, 'w') as handle:
        json.dump({'model': args.model, 'concurrency': args.concurrency,
                   'size_vram_gb': size, 'results': results}, handle, ensure_ascii=False, indent=2)
    print('  written: %s' % path)

    for item in results:
        print('\n--- %s (%.1fs)\n%s' % (item['image'], item['seconds'], item['text']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
