#!/usr/bin/env python3
"""Parse a whole corpus and report per-format cost and failure rate.

This is the acceptance run for the multi-format parser (plan.md §5.1): it walks
the directories built by `build_multiformat_dataset.py`, parses every file, and
prints what each format cost and what it produced. Deliberately verbose about
failures — a parse that silently yields zero chunks is a failure, not a pass.

    scripts/verify_multiformat.py [corpus_dir] [--no-cache]

Exits non-zero if any document failed, so it can gate a change.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sidecar'))

from documents import SUPPORTED_SUFFIXES  # noqa: E402
from pipeline import parse_document  # noqa: E402

DEFAULT_CORPUS = '/Users/cbn/workspace/datasets/multiformat100'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', nargs='?', default=DEFAULT_CORPUS)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='parse at most N files per format')
    args = parser.parse_args()

    by_format = collections.defaultdict(list)
    for root, _dirs, files in os.walk(args.corpus):
        for name in sorted(files):
            suffix = os.path.splitext(name)[1].lower()
            if suffix in SUPPORTED_SUFFIXES:
                by_format[os.path.basename(root)].append(os.path.join(root, name))

    if not by_format:
        print('no supported documents under %s' % args.corpus, file=sys.stderr)
        return 1

    total = failures = 0
    report = {}
    for fmt in sorted(by_format):
        paths = by_format[fmt]
        if args.limit:
            paths = paths[:args.limit]
        times = []
        chunks = []
        types = collections.Counter()
        errors = []

        started = time.time()
        for path in paths:
            total += 1
            begin = time.time()
            try:
                result = parse_document(path, use_cache=not args.no_cache)
            except Exception as error:  # one bad document must not stop the run
                failures += 1
                errors.append('%s: %s' % (os.path.basename(path), error))
                continue
            times.append(time.time() - begin)
            chunks.append(result['chunk_count'])
            types.update(c['metadata']['block_type'] for c in result['chunks'])
            if result['chunk_count'] == 0:
                failures += 1
                errors.append('%s: parsed but produced no chunks' % os.path.basename(path))

        elapsed = time.time() - started
        report[fmt] = {
            'files': len(paths),
            'errors': errors,
            'mean_seconds': sum(times) / len(times) if times else 0,
            'total_seconds': elapsed,
            'mean_chunks': sum(chunks) / len(chunks) if chunks else 0,
            'chunk_types': dict(types),
        }
        print('%-6s %3d files  %6.1fs total  %5.2fs/file  %5.1f chunks/file  %s'
              % (fmt, len(paths), elapsed,
                 report[fmt]['mean_seconds'], report[fmt]['mean_chunks'],
                 dict(types.most_common())))
        for message in errors:
            print('        FAIL %s' % message)

    print()
    print('%d documents, %d failures' % (total, failures))
    out = os.path.join(args.corpus, 'verification.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print('report: %s' % out)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
