#!/usr/bin/env python3
"""Build a 100-document, multi-format corpus for parser acceptance tests.

Why generated rather than downloaded: the parser has only ever been fed PDFs
(plan.md §5), so it needs formats it has never seen. Real PDFs come from
MultiHop-RAG; the .txt/.docx/.doc files carry text extracted from *different*
PDFs, converted with macOS `textutil`, so every file holds real document prose
instead of placeholder text — a chunker tested on lorem ipsum proves nothing.

Layout (one directory per format, so a failure names the format):

    <out>/pdf/   60  copied as-is
    <out>/docx/  15  textutil from extracted text
    <out>/doc/   10  textutil (Word 97)
    <out>/txt/   15  extracted text

Sampling is deterministic (sorted paths, even stride) so two runs produce the
same corpus and a regression can be compared against a known list.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

PDF_SOURCE = '/Users/cbn/workspace/benchmarks/MultiHop-RAG/pdfs'
DEFAULT_OUT = '/Users/cbn/workspace/datasets/multiformat100'

#: name -> (how many, extension)
COMPOSITION = (('pdf', 60), ('docx', 15), ('doc', 10), ('txt', 15))

#: A .txt smaller than this says the PDF was a scan with no text layer, and the
#: converted formats would then be empty documents that test nothing.
MIN_TEXT_CHARS = 2000


def sample(paths, count, stride_offset=0):
    """Evenly spaced sample: deterministic and spread across the corpus."""
    if len(paths) <= count:
        return list(paths)
    step = len(paths) / count
    return [paths[int((i + stride_offset) * step) % len(paths)] for i in range(count)]


def extract_text(pdf_path):
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return '\n\n'.join(page.get_text() for page in doc).strip()


def convert(text_path, out_path, fmt):
    subprocess.run(
        ['textutil', '-convert', fmt, '-output', out_path, text_path],
        check=True, capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--source', default=PDF_SOURCE)
    parser.add_argument('--force', action='store_true', help='rebuild over an existing dir')
    args = parser.parse_args()

    if os.path.exists(args.out):
        if not args.force:
            print('%s exists; pass --force to rebuild' % args.out)
            return 1
        shutil.rmtree(args.out)

    if not shutil.which('textutil'):
        print('textutil not found: this must run on macOS', file=sys.stderr)
        return 1

    pdfs = sorted(p for p in os.listdir(args.source) if p.lower().endswith('.pdf'))
    if len(pdfs) < sum(n for _, n in COMPOSITION):
        print('only %d PDFs in %s' % (len(pdfs), args.source), file=sys.stderr)
        return 1

    random.seed(20260930)
    random.shuffle(pdfs)

    want = dict(COMPOSITION)

    # 1. The PDFs themselves.
    os.makedirs(os.path.join(args.out, 'pdf'))
    for name in pdfs[:want['pdf']]:
        shutil.copy2(os.path.join(args.source, name), os.path.join(args.out, 'pdf', name))
    cursor = want['pdf']

    # 2. The other formats, from the NEXT PDFs in the shuffled list, so a
    #    document's text never appears twice in the corpus.
    text_dir = os.path.join(args.out, 'txt')
    os.makedirs(text_dir)
    os.makedirs(os.path.join(args.out, 'docx'))
    os.makedirs(os.path.join(args.out, 'doc'))
    scratch = tempfile.mkdtemp(prefix='multiformat-')
    built = {'pdf': want['pdf']}

    usable = []
    while len(usable) < sum(n for _, n in COMPOSITION[1:]) and cursor < len(pdfs):
        name = pdfs[cursor]
        cursor += 1
        try:
            text = extract_text(os.path.join(args.source, name))
        except Exception as error:  # a corrupt PDF must not abort the build
            print('  skip %s (%s)' % (name, error))
            continue
        if len(text) < MIN_TEXT_CHARS:
            continue
        usable.append((name, text))

    if len(usable) < sum(n for _, n in COMPOSITION[1:]):
        print('only %d PDFs had a text layer' % len(usable), file=sys.stderr)
        return 1

    counts = {'txt': len(usable) - want['docx'] - want['doc'], 'docx': want['docx'], 'doc': want['doc']}

    for index, (name, text) in enumerate(usable):
        stem = 'doc%03d' % index
        if index < want['txt']:
            target = os.path.join(text_dir, stem + '.txt')
        else:
            # The converted formats need a text file to read, but it is an
            # intermediate: leaving it in txt/ would put the same document in
            # the corpus twice and make the count 25 files more than 100.
            target = os.path.join(scratch, stem + '.txt')
        with open(target, 'w') as handle:
            handle.write(text + '\n')
        if index < want['docx']:
            convert(target, os.path.join(args.out, 'docx', stem + '.docx'), 'docx')
        elif index < want['docx'] + want['doc']:
            convert(target, os.path.join(args.out, 'doc', stem + '.doc'), 'doc')

    built.update(counts)

    shutil.rmtree(scratch, ignore_errors=True)
    print('built %d documents in %s' % (sum(built.values()), args.out))
    for fmt, count in built.items():
        total = sum(
            os.path.getsize(os.path.join(args.out, fmt, f))
            for f in os.listdir(os.path.join(args.out, fmt))
        )
        print('  %-5s %3d files  %6.1f MB' % (fmt, count, total / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
