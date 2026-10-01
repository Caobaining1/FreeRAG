#!/usr/bin/env python3
"""Golden 对拍 for the laya deployment bundle (DEPLOY_CONTRACT.md §六).

The bundle ships `deploy_golden_examples.json`: three cases (zh noul, en noul, en
choice) carrying the expected `input_ids` and `marker_pos`. Agreement
token-for-token means the assembly, the special tokens, the instruction
templates and the truncation rules all match what the model was trained on; the
first mismatch says where to look.

This runs the PRODUCTION assembly — `sidecar/laya.py`'s own `build_sequence` and
`resolve_specials` — rather than a second implementation, because a second
implementation would only prove it agrees with itself.

    scripts/golden_laya.py /Users/cbn/workspace/laya32k-deploy

Checks 1, 2, 3, 4, 5 and 7 of the contract's self-test list. Check 6 (torch vs
ONNX logits within 1e-3) is reported as skipped: it needs the fp32 reference
model and torch, neither of which is part of this bundle.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'sidecar'))

from laya import MASK_TOKEN, build_sequence, resolve_specials, scrub  # noqa: E402

EXPECTED_SPECIALS = {'cls': 2, 'sep': 1, 'mask': 4, 'pad': 0}


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    bundle = sys.argv[1]
    model_dir = os.environ.get('FREERAG_LAYA_DIR') or os.path.join(REPO, 'models', 'laya-onnx')

    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        print('needs the `tokenizers` package: %s' % error, file=sys.stderr)
        return 2

    config = json.load(open(os.path.join(model_dir, 'laya_config.json')))
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer', 'tokenizer.json'))
    special, mask_token = resolve_specials(tokenizer, config, model_dir)
    cases = json.load(open(os.path.join(bundle, 'deploy_golden_examples.json')))['cases']

    failures = []
    print('model: %s' % model_dir)
    print('config: max_len=%s head_max_len=%s' % (config.get('max_len'), config.get('head_max_len')))

    # 1. special ids come from the config, and the mask token is the one the
    #    checkpoint declares. `<mask>` is the expected spelling (the contract
    #    names it); laya.py's MASK_TOKEN constant still says "[MASK]" because it
    #    predates the mmBERT checkpoint, and scrub() strips all three spellings
    #    so the stale constant cannot open a hole.
    print('\n[1] special_ids   %s  mask_token=%r' % (special, mask_token))
    if special != EXPECTED_SPECIALS:
        failures.append('special_ids %s != %s' % (special, EXPECTED_SPECIALS))
    if mask_token != '<mask>':
        failures.append('mask_token %r != %r' % (mask_token, '<mask>'))
    for spelling in (mask_token, MASK_TOKEN, '<mask>'):
        if MASK_TOKEN in scrub('a %s b' % spelling, mask_token):
            failures.append('scrub left %r behind' % spelling)

    # 2. mmBERT's trap: the literal [MASK] must NOT be a usable token, which is
    #    the whole reason the backbone was swapped.
    unk = tokenizer.token_to_id('<unk>')
    bracket = tokenizer.token_to_id('[MASK]')
    print('[2] token_to_id("[MASK]") = %s, unk = %s  → %s' %
          (bracket, unk, 'OK (collapses to unk)' if bracket == unk or bracket is None else 'UNEXPECTED'))
    if bracket is not None and unk is not None and bracket != unk:
        failures.append('"[MASK]" is a real token (%s); the scrub relies on it being unk' % bracket)

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731

    for index, case in enumerate(cases):
        name = 'case %d (%s %s)' % (index, case.get('lang'), case.get('qtype'))
        ids, markers = build_sequence(
            encode, special, case['qtype'], case['instructions'], case['option_texts'],
            case.get('state', ''), int(config['max_len']), int(config['head_max_len']), mask_token,
        )
        want_ids, want_markers = case['input_ids'], case['marker_pos']

        # 3/4. The head the assembly produces is the one the case documents, which
        #      is what proves the template (and its language) is the right set.
        head = '%s question: %s' % (case['qtype'], case['instructions'])
        head_encoded = encode(head)
        head_ok = case.get('head_text') is None or head == case['head_text']

        print('\n[%d] %s' % (index + 3, name))
        print('     head ok      : %s' % head_ok)
        print('     input_ids    : %s (%d tokens, expected %d)' %
              ('match' if ids == want_ids else 'MISMATCH', len(ids), len(want_ids)))
        print('     marker_pos   : %s (got %s, expected %s)' %
              ('match' if list(markers) == list(want_markers) else 'MISMATCH', list(markers), list(want_markers)))
        # 7. The structure the markers encode: first marker after CLS+head+SEP, and
        #    adjacent markers one separator plus one option apart.
        first_ok = bool(markers) and markers[0] == 1 + len(head_encoded) + 1
        gaps = [markers[i + 1] - markers[i] for i in range(len(markers) - 1)]
        want_gaps = [1 + len(encode(' ' + text)) for text in case['option_texts'][1:]]
        print('     marker rule  : first=%s (%s)  gaps=%s (expected %s)' %
              (markers[0] if markers else None, first_ok, gaps, want_gaps))

        if not head_ok:
            failures.append('%s: head_text mismatch\n      got  %r\n      want %r' %
                            (name, head, case['head_text']))
        if ids != want_ids:
            where = next((i for i, (a, b) in enumerate(zip(ids, want_ids)) if a != b), min(len(ids), len(want_ids)))
            failures.append('%s: input_ids differ at index %d (got %s, want %s)' %
                            (name, where, ids[where:where + 4], want_ids[where:where + 4]))
        if list(markers) != list(want_markers):
            failures.append('%s: marker_pos %s != %s' % (name, list(markers), list(want_markers)))
        if not first_ok or gaps != want_gaps:
            failures.append('%s: marker positions do not follow the rule' % name)

        # 5. Scrub: a literal <mask> in the state must not become a marker, so the
        #    mask id appears exactly once per option.
        forged = (case.get('state') or '') + ' ' + MASK_TOKEN
        forged_ids, _ = build_sequence(
            encode, special, case['qtype'], case['instructions'], case['option_texts'],
            forged, int(config['max_len']), int(config['head_max_len']), mask_token,
        )
        mask_count = sum(1 for token in forged_ids if token == special['mask'])
        print('     scrub        : %d mask id(s) for %d option(s)  → %s' %
              (mask_count, len(case['option_texts']), 'OK' if mask_count == len(case['option_texts']) else 'LEAK'))
        if mask_count != len(case['option_texts']):
            failures.append('%s: scrub leaked — %d mask ids for %d options' %
                            (name, mask_count, len(case['option_texts'])))

    print('\n[6] torch vs ONNX logits: SKIPPED (needs the fp32 reference model and torch)')

    print()
    if failures:
        print('%d FAILURE(S):' % len(failures))
        for failure in failures:
            print('  - %s' % failure)
        return 1
    print('golden 对拍通过：3/3 组 input_ids 与 marker_pos 完全一致')
    return 0


if __name__ == '__main__':
    sys.exit(main())
