"""Laya: a non-generative typed-decision model over ONNX (docs/plan.md §6.6).

Laya does not generate text. Given a question, a set of named options and some
context, it scores each option; the decision is the argmax. That is what makes it
the right fit for the sufficiency check: ~33 ms, no hallucination surface, no
prompt-injection surface beyond the decision itself.

The sequence layout is fixed by the checkpoint and reproduced here from the
reference implementation (receptron/laya, ``src/sequence.ts``)::

    [CLS] {type} question: {instructions} [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] {state} [SEP]

Every option is prefixed with a literal ``[MASK]`` token; ``marker_pos`` records
those positions and the model emits one logit per option at them.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The process-wide ONNX session registry. Shared rather than reimplemented: the
# layout model learned this lesson first (a session costs 0.14s to build on CPU
# and 6.90s compiled for the ANE, which was being paid per document), and the
# same double-checked registry is what makes a second LayaDecider — or a second
# thread — reuse the session instead of building another 1.2 GB copy.
from layout_onnx import session_for  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: Question type -> qtype id, from the reference implementation's QTYPES.
QTYPES = {"choice": 0, "score": 1, "noul": 2}

#: Longest option text, in tokens (reference implementation caps at 48).
OPTION_TOKEN_CAP = 48

#: The marker token of the ModernBERT checkpoints.
#:
#: Kept only as a legacy default: on a multilingual backbone this string is not
#: the marker at all, and ``token_to_id("[MASK]")`` returns the unk id, so every
#: option's marker lands on the wrong position without anything raising.
MASK_TOKEN = "[MASK]"

#: Token roles -> the keys that declare them, in the order we try them.
#:
#: The roles are read from the tokenizer's **own declaration**, never from a
#: literal token name. Two failure modes motivate this, both silent:
#:
#: * mmBERT has no ``[MASK]``. ``token_to_id("[MASK]")`` there returns the unk id
#:   (3), so the marker positions are wrong and the output is noise.
#: * The id is only knowable from the declaration. mmBERT's ``config.json``
#:   carries a stale ``cls_token_id = 1`` while its ``tokenizer_config.json``
#:   declares ``cls_token = "<bos>"`` -> 2. Taking the wrong source puts ``<eos>``
#:   at position 0 — which is exactly where the act head reads the sequence from.
_ROLE_KEYS = {
    "cls": ("cls_token", "bos_token"),
    "sep": ("sep_token", "eos_token", "bos_token"),
    "mask": ("mask_token",),
    "pad": ("pad_token",),
}

#: Last-resort literal names, consulted only when nothing else resolves. Hitting
#: this is logged, because it is the shape that breaks on a multilingual
#: tokenizer rather than failing loudly.
_ROLE_LITERALS = {"cls": "[CLS]", "sep": "[SEP]", "mask": "[MASK]", "pad": "[PAD]"}

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "laya-onnx"
)

#: CPU intra-op threads for Laya.
#:
#: Measured on a 41-token decision, Apple M-series / 10 cores:
#:   CoreML 434 ms  |  CPU 1 thread 104 ms  |  CPU default 281 ms
#: This is the *opposite* of the layout model (where CoreML wins 46x): a 421M
#: text encoder over a handful of tokens is dominated by per-call graph upload,
#: and its ops do not parallelise, so extra threads only add contention.
DEFAULT_CPU_THREADS = 1


def model_dir() -> str:
    return os.environ.get("FREERAG_LAYA_DIR") or DEFAULT_MODEL_DIR


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return fallback


def size_bucket(count: int) -> str:
    """Option-count bucket, used to pick a per-cardinality temperature."""
    if count <= 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _token_text(value: Any) -> str:
    """A role declaration is either a plain string or a ``{"content": …}`` object.

    ``special_tokens_map.json`` uses the object form, ``tokenizer_config.json``
    the string form; both appear in the wild.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return ""


def _load_json_object(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_specials(
    tokenizer: Any, config: Optional[Dict[str, Any]] = None, directory: str = ""
) -> Tuple[Dict[str, int], str]:
    """Resolve ``{cls, sep, mask, pad}`` ids and the mask token string.

    Most authoritative first:

    1. ``laya_config.json``'s ``special_ids`` / ``mask_token`` — the training
       side's contract, i.e. what the checkpoint was actually built with.
    2. The model directory's ``tokenizer_config.json`` then
       ``special_tokens_map.json`` — the same declarations ``AutoTokenizer``
       reads, so this is the id that takes effect at encode time.
    3. Literal ``[CLS]``/``[SEP]``/``[MASK]``/``[PAD]`` — legacy fallback for a
       checkpoint that carries neither, logged because it is the shape that
       breaks silently on a multilingual tokenizer.

    Only ``token_to_id`` is used: the bare ``tokenizers.Tokenizer`` has no
    ``cls_token_id``-style attributes, and those attributes are on
    ``PreTrainedTokenizer``, which this sidecar deliberately does not import.
    """
    provided = (config or {}).get("special_ids") or {}

    declared: Dict[str, str] = {}
    if directory:
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            data = _load_json_object(os.path.join(directory, name))
            for role, keys in _ROLE_KEYS.items():
                if role in declared:
                    continue
                for key in keys:
                    text = _token_text(data.get(key))
                    if text and tokenizer.token_to_id(text) is not None:
                        declared[role] = text
                        break

    ids: Dict[str, int] = {}
    for role, keys in _ROLE_KEYS.items():
        value = provided.get(role)
        if value is not None:
            ids[role] = int(value)
            continue

        text = declared.get(role)
        if text:
            resolved = tokenizer.token_to_id(text)
            if resolved is not None:
                ids[role] = int(resolved)
                continue

        literal = _ROLE_LITERALS[role]
        resolved = tokenizer.token_to_id(literal)
        if resolved is None:
            raise ValueError(
                "cannot resolve the %r token: laya_config.json has no special_ids[%r], "
                "the tokenizer declares no %s, and %r is not in the vocabulary"
                % (role, role, "/".join(keys), literal)
            )
        LOGGER.warning(
            "Laya: %r resolved from the literal name %r (=%d). This is the legacy "
            "ModernBERT path; a multilingual checkpoint needs special_ids in "
            "laya_config.json or a role declaration in tokenizer_config.json.",
            role, literal, resolved,
        )
        ids[role] = int(resolved)

    unk = tokenizer.token_to_id("<unk>")
    for role in ("cls", "sep", "mask"):
        if unk is not None and ids[role] == unk:
            raise ValueError(
                "the %r token resolved to the unk id (%d); every marker would land on "
                "an unknown token. Check special_ids in laya_config.json." % (role, unk)
            )

    mask = (config or {}).get("mask_token")
    if not isinstance(mask, str) or not mask:
        mask = declared.get("mask") or MASK_TOKEN
    return ids, mask


def scrub(text: str, mask_token: str = MASK_TOKEN) -> str:
    """Remove marker tokens from untrusted text (the string half of the defence).

    Without it a document could embed the marker and shift the positions the
    model reads its decision from. This half is **not sufficient on its own** —
    see `scrub_ids`, which is applied to the encoded span as well.
    """
    cleaned = text or ""
    for token in (mask_token, MASK_TOKEN, "<mask>"):
        if token:
            cleaned = cleaned.replace(token, "")
    return cleaned


def scrub_ids(ids: Sequence[int], special: Dict[str, int]) -> List[int]:
    """Drop the marker and separator ids from an encoded span.

    This is the authoritative half. It is the only way to *prove* the span
    carries no marker — the string scrub cannot see an id that survived
    tokenisation in another form — and it also closes separator injection: a
    crafted state could otherwise embed the separator and forge a structural
    boundary, which the string scrub never covered at all.
    """
    drop = {special["mask"], special["cls"], special["sep"]}
    return [token for token in ids if token not in drop]


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    total = float(np.sum(exponent))
    return exponent / total if total > 0 else np.zeros_like(values)


def confidence(probs: np.ndarray) -> float:
    """Jev-style confidence: 1 - H(p)/log(k).

    0 means "evenly split", 1 means "certain". A model that returns one option
    for everything gets a low score here, which is the point.
    """
    if probs.size <= 1:
        return 1.0
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    return max(0.0, 1.0 - entropy / math.log(probs.size))


def render_options(kind: str, criteria: Any) -> List[str]:
    """Render option texts exactly as the reference implementation does."""
    if kind == "score":
        return ["level %d: %s" % (i + 1, str(item).strip()) for i, item in enumerate(criteria or [])]
    if kind == "noul":
        # Fixed two-way decision; index 1 carries the "noul" probability.
        return ["false", "true"]

    options: List[str] = []
    for key, value in (criteria or {}).items():
        if isinstance(value, str) and value.strip():
            options.append("%s: %s" % (key, value.strip()))
        else:
            options.append(str(key))
    return options


def build_sequence(
    encode,
    special: Dict[str, int],
    qtype_name: str,
    instructions: str,
    option_texts: Sequence[str],
    state: str,
    max_len: int,
    head_max_len: int,
    mask_token: str = MASK_TOKEN,
) -> Tuple[List[int], List[int]]:
    """Build ``input_ids`` and ``marker_pos`` for one request.

    Mirrors ``buildSequence`` in the reference implementation: the head and the
    options share ``head_max_len``, and when the options do not fit they are
    shortened together so the instruction survives.

    Every span goes through `scrub` (string) **and** `scrub_ids` (id), so neither
    a literal marker nor a marker that only appears after tokenisation can shift
    the marker positions, and a separator cannot be injected into the structure.
    """
    def span(text: str) -> List[int]:
        return scrub_ids(encode(scrub(text, mask_token)), special)

    head = span("%s question: %s" % (qtype_name, instructions))
    options = [[special["mask"], *span(" " + text)[:OPTION_TOKEN_CAP]] for text in option_texts]

    budget = head_max_len - sum(len(option) for option in options)
    if budget < 16 and options:
        per_option = max(4, (head_max_len - 16) // max(1, len(options)))
        options = [option[:per_option] for option in options]
        budget = head_max_len - sum(len(option) for option in options)
    head = head[: max(8, budget)]

    sequence: List[int] = [special["cls"], *head, special["sep"]]
    markers: List[int] = []
    for option in options:
        markers.append(len(sequence))
        sequence.extend(option)

    sequence.append(special["sep"])
    room = max(0, max_len - len(sequence) - 1)
    sequence.extend(span(state)[:room])
    sequence.append(special["sep"])

    sequence = sequence[:max_len]
    return sequence, [marker for marker in markers if marker < max_len]


class LayaDecider:
    """Loads the Laya ONNX checkpoint and answers typed decisions.

    The session is built on first use: the fp32 checkpoint is ~1.7 GB, and a
    parse-only workload should not pay for it.
    """

    def __init__(self, directory: str = "", providers: Optional[Sequence[str]] = None) -> None:
        self.directory = directory or model_dir()
        self.providers = list(providers) if providers else None
        self.threads = _env_int("FREERAG_LAYA_THREADS", DEFAULT_CPU_THREADS)
        self._session: Any = None
        self._tokenizer: Any = None
        self._config: Dict[str, Any] = {}
        # Resolved lazily from the checkpoint's own declarations; see
        # `resolve_specials` for why they are never hard-coded here.
        self._special: Optional[Dict[str, int]] = None
        self._mask_token: str = MASK_TOKEN
        # Guards the one-time load. The sidecar answers requests on a thread pool
        # (parse_server.serve), so two decisions can arrive before the checkpoint
        # is loaded, and an unguarded `if self._session is None` would build the
        # session twice — 1.2 GB twice over.
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return os.path.isfile(os.path.join(self.directory, "laya.onnx")) and os.path.isfile(
            os.path.join(self.directory, "tokenizer", "tokenizer.json")
        )

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            # Second look inside the lock: the other thread may have built it
            # while this one waited.
            if self._session is not None:
                return

            from tokenizers import Tokenizer

            if not self.available:
                raise FileNotFoundError("Laya ONNX checkpoint not found at %s" % self.directory)

            self._tokenizer = Tokenizer.from_file(
                os.path.join(self.directory, "tokenizer", "tokenizer.json")
            )
            config_path = os.path.join(self.directory, "laya_config.json")
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as handle:
                    self._config = json.load(handle)

            # CPU by default: see DEFAULT_CPU_THREADS — an accelerator costs more
            # in per-call graph upload than it saves on a handful of tokens.
            providers = self.providers or ["CPUExecutionProvider"]
            # The session is registered per PROCESS, not per instance, and built
            # under that registry's own lock (layout_onnx.session_for). Two
            # LayaDecider instances — or two threads — then share one session:
            # the weights are resident once, and a second build cannot happen.
            self._session = session_for(
                os.path.join(self.directory, "laya.onnx"),
                providers,
                self.threads,
                build=lambda: self._build(providers),
            )
            # Resolved here rather than lazily on first use: _specials writes two
            # attributes, and two worker threads resolving at once is the same
            # duplicated work the lock above exists to prevent.
            self._specials()

    def _build(self, providers: Sequence[str]) -> Any:
        """Build the inference session. Runs under the session registry's lock."""
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if providers[0] == "CPUExecutionProvider":
            options.intra_op_num_threads = self.threads
        return onnxruntime.InferenceSession(
            os.path.join(self.directory, "laya.onnx"),
            sess_options=options,
            providers=list(providers),
        )

    def _encode(self, text: str) -> List[int]:
        """Encode without special tokens — the template inserts them itself."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def _specials(self) -> Tuple[Dict[str, int], str]:
        """Resolve the special ids and mask token once per checkpoint.

        Never from a literal token name — see `resolve_specials`, which explains
        both the mmBERT `[MASK]` → unk trap and the `config.json` vs
        `tokenizer_config.json` disagreement.
        """
        if self._special is None:
            self._special, self._mask_token = resolve_specials(
                self._tokenizer, self._config, self.directory
            )
            LOGGER.info("Laya: special_ids=%s mask_token=%r", self._special, self._mask_token)
        return self._special, self._mask_token

    def _temperature(self, qtype_name: str, option_count: int) -> float:
        by_options = self._config.get("temperature_by_options") or {}
        key = "%s:%s" % (qtype_name, size_bucket(option_count))
        if key in by_options:
            return float(by_options[key])

        table = self._config.get("temperature")
        index = QTYPES.get(qtype_name, 0)
        if isinstance(table, list) and index < len(table):
            return float(table[index])
        return 1.0

    def decide(
        self,
        instructions: str,
        criteria: Any,
        *,
        state: str = "",
        qtype: str = "choice",
    ) -> Dict[str, Any]:
        """Answer one typed decision.

        ``criteria`` is a mapping of option -> description for ``choice``, a list
        of levels for ``score``, and ignored for ``noul``.
        """
        return self.decide_many([{
            "instructions": instructions,
            "criteria": criteria,
            "state": state,
            "qtype": qtype,
        }])[0]

    def decide_many(self, requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answer several typed decisions, scoring them in ONE forward pass.

        MEASURED SLOWER. Do not wire this up as an optimisation on this machine:
        scripts/bench_laya_batch.py, against the labelled test set, three runs
        alternating, best of each:

            rows   one-at-a-time   batched    speed
               1        0.27s       0.28s    0.97x   (the same code path — the
               2        0.68s       0.81s    0.84x    noise floor, about 20%)
               4        1.43s       2.21s    0.65x
               8        2.66s       4.72s    0.56x

        The reasoning that motivated it was that a 1.2 GB fp32 checkpoint must
        stream its weights every pass, so one pass over several rows would
        amortise that. The numbers say the cost is per TOKEN, not per pass:
        sequentially 2,202 tokens cost 1.21 ms each, batched the same rows cost
        2.03 ms each — 1.7x more, because every row is padded to the longest and
        attention is quadratic in that width, and because a batch's activations
        are larger than one row's. A batch of one is exactly the old single-row
        work, which is why rows=1 measures 0.97x.

        Kept for two reasons: `decide` routes through it, so the assembly and
        the scoring exist in one place instead of two; and it is the exact
        equivalence reference — every batched choice matches the unbatched one
        and the probabilities agree to 0.0000, so the padding is provably
        invisible. If a machine ever appears where batching pays, this is ready
        and correct; on this one it is a slower path and is not used as one.
        """
        if not requests:
            return []
        self._load()

        prepared = []
        for request in requests:
            qtype = request.get("qtype") or "choice"
            if qtype not in QTYPES:
                raise ValueError("unknown question type %r" % qtype)
            ids, markers, option_texts = self._prepare(
                request.get("instructions") or "",
                request.get("criteria"),
                request.get("state") or "",
                qtype,
            )
            prepared.append((ids, markers, option_texts, qtype))

        rows_count = len(prepared)
        options = max(len(item[2]) for item in prepared)
        width = max(len(item[0]) for item in prepared)
        pad = self._specials()[0]["pad"]

        input_ids = np.full((rows_count, width), pad, dtype=np.int64)
        attention = np.zeros((rows_count, width), dtype=np.int64)
        marker_pos = np.zeros((rows_count, options), dtype=np.int64)
        marker_mask = np.zeros((rows_count, options), dtype=bool)
        qtypes = np.zeros((rows_count,), dtype=np.int64)

        for row, (ids, markers, option_texts, qtype) in enumerate(prepared):
            input_ids[row, :len(ids)] = ids
            attention[row, :len(ids)] = 1
            for column, position in enumerate(markers[:options]):
                marker_pos[row, column] = position
                marker_mask[row, column] = True
            qtypes[row] = QTYPES[qtype]

        logits, act_probs = self._session.run(["logits", "act_probs"], {
            "input_ids": input_ids,
            "attention_mask": attention,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtypes,
        })

        scores = np.asarray(logits, dtype=np.float64).reshape(rows_count, -1)
        acts = np.asarray(act_probs, dtype=np.float64).reshape(rows_count, -1)
        return [
            self._finish(scores[row], marker_mask[row], option_texts, qtype, acts[row], options)
            for row, (_ids, _markers, option_texts, qtype) in enumerate(prepared)
        ]

    def _prepare(self, instructions: str, criteria: Any, state: str, qtype: str) -> Tuple[List[int], List[int], List[str]]:
        """Assemble one decision's token sequence.

        Shared by every path so that a batched row and a lone row are assembled
        by the same code — a second assembly would only prove it agrees with
        itself (DEPLOY_CONTRACT.md §六).
        """
        option_texts = render_options(qtype, criteria)
        if len(option_texts) < 2:
            raise ValueError("a decision needs at least two options")

        max_len = int(self._config.get("max_len") or 512)
        head_max_len = int(self._config.get("head_max_len") or 192)
        special, mask_token = self._specials()
        ids, markers = build_sequence(
            self._encode,
            special,
            qtype,
            instructions,
            option_texts,
            state,
            max_len,
            head_max_len,
            mask_token=mask_token,
        )
        return ids, markers, option_texts

    def _finish(
        self,
        raw_row: Any,
        mask_row: Any,
        option_texts: List[str],
        qtype: str,
        act_row: Any,
        width: int,
    ) -> Dict[str, Any]:
        """Turn one row's logits into a decision.

        ``width`` is the row's padded option count, which is what the logits and
        the mask are shaped by; ``option_texts`` is the row's own list, which is
        what the answer is reported in terms of.
        """
        count = len(option_texts)
        scores = np.asarray(raw_row, dtype=np.float64).reshape(-1)[:width]
        # Masked slots arrive as -1e4; honour the mask rather than trusting that.
        valid = np.asarray(mask_row, dtype=bool)[:width]
        scores = np.where(valid, scores, -np.inf)

        temperature = self._temperature(qtype, count)
        scaled = scores / temperature if temperature > 0 else scores
        probs = softmax(np.asarray(scaled, dtype=np.float64))
        # `probability` 也用**加了温度**的分布。
        #
        # 原来这里算的是未加温度的原始 softmax，而唯一读它的地方
        # （cmd/freerag 的 LayaChecker 闭包 → decision.Probability → 与
        # DefaultLayaMinProbability = 0.6 比较）正靠它做"够不够"的门控判定
        # —— 等于 `scripts/fit_temperature.py` 的全部校准工作被静默绕过，
        # 落在了没人读的 `confidence` 上。
        #
        # 改后：argmax 不受影响（T > 0 时除以正常数单调），但暴露出去的
        # 概率是校准过的，与 `confidence` 同源一致。
        probability = probs

        index = int(np.argmax(scores)) if np.isfinite(scores).any() else 0
        return {
            "qtype": qtype,
            "index": index,
            "choice": option_texts[index],
            "options": option_texts,
            "confidence": round(confidence(probs), 4),
            "probability": round(float(probability[index]), 4),
            "scores": [round(float(value), 4) for value in scores[:count]],
            "temperature": temperature,
            "act_probs": [round(float(value), 4) for value in np.asarray(act_row).reshape(-1)],
        }
