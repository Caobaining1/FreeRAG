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
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Question type -> qtype id, from the reference implementation's QTYPES.
QTYPES = {"choice": 0, "score": 1, "noul": 2}

#: Longest option text, in tokens (reference implementation caps at 48).
OPTION_TOKEN_CAP = 48

#: The literal marker token; scrubbed from user text so content cannot forge one.
MASK_TOKEN = "[MASK]"

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


def scrub(text: str) -> str:
    """Remove marker tokens from untrusted text.

    Without this, a document could embed ``[MASK]`` and shift the marker
    positions the model reads its decision from.
    """
    return text.replace(MASK_TOKEN, "")


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
) -> Tuple[List[int], List[int]]:
    """Build ``input_ids`` and ``marker_pos`` for one request.

    Mirrors ``buildSequence`` in the reference implementation: the head and the
    options share ``head_max_len``, and when the options do not fit they are
    shortened together so the instruction survives.
    """
    head = encode("%s question: %s" % (qtype_name, scrub(instructions)))
    options = [
        [special["mask"], *encode(" " + scrub(text))[:OPTION_TOKEN_CAP]] for text in option_texts
    ]

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
    sequence.extend(encode(scrub(state))[:room])
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

    @property
    def available(self) -> bool:
        return os.path.isfile(os.path.join(self.directory, "laya.onnx")) and os.path.isfile(
            os.path.join(self.directory, "tokenizer", "tokenizer.json")
        )

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime
        from tokenizers import Tokenizer

        if not self.available:
            raise FileNotFoundError("Laya ONNX checkpoint not found at %s" % self.directory)

        self._tokenizer = Tokenizer.from_file(os.path.join(self.directory, "tokenizer", "tokenizer.json"))
        config_path = os.path.join(self.directory, "laya_config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as handle:
                self._config = json.load(handle)

        # CPU by default: see DEFAULT_CPU_THREADS — an accelerator costs more in
        # per-call graph upload than it saves on a handful of tokens.
        providers = self.providers or ["CPUExecutionProvider"]
        options = onnxruntime.SessionOptions()
        if providers[0] == "CPUExecutionProvider":
            options.intra_op_num_threads = self.threads
        self._session = onnxruntime.InferenceSession(
            os.path.join(self.directory, "laya.onnx"), sess_options=options, providers=providers
        )

    def _encode(self, text: str) -> List[int]:
        """Encode without special tokens — the template inserts them itself."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def _special_ids(self) -> Dict[str, int]:
        tokenizer = self._tokenizer
        ids = {}
        for name, token in (("cls", "[CLS]"), ("sep", "[SEP]"), ("mask", "[MASK]"), ("pad", "[PAD]")):
            value = tokenizer.token_to_id(token)
            if value is None:
                raise ValueError("tokenizer has no %s token" % token)
            ids[name] = int(value)
        return ids

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
        if qtype not in QTYPES:
            raise ValueError("unknown question type %r" % qtype)

        self._load()
        option_texts = render_options(qtype, criteria)
        if len(option_texts) < 2:
            raise ValueError("a decision needs at least two options")

        max_len = int(self._config.get("max_len") or 512)
        head_max_len = int(self._config.get("head_max_len") or 192)

        ids, markers = build_sequence(
            self._encode,
            self._special_ids(),
            qtype,
            instructions,
            option_texts,
            state,
            max_len,
            head_max_len,
        )

        count = len(option_texts)
        marker_pos = np.zeros((1, count), dtype=np.int64)
        marker_mask = np.zeros((1, count), dtype=bool)
        for index, position in enumerate(markers[:count]):
            marker_pos[0, index] = position
            marker_mask[0, index] = True

        inputs = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": np.asarray([QTYPES[qtype]], dtype=np.int64),
        }
        logits, act_probs = self._session.run(["logits", "act_probs"], inputs)

        scores = np.asarray(logits, dtype=np.float64).reshape(-1)[:count]
        # Masked slots arrive as -1e4; honour the mask rather than trusting that.
        valid = np.asarray(marker_mask[0], dtype=bool)[:count]
        scores = np.where(valid, scores, -np.inf)

        temperature = self._temperature(qtype, count)
        scaled = scores / temperature if temperature > 0 else scores
        probs = softmax(np.asarray(scaled, dtype=np.float64))
        probability = softmax(np.asarray(scores, dtype=np.float64))

        index = int(np.argmax(scores)) if np.isfinite(scores).any() else 0
        return {
            "qtype": qtype,
            "index": index,
            "choice": option_texts[index],
            "options": option_texts,
            "confidence": round(confidence(probs), 4),
            "probability": round(float(probability[index]), 4),
            "scores": [round(float(value), 4) for value in scores],
            "temperature": temperature,
            "act_probs": [round(float(value), 4) for value in np.asarray(act_probs).reshape(-1)],
        }
