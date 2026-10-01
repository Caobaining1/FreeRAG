"""VLM figure understanding — PARSE TIME ONLY (docs/plan.md §5.1 step 3).

A ``Figure`` region has no text layer, so today it becomes a chunk whose body is
whatever its caption says. This module gives the figure its own words: the page
region is cropped, handed to a vision model over Ollama, and the description
replaces the empty body — so the figure participates in retrieval like any other
passage instead of being reachable only through its caption.

**Never called on the query path.** The model is loaded for an index run and
unloaded when that run ends (the kernel does the unload, because "one index run"
is a job the sidecar does not see). Keeping it out of `ask` is deliberate: the
answering model and a 3.5 GB vision model must not be resident at once on this
class of machine (16 GB, measured swapping already).

The prompt is the one benchmarked in ``scripts/bench_vlm.py`` — kept identical
because what it forbids is the whole point: a misread number that lands in the
index is invisible afterwards, so the model is told not to quote values at all.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence

LOGGER = logging.getLogger(__name__)

#: Ollama's default address. Overridable so a dedicated instance can be used.
DEFAULT_VLM_URL = "http://127.0.0.1:11434"

#: How long the model stays resident after a call.
#:
#: Long enough to survive a whole batch (so it is not reloaded per document) and
#: short enough not to linger if the kernel dies mid-run. The kernel unloads it
#: explicitly when the index job ends, so this is only a backstop.
DEFAULT_KEEP_ALIVE = "30m"

#: Per-call ceiling. A cold 3.5 GB load plus one image is well inside this; the
#: value exists so a wedged server fails the figure rather than the whole parse.
DEFAULT_TIMEOUT = 600

#: Ceiling on the description, in tokens. The prompt asks for 2-4 sentences, so
#: this is generous rather than tight — it exists so a model that starts rambling
#: cannot spend minutes on one figure.
DEFAULT_MAX_PREDICT = 256

#: Reasoning is requested OFF — but do NOT rely on this to save a reasoning
#: model, because Ollama does not always honour it.
#:
#: Measured on Ollama 0.34.4 with `qwen3-vl:4b`: `"think": false` is ignored.
#: The reply still carries a `thinking` field (466 chars on one crop) and an
#: empty `content`, because reasoning is generated against the same
#: `num_predict` as the description and consumes all of it. The result is a
#: figure that costs ~56 s and yields no text at all.
#:
#: A non-reasoning model is the real fix: `qwen2.5vl:3b` measured ~22 s/figure
#: with no `thinking` and a usable description (see README, 解析期视觉). The
#: field is still sent because it is correct where it is honoured, and because
#: `describe` now warns loudly when a reasoning-only reply comes back.
VLM_THINK = False

#: Figure crop resolution.
#:
#: Measured on the same flow-chart crop, `qwen2.5vl:3b`: 25.7 s at 150 dpi
#: (952x353 px) against 21.8 s at 100 dpi (636x235 px), with the description just
#: as usable. Vision cost scales with image tokens, so the lower value is not a
#: quality trade — it is the same answer for 15% less time.
DEFAULT_FIGURE_DPI = 100

#: Crops smaller than this (in PDF points) are skipped — a few points of a rule
#: or an arrow is nothing to describe, and each one costs a model call.
MIN_FIGURE_POINTS = 24

#: Ceiling on figures per document, so one pathological file cannot turn a parse
#: into a hundred model calls.
DEFAULT_MAX_FIGURES = 12

#: The prompt this feature ships with — verbatim from scripts/bench_vlm.py.
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


class VlmCaptioner:
    """Calls a vision model on Ollama to describe one figure image."""

    def __init__(
        self,
        model: str,
        url: str = "",
        *,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        self.model = (model or "").strip()
        self.url = (url or DEFAULT_VLM_URL).rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    @property
    def available(self) -> bool:
        """Whether a model name was configured.

        Reachability is not probed here: a machine without Ollama running should
        still parse (the figures simply keep their caption), and probing would
        add a round trip to every parse.
        """
        return bool(self.model)

    def describe(self, png: bytes) -> str:
        """Return the model's description of one PNG image.

        Raises on failure; ``caption_figures`` catches, so one unreadable figure
        never fails a document.
        """
        if not self.model:
            raise ValueError("vlm: no model configured")

        payload = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(png).decode("ascii")],
            }],
            "stream": False,
            "keep_alive": self.keep_alive,
            # Sent explicitly rather than left to the server: see VLM_THINK.
            "think": VLM_THINK,
            "options": {
                # Deterministic: two index runs over the same figure must produce
                # the same chunk, or the index is not reproducible.
                "temperature": 0,
                "seed": 7,
                "num_predict": DEFAULT_MAX_PREDICT,
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            self.url + "/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)

        message = body.get("message") or {}
        content = (message.get("content") or "").strip()
        if content:
            return content

        # An empty reply that carries reasoning is the budget being eaten: a
        # reasoning model generates its thinking against the same num_predict as
        # the answer, so once thinking fills the cap there is no answer left.
        # Named here because returning "" would make it look like the model
        # simply had nothing to say, and the figure would silently get no text.
        reasoning = message.get("thinking") or ""
        if reasoning:
            LOGGER.warning(
                "vlm: %s produced %d char(s) of reasoning and no description — "
                "num_predict was consumed by reasoning. Use a non-reasoning "
                "vision model (measured: qwen2.5vl:3b, ~22 s/figure).",
                self.model, len(reasoning))
        else:
            LOGGER.warning("vlm: %s returned an empty description for one figure", self.model)
        return ""

    def unload(self) -> bool:
        """Ask Ollama to drop the model now. Returns whether the call succeeded.

        ``keep_alive: 0`` is Ollama's documented unload. Failures are reported,
        not raised: this runs on the way out of an index job, where nothing is
        left to do about it — and the model expires on its own regardless.
        """
        if not self.model:
            return False
        payload = json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False


def caption_figures(
    path: str,
    blocks: Sequence[Any],
    captioner: VlmCaptioner,
    *,
    dpi: int = DEFAULT_FIGURE_DPI,
    max_figures: int = DEFAULT_MAX_FIGURES,
) -> int:
    """Give every ``Figure`` block a text body, in place. Returns how many.

    Only Figure regions are touched. A figure's caption is already a chunk of
    its own and gets merged onto the figure by the chunker (see chunking.py), so
    the description lands alongside the caption rather than replacing it.

    Every failure is local: a figure that cannot be cropped, or a call that
    errors or times out, leaves that block exactly as it was.
    """
    if not captioner.available or not blocks:
        return 0

    figures = [
        block for block in blocks
        if getattr(block, "block_type", "") == "Figure"
        and block.width >= MIN_FIGURE_POINTS
        and block.height >= MIN_FIGURE_POINTS
        and block.page_num > 0
    ]
    if not figures:
        return 0

    import pymupdf  # imported lazily: the module must import without the wheel

    described = 0
    doc = pymupdf.open(path)
    try:
        for block in figures:
            if described >= max_figures:
                break
            if block.page_num > len(doc):
                continue
            page = doc[block.page_num - 1]
            x0, y0, x1, y1 = (float(value) for value in block.bbox)
            try:
                pixmap = page.get_pixmap(clip=pymupdf.Rect(x0, y0, x1, y1), dpi=dpi)
                png = pixmap.tobytes("png")
            except Exception:
                continue
            try:
                text = captioner.describe(png)
            except Exception:
                continue
            if not text:
                continue
            existing = (block.text or "").strip()
            block.text = f"{existing}\n{text}" if existing else text
            described += 1
    finally:
        doc.close()

    return described
