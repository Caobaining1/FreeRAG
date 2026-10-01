"""Unit tests for parse-time figure captioning (sidecar/vlm.py).

No vision model is required: the Ollama calls are exercised against a local
HTTP server, and the crop path is guarded rather than driven (it needs a real
PDF, which the pipeline integration tests cover).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm import VlmCaptioner, caption_figures  # noqa: E402


class _Recorder(BaseHTTPRequestHandler):
    """Serves one canned reply and records the request it was given."""

    reply: dict = {}
    status: int = 200
    requests: list = []

    def do_POST(self):  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.requests.append((self.path, body))
        payload = json.dumps(self.__class__.reply).encode("utf-8")
        self.send_response(self.__class__.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return  # keep the test output clean


class _Server:
    def __init__(self, reply=None, status=200):
        _Recorder.reply = reply or {}
        _Recorder.status = status
        _Recorder.requests = []
        self.httpd = HTTPServer(("127.0.0.1", 0), _Recorder)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.httpd.server_address[1]

    @property
    def requests(self):
        return _Recorder.requests

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class VlmCaptionerTest(unittest.TestCase):
    def setUp(self):
        self.server = _Server()
        self.addCleanup(self.server.close)

    def test_available_only_when_a_model_is_named(self):
        self.assertFalse(VlmCaptioner("", self.server.url).available)
        self.assertTrue(VlmCaptioner("qwen3-vl:4b", self.server.url).available)

    def test_describe_posts_the_image_and_returns_the_text(self):
        server = _Server({"message": {"content": "  一张折线图。  "}})
        self.addCleanup(server.close)

        captioner = VlmCaptioner("qwen3-vl:4b", server.url)
        text = captioner.describe(b"\x89PNG fake")
        self.assertEqual(text, "一张折线图。")

        path, body = server.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["model"], "qwen3-vl:4b")
        self.assertFalse(body["stream"])
        # Deterministic: the same figure must yield the same chunk.
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 7)
        self.assertTrue(body["messages"][0]["images"], "the image was not attached")
        # Reasoning is off and the reply is bounded. Measured: with reasoning on
        # a Qwen3-VL figure took 213 s on average, and the reasoning would land
        # in the index as the figure's text.
        self.assertIs(body["think"], False)
        self.assertGreater(body["options"]["num_predict"], 0)

    def test_a_reasoning_only_reply_is_reported_not_written(self):
        # qwen3-vl:4b ignores "think": false on Ollama 0.34.4: reasoning eats the
        # whole num_predict budget and the description comes back empty. That
        # must be loud — a silent "" would look like "the model had nothing to
        # say" while the figure quietly got no text.
        server = _Server({"message": {"content": "", "thinking": "推理" * 40}})
        self.addCleanup(server.close)

        captioner = VlmCaptioner("qwen3-vl:4b", server.url)
        with self.assertLogs("vlm", level="WARNING") as captured:
            self.assertEqual(captioner.describe(b"\x89PNG"), "")
        self.assertIn("reasoning", " ".join(captured.output).lower())

    def test_unload_asks_ollama_to_evict_now(self):
        server = _Server({"done": True})
        self.addCleanup(server.close)

        self.assertTrue(VlmCaptioner("qwen3-vl:4b", server.url).unload())
        path, body = server.requests[0]
        self.assertEqual(path, "/api/generate")
        self.assertEqual(body["model"], "qwen3-vl:4b")
        self.assertEqual(body["keep_alive"], 0)

    def test_unload_reports_failure_instead_of_raising(self):
        # It runs on the way out of an index job; there is nothing left to do
        # about a failure, and the model expires on its own keep-alive.
        captioner = VlmCaptioner("qwen3-vl:4b", "http://127.0.0.1:1")  # nothing listens
        self.assertFalse(captioner.unload())

    def test_unload_without_a_model_is_a_no_op(self):
        self.assertFalse(VlmCaptioner("", self.server.url).unload())
        self.assertEqual(self.server.requests, [])


class CaptionFiguresTest(unittest.TestCase):
    def test_no_captioner_does_nothing(self):
        blocks = [_figure()]
        self.assertEqual(caption_figures("/nonexistent.pdf", blocks, VlmCaptioner("")), 0)
        self.assertEqual(blocks[0].text, "")

    def test_no_figures_does_nothing_without_opening_the_file(self):
        # A reflowable document has no Figure regions, so this must return
        # before touching PyMuPDF — otherwise a DOCX parse would fail here.
        blocks = [SimpleNamespace(block_type="Text", width=100, height=12,
                                  page_num=1, bbox=(0, 0, 100, 12), text="body")]
        self.assertEqual(
            caption_figures("/does/not/exist.docx", blocks, VlmCaptioner("qwen3-vl:4b")), 0)


def _figure():
    return SimpleNamespace(block_type="Figure", width=200.0, height=150.0, page_num=1,
                           bbox=(10.0, 20.0, 210.0, 170.0), text="")


if __name__ == "__main__":
    unittest.main()
