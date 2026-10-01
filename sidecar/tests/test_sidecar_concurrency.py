"""Concurrency of the sidecar: two lanes, one shared Laya session.

The kernel multiplexes by request id (internal/sidecar/client.go's readLoop), so
answers may come back in any order — which is what makes it possible for the
sidecar to work on several at once instead of one after another. These tests pin
the two properties that matter: a decision is not queued behind a parse, and the
heavy lane stays serial.
"""
import io
import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parse_server  # noqa: E402
from laya import LayaDecider  # noqa: E402
from layout_onnx import reset_session_cache  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYA_DIR = os.path.join(REPO, "models", "laya-onnx")


def request(identifier, method, params=None):
    return json.dumps({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params or {}})


def run_serve(lines):
    """Feed lines through serve() and return the decoded responses, in order."""
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    stdout = io.StringIO()
    parse_server.serve(stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


class ServeLanesTest(unittest.TestCase):
    def test_a_decision_is_not_queued_behind_a_parse(self):
        # The two methods hold each other: the slow one waits for the fast one
        # to start. If both run on one queue the slow one cannot be released
        # until it has already finished, so the fast answer cannot arrive first
        # and this test fails on the ORDER rather than on a timeout.
        started = threading.Event()
        release = threading.Event()

        def heavy(_params):
            started.set()
            release.wait(5)
            return {"done": "heavy"}

        def light(_params):
            self.assertTrue(started.wait(5), "the heavy request never started")
            release.set()
            return {"done": "light"}

        with mock.patch.dict(parse_server.METHODS, {"heavy": heavy, "light": light}), \
                mock.patch.object(parse_server, "_PARSE_LANE", frozenset({"heavy"})):
            responses = run_serve([request(1, "heavy"), request(2, "light")])

        self.assertEqual([r["id"] for r in responses], [2, 1],
                         "the light request was answered after the heavy one, so it waited for it")

    def test_the_parse_lane_stays_serial(self):
        # A parse holds hundreds of MB of activations, so several at once cost
        # memory and throughput both (measured: four concurrent parses took four
        # times the wall time of one). The lane is the budget.
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def busy(_params):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.1)
            with lock:
                state["active"] -= 1
            return {"ok": True}

        with mock.patch.dict(parse_server.METHODS, {"parse": busy}):
            responses = run_serve([request(i, "parse") for i in range(4)])

        self.assertEqual(len(responses), 4)
        self.assertEqual(state["peak"], 1, "the parse lane ran %d request(s) at once" % state["peak"])

    def test_the_decision_lane_is_bounded(self):
        # Bounded, not unlimited: the budget is a decision, so it is asserted
        # rather than left to whatever the pool happens to allow.
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def busy(_params):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.1)
            with lock:
                state["active"] -= 1
            return {"ok": True}

        with mock.patch.dict(parse_server.METHODS, {"decide": busy}):
            responses = run_serve([request(i, "decide") for i in range(6)])

        self.assertEqual(len(responses), 6)
        self.assertEqual(state["peak"], min(6, parse_server.DECIDE_THREADS))

    def test_every_response_is_a_whole_line(self):
        # Responses are written by several threads; the write is locked, and an
        # unlocked one would splice two answers into a line no reader can parse.
        def echo(params):
            return {"pad": (params or {}).get("pad", "")}

        with mock.patch.dict(parse_server.METHODS, {"ping": echo}):
            lines = [request(i, "ping", {"pad": "x" * 400}) for i in range(12)]
            responses = run_serve(lines)

        self.assertEqual(len(responses), 12)
        self.assertEqual(sorted(r["id"] for r in responses), list(range(12)))
        for response in responses:
            self.assertEqual(len(response["result"]["pad"]), 400)


@unittest.skipUnless(os.path.isdir(LAYA_DIR), "Laya checkpoint not downloaded")
class LayaSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_session_cache()
        cls.decider = LayaDecider(directory=LAYA_DIR)
        cls.decider._load()

    def test_two_deciders_share_one_session(self):
        # A session is 1.2 GB. The second decider must reuse the first rather
        # than build its own — the registry is keyed by (model, providers,
        # threads), not held on the instance.
        other = LayaDecider(directory=LAYA_DIR)
        other._load()
        self.assertIs(other._session, self.decider._session)

    def test_concurrent_decisions_share_the_session_safely(self):
        # The thread pool relies on onnxruntime.Run being thread-safe; this is
        # the assumption, verified rather than trusted.
        decider = self.decider
        answers = []
        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker():
            try:
                barrier.wait(5)
                result = decider.decide(
                    "Does the provided evidence sufficiently answer the question: When?",
                    {"true": "yes", "false": "no"},
                    state="It happened in 2019.",
                    qtype="noul",
                )
                with lock:
                    answers.append(result["choice"])
            except Exception as failure:  # noqa: BLE001
                with lock:
                    errors.append("%s: %s" % (type(failure).__name__, failure))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(answers), 4)
        self.assertEqual(len(set(answers)), 1, "the same input gave different answers: %s" % answers)


if __name__ == "__main__":
    unittest.main()
