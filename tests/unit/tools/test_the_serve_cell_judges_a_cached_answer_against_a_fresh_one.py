"""tools/gate_serve_cell.py: the repeated request (a prefix-cache hit) and the request after a reset are judged against
the fresh one, byte for byte, with the cached tokens the daemon reports.

What this test would do if the code were wrong: a repeated request answered differently after a cache hit judged
GREEN — the cell passing while the cache changed the answer; a missing cached_tokens counted as zero, a number
invented; a difference listed known with its proof still failing the run."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "gate_serve_cell", Path(__file__).resolve().parents[3] / "tools" / "gate_serve_cell.py")
cell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cell)

HASH = ["A", " hash", " table", " maps", " keys"]


@pytest.fixture
def daemon():
    """A fake daemon: each chat request streams the next scripted answer (its pieces, its usage)."""
    script = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pieces, usage = script.pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for p in pieces:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n".encode())
            if (body.get("stream_options") or {}).get("include_usage"):
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], script
    server.shutdown()


def _usage(cached):
    u = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    return u if cached is None else {**u, "prompt_tokens_details": {"cached_tokens": cached}}


def _run(daemon, answers, known=None):
    port, script = daemon
    script.extend(answers)
    asked = [cell.ask(port, "m", 5, 10.0) for _ in answers]
    return cell.judge_cache("m_triton", asked[:3], asked[3], known or {})


def test_four_byte_equal_answers_with_a_cache_hit_then_none_after_the_reset_are_green(daemon):
    got = _run(daemon, [(HASH, _usage(0)), (HASH, _usage(19)), (HASH, _usage(19)), (HASH, _usage(0))])
    assert got["verdict"] == "GREEN" and got["cached_tokens"] == [0, 19, 19, 0] and cell.passes(got["verdict"])


def test_a_repeated_request_answered_differently_after_a_cache_hit_is_red_at_its_first_differing_piece(daemon):
    other = HASH[:3] + [" stores", " keys"]
    got = _run(daemon, [(HASH, _usage(0)), (other, _usage(19)), (HASH, _usage(19)), (HASH, _usage(0))])
    assert got["verdict"] == "RED" and not cell.passes(got["verdict"])
    assert got["diffs"] == [{"request": 2, "cached_tokens": 19, "piece": 3, "fresh": " maps", "this": " stores",
                             "logit_margin": None}]
    assert "piece 3" in got["reason"] and "19 cached tokens" in got["reason"]


def test_a_difference_on_a_cell_listed_known_is_red_known_and_does_not_fail_the_run(daemon, tmp_path):
    (tmp_path / "proof.txt").write_text("the difference, measured\n")
    (tmp_path / "known.json").write_text(json.dumps({"m_triton": "proof.txt"}))
    known = cell.read_known(tmp_path / "known.json")
    other = HASH[:4] + [" values"]
    got = _run(daemon, [(HASH, _usage(0)), (HASH, _usage(19)), (HASH, _usage(19)), (other, _usage(0))], known)
    assert got["verdict"] == "RED-KNOWN" and cell.passes(got["verdict"]) and "proof.txt" in got["reason"]
    assert got["diffs"][0]["request"] == 4


def test_a_known_entry_without_its_proof_file_is_refused(tmp_path):
    (tmp_path / "known.json").write_text(json.dumps({"m_triton": "missing.txt"}))
    with pytest.raises(SystemExit, match="no existing proof file"):
        cell.read_known(tmp_path / "known.json")


def test_a_daemon_that_does_not_report_cached_tokens_needs_a_card_and_no_number_is_invented(daemon):
    got = _run(daemon, [(HASH, _usage(0)), (HASH, _usage(None)), (HASH, _usage(19)), (HASH, _usage(0))])
    assert got["verdict"] == "NEEDS CARD" and got["cached_tokens"] == [0, None, 19, 0]
    assert "routes.py" in got["reason"] and not cell.passes(got["verdict"])


def test_a_cache_never_hit_or_a_reset_that_kept_the_cache_is_red(daemon):
    assert _run(daemon, [(HASH, _usage(0)), (HASH, _usage(0)), (HASH, _usage(0)), (HASH, _usage(0))])["verdict"] == "RED"
    assert _run(daemon, [(HASH, _usage(0)), (HASH, _usage(19)), (HASH, _usage(19)), (HASH, _usage(19))])["verdict"] == "RED"


def test_the_table_has_at_most_ten_lines():
    r = {"cached_tokens": 0, "chunks": 5, "median_gap_s": 0.1, "text_sha": "ab"}
    row = {"cell": "m_triton", "requests": [r, r, r], "after_reset": r, "verdict": "GREEN", "reason": "x"}
    assert len(cell.table(row).splitlines()) <= 10
