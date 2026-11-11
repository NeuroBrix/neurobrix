"""tools/gate_serve_cell.py: the verdict of three successive requests to one daemon.

What this test would do if the code were wrong: a third request 2.5 times slower than the first judged GREEN — the
defect of 2026-11-11 (0.08 s then 0.20 s a token) passing the gate; a failed or empty request judged GREEN."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gate_serve_cell", Path(__file__).resolve().parents[3] / "tools" / "gate_serve_cell.py")
cell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cell)


def _r(gap, **kw):
    return {"first_s": 3.0, "chunks": 50, "median_gap_s": gap, "text_sha": "ab12", **kw}


def test_the_third_request_at_the_firsts_speed_is_green():
    assert cell.judge([_r(0.08), _r(0.09), _r(0.10)])[0] == "GREEN"


def test_a_slower_third_request_is_red_and_says_by_how_much():
    verdict, reason = cell.judge([_r(0.08), _r(0.20), _r(0.20)])
    assert verdict == "RED" and "2.50x" in reason


def test_the_second_request_alone_slow_is_not_the_criterion_but_a_failed_one_is_red():
    assert cell.judge([_r(0.08), _r(0.30), _r(0.09)])[0] == "GREEN"       # the rule reads the third against the first
    assert cell.judge([_r(0.08), {"error": "URLError: refused", "chunks": 0}])[0] == "RED"
    assert cell.judge([_r(0.08), _r(0.08), _r(0.08, error="timeout")])[0] == "RED"


def test_an_empty_answer_is_red():
    assert cell.judge([_r(0.08), _r(0.08), _r(0.08, text_sha=None)])[0] == "RED"
    assert cell.judge([_r(0.08), _r(0.08), _r(0.08, chunks=1)])[0] == "RED"
    assert cell.judge([_r(0.0), _r(0.08), _r(0.08)])[0] == "RED"
