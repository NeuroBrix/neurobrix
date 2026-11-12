"""tools/gate_serve_cell.py: a thinking model's reasoning is text the cell times and compares.

The server streams a thinking model's text between its markers as `reasoning_content` (serving/openai_api/turn.py). The
cell read `content` alone, so Qwen3-30B-A3B-Thinking-2507 served on two V100-32GB answered "request 1 returned 0
chunk(s): no text to time" — its first 128 tokens were all reasoning (2026-11-12).

What this test would do if the code were wrong: a delta or a message carrying only reasoning reads as empty."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gate_serve_cell", Path(__file__).resolve().parents[3] / "tools" / "gate_serve_cell.py")
cell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cell)


def test_reasoning_alone_is_text():
    assert cell.spoken({"reasoning_content": "Let me think"}) == "Let me think"


def test_reasoning_then_answer_and_an_answer_alone():
    assert cell.spoken({"reasoning_content": "why ", "content": "because"}) == "why because"
    assert cell.spoken({"content": "Paris"}) == "Paris"
    assert cell.spoken({"role": "assistant"}) == ""
