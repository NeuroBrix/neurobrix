"""A served generation whose request and container declare no max_tokens is bounded by the session's reach left
after the prompt (Devstral, Qwen3.8): neither refused nor given an engine number (orchestrator 23/11 09:19)."""
import pytest

from neurobrix.core.runtime.prompt_length import ContextLengthExceeded, decode_bound


def test_a_declared_bound_is_kept():
    assert decode_bound(100, 4096, "the card", 256) == 256


def test_no_declared_bound_takes_the_reach_left_after_the_prompt():
    assert decode_bound(100, 4096, "the card") == 3996


def test_a_prompt_that_fills_the_reach_is_refused_by_name():
    with pytest.raises(ContextLengthExceeded, match="leaves none to generate"):
        decode_bound(4096, 4096, "the card")


def test_the_served_engine_supplies_no_number_of_its_own():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[3] / "src/neurobrix/serving/engine.py").read_text()
    body = src.split("def _max_tokens(", 1)[1].split("\n    def ", 1)[0]
    assert "_resolve_runtime" not in body and "default" not in body.replace("defaults", "")
