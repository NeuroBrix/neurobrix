"""A vlm container answers a plain text request, and hands its language model the positions its graph declares.

The vlm flow had one text-only form, behind `--mode audio` on a speech build: a text request on any other vlm
container was refused, though the vendor's processor serves text with no image. And the legacy flow built the
three M-RoPE planes whatever the LM graph declared, so an LM that takes `position_ids [B, S]` received `[3, 1, S]`.
Both are read from data now, once, in `shared_steps.VLMSteps`, and both branches call it.
"""
from pathlib import Path

import pytest

from neurobrix.core.flow.shared_steps import VLMSteps

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_the_position_form_is_the_rank_the_lm_graph_declares():
    assert VLMSteps._lm_position_rank({"input::position_ids": {"shape": ["B", "S"]}}) == 2
    assert VLMSteps._lm_position_rank({"input::position_ids": {"shape": [3, "B", "S"]}}) == 3


@pytest.mark.parametrize("tensors", [{}, {"input::position_ids": {"shape": ["S"]}},
                                     {"input::position_ids": {"shape": [1, 3, "B", "S"]}}])
def test_a_graph_declaring_neither_form_is_refused_by_name(tensors):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: the vlm flow feeds the language model its positions"):
        VLMSteps._lm_position_rank(tensors)


def test_a_request_without_image_video_or_audio_is_the_text_only_form():
    assert VLMSteps._text_only_request(False, None)
    assert not VLMSteps._text_only_request(True, None)
    assert not VLMSteps._text_only_request(False, "speech.wav")


@pytest.mark.parametrize("branch", ["core/flow/vlm.py", "triton/flow/vlm.py"])
def test_both_branches_read_the_shared_decision(branch):
    text = (SRC / branch).read_text()
    assert "self._text_only_request(has_visual, audio_path)" in text
    assert "self._lm_position_rank(" in text
    assert "text-only requests are accepted only" not in text


def test_the_command_line_asks_a_vlm_request_for_its_prompt_alone():
    from argparse import Namespace
    from neurobrix.core.runtime.output_dispatch import validate_required_inputs
    validate_required_inputs("vlm", Namespace(prompt="Write a function.", input_image=None))
    with pytest.raises(RuntimeError, match="requires --prompt"):
        validate_required_inputs("vlm", Namespace(prompt=None, input_image="cat.png"))
