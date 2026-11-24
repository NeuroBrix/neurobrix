"""`neurobrix run --turns` on an audio_llm container (granite-speech, Voxtral, canary): one refusal, by name, first.

Before (the V100, 2026-11-24): the command line's two gates contradicted for audio_llm. Without --prompt:
"ZERO FALLBACK: family 'audio_llm' requires --prompt" (config/families/audio_llm.yml `inputs.required`); with
--prompt beside --turns: "--turns plays its own messages; --prompt beside it would be ignored". A user could neither
give the prompt nor omit it. The decision read from the code: a session of turns is played through
`serving.session.play_turn`, which hands the executor `global.input_token_ids` and a `kv_session`; the executor
refuses a session's turn for any flow but `autoregressive_generation` (core/runtime/executor.py), and the audio_llm
flow (core/flow/audio_llm.py, triton/flow/audio_llm.py) builds one context — prefix ids, audio embeddings, suffix
ids — and decodes it once. Multi-turn is not an audio_llm path: chat is for language models, and a family whose YAML
names no --turns among its inputs refuses it by name, before the family's required inputs are read
(core/runtime/output_dispatch.validate_required_inputs).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neurobrix.cli.commands.run import load_turns
from neurobrix.core.runtime.output_dispatch import validate_required_inputs


def _args(**kw):
    """An argparse-like namespace with the run's input flags, None unless given."""
    base = dict(prompt=None, turns=None, audio=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _turns_file(tmp_path):
    """A valid --turns file of two user messages."""
    p = tmp_path / "turns.json"
    p.write_text(json.dumps(["hello", "and then?"]))
    return str(p)


@pytest.mark.parametrize("prompt", [None, "transcribe"])
def test_audio_llm_with_turns_is_one_refusal_by_name(tmp_path, prompt):
    args = _args(turns=_turns_file(tmp_path), prompt=prompt, audio="a.wav")
    load_turns(args)                                   # the file is read; the family is not known yet
    with pytest.raises(RuntimeError) as exc:
        validate_required_inputs("audio_llm", args)
    msg = str(exc.value)
    assert "--turns is the chat path of a language model" in msg
    assert "audio_llm" in msg and "--audio" in msg and "--prompt" in msg
    assert "beside it would be ignored" not in msg


def test_audio_llm_without_turns_and_without_prompt_still_requires_prompt():
    args = _args(audio="a.wav")
    with pytest.raises(RuntimeError, match="requires --prompt"):
        validate_required_inputs("audio_llm", args)


def test_a_text_llm_with_turns_and_prompt_is_unchanged(tmp_path):
    args = _args(turns=_turns_file(tmp_path), prompt="x")
    assert load_turns(args) == ["hello", "and then?"]
    with pytest.raises(RuntimeError, match="--turns plays its own messages; --prompt beside it would be ignored"):
        validate_required_inputs("llm", args)
    validate_required_inputs("llm", _args(turns=_turns_file(tmp_path)))
