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
(core/runtime/output_dispatch.validate_required_inputs). A --prompt beside --turns is the command line's alone to
refuse, in load_turns, before the container loads (a first run from a .nbx unpacks the whole container): --turns
with --prompt is refused there for every family; --turns without --prompt then meets the family gate.
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


def test_audio_llm_with_turns_is_one_refusal_by_name(tmp_path):
    args = _args(turns=_turns_file(tmp_path), audio="a.wav")
    load_turns(args)                                   # the file is read; the family is not known yet
    with pytest.raises(RuntimeError) as exc:
        validate_required_inputs("audio_llm", args)
    msg = str(exc.value)
    assert "--turns is the chat path of a language model" in msg
    assert "audio_llm" in msg and "--audio" in msg and "--prompt" in msg
    assert "beside it would be ignored" not in msg


@pytest.mark.parametrize("family", ["llm", "audio_llm", "vlm", "multimodal"])
def test_a_prompt_beside_turns_is_refused_by_the_command_line_alone(tmp_path, family):
    args = _args(turns=_turns_file(tmp_path), prompt="x")
    with pytest.raises(SystemExit, match="^ERROR: --turns plays its own messages; --prompt beside it would be ignored"):
        load_turns(args)


def test_the_prompt_beside_turns_refusal_comes_before_the_container_loads(tmp_path, monkeypatch):
    """A first run from a .nbx unpacks the whole container: minutes before an error the command line alone decides."""
    import neurobrix.cli.commands.run as run_mod
    import neurobrix.cli.utils as cli_utils
    import neurobrix.cli.engine_choice as engine_choice
    from neurobrix.nbx import NBXContainer
    from neurobrix.serving.client import DaemonClient
    loads = []

    def _load(*a, **k):
        """The container load's stand-in: records that it was called."""
        loads.append(a)
        raise RuntimeError("the container load was reached")

    monkeypatch.setattr(run_mod, "refuse_unread_flags", lambda args: None)
    monkeypatch.setattr(cli_utils, "resolve_model_arg", lambda name: name)
    monkeypatch.setattr(DaemonClient, "is_running", staticmethod(lambda: False))
    monkeypatch.setattr(run_mod, "find_model", lambda name: tmp_path / "m.nbx")
    monkeypatch.setattr(engine_choice, "choose_engine", lambda args, log: "compiled")
    monkeypatch.setattr(NBXContainer, "load", staticmethod(_load))
    args = _args(model="m", turns=_turns_file(tmp_path), prompt="x", audio="a.wav")
    with pytest.raises((SystemExit, RuntimeError)) as exc:
        run_mod.cmd_run(args)
    assert loads == []
    assert "--prompt beside it would be ignored" in str(exc.value)


def test_audio_llm_without_turns_and_without_prompt_still_requires_prompt():
    args = _args(audio="a.wav")
    with pytest.raises(RuntimeError, match="requires --prompt"):
        validate_required_inputs("audio_llm", args)


def test_a_text_llm_with_turns_and_prompt_is_unchanged(tmp_path):
    args = _args(turns=_turns_file(tmp_path), prompt="x")
    with pytest.raises(SystemExit, match="--turns plays its own messages; --prompt beside it would be ignored"):
        load_turns(args)
    validate_required_inputs("llm", _args(turns=_turns_file(tmp_path)))
    validate_required_inputs("llm", args)              # the family accepts --turns; the beside refusal is load_turns
