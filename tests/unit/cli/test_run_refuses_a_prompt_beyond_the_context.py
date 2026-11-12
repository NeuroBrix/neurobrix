"""`neurobrix run` refuses by name a prompt longer than the model's context, before anything is planned or loaded — by
the one check `neurobrix serve` answers its 400 context_length_exceeded with.

The Mac's real-use campaign on the published 0.6.9 (2026-11-12, D1): TinyLlama-1.1B-Chat-v1.0 (context 2048) given a
4 094-token prompt by `run` reached the GPU — the Triton engine failed inside the rotary table's index kernel, the PyTorch
engine said the overflow at its attention and then aborted at exit (SIGABRT) — while the same prompt served was refused
before it ran. `run` planned the request (`request_input_config` counted the prompt's tokens for Prism) and never held
them against the model's window.

The request is planned here on a container directory with no weights: its defaults (a 64-position window), its topology,
and a stand-in tokenizer counting one token per character. What this test does if the code is wrong: the first cell
fails when the run's request is planned without a refusal (red on main: nothing raised); the second when the run and the
serving door refuse by two different checks.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import neurobrix.core.runtime.prompt_length as prompt_length
from neurobrix.cli import create_parser
from neurobrix.cli.commands.run import request_input_config

WINDOW, ASKED = 64, 8


class _Tokenizer:
    """One id per character of every message, plus one for the generation prompt."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **_):
        return [ord(c) for m in messages for c in m["content"]] + [1]


def _container(tmp_path, monkeypatch):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps({
        "type": "autoregressive_text", "chat_mode": True, "max_tokens": 512,
        "lm_config": {"max_position_embeddings": WINDOW}}))
    (tmp_path / "topology.json").write_text(json.dumps({
        "flow": {"generation": {"type": "autoregressive_text"}}, "components": {},
        "modules": {"tokenizer": {"path": "modules/tokenizer"}}}))
    monkeypatch.setattr(prompt_length, "container_tokenizer", lambda root: _Tokenizer())
    return {"family": "llm", "model_name": "stand-in", "dtype": "float16", "modules": {"tokenizer": {}}}


def _plan(tmp_path, manifest, prompt):
    args = create_parser().parse_args(["run", "--model", "stand-in", "--prompt", prompt, "--max-tokens", str(ASKED)])
    return request_input_config(args, manifest, "llm", tmp_path)


def test_a_run_s_prompt_past_the_model_s_context_is_refused_by_name_before_it_is_planned(tmp_path, monkeypatch):
    manifest = _container(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"needs 109 positions \(101 rendered \+ 8 to generate\)") as refused:
        _plan(tmp_path, manifest, "x" * 100)
    said = str(refused.value)
    assert type(refused.value).__name__ == "ContextLengthExceeded", said
    assert f"at most {WINDOW}" in said and "context window" in said, said
    fits = WINDOW - ASKED - 1                         # the rendered prompt plus the decode bound: exactly the window
    assert _plan(tmp_path, manifest, "x" * fits).seq_len == fits + 1


def test_the_run_and_the_serving_door_refuse_by_the_one_check(tmp_path, monkeypatch):
    from neurobrix.serving.engine import InferenceEngine
    calls = []
    real = prompt_length.refuse_past_reach
    monkeypatch.setattr(prompt_length, "refuse_past_reach", lambda *a, **k: calls.append(a) or real(*a, **k))
    manifest = _container(tmp_path, monkeypatch)
    with pytest.raises(prompt_length.ContextLengthExceeded):
        _plan(tmp_path, manifest, "x" * 100)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine._executor = SimpleNamespace(modules={"tokenizer": _Tokenizer()}, kv_tier=None)
    engine._pkg = SimpleNamespace(defaults={"max_tokens": ASKED, "lm_config": {"max_position_embeddings": WINDOW}})
    engine._plan = SimpleNamespace(kv_cache_plan=SimpleNamespace(session_reach=WINDOW, max_cache_len=WINDOW))
    with pytest.raises(prompt_length.ContextLengthExceeded, match="at most 64"):
        engine.check_context([{"role": "user", "content": "x" * 100}])
    from neurobrix.serving.session import ConversationSession, play_turn     # a served chat turn, and `run --turns`
    with pytest.raises(prompt_length.ContextLengthExceeded, match=r"/new"):
        play_turn(ConversationSession(_Tokenizer(), WINDOW, "the plan"), "x" * 100, ASKED, lambda ids, kv: {})
    assert [(c[1], c[2], c[4]) for c in calls] == [(101, ASKED, WINDOW)] * 3, calls
