"""A `generate` params key the engine does not read is refused by name before
any work, and the refusal says what the engine does read (the Dell's decision,
to_mac 2026-10-11 21:12, on the residual named in wire-protocol.md).

Before: `InferenceEngine.generate(**kwargs)` read the keys it knew and
ignored the rest, so a misspelt `step=4` ran at the model's default step
count and answered as if it had been obeyed.

The accepted set is not a list: it is the set of keys `generate` looked at
while it built the model's inputs, so it cannot drift from what is read.
The real `InferenceEngine.generate` runs here on a stand-in executor; no
model is loaded.
"""
from __future__ import annotations

import pytest

from neurobrix.serving.engine import InferenceEngine


class _Reached(Exception):
    """The executor was called: the request passed the refusal."""


class _Executor:
    def execute(self, inputs):
        raise _Reached(inputs)


def _engine() -> InferenceEngine:
    e = InferenceEngine.__new__(InferenceEngine)
    e._is_loaded, e._device, e._executor, e._family = True, "cpu", _Executor(), "image"
    return e


@pytest.mark.parametrize("key", ["step", "guidance", "num_inference_steps"])
def test_an_unread_key_is_refused_before_the_model_runs(key):
    with pytest.raises(ValueError) as refused:
        _engine().generate(prompt="a red apple", steps=4, **{key: 1})
    msg = str(refused.value)
    assert msg.startswith(f"params key {key!r} is not read by this engine; it reads: ")
    assert "steps" in msg and "prompt" in msg


def test_every_key_the_engine_reads_is_served():
    read = {"steps": 4, "height": 64, "width": 64, "cfg": 3.0, "temperature": 0.5,
            "repetition_penalty": 1.1, "max_tokens": 8, "chat_mode": False, "mode": "t2i",
            "speaker": "a", "num_frames": 1}
    with pytest.raises(_Reached) as reached:
        _engine().generate(prompt="a red apple", **read)
    inputs = reached.value.args[0]
    assert inputs["global.num_inference_steps"] == 4 and inputs["global.guidance_scale"] == 3.0


# chat and complete: the same rule (the Dell, to_mac 2026-10-11 21:16), checked
# before the conversation is touched and before any token is decoded.

class _Session:
    def __init__(self):
        self.user_messages = []

    def add_user_message(self, message):
        self.user_messages.append(message)

    def ensure_fits(self, **_):
        return False

    def build_token_ids(self):
        return [1, 2, 3]


class _Tokenizer:
    def apply_chat_template(self, messages, **_):
        return [1, 2, 3]


def _llm_engine(monkeypatch) -> InferenceEngine:
    from types import SimpleNamespace

    import neurobrix.core.config as config
    e = _engine()
    e._family, e._session = "llm", _Session()
    e._executor.modules = {"tokenizer": _Tokenizer()}
    e._pkg = SimpleNamespace(defaults={"max_tokens": 8})
    e._get_max_cache_len = lambda: 4096
    monkeypatch.setattr(config, "get_family_config", lambda family: {"modes": {"supported": ["agent"]}})
    return e


_CALLS = {
    "chat": lambda e, **kw: e.chat("hi", **kw),
    "complete": lambda e, **kw: e.complete([{"role": "user", "content": "hi"}], **kw),
}


@pytest.mark.parametrize("call", sorted(_CALLS))
@pytest.mark.parametrize("key", ["max_new_tokens", "temp", "steps"])
def test_an_unread_key_is_refused_by_chat_and_complete_before_any_decode(monkeypatch, call, key):
    e = _llm_engine(monkeypatch)
    with pytest.raises(ValueError) as refused:
        _CALLS[call](e, max_tokens=4, **{key: 1})
    msg = str(refused.value)
    assert msg.startswith(f"params key {key!r} is not read by this engine; it reads: ")
    assert "max_tokens" in msg and "temperature" in msg
    assert e._session.user_messages == [], "a refused chat request was added to the conversation"


@pytest.mark.parametrize("call", sorted(_CALLS))
def test_every_key_chat_and_complete_read_is_served(monkeypatch, call):
    e = _llm_engine(monkeypatch)
    monkeypatch.setattr("neurobrix.serving.engine.device_seed", lambda device, seed: None)
    with pytest.raises(_Reached) as reached:
        _CALLS[call](e, temperature=0.5, repetition_penalty=1.1, max_tokens=4, seed=7)
    inputs = reached.value.args[0]
    assert inputs["global.temperature"] == 0.5 and inputs["global.max_tokens"] == 4
    assert inputs["global.seed"] == 7 and inputs["global.input_token_ids"] == [1, 2, 3]
