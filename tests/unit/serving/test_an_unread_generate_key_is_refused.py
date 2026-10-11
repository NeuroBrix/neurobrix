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
