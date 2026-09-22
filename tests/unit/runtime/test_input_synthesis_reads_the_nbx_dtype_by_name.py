"""Triton-path input synthesis reads an NBXTensor's dtype by NAME, not value.

`InputSynthesizer.synthesize_missing_inputs` derives the dtype of a
synthesized input from an existing input tensor. On the Triton path the
existing input is an NBXTensor, and the dtype was read as
`str(val.nbx_dtype).split(".")[-1]`. `NBXDtype` is an IntEnum, so `str()`
yields the enum's VALUE ("0"/"1"/"2"), not "float16"/"bfloat16"/"float32".
`_np_dtype("0")` then raises `data type '' not understood` (numpy reads the
leading digit as a byte count with no type letter).

This is latent on every Triton run that must synthesize an input — it
surfaced in the Apple census shadow the moment Prism stopped mis-placing a
component on the host (b23105fe) and the run reached the main loop:
PixArt-XL and CogVideoX-2b both died in `_full2` on the `from_dimensions`
resolution input.

The fix reads `val.nbx_dtype.name`. Model-free: the synthesizer is driven
directly with an NBXTensor input and a `from_dimensions` rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.runtime.resolution.input_synthesizer import InputSynthesizer
from neurobrix.kernels.nbx_tensor import NBXTensor


class _Resolver:
    mode = "triton"
    defaults = {"height": 512, "width": 512}

    def get(self, _key):
        return None


class _Plan:
    components = {}
    primary_device = None            # no device move: keep the test card-free
    vendor = "apple"
    architecture = "apple_silicon"


def _synth():
    topology = {"synthesis": {"transformer": {"resolution": {"method": "from_dimensions"}}}}
    return InputSynthesizer(topology, _Resolver(), _Plan(), {}, {})


@pytest.mark.parametrize("np_dtype,want", [
    (np.float16, "float16"),
    (np.float32, "float32"),
])
def test_a_synthesized_input_takes_the_named_dtype_of_an_nbx_input(np_dtype, want):
    synth = _synth()
    # one existing NBXTensor input → the dtype is read off its nbx_dtype
    existing = NBXTensor.from_numpy(np.zeros((1, 8), dtype=np_dtype))
    inputs = {"hidden_states": existing}

    out = synth.synthesize_missing_inputs("transformer", inputs)

    assert "resolution" in out, "the from_dimensions rule did not fire"
    res = out["resolution"]
    assert hasattr(res, "nbx_dtype"), f"synthesized a non-NBX input: {type(res)}"
    assert res.nbx_dtype.name == want, (
        f"synthesized input dtype {res.nbx_dtype.name!r} != source {want!r} — "
        f"the dtype was read from the IntEnum's value, not its name")
    # the resolution row is [height, width]
    assert tuple(res.shape) == (1, 2)
