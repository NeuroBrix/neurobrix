"""The Triton-sequential dispatcher sets the wrappers' per-component context only while it runs.

`kernels.wrappers` carries a process-wide context the self-managed wrappers read at call time — the component's compute
dtype (`conv2d_wrapper` sizes its output from it) and the activations-fp16-safe flag. `TritonSequence.run` sets both at
entry and restores them at exit. The sequential dispatcher set them in its CONSTRUCTOR and restored nothing: after one
dispatcher had existed in a process, every later fp32 convolution run outside a sequence was computed under an fp16
compute dtype — the unit suite of 2026-10-18 saw 27 conv kernels red (an fp32 conv2d returning inf above the fp16 range,
depthwise and transposed convs at 3e-4) with every one of them green alone; the leak was
`test_every_triton_site_asks_its_engine`, which merely constructs a dispatcher.

What this test would do if the code were wrong: construction changes the context (red at the first assertion), or the
context is not restored after the run (red at the last).
"""
from __future__ import annotations

import pytest

from nbxtensor.ops import wrappers as W
from nbxtensor.dtype.nbx_dtype import NBXDtype

pytest.importorskip("triton")


def test_construction_leaves_the_context_and_the_run_restores_it(monkeypatch):
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    monkeypatch.setattr(W, "_NBX_COMPUTE_DTYPE", NBXDtype.float32)
    monkeypatch.setattr(W, "_NBX_ACTIVATIONS_FP16_SAFE", False)
    disp = TritonSequentialDispatcher(0, NBXDtype.float16, activations_fp16_safe=True, stores_fp64=True)
    assert W.get_compute_dtype() is NBXDtype.float32, "constructing a dispatcher changed the wrappers' compute dtype"
    assert W.get_activations_fp16_safe() is False
    with disp:
        assert W.get_compute_dtype() is NBXDtype.float16
        assert W.get_activations_fp16_safe() is True
    assert W.get_compute_dtype() is NBXDtype.float32, "the run did not restore the compute dtype it found"
    assert W.get_activations_fp16_safe() is False


def test_the_context_is_restored_when_the_run_raises(monkeypatch):
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    monkeypatch.setattr(W, "_NBX_COMPUTE_DTYPE", None)
    monkeypatch.setattr(W, "_NBX_ACTIVATIONS_FP16_SAFE", False)
    disp = TritonSequentialDispatcher(0, NBXDtype.float16, stores_fp64=True)
    with pytest.raises(RuntimeError, match="op failed"):
        with disp:
            raise RuntimeError("op failed")
    assert W.get_compute_dtype() is None
    assert W.get_activations_fp16_safe() is False
