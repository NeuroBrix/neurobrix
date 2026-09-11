"""Torch is asked for a device selection only where it has one.

`torch.cuda.set_device(idx)` on the compiled engine's multi-device path raised
`AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'` on
Apple. Measured 2026-09-11 on `CogVideoX-2b`: the compiled arm died in 14.7 s
at `compiled_sequence.py:4136`, and no model measured before it had ever taken
the multi-device path — so the line had never been reached on this machine,
and the E queue is what reached it.

Same shape as the `f"cuda:{idx}"` that `torch_device_str` replaced: the
engine's internal token for device memory handed to torch, which resolves it
against the build it was compiled with.

Doing nothing on a single-device backend is not a fallback — `torch.mps` has
no `set_device` and `torch.mps.device_count()` is 1, so there is no selection
to make. A backend that HAS one and is missing from the table is refused.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_torch_device_selection_is_a_capability.py -v
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import nbx_tensor as nt


def test_every_backend_says_whether_torch_selects_a_device():
    table = nt._TORCH_HAS_DEVICE_SELECTION
    assert table["cuda"] is True and table["hip"] is True
    assert table["metal"] is False
    assert set(table) == set(nt._GPU_BACKENDS) | {"metal"}, (
        "a backend without a row would be guessed at, not asked")


def test_an_unknown_backend_is_refused_not_guessed(monkeypatch):
    monkeypatch.setattr(nt, "_detect_gpu_backend", lambda: "tenstorrent")
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        nt.set_torch_device(0)


def test_a_single_device_backend_is_a_no_op_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(nt, "_detect_gpu_backend", lambda: "metal")
    nt.set_torch_device(0)
    nt.set_torch_device(3)          # an index it cannot honour is still not an error


def test_a_numbered_backend_is_asked(monkeypatch):
    """CUDA must still be told, or a launch lands on the wrong device."""
    monkeypatch.setattr(nt, "_detect_gpu_backend", lambda: "cuda")
    seen = []

    class _FakeCuda:
        @staticmethod
        def set_device(idx):
            seen.append(idx)

    import sys
    import types
    fake = types.ModuleType("torch")
    fake.cuda = _FakeCuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    nt.set_torch_device(2)
    assert seen == [2], "the CUDA path must keep selecting its device"


def test_the_compiled_engine_no_longer_calls_torch_cuda_directly():
    """The site that raised. A structural pin because the failure needs a
    multi-device plan to reproduce, which this machine cannot make."""
    import inspect
    from neurobrix.core.runtime.graph import compiled_sequence
    src = inspect.getsource(compiled_sequence)
    assert "torch.cuda.set_device(" not in src, (
        "the compiled multi-device path is calling torch.cuda directly again")
    assert "set_torch_device(" in src
