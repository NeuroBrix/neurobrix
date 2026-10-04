"""The grouped GEMM reads every expert of a projection through ONE stride pair,
taken from expert 0 (`_build_ptr_tables`). Stacked-slab views share it by
construction; a list mixing layouts — a per-expert [F, H] matrix beside a
transposed view of a slab — would be read through the wrong strides in silence.
The table builder refuses it by name before any table is built.

What this test would do if the code were wrong: without the door the builder
goes on toward the device — which this test replaces by a sentinel that raises
"past the door" — and never names the mismatch: the `pytest.raises` fails. No
card: the refusal happens before any allocation.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")


class _W:
    def __init__(self, shape, strides):
        self.shape, self._strides, self._device_idx = shape, strides, 0

    def stride(self, d):
        return self._strides[d]


def _past_the_door(*_a, **_k):
    raise RuntimeError("past the door: the builder went on to the device")


def test_a_projection_mixing_layouts_is_refused_by_name(monkeypatch):
    from neurobrix.triton import moe as M
    from neurobrix.triton.moe import _build_ptr_tables
    monkeypatch.setattr(M, "_detect_gpu_backend", lambda: "cuda")
    monkeypatch.setattr(M.DeviceAllocator, "set_device", staticmethod(_past_the_door))
    F, H = 4, 6
    linear = [_W((F, H), (H, 1)) for _ in range(3)]               # (out x in), k contiguous
    mixed = linear[:2] + [_W((F, H), (1, 2 * F))]                  # a transposed slab view
    with pytest.raises(RuntimeError, match=r"'up' expert 2 has \(shape, strides\)"):
        _build_ptr_tables(linear, mixed, linear)
