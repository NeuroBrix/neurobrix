"""The screen's oracle hands back every buffer in the bytes of the SPAN the screen snapshots, not in logical order.

The screen (`_writable_buffers`) keeps a dense permutation of a buffer — a conv input read NHWC as NCHW, a transposed
matmul operand — and snapshots the span from its `data_ptr()`: its PHYSICAL bytes. The provider answered an input
with `ascontiguousarray(numpy())`, its LOGICAL bytes. For a contiguous tensor they are one thing; for a permutation
they are the same elements in another order, so every candidate "changed" an input it only read, and the screen
refused the whole space of a correct kernel — measured on an fp32 conv2d whose input had strides (2048, 65536, 32, 1)
(2026-10-29: "the fp64 oracle contradicts EVERY candidate"; the kernel itself agreed with the fp64 reference to 8e-08
on that very input).

What this test would do if the code were wrong: a provider that answers in logical order gives the transposed bytes
for `a` (red); one that loses the reference for the output on a permuted `c` gives the wrong order there (red).
Card-less: host buffers stand in for device ones.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_the_screen_oracle_lays_a_strided_input_in_its_span.py -v
"""
from __future__ import annotations

import numpy as np

from nbxtensor.ops import screen_oracle as SO


class _View:
    """A view over a host buffer: its address is the buffer's, its elements are read through its strides."""

    def __init__(self, base, view):
        self._base = base                       # the physical buffer, kept alive
        self._view = view
        self.shape = view.shape

    def data_ptr(self):
        return self._base.ctypes.data

    @property
    def _nbytes(self):
        return self._base.nbytes

    def stride(self):
        return tuple(s // self._view.itemsize for s in self._view.strides)

    def is_contiguous(self):
        return bool(self._view.flags["C_CONTIGUOUS"])

    def numpy(self):
        return np.ascontiguousarray(self._view)


class _Tuner:
    def __init__(self, name, nargs):
        self.base_fn = type(name, (), {})
        self.nargs = nargs


def _transposed(rows, cols, seed):
    base = np.ascontiguousarray(np.random.default_rng(seed).standard_normal((cols, rows)), dtype=np.float32)
    return _View(base, base.T)                  # logical (rows, cols), physical (cols, rows)


def _buffers(*tensors):
    return [(t.data_ptr(), t._nbytes, "fp32") for t in tensors]


def test_a_permuted_input_is_answered_in_its_physical_bytes():
    a = _transposed(64, 48, 0)
    assert not a.is_contiguous(), "the premise: a dense permutation"
    b_arr = np.ascontiguousarray(np.random.default_rng(1).standard_normal((48, 32)), dtype=np.float32)
    b = _View(b_arr, b_arr)
    c_arr = np.zeros((64, 32), dtype=np.float32)
    c = _View(c_arr, c_arr)
    out = SO.provider(_Tuner("matmul_kernel", {"a_ptr": a, "b_ptr": b, "c_ptr": c}), ("k",), _buffers(a, b, c))
    assert out is not None
    assert out[0] == a._base.tobytes(), "the input was answered in logical order, not in the span the screen holds"
    want = (a.numpy().astype(np.float64) @ b_arr.astype(np.float64)).astype(np.float32)
    assert np.allclose(np.frombuffer(out[2], dtype=np.float32).reshape(64, 32), want, rtol=1e-5)


def test_a_permuted_output_gets_the_reference_in_its_physical_bytes():
    a_arr = np.ascontiguousarray(np.random.default_rng(2).standard_normal((64, 48)), dtype=np.float32)
    a = _View(a_arr, a_arr)
    b_arr = np.ascontiguousarray(np.random.default_rng(3).standard_normal((48, 32)), dtype=np.float32)
    b = _View(b_arr, b_arr)
    c = _transposed(64, 32, 4)
    out = SO.provider(_Tuner("matmul_kernel", {"a_ptr": a, "b_ptr": b, "c_ptr": c}), ("k",), _buffers(a, b, c))
    assert out is not None
    want = (a_arr.astype(np.float64) @ b_arr.astype(np.float64)).astype(np.float32)
    got = np.frombuffer(out[2], dtype=np.float32).reshape(32, 64)      # the physical layout: c.T row-major
    assert np.allclose(got.T, want, rtol=1e-5), "the output's reference is not laid in its span"
