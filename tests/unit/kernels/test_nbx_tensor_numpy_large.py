"""`NBXTensor.numpy()` reads any size back: the byte copy went through a C int and refused
2 GiB (the certifier on 2026-09-07). The host view path is exercised on small tensors here,
and the size arithmetic on a synthetic buffer without allocating 2 GiB."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor


def test_numpy_reads_back_the_values_and_never_a_bytes_copy(monkeypatch):
    a = (np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7) * 0.5)
    t = NBXTensor.from_numpy(a)
    if t._device != "cpu":
        t = t.to_cpu()
    monkeypatch.setattr(ctypes, "string_at", lambda *a, **k: (_ for _ in ()).throw(AssertionError("string_at must not be used")))
    b = t.numpy()
    assert b.dtype == np.float32 and b.shape == (3, 5, 7) and np.array_equal(a, b)
    b[0, 0, 0] = 99.0                                                     # a copy, not a view over the tensor
    assert t.numpy()[0, 0, 0] == 0.0


def test_an_empty_tensor_reads_back_empty():
    t = NBXTensor.from_numpy(np.zeros((0, 4), dtype=np.float16))
    if t._device != "cpu":
        t = t.to_cpu()
    assert t.numpy().shape == (0, 4)


def test_the_size_arithmetic_holds_above_two_gib():
    """The ctypes array type takes a Py_ssize_t: 2 GiB + 1 is a valid buffer type."""
    n = 2 ** 31 + 1
    arr_t = ctypes.c_uint8 * n
    assert ctypes.sizeof(arr_t) == n
