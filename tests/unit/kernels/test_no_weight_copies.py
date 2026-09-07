"""No weight copy after load (the copy lever, 2026-09-07). The matmul wrapper walks a
pre-transposed weight by its strides instead of materialising it once per prefill; the GEMV
wrapper takes a row-contiguous matrix and a strided vector as they are, and copies a matrix
whose rows are strided exactly once, saying why; RoPE's per-layer cast of the step's tables stays, justified in its line. Every path stays byte-identical to the copying one — the same tile math on the same
numbers — which is what these tests measure, with the launcher counted."""
from __future__ import annotations

import ctypes
import os as _os

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels import wrappers as W
from neurobrix.kernels import launcher as L

_NP = {NBXDtype.float16: (np.float16, 2), NBXDtype.float32: (np.float32, 4)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="a CUDA device is needed")


@pytest.fixture(autouse=True)
def volta_profile(monkeypatch):
    """The engine sets this from the hardware profile at start (V100: no native bf16, so an
    fp32 activation × fp16 weight promotes the B tile in-kernel); the unit test states it."""
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", False)


def _d2h(t):
    dt, sz = _NP[t._dtype]
    buf = (ctypes.c_char * (t.numel() * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), t.numel() * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


class _Count:
    """Copy-kernel launches through the NeuroBrix launcher during a block."""
    def __init__(self):
        self.copies = 0

    def __enter__(self):
        self._orig = L.launch
        me = self

        def counting(kernel, grid, *a, **k):
            if "copy" in getattr(kernel, "__name__", ""):
                me.copies += 1
            return me._orig(kernel, grid, *a, **k)
        L.launch = counting
        return self

    def __exit__(self, *exc):
        L.launch = self._orig


def _weight(N, K, seed=0):
    rng = np.random.default_rng(seed)
    return NBXTensor.from_numpy((rng.standard_normal((N, K)) * 0.05).astype(np.float16))


def _act(M, K, seed=1):
    rng = np.random.default_rng(seed)
    return NBXTensor.from_numpy((rng.standard_normal((M, K)) * 0.05).astype(np.float32))


@pytest.mark.parametrize("M", [16, 64])
def test_prefill_mm_walks_the_pretransposed_weight_without_a_copy(M):
    Wt, a = _weight(96, 128), _act(M, 128)
    b = Wt.t()                                    # the (K, N) stride view the graph hands the wrapper
    assert not b.is_contiguous() and b.stride(0) == 1
    with _Count() as c:
        out = W.mm(a, b)
    assert c.copies == 0, "the weight is read in place"
    ref = W.mm(a, b.contiguous())                 # the copying path, the same tile math
    assert np.array_equal(_d2h(out), _d2h(ref))


def test_prefill_mm_still_materialises_a_broadcast_weight():
    a = _act(16, 8)
    row = NBXTensor.from_numpy((np.arange(8, dtype=np.float32) * 0.01).astype(np.float16))
    b = row.view(8, 1).expand(8, 6)               # stride 0 along N: the kernel cannot walk it
    with _Count() as c:
        out = W.mm(a, b)
    assert c.copies >= 1
    assert np.array_equal(_d2h(out), _d2h(W.mm(a, b.contiguous())))


def test_gemv_takes_a_row_contiguous_matrix_and_a_strided_vector_as_they_are():
    mat = _weight(64, 256)
    rng = np.random.default_rng(3)
    two = NBXTensor.from_numpy((rng.standard_normal((256, 2)) * 0.05).astype(np.float32))
    vec = two[:, 0]                               # a strided vector (stride 2)
    assert vec.stride(0) == 2
    with _Count() as c:
        out = W.mv_wrapper(mat, vec)
    assert c.copies == 0
    assert np.array_equal(_d2h(out), _d2h(W.mv_wrapper(mat, vec.contiguous())))


def test_gemv_copies_a_row_strided_matrix_exactly_once():
    rng = np.random.default_rng(4)
    X = NBXTensor.from_numpy((rng.standard_normal((256, 64)) * 0.05).astype(np.float16))   # (K, N)
    mat = X.t()                                   # (N, K) with strided rows: the kernel's loads need K-contiguous rows
    vec = NBXTensor.from_numpy((rng.standard_normal(256) * 0.05).astype(np.float32))
    with _Count() as c:
        out = W.mv_wrapper(mat, vec)
    assert c.copies == 1, "the one justified copy"
    assert np.array_equal(_d2h(out), _d2h(W.mv_wrapper(mat.contiguous(), vec)))

