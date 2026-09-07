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



def _act16(M, K, seed=5):
    rng = np.random.default_rng(seed)
    return NBXTensor.from_numpy((rng.standard_normal((M, K)) * 0.05).astype(np.float16))


@pytest.mark.parametrize("M", [1, 4, 16, 64])
def test_fp16_activation_is_widened_in_registers_not_materialised(M):
    """On a card without native bf16 the fp16 activation used to be copied to fp32 before every
    matmul; the kernels widen it on load now — the same numbers, no copy. The reference is the
    same kernel handed the pre-widened activation (the former path, still reachable)."""
    Wt, a16 = _weight(96, 128), _act16(M, 128)
    b = Wt.t()
    a32 = a16.to(NBXDtype.float32)
    ref = W.mm(a32, b)                            # promote_a False (fp32 in), promote_b True: the old path
    with _Count() as c:
        out = W.mm(a16, b)
    assert c.copies == 0, "no activation upcast copy, no weight copy"
    assert out._dtype == ref._dtype == NBXDtype.float32
    assert np.array_equal(_d2h(out), _d2h(ref))


def test_rms_norm_widens_on_load_and_stores_the_dtype_asked():
    """The fp32-internal wrap used to copy a half input to fp32 before rms_norm and copy the
    fp32 result back; the kernel widens its loads and stores in the dtype asked, so the same
    numbers come out of one store — no copy either side."""
    rng = np.random.default_rng(7)
    x16 = NBXTensor.from_numpy((rng.standard_normal((6, 256)) * 0.5).astype(np.float16))
    w = NBXTensor.from_numpy((1.0 + rng.standard_normal(256) * 0.1).astype(np.float16))
    ref32 = W.rms_norm(x16.to(NBXDtype.float32), w)                  # the former path: materialised fp32 input
    with _Count() as c:
        out32 = W.rms_norm(x16, w, out_dtype=NBXDtype.float32)
    assert c.copies == 0 and out32._dtype == NBXDtype.float32
    assert np.array_equal(_d2h(out32), _d2h(ref32))
    ref16 = ref32.to(NBXDtype.float16)                                 # the former cast back
    with _Count() as c:
        out16 = W.rms_norm(x16, w, out_dtype=NBXDtype.float16)
    assert c.copies == 0 and out16._dtype == NBXDtype.float16
    assert np.array_equal(_d2h(out16), _d2h(ref16))


def test_the_fp32_internal_wrap_asks_a_widening_wrapper_for_its_output_dtype(monkeypatch):
    from neurobrix.triton import dtype as D
    from neurobrix.kernels import wrappers as _w
    calls = []

    def widening(x, out_dtype=None):
        calls.append((x, out_dtype)); return x
    widening._nbx_widens_on_load = True
    eng = D.TritonDtypeEngine.__new__(D.TritonDtypeEngine)
    eng.compute_dtype = NBXDtype.float16
    x16 = NBXTensor.from_numpy(np.ones((2, 8), dtype=np.float16))
    monkeypatch.setattr(_w, "_NBX_ACTIVATIONS_FP16_SAFE", False)
    with _Count() as c:
        eng._wrap_fp32_internal_compute_dtype_output(widening)(x16)
    assert c.copies == 0 and calls[-1][0] is x16 and calls[-1][1] == NBXDtype.float32, "conservative: fp32 asked, no input copy"
    monkeypatch.setattr(_w, "_NBX_ACTIVATIONS_FP16_SAFE", True)
    eng._wrap_fp32_internal_compute_dtype_output(widening)(x16)
    assert calls[-1][1] == NBXDtype.float16, "cast back on: the compute dtype asked of the store"
