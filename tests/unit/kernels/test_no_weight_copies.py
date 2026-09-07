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


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_index_select_along_any_axis_needs_no_layout_copy(dim):
    """A gather along a middle axis used to move the axis last (a copy), gather, and permute the
    result back (a copy): the middle-axis kernel writes the output in its final layout. The
    reference is numpy's take on the same data."""
    rng = np.random.default_rng(11)
    data = (rng.standard_normal((5, 7, 6)) * 0.5).astype(np.float16)
    n = data.shape[dim]
    idx = np.array([n - 1, 0, n // 2, n // 2, 1], dtype=np.int64)[:n]      # within [0, n)
    x = NBXTensor.from_numpy(data); index = NBXTensor.from_numpy(idx)
    with _Count() as c:
        out = W.index_select_wrapper(x, dim, index)
    assert c.copies == 0
    assert list(out.shape) == [data.shape[0] if dim != 0 else len(idx), data.shape[1] if dim != 1 else len(idx), data.shape[2] if dim != 2 else len(idx)]
    assert np.array_equal(_d2h(out).reshape(out.shape), np.take(data, idx, axis=dim))


class _Launches(_Count):
    """Every kernel launched through the launcher during a block, by name."""
    def __enter__(self):
        super().__enter__()
        me = self
        me.names = []
        orig = L.launch

        def naming(kernel, grid, *a, **k):
            me.names.append(getattr(kernel, "__name__", ""))
            return orig(kernel, grid, *a, **k)
        L.launch = naming
        return self


def test_a_size_one_dim_may_carry_any_stride():
    """PyTorch's contiguity: a transposed single-token head block (1, 32, 64, 1) and an
    expanded (1, 32, 1, 64) view are dense row-major; they are read in place by an
    element-wise kernel and re-viewed without a copy."""
    rng = np.random.default_rng(3)
    data = (rng.standard_normal((1, 32, 1, 64))).astype(np.float32)
    x = NBXTensor.from_numpy(data)
    t = x.transpose(2, 3)                       # (1, 32, 64, 1), strides (2048, 64, 1, 64)
    assert t.is_contiguous()
    with _Count() as c:
        y = W.mul(t, 2.0)
        v = x.expand(1, 32, 1, 64).view(1, 32, 64)
    assert c.copies == 0
    assert np.array_equal(_d2h(y).reshape(1, 32, 64, 1), data.transpose(0, 1, 3, 2) * 2.0)
    assert list(v.shape) == [1, 32, 64]
    # a dim that is walked keeps the rule
    assert not x.transpose(1, 3).is_contiguous()


def test_a_strided_write_into_a_strided_slice_casts_in_one_launch():
    """The KV cache's write: a head-strided fp32 V into an fp16 slice of the buffer. One launch
    of the nd copy, no contiguous copy of the source, no cast transient, no scatter; the bytes
    are those of the cast-then-scatter path (numpy's round-to-nearest fp16)."""
    rng = np.random.default_rng(4)
    src_np = (rng.standard_normal((1, 4, 8, 1, 64)) * 3).astype(np.float32)
    v = NBXTensor.from_numpy(src_np)[:, :, 0]                  # (1, 4, 1, 64), heads at stride 512
    assert not v.is_contiguous()
    buf = NBXTensor.from_numpy(np.zeros((1, 4, 16, 64), dtype=np.float16))
    with _Launches() as l:
        buf[:1, :, 5:6, :] = v
    assert l.names == ["strided_copy_nd_kernel"], l.names
    got = _d2h(buf).reshape(1, 4, 16, 64)
    assert np.array_equal(got[:, :, 5], src_np[:, :, 0, 0].astype(np.float16))
    assert not got[:, :, :5].any() and not got[:, :, 6:].any()


def test_a_cast_of_a_strided_source_is_one_launch():
    rng = np.random.default_rng(5)
    data = (rng.standard_normal((6, 5)) * 3).astype(np.float32)
    x = NBXTensor.from_numpy(data).transpose(0, 1)             # (5, 6) strided
    with _Launches() as l:
        y = x.to(NBXDtype.float16)
    assert l.names == ["strided_copy_nd_kernel"], l.names
    assert np.array_equal(_d2h(y).reshape(5, 6), data.T.astype(np.float16))


def test_copy_into_a_strided_view_broadcasts_the_source():
    """`buf[:, :3] = row` with a (3,) row: torch's copy_ semantics, the source broadcast by a
    stride 0 — no read past the source, one launch."""
    buf = NBXTensor.from_numpy(np.zeros((4, 8), dtype=np.float32))
    row = NBXTensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with _Launches() as l:
        buf[:, :3].copy_(row)
    assert l.names == ["strided_copy_nd_kernel"], l.names
    got = _d2h(buf).reshape(4, 8)
    assert np.array_equal(got[:, :3], np.tile([1.0, 2.0, 3.0], (4, 1))) and not got[:, 3:].any()


@pytest.mark.parametrize("q_dt,table_dt,S,copies", [(np.float16, np.float32, 1, 0), (np.float32, np.float16, 37, 2),
                                                     (np.float16, np.float16, 37, 0)])
def test_rope_casts_the_tables_in_kernel_to_the_bytes_of_the_cast_copy(q_dt, table_dt, S, copies):
    """A table wider than Q (fp32 tables, fp16 Q: the decode case) is rounded on load with no
    copy; a table narrower than Q (fp16 tables, fp32 Q: Sana's Gemma-2 encoder) keeps its two
    widening copies (the fp32 rotation's contraction moved when widened on load); tables of Q's
    dtype pass as they are. Every case: the bytes of the path that cast the tables beforehand."""
    rng = np.random.default_rng(6)
    B, Hq, Hk, D = 1, 32, 4, 64
    q_np = (rng.standard_normal((B, S, Hq, D)) * 0.5).astype(q_dt)
    k_np = (rng.standard_normal((B, S, Hk, D)) * 0.5).astype(q_dt)
    ang = rng.uniform(-np.pi, np.pi, (B, 1, S, D)).astype(np.float32)
    cos_np, sin_np = np.cos(ang).astype(table_dt), np.sin(ang).astype(table_dt)

    def run(cos_t, sin_t):
        q = NBXTensor.from_numpy(q_np).transpose(1, 2)      # (B, Hq, S, D) view, physical B,S,H,D
        k = NBXTensor.from_numpy(k_np).transpose(1, 2)
        with _Count() as c:
            qo, ko = W.rope_fused_wrapper(q, k, cos_t, sin_t)
        return c.copies, _d2h(qo), _d2h(ko)

    copies_ref, q_ref, k_ref = run(NBXTensor.from_numpy(cos_np.astype(q_dt)),
                                   NBXTensor.from_numpy(sin_np.astype(q_dt)))
    n_copies, q_out, k_out = run(NBXTensor.from_numpy(cos_np), NBXTensor.from_numpy(sin_np))
    assert n_copies == copies and copies_ref == 0
    assert np.array_equal(q_out, q_ref) and np.array_equal(k_out, k_ref)
    assert not np.array_equal(q_out, q_np.reshape(-1))     # the rotation happened


def test_decode_attention_reads_a_wider_q_in_the_cache_dtype_without_a_copy(monkeypatch):
    """The KV cache asks for Q in its dtype: on the vector decode route the kernel rounds the
    fp32 Q on load; the bytes are those of casting Q to fp16 first, with no copy launched."""
    monkeypatch.setenv("NBX_DECODE_VEC", "1")
    rng = np.random.default_rng(7)
    B, H, H_kv, T_k, D = 1, 8, 2, 40, 64
    q_np = (rng.standard_normal((B, H, 1, D))).astype(np.float32)
    k_np = (rng.standard_normal((B, H_kv, T_k, D))).astype(np.float16)
    v_np = (rng.standard_normal((B, H_kv, T_k, D))).astype(np.float16)
    k, v = NBXTensor.from_numpy(k_np), NBXTensor.from_numpy(v_np)
    scale = 1.0 / np.sqrt(D)
    with _Count() as c:
        out = W.scaled_dot_product_attention_wrapper(NBXTensor.from_numpy(q_np), k, v, scale=scale,
                                                     k_pre_transposed=False, q_dtype_of_kv=True)
    assert c.copies == 0
    assert out.nbx_dtype == NBXDtype.float16
    ref = W.scaled_dot_product_attention_wrapper(NBXTensor.from_numpy(q_np.astype(np.float16)), k, v,
                                                 scale=scale, k_pre_transposed=False)
    assert np.array_equal(_d2h(out), _d2h(ref))


@pytest.mark.parametrize("op", ["add", "mul"])
def test_a_broadcast_or_strided_binary_operand_is_read_by_its_strides(op):
    """A conv bias (C,) over an (N, C, H, W) image, an adaLN vector (B, 1, D) over (B, S, D),
    and a transposed operand: the strided kernels read them in place — no expand+contiguous
    transient, no copy launched — with the bytes of the flat kernel on materialised operands."""
    from neurobrix.kernels.ops.add import add_forward_kernel
    from neurobrix.kernels.ops.mul import mul_forward_kernel
    rng = np.random.default_rng(8)
    fn = W.add if op == "add" else W.mul
    cases = [
        ((2, 8, 16, 16), (1, 8, 1, 1)),        # a conv bias over the channels
        ((2, 64, 48), (2, 1, 48)),             # an adaLN shift over the tokens
        ((2, 64, 48), "transposed"),           # a transposed right operand
    ]
    for a_shape, b_spec in cases:
        a_np = rng.standard_normal(a_shape).astype(np.float16)
        if b_spec == "transposed":
            b_src = rng.standard_normal((2, 48, 64)).astype(np.float16)
            b = NBXTensor.from_numpy(b_src).transpose(1, 2); b_np = np.transpose(b_src, (0, 2, 1))
        else:
            b_np = rng.standard_normal(b_spec).astype(np.float16); b = NBXTensor.from_numpy(b_np)
        a = NBXTensor.from_numpy(a_np)
        with _Launches() as l:
            out = fn(a, b)
        assert not any("copy" in n for n in l.names), l.names
        assert l.names == [f"{op}_strided_nd_kernel"], l.names
        # the reference: the flat kernel on materialised operands (the copying path's bytes)
        bm = NBXTensor.from_numpy(np.ascontiguousarray(np.broadcast_to(b_np, a_shape)))
        ref = NBXTensor.from_numpy(np.zeros(a_shape, dtype=np.float16))
        n = ref.numel()
        if op == "add":
            L.launch(add_forward_kernel, (n // 1024 + 1,), a, bm, ref, n, 1.0, BLOCK_SIZE=1024, num_warps=4)
        else:
            L.launch(mul_forward_kernel, (n // 1024 + 1,), a, bm, ref, n, BLOCK_SIZE=1024, num_warps=4)
        assert np.array_equal(_d2h(out), _d2h(ref)), (a_shape, b_spec)


@pytest.mark.parametrize("M", [16, 64])
def test_addmm_takes_the_activation_weight_and_bias_as_they_are(M):
    """addmm on the card without native bf16: the fp16 activation is widened in registers, the
    pre-transposed fp16 weight walked by its strides, the fp16 bias widened on load — no copy
    launched — with the bytes of the path that copied all three beforehand."""
    N, K = 96, 80
    a = _act(M, K).to(NBXDtype.float16)                     # an fp16 activation
    w = _weight(N, K)                                       # (N, K) fp16, the loader's layout
    rng = np.random.default_rng(9)
    bias = NBXTensor.from_numpy((rng.standard_normal(N) * 0.1).astype(np.float16))
    with _Count() as c:
        out = W.addmm(bias, a, w.t())
    assert c.copies == 0
    ref = W.addmm(bias.to(NBXDtype.float32), a.to(NBXDtype.float32), w.t().contiguous())
    assert out.nbx_dtype == ref.nbx_dtype
    assert np.array_equal(_d2h(out), _d2h(ref))
