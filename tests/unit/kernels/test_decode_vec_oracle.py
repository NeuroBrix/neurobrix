"""Day-one float64 oracle for the vector (SIMT) decode-attention kernel.

The kernel (ops/decode_attn_vec.py) is born under the correctness-oracle
gate class (afcb8cb doctrine: a byte gate proves we repeat ourselves; an
oracle proves we are right) and under the route-activation rule
(2026-08-23 lesson: a route-dependent claim needs the route pinned —
the first strided-KV equivalence was vacuous because the test process
silently took another path).

Five proofs:
  1. CORRECTNESS vs the independently written float64 reference at the
     head-dim hazard triple D = 127/128/129 AND the canonical row
     shapes (GQA 32/8, bucketed long row, batch 2, MHA).
  2. DETERMINISM: three identical calls, identical bytes (replay
     requirement — the fixed-order reduce is shared with flash_decode).
  3. MASKED TAIL bitwise inert (the bias contract).
  4. ROUTE ACTIVATION: the armed flag actually reaches
     `_decode_attn_vec` (counted), and unarmed it does not.
  5. STRIDED K/V: the bucketed cache's prefix-slice views produce the
     same bytes as materialised K/V — consumed by strides, zero
     K/V-sized copies (the kernel's contract with the strided-KV lot).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_decode_vec_oracle.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_decode_vec_oracle.py
"""
from __future__ import annotations

import ctypes
import os as _os

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                def deco(fn):
                    fn._params = (a, k)
                    return fn
                return deco

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels import wrappers as W

# Same derived bound as the flash-decode oracle (fp16 inputs, fp32
# accumulation, ~4k-term softmax sums): relative error of a CORRECT
# kernel sits at a few 1e-4; 2e-3 is an order above that and two under
# the D=128 defect it exists to catch.
BOUND = 2e-03

_NP = {NBXDtype.float16: (np.float16, 2), NBXDtype.float32: (np.float32, 4)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _t(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x))


def _ref_float64(qn, kn, vn, bias, scale):
    """Independent decode attention in float64 (mirror of the
    flash-decode oracle's reference — GQA by repetition)."""
    B, H, _, D = qn.shape
    H_kv = kn.shape[1]
    g = H // H_kv
    K = np.repeat(kn.astype(np.float64), g, axis=1)
    V = np.repeat(vn.astype(np.float64), g, axis=1)
    Q = qn.astype(np.float64)
    s = np.einsum("bhqd,bhkd->bhqk", Q, K) * scale
    if bias is not None:
        s = s + bias.astype(np.float64)[None, None, None, :]
    m = s.max(-1, keepdims=True)
    m = np.where(np.isinf(m) & (m < 0), 0.0, m)
    e = np.exp(s - m)
    e = np.where(np.isneginf(s), 0.0, e)
    l = e.sum(-1, keepdims=True)
    l = np.where(l == 0.0, 1.0, l)
    return np.einsum("bhqk,bhkd->bhqd", e / l, V)


def _run(B, H, H_kv, T_k, D, bias_len=None, seed=0):
    """Arms NBX_DECODE_VEC for the call — this oracle tests THE KERNEL."""
    _os.environ["NBX_DECODE_VEC"] = "1"
    try:
        rng = np.random.default_rng(seed)
        qn = rng.standard_normal((B, H, 1, D)).astype(np.float16)
        kn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
        vn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
        bias = None
        bias_t = None
        if bias_len is not None:
            b = np.zeros(T_k, dtype=np.float32)
            b[bias_len:] = -np.inf
            bias = b
            bias_t = _t(b.reshape(1, 1, 1, T_k))
        q, k, v = _t(qn), _t(kn), _t(vn)
        out = W.scaled_dot_product_attention_wrapper(
            q, k, v, attn_mask=bias_t, is_causal=False,
            scale=1.0 / np.sqrt(D))
        got = _d2h(out).astype(np.float64).reshape(B, H, 1, D)
        ref = _ref_float64(qn, kn, vn, bias, 1.0 / np.sqrt(D))
        err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
        return err, got
    finally:
        _os.environ.pop("NBX_DECODE_VEC", None)


_SHAPES = [
    # B, H, H_kv, T_k, D, live_len
    (1, 32, 8, 256, 127, None),
    (1, 32, 8, 256, 128, None),      # the hazard specialisation
    (1, 32, 8, 256, 129, None),
    (1, 32, 8, 4352, 128, 4165),     # canonical long-row decode: bucketed
    (1, 32, 8, 256, 128, 13),        # short row: 13 live keys of 256
    (1, 32, 32, 256, 128, None),     # MHA
    (1, 16, 2, 1024, 64, None),      # 8:1 at D=64
    (2, 8, 2, 512, 128, 300),        # batch 2
    (1, 32, 8, 8448, 128, 8200),     # deeper than the long row
]


@pytest.mark.parametrize("B,H,H_kv,T_k,D,live", _SHAPES)
def test_decode_vec_matches_float64(B, H, H_kv, T_k, D, live) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    err, _ = _run(B, H, H_kv, T_k, D, live)
    assert err <= BOUND, (
        f"B={B} H={H} H_kv={H_kv} T_k={T_k} D={D} live={live}: relative "
        f"error {err:.3e} exceeds {BOUND:.1e} by {err/BOUND:.0f}x.")


@pytest.mark.parametrize("D", [127, 128, 129])
def test_decode_vec_is_deterministic(D) -> None:
    """Three calls, identical bytes — a replay-engine requirement."""
    if not _has_gpu():
        pytest.skip("no GPU")
    outs = [_run(1, 32, 8, 1024, D, 900, seed=5)[1].tobytes()
            for _ in range(3)]
    assert outs[0] == outs[1] == outs[2], (
        f"D={D}: outputs differ across identical calls — disqualified "
        f"for the replay engine regardless of accuracy")


def test_decode_vec_is_actually_routed() -> None:
    """ACTIVATION PROOF: armed, the wrapper reaches _decode_attn_vec;
    unarmed, it does not (the vacuous-equivalence lesson)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    calls = {"n": 0}
    orig = W._decode_attn_vec

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    W._decode_attn_vec = counting
    try:
        _run(1, 32, 8, 256, 128)
        assert calls["n"] == 1, "armed flag did not reach the vec kernel"
        n_before = calls["n"]
        # unarmed: the same call must NOT route here
        rng = np.random.default_rng(0)
        q = _t(rng.standard_normal((1, 32, 1, 128)).astype(np.float16))
        k = _t(rng.standard_normal((1, 8, 256, 128)).astype(np.float16))
        v = _t(rng.standard_normal((1, 8, 256, 128)).astype(np.float16))
        W.scaled_dot_product_attention_wrapper(
            q, k, v, attn_mask=None, is_causal=False, scale=0.088)
        assert calls["n"] == n_before, "unarmed call routed to the vec kernel"
    finally:
        W._decode_attn_vec = orig


def test_masked_tail_contributes_nothing() -> None:
    """Changing key values BEYOND the live length must not change the
    output at all — the bias contract, checked bitwise."""
    if not _has_gpu():
        pytest.skip("no GPU")
    _os.environ["NBX_DECODE_VEC"] = "1"
    try:
        rng = np.random.default_rng(11)
        B, H, H_kv, T_k, D, live = 1, 32, 8, 512, 128, 300
        qn = rng.standard_normal((B, H, 1, D)).astype(np.float16)
        kn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
        vn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
        b = np.zeros(T_k, dtype=np.float32)
        b[live:] = -np.inf
        bias_t = _t(b.reshape(1, 1, 1, T_k))

        out1 = W.scaled_dot_product_attention_wrapper(
            _t(qn), _t(kn), _t(vn), attn_mask=bias_t, is_causal=False,
            scale=1.0 / np.sqrt(D))
        kn2, vn2 = kn.copy(), vn.copy()
        kn2[:, :, live:, :] = 999.0
        vn2[:, :, live:, :] = -999.0
        out2 = W.scaled_dot_product_attention_wrapper(
            _t(qn), _t(kn2), _t(vn2), attn_mask=bias_t, is_causal=False,
            scale=1.0 / np.sqrt(D))
        assert _d2h(out1).tobytes() == _d2h(out2).tobytes(), (
            "masked-tail key/value bytes leaked into the output")
    finally:
        _os.environ.pop("NBX_DECODE_VEC", None)


def test_strided_bucketed_views_bit_equal_no_copy() -> None:
    """The kernel consumes the bucketed cache's prefix-slice views BY
    STRIDES: same bytes as materialised K/V, zero K/V-sized copies."""
    if not _has_gpu():
        pytest.skip("no GPU")
    from neurobrix.kernels import nbx_tensor as NT
    _os.environ["NBX_DECODE_VEC"] = "1"
    try:
        rng = np.random.default_rng(21)
        B, H, H_kv, PAD, CAP, D = 1, 32, 8, 256, 512, 128
        qn = rng.standard_normal((B, H, 1, D)).astype(np.float16)
        kb = _t(rng.standard_normal((B, H_kv, CAP, D)).astype(np.float16))
        vb = _t(rng.standard_normal((B, H_kv, CAP, D)).astype(np.float16))
        b = np.zeros(PAD, dtype=np.float32)
        b[200:] = -np.inf
        bias_t = _t(b.reshape(1, 1, 1, PAD))
        k_view = kb[:, :, :PAD, :]
        v_view = vb[:, :, :PAD, :]
        assert not k_view.is_contiguous()

        kv_numel = B * H_kv * PAD * D
        copies = {"n": 0}
        orig_sc = NT._strided_copy

        def counting_sc(src, dst):
            if getattr(src, "_numel", 0) == kv_numel:
                copies["n"] += 1
            return orig_sc(src, dst)

        NT._strided_copy = counting_sc
        try:
            out_v = W.scaled_dot_product_attention_wrapper(
                _t(qn), k_view, v_view, attn_mask=bias_t, is_causal=False,
                scale=1.0 / np.sqrt(D))
        finally:
            NT._strided_copy = orig_sc
        out_d = W.scaled_dot_product_attention_wrapper(
            _t(qn), k_view.contiguous(), v_view.contiguous(),
            attn_mask=bias_t, is_causal=False, scale=1.0 / np.sqrt(D))
        assert _d2h(out_v).tobytes() == _d2h(out_d).tobytes(), (
            "strided-view consumption changed the output bytes")
        assert copies["n"] == 0, (
            f"{copies['n']} K/V-sized strided copies ran — the kernel "
            f"materialised after all")
    finally:
        _os.environ.pop("NBX_DECODE_VEC", None)


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    for shp in _SHAPES:
        err, _ = _run(*shp)
        status = "OK " if err <= BOUND else "FAIL"
        print(f"  {status} {shp}: rel err {err:.3e}")
        assert err <= BOUND, shp
    for D in (127, 128, 129):
        outs = [_run(1, 32, 8, 1024, D, 900, seed=5)[1].tobytes()
                for _ in range(3)]
        assert outs[0] == outs[1] == outs[2], D
    print("  OK determinism 3x at D=127/128/129")
    test_decode_vec_is_actually_routed()
    print("  OK route activation (armed reaches, unarmed does not)")
    test_masked_tail_contributes_nothing()
    print("  OK masked tail bitwise inert")
    test_strided_bucketed_views_bit_equal_no_copy()
    print("  OK strided bucketed views: bit-equal, zero K/V copies")
    print("PASS: decode_attn_vec oracle complete")
