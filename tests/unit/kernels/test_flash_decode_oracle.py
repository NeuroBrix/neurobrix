"""CORRECTNESS ORACLE — flash-decoding kernel, from day one.

This kernel replaces `_math_attention` on the T_q == 1 decode path. It
was born WITH this oracle rather than gated after the fact, for a reason
this project paid weeks to learn: the flash forward kernel returned
wrong answers at head-dim specialisation 128 and 256 (a Triton codegen
defect), and every byte gate stayed green because both arms carried the
same error. A new kernel compiling the same class of specialisation does
not ship a single shape untested against the truth.

Checked, separately:
  1. CORRECTNESS vs an independently written float64 reference at
     D = 127, 128 AND 129 — the exact triple that isolated the codegen
     defect — plus the real decode shapes (GQA 8:1, bucketed T_k, pad
     bias) and MHA.
  2. DETERMINISM: three calls, identical bytes. The replay engine
     verifies frozen plans byte-equal before adopting them, so a
     non-deterministic kernel is disqualified regardless of accuracy.
  3. The masked-tail contract: keys beyond the bias's -inf region must
     contribute nothing; a fully-masked row must emit 0, matching the
     nan_to_num guard of both existing SDPA paths.

The bound is the attention family's: 2e-03 relative for fp16 inputs
with fp32 accumulation (derived in test_numeric_correctness_oracle.py).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_flash_decode_oracle.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_flash_decode_oracle.py
"""
from __future__ import annotations

import ctypes

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor

BOUND = 2e-03

_NP = {NBXDtype.float32: (np.float32, 4), NBXDtype.float16: (np.float16, 2)}


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
    """Independent decode attention in float64: q [B,H,1,D], GQA by
    repeating K/V (fine in a float64 reference — exactness, not speed)."""
    B, H, _, D = qn.shape
    H_kv, T_k = kn.shape[1], kn.shape[2]
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


import os as _os


def _run(B, H, H_kv, T_k, D, bias_len=None, seed=0):
    # The kernel is OPT-IN since f6032cf (measured slower than the math
    # path at real bucket sizes). This oracle tests THE KERNEL, so it
    # arms the flag explicitly for the duration of the call.
    _os.environ["NBX_FLASH_DECODE"] = "1"
    rng = np.random.default_rng(seed)
    qn = rng.standard_normal((B, H, 1, D)).astype(np.float16)
    kn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
    vn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
    bias = None
    bias_t = None
    if bias_len is not None:
        # bucketed pad mask: 0 for live keys, -inf beyond bias_len
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


# The head-dim triple that isolated the codegen defect, plus real shapes.
_SHAPES = [
    # B, H, H_kv, T_k, D, live_len
    (1, 32, 4, 256, 127, None),
    (1, 32, 4, 256, 128, None),      # the hazard specialisation
    (1, 32, 4, 256, 129, None),
    (1, 32, 4, 4352, 128, 4165),     # canonical long-row decode: bucketed
    (1, 32, 4, 256, 128, 13),        # short row: 13 live keys of 256
    (1, 32, 32, 256, 128, None),     # MHA
    (1, 16, 2, 1024, 64, None),      # 8:1 at D=64
    (2, 8, 2, 512, 128, 300),        # batch 2
    (1, 32, 4, 8448, 128, 8200),     # deeper than the long row
]


@pytest.mark.parametrize("B,H,H_kv,T_k,D,live", _SHAPES)
def test_flash_decode_matches_float64(B, H, H_kv, T_k, D, live) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    err, _ = _run(B, H, H_kv, T_k, D, live)
    assert err <= BOUND, (
        f"B={B} H={H} H_kv={H_kv} T_k={T_k} D={D} live={live}: relative "
        f"error {err:.3e} exceeds {BOUND:.1e} by {err/BOUND:.0f}x. If D is "
        f"128 or 256, suspect the same codegen specialisation defect the "
        f"forward kernel had.")


@pytest.mark.parametrize("D", [127, 128, 129])
def test_flash_decode_is_deterministic(D) -> None:
    """Three calls, identical bytes — a replay-engine requirement."""
    if not _has_gpu():
        pytest.skip("no GPU")
    outs = [_run(1, 32, 4, 1024, D, 900, seed=5)[1].tobytes()
            for _ in range(3)]
    assert outs[0] == outs[1] == outs[2], (
        f"D={D}: outputs differ across identical calls — disqualified "
        f"for the replay engine regardless of accuracy")


def test_masked_tail_contributes_nothing() -> None:
    """Changing key values BEYOND the live length must not change the
    output at all — the bias contract, checked bitwise."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T_k, D, live = 1, 32, 4, 512, 128, 100
    rng = np.random.default_rng(9)
    qn = rng.standard_normal((B, H, 1, D)).astype(np.float16)
    kn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
    vn = rng.standard_normal((B, H_kv, T_k, D)).astype(np.float16)
    b = np.zeros(T_k, dtype=np.float32); b[live:] = -np.inf
    bias_t = _t(b.reshape(1, 1, 1, T_k))
    scale = 1.0 / np.sqrt(D)
    out1 = _d2h(W.scaled_dot_product_attention_wrapper(
        _t(qn), _t(kn), _t(vn), attn_mask=bias_t, scale=scale))
    kn2, vn2 = kn.copy(), vn.copy()
    kn2[:, :, live:] = 7.5     # garbage in the dead region
    vn2[:, :, live:] = -3.25
    out2 = _d2h(W.scaled_dot_product_attention_wrapper(
        _t(qn), _t(kn2), _t(vn2), attn_mask=bias_t, scale=scale))
    assert np.array_equal(out1, out2), (
        "keys beyond the -inf bias changed the output — the masked tail "
        "is leaking into the softmax")


def test_flash_decode_is_actually_routed_when_armed() -> None:
    """Activation proof: with NBX_FLASH_DECODE=1, T_q=1 takes the new
    path. `_run` arms the flag; the probe proves arrival."""
    if not _has_gpu():
        pytest.skip("no GPU")
    called = {"n": 0}
    orig = W._flash_decode

    def probe(*a, **k):
        called["n"] += 1
        return orig(*a, **k)
    W._flash_decode = probe
    try:
        _run(1, 32, 4, 256, 128, None)
    finally:
        W._flash_decode = orig
    assert called["n"] == 1, (
        f"_flash_decode called {called['n']} times for a T_q=1 shape "
        f"with the flag armed — the routing hook is not reaching it")


def test_flash_decode_is_off_by_default() -> None:
    """The twin: WITHOUT the flag, T_q=1 must go to the math path — the
    default was flipped OFF by measurement (f6032cf: -10% short, -14%
    long vs the tuned math path) and this pins that decision. If someone
    re-enables it by default, this fails and demands the measurement be
    redone rather than the flag drifting silently."""
    if not _has_gpu():
        pytest.skip("no GPU")
    called = {"n": 0}
    orig = W._flash_decode

    def probe(*a, **k):
        called["n"] += 1
        return orig(*a, **k)
    W._flash_decode = probe
    _os.environ.pop("NBX_FLASH_DECODE", None)
    try:
        import numpy as _np
        rng = _np.random.default_rng(0)
        q = _t(rng.standard_normal((1, 32, 1, 128)).astype(_np.float16))
        k = _t(rng.standard_normal((1, 4, 256, 128)).astype(_np.float16))
        v = _t(rng.standard_normal((1, 4, 256, 128)).astype(_np.float16))
        W.scaled_dot_product_attention_wrapper(
            q, k, v, scale=1.0 / _np.sqrt(128))
    finally:
        W._flash_decode = orig
    assert called["n"] == 0, (
        "flash_decode ran WITHOUT NBX_FLASH_DECODE=1 — the measured "
        "default-off decision has drifted")


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    print(f"bound {BOUND:.1e}\n")
    print(f"{'B':>2s} {'H':>3s} {'Hkv':>4s} {'T_k':>6s} {'D':>4s} {'live':>5s} {'rel err':>10s}")
    for shape in _SHAPES:
        try:
            err, _ = _run(*shape)
            ok = err <= BOUND
            if not ok:
                fails += 1
            B, H, Hk, Tk, D, lv = shape
            print(f"{B:2d} {H:3d} {Hk:4d} {Tk:6d} {D:4d} {str(lv):>5s} "
                  f"{err:10.2e}  {'ok' if ok else f'FAIL ({err/BOUND:.0f}x)'}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {shape}: {type(e).__name__}: {str(e)[:60]}")
    for name, fn in (("determinism D=127", lambda: test_flash_decode_is_deterministic(127)),
                     ("determinism D=128", lambda: test_flash_decode_is_deterministic(128)),
                     ("determinism D=129", lambda: test_flash_decode_is_deterministic(129)),
                     ("masked tail bitwise inert", test_masked_tail_contributes_nothing),
                     ("activation: armed flag routes to _flash_decode",
                      test_flash_decode_is_actually_routed_when_armed),
                     ("twin: default-off goes to math",
                      test_flash_decode_is_off_by_default)):
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {name}: {type(e).__name__}: {str(e)[:60]}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
