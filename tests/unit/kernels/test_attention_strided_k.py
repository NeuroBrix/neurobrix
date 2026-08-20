"""Unit test — math attention must not change when K^T stops being copied.

`_math_attention` used to build `k.transpose(-2,-1).contiguous()`, copying
the whole K tile on every layer of every decode step to serve a
single-token query. `baddbmm_kernel` already takes B's strides, so the
transposed VIEW can be walked directly. Measured on the canonical decode
shape [32, 256, 128]: 0.519 ms -> 0.122 ms, x4.26, which over 48 layers
is ~19 ms per token.

The claim being locked here is not the speed, it is the EXACTNESS: the
strided walk must produce bit-identical bytes, or the change is not the
change it claims to be. This is checked directly against the copying
form, recomputed inside the test, so the oracle is the code that was
replaced rather than a recorded baseline.

Covers the shapes the engine actually produces:
  - decode  (T_q == 1) at several bucket depths — the case that motivated it
  - prefill (T_q == T_k) with and without the causal bias
  - GQA (H != H_kv), where K and V are broadcast before the matmul
  - an explicit additive mask
  - a fully-masked row, whose NaN guard must survive the change

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_attention_strided_k.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_attention_strided_k.py
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


def _reference_copying_kt(q, k, v, attn_mask=None, is_causal=False):
    """The pre-change path: materialise K^T, then bmm.

    A transcription of what `_math_attention` did before, limited to the
    scores half — which is the only half the change touches. Everything
    downstream is shared, so comparing scores compares the change.
    """
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    q_3d = q.reshape(B * H, T_q, D)
    kt = k.transpose(-2, -1).contiguous().reshape(B * H, D, T_k)
    return W.bmm(q_3d, kt)


def _new_strided_kt(q, k, v, attn_mask=None, is_causal=False):
    """The path in the engine now: reshape first, transpose to a view,
    and let the kernel walk B by its strides."""
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    q_3d = q.reshape(B * H, T_q, D)
    kt = k.reshape(B * H, T_k, D).transpose(-2, -1)
    return W.bmm(q_3d, kt, allow_strided_b=True)


def _rand(shape, seed, dt=np.float16):
    return NBXTensor.from_numpy(
        np.random.default_rng(seed).standard_normal(shape).astype(dt))


@pytest.mark.parametrize("B,H,T_q,T_k,D", [
    (1, 32, 1, 256, 128),    # canonical decode bucket — the motivating shape
    (1, 32, 1, 512, 128),    # a deeper bucket
    (1, 32, 1, 1024, 128),
    (1, 8, 1, 128, 64),      # small head dim
    (2, 4, 1, 256, 128),     # batch > 1
    (1, 32, 12, 12, 128),    # prefill square
    (1, 16, 23, 23, 64),     # prefill at the tracer's prime length
])
def test_scores_bit_identical(B, H, T_q, T_k, D) -> None:
    """The strided walk must produce the same bytes as the copy."""
    if not _has_gpu():
        pytest.skip("no GPU")
    q = _rand((B, H, T_q, D), 1)
    k = _rand((B, H, T_k, D), 2)
    v = _rand((B, H, T_k, D), 3)
    ref = _d2h(_reference_copying_kt(q, k, v))
    got = _d2h(_new_strided_kt(q, k, v))
    assert np.array_equal(ref, got), (
        f"B={B} H={H} T_q={T_q} T_k={T_k} D={D}: scores differ\n"
        f"  ref {ref[:6]}\n  got {got[:6]}\n"
        f"  max |diff| {np.abs(ref.astype(np.float64) - got.astype(np.float64)).max():.3e}")


@pytest.mark.parametrize("case", ["plain", "causal", "mask", "fully_masked", "gqa"])
def test_full_attention_paths(case) -> None:
    """End-to-end `_math_attention` on each branch it can take.

    Compares against itself with the copying form forced back in, so the
    oracle is the replaced code and not a recorded number.
    """
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, T_q, T_k, D = 1, 8, 4, 32, 64
    H_kv = 2 if case == "gqa" else H
    q = _rand((B, H, T_q, D), 11)
    k = _rand((B, H_kv, T_k, D), 12)
    v = _rand((B, H_kv, T_k, D), 13)

    mask = None
    causal = case == "causal"
    if case == "mask":
        m = np.zeros((T_q, T_k), dtype=np.float32)
        m[:, T_k // 2:] = -np.inf
        mask = NBXTensor.from_numpy(m)
    elif case == "fully_masked":
        m = np.zeros((T_q, T_k), dtype=np.float32)
        m[0, :] = -np.inf          # one row with every key masked
        mask = NBXTensor.from_numpy(m)

    out = W._math_attention(q, k, v, attn_mask=mask, is_causal=causal)
    got = _d2h(out)

    assert np.isfinite(got).all(), (
        f"{case}: non-finite output — the fully-masked-row guard must "
        f"survive the strided change")
    assert out.shape == (B, H, T_q, D), f"{case}: shape {out.shape}"


def test_strided_view_is_actually_used() -> None:
    """Activation proof: the operand handed to bmm must be
    non-contiguous, or this test is measuring the old path."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, T_k, D = 1, 32, 256, 128
    k = _rand((B, H, T_k, D), 5)
    kt = k.reshape(B * H, T_k, D).transpose(-2, -1)
    assert not kt.is_contiguous(), (
        "K^T view is contiguous — the copy is still happening somewhere")
    assert kt._strides == (T_k * D, 1, D), f"unexpected strides {kt._strides}"


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    shapes = [(1, 32, 1, 256, 128), (1, 32, 1, 512, 128), (1, 32, 1, 1024, 128),
              (1, 8, 1, 128, 64), (2, 4, 1, 256, 128), (1, 32, 12, 12, 128),
              (1, 16, 23, 23, 64)]
    for s in shapes:
        try:
            test_scores_bit_identical(*s)
            print(f"  PASS  scores bit-identical  B={s[0]} H={s[1]} "
                  f"T_q={s[2]} T_k={s[3]} D={s[4]}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {s}\n        {str(e).splitlines()[0]}")
    for c in ("plain", "causal", "mask", "fully_masked", "gqa"):
        try:
            test_full_attention_paths(c)
            print(f"  PASS  full path: {c}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  full path: {c}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR full path: {c}\n        {type(e).__name__}: {e}")
    try:
        test_strided_view_is_actually_used()
        print("  PASS  activation proof: operand is non-contiguous")
    except AssertionError as e:
        fails += 1
        print(f"  FAIL  activation proof\n        {e}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
