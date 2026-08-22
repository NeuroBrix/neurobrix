"""CORRECTNESS ORACLE — int4 dequant-GEMV, the published model's core.

This kernel executes every expert matmul of the int4 artifact published
to the hub, whose card claims a quality inside its class band. That claim
rests on this kernel being right.

=== WHAT THE EXISTING TEST DOES AND DOES NOT COVER ===

`test_dequant_gemv_parity.py` is a good BYTE gate and says so: it proves
the fused kernel is byte-identical to dequantise-then-GEMV built from the
family's own kernels. It also checks the dequantised weights against a
numpy mirror and against the original weights' RTN bound — but at ONE
shape (K=1024, N=256).

Its float64 cross-check is the gap:

    ref = dense.astype(np.float64).T @ x.astype(np.float64)

`dense` is the output of OUR OWN dequant kernel. A reference built from
the thing under test cannot fail when the thing under test is wrong in a
way both halves share — a misapplied scale, a swapped group index, a
nibble order read consistently backwards. The byte gate would stay green
and so would that check.

=== WHAT THIS FILE ADDS ===

A reference computed from the PACKED BYTES the kernel receives, unpacked
and dequantised in float64 from the format definition, with **no
NeuroBrix kernel anywhere in the reference path**, at the real shapes of
the published artifact.

If the kernel and the numpy packer shared a misunderstanding of the
format, this catches it; if the GEMV misaccumulates, this catches it.

=== THE BOUND, DERIVED ===

Inputs: activations fp16 (2^-11 = 4.9e-04 per value), weights exactly
representable given their fp16 scale/min and 4-bit level, accumulation in
fp32 over K.

The reference uses the SAME fp16 activations and the SAME packed weights,
so quantisation error is common to both sides and cancels. What remains
is fp32 accumulation-order difference over K terms, plus any fp16
intermediate the kernel keeps. Worst realistic case 4.9e-04 * sqrt(K);
at K = 4096 that is 3.1e-02.

**Bound: 5e-02 relative on the max element.** Same class and same
derivation as the matmul family oracle. `test_bound_is_not_vacuous`
asserts it is above one fp16 rounding unit and far below the size of
defect this audit exists to catch.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_dequant_gemv_correctness.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_dequant_gemv_correctness.py
"""
from __future__ import annotations

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

GROUP = 128
PACK = 8

DEQUANT_GEMV_REL_BOUND = 5e-02


def _has_gpu() -> bool:
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _nbx(arr):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _dl(t, np_dtype):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.data_ptr(), t._nbytes, 2)
    return buf.view(np_dtype).reshape(t.shape)


def _pack_int4_g128(w: np.ndarray):
    """fp16 [K,N] -> packed/scales/mins, asymmetric RTN per group of 128.

    Written from the format definition. Kept here rather than imported so
    that a change to the shipped packer shows up as a disagreement rather
    than propagating silently into the reference.
    """
    K, N = w.shape
    assert K % GROUP == 0, f"K={K} not a multiple of {GROUP}"
    wg = w.reshape(K // GROUP, GROUP, N).astype(np.float32)
    mins = wg.min(axis=1)
    scales = (wg.max(axis=1) - mins) / 15.0
    scales = np.where(scales == 0.0, 1.0, scales)
    q = np.rint((wg - mins[:, None, :]) / scales[:, None, :])
    q = np.clip(q, 0, 15).astype(np.uint32).reshape(K, N)
    packed = np.zeros((K // PACK, N), dtype=np.uint32)
    for i in range(PACK):
        packed |= q[i::PACK, :] << (4 * i)
    return (packed.view(np.int32), scales.astype(np.float16),
            mins.astype(np.float16))


def _dequant_float64(packed, scales, mins) -> np.ndarray:
    """THE INDEPENDENT REFERENCE. Unpack and dequantise in float64 from
    the packed bytes alone — no NeuroBrix kernel touches this path."""
    Kp, N = packed.shape
    K = Kp * PACK
    q = np.zeros((K, N), dtype=np.uint32)
    pu = packed.view(np.uint32)
    for i in range(PACK):
        q[i::PACK, :] = (pu >> (4 * i)) & 0xF
    s = np.repeat(scales.astype(np.float64), GROUP, axis=0)
    m = np.repeat(mins.astype(np.float64), GROUP, axis=0)
    return q.astype(np.float64) * s + m


def _run_fused(x_np, packed, scales, mins, K, N):
    """Launch ONLY the fused kernel — the thing under test."""
    import triton
    from neurobrix.kernels.nbx_tensor import (NBXTensor, NBXDtype,
                                              DeviceAllocator)
    from neurobrix.kernels.ops.dequant_gemv import (
        BLOCK_K, BLOCK_N, NUM_WARPS, dequant_gemv_int4_kernel)
    x = _nbx(x_np)
    wq, sc, mn = _nbx(packed), _nbx(scales), _nbx(mins)
    DeviceAllocator.set_device(x._device_idx)
    out = NBXTensor.empty((N,), dtype=NBXDtype.float32, device="cuda")
    dequant_gemv_int4_kernel[(triton.cdiv(N, BLOCK_N),)](
        x, wq, sc, mn, out, K, N,
        wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=BLOCK_N, GROUP_C=GROUP, PACK_C=PACK,
        num_warps=NUM_WARPS, num_stages=2)
    return _dl(out, np.float32)


# Real shapes of the published artifact, plus awkward neighbours.
# Qwen3-30B-A3B: hidden 2048; expert intermediate 768.
_SHAPES = [
    (2048, 2048),    # hidden-square
    (2048, 768),     # expert down-projection
    (768, 2048),     # expert up / gate projection
    (4096, 4096),    # dense 7B class
    (128, 128),      # exactly one group
    (256, 64),       # two groups, narrow N
    (1152, 320),     # K = 9 groups, N not a multiple of any block
    (2048, 1),       # single output column
]


@pytest.mark.parametrize("K,N", _SHAPES)
@pytest.mark.parametrize("seed", [3, 17])
def test_dequant_gemv_matches_float64(K, N, seed) -> None:
    """The check the byte gate cannot make."""
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(seed * 1000 + K)
    x = (rng.standard_normal(K) * 0.5).astype(np.float16)
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    packed, scales, mins = _pack_int4_g128(w)

    got = _run_fused(x, packed, scales, mins, K, N).astype(np.float64)
    ref = _dequant_float64(packed, scales, mins).T @ x.astype(np.float64)

    err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
    assert err <= DEQUANT_GEMV_REL_BOUND, (
        f"K={K} N={N} seed={seed}: relative error {err:.3e} exceeds "
        f"{DEQUANT_GEMV_REL_BOUND:.1e} by {err/DEQUANT_GEMV_REL_BOUND:.0f}x. "
        f"This is the execution core of the PUBLISHED int4 artifact — a "
        f"failure here is an escalation, not a test result.")


@pytest.mark.parametrize("K,N", [(2048, 768), (768, 2048)])
def test_dequant_weights_match_float64(K, N) -> None:
    """Separate the two halves: are the DEQUANTISED WEIGHTS right?

    If this passes and the GEMV test fails, the accumulation is at fault;
    if this fails, the format handling is. Splitting them means a failure
    names its own half.
    """
    if not _has_gpu():
        pytest.skip("no GPU")
    import triton
    from neurobrix.kernels.nbx_tensor import (NBXTensor, NBXDtype,
                                              DeviceAllocator)
    from neurobrix.kernels.ops.dequant_gemv import (
        BLOCK_K, BLOCK_N, NUM_WARPS, dequant_int4_kernel)
    rng = np.random.default_rng(K + N)
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    packed, scales, mins = _pack_int4_g128(w)
    wq, sc, mn = _nbx(packed), _nbx(scales), _nbx(mins)
    DeviceAllocator.set_device(wq._device_idx)
    dense = NBXTensor.empty((K, N), dtype=NBXDtype.float32, device="cuda")
    dequant_int4_kernel[(triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))](
        wq, sc, mn, dense, K, N,
        wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
        dense.stride(0), dense.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=BLOCK_N, GROUP_C=GROUP, PACK_C=PACK,
        num_warps=NUM_WARPS, num_stages=2)
    got = _dl(dense, np.float32).astype(np.float64)
    ref = _dequant_float64(packed, scales, mins)
    err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
    # dequant is q*scale+min in fp32 vs float64 — one rounding, no
    # accumulation, so this bound is much tighter than the GEMV's.
    assert err <= 1e-06, (
        f"K={K} N={N}: dequantised weights differ from an independent "
        f"float64 unpack by {err:.3e}. The format handling is wrong.")


def test_quantisation_error_is_within_the_int4_class() -> None:
    """The published card claims a class-band quality. That rests on the
    round-trip error being what 4-bit asymmetric RTN can deliver: at most
    half a level, i.e. scale/2 per element."""
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(101)
    K, N = 2048, 768
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    packed, scales, mins = _pack_int4_g128(w)
    deq = _dequant_float64(packed, scales, mins)
    err = np.abs(deq - w.astype(np.float64))
    bound = np.repeat(scales.astype(np.float64), GROUP, axis=0) * 0.5 + 1e-3
    assert (err <= bound).all(), (
        f"RTN round-trip exceeds half a quantisation level: "
        f"max {err.max():.3e}")


def test_bound_is_not_vacuous() -> None:
    assert DEQUANT_GEMV_REL_BOUND < 3.4e-01 / 5, "bound too loose"
    assert DEQUANT_GEMV_REL_BOUND >= 4.9e-04, "bound below one fp16 unit"


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    print(f"declared bound: {DEQUANT_GEMV_REL_BOUND:.1e} relative\n")
    print(f"{'K':>6s} {'N':>6s} {'seed':>5s} {'rel err':>10s}  verdict")
    for K, N in _SHAPES:
        for seed in (3, 17):
            try:
                rng = np.random.default_rng(seed * 1000 + K)
                x = (rng.standard_normal(K) * 0.5).astype(np.float16)
                w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
                packed, scales, mins = _pack_int4_g128(w)
                got = _run_fused(x, packed, scales, mins, K, N).astype(np.float64)
                ref = _dequant_float64(packed, scales, mins).T @ x.astype(np.float64)
                err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
                ok = err <= DEQUANT_GEMV_REL_BOUND
                if not ok:
                    fails += 1
                print(f"{K:6d} {N:6d} {seed:5d} {err:10.2e}  "
                      f"{'ok' if ok else f'FAIL ({err/DEQUANT_GEMV_REL_BOUND:.0f}x)'}")
            except Exception as e:
                fails += 1
                print(f"{K:6d} {N:6d} {seed:5d}  ERROR {type(e).__name__}: {str(e)[:45]}")
    print()
    for name, fn in (("dequant weights vs float64 (2048x768)",
                      lambda: test_dequant_weights_match_float64(2048, 768)),
                     ("dequant weights vs float64 (768x2048)",
                      lambda: test_dequant_weights_match_float64(768, 2048)),
                     ("RTN round-trip within half a level",
                      test_quantisation_error_is_within_the_int4_class),
                     ("bound not vacuous", test_bound_is_not_vacuous)):
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {name}: {type(e).__name__}: {str(e)[:60]}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} CORRECTNESS FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
