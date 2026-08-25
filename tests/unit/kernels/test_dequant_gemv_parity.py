"""E7 unit test — fused int4 dequant-GEMV vs the tier's own byte oracle.

The quantized tier contract (scoping 2026-08-16, clause 1): the fused
kernel must be BYTE-IDENTICAL to dequantize-then-GEMV with the same
math and the same accumulation order. The oracle is built from the
family's own kernels (standalone dequant -> reference fp16 GEMV with
the identical reduction structure); a cuBLAS/torch reference can never
be byte-matched (internal reduction order) and is used only as a
TOLERANCE cross-check here.

Also validates the numpy packing reference (the future Forge
quantizer's contract): pack -> kernel-dequant round-trips bit-exactly
vs the numpy dequant, and RTN+min group error is within the int4
asymmetric bound (|err| <= scale/2).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_dequant_gemv_parity.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_dequant_gemv_parity.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    pytest = None

import numpy as np


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not _cuda_available(), reason="CUDA device required")

GROUP = 128
PACK = 8


# ── numpy packing reference (the Forge quantizer's contract) ─────────

def quantize_int4_g128(w: np.ndarray):
    """fp16 [K, N] -> (packed int32 [K//8, N], scales fp16 [K//128, N],
    mins fp16 [K//128, N]). Asymmetric RTN: per group g along K,
    min = min(w_g), scale = (max-min)/15, q = round((w-min)/scale) in
    [0, 15]; dequant = q*scale + min."""
    K, N = w.shape
    assert K % GROUP == 0
    wg = w.reshape(K // GROUP, GROUP, N).astype(np.float32)
    mins = wg.min(axis=1)
    scales = (wg.max(axis=1) - mins) / 15.0
    scales = np.where(scales == 0.0, 1.0, scales)  # constant group
    q = np.rint((wg - mins[:, None, :]) / scales[:, None, :])
    q = np.clip(q, 0, 15).astype(np.uint32).reshape(K, N)
    packed = np.zeros((K // PACK, N), dtype=np.uint32)
    for i in range(PACK):
        packed |= q[i::PACK, :] << (4 * i)
    return (packed.view(np.int32),
            scales.astype(np.float16), mins.astype(np.float16))


def dequant_int4_np(packed: np.ndarray, scales: np.ndarray,
                    mins: np.ndarray) -> np.ndarray:
    """Numpy dequant mirror of the CANONICAL form: PURE fp32
    (llama.cpp dmmv default dfloat=float — no fp16 materialization;
    q*scale exact in fp32 -> contraction-immune). The fp16 round-trip
    variant was REJECTED by measurement: the compiler elides
    f32->f16->f32 in the fused kernel, so an fp16-dense oracle can
    never byte-match the fused path."""
    Kp, N = packed.shape
    K = Kp * PACK
    q = np.zeros((K, N), dtype=np.uint32)
    pu = packed.view(np.uint32)
    for i in range(PACK):
        q[i::PACK, :] = (pu >> (4 * i)) & 0xF
    s = np.repeat(scales.astype(np.float32), GROUP, axis=0)
    m = np.repeat(mins.astype(np.float32), GROUP, axis=0)
    return q.astype(np.float32) * s + m


# ── GPU plumbing ─────────────────────────────────────────────────────

def _nbx(arr):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _dl(t, np_dtype):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.data_ptr(), t._nbytes, 2)
    return buf.view(np_dtype).reshape(t.shape)


def _run_family(x_np, w_np, block_n=None):
    """Run oracle chain + fused kernel on GPU; return (dense_fp32,
    oracle_out_fp32, fused_out_fp32) as numpy arrays. `block_n`
    overrides the pinned parity BLOCK_N — oracle AND fused run the
    SAME width (the tier contract holds at every width; the wrapper's
    occupancy rule adapts it at small N)."""
    import triton
    from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
    from neurobrix.kernels.ops.dequant_gemv import (
        BLOCK_K, BLOCK_N, NUM_WARPS, dequant_int4_kernel,
        dequant_gemv_int4_kernel, gemv_ref_kernel)
    from neurobrix.kernels.nbx_tensor import NBXDtype

    K, N = w_np.shape
    if block_n is None:
        block_n = BLOCK_N
    packed, scales, mins = quantize_int4_g128(w_np)
    x = _nbx(x_np)
    wq = _nbx(packed)
    sc = _nbx(scales)
    mn = _nbx(mins)
    DeviceAllocator.set_device(x._device_idx)

    dense = NBXTensor.empty((K, N), dtype=NBXDtype.float32, device="cuda")
    dequant_int4_kernel[(triton.cdiv(K, BLOCK_K), triton.cdiv(N, block_n))](
        wq, sc, mn, dense, K, N,
        wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
        dense.stride(0), dense.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=block_n, GROUP_C=GROUP, PACK_C=PACK,
        num_warps=NUM_WARPS, num_stages=2)

    oracle = NBXTensor.empty((N,), dtype=NBXDtype.float32, device="cuda")
    gemv_ref_kernel[(triton.cdiv(N, block_n),)](
        x, dense, oracle, K, N, dense.stride(0), dense.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=block_n,
        num_warps=NUM_WARPS, num_stages=2)

    fused = NBXTensor.empty((N,), dtype=NBXDtype.float32, device="cuda")
    dequant_gemv_int4_kernel[(triton.cdiv(N, block_n),)](
        x, wq, sc, mn, fused, K, N,
        wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=block_n, GROUP_C=GROUP, PACK_C=PACK,
        num_warps=NUM_WARPS, num_stages=2)

    return (_dl(dense, np.float32), _dl(oracle, np.float32),
            _dl(fused, np.float32))


SHAPES = [(2048, 2048),   # Qwen3-30B-A3B hidden-square class
          (2048, 768),    # expert down-proj class
          (768, 2048),    # expert up-proj class
          (4096, 4096)]   # dense 7B class


def test_pack_roundtrip_and_rtn_bound():
    """Kernel dequant == numpy dequant bit-exactly; RTN error bound."""
    rng = np.random.default_rng(11)
    K, N = 1024, 256
    w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    packed, scales, mins = quantize_int4_g128(w)
    dense_gpu, _, _ = _run_family(
        rng.standard_normal(K).astype(np.float16), w)
    dense_np = dequant_int4_np(packed, scales, mins)
    assert np.array_equal(
        dense_gpu.view(np.uint32), dense_np.view(np.uint32)), \
        "kernel dequant != numpy dequant (bit compare)"
    err = np.abs(dense_np.astype(np.float32) - w.astype(np.float32))
    bound = np.repeat(scales.astype(np.float32), GROUP, axis=0) * 0.5 + 2e-3
    assert (err <= bound).all(), \
        f"RTN error exceeds scale/2 bound (max {err.max()})"


def test_fused_byte_equals_oracle():
    """THE implementation gate: fused == dequant->GEMV, byte-for-byte,
    across the proof-row shapes. Includes the activation proof (the
    fused output is non-trivial and K-dependent) and a tolerance
    cross-check vs float64 numpy."""
    rng = np.random.default_rng(7)
    for K, N in SHAPES:
        x = (rng.standard_normal(K) * 0.5).astype(np.float16)
        w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
        dense, oracle, fused = _run_family(x, w)
        assert np.array_equal(oracle.view(np.uint32),
                              fused.view(np.uint32)), \
            f"BYTE GATE FAILED at K={K} N={N}: " \
            f"maxdiff={np.abs(oracle - fused).max()}"
        # activation proof: a real reduction happened
        assert np.abs(fused).sum() > 0 and np.isfinite(fused).all()
        # tolerance cross-check vs f64 ground truth of the DEQUANT math
        ref = dense.astype(np.float64).T @ x.astype(np.float64)
        rel = np.abs(fused - ref) / (np.abs(ref) + 1e-6)
        assert rel.max() < 5e-3, f"drift vs f64 ref: {rel.max()}"


def test_fused_byte_equals_oracle_at_occupancy_widths():
    """The occupancy rule's parity proof at the EXPLICIT widths the
    V100 rule selects (router N=128 -> BLOCK_N=1; full-int4
    projection N=2048 -> BLOCK_N=8) — fused == oracle byte-for-byte
    and matches the float64 reference at every width. Explicit widths,
    not the rule's output, so the proof can never go vacuous when the
    process has no stashed hardware profile."""
    rng = np.random.default_rng(23)
    for (K, N), bn in [((2048, 128), 1),    # MoE router mv
                       ((2048, 2048), 8),   # full-int4 projection class
                       ((2048, 128), 64)]:  # pinned width still proven
        x = (rng.standard_normal(K) * 0.5).astype(np.float16)
        w = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
        dense, oracle, fused = _run_family(x, w, block_n=bn)
        assert np.array_equal(oracle.view(np.uint32),
                              fused.view(np.uint32)), \
            f"BYTE GATE FAILED at K={K} N={N} bn={bn}"
        assert np.abs(fused).sum() > 0 and np.isfinite(fused).all()
        ref = dense.astype(np.float64).T @ x.astype(np.float64)
        rel = np.abs(fused - ref) / (np.abs(ref) + 1e-6)
        assert rel.max() < 5e-3, \
            f"drift vs f64 ref at bn={bn}: {rel.max()}"


def test_occupancy_rule_resolves_vendor_config():
    """COUNTED activation proof for the rule itself: with a stashed
    profile whose arch is volta, _dequant_gemv_block_n narrows the
    router shape to 1 and leaves the huge-N grid pinned; with no
    profile, it degrades to the pinned width (prior behavior)."""
    import neurobrix.kernels.wrappers as W

    class _Dev:
        brand = "nvidia"
        architecture = "volta"

    class _Prof:
        devices = [_Dev()]

    saved_prof = W.get_hardware_profile()
    saved_cache = W._DEQUANT_GEMV_MIN_PROGRAMS
    try:
        W._NBX_HW_PROFILE = _Prof()
        W._DEQUANT_GEMV_MIN_PROGRAMS = None  # force re-resolution
        assert W._dequant_gemv_block_n(128) == 1, "router shape not narrowed"
        assert W._dequant_gemv_block_n(2048) == 8, "projection not narrowed"
        assert W._dequant_gemv_block_n(151936) == 64, "huge-N was touched"
        W._NBX_HW_PROFILE = None
        W._DEQUANT_GEMV_MIN_PROGRAMS = None
        assert W._dequant_gemv_block_n(128) == 64, \
            "no-profile fallback lost the pinned width"
        import os as _osl
        W._NBX_HW_PROFILE = _Prof()
        W._DEQUANT_GEMV_MIN_PROGRAMS = None
        _osl.environ["NBX_DGEMV_ADAPT"] = "0"
        try:
            assert W._dequant_gemv_block_n(128) == 64, \
                "kill switch '0' did not restore the pinned width"
        finally:
            _osl.environ.pop("NBX_DGEMV_ADAPT", None)
    finally:
        W._NBX_HW_PROFILE = saved_prof
        W._DEQUANT_GEMV_MIN_PROGRAMS = saved_cache


if __name__ == "__main__":
    if not _cuda_available():
        raise SystemExit("CUDA device required")
    test_pack_roundtrip_and_rtn_bound()
    print("pack round-trip + RTN bound: PASS")
    test_fused_byte_equals_oracle()
    print(f"BYTE GATE: fused == oracle on {len(SHAPES)} shapes — ALL PASS")
    test_fused_byte_equals_oracle_at_occupancy_widths()
    print("BYTE GATE at occupancy widths (router/full-int4): PASS")
    test_occupancy_rule_resolves_vendor_config()
    print("occupancy rule resolution + kill switch (counted): PASS")
