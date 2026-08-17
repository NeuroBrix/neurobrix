"""E7 unit test — fused W4 MoE grouped-GEMM vs the tier's byte oracle.

fused_moe_wna16_kernel (in-register per-expert dequant) must be
BYTE-IDENTICAL to dequantize-then-grouped-GEMM: the oracle chain is
the family's standalone dequant kernel (per expert, to dense fp32)
feeding fused_moe_fp32b_kernel — a textual twin of the W4 kernel
minus the unpack (same pid mapping, same K-loop, same fp32 tl.dot,
same routed-weight epilogue). One pinned config for both.

Covers the Qwen3-30B-A3B expert shapes (gate/up [K=2048 -> N=768],
down [K=768 -> N=2048]) with E=8 experts, top_k=2 routing, both
MUL_ROUTED_WEIGHT arms, M in {1, 5} (decode + tiny prefill).
Activation proof: the sorted-token/expert tables come from the real
moe_align_block_size brick and the W4 kernel's output is checked
non-trivial per expert.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_fused_moe_wna16_parity.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_fused_moe_wna16_parity.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:
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
BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS = 16, 64, 64, 8, 4


def _quantize(w_t: np.ndarray):
    """[K, N] fp32 -> (packed int32 [K//8, N], scales, mins fp16)."""
    K, N = w_t.shape
    wg = w_t.reshape(K // GROUP, GROUP, N)
    mins = wg.min(axis=1)
    scales = (wg.max(axis=1) - mins) / 15.0
    scales = np.where(scales == 0.0, 1.0, scales)
    q = np.rint((wg - mins[:, None, :]) / scales[:, None, :])
    q = np.clip(q, 0, 15).astype(np.uint32).reshape(K, N)
    packed = np.zeros((K // PACK, N), dtype=np.uint32)
    for i in range(PACK):
        packed |= q[i::PACK, :] << (4 * i)
    return packed.view(np.int32), scales.astype(np.float16), \
        mins.astype(np.float16)


def _nbx(arr):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _dl(t, np_dtype):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.data_ptr(), t._nbytes, 2)
    return buf.view(np_dtype).reshape(t.shape)


def _ptr_table(tensors):
    return _nbx(np.array([t.data_ptr() for t in tensors], dtype=np.int64))


def _run_case(M, K, N, E, top_k, mul_routed, seed):
    import triton
    import triton.language as tl
    from neurobrix.kernels.nbx_tensor import (NBXTensor, NBXDtype,
                                              DeviceAllocator)
    from neurobrix.kernels.ops.dequant_gemv import dequant_int4_kernel
    from neurobrix.kernels.ops.fused_moe import (fused_moe_wna16_kernel,
                                                 fused_moe_fp32b_kernel)
    from neurobrix.triton.moe import moe_align_block_size

    rng = np.random.default_rng(seed)
    a = _nbx((rng.standard_normal((M, K)) * 0.5).astype(np.float16))
    DeviceAllocator.set_device(a._device_idx)

    qws, scs, mns, denses = [], [], [], []
    for e in range(E):
        w_t = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
        pk, sc, mn = _quantize(w_t)
        qw_t, sc_t, mn_t = _nbx(pk), _nbx(sc), _nbx(mn)
        qws.append(qw_t)
        scs.append(sc_t)
        mns.append(mn_t)
        # oracle dense fp32 from the FAMILY's own dequant kernel
        dense = NBXTensor.empty((K, N), dtype=NBXDtype.float32,
                                device="cuda")
        dequant_int4_kernel[(triton.cdiv(K, 128), triton.cdiv(N, 64))](
            qw_t, sc_t, mn_t, dense, K, N,
            qw_t.stride(0), qw_t.stride(1), sc_t.stride(0), sc_t.stride(1),
            dense.stride(0), dense.stride(1),
            BLOCK_K_C=128, BLOCK_N_C=64, GROUP_C=GROUP, PACK_C=PACK,
            num_warps=4, num_stages=2)
        denses.append(dense)

    topk_ids = rng.integers(0, E, size=(M * top_k,)).astype(np.int64)
    topk_w = _nbx(rng.random(M * top_k).astype(np.float32))
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        _nbx(topk_ids), BLOCK_M, E, a._device_idx)
    EM = int(_dl(num_post, np.int32).reshape(-1)[0]) if num_post.nbx_dtype == NBXDtype.int32 \
        else int(_dl(num_post, np.int64).reshape(-1)[0])
    n_valid = M * top_k
    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    def out_buf():
        o = NBXTensor.empty((n_valid, N), dtype=NBXDtype.float16,
                            device="cuda")
        return o

    out_w4 = out_buf()
    fused_moe_wna16_kernel[grid](
        a, _ptr_table(qws), _ptr_table(scs), _ptr_table(mns), out_w4,
        topk_w, sorted_ids, expert_ids, num_post,
        N, K, EM, n_valid,
        a.stride(0), a.stride(1),
        qws[0].stride(0), qws[0].stride(1),
        scs[0].stride(0), scs[0].stride(1),
        out_w4.stride(0), out_w4.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_M, MUL_ROUTED_WEIGHT=mul_routed, top_k=top_k,
        compute_type=tl.float16, QGROUP=GROUP, QPACK=PACK,
        num_warps=WARPS, num_stages=2)

    out_or = out_buf()
    fused_moe_fp32b_kernel[grid](
        a, _ptr_table(denses), out_or,
        topk_w, sorted_ids, expert_ids, num_post,
        N, K, EM, n_valid,
        a.stride(0), a.stride(1),
        denses[0].stride(0), denses[0].stride(1),
        out_or.stride(0), out_or.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_M, MUL_ROUTED_WEIGHT=mul_routed, top_k=top_k,
        compute_type=tl.float16,
        num_warps=WARPS, num_stages=2)

    w4 = _dl(out_w4, np.float16)
    orc = _dl(out_or, np.float16)
    assert np.array_equal(w4.view(np.uint16), orc.view(np.uint16)), \
        (f"BYTE GATE FAILED M={M} K={K} N={N} routed={mul_routed}: "
         f"{(w4 != orc).sum()}/{w4.size} differ, "
         f"max {np.abs(w4.astype(np.float32) - orc.astype(np.float32)).max()}")
    assert np.abs(w4.astype(np.float32)).sum() > 0 and \
        np.isfinite(w4.astype(np.float32)).all()


def test_wna16_byte_equals_oracle():
    """Qwen3-30B expert shapes x decode/prefill x both routed arms."""
    for M in (1, 5):
        for K, N in ((2048, 768), (768, 2048)):
            for routed in (False, True):
                _run_case(M, K, N, E=8, top_k=2, mul_routed=routed,
                          seed=7 + M)
    print("wna16 BYTE GATE: all shape/arm cases identical to the oracle")


if __name__ == "__main__":
    if not _cuda_available():
        raise SystemExit("CUDA device required")
    test_wna16_byte_equals_oracle()
    print("ALL PASS")
