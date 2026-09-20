"""The grouped MoE kernel, driven through a pinned captured expert table.

This is the contract the future MoE wiring stands on, proven at the kernel the
refusal in `execute_moe_fused` protects, at the geometry of the model named for
the lift (granite-3.1-1b-a400m: 32 experts, 8 per token, hidden 1024, bf16),
judged against a numpy fp64 oracle — an instrument outside the engine.

Only the POSITIVE arm is asserted. The two failing arms — a data_ptr table
(today's `_build_ptr_tables` law) and the same captured table after its scope
died — are measured in `demo_moe_table_pinned.py` (zeros, nothing raised, both)
and are not asserted here for the reason `pinned_addresses`' own tests state:
outside the scope the read is an ACCIDENT, and a test on an accident is a coin.

The refusal itself stays until a real mixture-of-experts model runs and is
judged; this test is the kernel-level floor under that day, not the day.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from neurobrix.kernels.nbx_tensor import (  # noqa: E402
    NBXDtype, NBXTensor, bf16_carrier_to_float32,
)
from neurobrix.kernels.launcher import launch  # noqa: E402
from neurobrix.kernels.ops.fused_moe import fused_moe_kernel  # noqa: E402

E, TOP_K, K, N = 32, 8, 1024, 512
BM, BN, BK = 16, 64, 32


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the pinned-table contract is this driver's")
        from neurobrix.triton import triton_ext_driver as d
        return d
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


@triton.jit
def _capture(src_ptr, tab_ptr, i):
    tl.store(tab_ptr + i, tl.cast(src_ptr, tl.int64, bitcast=True))


def _bf16(a32):
    return ((np.asarray(a32, dtype=np.float32).view(np.uint32) + 0x7FFF)
            >> 16).astype(np.uint16)


def _val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def test_the_grouped_moe_kernel_matches_fp64_through_a_pinned_table():
    drv = _metal_or_skip()
    rng = np.random.default_rng(7)

    a_bits = _bf16(rng.standard_normal((1, K)) * 0.05)
    a = NBXTensor.from_numpy(a_bits, dtype=NBXDtype.bfloat16)
    experts, experts64 = [], []
    for _ in range(E):
        w = _bf16(rng.standard_normal((K, N)) * 0.05)
        experts64.append(_val(w))
        experts.append(NBXTensor.from_numpy(w, dtype=NBXDtype.bfloat16))

    chosen = np.array([3, 17, 4, 29, 11, 0, 25, 8], dtype=np.int32)
    weights = rng.random(TOP_K).astype(np.float32)
    weights /= weights.sum()

    EM = TOP_K * BM
    sorted_ids = np.full(EM, TOP_K, dtype=np.int32)
    for s in range(TOP_K):
        sorted_ids[s * BM] = s

    t_w = NBXTensor.from_numpy(weights)
    t_sid = NBXTensor.from_numpy(sorted_ids)
    t_eid = NBXTensor.from_numpy(chosen)
    t_np = NBXTensor.from_numpy(np.array([EM], dtype=np.int32))

    a64 = _val(a_bits)
    oracle = np.stack([weights[s] * (a64[0] @ experts64[chosen[s]])
                       for s in range(TOP_K)])

    with drv.pinned_addresses(*experts):
        tab = NBXTensor.from_numpy(np.zeros(E, dtype=np.int64))
        for e in range(E):
            launch(_capture, (1,), experts[e], tab, e)
        c = NBXTensor.from_numpy(_bf16(np.zeros((TOP_K, N))),
                                 dtype=NBXDtype.bfloat16)
        grid = (triton.cdiv(EM, BM) * triton.cdiv(N, BN),)
        launch(fused_moe_kernel, grid, a, tab, c, t_w, t_sid, t_eid, t_np,
               N, K, EM, TOP_K, K, 1, N, 1, N, 1,
               BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
               GROUP_SIZE_M=1, MUL_ROUTED_WEIGHT=True, top_k=TOP_K,
               compute_type=tl.bfloat16)
        got = bf16_carrier_to_float32(c.numpy()).astype(np.float64)

    rel = np.abs(got - oracle) / np.maximum(np.abs(oracle), 1e-3)
    live = int((np.abs(got) > 0).sum())
    assert live == TOP_K * N, (
        f"only {live}/{TOP_K*N} outputs are nonzero: the expert table did not "
        f"resolve — the silent-zeros defect, through a PINNED table")
    assert float(rel.max()) < 5e-2, (
        f"the grouped MoE kernel diverges from the fp64 oracle through a "
        f"pinned table: max rel {rel.max():.3e}")
