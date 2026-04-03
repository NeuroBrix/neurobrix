"""Matrix-vector multiply — pure @triton.jit kernel.

Ported from FlagGems mv (Apache-2.0 license).
Computes out[i] = sum_j(A[i, j] * v[j]) with tiled reduction.
"""

import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N, M,
    stride_an, stride_am,
    stride_bm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Matrix-vector multiply: C[n] = sum_m A[n, m] * B[m].

    Each program handles BLOCK_N rows of A, iterating over M in BLOCK_M tiles.
    Accumulates in fp32 for stability.
    """
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N

    A_ptrs = A_ptr + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B_ptr + offset_m * stride_bm

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    # Reduce over M dimension
    result = tl.sum(acc, axis=1)
    C_ptrs = C_ptr + offset_n * stride_cn
    tl.store(C_ptrs, result[:, None], mask=n_mask)
