"""Addmv — pure @triton.jit kernel.

Ported from FlagGems addmv (Apache-2.0 license).
Computes: out = beta * self + alpha * (mat @ vec)
where mat is [N, M], vec is [M], self is broadcastable to [N].
"""

import triton
import triton.language as tl


@triton.jit
def addmv_kernel(
    A_ptr,
    B_ptr,
    inp_ptr,
    out_ptr,
    N, M,
    alpha,
    beta,
    stride_an, stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Matrix-vector multiply with bias: out = beta*inp + alpha*(A @ B).

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

    # Reduce over M dimension, then add bias
    mv_result = tl.sum(acc, axis=1)[:, None]
    inp_ptrs = inp_ptr + offset_n * stride_in
    inp = tl.load(inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    out = mv_result * alpha + inp * beta

    out_ptrs = out_ptr + offset_n * stride_outn
    tl.store(out_ptrs, out, mask=n_mask)
