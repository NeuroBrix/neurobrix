"""Batched addmm (baddbmm) — pure @triton.jit kernel.

Ported from FlagGems baddbmm (Apache-2.0 license).
Computes: out = beta * bias + alpha * (batch1 @ batch2)
where batch1 is [B, M, K], batch2 is [B, K, N], bias is broadcastable to [B, M, N].
"""

import triton
import triton.language as tl


@triton.jit
def baddbmm_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    bias_ptr,
    alpha,
    beta,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    bias_batch_stride,
    bias_m_stride,
    bias_n_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Batched matrix multiply with bias: out = beta*bias + alpha*(A @ B).

    Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, batch)
    Accumulates in fp32 for numerical stability.
    """
    # Batch offset
    pid_b = tl.program_id(2)
    A_ptr += pid_b * stride_ab
    B_ptr += pid_b * stride_bb
    out_ptr += pid_b * stride_ob
    bias_ptr += pid_b * bias_batch_stride

    # 2D tile indexing with grouping for L2 locality
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate matmul in fp32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        mask_b = mask_k[:, None] & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load bias and apply alpha/beta
    bias_ptrs = bias_ptr + offs_m[:, None] * bias_m_stride + offs_n[None, :] * bias_n_stride
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bi = tl.load(bias_ptrs, mask=out_mask, other=0.0)
    result = accumulator * alpha + bi * beta

    # Store output
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, result, mask=out_mask)
