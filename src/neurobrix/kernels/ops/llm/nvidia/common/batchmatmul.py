"""
Batch Matrix Multiply - Batched matrix multiplication
ATen: aten::bmm
Pure Triton implementation.
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Batched matmul kernel - pure Triton."""
    pid_batch = tl.program_id(2)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + pid_batch * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + pid_batch * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_batch * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="batchmatmul")
def batchmatmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch matrix multiply - pure Triton."""
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 3 and b.ndim == 3

    a = a.contiguous()
    b = b.contiguous()

    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, f"Shape mismatch: {a.shape} @ {b.shape}"

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, B)

    with torch.cuda.device(a.device):
        _bmm_kernel[grid](
            a, b, c,
            B, M, N, K,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=4,
        )

    return c

