# Gemm - General Matrix Multiply (Y = alpha * A @ B + beta * C)
# Type: Triton Kernel (Composite)
# NeuroBrix - NVIDIA Common (All architectures)

import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    M, N, K,
    alpha, beta,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gemm: out = alpha * A @ B + beta * C"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply alpha
    acc = acc * alpha
    
    # Add beta * C if beta != 0
    if beta != 0.0:
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += beta * c
    
    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _gemm_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """Gemm using Triton."""
    assert a.is_cuda and b.is_cuda
    
    if trans_a:
        a = a.T.contiguous()
    if trans_b:
        b = b.T.contiguous()
    
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} @ {b.shape}"
    
    out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    
    if c is None:
        c = torch.zeros(M, N, dtype=a.dtype, device=a.device)
        beta = 0.0
    else:
        c = c.contiguous()
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        a, b, c, out,
        M, N, K,
        alpha, beta,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    
    return out


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="gemm")
def gemm_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """
    General Matrix Multiply: Y = alpha * A @ B + beta * C
    
    Args:
        a: Input matrix A
        b: Input matrix B  
        c: Optional bias matrix C
        alpha: Scalar for A @ B
        beta: Scalar for C
        trans_a: Transpose A
        trans_b: Transpose B
    
    Returns:
        Result matrix
    """
    return _gemm_impl(a, b, c, alpha, beta, trans_a, trans_b)
