"""
Einsum kernel - Einstein summation convention
ATen: aten::einsum
Pure Triton implementation for common patterns.
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _einsum_bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matmul for einsum - pure Triton."""
    pid_batch = tl.program_id(2)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_batch * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + pid_batch * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + pid_batch * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _einsum_outer_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_a, stride_b,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Outer product for einsum - pure Triton."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(a_ptr + offs_m * stride_a, mask=offs_m < M, other=0.0)
    b = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)

    c = a[:, None] * b[None, :]

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=c_mask)


def _einsum_bmm(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
    """Batched matrix multiply for einsum patterns."""
    assert a.is_cuda and b.is_cuda

    if trans_a:
        a = a.transpose(-2, -1)
    if trans_b:
        b = b.transpose(-2, -1)

    a = a.contiguous()
    b = b.contiguous()

    # Handle broadcasting
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    B = max(a.shape[0], b.shape[0])
    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    assert K == K2

    if a.shape[0] == 1:
        a = a.expand(B, -1, -1).contiguous()
    if b.shape[0] == 1:
        b = b.expand(B, -1, -1).contiguous()

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, B)

    with torch.cuda.device(a.device):
        _einsum_bmm_kernel[grid](
            a, b, c,
            B, M, N, K,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

    return c


def _einsum_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer product for einsum."""
    assert a.is_cuda and b.is_cuda
    a = a.contiguous().flatten()
    b = b.contiguous().flatten()

    M = a.numel()
    N = b.numel()

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    with torch.cuda.device(a.device):
        _einsum_outer_kernel[grid](
            a, b, c,
            M, N,
            a.stride(0), b.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )

    return c


def _parse_einsum(equation: str, operands: tuple):
    """Parse einsum equation and route to appropriate kernel."""
    # Split equation
    parts = equation.replace(' ', '').split('->')
    inputs = parts[0].split(',')

    # Common patterns
    if len(operands) == 2:
        a, b = operands
        in1, in2 = inputs

        # bmm patterns: bij,bjk->bik or similar
        if len(in1) == 3 and len(in2) == 3:
            # Check for matmul pattern
            if in1[0] == in2[0]:  # Same batch dim
                # bij,bjk -> bik (standard bmm)
                if in1[2] == in2[1]:
                    return _einsum_bmm(a, b, trans_a=False, trans_b=False)
                # bij,bkj -> bik (b transposed)
                if in1[2] == in2[2]:
                    return _einsum_bmm(a, b, trans_a=False, trans_b=True)
                # bji,bjk -> bik (a transposed)
                if in1[1] == in2[1]:
                    return _einsum_bmm(a, b, trans_a=True, trans_b=False)

        # 2D matmul: ij,jk->ik
        if len(in1) == 2 and len(in2) == 2:
            if in1[1] == in2[0]:
                return _einsum_bmm(a.unsqueeze(0), b.unsqueeze(0), False, False).squeeze(0)
            if in1[1] == in2[1]:
                return _einsum_bmm(a.unsqueeze(0), b.unsqueeze(0), False, True).squeeze(0)
            if in1[0] == in2[0]:
                return _einsum_bmm(a.unsqueeze(0), b.unsqueeze(0), True, False).squeeze(0)

        # Outer product: i,j->ij
        if len(in1) == 1 and len(in2) == 1:
            return _einsum_outer(a, b)

    # For unsupported patterns, fall back to decomposition via matmul
    raise NotImplementedError(f"Einsum pattern '{equation}' not implemented in pure Triton")


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="einsum")
def einsum_kernel(equation: str, *operands) -> torch.Tensor:
    """
    Einsum operation - pure Triton for common patterns.
    """
    try:
        return _parse_einsum(equation, operands)
    except NotImplementedError:
        # For complex patterns, use tensor decomposition
        # This still uses pure Triton via the matmul kernel
        raise RuntimeError(f"Einsum pattern '{equation}' not supported")
