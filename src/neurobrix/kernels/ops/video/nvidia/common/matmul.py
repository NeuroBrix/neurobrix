"""
Universal MatMul Kernel - NeuroBrix Standard (Triton Optimized)
Source: FlagGems / Liger Logic
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel

# --- TRITON KERNEL (PURE 2D) ---
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# --- OP LAUNCH (UNIVERSAL FLATTENING) ---

def _matmul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D matmul using Triton kernel."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} @ {b.shape}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    with torch.cuda.device(a.device):
        _matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=4,
        )

    return c


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="matmul")
def matmul_kernel(a: torch.Tensor, b: torch.Tensor,
                  trans_a: bool = False, trans_b: bool = False) -> torch.Tensor:
    """Matrix multiply - pure Triton."""
    assert a.is_cuda and b.is_cuda

    if trans_a:
        a = a.transpose(-2, -1)
    if trans_b:
        b = b.transpose(-2, -1)

    a = a.contiguous()
    b = b.contiguous()

    # Handle batched matmul by flattening batch dims
    if a.ndim > 2 or b.ndim > 2:
        # Broadcast and flatten to 3D then iterate
        a_shape = a.shape
        b_shape = b.shape

        # Flatten batch dimensions
        if a.ndim == 2:
            a = a.unsqueeze(0)
        if b.ndim == 2:
            b = b.unsqueeze(0)

        # Flatten all batch dims
        a_batch = a.shape[:-2]
        b_batch = b.shape[:-2]

        a_flat = a.reshape(-1, a.shape[-2], a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-2], b.shape[-1])

        # Broadcast batch dims
        if a_flat.shape[0] == 1:
            a_flat = a_flat.expand(b_flat.shape[0], -1, -1).contiguous()
        if b_flat.shape[0] == 1:
            b_flat = b_flat.expand(a_flat.shape[0], -1, -1).contiguous()

        batch_size = a_flat.shape[0]
        M, K = a_flat.shape[-2], a_flat.shape[-1]
        N = b_flat.shape[-1]

        c_flat = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)

        # Run 2D matmul for each batch
        for i in range(batch_size):
            c_flat[i] = _matmul_2d(a_flat[i], b_flat[i])

        # Reshape output
        out_batch = list(torch.broadcast_shapes(a_batch, b_batch))
        return c_flat.view(out_batch + [M, N])

    # Simple 2D case
    return _matmul_2d(a, b)

