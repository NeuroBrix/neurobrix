"""
RMSNorm Kernel - NVIDIA
Root Mean Square Normalization

Source: Liger-Kernel (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def _rms_norm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: y = x * w * rsqrt(mean(x^2) + eps)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointers
    row_x_ptr = X_ptr + row_idx * stride_x_row
    row_y_ptr = Y_ptr + row_idx * stride_y_row

    # Load x
    row_x = tl.load(row_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load weight
    row_w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean(x^2)
    row_x_sq = row_x * row_x
    mean_sq = tl.sum(row_x_sq, axis=0) / n_cols
    
    # Compute rsqrt
    rstd = tl.math.rsqrt(mean_sq + eps)
    
    # Normalize
    row_y = row_x * rstd * row_w
    
    # Store
    tl.store(row_y_ptr + col_offsets, row_y, mask=mask)


# =============================================================================
# WRAPPER
# =============================================================================

def _rmsnorm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Check shapes
    assert x.shape[-1] == weight.shape[0], "Weight shape must match input last dim"
    
    # Flatten to [M, N]
    N = x.shape[-1]
    M = x.numel() // N
    x_2d = x.view(-1, N)
    
    # Allocate output
    y = torch.empty_like(x_2d)
    
    # Block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    grid = (M,)
    
    _rms_norm_kernel[grid](
        x_2d,
        y,
        weight,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y.view_as(x)


# =============================================================================
# REGISTRATION
# =============================================================================

@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="rmsnorm")
def rmsnorm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm kernel compatible with all NVIDIA architectures."""
    return _rmsnorm_impl(x, weight, eps)
