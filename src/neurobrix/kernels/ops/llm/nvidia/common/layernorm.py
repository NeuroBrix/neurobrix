"""
Universal LayerNorm - Triton Optimized
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel

@triton.jit
def _layernorm_kernel(
    X, Y, W, B,
    stride,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Pointers
    x_ptr = X + row * stride + cols
    
    # Load Data
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Mean and Var
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize
    y = x_centered * rstd
    
    # Apply Scale/Bias if present (Load pointers handled by caller if None)
    if W is not None:
        w = tl.load(W + cols, mask=mask, other=1.0)
        y = y * w
    if B is not None:
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = y + b
        
    tl.store(Y + row * stride + cols, y, mask=mask)

def _layernorm_impl(
    x: torch.Tensor,
    normalized_shape: list = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm implementation using pure Triton."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    if normalized_shape is None:
        normalized_shape = [x.shape[-1]]
    
    N = normalized_shape[-1] if normalized_shape else x.shape[-1]
    M = x.numel() // N
    
    # Reshape to 2D for kernel
    x_2d = x.view(M, N)
    y = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    
    _layernorm_kernel[(M,)](
        x_2d, y,
        weight, bias,
        x_2d.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view_as(x)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="layernorm")
def layernorm_kernel(
    x: torch.Tensor,
    normalized_shape: list = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm using pure Triton.
    
    Args:
        x: Input tensor
        normalized_shape: Shape to normalize over
        weight: Scale parameter
        bias: Bias parameter
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor
    """
    return _layernorm_impl(x, normalized_shape, weight, bias, eps)

