"""
Universal GELU (Fast Approx) - Triton Optimized (fp16 compatible)
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _stable_tanh(x):
    """Numerically stable tanh that works with fp16."""
    # Clamp to avoid extreme values
    x_clamped = tl.minimum(tl.maximum(x, -20.0), 20.0)
    # Use the standard formula with clamped input
    exp_neg_2x = tl.exp(-2.0 * tl.abs(x_clamped))
    tanh_abs = (1.0 - exp_neg_2x) / (1.0 + exp_neg_2x)
    # Apply sign
    return tl.where(x >= 0, tanh_abs, -tanh_abs)


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    output = 0.5 * x * (1 + _stable_tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, output, mask=mask)


def _unary_impl(x: torch.Tensor) -> torch.Tensor:
    """Unary op implementation using pure Triton."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()

    output = torch.empty_like(x)
    n_elements = output.numel()

    if n_elements == 0:
        return output

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _gelu_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="gelu")
def gelu_kernel(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """
    Gelu activation using pure Triton.
    Uses tanh approximation by default.

    Args:
        x: Input tensor
        approximate: Approximation mode (ignored, always uses tanh)

    Returns:
        Result tensor
    """
    return _unary_impl(x)
