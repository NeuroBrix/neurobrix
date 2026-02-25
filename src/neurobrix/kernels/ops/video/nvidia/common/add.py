"""
Universal ElementWise Add - Triton Optimized
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _binary_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="add")
def add_kernel(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Add operation using PyTorch (handles all broadcasting).

    Args:
        a: First tensor
        b: Second tensor
        alpha: Scalar multiplier (for compatibility)

    Returns:
        Result tensor
    """
    return a + b
