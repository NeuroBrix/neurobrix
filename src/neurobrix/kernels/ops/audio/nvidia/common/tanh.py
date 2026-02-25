import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _tanh_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Numerically stable tanh implementation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Numerically stable tanh:
    # For x >= 0: tanh(x) = 1 - 2/(exp(2x) + 1)
    # For x < 0: tanh(x) = 2/(exp(-2x) + 1) - 1
    # This avoids overflow in exp(2x) for large positive x

    # Clamp to avoid extreme values
    x_clamped = tl.minimum(tl.maximum(x, -20.0), 20.0)

    # Use the standard formula with clamped input
    exp_neg_2x = tl.exp(-2.0 * tl.abs(x_clamped))
    tanh_abs = (1.0 - exp_neg_2x) / (1.0 + exp_neg_2x)

    # Apply sign
    out = tl.where(x >= 0, tanh_abs, -tanh_abs)

    tl.store(out_ptr + offsets, out, mask=mask)


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="tanh")
def tanh_kernel(x: torch.Tensor) -> torch.Tensor:
    """Tanh activation - pure Triton."""
    assert x.is_cuda
    x = x.contiguous()
    output = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return output

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    with torch.cuda.device(x.device):
        _tanh_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK, num_warps=4)

    return output

