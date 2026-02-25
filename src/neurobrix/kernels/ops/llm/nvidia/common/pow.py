"""
Power kernel - Pure Triton
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _pow_kernel(
    x_ptr, exp_ptr, out_ptr,
    n_elements,
    exp_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Power: x^exp - pure Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if exp_is_scalar:
        exp = tl.load(exp_ptr)
    else:
        exp = tl.load(exp_ptr + offsets, mask=mask)

    # pow(x, exp) = exp(exp * log(x)) for positive x
    # Handle special cases
    out = tl.exp(exp * tl.log(tl.abs(x) + 1e-10))

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _pow_scalar_kernel(
    x_ptr, out_ptr,
    exponent,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Power with scalar exponent - pure Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Common fast paths
    if exponent == 2.0:
        out = x * x
    elif exponent == 0.5:
        out = tl.sqrt(x)
    elif exponent == 1.0:
        out = x
    elif exponent == 0.0:
        out = tl.full([BLOCK_SIZE], 1.0, dtype=x.dtype)
    elif exponent == -1.0:
        out = 1.0 / x
    else:
        # General case: x^exp = exp(exp * log(|x|))
        out = tl.exp(exponent * tl.log(tl.abs(x) + 1e-10))

    tl.store(out_ptr + offsets, out, mask=mask)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="pow")
def pow_kernel(a: torch.Tensor, b, alpha: float = 1.0) -> torch.Tensor:
    """Power operation - pure Triton."""
    assert a.is_cuda
    a = a.contiguous()
    output = torch.empty_like(a)
    n = a.numel()

    if n == 0:
        return output

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    # Check if b is scalar
    if isinstance(b, (int, float)):
        exponent = float(b)
        with torch.cuda.device(a.device):
            _pow_scalar_kernel[grid](
                a, output,
                exponent,
                n,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
            )
    else:
        # b is a tensor
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=a.device, dtype=a.dtype)
        b = b.contiguous()

        # Check if b is effectively scalar (single element)
        if b.numel() == 1:
            b_scalar = torch.empty(1, device=a.device, dtype=a.dtype)
            b_scalar.copy_(b.flatten())
            with torch.cuda.device(a.device):
                _pow_kernel[grid](
                    a, b_scalar, output,
                    n,
                    exp_is_scalar=True,
                    BLOCK_SIZE=BLOCK,
                    num_warps=4,
                )
        else:
            # Element-wise power
            b = b.expand_as(a).contiguous()
            with torch.cuda.device(a.device):
                _pow_kernel[grid](
                    a, b, output,
                    n,
                    exp_is_scalar=False,
                    BLOCK_SIZE=BLOCK,
                    num_warps=4,
                )

    return output

