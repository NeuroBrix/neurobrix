"""Lerp (linear interpolation) kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.

Tensor variant: result = where(|w| < 0.5, a + w*(b-a), b - (b-a)*(1-w))
Scalar variant (head): result = a + w*(b-a)
Scalar variant (tail): result = b - (b-a)*(1-w)
"""

import triton
import triton.language as tl

@triton.jit
def lerp_tensor_forward_kernel(
    input_ptr,
    end_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Lerp with tensor weight: numerically stable two-branch formula."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    a = tl.load(input_ptr + offset, mask=mask)
    b = tl.load(end_ptr + offset, mask=mask)
    w = tl.load(weight_ptr + offset, mask=mask)

    result = tl.where(
        tl.abs(w) < 0.5,
        a + w * (b - a),
        b - (b - a) * (1 - w),
    )
    tl.store(output_ptr + offset, result, mask=mask)

@triton.jit
def lerp_scalar_head_kernel(
    input_ptr,
    end_ptr,
    output_ptr,
    n_elements,
    weight,
    BLOCK_SIZE: tl.constexpr,
):
    """Lerp with scalar weight < 0.5: a + w*(b-a)."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    a = tl.load(input_ptr + offset, mask=mask)
    b = tl.load(end_ptr + offset, mask=mask)

    result = a + weight * (b - a)
    tl.store(output_ptr + offset, result, mask=mask)

@triton.jit
def lerp_scalar_tail_kernel(
    input_ptr,
    end_ptr,
    output_ptr,
    n_elements,
    weight,
    BLOCK_SIZE: tl.constexpr,
):
    """Lerp with scalar weight >= 0.5: b - (b-a)*(1-w)."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    a = tl.load(input_ptr + offset, mask=mask)
    b = tl.load(end_ptr + offset, mask=mask)

    result = b - (b - a) * (1 - weight)
    tl.store(output_ptr + offset, result, mask=mask)
