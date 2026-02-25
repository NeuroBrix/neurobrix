# GELU - Pure Triton (fp16 compatible)
# NeuroBrix - NVIDIA Common

import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU with internal fp32 conversion for fp16 inputs (no memory copy)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # Convert to fp32 for math (in registers, no memory copy)
    x_f32 = x.to(tl.float32)

    # GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
    cdf = 0.5 * (1.0 + tl.math.erf(0.707106781 * x_f32))
    output_f32 = x_f32 * cdf

    # Store in original dtype
    tl.store(output_ptr + offsets, output_f32.to(x.dtype), mask=mask)
