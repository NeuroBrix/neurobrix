# Tanh - Pure Triton (fp16 compatible, numerically stable)
# NeuroBrix - NVIDIA Common

import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Tanh with internal fp32 conversion (no memory copy)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Convert to fp32 for math
    x_f32 = x.to(tl.float32)

    # Clamp to avoid extreme values
    x_clamped = tl.minimum(tl.maximum(x_f32, -20.0), 20.0)

    # Numerically stable formula
    exp_neg_2x = tl.exp(-2.0 * tl.abs(x_clamped))
    tanh_abs = (1.0 - exp_neg_2x) / (1.0 + exp_neg_2x)

    # Apply sign
    output_f32 = tl.where(x_f32 >= 0.0, tanh_abs, -tanh_abs)

    # Store in original dtype
    tl.store(out_ptr + offsets, output_f32.to(x.dtype), mask=mask)
