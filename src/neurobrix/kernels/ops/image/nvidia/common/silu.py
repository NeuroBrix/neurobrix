# SiLU - Pure Triton (fp16 compatible)
# NeuroBrix - NVIDIA Common

import triton
import triton.language as tl


@triton.jit
def _silu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """SiLU with internal fp32 conversion for fp16 inputs (no memory copy)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load (any dtype)
    x = tl.load(input_ptr + offsets, mask=mask)

    # Convert to fp32 for math (in registers, no memory copy)
    x_f32 = x.to(tl.float32)

    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_f32))
    output_f32 = x_f32 * sigmoid_x

    # Store in original dtype
    tl.store(output_ptr + offsets, output_f32.to(x.dtype), mask=mask)
