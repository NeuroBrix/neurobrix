# Sigmoid - Pure Triton (fp16 compatible)
# NeuroBrix - NVIDIA Common

import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Sigmoid with internal fp32 conversion (no memory copy)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Convert to fp32 for math
    x_f32 = x.to(tl.float32)

    # sigmoid = 1 / (1 + exp(-x))
    output_f32 = 1.0 / (1.0 + tl.exp(-x_f32))

    # Store in original dtype
    tl.store(out_ptr + offsets, output_f32.to(x.dtype), mask=mask)
