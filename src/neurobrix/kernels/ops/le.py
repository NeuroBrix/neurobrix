"""Less than or equal — pure @triton.jit kernel.

Logic from FlagGems le.py. Both tensor-tensor and scalar variants.
"""

import triton
import triton.language as tl

@triton.jit
def le_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, (x.to(tl.float32) <= y.to(tl.float32)).to(tl.int1), mask=mask)

@triton.jit
def le_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, (x.to(tl.float32) <= scalar).to(tl.int1), mask=mask)
