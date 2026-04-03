"""Copy / dtype cast — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def copy_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy tensor (with implicit dtype conversion via pointer types)."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x, mask=mask)
