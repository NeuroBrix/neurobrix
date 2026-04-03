"""Arange — pure @triton.jit kernel.

Ported from FlagGems arange_func (Apache-2.0 license).
Generates sequential values: start, start+step, start+2*step, ...
"""

import triton
import triton.language as tl


@triton.jit
def arange_kernel(
    output_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate sequential values: output[i] = start + i * step."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements

    value = start + idx * step
    tl.store(output_ptr + idx, value, mask=mask)
