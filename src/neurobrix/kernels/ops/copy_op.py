"""Copy — pure @triton.jit kernel.

Ported from FlagGems copy (Apache-2.0 license).
Copies elements from src to dst, with optional dtype conversion
handled by Triton's automatic casting on store.
"""

import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy src to dst. Dtype conversion happens automatically via Triton's
    store when src and dst pointer types differ."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    data = tl.load(src_ptr + offset, mask=mask)
    tl.store(dst_ptr + offset, data, mask=mask)
