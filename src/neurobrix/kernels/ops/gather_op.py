"""Gather along dim — pure @triton.jit kernel.

Approach from FlagGems: work on flattened index space. For each element in the
index tensor, compute its multi-dim position, load the gather index, then load
from source replacing the gather dim's index with the loaded value.

This static kernel handles any rank by taking pre-computed strides as args.
The wrapper decomposes the gather dim into (outer, gather, inner) strides.
"""

import triton
import triton.language as tl

@triton.jit
def gather_kernel(
    inp, index, out,
    N,
    inp_dim_stride,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    inp_stride_outer, inp_stride_inner,
    out_stride_outer, out_stride_dim, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather kernel — works for any tensor rank via pre-computed strides.

    The wrapper reshapes the problem into (outer, dim, inner) where:
    - outer = product of dims before gather dim
    - dim = gather dim size (from index tensor)
    - inner = product of dims after gather dim
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Decompose flat offset into (outer, dim, inner)
    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    # Load the gather index value
    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    # Load from input: same outer/inner position, but use cur_index for the gather dim
    inp_offset = outer_idx * inp_stride_outer + cur_index * inp_dim_stride + inner_idx * inp_stride_inner
    val = tl.load(inp + inp_offset, mask=mask, other=0.0)

    # Store to output
    out_offset = outer_idx * out_stride_outer + dim_idx * out_stride_dim + inner_idx * out_stride_inner
    tl.store(out + out_offset, val, mask=mask)
