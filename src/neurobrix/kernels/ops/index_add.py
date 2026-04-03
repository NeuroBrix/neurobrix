"""Index add along dim — pure @triton.jit kernel.

Logic from FlagGems index_add.py: for each element in src, load the index
value for that position's dim, then atomic_add into output at the indexed
position. Uses atomic_add for thread safety.
"""

import triton
import triton.language as tl

@triton.jit
def index_add_kernel(
    index, src, out,
    N, inp_numel,
    inp_dim_stride, inp_shape_dim, src_shape_dim, delta, alpha,
    src_stride_outer, src_stride_dim, src_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Index add: out[..., index[i], ...] += alpha * src[..., i, ...]."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Decompose flat offset into (outer, dim, inner)
    inner_idx = offsets % inner_size
    temp = offsets // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    # Compute source linear offset
    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner

    # Load the index value for this dim position
    src_dim_idx = tl.load(index + dim_idx, mask=mask, other=0).to(tl.int64)

    # Compute output offset: replace dim_idx with src_dim_idx
    # output_offset = src_offset + (delta * outer_idx_in_dim + src_dim_idx - dim_idx) * inp_dim_stride
    # Simplified: just compute directly from the 3D decomposition
    out_offset = (outer_idx * (inp_shape_dim * inner_size) + src_dim_idx * inner_size + inner_idx).to(tl.int64)

    # Scale factor
    input_mask = (out_offset >= 0) & (out_offset < inp_numel) & mask
    add_on = tl.load(src + src_offset, mask=mask, other=0.0) * alpha
    tl.atomic_add(out + out_offset, add_on, mask=input_mask)
