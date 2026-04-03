"""Scatter along dim — pure @triton.jit kernel.

Approach from FlagGems: work on flattened index space. For each element,
decompose into (outer, dim, inner), load the scatter index, then store
(or atomic_add) into the destination at the indexed position.

scatter: dst[outer][index[outer][dim][inner]][inner] = src[outer][dim][inner]
scatter_add: dst[...] += src[...]
"""

import triton
import triton.language as tl

@triton.jit
def scatter_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter: out[outer][index_val][inner] = src[outer][dim][inner]."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    # Load scatter index
    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    # Load source value
    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    # Store to output at scattered position
    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.store(out + out_offset, val, mask=mask)

@triton.jit
def scatter_add_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter add: out[outer][index_val][inner] += src[outer][dim][inner]."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_add(out + out_offset, val, mask=mask)

# ═══════════════════════════════════════════════════════════════════════════════
# scatter_reduce kernels — amax, amin using Triton native atomics
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def scatter_reduce_amax_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """scatter_reduce with reduce='amax': atomic max."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_max(out + out_offset, val, mask=mask)

@triton.jit
def scatter_reduce_amin_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """scatter_reduce with reduce='amin': atomic min."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_min(out + out_offset, val, mask=mask)
