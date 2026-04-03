"""Nonzero — pure @triton.jit kernel.

Extracts indices of non-zero elements from a flattened boolean tensor.
Requires a pre-computed prefix sum (cumsum of bool input) to determine
the output position of each non-zero element.

The Python wrapper must:
1. Flatten and convert to bool: inp_bool = (inp != 0)
2. Compute prefix sum: prefix_sum = inp_bool.cumsum(0)
3. Allocate output: out = empty(numel, ndim, dtype=int64)
4. Launch this kernel
5. Slice to actual count: out = out[:prefix_sum[-1]]

Extracted from FlagGems (Apache 2.0).

Grid: (ceil(n_elements / BLOCK_SIZE),)
"""

import triton
import triton.language as tl


@triton.jit
def nonzero_kernel(
    inp,
    prefix_sum,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write indices of non-zero elements into output tensor.

    For each non-zero element at flat index i, decomposes i into
    multi-dimensional indices using the original shape, then writes
    each dimension's index into out[prefix_sum[i]-1, dim].

    Args:
        inp: flattened boolean input (1 = nonzero)
        prefix_sum: cumulative sum of inp (1-based output position)
        out: output tensor of shape (n_elements, ndim), int64
        shape: original tensor shape stored as int32 on device
        ndim: number of dimensions (compile-time constant)
    """
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

    nonzero_mask = mask & inp_vals

    # Decompose flat index into multi-dimensional indices
    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat = idx_flat // dim_size
        tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)
