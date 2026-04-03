"""Count Nonzero — pure @triton.jit kernel.

Counts non-zero elements in a flattened tensor using atomic addition.
Each block counts its local non-zeros, then atomically adds to the output.
Extracted from FlagGems (Apache 2.0).

Grid: (ceil(numel / BLOCK_SIZE),)
"""

import triton
import triton.language as tl

@triton.jit
def count_nonzero_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Count non-zero elements via block-local sum + atomic global add.

    Each program loads a block of elements, compares != 0, sums the
    boolean result, then atomically adds the count to the output scalar.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)

    tl.atomic_add(out_ptr, nonzero_count)

@triton.jit
def count_nonzero_dim_kernel(
    x_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Count non-zero elements along a dimension (after dim_compress).

    Each program handles one row of the compressed tensor and iterates
    over columns in blocks, accumulating the non-zero count.

    x_ptr: flattened compressed input (M * N elements, reduction dim = N)
    out_ptr: output of shape (M,)
    """
    pid_x = tl.program_id(0)

    nonzero_count = tl.zeros((), dtype=tl.int64)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        col_mask = (offset < numel) & (cols_offsets < N)

        x = tl.load(x_ptr + offset, mask=col_mask, other=0)
        is_nonzero = (x != 0).to(tl.int64)
        nonzero_count += tl.sum(is_nonzero)

    tl.store(out_ptr + pid_x, nonzero_count)
