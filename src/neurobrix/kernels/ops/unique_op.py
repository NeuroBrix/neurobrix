"""Unique — pure @triton.jit kernels.

NOTE: The full unique implementation (FlagGems) requires a multi-phase
pipeline (sort -> local-ne -> cumsum -> scatter -> fan-out) orchestrated
by Python wrapper code that allocates intermediate buffers and chains
multiple kernel launches. The core algorithmic kernels are extracted below.

These kernels assume the input is PRE-SORTED. The Python wrapper must:
1. Sort the input: sorted_data, sorted_indices = sort(input.ravel())
2. Launch simple_unique_flat_kernel (for small inputs) or the
   local_ne -> global_cumsum pipeline (for large inputs)
3. Slice the output to [:unique_size]

Extracted from FlagGems (Apache 2.0).
"""

import triton
import triton.language as tl


@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr,
    sorted_indices_ptr,
    data_out_ptr,
    inverse_indices_ptr,
    idx_ptr,
    unique_size_ptr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks,
    tile_size: tl.constexpr,
):
    """Single-block unique on pre-sorted data (for small inputs <= 8192).

    Compares each element with its predecessor to detect boundaries,
    uses cumsum for compacted output indices, then scatters unique values.

    Grid: (1,)
    """
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    # Load current and previous elements
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # Detect unique boundaries: ne(a, prev(a)), first element always unique
    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)

    # Write unique_size (last element's cumsum = num_unique - 1)
    unique_size_mask = i0 == tile_size - 1
    tl.store(unique_size_ptr + tl.zeros_like(i0), cumsum, mask=unique_size_mask)

    # Scatter unique values to compacted output
    tl.store(data_out_ptr + cumsum, a, mask=mask)

    # Scatter inverse indices: original position -> compacted index
    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    # Record boundary positions for count computation
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)


@triton.jit
def output_counts_kernel(
    idx_ptr,
    origin_num_tasks,
    counts_ptr,
    num_tasks,
    tile_size: tl.constexpr,
):
    """Compute counts from boundary indices.

    counts[i] = idx[i+1] - idx[i], with the last count going to the end.

    Grid: (ceil(num_unique / tile_size),)
    """
    pid = tl.program_id(0)
    r = tl.arange(0, tile_size)
    i0 = pid * tile_size + r
    mask = i0 < num_tasks

    idx = tl.load(idx_ptr + i0, mask=mask)

    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)
    tl.store(counts_ptr + i0, counts, mask=mask)
