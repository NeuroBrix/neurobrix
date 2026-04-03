"""Cumulative Minimum — pure @triton.jit kernels.

Computes running minimum along a dimension, returning both values and indices.
Uses scan-then-fan algorithm (mirror of cummax with minimum).

Extracted from FlagGems (Apache 2.0).

Three kernels:
1. scan_part_min_kernel — local cummin within each block + partial min
2. add_base_min_kernel — fan-out: combine partial minima with local results
"""

import triton
import triton.language as tl


@triton.jit
def _cummin_combine(val_a, idx_a, val_b, idx_b):
    """Associative combiner: keep (value, index) of the minimum."""
    pick_b = val_b < val_a
    # Tie-break right: if equal, keep the later (right) index
    pick_b = pick_b | ((val_a == val_b) & (idx_b >= idx_a))
    val_out = tl.where(pick_b, val_b, val_a)
    idx_out = tl.where(pick_b, idx_b, idx_a)
    return val_out, idx_out


@triton.jit
def scan_part_min_kernel(
    inp,
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
):
    """Block-local cumulative minimum with optional partial output.

    Each block loads BLOCK_SIZE elements, computes cummin via associative
    scan, stores results, and optionally writes the block-minimum for
    the fan-out phase.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input, use +inf for masked positions
    inp_vals = tl.load(inp + offset, mask=mask, other=float('inf'))
    inp_vals = inp_vals.to(tl.float32)
    indices = offset

    # Associative scan for cummin
    result, cummin_idx = tl.associative_scan(
        (inp_vals, indices), 0, _cummin_combine
    )

    tl.store(out + offset, result, mask=mask)
    tl.store(out_indices + offset, cummin_idx, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        # Reduce to find block minimum (tie-break right)
        block_min = tl.min(inp_vals, axis=0)
        is_min = inp_vals == block_min
        min_idx = tl.max(tl.where(is_min, indices, -1), axis=0)
        if tl.arange(0, BLOCK_SIZE)[0] == 0:
            tl.store(partial_min + pid, block_min)
            tl.store(partial_min_indices + pid, min_idx)


@triton.jit
def add_base_min_kernel(
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fan-out phase: combine partial minima from prior blocks.

    For each block (except the first), loads the cumulative partial min
    from the previous block and merges with local results.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_vals = tl.load(out + offset, mask=mask)
    out_idx = tl.load(out_indices + offset, mask=mask)

    if pid > 0:
        prev_min = tl.load(partial_min + pid - 1)
        prev_idx = tl.load(partial_min_indices + pid - 1)

        # Merge: if previous partial min <= local, use previous
        use_prev = (prev_min < out_vals) | (
            (prev_min == out_vals) & (prev_idx >= out_idx)
        )
        final_vals = tl.where(use_prev, prev_min, out_vals)
        final_idx = tl.where(use_prev, prev_idx, out_idx)

        tl.store(out + offset, final_vals, mask=mask)
        tl.store(out_indices + offset, final_idx, mask=mask)
