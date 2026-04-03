"""Cumulative Maximum — pure @triton.jit kernels.

Computes running maximum along a dimension, returning both values and indices.
Uses scan-then-fan algorithm: each block computes a local cummax via
tl.associative_scan, then partial results are combined across blocks.

Extracted from FlagGems (Apache 2.0).

Three kernels for the scan-then-fan pipeline:
1. scan_part_max_kernel — local cummax within each block + partial max
2. add_base_max_kernel — fan-out: combine partial maxima with local results
3. cummax_loop_kernel — single-program loop variant for wide reductions
"""

import triton
import triton.language as tl


@triton.jit
def _cummax_combine(val_a, idx_a, val_b, idx_b):
    """Associative combiner: keep (value, index) of the maximum."""
    pick_b = val_b > val_a
    # Tie-break right: if equal, keep the later (right) index
    pick_b = pick_b | ((val_a == val_b) & (idx_b >= idx_a))
    val_out = tl.where(pick_b, val_b, val_a)
    idx_out = tl.where(pick_b, idx_b, idx_a)
    return val_out, idx_out


@triton.jit
def scan_part_max_kernel(
    inp,
    out,
    out_indices,
    partial_max,
    partial_max_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
):
    """Block-local cumulative maximum with optional partial output.

    Each block loads BLOCK_SIZE elements, computes cummax via associative
    scan, stores results, and optionally writes the block-maximum for
    the fan-out phase.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input, use -inf for masked positions
    inp_vals = tl.load(inp + offset, mask=mask, other=float('-inf'))
    inp_vals = inp_vals.to(tl.float32)
    indices = offset

    # Associative scan for cummax
    result, cummax_idx = tl.associative_scan(
        (inp_vals, indices), 0, _cummax_combine
    )

    tl.store(out + offset, result, mask=mask)
    tl.store(out_indices + offset, cummax_idx, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        # Reduce to find block maximum (tie-break right)
        block_max = tl.max(inp_vals, axis=0)
        # Find rightmost index with block_max value
        is_max = inp_vals == block_max
        max_idx = tl.max(tl.where(is_max, indices, -1), axis=0)
        # Only one thread writes the partial
        if tl.arange(0, BLOCK_SIZE)[0] == 0:
            tl.store(partial_max + pid, block_max)
            tl.store(partial_max_indices + pid, max_idx)


@triton.jit
def add_base_max_kernel(
    out,
    out_indices,
    partial_max,
    partial_max_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fan-out phase: combine partial maxima from prior blocks.

    For each block (except the first), loads the cumulative partial max
    from the previous block and merges with local results.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_vals = tl.load(out + offset, mask=mask)
    out_idx = tl.load(out_indices + offset, mask=mask)

    if pid > 0:
        prev_max = tl.load(partial_max + pid - 1)
        prev_idx = tl.load(partial_max_indices + pid - 1)

        # Merge: if previous partial max >= local, use previous
        use_prev = (prev_max > out_vals) | (
            (prev_max == out_vals) & (prev_idx >= out_idx)
        )
        final_vals = tl.where(use_prev, prev_max, out_vals)
        final_idx = tl.where(use_prev, prev_idx, out_idx)

        tl.store(out + offset, final_vals, mask=mask)
        tl.store(out_indices + offset, final_idx, mask=mask)
