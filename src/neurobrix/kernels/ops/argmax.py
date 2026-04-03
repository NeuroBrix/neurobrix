"""Argmax — pure @triton.jit kernels. Extracted from FlagGems.

Three-kernel approach:
  argmax_kernel_1: per-block max reduction (flat tensor, dim=None)
  argmax_kernel_2: final reduction across blocks (flat tensor, dim=None)
  argmax_kernel_inner: along last dim (dim=-1)
"""

import triton
import triton.language as tl


@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float('-inf'))
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float('-inf'))
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@triton.jit
def argmax_kernel_inner(
    inp,
    out_index,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Argmax along the innermost (last) dimension."""
    pid_m = tl.program_id(0)

    min_value = float('-inf')

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offset
        mask = n_offset < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, local_argmax)
    else:
        max_values = min_value
        argmax_values = 0

        loop_time = N // TILE_N
        remainder = N % TILE_N
        for start_n in range(0, loop_time):
            n_offset = start_n * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, start_n * TILE_N + local_argmax, argmax_values
            )

        if remainder:
            n_offset = loop_time * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            mask = n_offset < N
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, loop_time * TILE_N + local_argmax, argmax_values
            )

        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, argmax_values)
