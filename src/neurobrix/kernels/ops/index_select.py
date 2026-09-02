"""Index select — pure @triton.jit kernel.

Extracted from FlagGems (FlagOpen/FlagGems) index_select.py.
2D grid: rows x index blocks, with bounds checking.
"""

import triton
import triton.language as tl


@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask & (cols_offsets[None, :] < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    # torch semantics: a NEGATIVE index counts from the end (index + N).
    # The FlagGems port treated it as invalid and SKIPPED the store, leaving
    # the output element as whatever the allocation held — zeros on a fresh
    # cudaMalloc, stale data under the allocator pool (HAT's
    # relative_position_index_OCA carries negative entries; pool gate
    # 2026-09-02: the two HAT upscaler PNGs differed pool on vs off).
    indices = tl.where(indices < 0, indices + N, indices)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound

    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    final_mask = out_mask & index_valid_mask[None, :]
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=final_mask)
