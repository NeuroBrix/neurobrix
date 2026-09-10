"""Index select — pure @triton.jit kernel.

Extracted from FlagGems (FlagOpen/FlagGems) index_select.py.
2D grid: rows x index blocks, with bounds checking.
"""

import triton
import triton.language as tl

#: The contract this kernel enforces. Triton forbids reading a module global
#: from inside a @jit body, so the kernel repeats the literal; the permanent
#: test pins the two equal so they cannot drift.
INDEX_SELECT_OOB = "index_select: index out of range"


@triton.jit(debug=True)
def index_select_kernel(
    inp, out, M, N, index, index_len, fault_ptr,
    FAULT_CODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """`debug=True` keeps the `tl.device_assert` below in the compiled
    kernel regardless of TRITON_DEBUG: an out-of-range index TRAPS (the
    next device sync raises) exactly where torch's index_select raises.
    The FlagGems port skipped the store instead, leaving the output
    element as whatever the allocation held — silent, and invisible to
    a byte gate whose two sides share the defect
    (D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).

    That assert is honoured on CUDA and ROCm and NOT on Metal, which
    computes its predicate and discards it (measured 2026-09-10). This
    kernel never went out of bounds there — its load and its store are
    both masked by `index_valid_mask` — so what survived was exactly the
    FlagGems silence it was written to end: the output element stayed
    whatever the pool held. `FAULT_CODE` closes it. A non-zero code
    makes the kernel report through the fault word, which
    `check_device_faults` raises on at the next host observation; the
    wrapper passes 0 where the assert is honoured, and then not one
    instruction of it is emitted."""
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
    # Masked lanes loaded `other=0` and are in range by construction.
    tl.device_assert(index_valid_mask, "index_select: index out of range")
    if FAULT_CODE != 0:
        # Lanes past `index_len` loaded `other=0` and are in range by
        # construction; they must not report. Reduce to one scalar so the
        # store below is a single conditional word, not a vector.
        live = cols_offsets < index_len
        if tl.max(tl.where(live & (index_valid_mask == 0), 1, 0)) != 0:
            tl.store(fault_ptr, FAULT_CODE)

    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    final_mask = out_mask & index_valid_mask[None, :]
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=final_mask)
