"""Cumulative sum — pure @triton.jit kernels. Extracted from FlagGems.

Two-pass scan-then-fan approach for large 1D cumsum:
  scan_part_sum_kernel: per-block prefix sum + partial sums
  add_base_sum_kernel: propagate partial sums across blocks

Multi-dimensional (A,B,C) variants:
  scan_part_sum_abc_kernel: per-block scan along B dim
  add_base_sum_abc_kernel: propagate partial sums along B dim

Row-wise persistent kernels:
  reduce_then_scan_block_sum_kernel_row: block sums for row scan
  reduce_then_scan_root_scan_kernel_row: persistent root scan
  reduce_then_scan_block_scan_kernel_row: final block scan with prefix
"""

import triton
import triton.language as tl


@tl.constexpr
def get_scan_accum_type(inp_dtype: tl.dtype) -> tl.dtype:
    if inp_dtype.is_bf16() or inp_dtype.is_fp16():
        return tl.float32
    if inp_dtype.is_int():
        return tl.int64
    else:
        return inp_dtype


# --- 1D scan-then-fan kernels ---


@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_sum_kernel(
    inp,
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumsum(inp_vals, axis=0)

    part_sum_via_sum = tl.sum(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_sum_ptrs = partial_sum + pid
    tl.store(partial_sum_ptrs, part_sum_via_sum)


@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_sum_kernel(
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid > 0:
        partial_sum_ptrs = partial_sum + pid - 1
        last_part_sum_via_sum = tl.load(partial_sum_ptrs)

        final_vals = out_vals + last_part_sum_via_sum
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


# --- Multi-dim (A,B,C) kernels ---


@triton.jit(do_not_specialize=["part_num"])
def scan_part_sum_abc_kernel(
    inp,
    out,
    partial_sum,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumsum(inp_vals, axis=0)

    part_sum_via_sum = tl.sum(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_sum_ptrs = partial_sum + part_offset
    tl.store(partial_sum_ptrs, part_sum_via_sum)


@triton.jit(do_not_specialize=["part_num"])
def add_base_sum_abc_kernel(
    out,
    partial_sum,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid_b > 0:
        partial_sum_ptrs = partial_sum + last_part_offset
        last_part_sum_via_sum = tl.load(partial_sum_ptrs)

        final_vals = out_vals + last_part_sum_via_sum
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


# --- Row-wise persistent scan kernels ---


@triton.jit
def reduce_then_scan_block_sum_kernel_row(
    in_ptr,
    block_sum_ptr,
    N,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    """Block sum: each CTA reduces its assigned tiles into a single sum."""
    pid_n = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(0).to(tl.int64)
    num_programs_n = tl.num_programs(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    acc = tl.zeros((TILE_SIZE,), dtype=acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N).to(acc_dtype)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(
        block_sum_ptr + pid_m * num_programs_n + pid_n, block_sum, cache_modifier=".cg"
    )


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    """Persistent scan of block sums (or small row)."""
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0).to(acc_dtype)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


@triton.jit
def reduce_then_scan_block_scan_kernel_row(
    in_ptr,
    previous_sum_ptr,
    out_ptr,
    N,
    num_tiles_n,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    """Final block scan: apply prefix from previous blocks, then local cumsum."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)

    prefix = tl.load(
        previous_sum_ptr + pid_m * num_tiles_n + pid_n - 1, mask=pid_n > 0, other=0
    ).to(acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask).to(acc_dtype)
        tile_scan = prefix + tl.cumsum(x, 0)
        prefix += tl.sum(x, 0)
        tl.store(
            out_ptr + pid_m * N + offsets, tile_scan, mask=mask, cache_modifier=".cg"
        )
