"""Sort — pure @triton.jit radix sort helper kernels.

Ported from FlagGems sort (Apache-2.0 license).
Extracts the uint-conversion helpers and the two main radix sort kernels
(global histogram + sweep). The small-N bitonic sort kernel is also included.

The radix sort orchestration (multi-pass loop, buffer allocation) must be
done in the wrapper since it requires memory allocation.
"""

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Uint conversion helpers — preserve sort order across signed/float types
# --------------------------------------------------------------------------- #

# COMPILE-TIME helpers — `@tl.constexpr`, NEVER `@triton.jit`. Upstream
# FlagGems has carried them as constexpr functions since the one-sweep
# sort's birth commit (e669aee, PR #694); our vendoring delta hardened
# them into `@triton.jit`, which Triton >= 3.4 structurally rejects (the
# `_builder`→`_semantic` frontend refactor: a JIT callee compiles as a
# real typed function, so it can no longer return a dtype object, and
# `tl.core.get_int_dtype` — a plain undecorated python function — no
# longer resolves from inside JIT). `@tl.constexpr` runs them as plain
# compile-time python (legal 3.1→3.6), exactly upstream's shape.
def _unwrap_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_int_t(num_bits: tl.constexpr, signed: tl.constexpr):
    num_bits = _unwrap_constexpr(num_bits)
    signed = _unwrap_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def _one_zeros(num_bits: tl.constexpr):
    """1 followed by (num_bits-1) zeros, e.g. 0x80 for 8-bit."""
    num_bits = _unwrap_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def _zero_ones(num_bits: tl.constexpr):
    """0 followed by (num_bits-1) ones, e.g. 0x7F for 8-bit."""
    num_bits = _unwrap_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    """Unsigned → unsigned: negate bits if descending."""
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    """Signed int → unsigned, preserving sort order."""
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = _get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        bit_mask: tl.constexpr = _zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        sign_bit_mask: tl.constexpr = _one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    """Float → unsigned, preserving sort order (handles sign bit correctly)."""
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = _get_int_t(num_bits, True)
    udtype = _get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = _one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preserve_order(x, descending: tl.constexpr = False):
    """Convert any numeric type to unsigned, preserving sort order."""
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


# --------------------------------------------------------------------------- #
# Global histogram kernel — counts per-bin occurrences for radix sort
# --------------------------------------------------------------------------- #

@triton.jit
def radix_sort_histogram_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    """Compute per-bin histograms for radix sort.

    arr_ptr: (m, n) — input data
    out_ptr: (m, num_passes, r) — histogram output, r = 2^num_bits_per_pass
    """
    pid = tl.program_id(0).to(tl.int64)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2 ** num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = (CTA_TILE_N * pid_n).to(tl.int32)   # loop bounds are 32-bit counts: the Metal lowering refuses a 64-bit scf.for bound (addressing stays 64-bit through the program ids)
    cta_n_end = (tl.minimum(cta_n_start + CTA_TILE_N, n)).to(tl.int32)

    for p in range(0, num_passes):
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):
                n_offsets = n_start + tl.arange(0, TILE_N)
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preserve_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask
                matches = tl.where(mask, (bin_indices[:, None] == key), False)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


# --------------------------------------------------------------------------- #
# Sweep kernel — scatter elements to sorted positions using decoupled lookback
# --------------------------------------------------------------------------- #

@triton.jit
def radix_sort_tile_counts_kernel(
    arr_ptr,
    counts_ptr,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    bit_offset,
    descending: tl.constexpr,
):
    """How many elements of each bin this TILE holds. Stage 1 of three.

    This and `radix_sort_scatter_kernel` replace a single kernel that carried a
    DECOUPLED LOOKBACK: each tile published its count and then spun on its
    predecessors' flags to learn how many of its bin came before it.

    That construction needs forward progress across CTAs and cross-CTA
    visibility for a `.cg` store, and Triton promises neither — inside a kernel
    it promises nothing about other programs at all. Measured 2026-09-17 on
    triton-ext: correct to exactly 2048 elements (one tile) and wrong from 2049,
    with elements LOST rather than misordered (9 absent at n=2049, 1726 at
    n=4096) — two tiles computing the same destinations and overwriting each
    other, which is what a lookback reading stale flags produces.

    The replacement is not a new design. `cumsum_wrapper` in this same tree does
    its cross-tile scan as SEPARATE LAUNCHES (`scan_part_sum_kernel`, then
    `add_base_sum_kernel` when there is more than one part) and is correct at
    eight tiles, measured the same day. Triton guarantees ordering BETWEEN
    launches; that is the guarantee this needs.
    """
    pid = tl.program_id(0).to(tl.int64)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1).to(tl.int64)

    bfe_mask: tl.constexpr = (1 << k_bits) - 1
    r: tl.constexpr = 2 ** k_bits
    cta_r_start = (pid_r * TILE_R).to(tl.int32)
    cta_r_end = (tl.minimum(cta_r_start + TILE_R, r)).to(tl.int32)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preserve_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask

    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        tl.store(counts_ptr + pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n,
                 local_sum)


@triton.jit
def radix_sort_tile_prefix_kernel(
    counts_ptr,
    prefix_ptr,
    num_tiles,
    OUT_N,
    TILES_POW2: tl.constexpr,
):
    """Exclusive scan of the per-tile counts, along the TILE axis. Stage 2.

    One program per (batch, bin) row — the rows are independent, and the row is
    `num_tiles` long, which is `cdiv(n, 2048)`: small. This is the step the
    lookback was doing inside the scatter, done where Triton actually orders it.
    """
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, TILES_POW2)
    mask = offs < num_tiles
    counts = tl.load(counts_ptr + row * OUT_N + offs, mask=mask, other=0)
    prefix = tl.cumsum(counts, axis=0) - counts          # exclusive
    tl.store(prefix_ptr + row * OUT_N + offs, prefix, mask=mask)


@triton.jit
def radix_sort_scatter_kernel(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,
    tile_prefix_ptr,
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    """Scatter elements to their sorted positions. Stage 3.

    Identical to the old `radix_sort_sweep_kernel` except that
    `exclusive_prefix` — how many of this bin lie in EARLIER tiles — is READ
    from the scan of stage 2 instead of being discovered by spinning on other
    programs' flags.
    """
    pid = tl.program_id(0).to(tl.int64)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1).to(tl.int64)

    bfe_mask: tl.constexpr = (1 << k_bits) - 1
    r: tl.constexpr = 2 ** k_bits
    cta_r_start = (pid_r * TILE_R).to(tl.int32)
    cta_r_end = (tl.minimum(cta_r_start + TILE_R, r)).to(tl.int32)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preserve_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask

    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)
        exclusive_prefix = tl.load(
            tile_prefix_ptr + pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n)

        local_ex_cumsum = tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        ex_cumsum_in_bin = exclusive_prefix + local_ex_cumsum

        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
        )
        pos = ex_cumsum_bins + ex_cumsum_in_bin

        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            associate_arr = tl.load(
                associate_arr_ptr + pid_m * N + n_offsets, mask=mask
            )
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)