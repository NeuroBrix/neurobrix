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

@triton.jit
def _unwrap_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@triton.jit
def _get_int_t(num_bits: tl.constexpr, signed: tl.constexpr):
    num_bits = _unwrap_constexpr(num_bits)
    signed = _unwrap_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@triton.jit
def _one_zeros(num_bits: tl.constexpr):
    """1 followed by (num_bits-1) zeros, e.g. 0x80 for 8-bit."""
    num_bits = _unwrap_constexpr(num_bits)
    return 1 << (num_bits - 1)


@triton.jit
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
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2 ** num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

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
def radix_sort_sweep_kernel(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,
    status_ptr,
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
    """Scatter elements to their sorted positions using exclusive prefix sums.

    Uses decoupled lookback for inter-CTA prefix sum communication.
    """
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1

    r: tl.constexpr = 2 ** k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preserve_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask

    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # Decoupled lookback
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

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
