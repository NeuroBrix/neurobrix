"""FFT — pure @triton.jit Cooley-Tukey radix-2 kernels.

Implements forward FFT, rfft, and irfft as pure Triton kernels.
Based on FlagGems PR #1243 (Cooley-Tukey butterfly decomposition).

Algorithm (every row of a [rows, N] batch at once — one launch per step, not per row):
  1. Bit-reversal permutation (reorder input)
  2. log2(N) butterfly stages with twiddle factors
  3. For rfft: take first N//2+1 complex outputs
  4. For irfft: reconstruct full spectrum from half, inverse FFT, scale by 1/N

Constraint: Input size must be power of 2 (standard Cooley-Tukey).
For non-power-of-2: caller must zero-pad to next power of 2.
"""

import math

import triton
import triton.language as tl


@triton.jit
def bit_reverse_rows_kernel(
    real_in, imag_in, real_out, imag_out,
    n, total, log2n,
    BLOCK_SIZE: tl.constexpr,
):
    """Bit-reversal permutation of EVERY row of a [rows, n] pair in one launch:
    row r's element i lands at row r, position bit_reverse(i).

    One element per lane, `total` = rows * n lanes. The rows used to be launched one
    at a time from the host: an STFT of a spoken sentence is thousands of rows, and
    its FFT cost thousands of launches per stage (py-spy on chatterbox's census walk,
    2026-09-28: every sample inside that per-row loop).
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit: rows * n passes 2^31 on a long speech
    m = offs < total
    row = offs // n
    idx = offs % n

    # `rev` is an explicit int64 from the start: a bare 0 is int32, and the loop
    # re-assigns an int64 into it — Triton refuses a loop-carried type change
    # (found 2026-09-18 by chatterbox's `aten::stft`).
    rev = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    t = idx
    for _ in range(0, log2n):
        rev = (rev << 1) | (t & 1)
        t = t >> 1

    val_real = tl.load(real_in + offs, mask=m, other=0.0)
    val_imag = tl.load(imag_in + offs, mask=m, other=0.0)
    tl.store(real_out + row * n + rev, val_real, mask=m)
    tl.store(imag_out + row * n + rev, val_imag, mask=m)


@triton.jit
def fft_stage_rows_kernel(
    real_ptr, imag_ptr,
    n, pairs, stage,
    INVERSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One butterfly stage of the Cooley-Tukey FFT on EVERY row of a [rows, n] pair,
    in place, one butterfly pair per lane (`pairs` = rows * n // 2 lanes).

    The forward twiddle is e^(-i*pi*k/half_block); INVERSE takes its conjugate. The
    arithmetic of each pair is the one-row kernel's, written the same way, so a row
    computed in a batch is bit-identical to the row computed alone.
    """
    PI = math.pi
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    half_n = n // 2
    row = offs // half_n
    tid = offs % half_n
    half_block = 1 << (stage - 1)

    butterfly_group = tid // half_block
    pos_in_group = tid % half_block
    first_local = butterfly_group * half_block * 2 + pos_in_group
    second_local = first_local + half_block
    # A lane beyond the last pair, or a pair whose second element lies beyond its row,
    # reads and writes nothing (no early exit — the portable form is the mask).
    m = (offs < pairs) & (second_local < n)
    first_idx = row * n + first_local
    second_idx = row * n + second_local

    a_real = tl.load(real_ptr + first_idx, mask=m, other=0.0)
    a_imag = tl.load(imag_ptr + first_idx, mask=m, other=0.0)
    b_real = tl.load(real_ptr + second_idx, mask=m, other=0.0)
    b_imag = tl.load(imag_ptr + second_idx, mask=m, other=0.0)

    angle = PI * pos_in_group / half_block
    if INVERSE:
        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)
    else:
        w_real = tl.cos(-angle)
        w_imag = tl.sin(-angle)

    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real

    tl.store(real_ptr + first_idx, a_real + tw_real, mask=m)
    tl.store(imag_ptr + first_idx, a_imag + tw_imag, mask=m)
    tl.store(real_ptr + second_idx, a_real - tw_real, mask=m)
    tl.store(imag_ptr + second_idx, a_imag - tw_imag, mask=m)


@triton.jit
def scale_kernel(
    real_ptr, imag_ptr,
    n_elements, scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Scale all elements by 1/N after inverse FFT."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    r = tl.load(real_ptr + offset, mask=mask)
    i = tl.load(imag_ptr + offset, mask=mask)
    tl.store(real_ptr + offset, r * scale, mask=mask)
    tl.store(imag_ptr + offset, i * scale, mask=mask)
