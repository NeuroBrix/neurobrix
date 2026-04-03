"""FFT — pure @triton.jit Cooley-Tukey radix-2 kernels.

Implements forward FFT, rfft, and irfft as pure Triton kernels.
Based on FlagGems PR #1243 (Cooley-Tukey butterfly decomposition).

Algorithm:
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
def bit_reverse_kernel(
    real_in, imag_in, real_out, imag_out,
    n,
):
    """Bit-reversal permutation: input[i] → output[bit_reverse(i)].

    Each thread handles one element. Grid size = N.
    """
    tid = tl.program_id(0)

    if tid >= n:
        return

    # Compute bit-reversed index
    temp_n = n
    idx = tid
    rev_idx = 0
    temp_idx = idx
    while temp_n > 1:
        temp_n //= 2
        rev_idx = (rev_idx << 1) | (temp_idx & 1)
        temp_idx = temp_idx >> 1

    val_real = tl.load(real_in + idx)
    val_imag = tl.load(imag_in + idx)
    tl.store(real_out + rev_idx, val_real)
    tl.store(imag_out + rev_idx, val_imag)


@triton.jit
def fft_stage_kernel(
    real_ptr, imag_ptr,
    n, stage,
):
    """One butterfly stage of the Cooley-Tukey FFT.

    Each thread processes one butterfly pair. Grid size = N//2.
    Iterates log2(N) times for full FFT.
    """
    PI = math.pi
    tid = tl.program_id(0)

    if tid >= n // 2:
        return

    half_block = 1 << (stage - 1)

    # Which butterfly group and position within group
    butterfly_group = tid // half_block
    pos_in_group = tid % half_block

    # Indices of the two elements in this butterfly pair
    first_idx = butterfly_group * half_block * 2 + pos_in_group
    second_idx = first_idx + half_block

    if second_idx >= n:
        return

    # Load values
    a_real = tl.load(real_ptr + first_idx)
    a_imag = tl.load(imag_ptr + first_idx)
    b_real = tl.load(real_ptr + second_idx)
    b_imag = tl.load(imag_ptr + second_idx)

    # Twiddle factor: W = e^(-i * pi * k / half_block)
    angle = PI * pos_in_group / half_block
    w_real = tl.cos(-angle)
    w_imag = tl.sin(-angle)

    # Complex multiply: tw = b * W
    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real

    # Butterfly: a' = a + tw, b' = a - tw
    result_a_real = a_real + tw_real
    result_a_imag = a_imag + tw_imag
    result_b_real = a_real - tw_real
    result_b_imag = a_imag - tw_imag

    # Store
    tl.store(real_ptr + first_idx, result_a_real)
    tl.store(imag_ptr + first_idx, result_a_imag)
    tl.store(real_ptr + second_idx, result_b_real)
    tl.store(imag_ptr + second_idx, result_b_imag)


@triton.jit
def ifft_stage_kernel(
    real_ptr, imag_ptr,
    n, stage,
):
    """One butterfly stage of the INVERSE FFT.

    Same as forward but with conjugate twiddle factor (positive angle).
    """
    PI = math.pi
    tid = tl.program_id(0)

    if tid >= n // 2:
        return

    half_block = 1 << (stage - 1)
    butterfly_group = tid // half_block
    pos_in_group = tid % half_block

    first_idx = butterfly_group * half_block * 2 + pos_in_group
    second_idx = first_idx + half_block

    if second_idx >= n:
        return

    a_real = tl.load(real_ptr + first_idx)
    a_imag = tl.load(imag_ptr + first_idx)
    b_real = tl.load(real_ptr + second_idx)
    b_imag = tl.load(imag_ptr + second_idx)

    # INVERSE: positive angle (conjugate twiddle)
    angle = PI * pos_in_group / half_block
    w_real = tl.cos(angle)
    w_imag = tl.sin(angle)

    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real

    result_a_real = a_real + tw_real
    result_a_imag = a_imag + tw_imag
    result_b_real = a_real - tw_real
    result_b_imag = a_imag - tw_imag

    tl.store(real_ptr + first_idx, result_a_real)
    tl.store(imag_ptr + first_idx, result_a_imag)
    tl.store(real_ptr + second_idx, result_b_real)
    tl.store(imag_ptr + second_idx, result_b_imag)


@triton.jit
def scale_kernel(
    real_ptr, imag_ptr,
    n_elements, scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Scale all elements by 1/N after inverse FFT."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    r = tl.load(real_ptr + offset, mask=mask)
    i = tl.load(imag_ptr + offset, mask=mask)
    tl.store(real_ptr + offset, r * scale, mask=mask)
    tl.store(imag_ptr + offset, i * scale, mask=mask)
