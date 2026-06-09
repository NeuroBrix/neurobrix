"""DFT / iDFT basis matrices — pure @triton.jit kernels.

Builds the real cos/sin basis matrices for a Discrete Fourier Transform computed
by matrix multiplication. This is the standard small-N / non-power-of-2 path
(the DFT is an N×N matrix of complex exponentials applied to the signal) — used
when the radix-2 butterfly cannot run (n_fft not a power of 2, e.g. an iSTFT head
with n_fft=20). Model-agnostic: any non-pow2 FFT length routes here.

Layout (matches the rfft convention): frequency bins k in [0, N//2+1), samples
n in [0, N). The matrices are [N_bins, N], applied as `frames @ mat.T`.
"""

import triton
import triton.language as tl


@triton.jit
def dft_rfft_matrix_kernel(
    cos_ptr,      # (N_bins, N) — cos(2π k n / N)
    nsin_ptr,     # (N_bins, N) — -sin(2π k n / N)
    N_bins, N,
    two_pi_over_N,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward rfft basis: X[k] = Σ_n x[n] (cos - i sin)(2π k n / N).

    cos[k,n] = cos(2π k n / N), nsin[k,n] = -sin(2π k n / N) so that
    X_real = x @ cos.T and X_imag = x @ nsin.T.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N_bins * N
    mask = offs < total
    k = offs // N
    n = offs % N
    angle = (k * n).to(tl.float32) * two_pi_over_N
    tl.store(cos_ptr + offs, tl.cos(angle), mask=mask)
    tl.store(nsin_ptr + offs, -tl.sin(angle), mask=mask)


@triton.jit
def idft_c2r_matrix_kernel(
    cos_ptr,      # (N_bins, N) — weight_k cos(2π k n / N) / N
    sin_ptr,      # (N_bins, N) — -weight_k sin(2π k n / N) / N
    N_bins, N,
    two_pi_over_N,
    inv_N,
    n_even,       # 1 if N is even (Nyquist bin present), else 0
    BLOCK_SIZE: tl.constexpr,
):
    """Inverse rfft (irfft) basis, exploiting Hermitian symmetry.

    x[n] = (1/N) Σ_k weight_k [Re(X[k]) cos(2π k n / N) - Im(X[k]) sin(2π k n / N)]
    weight_k = 1 for k=0 and (N even) k=N/2, else 2. So
    real_out = X_real @ cos + X_imag @ sin.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N_bins * N
    mask = offs < total
    k = offs // N
    n = offs % N
    is_edge = (k == 0) | ((n_even == 1) & (k == N_bins - 1))
    weight = tl.where(is_edge, 1.0, 2.0) * inv_N
    angle = (k * n).to(tl.float32) * two_pi_over_N
    tl.store(cos_ptr + offs, weight * tl.cos(angle), mask=mask)
    tl.store(sin_ptr + offs, -weight * tl.sin(angle), mask=mask)
