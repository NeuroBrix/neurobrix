"""Vector norm — pure @triton.jit kernels.

Computes L2, L1, L0 (count non-zero), Linf (max abs), L-inf (min abs),
and general Lp norms.

Two kernel variants:
  - Row-wise: input is [M, N], computes norm over N for each of M rows
  - Flat (two-pass): input is flat [M], first pass reduces blocks, second
    pass reduces block results to scalar

Extracted from FlagGems reference (vector_norm.py).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Row-wise kernels: input [M, N] -> output [M]
# ---------------------------------------------------------------------------

@triton.jit
def l2_norm_kernel(
    X, Out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """L2 norm per row: out[m] = sqrt(sum(x[m,:]^2))."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += a * a

    s = tl.sum(_sum, axis=1)
    out = tl.sqrt(s)[:, None]
    tl.store(Out, out, mask=row_mask)


@triton.jit
def l1_norm_kernel(
    X, Out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """L1 norm per row: out[m] = sum(|x[m,:]|)."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.abs(a)

    s = tl.sum(_sum, axis=1)
    out = s[:, None]
    tl.store(Out, out, mask=row_mask)


@triton.jit
def linf_norm_kernel(
    X, Out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """L-infinity norm per row: out[m] = max(|x[m,:]|)."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _max = tl.maximum(tl.abs(a), _max)

    m = tl.max(_max, axis=1)
    out = m[:, None]
    tl.store(Out, out, mask=row_mask)


@triton.jit
def l0_norm_kernel(
    X, Out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """L0 norm per row: out[m] = count(x[m,:] != 0)."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.where(a != 0, 1.0, 0.0)

    s = tl.sum(_sum, axis=1)
    out = s[:, None]
    tl.store(Out, out, mask=row_mask)


@triton.jit(do_not_specialize=['ord'])
def lp_norm_kernel(
    X, Out,
    M, N,
    ord,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """General Lp norm per row: out[m] = sum(|x[m,:]|^p)^(1/p)."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.extra.cuda.libdevice.pow(tl.abs(a), ord)

    s = tl.sum(_sum, axis=1)
    out = tl.extra.cuda.libdevice.pow(s, 1.0 / ord)[:, None]
    tl.store(Out, out, mask=row_mask)


# ---------------------------------------------------------------------------
# Flat two-pass kernels: input [M] -> scalar
# ---------------------------------------------------------------------------

@triton.jit
def l2_norm_pass1_kernel(
    X, Mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass of flat L2 norm: each block computes partial sum of squares."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid + pid, mid)


@triton.jit
def l2_norm_pass2_kernel(
    Mid, Out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    """Second pass: reduce partial sums and take sqrt."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


@triton.jit
def linf_norm_pass1_kernel(
    X, Mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass of flat Linf norm: each block computes partial max abs."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    mid = tl.max(tl.abs(x))
    tl.store(Mid + pid, mid)


@triton.jit
def linf_norm_pass2_kernel(
    Mid, Out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    """Second pass: reduce partial maxes."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    out = tl.max(mid)
    tl.store(Out, out)
