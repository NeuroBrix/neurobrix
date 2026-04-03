"""Variance — pure @triton.jit kernels.

Welford online algorithm for numerically-stable variance.
Global reduction via two-pass and dim-specific Welford kernel.
Extracted from FlagGems var_mean (Apache-2.0).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Welford combine function (used by tl.reduce)
# ---------------------------------------------------------------------------

@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    """Combine two Welford partial aggregates."""
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


# ---------------------------------------------------------------------------
# Global reduction: two-pass (partial stats → combine)
# ---------------------------------------------------------------------------

@triton.jit
def var_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    """Pass 1: compute partial sum, sum-of-squares, and count per block."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@triton.jit(do_not_specialize=["correction"])
def var_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    """Pass 2: Welford-combine all blocks and output variance."""
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    _mean, _, nvar = tl.reduce(
        (average, count, acc), axis=0, combine_fn=welford_func
    )

    var = nvar / (N - correction)
    tl.store(Var, var)


# ---------------------------------------------------------------------------
# Dim-specific: Welford online variance along one axis
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=["correction"])
def var_welford_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Welford variance along compressed inner dim N.

    X: [M, N] (dim-compressed) → Var: [M]
    """
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

        count = _count + mask
        cnt = tl.maximum(count, 1)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask
        _mean = cur_mean
        _count = count

    _mean_r, _, acc_r = tl.reduce(
        (_mean, _count, _acc), axis=1, combine_fn=welford_func
    )
    var = acc_r / (N - correction)
    var = var[:, None]
    tl.store(Var, var, row_mask)
