"""The two passes of an exact tiled group norm on the Triton engine (`core/strategies/tiled_norm.py`'s NBXTensor twin):
the statistics of each (sample, group) over the WHOLE input, then a slice normalised with them.

`group_norm_stats_kernel` is the first pass of `groupnorm.group_norm_forward_kernel` alone — the same fp32 sum and
sum of squares, var = ss/n - mean^2 — without writing the normalised tensor (a stretch sliced along the axis holds its
input whole, never a whole output beside it). `group_norm_apply_kernel` is its second pass on a slice, reading the
given statistics. One program per (sample, group); the walk is a fixed chunk (no autotune key: nothing to certify), so
the summation order is fixed.

Pure @triton.jit (R33).
"""

import triton
import triton.language as tl

#: the chunk a program walks its group by — fixed, so the statistics' summation order is
TILED_NORM_BLOCK = 4096
TILED_NORM_WARPS = 8


@triton.jit
def group_norm_stats_kernel(X, Mean, Rstd, group_size, C, HW, num_groups, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    base = batch_idx * C * HW + group_idx * group_size * HW
    hidden = group_size * HW
    block = tl.arange(0, BLOCK_SIZE)
    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block
        mask = idx < hidden
        x = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x)
        ss += tl.sum(x * x)
    inv_n = 1.0 / hidden
    mean = s * inv_n
    var = ss * inv_n - mean * mean
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, 1.0 / tl.sqrt(var + eps))


@triton.jit
def group_norm_apply_kernel(X, Y, W, B, Mean, Rstd, group_size, C, HW, num_groups,
                            scale_by_weight: tl.constexpr, add_bias: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    chan_start = group_idx * group_size
    base = batch_idx * C * HW + chan_start * HW
    hidden = group_size * HW
    mean = tl.load(Mean + pid)
    rstd = tl.load(Rstd + pid)
    block = tl.arange(0, BLOCK_SIZE)
    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block
        mask = idx < hidden
        x = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        if scale_by_weight or add_bias:
            chan = chan_start + idx // HW
            if scale_by_weight:
                y = y * tl.load(W + chan, mask=mask, other=1.0).to(tl.float32)
            if add_bias:
                y = y + tl.load(B + chan, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + base + idx, y.to(Y.dtype.element_ty), mask=mask)
