"""Standard deviation — pure @triton.jit kernels.

Global reduction via map-reduce (sum + sum_sq) and dim-specific two-pass
(mean, then variance). Extracted from FlagGems (Apache-2.0).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Global reduction: map-reduce
# ---------------------------------------------------------------------------

@triton.jit
def std_map_kernel(X, Tmp_sum, Tmp_sum_sq, N, BLOCK_N: tl.constexpr):
    """Pass 1: compute partial sum and sum-of-squares per block."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x * x, axis=0)
    tl.store(Tmp_sum + pid, sum_val)
    tl.store(Tmp_sum_sq + pid, sum_sq_val)


@triton.jit
def std_reduce_kernel(
    Tmp_sum, Tmp_sum_sq, Out, N, correction, BLOCK_NUM, BLOCK_SIZE: tl.constexpr
):
    """Pass 2: combine blocks and compute std = sqrt(var)."""
    total_sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_sum_sq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, BLOCK_NUM, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < BLOCK_NUM
        tmp_sum_vals = tl.load(Tmp_sum + offset, mask=mask, other=0.0).to(tl.float32)
        tmp_sum_sq_vals = tl.load(Tmp_sum_sq + offset, mask=mask, other=0.0).to(
            tl.float32
        )
        total_sum_acc += tmp_sum_vals
        total_sum_sq_acc += tmp_sum_sq_vals
    total_sum = tl.sum(total_sum_acc, axis=0)
    total_sum_sq = tl.sum(total_sum_sq_acc, axis=0)
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Dim-specific: two-pass (mean, then variance)
# ---------------------------------------------------------------------------

@triton.jit
def std_dim_kernel(
    X,
    Out,
    stride_x_row,
    stride_x_col,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute std along compressed inner dim N.

    X: [M, N] (dim-compressed, strided) → Out: [M]
    Two-pass: first compute mean, then sum of squared deviations.
    """
    pid_group = tl.program_id(0)
    start_row = pid_group * BLOCK_M
    row_offsets = start_row + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    # Pass 1: compute mean
    mean_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_row_ptrs = X + row_offsets[:, None] * stride_x_row

    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        mean_acc += x.to(tl.float32)

    mean = tl.sum(mean_acc, axis=1) / N

    # Pass 2: compute variance
    var_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        diff = x.to(tl.float32) - mean[:, None]
        var_acc += tl.where(final_mask, diff * diff, 0.0)

    var = tl.sum(var_acc, axis=1)

    denom = N - correction
    var = var / tl.maximum(denom, 1e-12)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)

    out_ptrs = Out + row_offsets
    tl.store(out_ptrs, std_dev.to(Out.dtype.element_ty), mask=row_mask)
