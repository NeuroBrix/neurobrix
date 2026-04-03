"""Weight normalization (_weight_norm_interface) — pure @triton.jit kernels.

Computes w = g * v / ||v|| where g is a per-channel gain scalar.
Two variants: norm along dim=0 ("first") and dim=last ("last").

Ported from FlagGems weightnorm.py. Stripped FlagGems decorators.
"""

import triton
import triton.language as tl


# ---- Forward: dim = 0 (norm along first dimension, one gain per row) ----

@triton.jit
def weight_norm_kernel_first(
    output_ptr,
    norm_ptr,
    v_ptr,
    g_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Weight norm forward, normalizing along dim=0 (rows).

    v: [M, N], g: [M], norm: [M], output: [M, N]
    Each program handles BLOCK_M rows.
    """
    ty = tl.arange(0, BLOCK_M)[:, None]
    by = tl.program_id(0) * BLOCK_M
    row_offset = by + ty
    row_mask = row_offset < M

    tx = tl.arange(0, BLOCK_N)[None, :]

    # Pass 1: compute ||v|| per row
    v_sq = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for base in range(0, N, BLOCK_N):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_val = tl.load(v_ptr + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_sq += v_val * v_val

    norm_val = tl.sqrt(tl.sum(v_sq, axis=1) + eps)
    tl.store(norm_ptr + row_offset, norm_val[:, None], mask=row_mask)
    g_val = tl.load(g_ptr + row_offset, mask=row_mask).to(tl.float32)

    # Pass 2: output = g * v / ||v||
    for base in range(0, N, BLOCK_N):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_val = tl.load(v_ptr + row_offset * N + col_offset, mask=mask).to(tl.float32)
        out = (v_val / norm_val[:, None]) * g_val
        tl.store(output_ptr + row_offset * N + col_offset, out, mask=mask)


# ---- Forward: dim = last (norm along last dimension, one gain per column) ----

@triton.jit
def weight_norm_kernel_last(
    output_ptr,
    norm_ptr,
    v_ptr,
    g_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Weight norm forward, normalizing along last dim (columns).

    v: [M, N], g: [N], norm: [N], output: [M, N]
    Each program handles BLOCK_N columns.
    """
    tx = tl.arange(0, BLOCK_N)[:, None]
    bx = tl.program_id(0) * BLOCK_N
    col_offset = bx + tx
    col_mask = col_offset < N

    ty = tl.arange(0, BLOCK_M)[None, :]

    # Pass 1: compute ||v|| per column
    v_sq = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    for base in range(0, M, BLOCK_M):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_val = tl.load(v_ptr + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_sq += v_val * v_val

    norm_val = tl.sqrt(tl.sum(v_sq, axis=1) + eps)
    tl.store(norm_ptr + col_offset, norm_val[:, None], mask=col_mask)
    g_val = tl.load(g_ptr + col_offset, mask=col_mask).to(tl.float32)

    # Pass 2: output = g * v / ||v||
    for base in range(0, M, BLOCK_M):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_val = tl.load(v_ptr + row_offset * N + col_offset, mask=mask).to(tl.float32)
        out = (v_val / norm_val[:, None]) * g_val
        tl.store(output_ptr + row_offset * N + col_offset, out, mask=mask)
