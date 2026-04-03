"""Scaled softmax — pure @triton.jit kernel.

Computes softmax(input * scale_factor) for attention score tensors.
Input shape: [B, H, Q, K] (batch, heads, query_seq, key_seq).

Uses online softmax (two-pass): first pass computes max and exp-sum,
second pass normalizes. Numerically stable.

Extracted from FlagGems reference (scaled_softmax.py).
"""

import triton
import triton.language as tl


@triton.jit
def scaled_softmax_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scaled softmax: out = softmax(input * scale_factor) along last dim.

    Grid: (cdiv(query_seq_len, BLOCK_Q), num_heads, batch_size)
    """
    query_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_q = query_tile_idx * BLOCK_Q
    q_offsets = start_q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < query_seq_len

    row_start = (
        input_ptr
        + batch_idx * stride_b
        + head_idx * stride_h
        + q_offsets * stride_q
    )

    # Pass 1: find row max and exp-sum (online softmax)
    m = tl.full([BLOCK_Q], -float('inf'), dtype=tl.float32)
    exp_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        block_ptr = row_start[:, None] + k_offsets[None, :]

        row_mask = q_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        valid = row_mask & col_mask

        s = tl.load(block_ptr, mask=valid, other=-float('inf'))
        s = s * scale_factor

        m_new = tl.max(s, axis=1)
        m_old = m
        m = tl.maximum(m_old, m_new)

        # Rescale previous sum
        exp_sum = exp_sum * tl.exp(m_old - m)
        # Add current block
        s_exp = tl.exp(s - m[:, None])
        exp_sum = exp_sum + tl.sum(tl.where(valid, s_exp, 0.0), axis=1)

    exp_sum_inv = 1.0 / exp_sum

    # Pass 2: compute and store normalized values
    out_row_start = (
        output_ptr
        + batch_idx * stride_b
        + head_idx * stride_h
        + q_offsets * stride_q
    )

    for k_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        in_ptr = row_start[:, None] + k_offsets[None, :]
        out_ptr = out_row_start[:, None] + k_offsets[None, :]

        row_mask = q_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        valid = row_mask & col_mask

        s = tl.load(in_ptr, mask=valid, other=-float('inf'))
        s = s * scale_factor
        p = tl.exp(s - m[:, None]) * exp_sum_inv[:, None]

        tl.store(out_ptr, p, mask=valid)
