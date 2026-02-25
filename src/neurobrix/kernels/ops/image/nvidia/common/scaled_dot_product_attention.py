# Scaled Dot Product Attention - Pure Triton (Universal)
# NeuroBrix - NVIDIA Common
# Based on Flash Attention v2 algorithm

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, attn_mask, sm_scale, LSE, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch, heads, seq_q, seq_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    """
    Flash Attention Forward Kernel (Universal) with Masking Support.

    Uses online softmax with log-sum-exp normalization for numerical stability.
    BLOCK_DMODEL must be a power of 2 (handled by adapter via padding).
    """
    pid_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // heads
    off_h = off_hb % heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers with proper strides
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    # Initialize statistics
    # m_i: Running max of attention scores
    # l_i: Running sum of exp(score - m_i)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Query mask
    q_mask = offs_m[:, None] < seq_q

    # Load Q once
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Scale factor for log2-based softmax
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # Loop over K, V blocks
    for start_n in range(0, seq_k, BLOCK_N):
        k_mask = (start_n + offs_n) < seq_k

        # Load K
        k = tl.load(k_ptrs + start_n * stride_kn, mask=k_mask[None, :], other=0.0)

        # QK^T
        qk = tl.dot(q, k)
        
        # Apply sm_scale before masking
        qk = qk * qk_scale

        # Apply Attention Mask if present
        if HAS_ATTN_MASK:
            mask_off = off_b * stride_mask_b + off_h * stride_mask_h + \
                       offs_m[:, None] * stride_mask_m + (start_n + offs_n)[None, :] * stride_mask_n
            m_ptrs = attn_mask + mask_off
            # Load mask
            m = tl.load(m_ptrs, mask=q_mask & k_mask[None, :], other=-float("inf"))
            qk = qk + m

        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale factors
        # exp2(m_i - m_new)
        # Use where to avoid NaN if m_new is -inf
        alpha = tl.math.exp2(tl.where(m_new > -float("inf"), m_i - m_new, 0.0))
        
        # exp2(qk - m_new)
        p = tl.math.exp2(tl.where(m_new[:, None] > -float("inf"), qk - m_new[:, None], -float("inf")))

        l_ij = tl.sum(p, 1)
        l_new = l_i * alpha + l_ij

        # Rescale accumulator
        acc = acc * alpha[:, None]

        # Load V and accumulate
        v = tl.load(v_ptrs + start_n * stride_vn, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Final normalization
    # Avoid div by zero if l_i is 0 (though should not happen)
    acc = acc / tl.where(l_i[:, None] > 0, l_i[:, None], 1.0)

    # Store output with proper strides
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)
