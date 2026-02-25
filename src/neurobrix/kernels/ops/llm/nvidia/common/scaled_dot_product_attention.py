# Scaled Dot Product Attention (Flash Attention style)
# Type: Triton Kernel (Complex)
# NeuroBrix - NVIDIA Common (All architectures)
# Reference: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

import torch
import triton
import triton.language as tl
from typing import Optional

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention forward kernel.
    
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers to Q, K, V for this batch/head
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + \
             (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + \
             (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + \
             (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    
    # Initialize accumulator and max/sum for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q block (stays in SRAM)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        
        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # Mask out-of-bounds
        qk = tl.where(offs_m[:, None] < N_CTX, qk, float('-inf'))
        qk = tl.where((start_n + offs_n)[None, :] < N_CTX, qk, float('-inf'))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Scale accumulator
        acc = acc * alpha[:, None]
        
        # Load V and accumulate
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        p = tl.exp(qk - m_new[:, None])
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update running max and sum
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + \
             (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _sdpa_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled Dot Product Attention using Triton Flash Attention.

    Handles non-power-of-2 head dimensions (e.g., PixArt head_dim=72)
    by padding to the next power of 2.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda

    # Input shape: [batch, heads, seq_len, head_dim]
    batch, heads, seq_len, head_dim = q.shape

    # Auto scale (based on original head_dim, not padded)
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Compute padded dimension for Triton (must be power of 2)
    BLOCK_DMODEL = _next_power_of_2(head_dim)
    needs_padding = (BLOCK_DMODEL != head_dim)

    # Pad Q, K, V if needed
    if needs_padding:
        pad_size = BLOCK_DMODEL - head_dim
        q = torch.nn.functional.pad(q, (0, pad_size))  # Pad last dim
        k = torch.nn.functional.pad(k, (0, pad_size))
        v = torch.nn.functional.pad(v, (0, pad_size))

    # Output (padded shape)
    out = torch.empty_like(q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    _attn_fwd_kernel[grid](
        q, k, v, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=2,
    )

    # Remove padding from output
    if needs_padding:
        out = out[:, :, :, :head_dim]

    return out

@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="scaled_dot_product_attention")
def sdpa_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Scaled Dot Product Attention using Triton Flash Attention.
    """
    return _sdpa_impl(q, k, v, scale)

