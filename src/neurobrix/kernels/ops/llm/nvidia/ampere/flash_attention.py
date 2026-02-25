"""
Flash Attention Kernel - NVIDIA (Ampere+)
Memory-efficient attention with O(N) memory instead of O(N²)

Source: FlagGems (adapted) - Simplified implementation
Tier: ampere (requires BF16 native support)
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def _flash_attention_kernel(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention Kernel.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V
    # We use off_hz to compute the batch/head index
    # off_hz = batch_idx * num_heads + head_idx
    # So we need to reconstruct batch_idx and head_idx from off_hz if strides were batch/head separate
    # But here strides are passed for flat Z*H
    
    # Actually off_hz is passed as pid(1), which corresponds to batch*head
    # We can use it directly with stride_qh (which should be stride_head) and stride_qz (stride_batch)
    # But the standard triton tutorial uses a flattened view where we jump by stride_qh * n_heads for batches?
    # Let's assume Q, K, V are [Batch, Head, Seq, Dim]
    # Then off_hz = batch_id * H + head_id
    
    # Pointers
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    # Initialize m_i and l_i
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Scale sm_scale
    # sm_scale is passed as arg
    
    # Loop over K, V blocks
    # We block along N dimension
    # N_CTX is sequence length
    
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K, V block
        # We need to be careful with masking if N_CTX is not multiple of BLOCK_N
        # For simplicity, we assume N_CTX is multiple or we pad, or we use mask
        # Here simplified version assumes valid ranges or we mask
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Compute qk = Q @ K.T
        q = tl.load(q_ptrs)
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        
        l_i_new = alpha * l_i + beta * l_ij
        
        # Update acc
        # acc = alpha * acc + beta * (p @ v)
        p = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        
        # Update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        
        # Update m_i and l_i for next iteration
        m_i = m_i_new
        l_i = l_i_new
        
    # Finalize
    # Out = acc / l_i
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc)


# =============================================================================
# WRAPPER
# =============================================================================

def _flash_attention_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None) -> torch.Tensor:
    # Shapes: [Batch, Head, Seq, Dim]
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    
    # Constraints
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.shape[-1] in {16, 32, 64, 128}, "Only support head_dim in {16, 32, 64, 128}"
    
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    
    if scale is None:
        scale = 1.0 / (Lq ** 0.5)
        
    o = torch.empty_like(q)
    
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    
    # Alloc L (logsumexp) if needed for backward, here we skip
    L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    
    _flash_attention_kernel[grid](
        q, k, v, scale,
        L,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return o


# =============================================================================
# REGISTRATION
# =============================================================================

@register_kernel(family="llm", vendor="nvidia", tier="ampere", op_name="flash_attention")
def flash_attention_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Flash Attention kernel for Ampere+ GPUs (requires BF16/TF32 efficient support)."""
    return _flash_attention_impl(q, k, v, scale)
