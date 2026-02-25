# MatMul - Pure Triton (Robust Universal)
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _gemm_kernel(
    A, B, C, Bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias, # Bias is typically [N], so stride is for the N dim
    alpha, beta,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    General Matrix Multiply Kernel with Bias, Alpha, and Beta.
    Supports transposed inputs via stride manipulation in the adapter.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Grouping for cache efficiency
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    # Offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # Load A and B pointers
    # Use max_contiguous for performance hints
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    
    a_ptr = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + rk) < K
        a = tl.load(a_ptr, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptr, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk
        
    # Scale result by alpha
    if alpha != 1.0:
        acc *= alpha
        
    # Add bias if present
    if HAS_BIAS:
        # Bias is [N]
        bias = tl.load(Bias + rn * stride_bias, mask=rn < N, other=0.0)
        acc += bias[None, :]
        
    # Final scaling by beta? 
    # (Typically for addmm: out = beta * bias + alpha * mm)
    # Our kernel does: out = alpha * mm + bias
    # Adapter handles beta by pre-scaling bias if needed.
    
    # Store result - clamp to fp16 range before cast to prevent inf
    c_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    acc = tl.minimum(tl.maximum(acc, -65000.0), 65000.0)
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=mask)
