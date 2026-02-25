# Instance Normalization - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _instancenorm_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr,
    N, C, HW,
    eps,
    stride_n, stride_c, stride_hw,
    BLOCK_SIZE: tl.constexpr,
):
    """Instance Norm: normalize over H*W for each (N, C)."""
    pid = tl.program_id(0)
    n_idx = pid // C
    c_idx = pid % C
    if n_idx >= N: return
    
    mean = tl.zeros([1], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / HW
    
    var = tl.zeros([1], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / HW
    
    gamma = tl.load(gamma_ptr + c_idx) if gamma_ptr else 1.0
    beta_val = tl.load(beta_ptr + c_idx) if beta_ptr else 0.0
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std * gamma + beta_val
        tl.store(out_ptr + x_offs, y, mask=mask)
