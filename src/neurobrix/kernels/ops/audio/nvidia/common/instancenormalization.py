# Instance Normalization
# Type: Triton Kernel (Normalization)
# NeuroBrix - NVIDIA Common (All architectures)

import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


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
    
    # pid maps to (n, c)
    n_idx = pid // C
    c_idx = pid % C
    
    if n_idx >= N:
        return
    
    # Compute mean
    mean = tl.zeros([1], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / HW
    
    # Compute variance
    var = tl.zeros([1], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / HW
    
    # Load gamma, beta
    gamma = tl.load(gamma_ptr + c_idx) if gamma_ptr else 1.0
    beta_val = tl.load(beta_ptr + c_idx) if beta_ptr else 0.0
    
    # Normalize and store
    inv_std = 1.0 / tl.sqrt(var + eps)
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = hw_offs < HW
        x_offs = n_idx * stride_n + c_idx * stride_c + hw_offs * stride_hw
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std * gamma + beta_val
        tl.store(out_ptr + x_offs, y, mask=mask)


def _instancenorm_impl(
    x: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Instance Normalization using Triton."""
    assert x.is_cuda
    x = x.contiguous()
    
    N, C, H, W = x.shape
    HW = H * W
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(HW))
    grid = (N * C,)
    
    _instancenorm_kernel[grid](
        x, out,
        gamma, beta,
        N, C, HW,
        eps,
        x.stride(0), x.stride(1), 1,  # stride_hw = 1 for contiguous H*W
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return out

def _instance_norm_impl(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """InstanceNorm implementation using Triton."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    # Instance norm: normalize each (N, C) sample independently
    N, C = x.shape[:2]
    spatial = x.shape[2:]
    spatial_size = 1
    for s in spatial:
        spatial_size *= s
    
    # Reshape to (N*C, spatial_size)
    x_2d = x.view(N * C, spatial_size)
    y = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(spatial_size)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    if BLOCK_SIZE > 65536:
        BLOCK_SIZE = 65536
    
    _instance_norm_kernel[(N * C,)](
        x_2d, y,
        weight, bias,
        x_2d.stride(0),
        spatial_size, C, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view_as(x)


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="instancenormalization")
def instance_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    InstanceNorm using Triton.
    """
    return _instance_norm_impl(x, weight, bias, eps)

