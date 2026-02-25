"""
SiLU Kernel - NVIDIA
Sigmoid Linear Unit (SiLU) / Swish: x * sigmoid(x)

Source: FlagGems (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SiLU = x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    
    sigmoid = 1 / (1 + tl.exp(-x))
    output = x * sigmoid
    
    tl.store(out_ptr + offsets, output, mask=mask)


# =============================================================================
# WRAPPER
# =============================================================================

def _silu_impl(x: torch.Tensor, config: dict) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = (triton.cdiv(n_elements, config["BLOCK_SIZE"]),)
    
    _silu_kernel[grid](
        x, out, n_elements,
        BLOCK_SIZE=config["BLOCK_SIZE"],
    )
    return out


# =============================================================================
# REGISTRATION
# =============================================================================

COMMON_CONFIG = {"BLOCK_SIZE": 1024}

@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="silu")
def silu_kernel(x: torch.Tensor) -> torch.Tensor:
    """SiLU kernel compatible with all NVIDIA architectures."""
    return _silu_impl(x, COMMON_CONFIG)
