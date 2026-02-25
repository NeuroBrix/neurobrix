"""
SwiGLU Kernel - NVIDIA
SwiGLU activation: silu(gate) * up

Source: Liger-Kernel (adapted)
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
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load gate and up
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # SwiGLU = silu(gate) * up
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    silu_gate = gate / (1.0 + tl.exp(-gate))
    out = silu_gate * up
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


# =============================================================================
# WRAPPER
# =============================================================================

def _swiglu_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert gate.shape == up.shape, "Gate and Up projections must have same shape"
    assert gate.is_contiguous() and up.is_contiguous(), "Inputs must be contiguous"
    
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    
    # Heuristic
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _swiglu_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# =============================================================================
# REGISTRATION
# =============================================================================

@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="swiglu")
def swiglu_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU kernel compatible with all NVIDIA architectures."""
    return _swiglu_impl(gate, up)
