"""
Universal Where (Select) - Triton Optimized
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel

@triton.jit
def _where_kernel(cond_ptr, x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load condition (cast to bool/int1)
    cond = tl.load(cond_ptr + offsets, mask=mask).to(tl.int1)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = tl.where(cond, x, y)
    tl.store(out_ptr + offsets, out, mask=mask)

def _where_impl(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Where implementation using Triton."""
    assert condition.is_cuda

    # Broadcast to common shape
    out_shape = torch.broadcast_shapes(condition.shape, x.shape, y.shape)

    condition = condition.expand(out_shape).contiguous()
    x = x.expand(out_shape).contiguous()
    y = y.expand(out_shape).contiguous()

    output = torch.empty_like(x)
    n = output.numel()

    if n == 0:
        return output

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    with torch.cuda.device(condition.device):
        _where_kernel[grid](
            condition.view(-1), x.view(-1), y.view(-1), output.view(-1),
            n,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )

    return output.view(out_shape)


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="where")
def where_kernel(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Where operation - pure Triton."""
    return _where_impl(condition, x, y)

