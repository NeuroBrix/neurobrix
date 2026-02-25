"""
Tile - Repeat tensor along dimensions
ATen: aten::repeat, aten::tile
"""
import torch
import triton
import triton.language as tl
from typing import List

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _tile_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    # Input shape (4D padded)
    in_dim0, in_dim1, in_dim2, in_dim3,
    # Input strides
    in_stride0, in_stride1, in_stride2, in_stride3,
    # Output shape (4D padded)
    out_dim0, out_dim1, out_dim2, out_dim3,
    BLOCK_SIZE: tl.constexpr,
):
    """Tile by computing modulo indices."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert flat output offset to 4D indices
    idx = offsets
    o0 = idx // (out_dim1 * out_dim2 * out_dim3)
    idx = idx % (out_dim1 * out_dim2 * out_dim3)
    o1 = idx // (out_dim2 * out_dim3)
    idx = idx % (out_dim2 * out_dim3)
    o2 = idx // out_dim3
    o3 = idx % out_dim3
    
    # Compute input indices using modulo
    i0 = o0 % in_dim0
    i1 = o1 % in_dim1
    i2 = o2 % in_dim2
    i3 = o3 % in_dim3
    
    # Compute input offset
    in_offset = i0 * in_stride0 + i1 * in_stride1 + i2 * in_stride2 + i3 * in_stride3
    
    # Load and store
    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _tile_impl(x: torch.Tensor, repeats: List[int]) -> torch.Tensor:
    """Tile tensor using Triton."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    ndim = x.ndim
    
    # Pad repeats to match ndim
    if len(repeats) < ndim:
        repeats = [1] * (ndim - len(repeats)) + list(repeats)
    elif len(repeats) > ndim:
        # Prepend 1s to input shape
        x = x.view([1] * (len(repeats) - ndim) + list(x.shape))
        ndim = len(repeats)
    
    if ndim > 4:
        raise NotImplementedError("Tile currently supports up to 4D tensors")
    
    # Compute output shape
    out_shape = [x.shape[d] * repeats[d] for d in range(ndim)]
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    
    if n_elements == 0:
        return out
    
    # Pad to 4D
    in_shape_4d = list(x.shape) + [1] * (4 - ndim)
    in_strides_4d = list(x.stride()) + [0] * (4 - ndim)
    out_shape_4d = out_shape + [1] * (4 - ndim)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _tile_kernel[grid](
        x, out, n_elements,
        in_shape_4d[0], in_shape_4d[1], in_shape_4d[2], in_shape_4d[3],
        in_strides_4d[0], in_strides_4d[1], in_strides_4d[2], in_strides_4d[3],
        out_shape_4d[0], out_shape_4d[1], out_shape_4d[2], out_shape_4d[3],
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )
    
    return out

def _tile_impl(x: torch.Tensor, repeats: list) -> torch.Tensor:
    """Tile implementation."""
    assert x.is_cuda
    return x.repeat(*repeats)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="tile")
def tile_kernel(x: torch.Tensor, repeats: list = None) -> torch.Tensor:
    """
    Tile (repeat) operation.
    """
    if repeats is None:
        repeats = [1] * x.ndim
    return _tile_impl(x, repeats)

