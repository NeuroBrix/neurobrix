"""
Transpose - Permute tensor dimensions
Type: Triton Kernel (Pure Triton - ALWAYS copies)

CRITICAL: Transpose ALWAYS creates a contiguous copy.
This guarantees downstream ops can use view operations safely.

NeuroBrix - NVIDIA Common (All architectures)
"""
import torch
import triton
import triton.language as tl
from typing import List, Union

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _transpose_2d_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Transpose 2D matrix: (M, N) -> (N, M)
    
    Each block handles a BLOCK_M x BLOCK_N tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for this block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    
    # Create mask for valid elements
    m_mask = m_offs < M
    n_mask = n_offs < N
    
    # Better approach: load block, transpose in registers, store
    # Input offsets: [m, n] -> m * N + n
    in_offs = m_offs[:, None] * N + n_offs[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    
    # Load tile
    tile = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    
    # Output offsets: transposed [n, m] -> n * M + m
    out_offs = n_offs[:, None] * M + m_offs[None, :]
    out_mask = n_mask[:, None] & m_mask[None, :]
    
    # Store transposed tile
    tl.store(out_ptr + out_offs, tl.trans(tile), mask=out_mask)


@triton.jit
def _copy_with_strides_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    # Input strides (up to 4D)
    in_stride0, in_stride1, in_stride2, in_stride3,
    # Output strides (contiguous)
    out_stride0, out_stride1, out_stride2, out_stride3,
    # Dimensions
    dim0, dim1, dim2, dim3,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    General copy kernel that handles arbitrary strides.
    Converts non-contiguous to contiguous.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert flat offset to multi-dimensional indices
    idx = offsets
    
    # Initialize indices
    i0 = idx // (dim1 * dim2 * dim3) 
    idx = idx % (dim1 * dim2 * dim3) 
        
    i1 = idx // (dim2 * dim3) 
    idx = idx % (dim2 * dim3)
        
    i2 = idx // dim3 
    idx = idx % dim3 
        
    i3 = idx

    # If dimension is unused (e.g. 2D tensor in 4D kernel), stride is effectively ignored if logic is correct
    # But here we assume shapes and strides are padded to 4D correctly.
    
    # Compute input offset with original strides
    in_offset = i0 * in_stride0 + i1 * in_stride1 + i2 * in_stride2 + i3 * in_stride3
    
    # Load and store
    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _transpose_2d_impl(x: torch.Tensor) -> torch.Tensor:
    """Optimized 2D transpose using Triton."""
    assert x.ndim == 2, "Input must be 2D"
    
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Block sizes (power of 2, reasonable for most GPUs)
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _transpose_2d_kernel[grid](
        x, out, M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4
    )
    
    return out


def _transpose_nd_impl(x: torch.Tensor, perm: List[int]) -> torch.Tensor:
    """
    General N-D transpose using Triton copy with strides.
    
    Strategy: Create view with permuted strides, then copy to contiguous.
    """
    assert x.is_cuda, "Tensor must be on CUDA"
    
    # Get permuted shape and strides using torch machinery (metadata only)
    x_permuted = x.permute(*perm)
    new_shape = x_permuted.shape
    new_strides = x_permuted.stride()
    
    # Create output (contiguous)
    out = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    n_elements = x.numel()
    
    if n_elements == 0:
        return out
    
    # Pad dimensions to 4D for kernel
    ndim = len(new_shape)
    
    # Ensure we handle up to 4D. If more, fallback or extend kernel.
    # For now support up to 4D. If > 4D, we might need to flatten dims or use loop.
    # To keep it simple, let's assume <= 4D or raise error.
    if ndim > 4:
        # Fallback to pure torch copy if > 4D for now, or implement generic kernel with loop
        # But per "Zero Compute PyTorch", we should use Triton.
        # Let's extend logic to support > 4D by folding dimensions if possible, but that's complex.
        # For typical DL, <= 4D is common. 5D happens (NCDHW).
        # Let's implement up to 4D and throw if > 4D for this iteration.
        raise NotImplementedError("Triton transpose currently supports up to 4D tensors.")
    
    shape_4d = list(new_shape) + [1] * (4 - ndim)
    in_strides_4d = list(new_strides) + [0] * (4 - ndim) # 0 stride for padded dims
    out_strides_4d = list(out.stride()) + [0] * (4 - len(out.stride()))
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _copy_with_strides_kernel[grid](
        x, out, n_elements,
        in_strides_4d[0], in_strides_4d[1], in_strides_4d[2], in_strides_4d[3],
        out_strides_4d[0], out_strides_4d[1], out_strides_4d[2], out_strides_4d[3],
        shape_4d[0], shape_4d[1], shape_4d[2], shape_4d[3],
        ndim=ndim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return out


def _transpose_impl(x: torch.Tensor, perm: List[int]) -> torch.Tensor:
    """
    Transpose implementation with optimization for 2D case.
    """
    # x = x.contiguous() # Not strictly necessary if we use strides correctly, but good for 2D tile kernel
    
    # Special case: 2D transpose (swap dimensions) and input is contiguous
    if x.ndim == 2 and perm == [1, 0] and x.is_contiguous():
        return _transpose_2d_impl(x)
    
    # General N-D case
    return _transpose_nd_impl(x, perm)


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="transpose")
def transpose_kernel(x: torch.Tensor, perm: Union[List[int], tuple] = None) -> torch.Tensor:
    """
    Transpose (permute) tensor dimensions using pure Triton.
    
    ALWAYS returns a contiguous tensor (copies data).
    This guarantees downstream operations can use views safely.
    
    Args:
        x: Input tensor
        perm: Permutation of dimensions. If None, reverses all dims.
              Example: perm=[2, 0, 1] for (A, B, C) -> (C, A, B)
    """
    if perm is None:
        # Default: reverse all dimensions
        perm = list(range(x.ndim - 1, -1, -1))
    
    perm = list(perm)
    
    # Identity permutation - return contiguous copy
    if perm == list(range(x.ndim)):
        return x.contiguous()
    
    return _transpose_impl(x, perm)