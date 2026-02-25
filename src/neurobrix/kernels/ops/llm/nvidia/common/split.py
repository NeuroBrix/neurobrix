

import torch
import triton
import triton.language as tl
from typing import List, Union

from neurobrix.kernels.registry import register_kernel

# Split - Split tensor into chunks
# Type: Triton Kernel (Memory - copies data)
# NeuroBrix - NVIDIA Common (All architectures)

@triton.jit
def _split_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    in_offset,
    out_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy n_elements from in_ptr+in_offset to out_ptr+out_offset."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + in_offset + offsets, mask=mask)
    tl.store(out_ptr + out_offset + offsets, x, mask=mask)


@triton.jit
def _split_strided_copy_kernel(
    in_ptr,
    out_ptr,
    total_elements,
    # Dimensions
    before_dim,
    in_dim_size,
    out_dim_size,
    after_dim,
    # Offset in input along split dimension
    dim_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy a slice from input to output with proper striding.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert flat output index to 3D indices
    # total_elements = before_dim * out_dim_size * after_dim
    out_dim_after = out_dim_size * after_dim
    
    b_idx = offsets // out_dim_after
    remainder = offsets % out_dim_after
    d_idx = remainder // after_dim
    a_idx = remainder % after_dim
    
    # Compute input offset
    # Input index: [b_idx, dim_offset + d_idx, a_idx]
    in_dim_after = in_dim_size * after_dim
    in_offset = b_idx * in_dim_after + (dim_offset + d_idx) * after_dim + a_idx
    
    # Load and store
    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _split_impl(x: torch.Tensor, split_sizes: List[int], dim: int) -> List[torch.Tensor]:
    """Split tensor into chunks using Triton copies."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    ndim = x.ndim
    dim = dim if dim >= 0 else ndim + dim
    
    total_split = sum(split_sizes)
    assert total_split == x.shape[dim], \
        f"Split sizes {split_sizes} sum to {total_split}, expected {x.shape[dim]}"
    
    # Compute dimensions for 3D view: [before_dim, dim, after_dim]
    before_dim = 1
    for d in range(dim):
        before_dim *= x.shape[d]
    
    after_dim = 1
    for d in range(dim + 1, ndim):
        after_dim *= x.shape[d]
    
    in_dim_size = x.shape[dim]
    
    outputs = []
    BLOCK_SIZE = 1024
    
    dim_offset = 0
    for split_size in split_sizes:
        # Create output tensor
        out_shape = list(x.shape)
        out_shape[dim] = split_size
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        
        if split_size == 0:
            outputs.append(out)
            continue
        
        total_elements = before_dim * split_size * after_dim
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        _split_strided_copy_kernel[grid](
            x, out, total_elements,
            before_dim, in_dim_size, split_size, after_dim,
            dim_offset,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        
        outputs.append(out)
        dim_offset += split_size
    
    return outputs


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="split")
def split_kernel(x: torch.Tensor, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> List[torch.Tensor]:
    """
    Split tensor into chunks using pure Triton.
    
    Args:
        x: Input tensor
        split_size_or_sections: Size of each chunk (int) or list of sizes
        dim: Dimension to split along (default: 0)
    
    Returns:
        List of tensor chunks (all contiguous)
    """
    dim_size = x.shape[dim]
    
    if isinstance(split_size_or_sections, int):
        chunk_size = split_size_or_sections
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        n_full_chunks = dim_size // chunk_size
        remainder = dim_size % chunk_size
        
        split_sizes = [chunk_size] * n_full_chunks
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        split_sizes = list(split_size_or_sections)
    
    return _split_impl(x, split_sizes, dim)
