# Slice - Extract a slice from tensor
# Type: Pure Triton Kernel
# NeuroBrix - NVIDIA Common (All architectures)
# ATen API: dim, start, end, step (NO ONNX - axes removed)

import torch
import triton
import triton.language as tl
from typing import List, Optional

@triton.jit
def _strided_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    # 4D indexing params
    in_stride0, in_stride1, in_stride2, in_stride3,
    out_dim0, out_dim1, out_dim2, out_dim3,
    start0, start1, start2, start3,
    step0, step1, step2, step3,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy with strides for slicing - pure Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert flat output index to 4D indices
    idx = offsets
    i0 = idx // (out_dim1 * out_dim2 * out_dim3)
    idx = idx % (out_dim1 * out_dim2 * out_dim3)
    i1 = idx // (out_dim2 * out_dim3)
    idx = idx % (out_dim2 * out_dim3)
    i2 = idx // out_dim3
    i3 = idx % out_dim3

    # Compute input indices with start and step
    in_i0 = start0 + i0 * step0
    in_i1 = start1 + i1 * step1
    in_i2 = start2 + i2 * step2
    in_i3 = start3 + i3 * step3

    # Compute input offset
    in_offset = in_i0 * in_stride0 + in_i1 * in_stride1 + in_i2 * in_stride2 + in_i3 * in_stride3

    # Load and store
    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def _slice_impl(
    x: torch.Tensor,
    starts: List[int],
    ends: List[int],
    dims: Optional[List[int]] = None,
    steps: Optional[List[int]] = None
) -> torch.Tensor:
    """Slice tensor using pure Triton - NO PyTorch fallback."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()

    ndim = x.ndim
    if ndim > 4:
        raise NotImplementedError("Slice currently supports up to 4D tensors")

    # Default dims: all dimensions in order
    if dims is None:
        dims = list(range(len(starts)))

    # Default steps: all 1
    if steps is None:
        steps = [1] * len(starts)

    # Build full starts, ends, steps for all dimensions
    full_starts = [0] * ndim
    full_ends = list(x.shape)
    full_steps = [1] * ndim

    for i, dim in enumerate(dims):
        dim = dim if dim >= 0 else ndim + dim
        full_starts[dim] = starts[i]
        full_ends[dim] = ends[i]
        full_steps[dim] = steps[i]

    # Normalize negative indices and clamp
    for d in range(ndim):
        dim_size = x.shape[d]

        # Handle negative start/end
        if full_starts[d] < 0:
            full_starts[d] = max(0, dim_size + full_starts[d])
        if full_ends[d] < 0:
            full_ends[d] = max(0, dim_size + full_ends[d])

        # Clamp to valid range
        full_starts[d] = max(0, min(full_starts[d], dim_size))
        full_ends[d] = max(0, min(full_ends[d], dim_size))

        # Handle step direction
        if full_steps[d] < 0:
            # Negative step: swap and adjust
            full_starts[d], full_ends[d] = full_ends[d] + 1, full_starts[d] + 1
            full_steps[d] = -full_steps[d]

    # Compute output shape
    out_shape = []
    for d in range(ndim):
        size = (full_ends[d] - full_starts[d] + full_steps[d] - 1) // full_steps[d]
        size = max(0, size)
        out_shape.append(size)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()

    if n_elements == 0:
        return out

    # Pad to 4D
    shape_4d = list(x.shape) + [1] * (4 - ndim)
    out_shape_4d = out_shape + [1] * (4 - ndim)
    strides_4d = list(x.stride()) + [0] * (4 - ndim)
    starts_4d = full_starts + [0] * (4 - ndim)
    steps_4d = full_steps + [1] * (4 - ndim)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _strided_copy_kernel[grid](
        x, out, n_elements,
        strides_4d[0], strides_4d[1], strides_4d[2], strides_4d[3],
        out_shape_4d[0], out_shape_4d[1], out_shape_4d[2], out_shape_4d[3],
        starts_4d[0], starts_4d[1], starts_4d[2], starts_4d[3],
        steps_4d[0], steps_4d[1], steps_4d[2], steps_4d[3],
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )

    return out

def slice_kernel(
    x: torch.Tensor,
    starts: List[int],
    ends: List[int],
    dims: Optional[List[int]] = None,
    steps: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Slice tensor using pure Triton.

    Args:
        x: Input tensor
        starts: Start indices
        ends: End indices
        dims: Dimensions to slice - ATen API (not axes)
        steps: Step sizes

    Returns:
        Sliced tensor
    """
    return _slice_impl(x, starts, ends, dims, steps)
