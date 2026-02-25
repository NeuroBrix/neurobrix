# Softmax - Softmax along axis
# Type: Pure Triton Kernel
# NeuroBrix - NVIDIA Common (All architectures)
# ATen API: dim (NO ONNX - axis removed)

import torch
import triton
import triton.language as tl
from typing import Optional

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """Softmax along last axis - pure Triton."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load with boundary check
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Softmax logic: subtraction for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def _softmax_impl(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax implementation using Triton."""
    assert x.is_cuda
    x = x.contiguous()

    ndim = x.ndim

    # Normalize dim
    if dim < 0:
        dim = ndim + dim

    # Move target dim to last position
    if dim != ndim - 1:
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm).contiguous()
        need_unpermute = True
    else:
        need_unpermute = False

    # Flatten all but last dim
    orig_shape = list(x.shape)
    n_cols = orig_shape[-1]
    n_rows = 1
    for s in orig_shape[:-1]:
        n_rows *= s

    x_flat = x.view(n_rows, n_cols)
    out_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Cap block size for large inputs
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192

    grid = (n_rows,)

    with torch.cuda.device(x.device):
        _softmax_kernel[grid](
            out_flat, x_flat,
            x_flat.stride(0), out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    out = out_flat.view(orig_shape)

    if need_unpermute:
        # Reverse permutation
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        out = out.permute(perm)

    return out.contiguous()


@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="softmax")
def softmax_kernel(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax activation - pure Triton."""
    return _softmax_impl(x, dim)
