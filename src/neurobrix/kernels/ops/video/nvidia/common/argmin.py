# ArgMin - Argmin reduction along axis
# Type: Pure Triton Kernel (Reduction)
# NeuroBrix - NVIDIA Common (All architectures)
# ATen API: dim, keepdim (NO ONNX - axis/keepdims removed)

import torch
import triton
import triton.language as tl
from typing import Optional, List, Union

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _argmin_last_dim_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr
):
    """Argmin along last dimension - pure Triton."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Find min value and index
    min_val = float('inf')
    min_idx = 0

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x = tl.load(x_ptr + row_idx * stride_row + col_offsets, mask=mask, other=float('inf'))

        # Block min
        block_min = tl.min(x, axis=0)
        if block_min < min_val:
            idx_in_block = tl.argmin(x, axis=0)
            min_val = block_min
            min_idx = col_start + idx_in_block

    tl.store(out_ptr + row_idx, min_idx)


def _argmin_impl(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    """Argmin implementation using Triton."""
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

    # Flatten all but last dim
    orig_shape = list(x.shape)
    n_cols = orig_shape[-1]
    n_rows = 1
    for s in orig_shape[:-1]:
        n_rows *= s

    x_flat = x.view(n_rows, n_cols)

    # Output
    out = torch.empty(n_rows, dtype=torch.int64, device=x.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    grid = (n_rows,)

    with torch.cuda.device(x.device):
        _argmin_last_dim_kernel[grid](
            x_flat, out,
            n_rows, n_cols,
            x_flat.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    # Reshape output
    out_shape = list(orig_shape[:-1])
    if keepdim:
        out_shape.append(1)

    if not out_shape:
        out = out.squeeze()
    else:
        out = out.view(out_shape)

    return out


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="argmin")
def argmin_kernel(x: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """Argmin reduction - pure Triton."""
    if dim is None:
        # Global argmin - flatten and reduce
        x_flat = x.flatten()
        return _argmin_impl(x_flat.unsqueeze(0), dim=1, keepdim=False).squeeze()
    return _argmin_impl(x, dim, keepdim)
