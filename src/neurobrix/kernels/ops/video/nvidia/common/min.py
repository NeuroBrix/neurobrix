"""
Min reduction kernel - Pure Triton
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _min_reduce_kernel(
    x_ptr, out_ptr, idx_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr
):
    """Min reduction along last dimension - pure Triton."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    min_val = float('inf')
    min_idx = 0

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x = tl.load(x_ptr + row_idx * stride_row + col_offsets, mask=mask, other=float('inf'))

        block_min = tl.min(x, axis=0)
        if block_min < min_val:
            idx_in_block = tl.argmin(x, axis=0)
            min_val = block_min
            min_idx = col_start + idx_in_block

    tl.store(out_ptr + row_idx, min_val)
    tl.store(idx_ptr + row_idx, min_idx)


@triton.jit
def _min_global_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Global min reduction - pure Triton."""
    min_val = float('inf')

    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
        block_min = tl.min(x, axis=0)
        if block_min < min_val:
            min_val = block_min

    tl.store(out_ptr, min_val)


def _min_impl(x: torch.Tensor, dim: int, keepdim: bool):
    """Min implementation using Triton."""
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

    # Output tensors
    out_vals = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    out_idxs = torch.empty(n_rows, dtype=torch.int64, device=x.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    grid = (n_rows,)

    with torch.cuda.device(x.device):
        _min_reduce_kernel[grid](
            x_flat, out_vals, out_idxs,
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
        out_vals = out_vals.squeeze()
        out_idxs = out_idxs.squeeze()
    else:
        out_vals = out_vals.view(out_shape)
        out_idxs = out_idxs.view(out_shape)

    return out_vals, out_idxs


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="min")
def min_kernel(x: torch.Tensor, dim: int = None, keepdim: bool = False):
    """Min reduction - pure Triton."""
    if dim is None:
        # Global min
        x_flat = x.flatten().contiguous()
        out = torch.empty(1, dtype=x.dtype, device=x.device)

        n = x_flat.numel()
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n))

        with torch.cuda.device(x.device):
            _min_global_kernel[(1,)](
                x_flat, out,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
        return out.squeeze()

    return _min_impl(x, dim, keepdim)

