"""
Universal Gather - Triton Optimized (Auto-CUDA + Multidim Support + Scalar Fix)
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel

# --- KERNEL 1 : EMBEDDING LOOKUP (Indices -> Vectors) ---
# Cas: Gather axis=0 sur une table [Vocab, Dim]
@triton.jit
def _gather_embedding_kernel(
    out_ptr, table_ptr, indices_ptr,
    stride_out_row, stride_table_row, stride_idx,
    dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0) # Index dans le batch d'indices plats
    
    # 1. Lire l'index scalaire
    idx = tl.load(indices_ptr + pid * stride_idx)
    
    # 2. Copier la ligne correspondante de la table
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < dim
    
    val = tl.load(table_ptr + idx * stride_table_row + cols, mask=mask, other=0.0)
    tl.store(out_ptr + pid * stride_out_row + cols, val, mask=mask)

# --- KERNEL 2 : GATHER STANDARD (Axis -1) ---
@triton.jit
def _gather_kernel(
    out_ptr, x_ptr, indices_ptr,
    stride_out_row, stride_x_row, stride_idx_row,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    idx_val = tl.load(indices_ptr + row_idx * stride_idx_row + cols, mask=mask, other=0)
    val = tl.load(x_ptr + row_idx * stride_x_row + idx_val, mask=mask, other=0.0)
    
    tl.store(out_ptr + row_idx * stride_out_row + cols, val, mask=mask)

@triton.jit
def _gather_nd_kernel(
    x_ptr, indices_ptr, out_ptr,
    n_indices, index_dim,
    stride_x, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    """General gather along specified dimension."""
    pid = tl.program_id(0)
    if pid >= n_indices:
        return

    # Load the index for this position
    idx = tl.load(indices_ptr + pid)

    # Copy the value
    for d_start in range(0, index_dim, BLOCK_SIZE):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE)
        mask = d_offs < index_dim

        val = tl.load(x_ptr + idx * stride_x + d_offs, mask=mask, other=0.0)
        tl.store(out_ptr + pid * stride_out + d_offs, val, mask=mask)


def _gather_impl(
    x: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Gather implementation using Triton."""
    assert x.is_cuda and indices.is_cuda
    x = x.contiguous()
    indices = indices.contiguous()

    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    # For dim=0 gather (embedding-style), use the embedding kernel pattern
    if dim == 0 and x.ndim == 2:
        indices_flat = indices.flatten()
        num_indices = indices_flat.numel()
        embedding_dim = x.shape[1]

        out_flat = torch.empty(num_indices, embedding_dim, dtype=x.dtype, device=x.device)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(embedding_dim))
        grid = (num_indices,)

        with torch.cuda.device(x.device):
            _gather_nd_kernel[grid](
                x, indices_flat, out_flat,
                num_indices, embedding_dim,
                x.stride(0), out_flat.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )

        out_shape = list(indices.shape) + [embedding_dim]
        return out_flat.view(out_shape)

    # For gather along last dim
    if dim == ndim - 1:
        # Flatten to 2D
        orig_shape = list(x.shape)
        n_cols = orig_shape[-1]
        n_rows = 1
        for s in orig_shape[:-1]:
            n_rows *= s

        x_flat = x.view(n_rows, n_cols)
        indices_flat = indices.view(n_rows, -1)
        out_n_cols = indices_flat.shape[-1]

        out_flat = torch.empty(n_rows, out_n_cols, dtype=x.dtype, device=x.device)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(out_n_cols))
        grid = (n_rows,)

        with torch.cuda.device(x.device):
            _gather_kernel[grid](
                out_flat, x_flat, indices_flat,
                out_flat.stride(0), x_flat.stride(0), indices_flat.stride(0),
                out_n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )

        out_shape = list(orig_shape[:-1]) + [out_n_cols]
        return out_flat.view(out_shape)

    # General case: move dim to position 0 and handle
    perm = [dim] + [i for i in range(ndim) if i != dim]
    x_t = x.permute(perm).contiguous()

    indices_shape = indices.shape
    indices_flat = indices.flatten()
    num_indices = indices_flat.numel()

    # After permute, shape is [dim_size, ...]
    inner_size = x_t[0].numel()

    out_flat = torch.empty(num_indices, inner_size, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(inner_size))
    grid = (num_indices,)

    with torch.cuda.device(x.device):
        _gather_nd_kernel[grid](
            x_t.view(x_t.shape[0], -1), indices_flat, out_flat,
            num_indices, inner_size,
            x_t.view(x_t.shape[0], -1).stride(0), out_flat.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    # Reshape and unpermute
    out_shape = list(indices_shape) + list(x_t.shape[1:])
    return out_flat.view(out_shape)


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="gather")
def gather_kernel(
    x: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Gather operation - pure Triton."""
    return _gather_impl(x, indices, dim)

