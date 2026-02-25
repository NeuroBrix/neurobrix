"""
Concat kernel - Pure Triton
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Copy elements from src to dst."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


def _concat_impl(tensors: list, dim: int = 0) -> torch.Tensor:
    """Concat implementation using Triton."""
    if len(tensors) == 0:
        raise ValueError("Need at least one tensor to concatenate")

    if len(tensors) == 1:
        return tensors[0]

    # Ensure all tensors are contiguous and on same device
    device = tensors[0].device
    dtype = tensors[0].dtype
    assert all(t.is_cuda for t in tensors)

    tensors = [t.contiguous() for t in tensors]

    # Normalize dim
    ndim = tensors[0].ndim
    if dim < 0:
        dim = ndim + dim

    # Calculate output shape
    out_shape = list(tensors[0].shape)
    concat_size = sum(t.shape[dim] for t in tensors)
    out_shape[dim] = concat_size

    output = torch.empty(out_shape, dtype=dtype, device=device)

    # For dim=0 concat, we can just copy contiguous blocks
    if dim == 0:
        offset = 0
        for t in tensors:
            n = t.numel()
            if n > 0:
                BLOCK = 1024
                grid = (triton.cdiv(n, BLOCK),)

                with torch.cuda.device(device):
                    _copy_kernel[grid](
                        t.view(-1), output.view(-1)[offset:],
                        n,
                        BLOCK_SIZE=BLOCK,
                        num_warps=4,
                    )
                offset += n
        return output

    # For other dims, move dim to position 0, concat, move back
    # Permute so concat dim is first
    perm = [dim] + [i for i in range(ndim) if i != dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    tensors_t = [t.permute(perm).contiguous() for t in tensors]

    # After permute, concat along dim 0
    out_shape_t = list(tensors_t[0].shape)
    out_shape_t[0] = sum(t.shape[0] for t in tensors_t)

    output_t = torch.empty(out_shape_t, dtype=dtype, device=device)

    offset = 0
    for t in tensors_t:
        n = t.numel()
        if n > 0:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)

            with torch.cuda.device(device):
                _copy_kernel[grid](
                    t.view(-1), output_t.view(-1)[offset:],
                    n,
                    BLOCK_SIZE=BLOCK,
                    num_warps=4,
                )
            offset += n

    # Permute back
    return output_t.permute(inv_perm).contiguous()


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="concat")
def concat_kernel(tensors: list, dim: int = 0) -> torch.Tensor:
    """Concatenate tensors - pure Triton."""
    if isinstance(tensors, torch.Tensor):
        return tensors
    return _concat_impl(tensors, dim)

