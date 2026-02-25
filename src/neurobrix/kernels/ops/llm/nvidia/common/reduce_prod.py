# Reduce Prod - Product reduction along axis
# Type: Pure Triton Kernel (Reduction)
# NeuroBrix - NVIDIA Common (All architectures)
# ATen API: dim, keepdim (NO ONNX - axes/keepdims removed)

import torch
import triton
import triton.language as tl
from typing import List, Optional, Union

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _prod_combine(a, b):
    return a * b


@triton.jit
def _reduce_prod_1d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce entire tensor to product - one block writes one partial."""
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=1.0)
    block_prod = tl.reduce(x, axis=0, combine_fn=_prod_combine)

    tl.store(out_ptr + pid, block_prod)


@triton.jit
def _reduce_prod_axis_kernel(
    in_ptr,
    out_ptr,
    outer_size,
    reduce_size,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Product reduction along axis - pure Triton."""
    pid = tl.program_id(0)

    outer_idx = pid // inner_size
    inner_idx = pid % inner_size

    if outer_idx >= outer_size:
        return

    prod_val = tl.full([1], 1.0, dtype=tl.float32)

    for r_start in range(0, reduce_size, BLOCK_SIZE):
        r_offsets = r_start + tl.arange(0, BLOCK_SIZE)
        r_mask = r_offsets < reduce_size

        in_offsets = outer_idx * (reduce_size * inner_size) + r_offsets * inner_size + inner_idx
        x = tl.load(in_ptr + in_offsets, mask=r_mask, other=1.0)

        block_prod = tl.reduce(x, axis=0, combine_fn=_prod_combine)
        prod_val = prod_val * block_prod

    out_offset = outer_idx * inner_size + inner_idx
    tl.store(out_ptr + out_offset + tl.arange(0, 1), prod_val)


def _compute_reduce_dims(shape: List[int], dim: List[int], keepdim: bool):
    """Compute dimensions for reduction."""
    ndim = len(shape)
    dim = [(d if d >= 0 else ndim + d) for d in dim]
    dim = sorted(set(dim))

    outer_size = 1
    reduce_size = 1
    inner_size = 1

    for i, s in enumerate(shape):
        if i < dim[0]:
            outer_size *= s
        elif i <= dim[-1]:
            reduce_size *= s
        else:
            inner_size *= s

    if keepdim:
        out_shape = list(shape)
        for d in dim:
            out_shape[d] = 1
    else:
        out_shape = [s for i, s in enumerate(shape) if i not in dim]
        if not out_shape:
            out_shape = [1]

    return outer_size, reduce_size, inner_size, out_shape


def _reduce_prod_impl(
    x: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """Product reduction using pure Triton - NO PyTorch fallback."""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()

    # Full reduction - dim=None
    if dim is None:
        n_elements = x.numel()

        BLOCK_SIZE = 1024
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

        partial = torch.ones(n_blocks, dtype=x.dtype, device=x.device)

        grid = (n_blocks,)
        _reduce_prod_1d_kernel[grid](
            x, partial, n_elements,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Final product of partial results via Triton recursion
        while partial.numel() > 1:
            n_elements = partial.numel()
            n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
            new_partial = torch.ones(n_blocks, dtype=partial.dtype, device=partial.device)
            _reduce_prod_1d_kernel[(n_blocks,)](
                partial, new_partial, n_elements,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=4
            )
            partial = new_partial

        out = partial.squeeze()

        if keepdim:
            out = out.view([1] * x.ndim)
        return out

    # Axis reduction
    if isinstance(dim, int):
        dim = [dim]

    if len(dim) > 1:
        dim = sorted(dim, reverse=True)
        res = x
        for d in dim:
            res = _reduce_prod_impl(res, dim=[d], keepdim=keepdim)
        return res

    shape = list(x.shape)
    outer_size, reduce_size, inner_size, out_shape = _compute_reduce_dims(shape, dim, keepdim)

    out = torch.ones(out_shape, dtype=x.dtype, device=x.device)

    n_outputs = outer_size * inner_size
    BLOCK_SIZE = 1024
    if reduce_size < 1024:
        BLOCK_SIZE = triton.next_power_of_2(reduce_size)

    grid = (n_outputs,)

    _reduce_prod_axis_kernel[grid](
        x, out,
        outer_size, reduce_size, inner_size,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )

    return out

@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="reduce_prod")
def reduce_prod_kernel(
    x: torch.Tensor,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Product reduction using pure Triton.

    Args:
        x: Input tensor
        dim: Dimension(s) to reduce (None = all) - ATen API
        keepdim: Keep reduced dimensions as size 1 - ATen API

    Returns:
        Reduced tensor
    """
    if isinstance(dim, int):
        dim = [dim]
    return _reduce_prod_impl(x, dim, keepdim)
