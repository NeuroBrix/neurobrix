"""
Embedding - Lookup table
ATen: aten::embedding
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _embedding_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    num_indices,
    embedding_dim,
    stride_w0,
    stride_w1,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup kernel."""
    pid = tl.program_id(0)

    if pid >= num_indices:
        return

    idx = tl.load(indices_ptr + pid)

    for d_start in range(0, embedding_dim, BLOCK_SIZE):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE)
        mask = d_offs < embedding_dim

        w_ptr = weight_ptr + idx * stride_w0 + d_offs * stride_w1
        emb = tl.load(w_ptr, mask=mask, other=0.0)

        o_ptr = out_ptr + pid * embedding_dim + d_offs
        tl.store(o_ptr, emb, mask=mask)


def _embedding_impl(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Embedding lookup using Triton."""
    assert weight.is_cuda

    orig_shape = indices.shape
    indices_flat = indices.flatten().contiguous()

    num_indices = indices_flat.numel()
    vocab_size, embedding_dim = weight.shape

    weight = weight.contiguous()

    out_flat = torch.empty(num_indices, embedding_dim, dtype=weight.dtype, device=weight.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(embedding_dim))
    grid = (num_indices,)

    with torch.cuda.device(weight.device):
        _embedding_kernel[grid](
            indices_flat, weight, out_flat,
            num_indices, embedding_dim,
            weight.stride(0), weight.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    out_shape = list(orig_shape) + [embedding_dim]
    return out_flat.view(out_shape)

@register_kernel(family="audio", vendor="nvidia", tier="common", op_name="embedding")
def embedding_kernel(weight: torch.Tensor, indices: torch.Tensor, padding_idx: int = None) -> torch.Tensor:
    """Embedding lookup - pure Triton."""
    return _embedding_impl(indices, weight)

