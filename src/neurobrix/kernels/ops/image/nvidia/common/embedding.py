# Embedding - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(
    indices_ptr, weight_ptr, out_ptr,
    num_indices, embedding_dim,
    stride_w0, stride_w1,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup kernel."""
    pid = tl.program_id(0)
    if pid >= num_indices: return
    idx = tl.load(indices_ptr + pid)
    for d_start in range(0, embedding_dim, BLOCK_SIZE):
        d_offs = d_start + tl.arange(0, BLOCK_SIZE)
        mask = d_offs < embedding_dim
        w_ptr = weight_ptr + idx * stride_w0 + d_offs * stride_w1
        emb = tl.load(w_ptr, mask=mask, other=0.0)
        tl.store(out_ptr + pid * embedding_dim + d_offs, emb, mask=mask)
