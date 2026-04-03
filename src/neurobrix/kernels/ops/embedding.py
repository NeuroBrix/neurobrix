"""Embedding lookup — pure @triton.jit kernel. Ported from FlagGems (Apache 2.0)."""

import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    output_ptr, indices_ptr, weight_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding forward: output[i] = weight[indices[i]].

    indices: [M] (flattened)
    weight: [num_embeddings, N]
    output: [M, N]
    """
    pid = tl.program_id(0)
    output_ptr += pid * N
    indices_ptr += pid

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_idx = tl.load(indices_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(output_ptr + cols, embedding_weight, mask)
