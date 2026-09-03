"""Embedding lookup — pure @triton.jit kernel. Ported from FlagGems (Apache 2.0)."""

import triton
import triton.language as tl


@triton.jit(debug=True)
def embedding_kernel(
    output_ptr, indices_ptr, weight_ptr,
    V,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding forward: output[i] = weight[indices[i]].

    indices: [M] (flattened)
    weight: [V = num_embeddings, N]
    output: [M, N]

    An id outside [0, V) TRAPS via `tl.device_assert` (kept in the
    binary by `debug=True`) where torch's embedding raises — the port
    read the weight row at that address, silently
    (D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).
    """
    pid = tl.program_id(0)
    output_ptr += pid * N
    indices_ptr += pid

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_idx = tl.load(indices_ptr)
    tl.device_assert((row_idx >= 0) & (row_idx < V), "embedding: index out of range")
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(output_ptr + cols, embedding_weight, mask)
