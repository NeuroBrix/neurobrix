"""Embedding lookup — pure @triton.jit kernel. Ported from FlagGems (Apache 2.0)."""

import triton
import triton.language as tl

#: The contract this kernel enforces. Triton forbids reading a module global
#: from inside a @jit body, so the kernel repeats the literal; the permanent
#: test pins the two equal so they cannot drift.
EMBEDDING_OOB = "embedding: index out of range"


@triton.jit(debug=True)
def embedding_kernel(
    output_ptr, indices_ptr, weight_ptr,
    V, fault_ptr,
    FAULT_CODE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding forward: output[i] = weight[indices[i]].

    indices: [M] (flattened)
    weight: [V = num_embeddings, N]
    output: [M, N]

    An id outside [0, V) is a broken contract where torch's embedding
    raises, and the port read the weight row at that address, silently
    (D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).

    It is refused through TWO channels because one of them is not
    universal:

      `tl.device_assert`, kept in the binary by `debug=True`, halts the
      thread on CUDA and ROCm and the next sync raises.

      `FAULT_CODE`, a compile-time constant, is the same refusal for a
      backend that does NOT honour that assert. The Metal backend
      computes the assert predicate and discards it (measured
      2026-09-10), so this kernel read past the end of the weight there
      — 4 floats past a 24-float tensor in the eight-float reproducer.
      A non-zero code makes the kernel store it into the fault word,
      which `check_device_faults` raises on at the next host
      observation. The wrapper passes 0 where the assert is honoured,
      and then not one instruction of this is emitted.

    The address is also made LEGAL before it is used. That is not a
    fallback and it does not rescue the run — the run refuses at the
    next observation either way. It is there because the refusal is
    asynchronous on this path, and a kernel must not corrupt memory in
    the interval before it arrives. On CUDA the assert halts the thread
    first, so this changes nothing there.
    """
    pid = tl.program_id(0)
    output_ptr += pid * N
    indices_ptr += pid

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_idx = tl.load(indices_ptr)
    in_range = (row_idx >= 0) & (row_idx < V)
    tl.device_assert(in_range, "embedding: index out of range")
    if FAULT_CODE != 0:
        if not in_range:
            tl.store(fault_ptr, FAULT_CODE)

    weight_ptr += tl.where(in_range, row_idx, 0) * N
    embedding_weight = tl.load(weight_ptr + cols, mask & in_range, other=0.0)
    tl.store(output_ptr + cols, embedding_weight, mask)
