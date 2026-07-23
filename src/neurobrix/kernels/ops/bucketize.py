"""Bucketize (searchsorted over a small boundary vector) — pure
@triton.jit kernel.

aten::bucketize semantics: for each element v, the returned index i
counts the boundaries that sort strictly before v (right=False,
``boundaries[i-1] < v <= boundaries[i]``) or before-or-equal
(right=True, ``boundaries[i-1] <= v < boundaries[i]``).

Boundary vectors on this path are decode-control scale (block-diagonal
attention masks: one boundary per temporal frame / packed segment —
tens of entries), so a linear scan per element beats a binary search
and stays branch-free and bit-reproducible.
"""

import triton
import triton.language as tl


@triton.jit
def bucketize_kernel(
    x_ptr, boundaries_ptr, out_ptr,
    n_elements, n_boundaries,
    RIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offs < n_elements
    x = tl.load(x_ptr + offs, mask=in_range, other=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for i in range(0, n_boundaries):
        b = tl.load(boundaries_ptr + i)
        if RIGHT:
            acc += (b <= x).to(tl.int64)
        else:
            acc += (b < x).to(tl.int64)
    tl.store(out_ptr + offs, acc, mask=in_range)
