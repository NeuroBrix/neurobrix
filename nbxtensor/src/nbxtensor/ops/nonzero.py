"""1-D boolean-mask compaction (nonzero) — pure @triton.jit kernel.

Deterministic by construction: a SINGLE program scans the mask in
BLOCK-sized chunks with an in-block tl.cumsum and a running cursor, so
the compacted indices are emitted in ascending order without atomics.
Mask sizes on this path are decode-control scale (visual/attention
masks, thousands of elements) — the serial-block scan costs
microseconds there and keeps the op bit-reproducible across runs.
"""

import triton
import triton.language as tl


@triton.jit
def nonzero_1d_kernel(
    mask_ptr, out_ptr, count_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    cursor = tl.zeros((1,), dtype=tl.int32)
    for start in range(0, n_elements, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        in_range = offs < n_elements
        m = tl.load(mask_ptr + offs, mask=in_range, other=0)
        pred = (m != 0) & in_range
        p32 = pred.to(tl.int32)
        excl = tl.cumsum(p32, axis=0) - p32
        base = tl.sum(cursor, axis=0)
        tl.store(out_ptr + (base + excl).to(tl.int64), offs.to(tl.int64),
                 mask=pred)
        cursor += tl.sum(p32, axis=0)
    tl.store(count_ptr, tl.sum(cursor, axis=0))
