"""index_put — pure @triton.jit kernel.

Scatter-write `values` into `out` at positions given by a single 1-D
advanced index tensor on the leading dim (the form decomposed-ATen
emits for MoE-v2 expert-output aggregation, KV-cache indexed writes,
and post-`aten::nonzero` masked scatter):

    out[idx[s], t] = values[s, t]              (ACCUMULATE == 0)
    out[idx[s], t] += values[s, t]             (ACCUMULATE == 1)

`idx` may have any shape S (flattened, numel = Sn); `t` ranges over
the flattened tail `x.shape[1:]` (numel = T).  `out` is contiguous so
the leading-dim stride is exactly T.  Duplicate indices follow torch
semantics: summed when accumulating, last-writer-wins (unspecified
order) otherwise.

Index decomposition logic mirrors `index_add_op.py` (FlagGems
lineage, Apache-2.0).
"""

import triton
import triton.language as tl


@triton.jit
def index_put_kernel(
    out_ptr, idx_ptr, val_ptr,
    T, N,
    VAL_SCALAR: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out[idx[off // T], off % T] {=|+=} values[off]  for off < N."""
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    s = off // T                       # which index entry (row in idx)
    t = off % T                        # offset within the flattened tail
    row = tl.load(idx_ptr + s, mask=mask, other=0).to(tl.int64)
    dst = row * T + t

    if VAL_SCALAR:
        # All lanes read element 0 (off * 0 keeps it a vector for masking).
        v = tl.load(val_ptr + off * 0, mask=mask, other=0.0)
    else:
        v = tl.load(val_ptr + off, mask=mask, other=0.0)

    if ACCUMULATE:
        tl.atomic_add(out_ptr + dst, v, mask=mask)
    else:
        tl.store(out_ptr + dst, v, mask=mask)
