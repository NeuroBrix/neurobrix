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

#: The contract this kernel enforces. Triton forbids reading a module global
#: from inside a @jit body, so the kernel repeats the literal; the permanent
#: test pins the two equal so they cannot drift.
INDEX_PUT_OOB = "index_put: index out of range"


@triton.jit(debug=True)
def index_put_kernel(
    out_ptr, idx_ptr, val_ptr,
    T, N, R, fault_ptr,
    FAULT_CODE: tl.constexpr,
    VAL_SCALAR: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """out[idx[off // T], off % T] {=|+=} values[off]  for off < N.

    `R` = rows of `out` (its leading dim). A negative index counts from
    the end (torch semantics); an index still out of [0, R) after that
    TRAPS via `tl.device_assert` (kept in the binary by `debug=True`)
    where torch's index_put raises — the FlagGems-lineage kernel wrote
    through it to memory outside the tensor, silently
    (D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).

    That assert is honoured on CUDA and ROCm and NOT on Metal, which
    computes its predicate and discards it (measured 2026-09-10). Here
    that was not a wrong value, it was the WRITE ITSELF: the store's mask
    was `off < N` alone, so `dst = row * T + t` with an out-of-range row
    wrote outside the tensor — 8 floats past the end of a 24-float
    tensor in the eight-float reproducer. Two things close it. The
    address is made legal before it is used, so nothing outside the
    tensor is touched in the interval before the refusal arrives (on
    CUDA the assert halts the thread first, so it changes nothing
    there); and `FAULT_CODE`, a compile-time constant, carries the
    refusal itself through the fault word, which `check_device_faults`
    raises on at the next host observation. The wrapper passes 0 where
    the assert is honoured, and then not one instruction of it is
    emitted.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    s = off // T                       # which index entry (row in idx)
    t = off % T                        # offset within the flattened tail
    row = tl.load(idx_ptr + s, mask=mask, other=0).to(tl.int64)
    row = tl.where(row < 0, row + R, row)
    # Masked lanes loaded `other=0` and are in range by construction.
    row_valid = (row >= 0) & (row < R)
    tl.device_assert(row_valid, "index_put: index out of range")
    if FAULT_CODE != 0:
        # Lanes past N loaded `other=0`; they must not report (and `R == 0`
        # would make even row 0 invalid, so `mask` has to gate this).
        if tl.max(tl.where(mask & (row_valid == 0), 1, 0)) != 0:
            tl.store(fault_ptr, FAULT_CODE)
    # A legal address, so no memory outside the tensor is touched before the
    # refusal is observed. Not a rescue: the run refuses either way.
    dst = tl.where(row_valid, row, 0) * T + t

    if VAL_SCALAR:
        # All lanes read element 0 (off * 0 keeps it a vector for masking).
        v = tl.load(val_ptr + off * 0, mask=mask, other=0.0)
    else:
        v = tl.load(val_ptr + off, mask=mask, other=0.0)

    if ACCUMULATE:
        tl.atomic_add(out_ptr + dst, v, mask=mask & row_valid)
    else:
        tl.store(out_ptr + dst, v, mask=mask & row_valid)
