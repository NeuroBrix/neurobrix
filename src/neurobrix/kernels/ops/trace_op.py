"""Trace — pure @triton.jit kernel.

Sum of diagonal elements of a 2D matrix.
Single program iterates over the diagonal in blocks, accumulating in
float32 (or int64 for integer types) for numerical stability.

Extracted from FlagGems (Apache 2.0).

Grid: (1,)
"""

import triton
import triton.language as tl


@triton.jit
def trace_kernel(
    inp_ptr,
    out_ptr,
    num_diag,
    stride0,
    stride1,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum diagonal elements of a 2D matrix.

    Iterates over the diagonal in blocks of BLOCK_SIZE, accumulating
    partial sums. The diagonal stride is stride0 + stride1 (step to
    move from (i,i) to (i+1,i+1)).

    Args:
        inp_ptr: input 2D matrix pointer
        out_ptr: scalar output pointer
        num_diag: min(M, N) — number of diagonal elements
        stride0: row stride of input
        stride1: column stride of input
    """
    # Select accumulation dtype based on input type
    inp_dtype = inp_ptr.type.element_ty
    if inp_dtype.is_int():
        acc_dtype = tl.int64
        other_val = 0
    elif inp_dtype == tl.float64:
        acc_dtype = tl.float64
        other_val = 0.0
    else:
        acc_dtype = tl.float32
        other_val = 0.0

    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)

    diag_stride = stride0 + stride1

    for i in range(0, tl.cdiv(num_diag, BLOCK_SIZE)):
        block_start = i * BLOCK_SIZE
        current_indices = block_start + tl.arange(0, BLOCK_SIZE)
        mask = current_indices < num_diag

        ptr_offsets = current_indices * diag_stride
        vals = tl.load(inp_ptr + ptr_offsets, mask=mask, other=other_val)
        acc += vals.to(acc_dtype)

    final_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr, final_sum.to(out_ptr.type.element_ty))
