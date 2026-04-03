"""NanToNum — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import triton.language as tl

@triton.jit
def nan_to_num_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    nan_val, posinf_val, neginf_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    pos_inf = float('inf')
    neg_inf = float('-inf')
    out = tl.where(x != x, nan_val,
          tl.where(x == pos_inf, posinf_val,
          tl.where(x == neg_inf, neginf_val, x)))
    tl.store(output_ptr + offset, out, mask=mask)
