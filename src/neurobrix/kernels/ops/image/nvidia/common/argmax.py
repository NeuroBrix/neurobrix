# ArgMax - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _argmax_last_dim_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows: return
    max_val, max_idx = -float('inf'), 0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_idx * stride_row + offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        if block_max > max_val:
            idx_in_block = tl.argmax(x, axis=0)
            max_val, max_idx = block_max, col_start + idx_in_block
    tl.store(out_ptr + row_idx, max_idx)
