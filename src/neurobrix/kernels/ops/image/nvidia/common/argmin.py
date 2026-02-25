# ArgMin - Pure Triton
# Source: FlagGems
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _argmin_last_dim_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows: return
    
    # Simple stable implementation for last dim
    max_val = float('inf')
    min_idx = 0
    
    for start_n in range(0, n_cols, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        ptr = x_ptr + pid * stride_row + offsets
        val = tl.load(ptr, mask=mask, other=float('inf'))
        
        local_min, local_idx = tl.min(val, axis=0, return_indices=True)
        if local_min < max_val:
            max_val = local_min
            min_idx = start_n + local_idx
            
    tl.store(out_ptr + pid, min_idx)
