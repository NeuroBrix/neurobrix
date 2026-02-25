# Reduce Max - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _max_axis_kernel(in_ptr, out_ptr, outer_size, reduce_size, inner_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    outer_idx = pid // inner_size
    inner_idx = pid % inner_size
    if outer_idx >= outer_size: return

    acc = tl.full([1], value=-float('inf'), dtype=tl.float32)
    for r_start in range(0, reduce_size, BLOCK_SIZE):
        r_offsets = r_start + tl.arange(0, BLOCK_SIZE)
        r_mask = r_offsets < reduce_size
        in_offsets = outer_idx * (reduce_size * inner_size) + r_offsets * inner_size + inner_idx
        x = tl.load(in_ptr + in_offsets, mask=r_mask, other=-float('inf'))
        acc = tl.maximum(acc, tl.max(x, axis=0))

    out_offset = outer_idx * inner_size + inner_idx
    tl.store(out_ptr + out_offset + tl.arange(0, 1), acc)
