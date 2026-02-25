# Reduce Sum - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _sum_1d_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, block_sum)

@triton.jit
def _sum_axis_kernel(in_ptr, out_ptr, outer_size, reduce_size, inner_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    outer_idx = pid // inner_size
    inner_idx = pid % inner_size
    if outer_idx >= outer_size: return

    acc = tl.zeros([1], dtype=tl.float32)
    for r_start in range(0, reduce_size, BLOCK_SIZE):
        r_offsets = r_start + tl.arange(0, BLOCK_SIZE)
        r_mask = r_offsets < reduce_size
        in_offsets = outer_idx * (reduce_size * inner_size) + r_offsets * inner_size + inner_idx
        x = tl.load(in_ptr + in_offsets, mask=r_mask, other=0.0)
        acc += tl.sum(x, axis=0)

    out_offset = outer_idx * inner_size + inner_idx
    tl.store(out_ptr + out_offset + tl.arange(0, 1), acc)
