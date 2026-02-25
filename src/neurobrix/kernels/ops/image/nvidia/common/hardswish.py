# HardSwish - Pure Triton
# Source: Attorch
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _hardswish_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # relu6(x + 3) / 6
    x_plus_3 = x + 3
    relu6 = tl.maximum(0, tl.minimum(x_plus_3, 6))
    output = x * relu6 / 6
    
    tl.store(output_ptr + offsets, output, mask=mask)
