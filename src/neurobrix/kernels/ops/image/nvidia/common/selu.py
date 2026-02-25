# SELU - Pure Triton
# Source: Attorch
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _selu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    
    # scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    res = tl.maximum(0, x) + tl.minimum(0, alpha * (tl.exp(x) - 1))
    output = scale * res
    
    tl.store(output_ptr + offsets, output, mask=mask)
