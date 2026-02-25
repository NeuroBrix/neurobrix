# CELU - Pure Triton
# Source: Attorch
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _celu_kernel(input_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # relu(x) + min(0, alpha * (exp(x / alpha) - 1))
    relu = tl.maximum(0, x)
    exp_term = alpha * (tl.exp(x / alpha) - 1)
    output = relu + tl.minimum(0, exp_term)
    
    tl.store(output_ptr + offsets, output, mask=mask)
