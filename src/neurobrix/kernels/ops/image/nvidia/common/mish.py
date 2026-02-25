# Mish - Pure Triton
# Source: Attorch
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _mish_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # x * tanh(softplus(x))
    softplus = tl.log(1 + tl.exp(x))
    # tanh approximation
    tanh_res = tl.math.tanh(softplus)
    output = x * tanh_res
    
    tl.store(output_ptr + offsets, output, mask=mask)
