# ELU - Pure Triton
# Source: Attorch
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _elu_kernel(input_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))
    tl.store(output_ptr + offsets, output, mask=mask)
