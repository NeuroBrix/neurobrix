# Fused Operations - Pure Triton
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mul_add_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a * b + c, mask=mask)

@triton.jit
def _fused_add_gelu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask)
    # Fast Gelu approx
    res = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offs, res, mask=mask)

@triton.jit
def _fused_add_silu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x), mask=mask)

# Registry for discovery

# Others (Linear variants) are handled by Adapter decomposition
