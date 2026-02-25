# Floor - Pure Triton
import torch, triton, triton.language as tl

@triton.jit
def _floor_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE); mask = offs < n_elements
    tl.store(out_ptr + offs, tl.math.floor(tl.load(x_ptr + offs, mask=mask)), mask=mask)
