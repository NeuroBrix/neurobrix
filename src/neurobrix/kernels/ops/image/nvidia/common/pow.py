"""
Universal Pow - Pure Triton (fp16 compatible)
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _pow_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # libdevice.pow only supports fp32/fp64, cast if needed
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    output_fp32 = libdevice.pow(x_fp32, y_fp32)
    # Clamp to fp16 range before casting back to prevent inf
    output_fp32 = tl.minimum(tl.maximum(output_fp32, -65000.0), 65000.0)
    output = output_fp32.to(x.dtype)

    tl.store(out_ptr + offsets, output, mask=mask)
