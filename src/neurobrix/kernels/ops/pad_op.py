"""Constant padding — pure @triton.jit kernel.

Handles constant_pad_nd for the common 4D case (N,C,H,W) and generic N-D
via a flat-index approach. For each output element, compute the corresponding
input coordinate. If in bounds, load from input; otherwise, store pad_value.

Logic derived from FlagGems pad.py codegen approach.
"""

import triton
import triton.language as tl

@triton.jit
def constant_pad_2d_kernel(
    x_ptr, output_ptr,
    total,
    in_h, in_w, out_h, out_w,
    channels_times_batch,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    pad_left, pad_top,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
):
    """Constant pad for 4D tensors (N,C,H,W). Most common case."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Decompose flat index into (n*c, h, w)
    ow = offsets % out_w
    temp = offsets // out_w
    oh = temp % out_h
    nc = temp // out_h

    # Map output coords to input coords
    iw = ow - pad_left
    ih = oh - pad_top

    # Check if within input bounds
    in_bounds = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & (nc < channels_times_batch)

    # Flat stride approach: assumes contiguous input (wrapper ensures contiguous)
    inp_offset = nc * in_h * in_w + ih * in_w + iw

    val = tl.load(x_ptr + inp_offset, mask=mask & in_bounds, other=pad_value)
    out_offset = nc * out_h * out_w + oh * out_w + ow
    tl.store(output_ptr + out_offset, val, mask=mask)

@triton.jit
def constant_pad_1d_kernel(
    x_ptr, output_ptr,
    total,
    in_size, out_size,
    batch_size,
    pad_left,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
):
    """Constant pad for 2D/3D tensors along last dim."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    out_idx = offsets % out_size
    batch_idx = offsets // out_size

    in_idx = out_idx - pad_left
    in_bounds = (in_idx >= 0) & (in_idx < in_size) & (batch_idx < batch_size)

    inp_offset = batch_idx * in_size + in_idx
    val = tl.load(x_ptr + inp_offset, mask=mask & in_bounds, other=pad_value)

    out_offset = batch_idx * out_size + out_idx
    tl.store(output_ptr + out_offset, val, mask=mask)
