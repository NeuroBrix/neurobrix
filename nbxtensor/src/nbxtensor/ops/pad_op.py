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
    WIDE: tl.constexpr = True,
):
    """Constant pad for 4D tensors (N,C,H,W). Most common case."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
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
    WIDE: tl.constexpr = True,
):
    """Constant pad for 2D/3D tensors along last dim."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
    mask = offsets < total

    out_idx = offsets % out_size
    batch_idx = offsets // out_size

    in_idx = out_idx - pad_left
    in_bounds = (in_idx >= 0) & (in_idx < in_size) & (batch_idx < batch_size)

    inp_offset = batch_idx * in_size + in_idx
    val = tl.load(x_ptr + inp_offset, mask=mask & in_bounds, other=pad_value)

    out_offset = batch_idx * out_size + out_idx
    tl.store(output_ptr + out_offset, val, mask=mask)


@triton.jit
def constant_pad_nd_kernel(
    x_ptr, output_ptr,
    total,
    in0, in1, in2, in3, in4,
    out0, out1, out2, out3, out4,
    xs0, xs1, xs2, xs3, xs4,
    pb0, pb1, pb2, pb3, pb4,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Constant pad of up to 5 dims in ONE pass: each output element maps to its input element (any dim padded,
    a middle one included) or to `pad_value`. Dims a tensor lacks are given extent 1 and pad 0 by the wrapper, so one
    kernel serves 1-D to 5-D. The input is read through its strides (xs*); the output is written contiguous."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
    mask = offsets < total
    o4 = offsets % out4
    t = offsets // out4
    o3 = t % out3
    t = t // out3
    o2 = t % out2
    t = t // out2
    o1 = t % out1
    o0 = t // out1
    i0 = o0 - pb0
    i1 = o1 - pb1
    i2 = o2 - pb2
    i3 = o3 - pb3
    i4 = o4 - pb4
    inside = ((i0 >= 0) & (i0 < in0) & (i1 >= 0) & (i1 < in1) & (i2 >= 0) & (i2 < in2)
              & (i3 >= 0) & (i3 < in3) & (i4 >= 0) & (i4 < in4))
    src = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3 + i4 * xs4
    val = tl.load(x_ptr + src, mask=mask & inside, other=pad_value)
    tl.store(output_ptr + offsets, val, mask=mask)


@triton.jit
def replication_pad_nd_kernel(
    x_ptr, output_ptr,
    total,
    in0, in1, in2, in3, in4,
    out0, out1, out2, out3, out4,
    xs0, xs1, xs2, xs3, xs4,
    pb0, pb1, pb2, pb3, pb4,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Replication pad of up to 5 dims in ONE pass (PyTorch `replicate`): each output element reads the input element
    at its position clamped into the input, so a padded border repeats the edge. The twin of `constant_pad_nd_kernel`,
    the same index map with a clamp where that one masks; the input is read through its strides, the output written
    contiguous."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`)
    mask = offsets < total
    o4 = offsets % out4
    t = offsets // out4
    o3 = t % out3
    t = t // out3
    o2 = t % out2
    t = t // out2
    o1 = t % out1
    o0 = t // out1
    i0 = tl.minimum(tl.maximum(o0 - pb0, 0), in0 - 1)
    i1 = tl.minimum(tl.maximum(o1 - pb1, 0), in1 - 1)
    i2 = tl.minimum(tl.maximum(o2 - pb2, 0), in2 - 1)
    i3 = tl.minimum(tl.maximum(o3 - pb3, 0), in3 - 1)
    i4 = tl.minimum(tl.maximum(o4 - pb4, 0), in4 - 1)
    src = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3 + i4 * xs4
    val = tl.load(x_ptr + src, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)
