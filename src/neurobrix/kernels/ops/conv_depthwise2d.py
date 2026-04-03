"""Depthwise 2D convolution — pure @triton.jit kernel.

Depthwise convolution: each input channel is convolved independently with
its own filter (groups = C_in). Output channels = C_in * channel_multiplier
(typically channel_multiplier = 1).

Input:  [N, C, IH, IW]
Weight: [C_out, 1, KH, KW]  (groups = C_in)
Output: [N, C_out, OH, OW]

OH = (IH + 2*pad_h - dilation_h*(KH-1) - 1) // stride_h + 1
OW = (IW + 2*pad_w - dilation_w*(KW-1) - 1) // stride_w + 1
"""

import triton
import triton.language as tl


@triton.jit
def conv_depthwise2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    # Input shape
    N, C_in, IH, IW,
    # Weight shape: [C_out, 1, KH, KW]
    C_out, KH, KW,
    # Output shape
    OH, OW,
    # Conv parameters
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    # Channel multiplier: C_out // C_in (usually 1)
    ch_mult,
    BLOCK_X: tl.constexpr,  # OW tile
    BLOCK_Y: tl.constexpr,  # OH tile
):
    """Depthwise conv2d kernel.

    Grid: (cdiv(OW, BLOCK_X), cdiv(OH, BLOCK_Y), N * C_out)
    Each program computes a BLOCK_Y x BLOCK_X tile of one output channel.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C_out
    c_out = pid_nc % C_out
    c_in = c_out // ch_mult

    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)  # [BLOCK_X]
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)  # [BLOCK_Y]

    mask_h = oh < OH
    mask_w = ow < OW
    mask_2d = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros([BLOCK_Y, BLOCK_X], dtype=tl.float32)

    # Base pointer for input channel
    in_base = n * (C_in * IH * IW) + c_in * (IH * IW)

    # Base pointer for weight
    w_base = c_out * (KH * KW)  # weight shape is [C_out, 1, KH, KW], 1 is implicit

    for kh in range(0, KH):
        for kw in range(0, KW):
            # Input coordinates
            ih = oh[:, None] * stride_h - pad_h + kh * dilation_h  # [BLOCK_Y, 1]
            iw = ow[None, :] * stride_w - pad_w + kw * dilation_w  # [1, BLOCK_X]

            # Bounds check
            valid_h = (ih >= 0) & (ih < IH)
            valid_w = (iw >= 0) & (iw < IW)
            valid = mask_2d & valid_h & valid_w

            # Load input value
            in_offset = in_base + ih * IW + iw
            x_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)

            # Load weight value (scalar per kh, kw)
            w_offset = w_base + kh * KW + kw
            w_val = tl.load(weight_ptr + w_offset).to(tl.float32)

            acc += x_val * w_val

    # Store output
    out_base = n * (C_out * OH * OW) + c_out * (OH * OW)
    out_offset = out_base + oh[:, None] * OW + ow[None, :]
    tl.store(output_ptr + out_offset, acc, mask=mask_2d)


@triton.jit
def conv_depthwise2d_bias_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C_in, IH, IW,
    C_out, KH, KW,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    ch_mult,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Depthwise conv2d with bias, same structure as above."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C_out
    c_out = pid_nc % C_out
    c_in = c_out // ch_mult

    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)

    mask_h = oh < OH
    mask_w = ow < OW
    mask_2d = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros([BLOCK_Y, BLOCK_X], dtype=tl.float32)

    in_base = n * (C_in * IH * IW) + c_in * (IH * IW)
    w_base = c_out * (KH * KW)

    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh[:, None] * stride_h - pad_h + kh * dilation_h
            iw = ow[None, :] * stride_w - pad_w + kw * dilation_w

            valid_h = (ih >= 0) & (ih < IH)
            valid_w = (iw >= 0) & (iw < IW)
            valid = mask_2d & valid_h & valid_w

            in_offset = in_base + ih * IW + iw
            x_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)

            w_offset = w_base + kh * KW + kw
            w_val = tl.load(weight_ptr + w_offset).to(tl.float32)

            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + c_out).to(tl.float32)
    acc += bias_val

    out_base = n * (C_out * OH * OW) + c_out * (OH * OW)
    out_offset = out_base + oh[:, None] * OW + ow[None, :]
    tl.store(output_ptr + out_offset, acc, mask=mask_2d)
