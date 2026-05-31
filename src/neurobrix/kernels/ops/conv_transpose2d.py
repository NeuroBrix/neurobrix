"""ConvTranspose2d — pure @triton.jit kernel.

Transposed convolution (deconvolution) used in decoder/upsampling paths.
Implements the scatter-based approach: for each output pixel, accumulate
contributions from all input pixels whose receptive field covers it.

No reference project had a Triton conv_transpose2d, so this is written
from first principles using the standard relationship:
    output[n, co, oh, ow] = sum over ci,kh,kw of:
        input[n, ci, ih, iw] * weight[ci, co, kh, kw]
    where ih = (oh + pad_h - kh) / stride_h  (must be integer)
          iw = (ow + pad_w - kw) / stride_w  (must be integer)

Grid: (N * C_out, ceil(OH * OW / BLOCK_SIZE))
"""

import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,       # (N, C_in, IH, IW) contiguous
    weight_ptr,      # (C_in, C_out/groups, KH, KW) contiguous
    output_ptr,      # (N, C_out, OH, OW) contiguous
    N, C_in, IH, IW,
    C_out, KH, KW,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    C_in_per_g, C_out_per_g,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter-based transposed convolution kernel (groups + dilation aware).

    Each program computes a block of output pixels for one (batch, out_channel).
    For each output pixel (oh, ow), it iterates over the (c_in, kh, kw) of the
    output channel's group and checks if the corresponding input pixel exists
    (integer division constraint). Weight layout matches PyTorch ConvTranspose:
    (C_in, C_out/groups, KH, KW). groups=1 (C_in_per_g=C_in, C_out_per_g=C_out)
    reduces exactly to the original full-sum behaviour.
    """
    # pid_0 = n * C_out + co
    pid_nc = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    n = pid_nc // C_out
    co = pid_nc % C_out

    # Group mapping: which input-channel block + local out-channel index this
    # output channel belongs to (PyTorch grouped-ConvTranspose semantics).
    group = co // C_out_per_g
    co_local = co % C_out_per_g
    ci_start = group * C_in_per_g

    # Output spatial offsets for this block
    out_base = pid_spatial * BLOCK_SIZE
    out_offsets = out_base + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < (OH * OW)

    oh = out_offsets // OW
    ow = out_offsets % OW

    # Accumulate in float32 for numerical stability
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the group's input channels and kernel positions
    for ci_local in range(C_in_per_g):
        ci = ci_start + ci_local
        for kh in range(KH):
            for kw in range(KW):
                # Inverse mapping: ih = (oh + pad_h - kh*dil_h) / stride_h
                ih_num = oh + pad_h - kh * dil_h
                iw_num = ow + pad_w - kw * dil_w

                # Check divisibility and bounds
                ih_valid = (ih_num % stride_h) == 0
                iw_valid = (iw_num % stride_w) == 0
                ih = ih_num // stride_h
                iw = iw_num // stride_w

                valid = (
                    ih_valid & iw_valid
                    & (ih >= 0) & (ih < IH)
                    & (iw >= 0) & (iw < IW)
                    & out_mask
                )

                # Load input[n, ci, ih, iw]
                in_offset = (
                    n * (C_in * IH * IW)
                    + ci * (IH * IW)
                    + ih * IW
                    + iw
                )
                in_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)

                # Load weight[ci, co_local, kh, kw] — layout (C_in, C_out/groups, KH, KW)
                w_offset = (
                    ci * (C_out_per_g * KH * KW)
                    + co_local * (KH * KW)
                    + kh * KW
                    + kw
                )
                # w_offset is scalar (one weight element for the whole output
                # block) — load unmasked; per-position validity is applied below.
                w_val = tl.load(weight_ptr + w_offset)

                acc += tl.where(valid, in_val * w_val, 0.0)

    # Store output[n, co, oh, ow]
    out_offset = (
        n * (C_out * OH * OW)
        + co * (OH * OW)
        + out_offsets
    )
    tl.store(output_ptr + out_offset, acc, mask=out_mask)
