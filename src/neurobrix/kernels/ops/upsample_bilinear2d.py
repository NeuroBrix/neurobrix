"""Bilinear 2D upsampling — pure @triton.jit kernel.

For each output pixel (n, c, oh, ow), compute source coordinates in
input space, find the 4 nearest input pixels, and bilinearly interpolate.

Supports align_corners=True (maps corner-to-corner) and align_corners=False
(maps center-to-center with half-pixel offset).
"""

import triton
import triton.language as tl


@triton.jit
def upsample_bilinear2d_kernel(
    input_ptr,
    output_ptr,
    N, C,
    IH, IW,
    OH, OW,
    # Precomputed scale factors: how input coords map to output coords
    # For align_corners=False: scale_h = IH / OH, scale_w = IW / OW
    # For align_corners=True:  scale_h = (IH-1)/(OH-1), scale_w = (IW-1)/(OW-1)
    scale_h,
    scale_w,
    # 0.0 for align_corners=True, 0.5 for align_corners=False
    offset,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Bilinear interpolation upsampling.

    Grid: (cdiv(OW, BLOCK_X), cdiv(OH, BLOCK_Y), N*C)
    Each program handles a BLOCK_Y x BLOCK_X tile of output pixels for one (n, c).
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_nc = tl.program_id(2)

    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)  # [BLOCK_X]
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)  # [BLOCK_Y]

    # Map output coords to input coords (floating point)
    # align_corners=False: src = (dst + 0.5) * scale - 0.5
    # align_corners=True:  src = dst * scale
    src_h = (oh.to(tl.float32) + offset) * scale_h - offset  # [BLOCK_Y]
    src_w = (ow.to(tl.float32) + offset) * scale_w - offset  # [BLOCK_X]

    # Clamp to valid range
    src_h = tl.maximum(src_h, 0.0)
    src_w = tl.maximum(src_w, 0.0)
    # IH/IW may arrive as a Python int (Triton-specialized) or a tl scalar
    # depending on the call's shape — `.to()` fails on the former, float() on the
    # latter. `* 1.0` promotes to float robustly for both.
    src_h = tl.minimum(src_h, (IH - 1) * 1.0)
    src_w = tl.minimum(src_w, (IW - 1) * 1.0)

    # Floor and ceil indices
    h0 = src_h.to(tl.int32)                         # [BLOCK_Y]
    w0 = src_w.to(tl.int32)                         # [BLOCK_X]
    h1 = tl.minimum(h0 + 1, IH - 1)                # [BLOCK_Y]
    w1 = tl.minimum(w0 + 1, IW - 1)                # [BLOCK_X]

    # Fractional parts (bilinear weights)
    fh = src_h - h0.to(tl.float32)                  # [BLOCK_Y]
    fw = src_w - w0.to(tl.float32)                  # [BLOCK_X]

    # Bilinear weights for 4 corners
    w_h0 = 1.0 - fh   # [BLOCK_Y]
    w_h1 = fh          # [BLOCK_Y]
    w_w0 = 1.0 - fw   # [BLOCK_X]
    w_w1 = fw          # [BLOCK_X]

    # Base offset for this (n, c) plane
    base = pid_nc * IH * IW

    # Masks
    mask_h = oh < OH  # [BLOCK_Y]
    mask_w = ow < OW  # [BLOCK_X]
    mask_2d = mask_h[:, None] & mask_w[None, :]  # [BLOCK_Y, BLOCK_X]

    # Load 4 corner values
    # top-left:     input[n, c, h0, w0]
    # top-right:    input[n, c, h0, w1]
    # bottom-left:  input[n, c, h1, w0]
    # bottom-right: input[n, c, h1, w1]
    off_tl = base + h0[:, None] * IW + w0[None, :]
    off_tr = base + h0[:, None] * IW + w1[None, :]
    off_bl = base + h1[:, None] * IW + w0[None, :]
    off_br = base + h1[:, None] * IW + w1[None, :]

    v_tl = tl.load(input_ptr + off_tl, mask=mask_2d, other=0.0).to(tl.float32)
    v_tr = tl.load(input_ptr + off_tr, mask=mask_2d, other=0.0).to(tl.float32)
    v_bl = tl.load(input_ptr + off_bl, mask=mask_2d, other=0.0).to(tl.float32)
    v_br = tl.load(input_ptr + off_br, mask=mask_2d, other=0.0).to(tl.float32)

    # Bilinear interpolation
    # result = w_h0 * (w_w0 * v_tl + w_w1 * v_tr) + w_h1 * (w_w0 * v_bl + w_w1 * v_br)
    top = w_w0[None, :] * v_tl + w_w1[None, :] * v_tr
    bot = w_w0[None, :] * v_bl + w_w1[None, :] * v_br
    result = w_h0[:, None] * top + w_h1[:, None] * bot

    # Store output
    out_base = pid_nc * OH * OW
    out_off = out_base + oh[:, None] * OW + ow[None, :]
    tl.store(output_ptr + out_off, result, mask=mask_2d)
