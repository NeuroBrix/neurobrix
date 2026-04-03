"""Pixel unshuffle (inverse of pixel shuffle) — pure @triton.jit kernel.

Rearranges: [N, C, H*r, W*r] -> [N, C*r*r, H, W]

For each output element (n, oc, oh, ow):
    c = oc // (r * r)
    sub_h = (oc % (r * r)) // r
    sub_w = (oc % (r * r)) % r
    ih = oh * r + sub_h
    iw = ow * r + sub_w
    output[n, oc, oh, ow] = input[n, c, ih, iw]
"""

import triton
import triton.language as tl

@triton.jit
def pixel_unshuffle_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    # Input shape: [N, C, IH, IW] where IH = H*r, IW = W*r
    C,
    IH,
    IW,
    r,
    # Output shape: [N, C*r*r, H, W]
    C_out,  # = C * r * r
    H,      # = IH // r
    W,      # = IW // r
    # Input strides (contiguous NCHW)
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    # Output strides (contiguous NCHW)
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Pixel unshuffle: [N, C, H*r, W*r] -> [N, C*r*r, H, W].

    Each thread handles one output element.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Decompose flat output index -> (n, oc, oh, ow)
    ow = idx % W
    tmp = idx // W
    oh = tmp % H
    tmp2 = tmp // H
    oc = tmp2 % C_out
    n = tmp2 // C_out

    # Map output coords to input coords
    c = oc // (r * r)
    sub_idx = oc % (r * r)
    sub_h = sub_idx // r
    sub_w = sub_idx % r
    ih = oh * r + sub_h
    iw = ow * r + sub_w

    # Load from input, store to output
    in_offset = n * stride_in_n + c * stride_in_c + ih * stride_in_h + iw * stride_in_w
    out_offset = n * stride_out_n + oc * stride_out_c + oh * stride_out_h + ow * stride_out_w

    val = tl.load(input_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)
