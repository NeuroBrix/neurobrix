"""Pixel shuffle (sub-pixel convolution) — pure @triton.jit kernel.

Rearranges: [N, C*r*r, H, W] -> [N, C, H*r, W*r]

For each output element (n, c, oh, ow):
    ih = oh // r
    iw = ow // r
    ic = c * (r * r) + (oh % r) * r + (ow % r)
    output[n, c, oh, ow] = input[n, ic, ih, iw]
"""

import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    # Input shape: [N, C_in, H, W] where C_in = C_out * r * r
    C_out,
    H,
    W,
    r,
    # Output shape: [N, C_out, H*r, W*r]
    OH,  # = H * r
    OW,  # = W * r
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
    """Pixel shuffle: [N, C*r*r, H, W] -> [N, C, H*r, W*r].

    Each thread handles one output element.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Decompose flat output index -> (n, c, oh, ow)
    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C_out
    n = tmp2 // C_out

    # Map output coords to input coords
    ih = oh // r
    iw = ow // r
    # Sub-pixel position within the r x r block
    sub_h = oh % r
    sub_w = ow % r
    ic = c * (r * r) + sub_h * r + sub_w

    # Load from input, store to output
    in_offset = n * stride_in_n + ic * stride_in_c + ih * stride_in_h + iw * stride_in_w
    out_offset = n * stride_out_n + c * stride_out_c + oh * stride_out_h + ow * stride_out_w

    val = tl.load(input_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)
