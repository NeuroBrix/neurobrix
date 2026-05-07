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


@triton.jit
def pixel_shuffle_broadcast_aware_kernel(
    bcast_view_ptr,  # 5D NBX expand-output view (N, C_pre, B, H, W) with stride_b == 0
    output_ptr,      # 4D contiguous output (N, C_out, OH, OW)
    n_elements,
    C_out,           # output channels = (C_pre * B) // (r*r)
    OH,              # = H * r
    OW,              # = W * r
    r,               # pixel_shuffle upscale factor
    bcast,           # broadcast factor (= B, the size on the stride-0 dim)
    # 5D bcast_view strides (NBX expand sets stride_b = 0).
    stride_v_n,
    stride_v_c,
    stride_v_b,
    stride_v_h,
    stride_v_w,
    # 4D output strides (contiguous NCHW).
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Pixel shuffle that reads through an unmaterialized NBX broadcast view.

    Input layout: (N, C_pre, B, H, W) where the original DAG pattern is
    `unsqueeze(dim=2) -> expand(dim=2, B) -> clone -> view -> pixel_shuffle`.
    NBX expand sets stride_b=0 (kernels/nbx_tensor.py:1581), so the 5D view
    aliases the pre-expand tensor with no materialization. We read through
    that view directly and skip the clone's 8 GiB copy.

    The post-clone-post-view shape consumed by a normal pixel_shuffle is
    (N, C_pre*B, H, W); here the kernel does the equivalent decomposition
    inline:
      ic_view = c*r*r + (oh%r)*r + (ow%r)         in [0, C_pre*B)
      c_pre   = ic_view // bcast                   in [0, C_pre)
      b       = ic_view %  bcast                   in [0, B)
      ih      = oh // r
      iw      = ow // r
    Read offset uses all 5D strides; with stride_v_b == 0, the `b` index
    contributes nothing and the load aliases naturally.
    """
    pid = tl.program_id(0)
    # Cast to int64 to support tensors with >= 2^31 elements
    # (e.g. Sana 4Kpx VAE 1*128*4096*4096 = 2^31 — at INT32_MAX
    # boundary; signed int32 multiplication of strides overflows
    # silently and corrupts offsets, producing garbage output).
    idx = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = idx < n_elements

    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C_out
    n = tmp2 // C_out

    sub_h = oh % r
    sub_w = ow % r
    ic_view = c * (r * r) + sub_h * r + sub_w
    c_pre = ic_view // bcast
    b = ic_view % bcast

    ih = oh // r
    iw = ow // r

    in_offset = (n * stride_v_n + c_pre * stride_v_c + b * stride_v_b
                 + ih * stride_v_h + iw * stride_v_w)
    out_offset = (n * stride_out_n + c * stride_out_c
                  + oh * stride_out_h + ow * stride_out_w)

    val = tl.load(bcast_view_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)
