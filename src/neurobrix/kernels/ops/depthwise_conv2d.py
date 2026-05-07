"""Depthwise conv2d — direct stencil @triton.jit kernel.

Routed by `kernels/wrappers.py:conv2d_wrapper` when the convolution
matches the depthwise signature (`groups == in_c == out_c`, weight
shape `(C, 1, kh, kw)`). The generic `conv2d_forward_kernel` (im2col +
tl.dot) is structurally inefficient for this case: each group has K =
kh*kw (often 9) so the inner reduction loop runs once with most of the
BLOCK_INF lane idle, yet the kernel still pays full launch + im2col
cost for every group. On Sana 4Kpx VAE (the DC-AE depthwise blocks
groups=in=out=11200), the generic kernel measures ~4.8 s per call
versus ~2.6 ms for cuDNN's dedicated depthwise path — a ~1800× gap
that closes under the dedicated stencil.

Pattern reference (P-SANA-4KPX-RUNTIME Étape 3 study):
- MultiPath/DepthwiseConv2d (CUTLASS sm_70 NCHW depthwise iterator,
  threadblock 64x128x32, warp 32x32x32, instruction 8x8x4 HMMA)
- PyTorch PR #22302 — confirmed cuDNN has a separate depthwise code
  path on Volta/Turing fp16 (cudnn >= 7600)

Our Triton-pure design: each program block computes a tile of
`(BLOCK_HW x BLOCK_C)` output positions for one batch, accumulating
the kh*kw stencil contributions in fp32. No im2col buffer, no
cross-channel reduction, no HMMA pseudo-GEMM packing. K-loop is
unrolled at compile time via `tl.constexpr` for kh, kw.
"""

import triton
import triton.language as tl

from ._autotune_policy import maybe_pin_single, is_depthwise_conv2d_pinned


_DEPTHWISE_CONV2D_CONFIGS = [
    triton.Config({'BLOCK_HW': 32, 'BLOCK_C': 32}, num_stages=2, num_warps=2),
    triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_HW': 32, 'BLOCK_C': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_stages=2, num_warps=8),
]


@triton.autotune(
    configs=maybe_pin_single(_DEPTHWISE_CONV2D_CONFIGS, is_depthwise_conv2d_pinned),
    key=['C', 'H_in', 'W_in', 'H_out', 'W_out', 'kh', 'kw',
         'stride_h', 'stride_w', 'pad_h', 'pad_w', 'fp16'],
    cache_results=True,
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C,
    H_in, W_in, H_out, W_out,
    x_n_stride, x_c_stride, x_h_stride, x_w_stride,
    w_c_stride, w_kh_stride, w_kw_stride,
    out_n_stride, out_c_stride, out_h_stride, out_w_stride,
    kh: tl.constexpr, kw: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_HW: tl.constexpr = 64,
    BLOCK_C: tl.constexpr = 32,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = offs_hw // W_out
    ow = offs_hw % W_out

    c_mask = offs_c < C
    hw_mask = offs_hw < H_out * W_out

    accum = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    # Unrolled stencil over kh x kw. Each (kh_i, kw_i) is a constexpr
    # because kh, kw are constexpr — Triton fully unrolls the loop body.
    for kh_i in tl.static_range(0, kh):
        for kw_i in tl.static_range(0, kw):
            ih = oh * stride_h + kh_i - pad_h
            iw = ow * stride_w + kw_i - pad_w
            spatial_mask = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
            full_mask = spatial_mask[:, None] & c_mask[None, :] & hw_mask[:, None]

            x_addr = (x_ptr + pid_n * x_n_stride
                      + offs_c[None, :] * x_c_stride
                      + ih[:, None] * x_h_stride
                      + iw[:, None] * x_w_stride)
            x_block = tl.load(x_addr, mask=full_mask, other=0.0)

            w_addr = (w_ptr + offs_c * w_c_stride
                      + kh_i * w_kh_stride
                      + kw_i * w_kw_stride)
            w_block = tl.load(w_addr, mask=c_mask, other=0.0)

            if fp16:
                accum += (x_block.to(tl.float32) * w_block.to(tl.float32)[None, :])
            else:
                accum += x_block * w_block[None, :]

    out_addr = (out_ptr + pid_n * out_n_stride
                + offs_c[None, :] * out_c_stride
                + oh[:, None] * out_h_stride
                + ow[:, None] * out_w_stride)
    out_mask = c_mask[None, :] & hw_mask[:, None]
    if fp16:
        tl.store(out_addr, accum.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_addr, accum, mask=out_mask)
