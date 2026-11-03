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

Pattern reference (P-SANA-4KPX-RUNTIME Step 3 study):
- MultiPath/DepthwiseConv2d (CUTLASS sm_70 NCHW depthwise iterator,
  threadblock 64x128x32, warp 32x32x32, instruction 8x8x4 HMMA)
- PyTorch PR #22302 — confirmed cuDNN has a separate depthwise code
  path on Volta/Turing fp16 (cudnn >= 7600)

Our Triton-pure design: each program computes a (BC channels x TH rows x TW columns) output tile of one batch
sample, width the fastest axis (NCHW's contiguous one), accumulating the kh*kw stencil contributions in fp32: the
kh*kw shifted input windows of a compact 2-D tile overlap and are served from the L1. No im2col buffer, no
cross-channel reduction, no HMMA pseudo-GEMM packing; the stencil is unrolled at compile time (kh, kw constexpr).
The tile was (BLOCK_HW flattened positions x BLOCK_C channels) with the channels fastest, H*W elements apart in
memory: no load coalesced — SANA-Video's 11200-channel 3x3 fp32 [2, 11200, 24, 40] ran 2.63 ms where this one runs
0.33 ms (529 GB/s; torch eager 0.38 ms), [22, 13440, 21, 42] 18.9 -> 4.2 ms (V100-SXM2, 2026-11-03).
"""

import triton

from nbxtensor.ops._configs import nbx_autotune
import triton.language as tl

from nbxtensor.ops._autotune_policy import maybe_pin_single, is_depthwise_conv2d_pinned


_DEPTHWISE_CONV2D_CONFIGS = [
    triton.Config({'BC': 1, 'TH': 8, 'TW': 64}, num_stages=2, num_warps=4),
    triton.Config({'BC': 1, 'TH': 16, 'TW': 64}, num_stages=2, num_warps=4),
    triton.Config({'BC': 1, 'TH': 32, 'TW': 64}, num_stages=2, num_warps=8),
    triton.Config({'BC': 2, 'TH': 8, 'TW': 64}, num_stages=2, num_warps=4),
    triton.Config({'BC': 4, 'TH': 8, 'TW': 32}, num_stages=2, num_warps=4),
    triton.Config({'BC': 4, 'TH': 8, 'TW': 64}, num_stages=2, num_warps=8),
    triton.Config({'BC': 8, 'TH': 4, 'TW': 64}, num_stages=2, num_warps=8),
    triton.Config({'BC': 16, 'TH': 4, 'TW': 32}, num_stages=2, num_warps=8),
    triton.Config({'BC': 1, 'TH': 4, 'TW': 128}, num_stages=2, num_warps=4),
    # one-row tiles: the census's depthwise rows are mostly conv1d run at H = 1 (1 x 7, 64-1024 channels), where a
    # taller tile is all masked rows — measured best there (1024 x 2048 fp16: 0.027 ms, the flattened tile 0.062)
    triton.Config({'BC': 4, 'TH': 1, 'TW': 128}, num_stages=2, num_warps=4),
    triton.Config({'BC': 8, 'TH': 1, 'TW': 64}, num_stages=2, num_warps=4),
    triton.Config({'BC': 16, 'TH': 1, 'TW': 32}, num_stages=2, num_warps=4),
    triton.Config({'BC': 16, 'TH': 1, 'TW': 64}, num_stages=2, num_warps=8),
    triton.Config({'BC': 32, 'TH': 1, 'TW': 32}, num_stages=2, num_warps=8),
]


@nbx_autotune(
    configs=maybe_pin_single(_DEPTHWISE_CONV2D_CONFIGS, is_depthwise_conv2d_pinned),
    key=['C', 'H_in', 'W_in_key', 'H_out', 'W_out_key', 'kh', 'kw',
         'stride_h', 'stride_w', 'pad_h', 'pad_w', 'fp16'],
    cache_results=True,
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C,
    H_in, W_in, H_out, W_out,
    W_in_key, W_out_key,
    x_n_stride, x_c_stride, x_h_stride, x_w_stride,
    w_c_stride, w_kh_stride, w_kw_stride,
    out_n_stride, out_c_stride, out_h_stride, out_w_stride,
    kh: tl.constexpr, kw: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    fp16: tl.constexpr,
    BC: tl.constexpr = 2,
    TH: tl.constexpr = 8,
    TW: tl.constexpr = 64,
    WIDE_OFFSETS: tl.constexpr = True,
):
    # 64-bit offsets only where an operand's extent needs them (`WIDE_OFFSETS`, the wrapper's `_conv_offsets_wide`, as
    # conv2d): 32-bit offsets are exact below 2^31, and the 64-bit ones are emulated on Apple's ALUs — under the i64 fix of
    # the Metal backend (b06cbb0d) the fp32 3x3 depthwise (13440 ch, 42x21) ran 8.4 -> 12.5 ms (Mac, 2026-10-21)
    if WIDE_OFFSETS:
        pid_nc = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        pid_w = tl.program_id(2).to(tl.int64)
    else:
        pid_nc = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_w = tl.program_id(2)
    n_cb = tl.cdiv(C, BC)
    pid_n = pid_nc // n_cb
    offs_c = (pid_nc % n_cb) * BC + tl.arange(0, BC)
    oh = pid_h * TH + tl.arange(0, TH)
    ow = pid_w * TW + tl.arange(0, TW)
    c_mask = offs_c < C

    accum = tl.zeros((BC, TH, TW), dtype=tl.float32)
    x_base = x_ptr + pid_n * x_n_stride + offs_c[:, None, None] * x_c_stride

    # Unrolled stencil over kh x kw (constexpr): each tap reads the input window the output tile shifted by it.
    for kh_i in tl.static_range(0, kh):
        ih = oh * stride_h + kh_i - pad_h
        h_ok = (ih >= 0) & (ih < H_in)
        for kw_i in tl.static_range(0, kw):
            iw = ow * stride_w + kw_i - pad_w
            w_ok = (iw >= 0) & (iw < W_in)
            full_mask = c_mask[:, None, None] & h_ok[None, :, None] & w_ok[None, None, :]
            x_block = tl.load(x_base + ih[None, :, None] * x_h_stride + iw[None, None, :] * x_w_stride,
                              mask=full_mask, other=0.0)
            w_block = tl.load(w_ptr + offs_c * w_c_stride + kh_i * w_kh_stride + kw_i * w_kw_stride,
                              mask=c_mask, other=0.0)
            # Upcast for EVERY dtype, not just fp16. The accumulator is fp32 either way, so
            # this costs nothing — and multiplying in the operands' own dtype made bf16 with
            # padding wrong on Metal: deviation 0.754 against the fp64 oracle where unpadded
            # bf16 sat at its 3.2e-03 mantissa floor, with the error in the ALL-IN-BOUNDS
            # interior while the genuinely masked border row was clean. fp32 and fp16 were
            # unaffected at both paddings, and the same source is exact to the mantissa on
            # CUDA (Triton 3.8.0, V100 sm_70: bf16 pad0 == pad1), so the bad lowering is the
            # Metal backend's for a native bf16 product of a masked-loaded operand. Upcasting
            # removes the dependence on it rather than relying on it being fixed below us.
            # Gate: tests/unit/kernels/test_depthwise_bf16_padding.py (2026-09-22).
            accum += x_block.to(tl.float32) * w_block.to(tl.float32)[:, None, None]

    out_addr = (out_ptr + pid_n * out_n_stride + offs_c[:, None, None] * out_c_stride
                + oh[None, :, None] * out_h_stride + ow[None, None, :] * out_w_stride)
    out_mask = c_mask[:, None, None] & (oh < H_out)[None, :, None] & (ow < W_out)[None, None, :]
    if fp16:
        tl.store(out_addr, accum.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_addr, accum, mask=out_mask)
