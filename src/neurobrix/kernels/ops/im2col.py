"""aten::im2col (nn.Unfold) — pure @triton.jit kernel.

Extracts sliding local blocks from a batched NCHW tensor into columns:
    input  [N, C, IH, IW]
    output [N, C*KH*KW, L]   with L = OH*OW

    out[n, c*KH*KW + kh*KW + kw, oh*OW + ow]
        = input[n, c, oh*stride_h - pad_h + kh*dil_h,
                       ow*stride_w - pad_w + kw*dil_w]   (0 outside the image)

This is the operation behind HAT's OCAB overlapping-window cross-attention
(P-TRITON-IM2COL-KERNEL). Pure Triton (R33): NBXTensor in/out, tl.load with
boundary masking for the padding halo (no F.pad).
"""

import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    inp_ptr, out_ptr,
    C, IH, IW,
    OW, L,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr,
    in_sn, in_sc, in_sh, in_sw,
    out_sn, out_sc, out_sl,
    BLOCK_L: tl.constexpr,
):
    # One program per (n, output-channel) = (n, c*KH*KW + kh*KW + kw); it
    # writes one full output row of L block-positions (looped in BLOCK_L tiles).
    pid = tl.program_id(0)
    CKK = C * KH * KW
    n = pid // CKK
    oc = pid % CKK
    c = oc // (KH * KW)
    rem = oc % (KH * KW)
    kh = rem // KW
    kw = rem % KW

    in_base = n * in_sn + c * in_sc
    out_base = n * out_sn + oc * out_sc

    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        oh = l // OW
        ow = l % OW
        ih = oh * stride_h - pad_h + kh * dil_h
        iw = ow * stride_w - pad_w + kw * dil_w
        in_bounds = mask_l & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        vals = tl.load(inp_ptr + in_base + ih * in_sh + iw * in_sw,
                       mask=in_bounds, other=0.0)
        tl.store(out_ptr + out_base + l * out_sl, vals, mask=mask_l)
