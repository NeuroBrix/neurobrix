"""aten::grid_sampler_2d — pure Triton 2D grid sampler.

Clean-room implementation from the PyTorch semantics (ATen GridSampler.h /
torch._decomp grid sampler decomposition — formulas only, no code vendored):

- Coordinate unnormalization:
    align_corners=True:  ix = (x + 1)/2 * (W - 1)
    align_corners=False: ix = ((x + 1)*W - 1)/2
- Bicubic: cubic-convolution weights with A = -0.75 over a 4x4 tap window;
  the fractional offset is computed on the RAW unnormalized coordinate
  (floor before padding), padding is applied PER INTEGER TAP (this staging
  differs from bilinear and changes border behavior).
- Bilinear: padding (clamp) is applied to the FLOAT coordinate before
  flooring; 4 corners accumulated with per-corner in-bounds masks.
- zeros padding: out-of-bounds taps contribute 0 (masked loads);
  border padding: tap indices clamp into [0, size-1].
- All coordinate/weight math and accumulation in float32 (ATen opmath),
  output cast to the input dtype by the wrapper.

Modes follow the ATen integer encoding: interpolation 0=bilinear,
2=bicubic (1=nearest unsupported → wrapper raises); padding 0=zeros,
1=border (2=reflection unsupported → wrapper raises).

Parallelization: one program per (batch, output-location, channel-block);
coordinates and weights are computed once per location, taps are gathered
as BLOCK_C-wide strided loads (mirrors the CUDA kernel's channel loop).
"""

import triton
import triton.language as tl


@triton.jit
def _cc1(x, A: tl.constexpr):
    # |x| <= 1 branch of the cubic convolution kernel
    return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0


@triton.jit
def _cc2(x, A: tl.constexpr):
    # 1 < |x| < 2 branch of the cubic convolution kernel
    return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A


@triton.jit
def grid_sampler_2d_kernel(
    inp_ptr, grid_ptr, out_ptr,
    C, H, W, OUT_SPATIAL,
    stride_in, stride_ic, stride_ih, stride_iw,
    stride_gn, stride_gs, stride_gc,
    stride_on, stride_oc, stride_os,
    MODE: tl.constexpr,          # 0 = bilinear, 2 = bicubic
    PAD_BORDER: tl.constexpr,    # 0 = zeros, 1 = border
    ALIGN: tl.constexpr,         # 0 / 1
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0)     # output spatial location (oH*oW flattened)
    pid_c = tl.program_id(1)     # channel block
    pid_n = tl.program_id(2)     # batch

    if pid_s >= OUT_SPATIAL:
        return

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # grid holds (x, y) pairs, channels-last
    gx = tl.load(grid_ptr + pid_n * stride_gn + pid_s * stride_gs
                 + 0 * stride_gc).to(tl.float32)
    gy = tl.load(grid_ptr + pid_n * stride_gn + pid_s * stride_gs
                 + 1 * stride_gc).to(tl.float32)

    Wf = W.to(tl.float32)
    Hf = H.to(tl.float32)
    if ALIGN:
        ix = (gx + 1.0) * 0.5 * (Wf - 1.0)
        iy = (gy + 1.0) * 0.5 * (Hf - 1.0)
    else:
        ix = ((gx + 1.0) * Wf - 1.0) * 0.5
        iy = ((gy + 1.0) * Hf - 1.0) * 0.5

    inp_base = inp_ptr + pid_n * stride_in + c_offs * stride_ic
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    if MODE == 0:
        # ── bilinear: pad the FLOAT coordinate first, then corners ──
        if PAD_BORDER:
            ix = tl.minimum(tl.maximum(ix, 0.0), Wf - 1.0)
            iy = tl.minimum(tl.maximum(iy, 0.0), Hf - 1.0)
        x0 = tl.floor(ix)
        y0 = tl.floor(iy)
        tx = ix - x0
        ty = iy - y0
        for j in tl.static_range(2):
            for i in tl.static_range(2):
                xi = x0 + i
                yi = y0 + j
                wgt = (tl.where(i == 0, 1.0 - tx, tx)
                       * tl.where(j == 0, 1.0 - ty, ty))
                inb = (xi >= 0) & (xi < Wf) & (yi >= 0) & (yi < Hf)
                xs = tl.minimum(tl.maximum(xi, 0.0), Wf - 1.0).to(tl.int32)
                ys = tl.minimum(tl.maximum(yi, 0.0), Hf - 1.0).to(tl.int32)
                val = tl.load(inp_base + ys * stride_ih + xs * stride_iw,
                              mask=c_mask & inb, other=0.0).to(tl.float32)
                acc += tl.where(inb, wgt, 0.0) * val
    else:
        # ── bicubic: floor on the RAW coordinate, pad per integer tap ──
        A: tl.constexpr = -0.75
        x0 = tl.floor(ix)
        y0 = tl.floor(iy)
        tx = ix - x0
        ty = iy - y0
        wx0 = _cc2(tx + 1.0, A)
        wx1 = _cc1(tx, A)
        wx2 = _cc1(1.0 - tx, A)
        wx3 = _cc2(2.0 - tx, A)
        wy0 = _cc2(ty + 1.0, A)
        wy1 = _cc1(ty, A)
        wy2 = _cc1(1.0 - ty, A)
        wy3 = _cc2(2.0 - ty, A)
        for j in tl.static_range(4):
            wy = tl.where(j == 0, wy0,
                 tl.where(j == 1, wy1,
                 tl.where(j == 2, wy2, wy3)))
            yi = y0 + (j - 1)
            for i in tl.static_range(4):
                wx = tl.where(i == 0, wx0,
                     tl.where(i == 1, wx1,
                     tl.where(i == 2, wx2, wx3)))
                xi = x0 + (i - 1)
                inb = (xi >= 0) & (xi < Wf) & (yi >= 0) & (yi < Hf)
                xs = tl.minimum(tl.maximum(xi, 0.0), Wf - 1.0).to(tl.int32)
                ys = tl.minimum(tl.maximum(yi, 0.0), Hf - 1.0).to(tl.int32)
                if PAD_BORDER:
                    val = tl.load(inp_base + ys * stride_ih + xs * stride_iw,
                                  mask=c_mask, other=0.0).to(tl.float32)
                    acc += wy * wx * val
                else:
                    val = tl.load(inp_base + ys * stride_ih + xs * stride_iw,
                                  mask=c_mask & inb, other=0.0).to(tl.float32)
                    acc += tl.where(inb, wy * wx, 0.0) * val

    out = out_ptr + pid_n * stride_on + c_offs * stride_oc + pid_s * stride_os
    tl.store(out, acc, mask=c_mask)
