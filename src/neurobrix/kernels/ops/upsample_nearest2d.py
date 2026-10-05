"""Upsample nearest 2D — pure @triton.jit kernel.

Extracted from FlagGems (FlagOpen/FlagGems) upsample_nearest2d.py.
Stripped FlagGems-specific decorators (@libentry, runtime heuristics).
"""

import triton
import triton.language as tl


@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    s_n,
    s_c,
    s_h,
    s_w,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    NC = N * C
    nc_stride = tl.num_programs(axis=1)
    nc_iter = (tl.program_id(axis=1).to(tl.int64)).to(tl.int32)   # loop bounds are 32-bit counts: the Metal lowering refuses a 64-bit scf.for bound (addressing stays 64-bit through the program ids)
    pid = tl.program_id(axis=0).to(tl.int64)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    ow = idx % OW
    oh = idx // OW % OH

    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # The input is read through its strides: a first-frame branch feeds this op `x[:, :, 0]` of a
    # 5-D activation, a view whose channel stride spans every frame — flat indexing read another
    # channel's frames there (CogVideoX / Open-Sora frame 0). The output is fresh and contiguous.
    offset_o = (nc_iter * OH + oh) * OW + ow
    in_hw = ih.to(tl.int64) * s_h + iw.to(tl.int64) * s_w
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < NC:
        n = nc_iter // C
        c = nc_iter % C
        data = tl.load(ptr_i + n.to(tl.int64) * s_n + c.to(tl.int64) * s_c + in_hw)
        tl.store(ptr_o + offset_o, data)
        ptr_o += dst_index_stride
        nc_iter += nc_stride
