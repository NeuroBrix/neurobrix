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

    offset_o = (nc_iter * OH + oh) * OW + ow
    offset_i = (nc_iter * IH + ih) * IW + iw
    src_index_stride = nc_stride * IH * IW
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride
