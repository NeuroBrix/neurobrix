"""Upsample nearest 3D — pure @triton.jit kernel.

3D nearest neighbor upsampling: [N,C,D,H,W] -> [N,C,D*s_d,H*s_h,W*s_w].
Adapted from our upsample_nearest2d.py by adding the depth dimension.
"""

import triton
import triton.language as tl


@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Nearest-neighbor 3D upsample.

    Grid: (cdiv(OD*OH*OW, BLOCK_SIZE), cdiv(N*C, 4))
    Each program handles BLOCK_SIZE output pixels across one (n,c) slice,
    iterating over (n,c) in strides of nc_stride.
    """
    NC = N * C
    nc_stride = tl.num_programs(axis=1)
    nc_iter = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Decompose flat index -> (od, oh, ow)
    ow = idx % OW
    oh = (idx // OW) % OH
    od = (idx // (OW * OH)) % OD

    # Map output coords to input coords (nearest neighbor)
    id_ = tl.minimum((od * reciprocal_scale_d).to(tl.int32), ID - 1)
    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # Compute offsets for contiguous [N, C, D, H, W] layout
    offset_o = ((nc_iter * OD + od) * OH + oh) * OW + ow
    offset_i = ((nc_iter * ID + id_) * IH + ih) * IW + iw
    src_index_stride = nc_stride * ID * IH * IW
    dst_index_stride = nc_stride * OD * OH * OW

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride
