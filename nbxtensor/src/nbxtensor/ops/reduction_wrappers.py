"""Reduction wrappers: sum, mean, prod, min/max, argmin/argmax, all/any, std/var, cumulative scans. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

from __future__ import annotations

import os
import triton
from nbxtensor.ops.autotune_bucket import bucket_of as _bucket_of
from nbxtensor.ops import launch_keys as _lk
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.backend.runtime import set_device
from nbxtensor.shape import MAX_NDIM as _NBX_MAX_NDIM, broadcast_shapes, max_linear_offset
from nbxtensor.dtype.itemsize import canonical_name
from nbxtensor.dtype.promotion import binary_result, float_math_input
from nbxtensor.dtype.nbx_dtype import parse_dtype
from nbxtensor.memory.allocator import DeviceOOMError
from nbxtensor.backend.capabilities import fa_min_tile as _fa_min_tile
from nbxtensor.backend.capabilities import reduce_tile_max as _reduce_tile_max
from nbxtensor.backend.faults import device_fault_buffer, device_fault_code_cached, fault_channel
from nbxtensor.tensor import _upload_int64_array as _nbx_upload_int64, _saturating_cast as _nbx_saturating_cast
from nbxtensor.ops._configs import sdpa_block_ceiling as _sdpa_block_ceiling
from nbxtensor.ops._configs import sdpa_launch_meta as _sdpa_launch_meta
from nbxtensor.ops._configs import matrix_unit as _matrix_unit, matrix_unit_mm_tile as _matrix_unit_mm_tile
from nbxtensor.ops._configs import largest_tile_within_smem as _largest_tile_within_smem
from nbxtensor.backend import launcher as _nbx_launcher
from nbxtensor.backend.launcher import install as _install_launcher
from nbxtensor.ops.relu import relu_forward_kernel
from nbxtensor.ops.silu import silu_forward_kernel
from nbxtensor.ops.gelu import gelu_forward_kernel
from nbxtensor.ops.sigmoid import sigmoid_forward_kernel
from nbxtensor.ops.tanh import tanh_forward_kernel
from nbxtensor.ops.hardsigmoid import hardsigmoid_forward_kernel
from nbxtensor.ops.hardswish import hardswish_forward_kernel
from nbxtensor.ops.leaky_relu import leaky_relu_forward_kernel
from nbxtensor.ops.elu import elu_forward_kernel
from nbxtensor.ops.mish import mish_forward_kernel
from nbxtensor.ops.selu import selu_forward_kernel
from nbxtensor.ops.neg import neg_forward_kernel
from nbxtensor.ops.exp import exp_forward_kernel
from nbxtensor.ops.sin import sin_forward_kernel
from nbxtensor.ops.cos import cos_forward_kernel
from nbxtensor.ops.rsqrt import rsqrt_forward_kernel
from nbxtensor.ops.sqrt import sqrt_forward_kernel
from nbxtensor.ops.abs import abs_forward_kernel
from nbxtensor.ops.log import log_forward_kernel
from nbxtensor.ops.reciprocal import reciprocal_forward_kernel
from nbxtensor.ops.pow import pow_forward_kernel
from nbxtensor.ops.clamp import clamp_forward_kernel
from nbxtensor.ops.erf import erf_forward_kernel
from nbxtensor.ops.copy import copy_forward_kernel
from nbxtensor.ops.add import add_forward_kernel, add_scalar_kernel, add_scalar_dev_kernel, add_bias_broadcast_kernel
from nbxtensor.ops.binary_strided import add_strided_nd_kernel, mul_strided_nd_kernel, sub_strided_nd_kernel, div_strided_nd_kernel, add_scalar_strided_nd_kernel, mul_scalar_strided_nd_kernel
from nbxtensor.ops.mul import mul_forward_kernel, mul_scalar_kernel, mul_scalar_dev_kernel
from nbxtensor.ops.div import div_forward_kernel, div_scalar_kernel, div_scalar_dev_kernel, int_div_kernel
from nbxtensor.ops.sub import sub_forward_kernel, rsub_forward_kernel
from nbxtensor.ops.where import where_forward_kernel
from nbxtensor.ops.maximum import maximum_forward_kernel
from nbxtensor.ops.minimum import minimum_forward_kernel
from nbxtensor.ops.masked_fill import masked_fill_forward_kernel
from nbxtensor.ops.gt import gt_forward_kernel, gt_scalar_kernel
from nbxtensor.ops.ge import ge_forward_kernel, ge_scalar_kernel
from nbxtensor.ops.lt import lt_forward_kernel, lt_scalar_kernel
from nbxtensor.ops.eq import eq_forward_kernel, eq_scalar_kernel
from nbxtensor.ops.ne import ne_forward_kernel, ne_scalar_kernel
from nbxtensor.ops.layernorm import layer_norm_forward_kernel
from nbxtensor.ops.rmsnorm import rms_norm_forward_kernel
from nbxtensor.ops.softmax import softmax_forward_kernel, softmax_forward_looped_kernel
from nbxtensor.ops.matmul import matmul_kernel, addmm_kernel
from nbxtensor.ops._autotune_policy import NBX_FORCE_FP32_ACCUM
from nbxtensor.ops.fused_moe import fused_moe_kernel, silu_and_mul_kernel, silu_mul_split_kernel
from nbxtensor.ops.dtype_convert import bf16_to_fp16_kernel
from nbxtensor.ops.embedding import embedding_kernel, EMBEDDING_OOB
from nbxtensor.ops.sum import sum_forward_kernel
from nbxtensor.ops.mean import mean_forward_kernel
from nbxtensor.ops.amax import amax_forward_kernel
from nbxtensor.ops.glu import glu_forward_kernel
from nbxtensor.ops.le import le_forward_kernel, le_scalar_kernel
from nbxtensor.ops.softplus import softplus_forward_kernel
from nbxtensor.ops.dropout import dropout_inference_kernel
from nbxtensor.ops.upsample_nearest2d import upsample_nearest2d_kernel
from nbxtensor.ops.groupnorm import group_norm_forward_kernel
from nbxtensor.ops.index_select import index_select_kernel, INDEX_SELECT_OOB
from nbxtensor.ops.triu import triu_kernel, triu_batch_kernel
from nbxtensor.ops.tril import tril_kernel, tril_batch_kernel
from nbxtensor.ops.argmax import argmax_kernel_1, argmax_kernel_2, argmax_kernel_inner
from nbxtensor.ops.argmin import argmin_kernel_1, argmin_kernel_2, argmin_kernel
from nbxtensor.ops.conv2d import conv2d_forward_kernel
from nbxtensor.ops.depthwise_conv2d import depthwise_conv2d_kernel
from nbxtensor.ops.batch_norm import batch_norm_forward_kernel
from nbxtensor.ops.cumsum import scan_part_sum_kernel, add_base_sum_kernel
from nbxtensor.ops.scatter_op import scatter_kernel, scatter_add_kernel, scatter_reduce_amax_kernel, scatter_reduce_amin_kernel
from nbxtensor.ops.gather_op import gather_kernel
from nbxtensor.ops.avg_pool2d import avg_pool2d_forward_kernel
from nbxtensor.ops.max_pool2d import max_pool2d_forward_kernel
from nbxtensor.ops.max_pool3d import max_pool3d_forward_kernel
from nbxtensor.ops.cross_entropy import cross_entropy_loss_forward_kernel
from nbxtensor.ops.exp2 import exp2_forward_kernel
from nbxtensor.ops.tan import tan_forward_kernel
from nbxtensor.ops.celu import celu_forward_kernel
from nbxtensor.ops.log_sigmoid import log_sigmoid_forward_kernel
from nbxtensor.ops.isfinite import isfinite_forward_kernel
from nbxtensor.ops.isinf import isinf_forward_kernel
from nbxtensor.ops.isnan import isnan_forward_kernel
from nbxtensor.ops.nan_to_num import nan_to_num_forward_kernel
from nbxtensor.ops.threshold import threshold_forward_kernel
from nbxtensor.ops.floor import floor_forward_kernel, ceil_forward_kernel, round_forward_kernel, trunc_forward_kernel
from nbxtensor.ops.addcdiv import addcdiv_forward_kernel
from nbxtensor.ops.addcmul import addcmul_forward_kernel
from nbxtensor.ops.all_reduce import all_kernel_mid, all_kernel_result, all_kernel_dim
from nbxtensor.ops.any_reduce import any_kernel_mid, any_kernel_result, any_kernel_dim
from nbxtensor.ops.bitwise_and import bitwise_and_forward_kernel
from nbxtensor.ops.bitwise_or import bitwise_or_forward_kernel
from nbxtensor.ops.bitwise_not import bitwise_not_forward_kernel
from nbxtensor.ops.logical_and import logical_and_forward_kernel
from nbxtensor.ops.logical_or import logical_or_forward_kernel
from nbxtensor.ops.logical_not import logical_not_forward_kernel
from nbxtensor.ops.conv1d import conv1d_forward_kernel
from nbxtensor.ops.lerp import lerp_tensor_forward_kernel, lerp_scalar_head_kernel, lerp_scalar_tail_kernel
from nbxtensor.ops.dot_op import dot_kernel_small, dot_kernel_partial, dot_kernel_reduce
from nbxtensor.ops.flip_op import flip_1d_kernel, flip_strided_kernel
from nbxtensor.ops.cat_op import cat_copy_kernel_4
from nbxtensor.ops.fill_op import fill_kernel
from nbxtensor.ops.arange_op import arange_kernel
from nbxtensor.ops.prod import prod_kernel_mid, prod_kernel_result, prod_kernel
from nbxtensor.ops.min_reduce import min_kernel_mid, min_kernel_result, min_kernel
from nbxtensor.ops.max_reduce import max_kernel_mid, max_kernel_result, max_kernel
from nbxtensor.ops.baddbmm_op import baddbmm_kernel
from nbxtensor.ops.addmv_op import addmv_kernel
from nbxtensor.ops.mse_loss import mse_loss_partial_kernel, mse_loss_reduce_kernel, mse_loss_none_kernel
from nbxtensor.ops.nllloss import nll_loss_forward_kernel
from nbxtensor.ops.std import std_map_kernel, std_reduce_kernel, std_dim_kernel
from nbxtensor.ops.var import var_kernel_1, var_kernel_2, var_welford_kernel
from nbxtensor.ops.index_add import index_add_gather_kernel
from nbxtensor.ops.index_put_op import index_put_kernel, INDEX_PUT_OOB
from nbxtensor.ops.sort_op import (radix_sort_histogram_kernel,
                          radix_sort_scatter_kernel,
                          radix_sort_tile_counts_kernel,
                          radix_sort_tile_prefix_kernel)
from nbxtensor.ops.rope import rope_forward_kernel
from nbxtensor.ops.pixel_shuffle import (
    pixel_shuffle_kernel,
    pixel_shuffle_broadcast_aware_kernel,
)
from nbxtensor.ops.pixel_unshuffle import pixel_unshuffle_kernel
from nbxtensor.ops.upsample_bilinear2d import upsample_bilinear2d_kernel
from nbxtensor.ops.adaptive_avg_pool2d import adaptive_avg_pool2d_kernel
from nbxtensor.ops.conv_depthwise2d import conv_depthwise2d_kernel
from nbxtensor.ops.scaled_softmax import scaled_softmax_kernel
from nbxtensor.ops.vector_norm import (
    l2_norm_kernel, l1_norm_kernel, linf_norm_kernel, l0_norm_kernel, lp_norm_kernel,
    l2_norm_pass1_kernel, l2_norm_pass2_kernel,
    linf_norm_pass1_kernel, linf_norm_pass2_kernel,
)
from nbxtensor.ops.addr import addr_kernel
from nbxtensor.ops.logical_xor import logical_xor_kernel
from nbxtensor.ops.bitwise_left_shift import bitwise_left_shift_kernel
from nbxtensor.ops.bitwise_right_shift import bitwise_right_shift_kernel
from nbxtensor.ops.rand_op import rand_kernel, randn_kernel
from nbxtensor.ops.clamp_min import clamp_min_forward_kernel
from nbxtensor.ops.conv_transpose2d import conv_transpose2d_kernel
from nbxtensor.ops.cummax import scan_part_max_kernel
from nbxtensor.ops.cummin import scan_part_min_kernel
from nbxtensor.ops.count_nonzero import count_nonzero_kernel
from nbxtensor.ops.trace_op import trace_kernel
from nbxtensor.ops.linspace_op import linspace_kernel
from nbxtensor.ops.full_op import full_kernel
from nbxtensor.ops.stack_op import stack_copy_kernel
from nbxtensor.ops.repeat_op import repeat_1d_kernel
from nbxtensor.ops.upsample_nearest3d import upsample_nearest3d_kernel
from nbxtensor.ops.weight_norm import weight_norm_kernel_first, weight_norm_kernel_last
from nbxtensor.ops.repeat_interleave import repeat_interleave_tensor_kernel
from nbxtensor.ops.elementwise_launch import ew_meta as _ew_meta, ew_grid as _ew_grid
from nbxtensor.ops import tiling_sizes as _ts
import math
from nbxtensor.ops import elementwise_wrappers as _ew
from nbxtensor.ops import wrappers as _core


def _clamp_reduce_tile(bm, bn):
    """(block_m, block_n) clamped so a 2-D reduce tile fits the backend's cap.

    A 2-D axis reduce stages BLOCK_M*BLOCK_N one element per thread; above the
    cap (Metal 1024, cuda/hip far higher, so inert there) the lowering refuses
    -- l2_norm on swin2SR at 8192, weight_norm on Kokoro at 16384. Shrinking
    BLOCK_N adds loop iterations and changes no result; BLOCK_M is kept because
    a caller may use it for its grid. THE clamp, one place: std/var/norm and
    weight_norm all call it, a twin copy being a fix that lands in only one.
    """
    cap = _reduce_tile_max()
    if bm * bn > cap:
        bn = max(1, cap // bm)
    return bm, bn


# The argmax row tile: a power-of-two cap on the vocabulary row a single
# program scans, and the warp count that scans it. 4096 with Triton's default
# 4 warps is what every architecture used before this became readable.
_ARGMAX_TILE = (4096, 4)


# The argmin-over-a-dim tile. Its kernel takes BLOCK_M rows and a BLOCK_N
# column cap; the adopted 4 x 4096 with 4 warps.
_ARGMIN_TILE = (4, 4096, 4)


def _argmin_tile() -> tuple:
    """`(BLOCK_M, BLOCK_N cap, num_warps)` from `block_sizes.argmin`.

    Companion to `_argmax_tile`; argmin's kernel carries a row block as well
    as a column cap, so it has its own entry rather than sharing one.
    """
    bm, bn, warps = _ARGMIN_TILE
    try:
        from nbxtensor.ops._configs import active_vendor_profile
        cfg = (active_vendor_profile().get("block_sizes", {}) or {}).get("argmin")
        if isinstance(cfg, dict):
            bm = int(cfg.get("block_m") or bm)
            bn = int(cfg.get("tile_n") or bn)
            warps = int(cfg.get("num_warps") or warps)
    except Exception:
        pass
    return bm, bn, warps


def _argmax_tile() -> tuple:
    """`(TILE_N cap, num_warps)` for the row-wise argmax, from
    `block_sizes.argmax` in config/vendors/<vendor>/<arch>.yml.

    argmax is a MULTI-VALUE reduce — it carries a value and its index — and a
    backend may need the whole tile to fit its threadgroup to aggregate the
    pair. That is a property of the hardware and its compiler, so it is
    written where such properties are written. Absent key / no profile -> the
    values every architecture used before, so nothing measured moves.
    """
    cap, warps = _ARGMAX_TILE
    try:
        from nbxtensor.ops._configs import active_vendor_profile
        cfg = (active_vendor_profile().get("block_sizes", {}) or {}).get("argmax")
        if isinstance(cfg, dict):
            cap = int(cfg.get("tile_n") or cap)
            warps = int(cfg.get("num_warps") or warps)
    except Exception:
        pass
    return cap, warps


# ===========================================================================
# REDUCTION WRAPPERS
# ===========================================================================

def _reduce_over_dims(single_fn, x, dims, keepdim, **kw):
    """Reduce ``x`` over MULTIPLE dims via one single-dim reduction.

    The per-reduction wrappers below each handle a single ``dim`` (or a 1-elem
    list) by movedim+flatten to 2D; a list of dims used to collapse to its
    first element (``dim = dim[0]``), silently reducing only one axis. This
    helper makes the multi-dim case correct: permute the reduced dims to the
    end, flatten them into ONE axis, reduce that axis with ``single_fn`` (the
    same wrapper, called with an int dim). Correct for separable reductions
    (sum/mean/amax) AND non-separable ones (std/var — the correction divides
    by the TOTAL reduced count, which the single flattened axis preserves).
    R33-pure: permute/contiguous/reshape + the wrapper's own kernel.
    """
    nd = x.ndim
    dims = sorted(int(d) % nd for d in dims)
    kept = [d for d in range(nd) if d not in dims]
    red = 1
    for d in dims:
        red *= x.shape[d]
    xp = x.permute(*(kept + dims)).contiguous()
    flat_shape = [x.shape[d] for d in kept] + [red]
    xf = xp.reshape(*flat_shape)                     # [...kept..., prod(reduced)]
    out = single_fn(xf, dim=len(kept), keepdim=False, **kw)
    if keepdim:
        full = list(x.shape)
        for d in dims:
            full[d] = 1
        out = out.reshape(*full)                     # kept dims keep original order
    return out


def mean_wrapper(x, dim=None, keepdim=False, dtype=None) :
    """Mean reduction. Reduces over one dim, a 1-elem list, or (via
    _reduce_over_dims) an arbitrary list of dims. `dtype`: aten::mean's keyword, the input cast to it first."""
    dtype = _core._reduction_dtype("mean", dtype)
    if dtype is not None and x._dtype != dtype:
        x = x.to(dtype)
    if isinstance(dim, (list, tuple)) and len(dim) > 1:
        return _reduce_over_dims(mean_wrapper, x, dim, keepdim)
    if dim is None:
        # Full reduction — flatten and reduce
        x_flat = x.contiguous().view(-1)
        batch_dim = 1
        feat_dim = x_flat.numel()
        x_2d = x_flat.unsqueeze(0)
    else:
        if isinstance(dim, (list, tuple)):
            dim = dim[0]
        dim = dim % x.ndim
        feat_dim = x.shape[dim]
        batch_dim = x.numel() // feat_dim
        x_perm = x.movedim(dim, -1).contiguous()
        x_2d = x_perm.reshape(batch_dim, feat_dim)

    output = NBXTensor.empty(batch_dim, dtype=NBXDtype.float32, device=x.device)
    # rows per program, the feature tile and the warps: the certified configuration of this key (chantier 1), the
    # key the derived census forms (`launch_keys.row_reduce_launches`)
    grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    set_device(x_2d)
    mean_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
    )

    if dim is None:
        return output.squeeze(0).to(x.dtype)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.to(x.dtype).view(shape)


def sum_wrapper(x, dim=None, keepdim=False, dtype=None) :
    """Sum reduction.

    Output dtype follows PyTorch ``torch.sum`` promotion: floating inputs keep
    their dtype; bool/integer inputs promote to int64. Casting the (fp32-
    accumulated) result back to a narrow input dtype silently overflows — e.g.
    summing a bool mask of 354 True elements and casting to uint8 wraps to
    354 % 256 = 98 (the chatterbox s3gen mel-length mask bug). Accumulation
    stays fp32 either way (exact for counts well under the 2^24 mantissa).

    A multi-dim reduction (``dim`` a list with >1 entry, e.g. the Wan RoPE
    grid-size sum over [0,1,2,4]) is reduced atomically via _reduce_over_dims;
    a single dim / 1-elem list keeps the fast path below unchanged.
    """
    dtype = _core._reduction_dtype("sum", dtype)
    if dtype is not None and x._dtype != dtype:
        x = x.to(dtype)
    if isinstance(dim, (list, tuple)) and len(dim) > 1:
        return _reduce_over_dims(sum_wrapper, x, dim, keepdim)
    # Complex sum = (sum real, sum imag), summed over the same dim. The flat
    # kernel would reduce the interleaved re/im storage on the wrong stride.
    # Route via the real view: the appended [.,2] axis is NOT reduced (dim is
    # resolved on the complex rank), so re/im are summed independently.
    if isinstance(x, NBXTensor) and x.is_complex():
        d = (dim[0] if isinstance(dim, (list, tuple)) else dim)
        d = d % x.ndim
        outr = sum_wrapper(x.view_as_real(), d, keepdim)
        return outr.view_as_complex()
    # x.dtype is a triton dtype handle (not NBXDtype) — gate on the tensor's
    # own is_floating_point() rather than dtype membership.
    out_dtype = x.dtype if x.is_floating_point() else NBXDtype.int64

    if dim is None:
        x_flat = x.contiguous().view(-1)
        batch_dim = 1
        feat_dim = x_flat.numel()
        x_2d = x_flat.unsqueeze(0)
    else:
        if isinstance(dim, (list, tuple)):
            dim = dim[0]
        dim = dim % x.ndim
        feat_dim = x.shape[dim]
        batch_dim = x.numel() // feat_dim
        x_perm = x.movedim(dim, -1).contiguous()
        x_2d = x_perm.reshape(batch_dim, feat_dim)

    output = NBXTensor.empty(batch_dim, dtype=NBXDtype.float32, device=x.device)
    # rows per program, the feature tile and the warps: the certified configuration of this key (chantier 1), the
    # key the derived census forms (`launch_keys.row_reduce_launches`)
    grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    set_device(x_2d)
    sum_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
    )

    if dim is None:
        return output.squeeze(0).to(out_dtype)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.to(out_dtype).view(shape)


def amax_wrapper(x, dim=None, keepdim=False) :
    """Max reduction (one dim, a 1-elem list, or an arbitrary list of dims)."""
    if isinstance(dim, (list, tuple)) and len(dim) > 1:
        return _reduce_over_dims(amax_wrapper, x, dim, keepdim)
    if dim is None:
        x_flat = x.contiguous().view(-1)
        batch_dim = 1
        feat_dim = x_flat.numel()
        x_2d = x_flat.unsqueeze(0)
    else:
        if isinstance(dim, (list, tuple)):
            dim = dim[0]
        dim = dim % x.ndim
        feat_dim = x.shape[dim]
        batch_dim = x.numel() // feat_dim
        x_perm = x.movedim(dim, -1).contiguous()
        x_2d = x_perm.reshape(batch_dim, feat_dim)

    output = NBXTensor.empty(batch_dim, dtype=x.dtype, device=x.device)
    # rows per program, the feature tile and the warps: the certified configuration of this key (chantier 1), the
    # key the derived census forms (`launch_keys.row_reduce_launches`)
    grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    set_device(x_2d)
    amax_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
    )

    if dim is None:
        return output.squeeze(0)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.view(shape)


def argmax_wrapper(x, dim=None, keepdim=False) :
    """Argmax reduction. Wrapper from FlagGems."""
    import math
    if dim is None:
        M = x.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = NBXTensor.empty((mid_size,), dtype=x.dtype, device=x.device)
        mid_index = NBXTensor.empty((mid_size,), dtype=NBXDtype.int64, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.int64, device=x.device)

        set_device(x.contiguous())
        argmax_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
        set_device(mid_value)
        argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        dim = dim % x.ndim
        N = x.shape[dim]
        M = x.numel() // N
        x = x.contiguous()
        shape_list = list(x.shape)
        shape_list[dim] = 1
        out_index = NBXTensor.empty(shape_list, dtype=NBXDtype.int64, device=x.device)
        if not keepdim:
            out_index = out_index.squeeze(dim)

        cap, warps = _argmax_tile()
        TILE_N = min(triton.next_power_of_2(N), cap)
        grid = (M, 1, 1)
        set_device(x)
        argmax_kernel_inner[grid](x, out_index, M, N, TILE_N=TILE_N,
                                  ONE_TILE_PER_CTA=(TILE_N >= N),
                                  num_warps=warps)
        return out_index


def argmin_wrapper(x, dim=None, keepdim=False) :
    """Argmin reduction. Wrapper from FlagGems (adapted from argmax)."""
    import math
    if dim is None:
        M = x.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = NBXTensor.empty((mid_size,), dtype=x.dtype, device=x.device)
        mid_index = NBXTensor.empty((mid_size,), dtype=NBXDtype.int64, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.int64, device=x.device)

        set_device(x.contiguous())
        argmin_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
        set_device(mid_value)
        argmin_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        dim = dim % x.ndim
        N = x.shape[dim]
        M = x.numel() // N
        x = x.contiguous()
        shape = x.shape
        N = shape[dim]
        M = 1
        for i in range(dim):
            M *= shape[i]
        K = x.numel() // M // N

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = NBXTensor.empty(shape_list, dtype=NBXDtype.int64, device=x.device)
        if not keepdim:
            out_index = out_index.squeeze(dim)

        _amin_bm, _amin_cap, _amin_warps = _argmin_tile()
        BLOCK_N = min(_amin_cap, triton.next_power_of_2(N))
        BLOCK_M = _amin_bm
        grid = (triton.cdiv(M, BLOCK_M), K)
        set_device(x)
        argmin_kernel[grid](x, out_index, M, N, K, BLOCK_M, BLOCK_N,
                            num_warps=_amin_warps)
        return out_index


# ===========================================================================
# CUMSUM WRAPPER — Extracted from FlagGems (1D scan-then-fan)
# ===========================================================================

def cumsum_wrapper(x, dim: int = 0, dtype=None) :
    """Cumulative sum along dimension. Wrapper from FlagGems.

    `dtype` mirrors aten::cumsum's dtype kwarg (vendor cu_seqlens passes
    int32): per ATen semantics the INPUT is cast to `dtype` BEFORE
    accumulation. Accepted as string or NBXDtype; unknown strings raise
    (ZERO FALLBACK)."""
    _out_dtype = dtype
    if isinstance(_out_dtype, str):
        _name = _out_dtype.replace("torch.", "")
        _out_dtype = getattr(NBXDtype, _name, None)
        if _out_dtype is None:
            raise RuntimeError(
                f"ZERO FALLBACK: cumsum dtype '{dtype}' is not a known "
                f"NBXDtype.")
    if _out_dtype is not None and x.dtype != _out_dtype:
        x = x.to(_out_dtype)
    dim = dim % x.ndim

    # Reshape to [batch, scan_dim] where scan_dim is the cumsum axis
    scan_size = x.shape[dim]
    batch_size = x.numel() // scan_size

    # Move cumsum dim to last, flatten batch dims
    x_perm = x.movedim(dim, -1).contiguous()
    x_2d = x_perm.reshape(batch_size, scan_size)

    BLOCK_SIZE = min(triton.next_power_of_2(scan_size), 4096)

    # Process each row independently
    output_2d = NBXTensor.empty_like(x_2d)
    for row in range(batch_size):
        row_in = x_2d[row]
        row_out = output_2d[row]
        part_num = triton.cdiv(scan_size, BLOCK_SIZE)
        partial_sum = NBXTensor.empty(part_num, dtype=NBXDtype.float32, device=x.device)

        set_device(row_in)
        scan_part_sum_kernel[(part_num,)](
            row_in, row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

        if part_num > 1:
            for i in range(1, part_num):
                partial_sum[i] += partial_sum[i - 1]
            set_device(row_out)
            add_base_sum_kernel[(part_num,)](
                row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

    result = output_2d.view_as(x_perm).movedim(-1, dim)
    if _out_dtype is not None and result.dtype != _out_dtype:
        result = result.to(_out_dtype)
    return result


# ---------------------------------------------------------------------------
# All / Any (boolean reductions)
# ---------------------------------------------------------------------------

def all_wrapper(x, dim=None, keepdim=False) :
    """Test if all elements are non-zero. Two-pass global reduction."""
    import math
    x = x.contiguous()
    if dim is None:
        M = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        mid = NBXTensor.empty(mid_size, dtype=NBXDtype.bool_, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.bool_, device=x.device)
        set_device(x)
        all_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
        set_device(mid)
        all_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=NBXDtype.bool_, device=x.device)
        BLOCK_N = min(4096, triton.next_power_of_2(N))
        _red_bm, _red_bn, _red_warps = _ew._reduction_2d_tile()
        grid = (triton.cdiv(M, _red_bm),)
        set_device(x_perm)
        all_kernel_dim[grid](x_perm, out, M, N, BLOCK_M=_red_bm, BLOCK_N=BLOCK_N,
                             num_warps=_red_warps)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


def any_wrapper(x, dim=None, keepdim=False) :
    """Test if any element is non-zero. Two-pass global reduction."""
    import math
    x = x.contiguous()
    if dim is None:
        M = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        mid = NBXTensor.empty(mid_size, dtype=NBXDtype.bool_, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.bool_, device=x.device)
        set_device(x)
        any_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
        set_device(mid)
        any_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=NBXDtype.bool_, device=x.device)
        BLOCK_N = min(4096, triton.next_power_of_2(N))
        _red_bm, _red_bn, _red_warps = _ew._reduction_2d_tile()
        grid = (triton.cdiv(M, _red_bm),)
        set_device(x_perm)
        any_kernel_dim[grid](x_perm, out, M, N, BLOCK_M=_red_bm, BLOCK_N=BLOCK_N,
                             num_warps=_red_warps)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


# ---------------------------------------------------------------------------
# Prod (product reduction)
# ---------------------------------------------------------------------------

def prod_wrapper(x, dim=None, keepdim=False) :
    """Product reduction. Two-pass global or dim-specific."""
    import math
    x = x.contiguous()
    if dim is None:
        M = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        mid = NBXTensor.empty(mid_size, dtype=x.dtype, device=x.device)
        out = NBXTensor.empty([], dtype=x.dtype, device=x.device)
        set_device(x.view(-1))
        prod_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        set_device(mid)
        prod_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        _red_bm, _red_bn, _red_warps = _ew._reduction_2d_tile()
        grid = (triton.cdiv(M, _red_bm),)
        set_device(x_perm)
        prod_kernel[grid](x_perm, out, M, N,
                          BLOCK_M=_red_bm, BLOCK_N=_red_bn,
                          num_warps=_red_warps)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


# ---------------------------------------------------------------------------
# Std / Var
# ---------------------------------------------------------------------------

def std_wrapper(x, dim=None, correction=1, keepdim=False) :
    """Standard deviation. Global (map-reduce), single-dim, or multi-dim (the
    reduced dims are flattened to one axis so the correction divides by the
    total reduced count — matching torch.std over a list of dims)."""
    import math
    x = x.contiguous()
    if isinstance(dim, (list, tuple)) and len(dim) > 1:
        return _reduce_over_dims(std_wrapper, x, dim, keepdim, correction=correction)
    if dim is None:
        N = x.numel()
        BLOCK_N = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), 4096)
        num_blocks = triton.cdiv(N, BLOCK_N)
        tmp_sum = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        tmp_sum_sq = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        set_device(x.view(-1))
        std_map_kernel[(num_blocks,)](x.view(-1), tmp_sum, tmp_sum_sq, N, BLOCK_N=BLOCK_N)
        BLOCK_SIZE = triton.next_power_of_2(num_blocks)
        set_device(tmp_sum)
        std_reduce_kernel[(1,)](
            tmp_sum, tmp_sum_sq, out, N, float(correction), num_blocks,
            BLOCK_SIZE=BLOCK_SIZE)
        return out.to(x.dtype)
    else:
        if isinstance(dim, (list, tuple)):
            dim = dim[0]
        dim = dim % x.ndim
        N = x.shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous()
        x_2d = x_perm.reshape(M, N)
        out_flat = NBXTensor.empty(M, dtype=NBXDtype.float32, device=x.device)
        grid, _bm, _bn = _ew._reduction_tile(M)
        set_device(x_2d)
        std_dim_kernel[grid](
            x_2d, out_flat,
            x_2d.stride(0), x_2d.stride(1),
            M, N, float(correction),
            BLOCK_M=_bm, BLOCK_N=_bn,
            num_warps=4,
        )
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return out_flat.to(x.dtype).view(shape)


def var_wrapper(x, dim=None, correction=1, keepdim=False) :
    """Variance. Global (Welford), single-dim, or multi-dim (reduced dims
    flattened to one axis → correction over the total reduced count)."""
    import math
    x = x.contiguous()
    if isinstance(dim, (list, tuple)) and len(dim) > 1:
        return _reduce_over_dims(var_wrapper, x, dim, keepdim, correction=correction)
    if dim is None:
        N = x.numel()
        BLOCK_N = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), 4096)
        num_blocks = triton.cdiv(N, BLOCK_N)
        acc = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        average = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        count = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        set_device(x.view(-1))
        var_kernel_1[(num_blocks,)](x.view(-1), acc, average, count, N, BLOCK_N=BLOCK_N)
        BLOCK_NUM = triton.next_power_of_2(num_blocks)
        set_device(acc)
        var_kernel_2[(1,)](
            acc, average, count, out, N, float(correction), num_blocks,
            BLOCK_N=BLOCK_NUM)
        return out.to(x.dtype)
    else:
        if isinstance(dim, (list, tuple)):
            dim = dim[0]
        dim = dim % x.ndim
        N = x.shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous()
        x_2d = x_perm.reshape(M, N)
        out_flat = NBXTensor.empty(M, dtype=NBXDtype.float32, device=x.device)
        grid, _bm, _bn = _ew._reduction_tile(M)
        set_device(x_2d)
        var_welford_kernel[grid](x_2d, out_flat, M, N, float(correction),
                                 BLOCK_M=_bm, BLOCK_N=_bn,
                                 num_warps=4)
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return out_flat.to(x.dtype).view(shape)


def scatter_reduce_wrapper(input, dim: int, index,
                           src, reduce: str,
                           include_self: bool = True) :
    """scatter_reduce: accumulate src into input along dim using reduce mode.
    Supported modes: sum, amax, amin. All use Triton atomic ops."""
    input = input.contiguous()
    src = src.contiguous()
    index = index.contiguous()
    dim = dim % input.ndim

    if include_self:
        out = input.clone()
    else:
        if reduce == "sum" or reduce == "add":
            out = NBXTensor.zeros_like(input)
        elif reduce == "amax":
            out = NBXTensor.empty_like(input, float('-inf'))
        elif reduce == "amin":
            out = NBXTensor.empty_like(input, float('inf'))
        else:
            out = input.clone()

    N = index.numel()
    if N == 0:
        return out

    outer_size = 1
    for i in range(dim):
        outer_size *= index.shape[i]
    inner_size = 1
    for i in range(dim + 1, index.ndim):
        inner_size *= index.shape[i]
    dim_size = index.shape[dim]

    stride_args = (
        src, index, out, N,
        out.stride(dim),
        src.stride(0) if dim > 0 else 0, src.stride(dim), src.stride(-1) if dim < src.ndim - 1 else 0,
        index.stride(0) if dim > 0 else 0, index.stride(dim), index.stride(-1) if dim < index.ndim - 1 else 0,
        out.stride(0) if dim > 0 else 0, out.stride(-1) if dim < out.ndim - 1 else 0,
        outer_size, dim_size, inner_size,
    )

    set_device(out)
    if reduce == "sum" or reduce == "add":
        scatter_add_kernel[_ew._1d_grid(N)](*stride_args, **_ew_meta(N))
    elif reduce == "amax":
        scatter_reduce_amax_kernel[_ew._1d_grid(N)](*stride_args, **_ew_meta(N))
    elif reduce == "amin":
        scatter_reduce_amin_kernel[_ew._1d_grid(N)](*stride_args, **_ew_meta(N))
    else:
        raise RuntimeError(f"[--triton] scatter_reduce mode '{reduce}' not implemented")

    return out
