"""Normalisation wrappers: layer, RMS, group and batch norm, weight norm, vector norms. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

from __future__ import annotations

import os
import triton
from nbxtensor.ops.autotune_bucket import bucket_of as _bucket_of
from nbxtensor.ops import launch_keys as _lk
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.backend.runtime import _set_device
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
from nbxtensor.ops import random_wrappers as _rnd
from nbxtensor.ops import reduction_wrappers as _red
from nbxtensor.ops import wrappers as _core
_WNORM_BM, _WNORM_BN = 64, 256           # weight_norm


# ===========================================================================
# NORMALIZATION WRAPPERS
# ===========================================================================

def native_layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    """LayerNorm matching ATen native_layer_norm signature.

    Returns (output, mean, rstd).
    """
    # Flatten to [batch, feat] for the kernel
    feat_dim = 1
    for s in normalized_shape:
        feat_dim *= s
    batch_dim = x.numel() // feat_dim

    x_2d = x.contiguous().view(batch_dim, feat_dim)
    output_2d = NBXTensor.empty_like(x_2d)
    mean = NBXTensor.empty(batch_dim, dtype=NBXDtype.float32, device=x.device)
    inv_std = NBXTensor.empty(batch_dim, dtype=NBXDtype.float32, device=x.device)

    has_weight = weight is not None
    has_bias = bias is not None and has_weight

    # rows per program and warps: the certified configuration of this key (workstream 1), the rows' bucket from the
    # launcher's own `bucket_of` — the key the derived census forms (`launch_keys.layer_norm_launches`)
    grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    _set_device(x_2d)
    layer_norm_forward_kernel[grid](
        x_2d, weight if has_weight else x_2d, bias if has_bias else x_2d,
        mean, inv_std, output_2d,
        batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        eps,
        scale_by_weight=has_weight,
        add_bias=has_bias,
        save_stats=True,
    )
    return output_2d.view_as(x), mean, inv_std


def layer_norm_wrapper(x, normalized_shape, weight=None, bias=None, eps=1e-5,
                       cudnn_enable=True) :
    """aten::layer_norm — the high-level op returns ONLY the normalized output
    (native_layer_norm also returns mean/rstd). The triton runtime stores a
    tuple result as out_0=the-whole-tuple when the graph declares a single
    output, so a downstream op (dropout) then receives a tuple. Return just the
    output tensor here."""
    out = native_layer_norm(x, normalized_shape, weight, bias, eps)
    return out[0] if isinstance(out, (tuple, list)) else out


def rms_norm(x, weight, eps=1e-6, epsilon=None, out_dtype=None):
    """RMSNorm wrapper. `out_dtype`: the dtype the output is stored in (default: x's).

    The kernel widens every input load to fp32 and stores in the output buffer's dtype, so a
    half input needs no materialised fp32 copy before the call and an fp32 output no cast
    after it: the dtype engine's fp32-internal wrap asks for the output dtype instead (the
    copy lever, 2026-09-07 — 48 + 48 copies a token on TinyLlama). The rounding at the store
    is the one the materialised path applied at its cast back.

    When `x.contiguous()` materializes a new tensor (input is a strided
    view such as the NHWC permute pattern in DC-AE VAEs), the rms_norm
    kernel can safely write its output back into that fresh contiguous
    buffer in place: each program block reads its full BLOCK_SIZE_BATCH
    x BLOCK_SIZE_FEAT tile via `tl.load` before any `tl.store`, so
    input==output is per-tile safe and no other Python consumer sees
    the materialized copy. This saves an 8 GiB transient on the
    Sana 4Kpx VAE rms_norm::24 (post-pixel_shuffle NHWC chain).
    """
    if epsilon is not None:
        eps = epsilon
    x = _core._ensure_cuda(x)
    weight = _core._ensure_cuda(weight)
    feat_dim = x.shape[-1]
    batch_dim = x.numel() // feat_dim

    x_contig = x.contiguous()
    x_2d = x_contig.view(batch_dim, feat_dim)
    out_dt = out_dtype if out_dtype is not None else x_2d.nbx_dtype
    if x_contig is not x and out_dt == x_2d.nbx_dtype:
        # contiguous() allocated a fresh buffer with no other holder —
        # write rms_norm output directly into it (in-place) instead of
        # paying for a second 8 GiB allocation.
        output_2d = x_2d
    else:
        output_2d = NBXTensor.empty(x_2d.shape, dtype=out_dt, device=x_2d.device)

    has_weight = weight is not None

    # rows per program and warps: the certified configuration of this key (workstream 1) — the key the derived census
    # forms (`launch_keys.rms_norm_launches`)
    grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    rms_norm_forward_kernel[grid](
        x_2d, weight if has_weight else x_2d,
        output_2d,
        batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        eps,
        scale_by_weight=has_weight,
        SATURATE_F16=_nbx_saturating_cast(NBXDtype.float32, out_dt),
    )
    return output_2d.view_as(x)


# NOT `_nbx_widens_on_load`: the kernel widening an fp16 INPUT on load is another
# compilation than the fp32 input the certified rows ran it with — 12–15 % of the
# elements differ at T5's shapes (PixArt-XL-2, the machine-set gate of 2026-09-08),
# invisible on TinyLlama whose stream reaches rms_norm in fp32. The dtype engine's
# wrap pre-casts as it always did; `out_dtype` stays for callers that store fp32.
rms_norm._nbx_widens_on_load = False


def norm_wrapper(x, p=2, dim=None, keepdim=False):
    """aten::norm(self, p, dim, keepdim) — Lp norm reduction over ``dim``.

    The DAC codec's only norm is weight-norm: ``||v|| = sqrt(sum(v*v, dim))``
    with p=2 over a single dim. Composed from existing triton kernels
    (mul + sum + sqrt) — R33-pure, no new @triton.jit needed; sum_wrapper
    accumulates in fp32 so the reduction over a long row does not overflow fp16.
    Other p raise ZERO-FALLBACK (named follow-up) rather than mis-computing.
    """
    pv = _core._to_scalar(p) if isinstance(p, NBXTensor) else (2 if p is None else p)
    if float(pv) != 2.0:
        raise NotImplementedError(
            f"aten::norm p={pv} unwired (DAC weight-norm uses p=2) — "
            "follow-up P-TRITON-NORM-GENERAL-P.")
    return _ew.sqrt_wrapper(_red.sum_wrapper(_ew.mul(x, x), dim=dim, keepdim=keepdim))


def group_norm_wrapper(x, num_groups: int, weight, bias, eps=1e-5):
    """GroupNorm. Returns (output, mean, rstd) matching ATen native_group_norm.

    Friendly signature (x, num_groups, weight, bias, eps) for direct callers.
    The graph-level `aten::native_group_norm` op uses a longer signature
    (input, weight, bias, N, C, HxW, num_groups, eps) — see
    `native_group_norm_wrapper` below for the adapter.
    """
    x = x.contiguous()
    N = x.shape[0]
    C = x.shape[1]
    HxW = x.numel() // (N * C)
    group_size = C // num_groups

    y = NBXTensor.empty_like(x)
    mean = NBXTensor.empty((N, num_groups), dtype=NBXDtype.float32, device=x.device)
    rstd = NBXTensor.empty((N, num_groups), dtype=NBXDtype.float32, device=x.device)

    grid = (N * num_groups,)
    _set_device(x)
    # the chunk along hidden = group_size * HxW and the warps: the certified configuration of this key (chantier 1),
    # the key the derived census forms (`launch_keys.group_norm_launches`)
    hidden = group_size * HxW
    group_norm_forward_kernel[grid](
        x, y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        mean, rstd,
        group_size, C, HxW, num_groups, eps,
        _lk.bucket_of("M", N * num_groups), _lk.softmax_feat_bucket(hidden),
        scale_by_weight=(weight is not None),
        add_bias=(bias is not None),
    )
    return y, mean, rstd


def native_group_norm_wrapper(input, weight, bias, N, C, HxW, num_groups, eps):
    """Adapter for ATen's low-level `aten::native_group_norm`.

    The op is dispatched with 8 positional args (input, weight, bias, N, C,
    HxW, num_groups, eps). We recompute N/C/HxW inside the kernel wrapper
    from `input.shape`, so we just forward the friendly call — dropping the
    redundant N/C/HxW scalars.
    """
    return group_norm_wrapper(input, int(num_groups), weight, bias, float(eps))


# ===========================================================================
# BATCH NORM WRAPPER — Extracted from FlagGems
# ===========================================================================

def batch_norm_wrapper(
    x, weight, bias,
    running_mean, running_var,
    training: bool = False, momentum: float = 0.1, eps: float = 1e-5,
) :
    """BatchNorm / InstanceNorm forward (kernel from FlagGems).

    training=False → inference batch_norm using running_mean/running_var.
    training=True  → statistics computed from the input — covers batch_norm
                     training AND instance_norm / AdaIN, which the tracer emits
                     as cudnn_batch_norm(training=True) with no affine and no
                     running buffers (it reshapes [N,C,L]→[1,N*C,L] first, so
                     each instance-channel normalises over the spatial axis).
    Absent weight/bias/running reach the kernel as null pointers, handled by its
    `if *_pointer:` guards (weight→1, bias→0, running update skipped). Never
    substitute the input tensor for an absent buffer — that silently corrupted
    instance_norm (kernel read x as the affine scale and overwrote x with the
    momentum-blended running stats).
    """
    if not training and running_mean is None:
        raise ValueError(
            "batch_norm eval mode (training=False) requires running_mean and "
            "running_var; got None. For statistics-from-input use training=True."
        )
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    elif x.ndim >= 4:
        x = x.flatten(2, -1)
    # x is now [N, C, spatial_dim]
    N, C, spatial = x.shape

    output = NBXTensor.empty_like(x)
    mean_out = NBXTensor.empty(C, dtype=NBXDtype.float32, device=x.device)
    inv_std_out = NBXTensor.empty(C, dtype=NBXDtype.float32, device=x.device)

    grid = (C,)
    xc = x.contiguous()
    _set_device(xc)
    # Pass null pointers (None) — NOT x — for absent weight/bias/running. The
    # kernel guards each with `if *_pointer:`; substituting x defeated the guard
    # (kernel read x as the affine scale/shift and wrote running stats back into
    # the input), which silently corrupted instance_norm (AdaIN has no affine
    # nor running stats). batch_norm callers still pass real buffers unchanged.
    batch_norm_forward_kernel[grid](
        xc,
        weight.contiguous() if weight is not None else None,
        bias.contiguous() if bias is not None else None,
        mean_out, inv_std_out, output,
        running_mean if running_mean is not None else None,
        running_var if running_var is not None else None,
        N, spatial,
        xc.stride(0), xc.stride(1), xc.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        momentum, eps,
        is_train=training,
    )
    return output


def vector_norm_wrapper(x, ord: float = 2.0, dim=None, keepdim=False) :
    """Vector norm — dispatches by ord to L0/L1/L2/Linf/Lp kernels. Triton kernel compute."""
    import math as _math

    if dim is None:
        dim_list = list(range(x.ndim))
    elif isinstance(dim, int):
        dim_list = [dim % x.ndim]
    else:
        dim_list = [d % x.ndim for d in dim]

    # Compute M (outer), N (reduction), shape manipulation
    shape = list(x.shape)
    N = 1
    for d in dim_list:
        N *= shape[d]
    M = x.numel() // N

    # Move reduction dims to last, flatten
    # Build permutation: non-reduced dims first, reduced dims last
    non_reduced = [i for i in range(x.ndim) if i not in dim_list]
    perm = non_reduced + dim_list
    x_perm = x.permute(perm).contiguous()
    x_2d = x_perm.reshape(M, N)

    out_shape = [shape[i] for i in range(x.ndim)]
    for d in dim_list:
        out_shape[d] = 1
    out = NBXTensor.empty(M, dtype=NBXDtype.float32, device=x.device)

    grid, _bm, _bn = _ew._reduction_tile(M)

    if ord == 2 or ord == 2.0:
        _set_device(x_2d)
        l2_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_bm, BLOCK_N=_bn, num_warps=4)
    elif ord == float('inf'):
        _set_device(x_2d)
        linf_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_bm, BLOCK_N=_bn, num_warps=4)
    elif ord == 0 or ord == 0.0:
        _set_device(x_2d)
        l0_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_bm, BLOCK_N=_bn, num_warps=4)
    elif ord == 1 or ord == 1.0:
        _set_device(x_2d)
        l1_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_bm, BLOCK_N=_bn, num_warps=4)
    else:
        _set_device(x_2d)
        lp_norm_kernel[grid](x_2d, out, M, N, ord, BLOCK_M=_bm, BLOCK_N=_bn, num_warps=4)

    result = out.to(x.dtype).view(out_shape)
    if not keepdim:
        for d in sorted(dim_list, reverse=True):
            result = result.squeeze(d)
    return result


def weight_norm_interface_wrapper(
    v, g, dim: int = 0
) -> tuple:
    """Weight normalization: w = g * v / ||v||.

    Returns (output, norm) matching aten::_weight_norm_interface.
    """
    v = v.contiguous()
    g = g.contiguous()
    output = NBXTensor.empty_like(v)
    norm = NBXTensor.empty_like(g)
    eps = 1e-45

    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        _wbm, _wbn = _red._clamp_reduce_tile(_WNORM_BM, _WNORM_BN)
        grid = (triton.cdiv(M, _wbm),)
        _set_device(output)
        weight_norm_kernel_first[grid](output, norm, v, g, M, N, eps,
                                       BLOCK_M=_wbm, BLOCK_N=_wbn,
                                       num_warps=4)
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        _wbm, _wbn = _red._clamp_reduce_tile(_WNORM_BM, _WNORM_BN)
        grid = (triton.cdiv(N, _wbn),)
        _set_device(output)
        weight_norm_kernel_last[grid](output, norm, v, g, M, N, eps,
                                      BLOCK_M=_wbm, BLOCK_N=_wbn,
                                      num_warps=4)
    else:
        raise ValueError(f"weight_norm only supports dim=0 or dim={v.ndim - 1}, got {dim}")

    return output, norm


def normal_wrapper(*args, **kwargs) :
    """normal(mean, std, size) → N(mean, std) via Triton randn + transform."""
    from nbxtensor.ops.rand_op import randn_kernel
    # ATen: normal(float mean, float std, SymInt[] size, ...)
    mean = float(args[0]) if len(args) > 0 else kwargs.get('mean', 0.0)
    std = float(args[1]) if len(args) > 1 else kwargs.get('std', 1.0)
    size = args[2] if len(args) > 2 else kwargs.get('size', [])
    if isinstance(mean, NBXTensor):
        mean = mean.item()
    if isinstance(std, NBXTensor):
        std = std.item()
    dtype = kwargs.get('dtype', NBXDtype.float32)
    device = kwargs.get('device', None)
    if device is None:
        device = str('cuda')
    output = _ew._stream_draw("normal", size, dtype, device)
    if output is None:
        output = NBXTensor.empty(size, dtype=dtype, device=device)
        n = output.numel()
        if n == 0:
            return output
        seed = _rnd._draw_seed()
        _set_device(output)
        randn_kernel[_ew._1d_grid(n)](output, n, seed, **_ew_meta(n))
    # Transform N(0,1) → N(mean, std)
    if std != 1.0:
        output.mul_(std)
    if mean != 0.0:
        output.add_(mean)
    return output


def group_norm_statistics(x, num_groups: int, eps: float):
    """(mean, rstd) [N, G] fp32 of `x` [N, C, ...] over each group — the first pass of an exact tiled group norm
    (`core/strategies/tiled_norm.py`), over the WHOLE input, without its normalised tensor."""
    from nbxtensor.ops.group_norm_tiled import group_norm_stats_kernel, TILED_NORM_BLOCK, TILED_NORM_WARPS
    x = x.contiguous()
    N, C = int(x.shape[0]), int(x.shape[1])
    if C % int(num_groups):
        raise ValueError(f"tiled group norm: {C} channels in {num_groups} groups")
    HW = x.numel() // (N * C)
    mean = NBXTensor.empty((N, int(num_groups)), dtype=NBXDtype.float32, device=x.device)
    rstd = NBXTensor.empty((N, int(num_groups)), dtype=NBXDtype.float32, device=x.device)
    _set_device(x)
    group_norm_stats_kernel[(N * int(num_groups),)](x, mean, rstd, C // int(num_groups), C, HW, int(num_groups),
                                                     float(eps), BLOCK_SIZE=TILED_NORM_BLOCK, num_warps=TILED_NORM_WARPS)
    return mean, rstd


def group_norm_apply(x, weight, bias, num_groups: int, mean, rstd):
    """`aten::native_group_norm`'s three outputs for a slice `x` given the whole input's statistics — the second pass;
    the output in `x`'s dtype, as the engine's group norm writes it (`group_norm_wrapper`)."""
    from nbxtensor.ops.group_norm_tiled import group_norm_apply_kernel, TILED_NORM_BLOCK, TILED_NORM_WARPS
    x = x.contiguous()
    N, C = int(x.shape[0]), int(x.shape[1])
    HW = x.numel() // (N * C)
    if mean._device != "cuda" or getattr(mean, "_device_idx", None) != getattr(x, "_device_idx", None):
        mean, rstd = mean.to_cuda(x._device_idx), rstd.to_cuda(x._device_idx)
    y = NBXTensor.empty_like(x)
    _set_device(x)
    group_norm_apply_kernel[(N * int(num_groups),)](
        x, y, weight if weight is not None else x, bias if bias is not None else x, mean, rstd,
        C // int(num_groups), C, HW, int(num_groups), scale_by_weight=weight is not None, add_bias=bias is not None,
        BLOCK_SIZE=TILED_NORM_BLOCK, num_warps=TILED_NORM_WARPS)
    return y, mean, rstd
