"""Element-wise wrappers: activations, unary and binary arithmetic, comparisons, bitwise and logical ops, interpolation, the device bf16->fp16 conversion. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import attention_wrappers as _attn
from nbxtensor.ops import index_wrappers as _idx
from nbxtensor.ops import linalg_wrappers as _la
from nbxtensor.ops import random_wrappers as _rnd
from nbxtensor.ops import reduction_wrappers as _red
from nbxtensor.ops import shape_wrappers as _shp
from nbxtensor.ops import wrappers as _core


# A run of a Triton sequence (both engines say so at entry, `begin_run`): the
# caches that live for one run empty here. `_ROPE_TABLES`: the step's rotary
# tables widened ONCE per run — the graph hands every layer its own view of one
# base tensor, and each layer widened it again (2 × the layer count a token).
# Keyed by the base object's identity, held strongly for the run so the id
# cannot be reused; a later view of another base is another entry; nothing
# outlives the run (the arena may reuse the base's slot after its last read,
# and no consumer reads a view of it after that by construction).
_RUN_EPOCH = 0
_ROPE_TABLES: dict = {}


def begin_run() -> None:
    """A Triton sequence's run begins: the per-run caches empty."""
    global _RUN_EPOCH
    _RUN_EPOCH += 1
    _ROPE_TABLES.clear()


def _widened_once_per_run(t, target):
    """`t.to(target)` computed once per run for a base tensor and its view geometry."""
    base = t._base if getattr(t, "_base", None) is not None else t
    key = (id(base), t.data_ptr(), tuple(t.shape), tuple(t._strides), t.nbx_dtype)
    hit = _ROPE_TABLES.get(key)
    if hit is not None and hit[0] is base:
        return hit[1]
    widened = t.to(target)
    _ROPE_TABLES[key] = (base, widened)
    return widened


def _1d_grid(n_elements):
    """Grid for element-wise (memory-bound) kernels, at the certified tile of the count's class (`_ew_meta`)."""
    return _ew_grid(n_elements, _ew_meta(n_elements))


def _batch_block(batch_dim, feat_dim):
    """Replicate batch_block_heuristic from _configs.py for grid computation.

    Must match the @triton.heuristics decorator on norm/softmax kernels.
    """
    if feat_dim < 64:
        return min(max(1, triton.next_power_of_2(batch_dim // 1024)), 128)
    return 1
_RED_BM, _RED_BN = 8, 1024                 # reduction kernels (prod, min, max, std, var, norm, all, any)


def _reduction_tile(M):
    """(grid, block_m, block_n) for a full-row reduction gridded on M."""
    bm, bn = _red._clamp_reduce_tile(_RED_BM, _RED_BN)
    return (triton.cdiv(M, bm),), bm, bn


# ===========================================================================
# ACTIVATION WRAPPERS
# ===========================================================================

def _dense_span(x):
    """`x` as the flat run of memory its elements fill when it is DENSE in some order of its dims (a permute of a
    contiguous tensor: its strides, sorted, are the contiguous strides of its shape so permuted), else None."""
    shape, stride = tuple(x.shape), tuple(x.stride())
    expect = 1
    for i in sorted((i for i in range(len(shape)) if shape[i] != 1), key=lambda i: stride[i]):
        if stride[i] != expect:
            return None
        expect *= shape[i]
    return x.as_strided((x.numel(),), (1,))


def _flat_unary(x, kernel, out=None):
    """An element-wise kernel over `x`'s elements, its output laid out as `x` is. A dense input in another order of
    its dims (a permute's output) is read where it lies and the result takes its strides, as ATen's element-wise ops
    keep a dense layout; a contiguous copy of it first cost one input-sized buffer the graph never priced
    (mochi-1-preview's VAE: a 3 GB silu input, 2026-10-28). `out` (a caller that proved the input dies here) keeps the
    contiguous path."""
    if out is None and not x.is_contiguous():
        span = _dense_span(x)
        if span is not None:
            n = span.numel()
            res = NBXTensor.empty_like(span)
            _set_device(span)
            kernel[_1d_grid(n)](span, res, n, **_ew_meta(n))
            return res.as_strided(tuple(x.shape), tuple(x.stride()))
    x = x.contiguous()
    output = out if out is not None else NBXTensor.empty_like(x)
    _set_device(x)
    kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def relu(x, out=None) :
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    return _flat_unary(x, relu_forward_kernel, out)


def silu(x, out=None) :
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    return _flat_unary(x, silu_forward_kernel, out)


def gelu(x, approximate: str = 'none', out=None) :
    x = x.contiguous()
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    output = out if out is not None else NBXTensor.empty_like(x)
    _set_device(x)
    gelu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), approximate=(approximate == 'tanh'),
        **_ew_meta(x.numel()))
    return output


def sigmoid_wrapper(x) :
    return _flat_unary(x, sigmoid_forward_kernel)


def tanh_wrapper(x) :
    return _flat_unary(x, tanh_forward_kernel)


def hardsigmoid(x) :
    return _flat_unary(x, hardsigmoid_forward_kernel)


def hardswish(x, out=None) :
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    return _flat_unary(x, hardswish_forward_kernel, out)


def leaky_relu(x, negative_slope: float = 0.01, out=None) :
    x = x.contiguous()
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    output = out if out is not None else NBXTensor.empty_like(x)
    _set_device(x)
    leaky_relu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), negative_slope,
        **_ew_meta(x.numel()))
    return output


def elu(x, alpha: float = 1.0, out=None) :
    x = x.contiguous()
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    output = out if out is not None else NBXTensor.empty_like(x)
    _set_device(x)
    elu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, **_ew_meta(x.numel()))
    return output


def mish(x, out=None) :
    # `out` lets a caller that has PROVED this tensor's input dies here
    # write the result into the input's own buffer. These kernels are
    # strictly element-wise with matched offsets -- one load at
    # `input_ptr + offset`, one store at `output_ptr + offset`, no
    # cross-lane read -- so one pointer passed twice is correct by
    # construction, not by luck. The liveness proof is the caller's:
    # see `OpLevelTilingEngine._detect_inplace_unary_candidates`.
    return _flat_unary(x, mish_forward_kernel, out)


def selu_wrapper(x) :
    return _flat_unary(x, selu_forward_kernel)


# ===========================================================================
# UNARY ELEMENT-WISE WRAPPERS
# ===========================================================================

def neg(x) :
    return _flat_unary(x, neg_forward_kernel)


def exp(x) :
    if isinstance(x, NBXTensor) and x.is_complex():
        # exp(a + bi) = e^a (cos b + i sin b)
        a = x.real.contiguous()
        b = x.imag.contiguous()
        ea = exp(a)
        return complex_wrapper(mul(ea, cos(b)), mul(ea, sin(b)))
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    exp_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def sin(x) :
    return _flat_unary(x, sin_forward_kernel)


def cos(x) :
    return _flat_unary(x, cos_forward_kernel)


def _promote_int_unary(x):
    """Float-math unary ops (rsqrt/sqrt/log/exp/reciprocal/...) promote integer
    or bool input to float32, matching PyTorch type promotion. Without it the
    kernel computes in the integer dtype AND NBXTensor.empty_like keeps the int
    output — e.g. rsqrt(int64 2) truncates 0.7071 to 0, which silently collapsed
    the Kokoro AdainResBlk1d residual `(h + sc) * rsqrt(2)` to all-zeros (the
    rsqrt(2) constant was traced as an int64 scalar; torch auto-promotes it).
    No-op for tensors already in a floating dtype."""
    want = float_math_input(canonical_name(x.nbx_dtype)) if hasattr(x, "nbx_dtype") else None
    if want is not None and want != canonical_name(x.nbx_dtype):
        return x.to(parse_dtype(want))
    return x


def rsqrt(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    rsqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def sqrt_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    sqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def abs_wrapper(x) :
    if x.is_complex():
        # abs of a complex tensor is the REAL magnitude sqrt(re^2 + im^2). The
        # element-wise float path below is doubly wrong for complex64: empty_like
        # keeps the complex dtype AND the kernel runs over x.numel() interleaved
        # [re,im] floats (|interleaved float|, not the magnitude) — half the
        # storage, mis-typed output, heap corruption. Exposed by the Kokoro iSTFT
        # source STFT (abs of _fft_r2c) feeding generator.noise_convs. Computed
        # R33-pure from the stride-2 real/imag views.
        re = x.real
        im = x.imag
        return sqrt_wrapper(add(mul(re, re), mul(im, im)))
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    abs_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def log_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    log_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def reciprocal(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    reciprocal_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def pow_wrapper(x, exponent) :
    # aten::pow has four forms — the base and/or exponent may be a scalar:
    #   tensor ** scalar  → fused kernel (fast path)
    #   scalar ** tensor  → e.g. 10000.0 ** (arange/dim), the sinusoidal/RoPE
    #                       frequency base (Flex DiT) — a**b = exp(b·ln a)
    #   tensor ** tensor  → exp(b·ln a)
    #   scalar ** scalar  → constant
    # All non-fast cases route through exp/mul/log (R33-pure, no torch).
    import math
    x_is_t = isinstance(x, NBXTensor)
    e_is_t = isinstance(exponent, NBXTensor)
    if x_is_t and not e_is_t:
        x = x.contiguous()
        output = NBXTensor.empty_like(x)
        _set_device(x)
        # A small integer exponent takes the exact multiplication route at
        # compile time (kernels/ops/pow.py, INT_EXP); everything else the
        # general exp/log route. Measured 2026-09-20 on CUDA: x**2 by exp/log is
        # up to 15 ulps from the fp64 oracle, by multiplication within 1.
        _e = float(exponent)
        _int_exp = int(_e) if (_e == int(_e) and 1 <= abs(_e) <= 8) else 0
        pow_forward_kernel[_1d_grid(x.numel())](
            x, output, x.numel(), _e,
            **_ew_meta(x.numel()), INT_EXP=_int_exp)
        return output
    if (not x_is_t) and e_is_t:
        # scalar ** tensor = exp(exponent · ln(scalar))
        return exp(mul(exponent, math.log(float(x))))
    if x_is_t and e_is_t:
        # tensor ** tensor = exp(exponent · ln(base))
        return exp(mul(exponent, log_wrapper(x)))
    return float(x) ** float(exponent)


def clamp(x, min_val=None, max_val=None) :
    # aten::clamp.Tensor: a min/max BOUND may itself be an NBXTensor (e.g. Flex
    # DiT). The fused scalar kernel can't take a tensor bound, so route tensor
    # bounds through elementwise maximum/minimum (R33-pure); scalar bounds keep
    # the fused kernel. Mixed (one tensor, one scalar) handled per side.
    if isinstance(min_val, NBXTensor) or isinstance(max_val, NBXTensor):
        out = x
        if min_val is not None:
            out = (maximum_wrapper(out, min_val) if isinstance(min_val, NBXTensor)
                   else clamp(out, min_val=float(min_val), max_val=None))
        if max_val is not None:
            out = (minimum_wrapper(out, max_val) if isinstance(max_val, NBXTensor)
                   else clamp(out, min_val=None, max_val=float(max_val)))
        return out
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _min = float(min_val) if min_val is not None else 0.0
    _max = float(max_val) if max_val is not None else 0.0
    _set_device(x)
    clamp_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), _min, _max,
        has_min=(min_val is not None), has_max=(max_val is not None),
        **_ew_meta(x.numel()))
    return output


def erf(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    erf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def floor_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    floor_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def ceil_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    ceil_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def floor_divide_wrapper(a, b):
    """aten::floor_divide — floor(a / b), result in a's dtype.

    Two integer operands divide in the integers (`_int_divide`), exact at every magnitude. A
    float operand uses the float division (floor(a/b) is the ATen semantic, no exactness cliff
    involved); an integer tensor beside it widens through the backend's float
    (`_int_division_widening_dtype`)."""
    if _both_integer(a, b):
        return _int_divide(a, b, floor=True)
    a_dtype = a.dtype
    _is_int = hasattr(a_dtype, "is_floating") and not a_dtype.is_floating()
    if _is_int:
        wide = _core._int_division_widening_dtype(a, b)
        a_f = a.to(wide)
        b_f = b.to(wide) if hasattr(b, "to") else b
    else:
        a_f = a
        b_f = b
    out = floor_wrapper(div(a_f, b_f))
    return out.to(a_dtype) if out.dtype != a_dtype else out


def _both_integer(a, b):
    return _core._is_integer_operand(a) and _core._is_integer_operand(b) and (isinstance(a, NBXTensor) or isinstance(b, NBXTensor))


def _int_divide(a, b, floor):
    """Integer // integer in the integers (`int_div_kernel`): truncated, or floored when `floor`.
    A Python int operand takes the tensor's dtype; two tensors take the wider; both broadcast."""
    import numpy as np
    ref = a if isinstance(a, NBXTensor) else b
    dtype = _core._wider_dtype(a.dtype, b.dtype) if isinstance(a, NBXTensor) and isinstance(b, NBXTensor) else ref.dtype
    a, b = (x if isinstance(x, NBXTensor) else NBXTensor.from_numpy(np.array(x, dtype=np.dtype(dtype.name)))
            for x in (a, b))
    if hasattr(a, "_device_idx") and hasattr(b, "_device_idx") and a._device_idx != b._device_idx:
        b = _core._transfer_to_device(b, a._device_idx)
    a, b = (x if x.dtype == dtype else x.to(dtype) for x in (a, b))
    shape = broadcast_shapes(a.shape, b.shape)
    a, b = (x.expand(shape).contiguous() if tuple(x.shape) != tuple(shape) else x.contiguous() for x in (a, b))
    _set_device(a)
    output = NBXTensor.empty_like(a)
    n = output.numel()
    if n:
        int_div_kernel[_1d_grid(n)](a, b, output, n, FLOOR=floor, **_ew_meta(n))
    return output


def round_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    round_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def trunc_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    trunc_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


# ===========================================================================
# BINARY ELEMENT-WISE WRAPPERS — use universal launch layer
# ===========================================================================

def _complex_addsub(a, b, alpha: float, is_sub: bool):
    """Complex elementwise add/sub with correct broadcasting.

    The plain float add/sub kernels broadcast the interleaved [re,im] storage of
    a complex tensor on the wrong stride (they treat it as flat floats, so re/im
    misalign across operands when shapes differ — e.g. the Wan rotary-embedding
    freqs grid, built by broadcasting complex per-axis freqs [T,1,1,d]+[T,H,W,1]).
    Routing through view_as_real appends the [.,2] real/imag axis (broadcasts as
    2==2), keeping re aligned with re and im with im. R33-pure.
    """
    ac = isinstance(a, NBXTensor) and a.is_complex()
    bc = isinstance(b, NBXTensor) and b.is_complex()
    if ac and bc:
        ar, br = a.view_as_real(), b.view_as_real()
        out = sub(ar, br, alpha) if is_sub else add(ar, br, alpha)
        return out.view_as_complex()
    # Mixed complex + real: the real operand affects only the real part. Rare in
    # current models; route via the real-view interior and leave imag untouched.
    cplx, real = (a, b) if ac else (b, a)
    vr = cplx.view_as_real().contiguous()
    re = vr.select(vr.ndim - 1, 0)
    im = vr.select(vr.ndim - 1, 1)
    if is_sub and not ac:        # real - complex = (real-re) + i(-im)
        re2 = sub(real, re, 1.0); im2 = mul(im, -1.0)
    elif is_sub:                 # complex - real
        re2 = sub(re, real, alpha); im2 = im
    else:                        # complex + real
        re2 = add(re, real, alpha); im2 = im
    return complex_wrapper(re2.contiguous(), im2.contiguous())


def add(a, b, alpha: float = 1.0) :
    if (isinstance(a, NBXTensor) and a.is_complex()) or (isinstance(b, NBXTensor) and b.is_complex()):
        return _complex_addsub(a, b, alpha, is_sub=False)
    # Bias-broadcast fast path: when `b` is a 1D tensor matching `a`'s
    # last dim and `a` is already contiguous, route to the broadcast-aware
    # kernel that reads `bias[offset % feat_dim]` instead of materializing
    # an 8 GiB contiguous expand of the bias. Sana 4Kpx VAE add::88
    # (`mul::58::out_0 (1, 4096, 4096, 128) + bias (128)`) is the
    # canonical case — saves an 8 GiB transient that otherwise crowds
    # out the rms_norm output during the post-pixel_shuffle chain.
    if (not _core._is_scalar(b)
            and hasattr(a, '_dtype') and hasattr(b, '_dtype')
            and a.ndim >= 1 and b.ndim == 1
            and a.shape[-1] == b.shape[0]
            and a.is_contiguous()):
        # Dtype align (mirror _prepare_binary).
        common_dtype = _core._wider_dtype(a._dtype, b._dtype)
        if a._dtype != common_dtype:
            a = a.to(common_dtype)
        if b._dtype != common_dtype:
            b = b.to(common_dtype)
        # Device align.
        if (hasattr(a, '_device_idx') and hasattr(b, '_device_idx')
                and a._device_idx != b._device_idx):
            b = _core._transfer_to_device(b, a._device_idx)
        n = a.numel()
        feat_dim = b.shape[0]
        output = NBXTensor.empty_like(a)
        _set_device(a)
        add_bias_broadcast_kernel[_1d_grid(n)](
            a, b, output, n, feat_dim, alpha,
            **_ew_meta(n))
        return output

    sc = _core._prepare_scalar_strided(a, b)
    if sc is not None:
        x, scalar, output, n, shp, x_st, ndim = sc
        add_scalar_strided_nd_kernel[_1d_grid(n)](x, output, n, float(scalar) * alpha, shp, x_st,
                                                  **_ew_meta(n), NDIM=ndim)
        return output
    strided = _core._prepare_binary_strided(a, b)
    if strided[0] is not None:
        a, b, output, n, shp, a_st, b_st, ndim = strided
        add_strided_nd_kernel[_1d_grid(n)](a, b, output, n, alpha, shp, a_st, b_st,
                                           **_ew_meta(n), NDIM=ndim)
        return output
    _, a, b = strided
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":
        add_scalar_dev_kernel[_1d_grid(n)](a, b, output, n, float(alpha), **_ew_meta(n))
    elif scalar:
        add_scalar_kernel[_1d_grid(n)](a, output, n, float(b) * alpha, **_ew_meta(n))
    else:
        add_forward_kernel[_1d_grid(n)](a, b, output, n, alpha, **_ew_meta(n))
    return output


def add_inplace_nbx(target, other, alpha: float = 1.0):
    """In-place element-wise add: target += alpha * other.

    Used by op-level tiling for residual adds where Prism's liveness
    analysis proved that one input has its last use at this op (so its
    buffer can be safely reused as output, avoiding a 3rd allocation).
    Saves up to N bytes per call where N = output tensor size — on
    Sana 4Kpx VAE residual adds at (1, 128, 4096, 4096) fp32 this is
    8 GiB per call.

    CONTRACT (caller-enforced by Prism liveness analysis):
      - target.shape == other.shape (no broadcast, by construction since
        residual adds always operate on identical shapes)
      - target.dtype == other.dtype
      - target.device == other.device
      - target is contiguous (produced by upstream kernels which write
        contiguous tensors by default)
      - target has no other consumer in the graph (last use here)

    Returns target (same object, mutated in place).
    """
    assert target.shape == other.shape, (
        f"add_inplace_nbx requires identical shapes; got "
        f"target={target.shape} other={other.shape}")
    # POINT 6 H2 FIX: kernel uses flat 1D indexing
    # `tl.load/store(target_ptr + offset, ...)` which assumes target
    # is contiguous in memory. If target is a non-contiguous view
    # (permute, transpose, slice with stride), flat indexing reads/
    # writes wrong addresses — silent corruption. The caller's "last-
    # use" liveness analysis from Prism does not check contiguity.
    # Pre-fix on Sana 4Kpx VAE: residual adds at op 649+ (add::69
    # onward) showed +1.3% rel divergence introducing horizontal-
    # band garbage that then amplified through downstream tiled
    # kernels. Fall back to the standard non-in-place `add` when
    # target is not contiguous (handles strided views correctly via
    # `_prepare_binary` + `expand+contiguous`).
    if not target.is_contiguous():
        return add(target, other, alpha=alpha)
    if hasattr(target, '_device_idx') and hasattr(other, '_device_idx') \
            and target._device_idx != other._device_idx:
        other = _core._transfer_to_device(other, target._device_idx)
    # Dtype alignment. In-place semantics require writing into target's
    # buffer, so target's dtype is the result dtype. If `other` is at a
    # different precision, cast it (the cast allocates a transient sized
    # like other — same VRAM cost as a non-in-place add only when other
    # < target dtype). If target is NARROWER than other, in-place would
    # lose precision: fall back to the standard non-in-place `add` so the
    # standard `_wider_dtype` upcast path kicks in.
    if target._dtype != other._dtype:
        if _core._wider_dtype(target._dtype, other._dtype) == target._dtype:
            other = other.to(target._dtype)
        else:
            return add(target, other, alpha=alpha)
    other = other.contiguous()
    n = target.numel()
    _set_device(target)
    add_forward_kernel[_1d_grid(n)](
        target, other, target, n, alpha,
        **_ew_meta(n))
    return target


def _as_complex_scalar(v):
    """Parse a complex literal ('1j', '(1+2j)', a Python complex) → complex, else None.
    The tracer serializes complex constants like the imaginary unit as an
    'unknown'-typed string ('1j')."""
    if isinstance(v, complex):
        return v
    if isinstance(v, str):
        try:
            return complex(v.strip())
        except (ValueError, TypeError):
            return None
    return None


def _cmul_scalar(x, c):
    """x (real or complex NBXTensor) * complex scalar c → complex64."""
    cr, ci = c.real, c.imag
    if isinstance(x, NBXTensor) and x.is_complex():
        xr, xi = x.real, x.imag
        return complex_wrapper(sub(mul(xr, cr), mul(xi, ci)),
                               add(mul(xr, ci), mul(xi, cr)))
    real = mul(x, cr) if cr != 0.0 else NBXTensor.zeros_like(x)
    imag = mul(x, ci) if ci != 0.0 else NBXTensor.zeros_like(x)
    return complex_wrapper(real, imag)


def _complex_mul(a, b):
    """Handle a mul involving a complex scalar/tensor. Returns the complex result,
    or None to fall through to the real elementwise mul (the common case)."""
    ca = _as_complex_scalar(a)
    cb = _as_complex_scalar(b)
    a_t = isinstance(a, NBXTensor) and a.is_complex()
    b_t = isinstance(b, NBXTensor) and b.is_complex()
    if ca is None and cb is None and not a_t and not b_t:
        return None  # purely real → real mul
    if cb is not None and isinstance(a, NBXTensor):
        return _cmul_scalar(a, cb)
    if ca is not None and isinstance(b, NBXTensor):
        return _cmul_scalar(b, ca)
    # complex tensor * (real or complex) tensor
    if a_t or b_t:
        if a_t and not b_t:
            return complex_wrapper(mul(a.real, b), mul(a.imag, b))
        if b_t and not a_t:
            return complex_wrapper(mul(b.real, a), mul(b.imag, a))
        ar, ai, br, bi = a.real, a.imag, b.real, b.imag
        return complex_wrapper(sub(mul(ar, br), mul(ai, bi)),
                               add(mul(ar, bi), mul(ai, br)))
    return None


def mul(a, b) :
    cr = _complex_mul(a, b)
    if cr is not None:
        return cr
    sc = _core._prepare_scalar_strided(a, b)
    if sc is not None:
        x, scalar, output, n, shp, x_st, ndim = sc
        mul_scalar_strided_nd_kernel[_1d_grid(n)](x, output, n, float(scalar), shp, x_st,
                                                  **_ew_meta(n), NDIM=ndim)
        return output
    strided = _core._prepare_binary_strided(a, b)
    if strided[0] is not None:
        a, b, output, n, shp, a_st, b_st, ndim = strided
        mul_strided_nd_kernel[_1d_grid(n)](a, b, output, n, shp, a_st, b_st,
                                           **_ew_meta(n), NDIM=ndim)
        return output
    _, a, b = strided
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":
        mul_scalar_dev_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    elif scalar:
        # Contract: tensor is always `a`, scalar is always `b`
        mul_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        mul_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def div(a, b, rounding_mode=None) :
    # aten::div carries an optional rounding_mode kwarg ("floor"/"trunc",
    # e.g. the SD3-class pos_embed crop offset (max-h)//2 traces as
    # div(rounding_mode=floor) on a 0-dim int64). Routed onto the
    # existing bricks — floor_divide_wrapper's fp64 int discipline for
    # floor, the same composition with trunc for trunc.
    if rounding_mode == "floor":
        return floor_divide_wrapper(a, b)
    if rounding_mode == "trunc":
        if _both_integer(a, b):
            return _int_divide(a, b, floor=False)
        a_dtype = a.dtype
        _is_int = hasattr(a_dtype, "is_floating") and not a_dtype.is_floating()
        if _is_int:
            wide = _core._int_division_widening_dtype(a, b)
            a_f = a.to(wide)
            b_f = b.to(wide) if hasattr(b, "to") else b
        else:
            a_f, b_f = a, b
        out = trunc_wrapper(div(a_f, b_f))
        return out.to(a_dtype) if out.dtype != a_dtype else out
    if rounding_mode is not None:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::div rounding_mode '{rounding_mode}' "
            "is not a known ATen mode (floor/trunc/None).")
    strided = _core._prepare_binary_strided(a, b)
    if strided[0] is not None:
        a, b, output, n, shp, a_st, b_st, ndim = strided
        div_strided_nd_kernel[_1d_grid(n)](a, b, output, n, shp, a_st, b_st,
                                           **_ew_meta(n), NDIM=ndim)
        return output
    _, a, b = strided
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":
        div_scalar_dev_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    elif scalar:
        div_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        div_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def sub(a, b, alpha: float = 1.0) :
    if (isinstance(a, NBXTensor) and a.is_complex()) or (isinstance(b, NBXTensor) and b.is_complex()):
        return _complex_addsub(a, b, alpha, is_sub=True)
    strided = _core._prepare_binary_strided(a, b)
    if strided[0] is not None:
        a, b, output, n, shp, a_st, b_st, ndim = strided
        sub_strided_nd_kernel[_1d_grid(n)](a, b, output, n, alpha, shp, a_st, b_st,
                                           **_ew_meta(n), NDIM=ndim)
        return output
    _, a, b = strided
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":
        add_scalar_dev_kernel[_1d_grid(n)](a, b, output, n, -float(alpha), **_ew_meta(n))
    elif scalar:
        add_scalar_kernel[_1d_grid(n)](a, output, n, -float(b) * alpha, **_ew_meta(n))
    else:
        sub_forward_kernel[_1d_grid(n)](a, b, output, n, alpha, **_ew_meta(n))
    return output


def rsub(a, b) :
    """Reverse subtraction: b - a."""
    a, _, _, n, dev_ctx, _ = _core._prepare_binary(a, a)  # just prepare a
    output = NBXTensor.empty_like(a)
    rsub_forward_kernel[_1d_grid(n)](a, output, n, float(_core._to_scalar(b)), **_ew_meta(n))
    return output


def maximum_wrapper(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":  # off the timestep hot path: host extract, exact old behavior
        b, scalar = _core._to_scalar(b), True
    if scalar:
        # scalar maximum: clamp_min
        clamp_min_forward_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        maximum_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def minimum_wrapper(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":  # off the timestep hot path: host extract, exact old behavior
        b, scalar = _core._to_scalar(b), True
    if not scalar:
        minimum_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    else:
        # scalar minimum: clamp(max=b)
        clamp_forward_kernel[_1d_grid(n)](a, output, n, 0.0, float(b), has_min=False, has_max=True, **_ew_meta(n))
    return output


# ===========================================================================
# COMPARISON WRAPPERS
# ===========================================================================

def gt(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        gt_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        gt_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def ge(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        ge_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        ge_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def lt(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        lt_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        lt_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def eq(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        eq_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        eq_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def ne(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        ne_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        ne_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


# The 2-D reduction tile (min / max / prod / std / var / norm / all / any over
# a dim). The adopted 8 x 1024 with 4 warps; `block_sizes.reduction_2d` in the
# hardware profile overrides it.
_REDUCTION_2D_TILE = (8, 1024, 4)


def _reduction_2d_tile() -> tuple:
    """`(BLOCK_M, BLOCK_N, num_warps)` for the row-wise reductions, from
    `block_sizes.reduction_2d` in config/vendors/<vendor>/<arch>.yml.

    `min` and `max` over a dim return a value AND its index, so their reduce
    carries a pair. A backend may need the whole tile to fit its threadgroup
    to aggregate one — the same property that governs `argmax` — and the
    threadgroup is `num_warps * 32`, which is why the warp count belongs to
    the tile. Absent key / no profile -> the adopted values.
    """
    bm, bn, warps = _REDUCTION_2D_TILE
    try:
        from nbxtensor.ops._configs import active_vendor_profile
        cfg = (active_vendor_profile().get("block_sizes", {}) or {}).get("reduction_2d")
        if isinstance(cfg, dict):
            bm = int(cfg.get("block_m") or bm)
            bn = int(cfg.get("block_n") or bn)
            warps = int(cfg.get("num_warps") or warps)
    except Exception:
        pass
    return bm, bn, warps


def isin_wrapper(elements, test_elements, *, invert: bool = False,
                 assume_unique: bool = False):
    """aten::isin — elementwise membership test.

    elements      : NBXTensor (any int dtype), shape (...)
    test_elements : NBXTensor (same dtype), shape (K,) typically small
    returns       : NBXTensor bool, same shape as `elements`

    Small-op fast path: D2H both, numpy.isin on host, H2D the bool mask.
    Typical sizes in audio models are elements=(1, S<=2048), test_elements<=4K
    — a handful of microseconds on CPU. No torch imports.
    """
    import numpy as np
    elements_np = _core._nbx_to_numpy(elements.contiguous())
    test_np = _core._nbx_to_numpy(test_elements.contiguous())
    out_np = np.isin(elements_np, test_np,
                     assume_unique=bool(assume_unique),
                     invert=bool(invert))
    # isin is a boolean membership predicate — return a true bool_ NBXTensor so
    # downstream bool-aware ops recognise it. Staging through uint8 (the earlier
    # workaround) lost the bool-ness: bitwise_not's `nbx_dtype == bool_` guard
    # then missed and did a raw byte complement (~0 -> 255) instead of the LOGICAL
    # NOT, which corrupted the openaudio DualAR attention mask (op #46 of the slow
    # backbone, first triton-vs-sequential divergence). from_numpy handles |b1
    # cleanly now (verified empirically), so the uint8 staging is no longer needed.
    return NBXTensor.from_numpy(out_np.astype(np.bool_))


def invoke_silu_and_mul(input_tensor, M, N):
    """Launch fused SwiGLU kernel: silu(input[:, :N]) * input[:, N:2*N].

    Args:
        input_tensor: NBXTensor [M, 2*N]
        M: number of rows
        N: output columns (half of input columns)

    Returns:
        NBXTensor [M, N]
    """
    output = NBXTensor.empty((M, N), dtype=input_tensor._dtype,
                             device=f"cuda:{input_tensor._device_idx}")
    BLOCK_M = 16
    BLOCK_N = min(triton.next_power_of_2(N), 1024)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _set_device(input_tensor)

    silu_and_mul_kernel[grid](
        input_tensor, output,
        M, N,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return output


def _row_matrix(t):
    """`t` as an (M, N) matrix over its last dim, WITHOUT a copy when its rows are evenly strided — the last dim
    unit-stride and every leading dim mergeable into one row axis, as a split of a fused projection's last dim is (the
    gate and up halves of a SwiGLU: (B, S, N) views of a (B, S, 2N) buffer, row stride 2N). A copy otherwise. The
    kernels reading it take its row stride. Copied unconditionally, the two halves of mochi-1-preview's (2, 7 950,
    16 384) fp32 projection cost 2 x 497 MB the graph never priced, and its transformer overran the 8 GB rung by
    them (2026-10-28)."""
    shape, stride = tuple(t.shape), tuple(t.stride())
    n = shape[-1] if shape else 1
    m = 1
    for d in shape[:-1]:
        m *= d
    if len(shape) >= 2 and (stride[-1] == 1 or n == 1):
        dims = [i for i in range(len(shape) - 1) if shape[i] != 1]
        if all(stride[a] == stride[b] * shape[b] for a, b in zip(dims, dims[1:])):
            row = stride[dims[-1]] if dims else n
            return t.as_strided((m, n), (row, 1))
    return t.contiguous().view(m, n)


def swiglu_fused_wrapper(gate, up):
    """Fused SwiGLU with split gate/up tensors: silu(gate) * up.

    Used by the _fuse_swiglu_ops pass in TritonSequence to replace
    aten::silu + aten::mul with a single kernel launch. Gate and up may
    be N-D (typically [B, S, N]); we flatten everything but the last dim
    for the 2-D kernel.

    Args:
        gate: NBXTensor [..., N]
        up:   NBXTensor [..., N] (same shape as gate)

    Returns:
        NBXTensor same shape as gate.
    """
    # Align dtypes (matches the semantics of the pre-fusion aten::mul,
    # which relied on PyTorch dtype promotion).
    if gate.dtype != up.dtype:
        _order = (NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.float16)
        widest = next(d for d in _order if d in (gate.dtype, up.dtype))
        if gate.dtype != widest:
            gate = gate.to(widest)
        if up.dtype != widest:
            up = up.to(widest)

    orig_shape = gate.shape
    gate2d = _row_matrix(gate)
    up2d = _row_matrix(up)
    M, N = gate2d.shape
    output = NBXTensor.empty((M, N), dtype=gate2d._dtype,
                             device=f"cuda:{gate2d._device_idx}")

    BLOCK_M = 16
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _set_device(gate2d)

    silu_mul_split_kernel[grid](
        gate2d, up2d, output,
        M, N,
        gate2d.stride(0), gate2d.stride(1),
        up2d.stride(0), up2d.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return output.view(*orig_shape)



def rope_fused_wrapper(q_raw, k_raw, cos, sin):
    """Fused RoPE (Liger-style rotate_half) — applies to Q and K in one launch.

    Replaces the graph chain slice×4 + neg×2 + cat×2 + mul×4 + add×2 (plus
    the per-side _to_copy done downstream) with a single call to
    rope_forward_kernel (see kernels/ops/rope.py).

    Inputs arrive in the HF-Llama layout, post-transpose:
        q_raw : [B, H_q,  S, D]  (logical view; physical memory [B,S,H_q,D])
        k_raw : [B, H_kv, S, D]  (logical view; physical memory [B,S,H_kv,D])
        cos   : [1|B, 1, S, D]   (broadcast over heads; from unsqueeze)
        sin   : [1|B, 1, S, D]

    The kernel expects physical (B, S, H, D) layout with q/k row-stride =
    H*D. Since transpose is a zero-copy stride permutation, we do the
    inverse transpose+contiguous to materialise a fresh (B, S, H, D) buffer,
    apply the kernel in-place on the fresh buffer, then view it back as
    (B, H, S, D). The contiguous copy doubles as the arena-safety clone.
    """
    # Squeeze cos/sin broadcast dims: [1|B, 1, S, D] -> [1|B, S, D].
    if cos.ndim == 4 and cos.shape[1] == 1:
        cos = cos.view(cos.shape[0], cos.shape[2], cos.shape[3])
    if sin.ndim == 4 and sin.shape[1] == 1:
        sin = sin.view(sin.shape[0], sin.shape[2], sin.shape[3])

    # The kernel computes in Q's dtype. A table WIDER than Q (fp32 tables, fp16 Q:
    # the decode case) is rounded to it on load — round-to-nearest, the stored
    # cast's conversion; the tables lie in [-1, 1], so the protected fp16 clamp
    # never applied — and the per-layer cast copy this replaces (TinyLlama: 44 a
    # token) is gone. A table NARROWER than Q (fp16 tables, fp32 Q: Sana's Gemma-2
    # encoder) is widened here, once per run (`_widened_once_per_run`). JUSTIFIED
    # COPY, 2 per run, no longer 2 per layer:
    # widened on load, the fp32 rotation's fused multiply-add moved with the
    # convert before it (1 ulp on a sixth of the elements, 2026-09-07), and an
    # explicit fma matched the fp32 path but not the fp16 one — the arithmetic
    # stays the plain expression, and the values arrive as they did. K is rotated
    # in place, so a K of another dtype than Q needs a buffer of Q's dtype.
    target_dt = q_raw.dtype
    if _core._wider_dtype(cos.nbx_dtype, q_raw.nbx_dtype) == q_raw.nbx_dtype and cos.nbx_dtype != q_raw.nbx_dtype:
        cos = _widened_once_per_run(cos, target_dt)      # once per run, not once per layer
    if _core._wider_dtype(sin.nbx_dtype, q_raw.nbx_dtype) == q_raw.nbx_dtype and sin.nbx_dtype != q_raw.nbx_dtype:
        sin = _widened_once_per_run(sin, target_dt)
    if k_raw.dtype != target_dt:
        k_raw = k_raw.to(target_dt)

    cos = cos.contiguous()
    sin = sin.contiguous()

    # Transpose back to (B, S, H, D) physical layout, then contiguous copy:
    # the kernel writes in-place, and this copy is our arena-safe clone.
    q_bshd = q_raw.transpose(1, 2).contiguous()
    k_bshd = k_raw.transpose(1, 2).contiguous()

    bs, sl, n_qh, hd = q_bshd.shape
    _, _, n_kh, _ = k_bshd.shape
    cos_bs = cos.shape[0]  # 1 (broadcast over batch) or bs

    # Row stride = H*D for (B, S, H, D) contiguous layout.
    q_row_stride = n_qh * hd
    k_row_stride = n_kh * hd

    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd = triton.next_power_of_2(hd)
    BLOCK_SIZE = max(pad_n_qh, pad_n_kh)

    grid = (bs * sl,)
    _set_device(q_bshd)

    rope_forward_kernel[grid](
        q_bshd, q_row_stride,
        k_bshd, k_row_stride,
        cos, cos.stride(-2),
        sin, sin.stride(-2),
        sl,
        bs=bs, cos_bs=cos_bs,
        n_qh=n_qh, n_kh=n_kh, hd=hd,
        pad_n_qh=pad_n_qh, pad_n_kh=pad_n_kh, pad_hd=pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
        num_warps=4,
    )

    # View back as (B, H, S, D) to match the pre-fusion output layout.
    q_out = q_bshd.transpose(1, 2)
    k_out = k_bshd.transpose(1, 2)
    return q_out, k_out


# ===========================================================================
# DTYPE CONVERSION WRAPPER
# ===========================================================================

def bf16_to_fp16_gpu(src: NBXTensor, out: "NBXTensor | None" = None) -> NBXTensor:
    """bfloat16 patterns (a 2-byte integer tensor holding the raw bits) to float16 on the device: the one device
    conversion of the pair, through float32 (`ops/dtype_convert.bf16_to_fp16_kernel`).

    `out`: a float16 tensor of the same number of elements to write into — the SAME buffer as `src` converts in
    place (both are two bytes an element, and each element is read before it is written); none: a new tensor."""
    N = src.numel()
    dst = out if out is not None else NBXTensor.empty(src._shape, NBXDtype.float16,
                                                     device=f"cuda:{src._device_idx}")
    if dst.nbx_dtype != NBXDtype.float16 or dst.numel() != N:
        raise ValueError(f"bf16_to_fp16_gpu: out must be float16 with {N} elements, got {dst.nbx_dtype.name} "
                         f"with {dst.numel()}")
    _set_device(src)
    _m = _ew_meta(N)
    bf16_to_fp16_kernel[_ew_grid(N, _m)](src, dst, N, BLOCK=_m['BLOCK_SIZE'], num_warps=_m['num_warps'])
    return dst


# ===========================================================================
# GLU WRAPPER
# ===========================================================================

def glu_wrapper(x, dim: int = -1, act_func: str = 'sigmoid') :
    """Gated Linear Unit: out = x1 * act(x2) where x is split on dim."""
    dim = dim % x.ndim
    half_size = x.shape[dim] // 2
    x1, x2 = x.narrow(dim, 0, half_size), x.narrow(dim, half_size, half_size)
    x1, x2 = x1.contiguous(), x2.contiguous()
    output = NBXTensor.empty_like(x1)
    _set_device(x1)
    glu_forward_kernel[_1d_grid(x1.numel())](
        x1, x2, output, x1.numel(), act_func,
        **_ew_meta(x1.numel()))
    return output


# ===========================================================================
# PHASE 2 WRAPPERS — Extracted from FlagGems reference
# ===========================================================================

def le(a, b) :
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        le_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        le_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def softplus_wrapper(x, beta: float = 1.0, threshold: float = 20.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    softplus_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), beta, threshold,
        **_ew_meta(x.numel()))
    return output


def remainder_wrapper(a, b) :
    """Remainder (modulo): a % b."""
    from nbxtensor.ops.remainder import remainder_forward_kernel, remainder_scalar_kernel
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":  # off the timestep hot path: host extract, exact old behavior
        b, scalar = _core._to_scalar(b), True
    if scalar:
        remainder_scalar_kernel[_1d_grid(n)](a, output, n, float(b), **_ew_meta(n))
    else:
        remainder_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


# ===========================================================================
# PHASE 3 — Simple element-wise wrappers
# ===========================================================================

def exp2_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    exp2_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def tan_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    tan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def celu_wrapper(x, alpha: float = 1.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    celu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, **_ew_meta(x.numel()))
    return output


def log_sigmoid_wrapper(x) :
    return _flat_unary(x, log_sigmoid_forward_kernel)


def isfinite_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isfinite_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def isinf_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isinf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def isnan_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isnan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


def nan_to_num_wrapper(x, nan: float = 0.0,
                       posinf: float = 3.4028235e+38,
                       neginf: float = -3.4028235e+38, inplace: bool = False) :
    """`inplace`: write over a contiguous `x` (an element-wise map: each element read once, then written) — the math
    attention's probabilities, whose copy was one more [B*H, Tq, Tk] buffer at its peak."""
    x = x.contiguous()
    output = x if inplace else NBXTensor.empty_like(x)
    _set_device(x)
    nan_to_num_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), nan, posinf, neginf,
        **_ew_meta(x.numel()))
    return output


def threshold_wrapper(x, threshold: float, value: float) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    threshold_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), threshold, value,
        **_ew_meta(x.numel()))
    return output


# ===========================================================================
# PHASE 4 — Full coverage wrappers
# ===========================================================================

# ---------------------------------------------------------------------------
# Addcdiv / Addcmul (3-tensor + scalar)
# ---------------------------------------------------------------------------

def addcdiv_wrapper(input, tensor1,
                    tensor2, value: float = 1.0) :
    """result = input + value * (tensor1 / tensor2)"""
    input, tensor1, tensor2 = input.contiguous(), tensor1.contiguous(), tensor2.contiguous()
    output = NBXTensor.empty_like(input)
    _set_device(input)
    addcdiv_forward_kernel[_1d_grid(input.numel())](
        input, tensor1, tensor2, output, input.numel(), value,
        **_ew_meta(input.numel()))
    return output


def addcmul_wrapper(input, tensor1,
                    tensor2, value: float = 1.0) :
    """result = input + value * tensor1 * tensor2"""
    input, tensor1, tensor2 = input.contiguous(), tensor1.contiguous(), tensor2.contiguous()
    output = NBXTensor.empty_like(input)
    _set_device(input)
    addcmul_forward_kernel[_1d_grid(input.numel())](
        input, tensor1, tensor2, output, input.numel(), value,
        **_ew_meta(input.numel()))
    return output


# ---------------------------------------------------------------------------
# Bitwise ops (binary: AND/OR, unary: NOT)
# ---------------------------------------------------------------------------

def bitwise_and_wrapper(a, b) :
    # _prepare_binary = the universal binary contract (dtype align, device
    # align, BROADCAST, contiguous). The former `a.contiguous(); empty_like(a)`
    # form silently ignored broadcasting: bitwise_and([1,1,S,1], [1,1,1,S])
    # (the T5 extended-attention-mask outer product q_valid & k_valid, Allegro
    # text_encoder) returned the elementwise diagonal [1,1,S,1] instead of
    # [1,1,S,S] — same numel, so no OOB crash — and the downstream expand
    # stretched the q-valid column over the key axis: NO key masking, every
    # real token attended all pad tokens in every T5 layer (triton cond-branch
    # 35.8% drift vs compiled at native, uncond clean because the empty-prompt
    # mask is all-ones = inert).
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":  # off the timestep hot path: host extract, exact old behavior
        b, scalar = _core._to_scalar(b), True
    if scalar:
        # 0-dim / python-scalar operand (e.g. the traced `mask.all() & qside`
        # opening the T5 extended-mask chain): materialize to a full tensor of
        # a's dtype and run the tensor kernel. Cold path, exact for every
        # dtype. The former code fed the 0-dim tensor straight to the flat
        # kernel, reading a.numel() elements past its 1-element allocation
        # (out-of-bounds, undefined values).
        b = NBXTensor.empty_like(a).fill_(b)
    bitwise_and_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def bitwise_or_wrapper(a, b) :
    # Same universal binary contract as bitwise_and_wrapper (broadcast fix).
    a, b, output, n, dev_ctx, scalar = _core._prepare_binary(a, b)
    if scalar == "dev":  # off the timestep hot path: host extract, exact old behavior
        b, scalar = _core._to_scalar(b), True
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    bitwise_or_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def bitwise_not_wrapper(x) :
    # torch.bitwise_not on a BOOL tensor is the LOGICAL NOT (0<->1), NOT the
    # bitwise complement: ~True(1)=254, ~False(0)=255 are both non-zero, so the
    # bitwise kernel yields an all-True bool tensor. Route bool through
    # logical_not. Surfaced by the parakeet conformer attention padding mask
    # (`~(arange<len)`): an all-True mask made masked_fill overwrite every score
    # with -1e4 -> uniform softmax -> dead self-attention -> garbage transcription.
    if getattr(x, "nbx_dtype", None) == NBXDtype.bool_:
        return logical_not_wrapper(x)
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    bitwise_not_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


# ---------------------------------------------------------------------------
# Logical ops (binary: AND/OR → bool, unary: NOT → bool)
# ---------------------------------------------------------------------------

def logical_and_wrapper(a, b) :
    # _prepare_comparison = the universal binary contract with a bool output
    # (dtype align, device align, BROADCAST, contiguous). Same broadcast fix
    # as bitwise_and_wrapper — the former form silently ignored broadcasting.
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    logical_and_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def logical_or_wrapper(a, b) :
    # Same universal binary contract as logical_and_wrapper (broadcast fix).
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    logical_or_forward_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def logical_not_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    logical_not_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
    return output


# ---------------------------------------------------------------------------
# Lerp (linear interpolation)
# ---------------------------------------------------------------------------

def lerp_wrapper(input, end, weight) :
    """Linear interpolation: numerically stable two-branch formula."""
    input, end = input.contiguous(), end.contiguous()
    output = NBXTensor.empty_like(input)
    if isinstance(weight, NBXTensor):
        weight = weight.contiguous()
        _set_device(input)
        lerp_tensor_forward_kernel[_1d_grid(input.numel())](
            input, end, weight, output, input.numel(),
            **_ew_meta(input.numel()))
    elif abs(weight) < 0.5:
        _set_device(input)
        lerp_scalar_head_kernel[_1d_grid(input.numel())](
            input, end, output, input.numel(), float(weight),
            **_ew_meta(input.numel()))
    else:
        _set_device(input)
        lerp_scalar_tail_kernel[_1d_grid(input.numel())](
            input, end, output, input.numel(), float(weight),
            **_ew_meta(input.numel()))
    return output


# ---------------------------------------------------------------------------
# Min / Max (value reductions, two-pass global)
# ---------------------------------------------------------------------------

def min_wrapper(x, dim=None, keepdim=False):
    """Min reduction. Two-pass global. Returns scalar for dim=None."""
    import math
    x = x.contiguous()
    if dim is None:
        M = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        mid = NBXTensor.empty(mid_size, dtype=x.dtype, device=x.device)
        out = NBXTensor.empty([], dtype=x.dtype, device=x.device)
        _set_device(x.view(-1))
        min_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        _set_device(mid)
        min_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        out_index = NBXTensor.empty(M, dtype=NBXDtype.int64, device=x.device)
        _red_bm, _red_bn, _red_warps = _reduction_2d_tile()
        grid = (triton.cdiv(M, _red_bm),)
        _set_device(x_perm)
        # The kernel writes the value AND its index (ATen's min.dim returns
        # both); the wrapper passed four arguments to a five-argument kernel
        # — a latent defect on a path no model of the zoo exercised, found by
        # the kernel reference bank (2026-09-05).
        min_kernel[grid](x_perm, out, out_index, M, N,
                         BLOCK_M=_red_bm, BLOCK_N=_red_bn,
                         num_warps=_red_warps)
        shape[dim] = 1
        values = out.view(shape)
        indices = out_index.view(shape)
        if not keepdim:
            values = values.squeeze(dim)
            indices = indices.squeeze(dim)
        return values, indices


def max_wrapper(x, dim=None, keepdim=False):
    """Max reduction. Two-pass global. Returns scalar for dim=None."""
    import math
    x = x.contiguous()
    if dim is None:
        M = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        mid = NBXTensor.empty(mid_size, dtype=x.dtype, device=x.device)
        out = NBXTensor.empty([], dtype=x.dtype, device=x.device)
        _set_device(x.view(-1))
        max_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        _set_device(mid)
        max_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        out_index = NBXTensor.empty(M, dtype=NBXDtype.int64, device=x.device)
        _red_bm, _red_bn, _red_warps = _reduction_2d_tile()
        grid = (triton.cdiv(M, _red_bm),)
        _set_device(x_perm)
        # The kernel writes the value AND its index (ATen's max.dim returns
        # both); the wrapper passed four arguments to a five-argument kernel
        # — a latent defect on a path no model of the zoo exercised, found by
        # the kernel reference bank (2026-09-05).
        max_kernel[grid](x_perm, out, out_index, M, N,
                         BLOCK_M=_red_bm, BLOCK_N=_red_bn,
                         num_warps=_red_warps)
        shape[dim] = 1
        values = out.view(shape)
        indices = out_index.view(shape)
        if not keepdim:
            values = values.squeeze(dim)
            indices = indices.squeeze(dim)
        return values, indices


# ---------------------------------------------------------------------------
# Bincount — small histogram for MoE routing
# ---------------------------------------------------------------------------

def bincount_wrapper(x, minlength: int = 0):
    """Bincount via CPU numpy (input is small 1D int tensor from MoE routing)."""
    import numpy as np
    import ctypes as _ctypes
    # D2H transfer (small: num_tokens * top_k ints)
    n = x.numel()
    buf = (_ctypes.c_int64 * n)()
    DeviceAllocator.memcpy(_ctypes.addressof(buf), x.data_ptr(), n * 8, kind=2)
    arr = np.frombuffer(buf, dtype=np.int64)
    counts = np.bincount(arr, minlength=minlength).astype(np.int64)
    DeviceAllocator.set_device(x._device_idx)
    return NBXTensor.from_numpy(counts)


# ---------------------------------------------------------------------------
# Sort (radix sort — complex, uses Triton radix sort kernels)
# ---------------------------------------------------------------------------

def sort_wrapper(x, dim: int = -1,
                 descending: bool = False, stable: bool = False):
    """Sort via Triton radix sort. Uses histogram + cumsum + sweep pipeline.
    All compute via Triton kernels, allocation via NBXTensor.

    For the exclusive prefix sum step, we use our Triton cumsum kernel
    uses Triton radix sort kernels.
    """
    from nbxtensor.ops.sort_op import (
        convert_to_uint_preserve_order,
        radix_sort_histogram_kernel,
        radix_sort_tile_counts_kernel,
        radix_sort_tile_prefix_kernel,
        radix_sort_scatter_kernel,
    )
    import math as _math

    if dim < 0:
        dim = dim + x.ndim
    assert dim == x.ndim - 1, "sort currently supports last dim only"

    n = x.shape[dim]
    m = x.numel() // n
    num_bits = x.element_size() * 8
    k_bits = 8
    n_passes = triton.cdiv(num_bits, k_bits)
    num_bins = 2 ** k_bits

    TILE_N = 1024
    tiles_per_cta = 8
    CTA_TILE = tiles_per_cta * TILE_N
    grid_n = triton.cdiv(n, CTA_TILE)

    # Allocate all buffers
    global_hist = NBXTensor.zeros((m, n_passes, num_bins), device=x.device, dtype=NBXDtype.int32)

    TILE_R = 8

    # Stage 1: compute global histogram
    _set_device(x.contiguous())
    radix_sort_histogram_kernel[(m * grid_n,)](
        x.contiguous(), global_hist,
        n_passes, m, n, tiles_per_cta,
        TILE_N, TILE_R, k_bits, descending)

    # Exclusive prefix sum on histogram (use our cumsum kernel)
    # cumsum along last dim, then subtract self → exclusive prefix sum
    hist_cumsum = NBXTensor.empty_like(global_hist)
    for b in range(m):
        for p in range(n_passes):
            row = global_hist[b, p]
            cs = _red.cumsum_wrapper(row, dim=0)
            hist_cumsum[b, p] = cs - row
    ex_cumsum = hist_cumsum.to(NBXDtype.int32)

    # Double buffers. The sweep kernel CARRIES the associate array (the
    # argsort indices) through the radix passes — it never synthesizes
    # initial positions, so the buffer must start as arange(n) PER ROW
    # (R33-pure via the triton arange kernel). First live firing (the
    # Ming windowed-ViT argsort) exposed the old init: an empty (0, n)
    # zeros buffer from an invalid zeros(0, n, dtype=...) call.
    arr_in = x.contiguous().clone()
    indices_in = NBXTensor.empty((m, n), dtype=NBXDtype.int64,
                                 device=x.device)
    _set_device(indices_in)
    for b in range(m):
        arange_kernel[_1d_grid(n)](
            indices_in[b], 0, 1, n, **_ew_meta(n))
    arr_out = NBXTensor.empty_like(arr_in)
    indices_out = NBXTensor.empty_like(indices_in)

    grid_r = triton.cdiv(num_bins, TILE_R)
    SWEEP_TILE = 2048
    sweep_grid_n = triton.cdiv(n, SWEEP_TILE)
    # `status` holds each tile's per-bin COUNT; `tile_prefix` the exclusive scan
    # of those counts along the tile axis. Two buffers of the same shape, and
    # the second is what the scatter reads instead of spinning on the first.
    status = NBXTensor.empty((m, num_bins, sweep_grid_n), device=x.device, dtype=NBXDtype.int32)
    tile_prefix = NBXTensor.empty((m, num_bins, sweep_grid_n), device=x.device, dtype=NBXDtype.int32)

    # Stage 2: sweep per radix pass. OUT_N is the STATUS row stride the
    # kernel indexes with pid_n ∈ [0, sweep_grid_n) — it must be the CTA
    # count along n, never n itself (the old call passed n: every status
    # write past bin 0 landed out of the (m, bins, sweep_grid_n) buffer).
    for i in range(n_passes):
        status.fill_(0)
        _set_device(arr_in)
        # THREE launches, not one. The single-kernel form carried a decoupled
        # lookback — a cross-CTA spin — which Triton does not order and Metal
        # does not make co-resident: measured wrong from n=2049 (the first size
        # past one 2048 tile), losing elements. `cumsum_wrapper` in this file
        # already does its cross-tile scan as separate launches and is correct
        # at eight tiles; this follows it.
        radix_sort_tile_counts_kernel[(m * sweep_grid_n, grid_r)](
            arr_in, status, m, n, sweep_grid_n,
            SWEEP_TILE, TILE_R, k_bits, i * k_bits, descending)
        radix_sort_tile_prefix_kernel[(m * num_bins,)](
            status, tile_prefix, sweep_grid_n, sweep_grid_n,
            TILES_POW2=triton.next_power_of_2(sweep_grid_n))
        radix_sort_scatter_kernel[(m * sweep_grid_n, grid_r)](
            arr_in, indices_in, arr_out, indices_out,
            ex_cumsum, tile_prefix,
            n_passes, i, i * k_bits, m, n, sweep_grid_n,
            SWEEP_TILE, TILE_R, k_bits, descending)
        arr_in, arr_out = arr_out, arr_in
        indices_in, indices_out = indices_out, indices_in

    # Values keep x's shape (clone lineage); indices were built (m, n) —
    # fold them back to the input shape for the aten::sort contract.
    return arr_in, indices_in.view(x.shape)


def logical_xor_wrapper(a, b) :
    # The universal binary contract (dtype align, device align, BROADCAST,
    # contiguous) like logical_and / logical_or: the former form read a
    # broadcast operand flat past its end — 34 % of the bank's broadcast
    # case was wrong (found by the reference bank, 2026-09-05).
    a, b, output, n, dev_ctx, scalar = _core._prepare_comparison(a, b)
    if scalar:
        b = NBXTensor.empty_like(a).fill_(b)
    logical_xor_kernel[_1d_grid(n)](a, b, output, n, **_ew_meta(n))
    return output


def bitwise_left_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        _set_device(a)
        bitwise_left_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), **_ew_meta(a.numel()))
    else:
        from nbxtensor.ops.bitwise_left_shift import bitwise_left_shift_scalar_kernel
        _set_device(a)
        bitwise_left_shift_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), int(b), **_ew_meta(a.numel()))
    return output


def bitwise_right_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        _set_device(a)
        bitwise_right_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), **_ew_meta(a.numel()))
    else:
        from nbxtensor.ops.bitwise_right_shift import bitwise_right_shift_scalar_kernel
        _set_device(a)
        bitwise_right_shift_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), int(b), **_ew_meta(a.numel()))
    return output


def addr_wrapper(input, vec1, vec2,
                 beta: float = 1.0, alpha: float = 1.0) :
    """Outer product add: out = beta*input + alpha*(vec1 outer vec2)."""
    M = vec1.numel()
    N = vec2.numel()
    inp_c = input.contiguous()
    v1_c = vec1.contiguous()
    v2_c = vec2.contiguous()
    output = NBXTensor.empty((M, N), device=input.device, dtype=input.dtype)
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_N = min(32, triton.next_power_of_2(N))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _set_device(inp_c)
    addr_kernel[grid](
        inp_c, v1_c, v2_c, output,
        beta, alpha, M, N,
        inp_c.stride(0) if inp_c.ndim >= 2 else 0,
        inp_c.stride(1) if inp_c.ndim >= 2 else 1,
        v1_c.stride(0), v2_c.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N)
    return output


def clamp_min_wrapper(x, min_val) :
    """Clamp minimum: out = max(x, min_val)."""
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    if isinstance(min_val, NBXTensor):
        min_val = min_val.item()
    _set_device(x)
    clamp_min_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), float(min_val), **_ew_meta(x.numel()))
    return output


# Bounds for the two-stage top-k, measured on sm_70. See the comment at
# the chunk-widening loop in topk_wrapper for the numbers behind them.
_TOPK_MERGE_TILE_TARGET = 2048   # widen chunks to keep the merge tile here
_TOPK_MERGE_TILE_MAX = 8192      # past this, refuse rather than stall
_TOPK_CHUNK_MAX = 8192           # stage-1 window still flat at this width


def topk_wrapper(x, k: int, dim: int = -1,
                 largest: bool = True, sorted: bool = True):
    """Top-k via 2-stage Triton pipeline. Stage 1: per-chunk top-k. Stage 2: merge via bitonic sort.
    Triton kernels + NBXTensor allocation."""
    from nbxtensor.ops.topk import topk_stage1_kernel, topk_stage2_kernel
    import math as _math

    if dim < 0:
        dim = dim + x.ndim
    assert dim == x.ndim - 1, "topk currently supports last dim only"

    # Kernel uses _get_finfo_val which handles fp16/bf16/fp32.
    # Cast fp32→fp16 for the kernel (it upcasts to fp32 internally anyway),
    # then cast output values back.
    input_dtype = x._dtype
    if input_dtype == NBXDtype.float32:
        x = x.to(NBXDtype.float16)

    descending = largest
    topk_n = x.shape[dim]
    batch_size = x.numel() // topk_n

    # Heuristic chunk size
    if topk_n < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    # Keep the stage-2 merge tile small.
    #
    # Stage 2 sorts next_power_of_2(chunk_num * k) elements inside ONE
    # Triton program, fully unrolled over log2(tile) bitonic stages, so
    # its FIRST-COMPILE cost explodes with the tile: measured on sm_70,
    # 2048 -> 15 s, 4096 -> 48 s, 8192 -> over 150 s, and 16384 did not
    # finish in the budget. That cost is paid once per (k, N, tile)
    # triple and then cached, but it lands on the first token of a
    # request, where minutes of apparent hang is indistinguishable from
    # a deadlock. Qwen2's 152k vocabulary with top_k=64 sat squarely in
    # that hole.
    #
    # Widening the chunk trades the merge tile for stage-1 work, which
    # is a k-iteration reduction rather than a sort — measured flat at
    # 1.1 s up to CHUNK 8192. Partitioning does not change the answer:
    # the top-k of the per-chunk top-k's IS the global top-k for any
    # partition. Only the index chosen among exactly-tied values can
    # move, so this is a FLOATING-class change, not an EXACT one.
    while (chunk_size < _TOPK_CHUNK_MAX
           and triton.cdiv(topk_n, chunk_size) * k > _TOPK_MERGE_TILE_TARGET):
        chunk_size *= 2

    chunk_num = triton.cdiv(topk_n, chunk_size)

    if chunk_num > 1 and chunk_num * k > _TOPK_MERGE_TILE_MAX:
        raise ValueError(
            f"topk: k={k} over {topk_n} elements needs a merge tile of "
            f"{triton.next_power_of_2(chunk_num * k)} even at the widest "
            f"chunk ({_TOPK_CHUNK_MAX}), past the {_TOPK_MERGE_TILE_MAX} "
            f"bound where the bitonic merge's first-compile cost runs to "
            f"minutes. Reduce k, or extend topk with a hierarchical "
            f"reduction stage.")

    # Stage 1 buffers
    s1_vals = NBXTensor.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    s1_idxs = NBXTensor.empty(batch_size * chunk_num * k, device=x.device, dtype=NBXDtype.int64)

    # Stage 1: per-chunk top-k (iterative argmax → already sorted descending
    # for largest=True).
    _set_device(s1_vals)
    topk_stage1_kernel[batch_size, chunk_num](
        s1_vals, s1_idxs, x.contiguous(),
        k, topk_n, chunk_size, descending)

    out_shape = x.shape[:-1] + (k,)

    # Single-chunk fast path: stage 1 already produces the correct top-k in
    # sorted order, so stage 2's bitonic merge is redundant. Skipping it also
    # avoids a kernel bug where the BLOCK=next_pow2(k) padding corrupts the
    # last entries for power-of-2-misaligned k (e.g. k=6 → BLOCK=8, with
    # DeepSeek-MoE's 64-expert softmax the 2 pad slots leaked near-zero
    # garbage into the real top-k, producing negative scores for slots 5–6
    # and a 28× routed-MoE magnitude collapse). Also saves one kernel launch
    # for every MoE model whose num_experts ≤ chunk_size.
    if chunk_num == 1:
        s1_vals = s1_vals.reshape(out_shape)
        s1_idxs = s1_idxs.reshape(out_shape)
        if input_dtype == NBXDtype.float32:
            s1_vals = s1_vals.to(NBXDtype.float32)
        return s1_vals, s1_idxs

    # Stage 2 buffers
    s2_vals = NBXTensor.empty(out_shape, device=x.device, dtype=x.dtype)
    s2_idxs = NBXTensor.empty(out_shape, device=x.device, dtype=NBXDtype.int64)

    # Stage 2: merge chunks via bitonic sort
    stage2_n = chunk_num * k
    BLOCK = triton.next_power_of_2(stage2_n)
    _set_device(s2_vals)
    topk_stage2_kernel[batch_size,](
        s2_vals, s2_idxs, s1_vals, s1_idxs,
        dim, k, stage2_n, BLOCK, descending)

    # Cast values back to original dtype if we downcast for the kernel
    if input_dtype == NBXDtype.float32:
        s2_vals = s2_vals.to(NBXDtype.float32)

    return s2_vals, s2_idxs


def trace_wrapper(x) :
    """Sum of diagonal elements."""
    assert x.ndim >= 2
    n = min(x.shape[-2], x.shape[-1])
    out = NBXTensor.empty(1, dtype=NBXDtype.float32, device=x.device)
    _set_device(x.contiguous())
    trace_kernel[(1,)](
        x.contiguous(), out, n,
        x.stride(-2), x.stride(-1),
        BLOCK_SIZE=triton.next_power_of_2(n))
    return out.squeeze().to(x.dtype)


def _stream_draw(kind, shape, dtype, device):
    """A seeded run's draw (nbxtensor/random/rng_stream: the vendor's CPU stream, continued in consumption order — the same
    values the compiled branch draws), as an NBXTensor of `dtype` on `device`; None when the run has no seed."""
    from nbxtensor.random import rng_stream
    if not rng_stream.active():
        return None
    arr = rng_stream.next_normal(shape) if kind == "normal" else rng_stream.next_uniform(shape)
    if isinstance(device, NBXTensor):            # a tensor is its own device context (NBXTensor.device)
        DeviceAllocator.set_device(int(device._device_idx))
    elif device is not None and ":" in str(device):
        DeviceAllocator.set_device(int(str(device).split(":")[1]))
    out = NBXTensor.from_numpy(arr)
    return out if out._dtype == dtype else out.to(dtype)


def bernoulli_wrapper(x, p: float = 0.5, **kwargs) :
    """bernoulli(tensor) → 0/1 values based on probabilities."""
    from nbxtensor.ops.rand_op import rand_kernel
    x = _core._ensure_cuda(x)
    output = NBXTensor.empty_like(x)
    # Generate uniform random values
    temp = NBXTensor.empty_like(x, dtype=NBXDtype.float32)
    n = temp.numel()
    if n == 0:
        return output
    seed = _rnd._draw_seed()
    rand_kernel[_1d_grid(n)](temp, n, seed, **_ew_meta(n))
    # Compare with probability: if rand < p → 1.0, else → 0.0
    # If x is the probability tensor, compare temp < x
    if x.is_floating_point():
        result = (temp < x.float()).to(output.dtype)
    else:
        result = (temp < p).to(output.dtype)
    output.copy_(result)
    return output


# ===========================================================================
# FOLD / UNFOLD_BACKWARD — compute ops
# ===========================================================================

def fold_wrapper(*args, **kwargs) :
    """fold — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] fold (col2im) has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/fold_op.py."
    )


# ===========================================================================
# ANGLE — atan2(imag, real) for complex tensors
# ===========================================================================

# ===========================================================================
# SCALED DOT-PRODUCT ATTENTION — Dao-AILab Flash Attention v2 Triton kernel
#
# Reference: Dao-AILab/flash-attention flash_attn_triton.py
# Kernel: ops/flash_attention.py (extracted from same source)
# ===========================================================================

# Cache of zero-bias buffers for the no-mask flash attention path. The
# kernel always reads bias from memory (BIAS_TYPE="vector"|"matrix" only)
# to guarantee bit-equivalent MMA selection across configurations — a
# tl.zeros in-register was empirically shown to produce a different MMA
# than a tl.load on Volta SIMT (and likely on AMD CDNA), so every call
# without an explicit mask is routed through tl.load via this shared
# zero buffer. Cache key = (device_idx, seqlen_q, seqlen_k, dtype). One
# allocation per distinct shape per device — typically caps at 2-4 keys
# per inference session. Memory: ~4-16 MB total for typical attention
# shapes, negligible vs model weights.
_zero_bias_cache: dict = {}
_causal_bias_cache: dict = {}


def _get_zero_bias(device_idx, seqlen_q, seqlen_k, dtype):
    """Zero additive bias for the no-mask flash path, as a STRIDE-0
    BROADCAST view over the query axis: physical storage is a single
    [1, seqlen_k] zero row, expanded to [seqlen_q, seqlen_k] with
    stride_q = 0. The kernel reads the bias through its strides and only
    requires unit stride on the KEY axis (offs_n is indexed with implicit
    stride 1), so the query axis can broadcast freely — exactly the same
    trick the wrapper already uses for key-padding masks. A materialized
    [Sq, Sk] zero matrix is fatal at video-native scale: Allegro 720p×88f
    self-attention is 79200×79200 fp32 = 25 GB for a tensor of ZEROS
    (first real trigger of this path at native scale, 2026-07-04)."""
    key = (device_idx, seqlen_q, seqlen_k, dtype)
    cached = _zero_bias_cache.get(key)
    if cached is not None:
        return cached
    DeviceAllocator.set_device(device_idx)
    row = NBXTensor.zeros((1, seqlen_k), dtype=dtype,
                          device=f"cuda:{device_idx}")
    bias = row.expand(seqlen_q, seqlen_k)
    _zero_bias_cache[key] = bias
    return bias


def _get_causal_bias(device_idx, seqlen_q, seqlen_k, dtype):
    """Return a cached causal additive mask [seqlen_q, seqlen_k]:
    0.0 for positions k <= q + (seqlen_k - seqlen_q), -inf otherwise.

    Materialized via memory so the flash kernel reads it through
    tl.load — same physical path as user-provided explicit masks.
    Required for numerical parity: Triton compiles tl.where(constexpr,
    0, -inf) and tl.load(memory) with different IR optimization
    passes, propagating to different MMA selections downstream.
    Empirical: fp32 hd=256 IS_CAUSAL via constexpr produced cosine
    0.937; via memory produces 0.999997.
    """
    key = (device_idx, seqlen_q, seqlen_k, dtype)
    cached = _causal_bias_cache.get(key)
    if cached is not None:
        return cached
    DeviceAllocator.set_device(device_idx)
    # Build via tril + masked_fill: ones[Q,K] -> tril(diagonal=offset)
    # gives 1 in valid positions, 0 in masked positions. masked_fill
    # those zero positions with -inf, the rest stays as 0 baseline.
    offset = seqlen_k - seqlen_q
    base = NBXTensor.zeros((seqlen_q, seqlen_k), dtype=dtype,
                            device=f"cuda:{device_idx}")
    ones = add(base, 1.0)
    tril_mask = _shp.tril_wrapper(ones, diagonal=offset)
    inverted = eq(tril_mask, 0.0)
    bias = _idx.masked_fill(base, inverted, float('-inf'))
    _causal_bias_cache[key] = bias
    return bias


def clear_bias_caches() -> None:
    """Evict the module-global attention bias caches.

    `_causal_bias_cache` stores MATERIALIZED [Sq, Sk] additive masks keyed
    by (device, Sq, Sk, dtype); `_zero_bias_cache` stores stride-0 broadcast
    rows. Both persist for the process lifetime otherwise. In a cold CLI run
    that is negligible (a handful of shapes), but a warm serve that sweeps
    many sequence lengths accumulates one materialized [Sq, Sk] tensor per
    distinct length — unbounded growth that the runtime never reclaimed
    (audit #2 F7). This hook is called from the single memory-cleanup
    surface (`MemoryManager.unload_weights`), so eviction rides the same
    phase boundaries as weight unload (encoder→loop→VAE, request teardown).
    The caches rebuild lazily on next use, so clearing is always safe;
    within a diffusion loop `unload_weights` is not called per step, so the
    hot reuse of a single bias table is untouched.

    R33-pure: clears two Python dicts of NBXTensors — no torch, no compute.
    The dropped NBXTensor refs trigger their finalizers (cudaFree) once the
    caller's per-device sync in `unload_weights` has completed.
    """
    _zero_bias_cache.clear()
    _causal_bias_cache.clear()


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


_DEFERRED_DRAIN_CLIFF_DEFAULT = 2_000_000_000   # 2 GB — throughput ceiling


def deferred_drain_policy():
    """Return the triton deferred-free drain policy as a 4-tuple
    `(cliff_bytes, floor_bytes, pressure_reserve_bytes, freecheck_interval)`.

    The triton hot loop batches every dead/overwritten NBXTensor into a
    `_deferred` list, drained (after a device sync) when a threshold trips —
    the amount by which the triton live watermark can exceed the compiled
    free-at-last-use watermark (compiled frees at the kill point via torch's
    caching allocator). The historical policy was a flat 2 GB cap (audit #2
    F1). Draining is always correctness-safe (sync-before-free) but NOT free:
    with the caching pool off (default), every drain is a cudaFree→cudaMalloc
    cycle, so draining EAGERLY on a model that has headroom is pure churn
    (measured: a fixed 650 MiB cap regressed CogVideoX-2b VAE decode ~49% for
    no capacity benefit — the device had ~21 GB free).

    Policy is therefore PRESSURE-SENSITIVE:
    - `cliff`   — the queue is ALWAYS drained at this size (throughput ceiling,
      bounds worst-case retention). On a device with headroom this is the ONLY
      trigger, so runtime is identical to the historical 2 GB behavior.
    - `floor`   — below this the queue is never drained mid-run (small-model /
      ample-headroom fast path).
    - `reserve` — when the queue is >= floor, the engine probes driver-free
      memory; if free < reserve the device is near capacity and the queue is
      drained early to keep the watermark from reaching OOM. Draining more
      (slower) to FIT beats an OOM — capacity is the hard constraint.
    - `interval`— minimum ops between free probes (bounds the probe rate; the
      probe is `DeviceAllocator.device_free_bytes` = cudaMemGetInfo, no sync,
      the sanctioned feasibility API — NOT a kernel-config source).

    Data-driven (R7/R24) from `config/vendors/<vendor>/<arch>.yml memory:`,
    keyed on the detected device's vendor+architecture. Safe fallback (no
    profile / keys absent / exception): `cliff = 2 GB, floor = cliff,
    reserve = 0` → the pressure branch can never trip (`free < 0` is
    impossible) and the queue only ever drains at the 2 GB cliff = the exact
    prior behavior.
    """
    cliff = _DEFERRED_DRAIN_CLIFF_DEFAULT
    fallback = (cliff, cliff, 0, 1 << 30)
    prof = _core.get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return fallback
    try:
        from nbxtensor.host import need as _need   # the application's core.config.loader (nbxtensor.host)
        get_vendor_config = _need("vendor_config")
        _brand = devices[0].brand
        _vendor = getattr(_brand, "value", _brand)
        cfg = get_vendor_config(_vendor, devices[0].architecture)
        # The ONE reading of the policy, shared with the plan that prices the cliff
        # (`core/prism/memory_budget.deferred_drain_policy_of`): the run and the plan
        # cannot read two figures.
        from nbxtensor.host import need as _need   # the application's core.prism.memory_budget (nbxtensor.host)
        deferred_drain_policy_of = _need("deferred_drain_policy_of")
        return deferred_drain_policy_of(cfg.get("memory", {}))
    except Exception:
        return fallback


def _try_decode_vec(q, k, v, attn_mask, softmax_scale,
                    batch, nheads, nheads_k, seqlen_k, headdim, q_round=None):
    """Route guard for the vector decode kernel: returns the output or
    None when the shape/mask is outside the kernel's contract."""
    if not _lk.decode_vec_takes(headdim, v.shape[3],
                                None if attn_mask is None else attn_mask.numel(), seqlen_k):
        return None
    bias = attn_mask.reshape(seqlen_k) if attn_mask is not None else None
    return _attn._decode_attn_vec(q, k, v, bias, softmax_scale,
                            batch, nheads, nheads_k, seqlen_k, headdim, q_round=q_round)


def angle_wrapper(x) :
    """angle(complex_tensor) → atan2(imag, real).

    For real inputs: 0 where x >= 0, pi where x < 0.
    Uses Triton atan2 via libdevice.
    """
    x = _core._ensure_cuda(x)

    if x.is_complex():
        # Complex: atan2(imag, real) element-wise (libdevice, fp32).
        from nbxtensor.ops.atan2 import atan2_kernel
        real = x.real.contiguous()
        imag = x.imag.contiguous()
        out = NBXTensor.empty(tuple(real.shape), dtype=NBXDtype.float32,
                              device=real.device)
        _set_device(real)
        n = real.numel()
        atan2_kernel[_1d_grid(n)](imag, real, out, n, **_ew_meta(n))
        return out
    else:
        # Real input: angle is 0 where x >= 0, pi where x < 0. Not on the
        # complex-FFT path (kept minimal: zeros for the x>=0 common case).
        return NBXTensor.zeros_like(x)


# ===========================================================================
# FFT — Pure Triton Cooley-Tukey radix-2 implementation
# ===========================================================================

def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ===========================================================================
# COMPLEX — tensor creation from real/imag parts
# ===========================================================================

def complex_wrapper(real, imag) :
    """complex(real, imag) → complex64 tensor (interleaved [real, imag] pairs).

    Builds the NBX complex64 representation: stack real/imag on a new last dim
    → [..,N,2] → reinterpret as complex64 [..,N] (NBXTensor.view_as_complex).
    R33-pure (cat + view; no torch)."""
    real = _core._ensure_cuda(real)
    imag = _core._ensure_cuda(imag)
    if real.nbx_dtype != NBXDtype.float32:
        real = real.to(NBXDtype.float32)
    if imag.nbx_dtype != NBXDtype.float32:
        imag = imag.to(NBXDtype.float32)
    stacked = NBXTensor.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1).contiguous()
    return stacked.view_as_complex()


# ============================================================================
# LSTM — triton-pure aten::lstm (mirror of F.lstm / cuDNN, the compiled ref).
# Level-1 R33-pure: NBXTensor + Triton wrappers only (zero torch, zero NumPy
# compute). Per-step recurrence assembled from mm/sigmoid/tanh/add/mul; a fused
# @triton.jit LSTM is the deferred level-2 optimisation. Replaces the parakeet
# NumPy LSTM (triton/flow/rnnt.py) once that flow is wired to it.
# ============================================================================
def _lstm_run_direction(x, w_ih_t, w_hh_t, b_ih, b_hh, h, c, hidden, reverse):
    """One LSTM direction over the time axis.

    x         : [B, T, I]   w_ih_t : [I, 4H]   w_hh_t : [H, 4H]
    b_ih/b_hh : [4H] or None        h, c   : [B, H]
    returns (out [B,T,H], h_last [B,H], c_last [B,H]).

    Gate order i/f/g/o (PyTorch), mirrors triton/flow/rnnt.py:_lstm_cell_np:
        gates = x@W_ih.T + b_ih + h@W_hh.T + b_hh
        i=sigmoid(g[:H]); f=sigmoid(g[H:2H]); g_=tanh(g[2H:3H]); o=sigmoid(g[3H:])
        c = f*c + i*g_;  h = o*tanh(c)
    mm/bmm accumulate in fp32 (Triton tl.dot) → matches cuDNN's fp16 math.

    The fp16 precision contract does NOT reach the recurrence: the calibration that grants it observed the graph's
    ops, and an LSTM is ONE graph op whose internal products it never saw — their state is carried through T steps,
    and fp16 stores there drift the recurrence (Kokoro-82M's predictor, 2026-10-26: the speech became gibberish under
    the contract, correct with it off inside this function only). The products run under the conservative store.
    """
    _contract = _core.get_activations_fp16_safe()
    _core.set_activations_fp16_safe(False)
    try:
        return _lstm_run_direction_steps(x, w_ih_t, w_hh_t, b_ih, b_hh, h, c, hidden, reverse)
    finally:
        _core.set_activations_fp16_safe(_contract)


def _lstm_run_direction_steps(x, w_ih_t, w_hh_t, b_ih, b_hh, h, c, hidden, reverse):
    """`_lstm_run_direction`'s body, under the conservative store."""
    T = x.shape[1]
    H = hidden
    Wx = _la.matmul_wrapper(x, w_ih_t)                       # [B, T, 4H]
    if b_ih is not None:
        Wx = add(Wx, b_ih)                                # broadcast [4H]
    outs = [None] * T
    step_range = range(T - 1, -1, -1) if reverse else range(T)
    for t in step_range:
        gates = add(Wx.select(1, t), _la.matmul_wrapper(h, w_hh_t))   # [B, 4H]
        if b_hh is not None:
            gates = add(gates, b_hh)
        i_gate = sigmoid_wrapper(gates.narrow(1, 0, H))
        f_gate = sigmoid_wrapper(gates.narrow(1, H, H))
        g_gate = tanh_wrapper(gates.narrow(1, 2 * H, H))
        o_gate = sigmoid_wrapper(gates.narrow(1, 3 * H, H))
        c = add(mul(f_gate, c), mul(i_gate, g_gate))      # f*c + i*g_
        h = mul(o_gate, tanh_wrapper(c))                  # o*tanh(c)
        outs[t] = h.unsqueeze(1)                          # [B, 1, H]
    return NBXTensor.cat(outs, dim=1), h, c               # [B, T, H]


def lstm_wrapper(input_, hx, params, has_biases=True, num_layers=1,
                 dropout=0.0, train=False, bidirectional=False,
                 batch_first=False):
    """Triton-pure aten::lstm. See _lstm_run_direction for the cell math.

    input_ : [B,T,I] (batch_first) or [T,B,I] or [T,I]
    hx     : [h0, c0], each [num_layers*num_dir, B, H]
    params : flat list per layer per direction: [W_ih, W_hh, b_ih, b_hh] (biases)
             or [W_ih, W_hh] (no biases); reverse direction follows forward.
    returns (output [B,T,num_dir*H], h_n, c_n).
    """
    num_dir = 2 if bidirectional else 1
    ppl = 4 if has_biases else 2
    stride = ppl * num_dir

    x = input_
    if not batch_first and x.ndim == 3:
        x = x.transpose(0, 1).contiguous()                # [T,B,I] -> [B,T,I]
    if x.ndim == 2:
        x = x.unsqueeze(0)                                # [T,I] -> [1,T,I]
    cdt = x.dtype

    h0_all, c0_all = hx[0], hx[1]                          # [layers*dir, B, H]
    h_parts, c_parts = [], []
    layer_in = x
    for layer in range(int(num_layers)):
        dir_outs = []
        for d in range(num_dir):
            base = layer * stride + d * ppl
            w_ih = params[base].to(cdt)
            w_hh = params[base + 1].to(cdt)
            H = w_hh.shape[1]                              # w_hh: [4H, H]
            b_ih = params[base + 2].to(cdt) if has_biases else None
            b_hh = params[base + 3].to(cdt) if has_biases else None
            si = layer * num_dir + d
            h = h0_all.select(0, si).to(cdt)               # [B, H]
            c = c0_all.select(0, si).to(cdt)
            out_d, h_last, c_last = _lstm_run_direction(
                layer_in,
                w_ih.transpose(0, 1).contiguous(),
                w_hh.transpose(0, 1).contiguous(),
                b_ih, b_hh, h, c, H, reverse=(d == 1))
            dir_outs.append(out_d)
            h_parts.append(h_last.unsqueeze(0))
            c_parts.append(c_last.unsqueeze(0))
        layer_in = (NBXTensor.cat(dir_outs, dim=-1).contiguous()
                    if num_dir == 2 else dir_outs[0])
    h_n = NBXTensor.cat(h_parts, dim=0)                    # [layers*dir, B, H]
    c_n = NBXTensor.cat(c_parts, dim=0)
    return layer_in, h_n, c_n


lstm_wrapper.self_manages_dtype = True
