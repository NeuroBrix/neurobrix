"""Triton Kernel Wrappers — triton mode execution layer.

Each wrapper:
  1. Receives NBXTensor
  2. Allocates output via NBXTensor.empty (cudaMalloc)
  3. Launches @triton.jit kernel
  4. Returns NBXTensor

Dependencies: triton, NBXTensor. Used exclusively by dispatch.py.
"""

import triton

from .nbx_tensor import NBXTensor, NBXDtype, _broadcast_shapes

# === Activations ===

from .ops.relu import relu_forward_kernel
from .ops.silu import silu_forward_kernel
from .ops.gelu import gelu_forward_kernel
from .ops.sigmoid import sigmoid_forward_kernel
from .ops.tanh import tanh_forward_kernel
from .ops.hardsigmoid import hardsigmoid_forward_kernel
from .ops.hardswish import hardswish_forward_kernel
from .ops.leaky_relu import leaky_relu_forward_kernel
from .ops.elu import elu_forward_kernel
from .ops.mish import mish_forward_kernel
from .ops.selu import selu_forward_kernel

# === Unary element-wise ===

from .ops.neg import neg_forward_kernel
from .ops.exp import exp_forward_kernel
from .ops.sin import sin_forward_kernel
from .ops.cos import cos_forward_kernel
from .ops.rsqrt import rsqrt_forward_kernel
from .ops.sqrt import sqrt_forward_kernel
from .ops.abs import abs_forward_kernel
from .ops.log import log_forward_kernel
from .ops.reciprocal import reciprocal_forward_kernel
from .ops.pow import pow_forward_kernel
from .ops.clamp import clamp_forward_kernel
from .ops.erf import erf_forward_kernel
from .ops.copy import copy_forward_kernel

# === Binary element-wise ===

from .ops.add import add_forward_kernel, add_scalar_kernel
from .ops.mul import mul_forward_kernel, mul_scalar_kernel
from .ops.div import div_forward_kernel, div_scalar_kernel
from .ops.sub import sub_forward_kernel, rsub_forward_kernel
from .ops.where import where_forward_kernel
from .ops.maximum import maximum_forward_kernel
from .ops.minimum import minimum_forward_kernel
from .ops.masked_fill import masked_fill_forward_kernel

# === Comparisons ===

from .ops.gt import gt_forward_kernel, gt_scalar_kernel
from .ops.ge import ge_forward_kernel, ge_scalar_kernel
from .ops.lt import lt_forward_kernel, lt_scalar_kernel
from .ops.eq import eq_forward_kernel, eq_scalar_kernel
from .ops.ne import ne_forward_kernel, ne_scalar_kernel

# === Normalization ===

from .ops.layernorm import layer_norm_forward_kernel
from .ops.rmsnorm import rms_norm_forward_kernel

# === Softmax ===

from .ops.softmax import softmax_forward_kernel

# === Matmul ===

from .ops.matmul import matmul_kernel, addmm_kernel

# === Embedding ===

from .ops.embedding import embedding_kernel

# === Reductions ===

from .ops.sum import sum_forward_kernel
from .ops.mean import mean_forward_kernel
from .ops.amax import amax_forward_kernel

# === GLU ===

from .ops.glu import glu_forward_kernel

# === New kernels (Phase 2) ===

from .ops.le import le_forward_kernel, le_scalar_kernel
from .ops.softplus import softplus_forward_kernel
from .ops.dropout import dropout_inference_kernel
from .ops.upsample_nearest2d import upsample_nearest2d_kernel
from .ops.groupnorm import group_norm_forward_kernel
from .ops.index_select import index_select_kernel
from .ops.triu import triu_kernel, triu_batch_kernel
from .ops.tril import tril_kernel, tril_batch_kernel
from .ops.argmax import argmax_kernel_1, argmax_kernel_2, argmax_kernel_inner
from .ops.argmin import argmin_kernel_1, argmin_kernel_2, argmin_kernel
from .ops.conv2d import conv2d_forward_kernel
from .ops.batch_norm import batch_norm_forward_kernel
from .ops.cumsum import scan_part_sum_kernel, add_base_sum_kernel
from .ops.scatter_op import scatter_kernel, scatter_add_kernel, scatter_reduce_amax_kernel, scatter_reduce_amin_kernel
from .ops.gather_op import gather_kernel
from .ops.avg_pool2d import avg_pool2d_forward_kernel
from .ops.max_pool2d import max_pool2d_forward_kernel
from .ops.cross_entropy import cross_entropy_loss_forward_kernel

# === Simple element-wise (Phase 3) ===

from .ops.exp2 import exp2_forward_kernel
from .ops.tan import tan_forward_kernel
from .ops.celu import celu_forward_kernel
from .ops.log_sigmoid import log_sigmoid_forward_kernel
from .ops.isfinite import isfinite_forward_kernel
from .ops.isinf import isinf_forward_kernel
from .ops.isnan import isnan_forward_kernel
from .ops.nan_to_num import nan_to_num_forward_kernel
from .ops.threshold import threshold_forward_kernel

# === Phase 4: Remaining kernels ===

from .ops.addcdiv import addcdiv_forward_kernel
from .ops.addcmul import addcmul_forward_kernel
from .ops.all_reduce import all_kernel_mid, all_kernel_result, all_kernel_dim
from .ops.any_reduce import any_kernel_mid, any_kernel_result, any_kernel_dim
from .ops.bitwise_and import bitwise_and_forward_kernel
from .ops.bitwise_or import bitwise_or_forward_kernel
from .ops.bitwise_not import bitwise_not_forward_kernel
from .ops.logical_and import logical_and_forward_kernel
from .ops.logical_or import logical_or_forward_kernel
from .ops.logical_not import logical_not_forward_kernel
from .ops.conv1d import conv1d_forward_kernel
from .ops.lerp import lerp_tensor_forward_kernel, lerp_scalar_head_kernel, lerp_scalar_tail_kernel
from .ops.dot_op import dot_kernel_small, dot_kernel_partial, dot_kernel_reduce
from .ops.mv_op import mv_kernel
from .ops.flip_op import flip_1d_kernel, flip_strided_kernel
from .ops.cat_op import cat_copy_kernel_4
from .ops.fill_op import fill_kernel
from .ops.arange_op import arange_kernel
from .ops.prod import prod_kernel_mid, prod_kernel_result, prod_kernel
from .ops.min_reduce import min_kernel_mid, min_kernel_result, min_kernel
from .ops.max_reduce import max_kernel_mid, max_kernel_result, max_kernel
from .ops.baddbmm_op import baddbmm_kernel
from .ops.addmv_op import addmv_kernel
from .ops.mse_loss import mse_loss_partial_kernel, mse_loss_reduce_kernel, mse_loss_none_kernel
from .ops.nllloss import nll_loss_forward_kernel
from .ops.std import std_map_kernel, std_reduce_kernel, std_dim_kernel
from .ops.var import var_kernel_1, var_kernel_2, var_welford_kernel
from .ops.index_add import index_add_kernel
from .ops.sort_op import radix_sort_histogram_kernel, radix_sort_sweep_kernel

# === Phase 5: RoPE, spatial, RNG, remaining ===

from .ops.rope import rope_forward_kernel
from .ops.pixel_shuffle import pixel_shuffle_kernel
from .ops.pixel_unshuffle import pixel_unshuffle_kernel
from .ops.upsample_bilinear2d import upsample_bilinear2d_kernel
from .ops.adaptive_avg_pool2d import adaptive_avg_pool2d_kernel
from .ops.conv_depthwise2d import conv_depthwise2d_kernel
from .ops.scaled_softmax import scaled_softmax_kernel
from .ops.vector_norm import (
    l2_norm_kernel, l1_norm_kernel, linf_norm_kernel, l0_norm_kernel, lp_norm_kernel,
    l2_norm_pass1_kernel, l2_norm_pass2_kernel,
    linf_norm_pass1_kernel, linf_norm_pass2_kernel,
)
from .ops.addr import addr_kernel
from .ops.logical_xor import logical_xor_kernel
from .ops.bitwise_left_shift import bitwise_left_shift_kernel
from .ops.bitwise_right_shift import bitwise_right_shift_kernel
from .ops.rand_op import rand_kernel, randn_kernel
from .ops.clamp_min import clamp_min_forward_kernel
from .ops.conv_transpose2d import conv_transpose2d_kernel
from .ops.cummax import scan_part_max_kernel
from .ops.cummin import scan_part_min_kernel
from .ops.count_nonzero import count_nonzero_kernel
from .ops.trace_op import trace_kernel
from .ops.linspace_op import linspace_kernel
from .ops.full_op import full_kernel
from .ops.stack_op import stack_copy_kernel
from .ops.repeat_op import repeat_1d_kernel

# === Phase 6: Video + Audio ops ===

from .ops.upsample_nearest3d import upsample_nearest3d_kernel
from .ops.weight_norm import weight_norm_kernel_first, weight_norm_kernel_last
from .ops.repeat_interleave import repeat_interleave_tensor_kernel


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_EW_BLOCK = 1024
_EW_WARPS = 4


def _1d_grid(n_elements):
    """Grid for element-wise (memory-bound) kernels. Fixed BLOCK_SIZE=1024."""
    return (triton.cdiv(n_elements, _EW_BLOCK),)


def _batch_block(batch_dim, feat_dim):
    """Replicate batch_block_heuristic from _configs.py for grid computation.

    Must match the @triton.heuristics decorator on norm/softmax kernels.
    """
    if feat_dim < 64:
        return min(max(1, triton.next_power_of_2(batch_dim // 1024)), 128)
    return 1


# --- Fixed block sizes for kernels without @triton.autotune ---
_MM_BM, _MM_BN, _MM_BK = 64, 64, 32      # matmul / addmm / baddbmm
_MM_GROUP = 8                              # matmul GROUP_M
_RED_BM, _RED_BN = 8, 1024                 # reduction kernels (prod, min, max, std, var, norm, all, any)
_MV_BN, _MV_BM = 64, 256                  # mv / addmv  (BLOCK_N rows, BLOCK_M reduction tile)
_TRIU_MBS, _TRIU_NBS = 64, 512            # triu / tril 2D
_TRIU_BATCH_BS, _TRIU_MN_BS = 32, 1024    # triu / tril batch
_CONV_BHW, _CONV_OUTF, _CONV_INF = 64, 64, 32  # conv2d
_POOL_BH, _POOL_BW = 4, 16                # avg_pool2d / max_pool2d
_BILINEAR_BX, _BILINEAR_BY = 16, 16       # upsample_bilinear2d
_WNORM_BM, _WNORM_BN = 64, 256           # weight_norm


# ===========================================================================
# UNIVERSAL LAUNCH LAYER
#
# Handles broadcasting, device context, scalar tensors, and contiguity
# for ALL Triton kernel wrappers. One place, all concerns, all ops.
# ===========================================================================

def _to_scalar(x):
    """Extract Python scalar from 0-d tensor or scalar."""
    if isinstance(x, (int, float, bool)):
        return x
    if hasattr(x, 'ndim') and x.ndim == 0:
        return x.item()
    if hasattr(x, 'numel') and x.numel() == 1:
        return x.item()
    return x


def _is_scalar(x):
    """Check if x is a scalar (Python number or 0-d tensor)."""
    if isinstance(x, (int, float, bool)):
        return True
    if hasattr(x, 'ndim') and x.ndim == 0:
        return True
    return False


def _ensure_cuda(t):
    """Ensure CUDA runtime device matches tensor device."""
    _set_device(t)
    return t


def _set_device(t):
    """Ensure CUDA device context matches tensor device.

    Triton JIT reads torch.cuda.current_device() internally.
    Reference: triton-lang/triton#2925
    """
    if hasattr(t, '_device_idx'):
        from .nbx_tensor import DeviceAllocator
        DeviceAllocator.ensure_triton_device(t._device_idx)


def _prepare_unary(x):
    """Prepare a single tensor for kernel launch."""
    x = x.contiguous()
    _set_device(x)
    return x, x.numel(), None


def _prepare_binary(a, b):
    """Prepare two tensors for a binary element-wise kernel.

    Handles scalar detection, broadcasting, contiguity.
    Returns: (a, b_or_scalar, output, n_elements, device_ctx, is_scalar)
    """
    if _is_scalar(b):
        a = a.contiguous()
        output = NBXTensor.empty_like(a)
        return a, _to_scalar(b), output, a.numel(), None, True

    if _is_scalar(a):
        b = b.contiguous()
        output = NBXTensor.empty_like(b)
        return _to_scalar(a), b, output, b.numel(), None, True

    if a.shape != b.shape:
        out_shape = _broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    output = NBXTensor.empty_like(a)
    return a, b, output, a.numel(), None, False


def _prepare_comparison(a, b):
    """Prepare two tensors for a comparison kernel (output is bool)."""
    if _is_scalar(b):
        a = a.contiguous()
        output = NBXTensor.empty(a.shape, NBXDtype.bool_)
        return a, _to_scalar(b), output, a.numel(), None, True

    if a.shape != b.shape:
        out_shape = _broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    output = NBXTensor.empty(a.shape, NBXDtype.bool_)
    return a, b, output, a.numel(), None, False


# ===========================================================================
# ACTIVATION WRAPPERS
# ===========================================================================

def relu(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    relu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def silu(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    silu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def gelu(x, approximate: str = 'none') :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    gelu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), approximate=(approximate == 'tanh'),
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sigmoid_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    sigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def tanh_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    tanh_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def hardsigmoid(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    hardsigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def hardswish(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    hardswish_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def leaky_relu(x, negative_slope: float = 0.01) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    leaky_relu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), negative_slope,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def elu(x, alpha: float = 1.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    elu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def mish(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    mish_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def selu_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    selu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# UNARY ELEMENT-WISE WRAPPERS
# ===========================================================================

def neg(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    neg_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def exp(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    exp_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sin(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    sin_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def cos(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    cos_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def rsqrt(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    rsqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sqrt_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    sqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def abs_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    abs_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def log_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    log_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def reciprocal(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    reciprocal_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def pow_wrapper(x, exponent) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    if isinstance(exponent, NBXTensor):
        exponent = exponent.item()
    pow_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), exponent, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def clamp(x, min_val=None, max_val=None) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _min = float(min_val) if min_val is not None else 0.0
    _max = float(max_val) if max_val is not None else 0.0
    clamp_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), _min, _max,
        has_min=(min_val is not None), has_max=(max_val is not None),
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def erf(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    erf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def copy_to(x, dtype: object) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x, dtype=dtype)
    copy_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# BINARY ELEMENT-WISE WRAPPERS — use universal launch layer
# ===========================================================================

def add(a, b, alpha: float = 1.0) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        add_scalar_kernel[_1d_grid(n)](a, output, n, float(b) * alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        add_forward_kernel[_1d_grid(n)](a, b, output, n, alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def mul(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        mul_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        mul_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def div(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        div_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        div_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sub(a, b, alpha: float = 1.0) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        add_scalar_kernel[_1d_grid(n)](a, output, n, -float(b) * alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        sub_forward_kernel[_1d_grid(n)](a, b, output, n, alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def rsub(a, b) :
    """Reverse subtraction: b - a."""
    a, _, _, n, dev_ctx, _ = _prepare_binary(a, a)  # just prepare a
    output = NBXTensor.empty_like(a)
    rsub_forward_kernel[_1d_grid(n)](a, output, n, float(_to_scalar(b)), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def where_wrapper(cond, x, y) :
    # Broadcast all three to same shape
    if cond.shape != x.shape or x.shape != y.shape:
        out_shape = _broadcast_shapes(cond.shape, x.shape, y.shape)
        cond = cond.expand(out_shape).contiguous()
        x = x.expand(out_shape).contiguous()
        y = y.expand(out_shape).contiguous()
    else:
        cond, x, y = cond.contiguous(), x.contiguous(), y.contiguous()
    output = NBXTensor.empty_like(x)
    where_forward_kernel[_1d_grid(x.numel())](cond, x, y, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def maximum_wrapper(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        # scalar maximum: clamp_min
        clamp_min_forward_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        maximum_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def minimum_wrapper(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if not scalar:
        minimum_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        # scalar minimum: clamp(max=b)
        clamp_forward_kernel[_1d_grid(n)](a, output, n, 0.0, float(b), has_min=False, has_max=True, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def masked_fill(x, mask, value: float) :
    # Broadcast mask to match x shape
    if x.shape != mask.shape:
        out_shape = _broadcast_shapes(x.shape, mask.shape)
        x = x.expand(out_shape).contiguous()
        mask = mask.expand(out_shape).contiguous()
    else:
        x, mask = x.contiguous(), mask.contiguous()
    value = _to_scalar(value) if isinstance(value, NBXTensor) else value
    output = NBXTensor.empty_like(x)
    masked_fill_forward_kernel[_1d_grid(x.numel())](
    x, mask, output, x.numel(), float(value),
    BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# COMPARISON WRAPPERS
# ===========================================================================

def gt(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        gt_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        gt_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def ge(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        ge_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        ge_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def lt(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        lt_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        lt_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def eq(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        eq_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        eq_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def ne(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        ne_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        ne_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


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

    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    layer_norm_forward_kernel[grid](
        x_2d, weight if has_weight else x_2d, bias if has_bias else x_2d,
        mean, inv_std, output_2d,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        eps,
        scale_by_weight=has_weight,
        add_bias=has_bias,
        save_stats=True,
        num_warps=4,
    )
    return output_2d.view_as(x), mean, inv_std


def rms_norm(x, weight, eps=1e-6, epsilon=None):
    """RMSNorm wrapper."""
    if epsilon is not None:
        eps = epsilon
    x = _ensure_cuda(x)
    weight = _ensure_cuda(weight)
    feat_dim = x.shape[-1]
    batch_dim = x.numel() // feat_dim

    x_2d = x.contiguous().view(batch_dim, feat_dim)
    output_2d = NBXTensor.empty_like(x_2d)

    has_weight = weight is not None

    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    rms_norm_forward_kernel[grid](
        x_2d, weight if has_weight else x_2d,
        output_2d,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        eps,
        scale_by_weight=has_weight,
        num_warps=4,
    )
    return output_2d.view_as(x)


# ===========================================================================
# SOFTMAX WRAPPER
# ===========================================================================

def softmax(x, dim: int = -1, half_to_float: bool = False) :
    """Softmax over given dimension."""
    dim = dim % x.ndim
    # Reshape to [batch, feat] where feat = softmax dim
    feat_dim = x.shape[dim]
    batch_dim = x.numel() // feat_dim

    # Move softmax dim to last, flatten
    x_perm = x.movedim(dim, -1).contiguous()
    x_2d = x_perm.view(batch_dim, feat_dim)
    output_2d = NBXTensor.empty_like(x_2d)

    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    softmax_forward_kernel[grid](
        x_2d, output_2d,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        log=False,
        num_warps=4,
    )
    return output_2d.view_as(x_perm).movedim(-1, dim)


def log_softmax(x, dim: int = -1) :
    """Log softmax over given dimension."""
    dim = dim % x.ndim
    feat_dim = x.shape[dim]
    batch_dim = x.numel() // feat_dim

    x_perm = x.movedim(dim, -1).contiguous()
    x_2d = x_perm.view(batch_dim, feat_dim)
    output_2d = NBXTensor.empty_like(x_2d)

    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    softmax_forward_kernel[grid](
        x_2d, output_2d,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        log=True,
        num_warps=4,
    )
    return output_2d.view_as(x_perm).movedim(-1, dim)


# ===========================================================================
# MATMUL WRAPPERS
# ===========================================================================

def mm(a, b) :
    """Matrix multiplication: C = A @ B.
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles fp16/bf16 overflow protection externally."""
    a, b = _ensure_cuda(a).contiguous(), _ensure_cuda(b).contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {K} vs {K2}"

    c = NBXTensor.empty((M, N), device=f"cuda:{a._device_idx}" if hasattr(a, '_device_idx') else 'cuda',
                        dtype=a.nbx_dtype if hasattr(a, 'nbx_dtype') else a.dtype)
    grid = (triton.cdiv(M, _MM_BM) * triton.cdiv(N, _MM_BN),)
    _set_device(a)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_MM_BM, BLOCK_N=_MM_BN, BLOCK_K=_MM_BK, GROUP_M=_MM_GROUP,
        num_warps=4, num_stages=2,
    )
    return c


def bmm(a, b) :
    """Batched matrix multiplication: C[i] = A[i] @ B[i].
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally."""
    a, b = _ensure_cuda(a).contiguous(), _ensure_cuda(b).contiguous()
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2

    c = NBXTensor.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, _MM_BM) * triton.cdiv(N, _MM_BN),)

    for i in range(B):
        matmul_kernel[grid](
            a[i], b[i], c[i],
            M, N, K,
            a[i].stride(0), a[i].stride(1),
            b[i].stride(0), b[i].stride(1),
            c[i].stride(0), c[i].stride(1),
            BLOCK_M=_MM_BM, BLOCK_N=_MM_BN, BLOCK_K=_MM_BK, GROUP_M=_MM_GROUP,
            num_warps=4, num_stages=2,
        )
    return c


def addmm(bias, a, b,
          beta: float = 1.0, alpha: float = 1.0) :
    """C = beta * bias + alpha * (A @ B).
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally."""
    a, b, bias = _ensure_cuda(a).contiguous(), _ensure_cuda(b).contiguous(), _ensure_cuda(bias).contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = NBXTensor.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, _MM_BM) * triton.cdiv(N, _MM_BN),)
    addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        alpha, beta,
        BLOCK_M=_MM_BM, BLOCK_N=_MM_BN, BLOCK_K=_MM_BK, GROUP_M=_MM_GROUP,
        num_warps=4, num_stages=2,
    )
    return c


# ===========================================================================
# EMBEDDING WRAPPER
# ===========================================================================

def embedding(weight, indices, padding_idx=-1, **kwargs) :
    """Embedding lookup: output[i] = weight[indices[i]]."""
    indices = indices.contiguous()
    weight = weight.contiguous()
    M = indices.numel()
    N = weight.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Allocate output on same device as weight (all args must be same device)
    dev = f"cuda:{weight._device_idx}" if hasattr(weight, '_device_idx') else 'cuda'
    # Move indices to weight's device if needed
    if hasattr(indices, '_device_idx') and hasattr(weight, '_device_idx') and indices._device_idx != weight._device_idx:
        # Cross-device: copy indices to weight's device
        new_indices = NBXTensor.empty(indices._shape, indices._dtype, dev)
        from .nbx_tensor import DeviceAllocator
        DeviceAllocator.memcpy(new_indices.data_ptr(), indices.data_ptr(), indices._nbytes)
        indices = new_indices
    output = NBXTensor.empty((*indices.shape, N), dtype=weight.nbx_dtype if hasattr(weight, 'nbx_dtype') else weight.dtype, device=dev)
    _set_device(weight)
    embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)
    return output


# ===========================================================================
# REDUCTION WRAPPERS
# ===========================================================================

def mean_wrapper(x, dim=None, keepdim=False) :
    """Mean reduction. Currently supports reducing over last dim."""
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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    mean_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        num_warps=4,
    )

    if dim is None:
        return output.squeeze(0).to(x.dtype)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.to(x.dtype).view(shape)


def sum_wrapper(x, dim=None, keepdim=False) :
    """Sum reduction."""
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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    sum_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        num_warps=4,
    )

    if dim is None:
        return output.squeeze(0).to(x.dtype)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.to(x.dtype).view(shape)


def amax_wrapper(x, dim=None, keepdim=False) :
    """Max reduction."""
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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    amax_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        num_warps=4,
    )

    if dim is None:
        return output.squeeze(0)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.view(shape)


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
    glu_forward_kernel[_1d_grid(x1.numel())](
        x1, x2, output, x1.numel(), act_func,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# PHASE 2 WRAPPERS — Extracted from FlagGems reference
# ===========================================================================

def le(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty(a.shape, dtype=NBXDtype.bool_, device=a.device)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        le_forward_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        le_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def softplus_wrapper(x, beta: float = 1.0, threshold: float = 20.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    softplus_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), beta, threshold,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def dropout_wrapper(x, p: float = 0.5, train: bool = False) :
    """Dropout at inference = identity. Training dropout not implemented in Triton yet."""
    if not train or p == 0.0:
        # Inference mode: passthrough
        x = x.contiguous()
        output = NBXTensor.empty_like(x)
        dropout_inference_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
        return output
    raise NotImplementedError("Training dropout with Triton RNG not implemented yet")


def remainder_wrapper(a, b) :
    """Remainder (modulo): a % b."""
    from .ops.remainder import remainder_forward_kernel, remainder_scalar_kernel
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        remainder_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        remainder_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def upsample_nearest1d_wrapper(
    x, output_size, scales=None
) :
    """Upsample nearest 1D via 2D: [N,C,L] → unsqueeze → [N,C,1,L] → upsample2d → squeeze."""
    x_2d = x.unsqueeze(2)  # [N, C, 1, L]
    if isinstance(output_size, (list, tuple)):
        out_size_2d = [1, output_size[0]]
    else:
        out_size_2d = [1, output_size]
    out_2d = upsample_nearest2d_wrapper(x_2d, out_size_2d, scales_h=None, scales_w=scales)
    return out_2d.squeeze(2)


def upsample_nearest2d_wrapper(
    x, output_size, scales_h=None, scales_w=None
) :
    """Upsample nearest 2D. Wrapper from FlagGems."""
    assert x.ndim == 4
    N, C, IH, IW = x.shape
    if isinstance(output_size, (list, tuple)):
        OH, OW = output_size[0], output_size[1]
    else:
        OH, OW = output_size, output_size

    reciprocal_scale_h = (1.0 / scales_h) if scales_h is not None else (IH / OH)
    reciprocal_scale_w = (1.0 / scales_w) if scales_w is not None else (IW / OW)

    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total_threads = OH * OW
    grid = (
        triton.cdiv(total_threads, _EW_BLOCK),
        triton.cdiv(N * C, 4),
    )
    upsample_nearest2d_kernel[grid](
        output, x, N, C, OH, OW, IH, IW,
        reciprocal_scale_h, reciprocal_scale_w,
        BLOCK_SIZE=_EW_BLOCK,
    )
    return output


def group_norm_wrapper(x, num_groups: int, weight, bias, eps=1e-5):
    """GroupNorm. Returns (output, mean, rstd) matching ATen native_group_norm."""
    x = x.contiguous()
    N = x.shape[0]
    C = x.shape[1]
    HxW = x.numel() // (N * C)
    group_size = C // num_groups

    y = NBXTensor.empty_like(x)
    mean = NBXTensor.empty((N, num_groups), dtype=NBXDtype.float32, device=x.device)
    rstd = NBXTensor.empty((N, num_groups), dtype=NBXDtype.float32, device=x.device)

    grid = (N * num_groups,)
    group_norm_forward_kernel[grid](
        x, y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        mean, rstd,
        group_size, C, HxW, num_groups, eps,
        scale_by_weight=(weight is not None),
        add_bias=(bias is not None),
        BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
        BLOCK_HW_SIZE=triton.next_power_of_2(HxW),
    )
    return y, mean, rstd


def index_select_wrapper(x, dim: int, index) :
    """Index select along dimension. Wrapper from FlagGems."""
    assert dim >= -x.ndim and dim < x.ndim
    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % x.ndim
    inp_shape = list(x.shape)
    index_len = index.numel()

    # dim_compress: move target dim to last
    if dim != x.ndim - 1:
        x = x.movedim(dim, -1).contiguous()
    else:
        x = x.contiguous()
    N = inp_shape[dim]
    M = x.numel() // N
    out_shape = list(x.shape)
    out_shape[-1] = index_len
    out = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)

    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(index_len))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(index_len, BLOCK_N))
    index_select_kernel[grid](x, out, M, N, index, index_len, BLOCK_M, BLOCK_N)

    if dim != x.ndim - 1:
        order = list(range(out.ndim - 1))
        order.insert(dim, out.ndim - 1)
        out = out.permute(order).contiguous()
    return out


def triu_wrapper(x, diagonal: int = 0) :
    """Upper triangular. Wrapper from FlagGems."""
    x = x.contiguous()
    out = NBXTensor.empty_like(x)
    assert x.ndim >= 2
    M, N = x.shape[-2:]

    if x.ndim == 2:
        grid = (triton.cdiv(M, _TRIU_MBS),)
        triu_kernel[grid](x, out, M, N, diagonal,
                          M_BLOCK_SIZE=_TRIU_MBS, N_BLOCK_SIZE=_TRIU_NBS,
                          num_warps=4)
    else:
        batch = x.numel() // (M * N)
        B = x.reshape(batch, M * N)
        grid = (
            triton.cdiv(batch, _TRIU_BATCH_BS),
            triton.cdiv(M * N, _TRIU_MN_BS),
        )
        triu_batch_kernel[grid](B, out.reshape(batch, M * N), batch, M * N, N, diagonal,
                                BATCH_BLOCK_SIZE=_TRIU_BATCH_BS, MN_BLOCK_SIZE=_TRIU_MN_BS,
                                num_warps=4)
    return out


def tril_wrapper(x, diagonal: int = 0) :
    """Lower triangular. Adapted from FlagGems triu."""
    x = x.contiguous()
    out = NBXTensor.empty_like(x)
    assert x.ndim >= 2
    M, N = x.shape[-2:]

    if x.ndim == 2:
        grid = (triton.cdiv(M, _TRIU_MBS),)
        tril_kernel[grid](x, out, M, N, diagonal,
                          M_BLOCK_SIZE=_TRIU_MBS, N_BLOCK_SIZE=_TRIU_NBS,
                          num_warps=4)
    else:
        batch = x.numel() // (M * N)
        B = x.reshape(batch, M * N)
        grid = (
            triton.cdiv(batch, _TRIU_BATCH_BS),
            triton.cdiv(M * N, _TRIU_MN_BS),
        )
        tril_batch_kernel[grid](B, out.reshape(batch, M * N), batch, M * N, N, diagonal,
                                BATCH_BLOCK_SIZE=_TRIU_BATCH_BS, MN_BLOCK_SIZE=_TRIU_MN_BS,
                                num_warps=4)
    return out


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

        argmax_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
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

        TILE_N = min(triton.next_power_of_2(N), 4096)
        grid = (M, 1, 1)
        argmax_kernel_inner[grid](x, out_index, M, N, TILE_N=TILE_N, ONE_TILE_PER_CTA=(TILE_N >= N))
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

        argmin_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
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

        BLOCK_N = min(4096, triton.next_power_of_2(N))
        BLOCK_M = 4
        grid = (triton.cdiv(M, BLOCK_M), K)
        argmin_kernel[grid](x, out_index, M, N, K, BLOCK_M, BLOCK_N)
        return out_index


# ===========================================================================
# CONV2D WRAPPER — Extracted from FlagGems
# ===========================================================================

def conv2d_wrapper(
    x, weight, bias=None,
    stride=1, padding=0, dilation=1,
    transposed=False, output_padding=0, groups=1,
) :
    """Convolution forward. Handles both 1D (3D tensors) and 2D (4D tensors).
    Routes to conv1d_wrapper for 3D inputs."""
    # Route 1D convolutions to conv1d_wrapper
    if weight.ndim == 3:
        return conv1d_wrapper(x, weight, bias, stride, padding, dilation, groups)

    assert weight.ndim == 4

    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride[0], stride[1]
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        pad_h, pad_w = padding[0], padding[1]
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (list, tuple)):
        dil_h, dil_w = dilation[0], dilation[1]
    else:
        dil_h = dil_w = dilation

    N, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = weight.shape
    out_h = (in_h + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    x_c = x.contiguous()
    w_c = weight.contiguous()
    output = NBXTensor.empty((N, out_c, out_h, out_w), device=x.device, dtype=x.dtype)

    fp16 = x.dtype == NBXDtype.float16

    grid = (
        triton.cdiv(N * out_h * out_w, _CONV_BHW),
        triton.cdiv(out_c // groups, _CONV_OUTF),
        groups,
    )
    conv2d_forward_kernel[grid](
    x_c, w_c, output,
    N, in_c, in_h, in_w,
    out_c, out_h, out_w,
    *x_c.stride(), *w_c.stride(), *output.stride(),
    kernel_height=kh, kernel_width=kw,
    stride_height=stride_h, stride_width=stride_w,
    padding_height=pad_h, padding_width=pad_w,
    groups=groups, fp16=fp16,
    BLOCK_SIZE_BHW=_CONV_BHW, BLOCK_SIZE_INF=_CONV_INF, BLOCK_SIZE_OUTF=_CONV_OUTF,
    num_warps=4, num_stages=2,
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


# ===========================================================================
# BATCH NORM WRAPPER — Extracted from FlagGems
# ===========================================================================

def batch_norm_wrapper(
    x, weight, bias,
    running_mean, running_var,
    training: bool = False, momentum: float = 0.1, eps: float = 1e-5,
) :
    """BatchNorm forward (inference mode). Wrapper from FlagGems."""
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
    batch_norm_forward_kernel[grid](
        x.contiguous(),
        weight.contiguous() if weight is not None else x,
        bias.contiguous() if bias is not None else x,
        mean_out, inv_std_out, output,
        running_mean if running_mean is not None else x,
        running_var if running_var is not None else x,
        N, spatial,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        momentum, eps,
        is_train=training,
    )
    return output


# ===========================================================================
# CUMSUM WRAPPER — Extracted from FlagGems (1D scan-then-fan)
# ===========================================================================

def cumsum_wrapper(x, dim: int = 0) :
    """Cumulative sum along dimension. Wrapper from FlagGems."""
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

        scan_part_sum_kernel[(part_num,)](
            row_in, row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

        if part_num > 1:
            for i in range(1, part_num):
                partial_sum[i] += partial_sum[i - 1]
            add_base_sum_kernel[(part_num,)](
                row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

    return output_2d.view_as(x_perm).movedim(-1, dim)


# ===========================================================================
# PHASE 3 — Simple element-wise wrappers
# ===========================================================================

def exp2_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    exp2_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def tan_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    tan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def celu_wrapper(x, alpha: float = 1.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    celu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def log_sigmoid_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    log_sigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isfinite_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    isfinite_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isinf_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    isinf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isnan_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    isnan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def nan_to_num_wrapper(x, nan: float = 0.0,
                       posinf: float = 3.4028235e+38,
                       neginf: float = -3.4028235e+38) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    nan_to_num_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), nan, posinf, neginf,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def threshold_wrapper(x, threshold: float, value: float) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    threshold_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), threshold, value,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
    addcdiv_forward_kernel[_1d_grid(input.numel())](
        input, tensor1, tensor2, output, input.numel(), value,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def addcmul_wrapper(input, tensor1,
                    tensor2, value: float = 1.0) :
    """result = input + value * tensor1 * tensor2"""
    input, tensor1, tensor2 = input.contiguous(), tensor1.contiguous(), tensor2.contiguous()
    output = NBXTensor.empty_like(input)
    addcmul_forward_kernel[_1d_grid(input.numel())](
        input, tensor1, tensor2, output, input.numel(), value,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


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
        all_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
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
        grid = (triton.cdiv(M, _RED_BM),)
        all_kernel_dim[grid](x_perm, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=BLOCK_N,
                             num_warps=4)
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
        any_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
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
        grid = (triton.cdiv(M, _RED_BM),)
        any_kernel_dim[grid](x_perm, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=BLOCK_N,
                             num_warps=4)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


# ---------------------------------------------------------------------------
# Bitwise ops (binary: AND/OR, unary: NOT)
# ---------------------------------------------------------------------------

def bitwise_and_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty_like(a)
    bitwise_and_forward_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_or_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty_like(a)
    bitwise_or_forward_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_not_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    bitwise_not_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ---------------------------------------------------------------------------
# Logical ops (binary: AND/OR → bool, unary: NOT → bool)
# ---------------------------------------------------------------------------

def logical_and_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty(a.shape, dtype=NBXDtype.bool_, device=a.device)
    logical_and_forward_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def logical_or_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty(a.shape, dtype=NBXDtype.bool_, device=a.device)
    logical_or_forward_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def logical_not_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    logical_not_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ---------------------------------------------------------------------------
# Conv1D
# ---------------------------------------------------------------------------

def conv1d_wrapper(
    x, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1,
) :
    """Conv1D via conv2d with unsqueeze. Reuses the V100-compatible attorch conv2d kernel.
    Input [N,C,L] → unsqueeze → [N,C,1,L] → conv2d → [N,Co,1,Lo] → squeeze → [N,Co,Lo]."""
    # Convert 1D → 2D by adding H=1 dimension
    x_2d = x.unsqueeze(2)  # [N, C_in, 1, L]
    w_2d = weight.unsqueeze(2)  # [C_out, C_in/groups, 1, K]

    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    out_2d = conv2d_wrapper(
        x_2d, w_2d, bias,
        stride=(1, stride), padding=(0, padding),
        dilation=(1, dilation), groups=groups)

    return out_2d.squeeze(2)  # [N, C_out, L_out]


# ---------------------------------------------------------------------------
# Lerp (linear interpolation)
# ---------------------------------------------------------------------------

def lerp_wrapper(input, end, weight) :
    """Linear interpolation: numerically stable two-branch formula."""
    input, end = input.contiguous(), end.contiguous()
    output = NBXTensor.empty_like(input)
    if isinstance(weight, NBXTensor):
        weight = weight.contiguous()
        lerp_tensor_forward_kernel[_1d_grid(input.numel())](
            input, end, weight, output, input.numel(),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    elif abs(weight) < 0.5:
        lerp_scalar_head_kernel[_1d_grid(input.numel())](
            input, end, output, input.numel(), float(weight),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        lerp_scalar_tail_kernel[_1d_grid(input.numel())](
            input, end, output, input.numel(), float(weight),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ---------------------------------------------------------------------------
# Dot product (2 vectors → scalar)
# ---------------------------------------------------------------------------

def dot_wrapper(x, y) :
    """Dot product: sum(x * y). Two-phase reduction for large N."""
    x, y = x.contiguous().view(-1), y.contiguous().view(-1)
    N = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)

    if N <= BLOCK_SIZE:
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        dot_kernel_small[(1,)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
        return out.to(x.dtype)
    else:
        mid_size = triton.cdiv(N, BLOCK_SIZE)
        mid = NBXTensor.empty(mid_size, dtype=NBXDtype.float32, device=x.device)
        dot_kernel_partial[(mid_size,)](x, y, mid, N, BLOCK_SIZE=BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        dot_kernel_reduce[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# MV (matrix-vector multiply)
# ---------------------------------------------------------------------------

def mv_wrapper(mat, vec) :
    """Matrix-vector multiply: out[i] = sum_j(mat[i, j] * vec[j])."""
    mat, vec = mat.contiguous(), vec.contiguous()
    N, M = mat.shape
    out = NBXTensor.empty(N, device=mat.device, dtype=mat.dtype)
    grid = (triton.cdiv(N, _MV_BN),)
    mv_kernel[grid](
        mat, vec, out,
        N, M,
        mat.stride(0), mat.stride(1),
        vec.stride(0),
        out.stride(0),
        BLOCK_M=_MV_BM, BLOCK_N=_MV_BN,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------

def flip_wrapper(x, dims) :
    """Reverse elements along the given dimension(s)."""
    x = x.contiguous()
    if isinstance(dims, int):
        dims = [dims]

    result = x
    for dim in dims:
        dim = dim % result.ndim
        if result.ndim == 1:
            output = NBXTensor.empty_like(result)
            flip_1d_kernel[_1d_grid(result.numel())](
                result, output, result.numel(),
                BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
            result = output
        else:
            output = NBXTensor.empty_like(result)
            n_elements = result.numel()
            dim_size = result.shape[dim]
            # stride_dim = product of dims after flip dim
            stride_dim = 1
            for i in range(dim + 1, result.ndim):
                stride_dim *= result.shape[i]
            grid = (triton.cdiv(n_elements, _EW_BLOCK),)
            flip_strided_kernel[grid](
                result, output, n_elements,
                dim_size, stride_dim, stride_dim,
                BLOCK_SIZE=_EW_BLOCK)
            result = output
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
        prod_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        prod_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(M, _RED_BM),)
        prod_kernel[grid](x_perm, out, M, N,
                          BLOCK_M=_RED_BM, BLOCK_N=_RED_BN,
                          num_warps=4)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


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
        min_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        min_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(M, _RED_BM),)
        min_kernel[grid](x_perm, out, M, N,
                         BLOCK_M=_RED_BM, BLOCK_N=_RED_BN,
                         num_warps=4)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


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
        max_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        max_kernel_result[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out
    else:
        dim = dim % x.ndim
        shape = list(x.shape)
        N = shape[dim]
        M = x.numel() // N
        x_perm = x.movedim(dim, -1).contiguous().reshape(M, N)
        out = NBXTensor.empty(M, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(M, _RED_BM),)
        max_kernel[grid](x_perm, out, M, N,
                         BLOCK_M=_RED_BM, BLOCK_N=_RED_BN,
                         num_warps=4)
        shape[dim] = 1
        result = out.view(shape)
        if not keepdim:
            result = result.squeeze(dim)
        return result


# ---------------------------------------------------------------------------
# Baddbmm (batched addmm)
# ---------------------------------------------------------------------------

def baddbmm_wrapper(
    input, batch1, batch2,
    beta: float = 1.0, alpha: float = 1.0,
) :
    """Batched addmm: out = beta * input + alpha * (batch1 @ batch2)."""
    batch1, batch2 = batch1.contiguous(), batch2.contiguous()
    input = input.contiguous()
    B, M, K = batch1.shape
    _, K2, N = batch2.shape
    assert K == K2

    output = NBXTensor.empty((B, M, N), device=batch1.device, dtype=batch1.dtype)

    # Handle bias broadcasting: input may be [M, N], [B, M, N], [1, M, N], etc.
    if input.ndim == 2:
        bias_batch_stride = 0
        bias_m_stride = input.stride(0)
        bias_n_stride = input.stride(1)
    else:
        bias_batch_stride = input.stride(0) if input.shape[0] > 1 else 0
        bias_m_stride = input.stride(-2)
        bias_n_stride = input.stride(-1)

    grid = (
        triton.cdiv(M, _MM_BM) * triton.cdiv(N, _MM_BN),
        1,
        B,
    )
    baddbmm_kernel[grid](
        batch1, batch2, output, input,
        alpha, beta,
        M, N, K,
        batch1.stride(0), batch1.stride(1), batch1.stride(2),
        batch2.stride(0), batch2.stride(1), batch2.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        bias_batch_stride, bias_m_stride, bias_n_stride,
        BLOCK_M=_MM_BM, BLOCK_N=_MM_BN, BLOCK_K=_MM_BK, GROUP_M=_MM_GROUP,
        num_warps=4, num_stages=2,
    )
    return output


# ---------------------------------------------------------------------------
# Addmv (matrix-vector with bias)
# ---------------------------------------------------------------------------

def addmv_wrapper(
    input, mat, vec,
    beta: float = 1.0, alpha: float = 1.0,
) :
    """Matrix-vector with bias: out = beta * input + alpha * (mat @ vec)."""
    mat, vec = mat.contiguous(), vec.contiguous()
    input = input.contiguous()
    N, M = mat.shape
    out = NBXTensor.empty(N, device=mat.device, dtype=mat.dtype)
    grid = (triton.cdiv(N, _MV_BN),)
    addmv_kernel[grid](
        mat, vec, input, out,
        N, M,
        alpha, beta,
        mat.stride(0), mat.stride(1),
        vec.stride(0),
        input.stride(0),
        out.stride(0),
        BLOCK_M=_MV_BM, BLOCK_N=_MV_BN,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# MSE Loss
# ---------------------------------------------------------------------------

def mse_loss_wrapper(input, target,
                     reduction: int = 1) :
    """MSE Loss. reduction: 0=none, 1=mean, 2=sum."""
    import math
    input, target = input.contiguous(), target.contiguous()
    if reduction == 0:
        output = NBXTensor.empty_like(input)
        mse_loss_none_kernel[_1d_grid(input.numel())](
            input, target, output, input.numel(),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
        return output

    M = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, BLOCK_SIZE)
    BLOCK_MID = triton.next_power_of_2(mid_size)
    mid = NBXTensor.empty(mid_size, dtype=NBXDtype.float32, device=input.device)
    out = NBXTensor.empty([], dtype=NBXDtype.float32, device=input.device)
    mse_loss_partial_kernel[(mid_size,)](
        input, target, mid, M, BLOCK_SIZE=BLOCK_SIZE, reduction=reduction)
    mse_loss_reduce_kernel[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
    return out.to(input.dtype)


# ---------------------------------------------------------------------------
# NLL Loss
# ---------------------------------------------------------------------------

def nll_loss_wrapper(input, target,
                     weight=None, ignore_index: int = -100,
                     reduction: int = 1) :
    """NLL Loss. reduction: 0=none, 1=mean, 2=sum."""
    input = input.contiguous()
    target = target.contiguous()
    N, C = input.shape

    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    num_blocks = triton.cdiv(N, BLOCK_N)

    if reduction == 0:
        out = NBXTensor.zeros(N, dtype=NBXDtype.float32, device=input.device)
    elif reduction == 1:
        # [sum_loss, sum_weight, counter, final_result]
        out = NBXTensor.zeros(4, dtype=NBXDtype.float32, device=input.device)
    else:
        out = NBXTensor.zeros(1, dtype=NBXDtype.float32, device=input.device)

    wgt_ptr = weight.contiguous() if weight is not None else None

    nll_loss_forward_kernel[(num_blocks,)](
        input, target, wgt_ptr, out,
        ignore_index, N, C,
        reduction=reduction, BLOCK_N=BLOCK_N,
    )

    if reduction == 0:
        return out.to(input.dtype)
    elif reduction == 1:
        return out[3].to(input.dtype)
    else:
        return out[0].to(input.dtype)


# ---------------------------------------------------------------------------
# Cross Entropy Loss
# ---------------------------------------------------------------------------

def cross_entropy_wrapper(input, target,
                          weight=None, reduction: int = 1) :
    """Cross entropy loss: -log(softmax(input)[target])."""
    input = input.contiguous()
    target = target.contiguous()
    batch_dim = input.shape[0]
    feat_dim = input.shape[-1]

    weighted = weight is not None
    BLOCK_SIZE_BATCH = min(triton.next_power_of_2(batch_dim), 64)
    num_blocks = triton.cdiv(batch_dim, BLOCK_SIZE_BATCH)

    output = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=input.device)
    sum_weights = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=input.device) if weighted else output

    cross_entropy_loss_forward_kernel[(num_blocks,)](
        input, target,
        weight.contiguous() if weighted else input,
        sum_weights, output,
        batch_dim, feat_dim,
        input.stride(0), input.stride(1),
        weighted=weighted,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
    )

    total_loss = output.sum()
    if weighted and reduction == 1:
        total_loss = total_loss / sum_weights.sum()
    return total_loss.to(input.dtype)


# ---------------------------------------------------------------------------
# Std / Var
# ---------------------------------------------------------------------------

def std_wrapper(x, dim=None, correction=1, keepdim=False) :
    """Standard deviation. Global (map-reduce) or dim-specific."""
    import math
    x = x.contiguous()
    if dim is None:
        N = x.numel()
        BLOCK_N = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), 4096)
        num_blocks = triton.cdiv(N, BLOCK_N)
        tmp_sum = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        tmp_sum_sq = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        std_map_kernel[(num_blocks,)](x.view(-1), tmp_sum, tmp_sum_sq, N, BLOCK_N=BLOCK_N)
        BLOCK_SIZE = triton.next_power_of_2(num_blocks)
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
        grid = (triton.cdiv(M, _RED_BM),)
        std_dim_kernel[grid](
            x_2d, out_flat,
            x_2d.stride(0), x_2d.stride(1),
            M, N, float(correction),
            BLOCK_M=_RED_BM, BLOCK_N=_RED_BN,
            num_warps=4,
        )
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return out_flat.to(x.dtype).view(shape)


def var_wrapper(x, dim=None, correction=1, keepdim=False) :
    """Variance. Global (Welford) or dim-specific."""
    import math
    x = x.contiguous()
    if dim is None:
        N = x.numel()
        BLOCK_N = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), 4096)
        num_blocks = triton.cdiv(N, BLOCK_N)
        acc = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        average = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        count = NBXTensor.empty(num_blocks, dtype=NBXDtype.float32, device=x.device)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        var_kernel_1[(num_blocks,)](x.view(-1), acc, average, count, N, BLOCK_N=BLOCK_N)
        BLOCK_NUM = triton.next_power_of_2(num_blocks)
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
        grid = (triton.cdiv(M, _RED_BM),)
        var_welford_kernel[grid](x_2d, out_flat, M, N, float(correction),
                                 BLOCK_M=_RED_BM, BLOCK_N=_RED_BN,
                                 num_warps=4)
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return out_flat.to(x.dtype).view(shape)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather_wrapper(input, dim: int, index) :
    """Gather along dimension."""
    input = input.contiguous()
    index = index.contiguous()
    dim = dim % input.ndim

    # Decompose into (outer, dim, inner) strides
    outer_size = 1
    for i in range(dim):
        outer_size *= input.shape[i]
    inner_size = 1
    for i in range(dim + 1, input.ndim):
        inner_size *= input.shape[i]
    dim_size = index.shape[dim]

    out = NBXTensor.empty_like(index, dtype=input.dtype)
    N = index.numel()

    # Pre-compute strides for the kernel
    inp_dim_stride = input.stride(dim)
    idx_stride_outer = index.stride(0) if dim > 0 else 0
    idx_stride_dim = index.stride(dim)
    idx_stride_inner = index.stride(-1) if dim < input.ndim - 1 else 0
    inp_stride_outer = input.stride(0) if dim > 0 else 0
    inp_stride_inner = input.stride(-1) if dim < input.ndim - 1 else 0
    out_stride_outer = out.stride(0) if dim > 0 else 0
    out_stride_dim = out.stride(dim)
    out_stride_inner = out.stride(-1) if dim < input.ndim - 1 else 0

    gather_kernel[_1d_grid(N)](
        input, index, out,
        N,
        inp_dim_stride,
        idx_stride_outer, idx_stride_dim, idx_stride_inner,
        inp_stride_outer, inp_stride_inner,
        out_stride_outer, out_stride_dim, out_stride_inner,
        outer_size, dim_size, inner_size,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


# ---------------------------------------------------------------------------
# Scatter / Scatter Add
# ---------------------------------------------------------------------------

def scatter_wrapper(input, dim: int, index,
                    src) :
    """Scatter: out[outer][index_val][inner] = src[outer][dim][inner]."""
    input = input.contiguous()
    src = src.contiguous()
    index = index.contiguous()
    dim = dim % input.ndim

    out = input.clone()
    N = index.numel()

    outer_size = 1
    for i in range(dim):
        outer_size *= index.shape[i]
    inner_size = 1
    for i in range(dim + 1, index.ndim):
        inner_size *= index.shape[i]
    dim_size = index.shape[dim]

    scatter_kernel[_1d_grid(N)](
        src, index, out,
        N,
        out.stride(dim),
        src.stride(0) if dim > 0 else 0, src.stride(dim), src.stride(-1) if dim < src.ndim - 1 else 0,
        index.stride(0) if dim > 0 else 0, index.stride(dim), index.stride(-1) if dim < index.ndim - 1 else 0,
        out.stride(0) if dim > 0 else 0, out.stride(-1) if dim < out.ndim - 1 else 0,
        outer_size, dim_size, inner_size,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


def scatter_add_wrapper(input, dim: int, index,
                        src) :
    """Scatter add: out[outer][index_val][inner] += src[outer][dim][inner]."""
    input = input.contiguous()
    src = src.contiguous()
    index = index.contiguous()
    dim = dim % input.ndim

    out = input.clone()
    N = index.numel()

    outer_size = 1
    for i in range(dim):
        outer_size *= index.shape[i]
    inner_size = 1
    for i in range(dim + 1, index.ndim):
        inner_size *= index.shape[i]
    dim_size = index.shape[dim]

    scatter_add_kernel[_1d_grid(N)](
        src, index, out,
        N,
        out.stride(dim),
        src.stride(0) if dim > 0 else 0, src.stride(dim), src.stride(-1) if dim < src.ndim - 1 else 0,
        index.stride(0) if dim > 0 else 0, index.stride(dim), index.stride(-1) if dim < index.ndim - 1 else 0,
        out.stride(0) if dim > 0 else 0, out.stride(-1) if dim < out.ndim - 1 else 0,
        outer_size, dim_size, inner_size,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


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

    if reduce == "sum" or reduce == "add":
        scatter_add_kernel[_1d_grid(N)](*stride_args, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    elif reduce == "amax":
        scatter_reduce_amax_kernel[_1d_grid(N)](*stride_args, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    elif reduce == "amin":
        scatter_reduce_amin_kernel[_1d_grid(N)](*stride_args, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        raise RuntimeError(f"[--triton] scatter_reduce mode '{reduce}' not implemented")

    return out


# ---------------------------------------------------------------------------
# Index Add
# ---------------------------------------------------------------------------

def index_add_wrapper(x, dim: int, index,
                      source, alpha: float = 1.0) :
    """Index add: out[..., index[i], ...] += alpha * source[..., i, ...]."""
    x = x.contiguous()
    source = source.contiguous()
    index = index.contiguous()
    dim = dim % x.ndim

    out = x.clone()
    N = source.numel()

    outer_size = 1
    for i in range(dim):
        outer_size *= source.shape[i]
    inner_size = 1
    for i in range(dim + 1, source.ndim):
        inner_size *= source.shape[i]
    dim_size = source.shape[dim]

    inp_dim_stride = x.stride(dim)
    inp_shape_dim = x.shape[dim]
    src_shape_dim = source.shape[dim]
    delta = inp_shape_dim - src_shape_dim

    src_stride_outer = source.stride(0) if dim > 0 else 0
    src_stride_dim = source.stride(dim)
    src_stride_inner = source.stride(-1) if dim < source.ndim - 1 else 0

    index_add_kernel[_1d_grid(N)](
        index, source, out,
        N, x.numel(),
        inp_dim_stride, inp_shape_dim, src_shape_dim, delta, alpha,
        src_stride_outer, src_stride_dim, src_stride_inner,
        outer_size, dim_size, inner_size,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


# ---------------------------------------------------------------------------
# Avg Pool 2D
# ---------------------------------------------------------------------------

def avg_pool2d_wrapper(
    x, kernel_size, stride=None, padding=0,
    ceil_mode=False, count_include_pad=True, divisor_override=None,
) :
    """Average pooling 2D forward."""
    assert x.ndim == 4
    N, C, H, W = x.shape

    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size[0], kernel_size[1]

    if stride is None:
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride[0], stride[1]

    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding[0], padding[1]

    if ceil_mode:
        oh = (H + 2 * ph - kh + sh - 1) // sh + 1
        ow = (W + 2 * pw - kw + sw - 1) // sw + 1
    else:
        oh = (H + 2 * ph - kh) // sh + 1
        ow = (W + 2 * pw - kw) // sw + 1

    output = NBXTensor.empty((N, C, oh, ow), device=x.device, dtype=x.dtype)

    grid = (
        N * C,
        triton.cdiv(oh, _POOL_BH) * triton.cdiv(ow, _POOL_BW),
    )
    avg_pool2d_forward_kernel[grid](
        x.contiguous(), output,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        C, H, W,
        oh, ow,
        kh, kw, sh, sw, ph, pw,
        1, 1,  # dilation_h, dilation_w
        count_include_pad,
        0 if divisor_override is None else divisor_override,
        BLOCK_H=_POOL_BH, BLOCK_W=_POOL_BW,
        num_warps=4,
    )
    return output


# ---------------------------------------------------------------------------
# Max Pool 2D
# ---------------------------------------------------------------------------

def max_pool2d_wrapper(
    x, kernel_size, stride=None, padding=0,
    dilation=1, ceil_mode=False, return_indices=False,
) :
    """Max pooling 2D forward."""
    assert x.ndim == 4
    N, C, H, W = x.shape

    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size[0], kernel_size[1]

    if stride is None:
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride[0], stride[1]

    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding[0], padding[1]

    if isinstance(dilation, int):
        dh, dw = dilation, dilation
    else:
        dh, dw = dilation[0], dilation[1]

    if ceil_mode:
        oh = (H + 2 * ph - dh * (kh - 1) - 1 + sh - 1) // sh + 1
        ow = (W + 2 * pw - dw * (kw - 1) - 1 + sw - 1) // sw + 1
    else:
        oh = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    output = NBXTensor.empty((N, C, oh, ow), device=x.device, dtype=x.dtype)
    indices = NBXTensor.empty((N, C, oh, ow), device=x.device, dtype=NBXDtype.int64)

    grid = (
        N * C,
        triton.cdiv(oh, _POOL_BH) * triton.cdiv(ow, _POOL_BW),
    )
    max_pool2d_forward_kernel[grid](
        x.contiguous(), output, indices,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        C, H, W,
        oh, ow,
        kh, kw, sh, sw, ph, pw, dh, dw,
        BLOCK_H=_POOL_BH, BLOCK_W=_POOL_BW,
        num_warps=4,
    )
    if return_indices:
        return output, indices
    return output


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
    from .ops.sort_op import (
        convert_to_uint_preserve_order,
        radix_sort_histogram_kernel,
        radix_sort_sweep_kernel,
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

    # Stage 1: compute global histogram
    radix_sort_histogram_kernel[(m * grid_n,)](
        x.contiguous(), global_hist,
        n, m, k_bits, n_passes, num_bins,
        CTA_TILE, TILE_N)

    # Exclusive prefix sum on histogram (use our cumsum kernel)
    # cumsum along last dim, then subtract self → exclusive prefix sum
    hist_cumsum = NBXTensor.empty_like(global_hist)
    for b in range(m):
        for p in range(n_passes):
            row = global_hist[b, p]
            cs = cumsum_wrapper(row, dim=0)
            hist_cumsum[b, p] = cs - row
    ex_cumsum = hist_cumsum.to(NBXDtype.int32)

    # Double buffers
    arr_in = x.contiguous().clone()
    indices_in = NBXTensor.zeros(0, n, dtype=NBXDtype.int64, device=x.device)
    if m > 1:
        indices_in = indices_in.unsqueeze(0).expand(m, -1).contiguous()
    arr_out = NBXTensor.empty_like(arr_in)
    indices_out = NBXTensor.empty_like(indices_in)

    TILE_R = 8
    grid_r = triton.cdiv(num_bins, TILE_R)
    SWEEP_TILE = 2048
    sweep_grid_n = triton.cdiv(n, SWEEP_TILE)
    status = NBXTensor.empty((m, num_bins, sweep_grid_n), device=x.device, dtype=NBXDtype.int32)

    # Stage 2: sweep per radix pass
    for i in range(n_passes):
        status.zero_()
        radix_sort_sweep_kernel[(m * sweep_grid_n, grid_r)](
            arr_in, indices_in, arr_out, indices_out,
            ex_cumsum, status,
            n, m, i * k_bits, k_bits, num_bins,
            SWEEP_TILE, TILE_R, descending)
        arr_in, arr_out = arr_out, arr_in
        indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


# ===========================================================================
# PHASE 5 — RoPE, spatial, RNG, remaining wrappers
# ===========================================================================

def rope_wrapper(q, k,
                 cos, sin,
                 position_ids=None) -> tuple:
    """RoPE — Rotary Position Embedding (in-place on Q and K)."""
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    B, H, T, D = q.shape
    n_elements = B * H * T * (D // 2)
    grid = (triton.cdiv(n_elements, _EW_BLOCK),)
    rope_forward_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos.stride(0), cos.stride(1),
        B, H, T, D, D // 2, n_elements,
        BACKWARD_PASS=False,
        BLOCK_SIZE=_EW_BLOCK,
    )
    return q, k


def pixel_shuffle_wrapper(x, upscale_factor: int) :
    """Pixel shuffle: [N,C*r*r,H,W] -> [N,C,H*r,W*r]."""
    N, C_in, H, W = x.shape
    r = upscale_factor
    C_out = C_in // (r * r)
    OH, OW = H * r, W * r
    x = x.contiguous()
    output = NBXTensor.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C_out * OH * OW
    grid = (triton.cdiv(total, _EW_BLOCK),)
    pixel_shuffle_kernel[grid](
        x, output, total,
        C_out, H, W, r, OH, OW,
        *x.stride(), *output.stride(),
        BLOCK_SIZE=_EW_BLOCK)
    return output


def pixel_unshuffle_wrapper(x, downscale_factor: int) :
    """Pixel unshuffle: [N,C,H*r,W*r] -> [N,C*r*r,H,W]."""
    N, C, IH, IW = x.shape
    r = downscale_factor
    C_out = C * r * r
    OH, OW = IH // r, IW // r
    x = x.contiguous()
    output = NBXTensor.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C_out * OH * OW
    grid = (triton.cdiv(total, _EW_BLOCK),)
    pixel_unshuffle_kernel[grid](
        x, output, total,
        C, IH, IW, r, C_out, OH, OW,
        *x.stride(), *output.stride(),
        BLOCK_SIZE=_EW_BLOCK)
    return output


def upsample_bilinear2d_wrapper(x, output_size, align_corners=False,
                                scales_h=None, scales_w=None) :
    """Bilinear upsampling."""
    N, C, IH, IW = x.shape
    if isinstance(output_size, (list, tuple)):
        OH, OW = output_size
    else:
        OH, OW = output_size, output_size
    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    if align_corners and OH > 1:
        scale_h = (IH - 1) / (OH - 1)
        scale_w = (IW - 1) / (OW - 1)
        offset = 0.0
    else:
        scale_h = IH / OH
        scale_w = IW / OW
        offset = 0.5
    grid = (
        triton.cdiv(OH, _BILINEAR_BX),
        triton.cdiv(OW, _BILINEAR_BY),
        N * C,
    )
    upsample_bilinear2d_kernel[grid](
        x.contiguous(), output,
        N, C, IH, IW, OH, OW,
        scale_h, scale_w, offset,
        BLOCK_X=_BILINEAR_BX, BLOCK_Y=_BILINEAR_BY,
        num_warps=4)
    return output


def adaptive_avg_pool2d_wrapper(x, output_size) :
    """Adaptive average pooling 2D."""
    N, C, IH, IW = x.shape
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH, OW = output_size
    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = (triton.cdiv(total, _EW_BLOCK),)
    x_c = x.contiguous()
    adaptive_avg_pool2d_kernel[grid](
        x_c, output, total,
        IH, IW, OH, OW,
        *x_c.stride(),
        BLOCK_SIZE=_EW_BLOCK)
    return output


def conv_depthwise2d_wrapper(x, weight, bias=None,
                             stride=1, padding=0, dilation=1) :
    """Depthwise conv2d."""
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    N, C, IH, IW = x.shape
    _, _, KH, KW = weight.shape
    OH = (IH + 2*padding[0] - dilation[0]*(KH-1) - 1) // stride[0] + 1
    OW = (IW + 2*padding[1] - dilation[1]*(KW-1) - 1) // stride[1] + 1
    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = (
        triton.cdiv(OW, _BILINEAR_BX),
        triton.cdiv(OH, _BILINEAR_BY),
        N * C,
    )
    conv_depthwise2d_kernel[grid](
        x.contiguous(), weight.contiguous(), output,
        N, C, IH, IW, C, KH, KW, OH, OW,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        1,  # ch_mult
        BLOCK_X=_BILINEAR_BX, BLOCK_Y=_BILINEAR_BY,
        num_warps=4)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output


def logical_xor_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty(a.shape, dtype=NBXDtype.bool_, device=a.device)
    logical_xor_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_left_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        bitwise_left_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        from .ops.bitwise_left_shift import bitwise_left_shift_scalar_kernel
        bitwise_left_shift_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), int(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_right_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        bitwise_right_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        from .ops.bitwise_right_shift import bitwise_right_shift_scalar_kernel
        bitwise_right_shift_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), int(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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

    grid = (triton.cdiv(M, _RED_BM),)

    if ord == 2 or ord == 2.0:
        l2_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == float('inf'):
        linf_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == 0 or ord == 0.0:
        l0_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == 1 or ord == 1.0:
        l1_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    else:
        lp_norm_kernel[grid](x_2d, out, M, N, ord, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)

    result = out.to(x.dtype).view(out_shape)
    if not keepdim:
        for d in sorted(dim_list, reverse=True):
            result = result.squeeze(d)
    return result


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
    clamp_min_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), float(min_val), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def topk_wrapper(x, k: int, dim: int = -1,
                 largest: bool = True, sorted: bool = True):
    """Top-k via 2-stage Triton pipeline. Stage 1: per-chunk top-k. Stage 2: merge via bitonic sort.
    Triton kernels + NBXTensor allocation."""
    from .ops.topk import topk_stage1_kernel, topk_stage2_kernel
    import math as _math

    if dim < 0:
        dim = dim + x.ndim
    assert dim == x.ndim - 1, "topk currently supports last dim only"

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

    chunk_num = triton.cdiv(topk_n, chunk_size)

    # Stage 1 buffers
    s1_vals = NBXTensor.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    s1_idxs = NBXTensor.empty(batch_size * chunk_num * k, device=x.device, dtype=NBXDtype.int64)

    # Stage 1: per-chunk top-k
    topk_stage1_kernel[batch_size, chunk_num](
        s1_vals, s1_idxs, x.contiguous(),
        k, topk_n, chunk_size, descending)

    # Stage 2 buffers
    out_shape = x.shape[:-1] + (k,)
    s2_vals = NBXTensor.empty(out_shape, device=x.device, dtype=x.dtype)
    s2_idxs = NBXTensor.empty(out_shape, device=x.device, dtype=NBXDtype.int64)

    # Stage 2: merge chunks via bitonic sort
    stage2_n = chunk_num * k
    BLOCK = triton.next_power_of_2(stage2_n)
    topk_stage2_kernel[batch_size,](
        s2_vals, s2_idxs, s1_vals, s1_idxs,
        dim, k, stage2_n, BLOCK, descending)

    return s2_vals, s2_idxs


def trace_wrapper(x) :
    """Sum of diagonal elements."""
    assert x.ndim >= 2
    n = min(x.shape[-2], x.shape[-1])
    out = NBXTensor.empty(1, dtype=NBXDtype.float32, device=x.device)
    trace_kernel[(1,)](
        x.contiguous(), out, n,
        x.stride(-2), x.stride(-1),
        BLOCK_SIZE=triton.next_power_of_2(n))
    return out.squeeze().to(x.dtype)


# ===========================================================================
# PHASE 6: VIDEO + AUDIO WRAPPERS
# ===========================================================================

def upsample_nearest3d_wrapper(
    x, output_size, scales_d=None, scales_h=None, scales_w=None
) :
    """Upsample nearest 3D: [N,C,D,H,W] -> [N,C,OD,OH,OW]."""
    assert x.ndim == 5, f"Expected 5D input, got {x.ndim}D"
    x = x.contiguous()
    N, C, ID, IH, IW = x.shape
    if isinstance(output_size, (list, tuple)):
        OD, OH, OW = output_size[0], output_size[1], output_size[2]
    else:
        OD = OH = OW = output_size

    reciprocal_scale_d = (1.0 / scales_d) if scales_d is not None else (ID / OD)
    reciprocal_scale_h = (1.0 / scales_h) if scales_h is not None else (IH / OH)
    reciprocal_scale_w = (1.0 / scales_w) if scales_w is not None else (IW / OW)

    output = NBXTensor.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    total_threads = OD * OH * OW
    grid = (
        triton.cdiv(total_threads, _EW_BLOCK),
        triton.cdiv(N * C, 4),
    )
    upsample_nearest3d_kernel[grid](
        output, x, N, C, OD, OH, OW, ID, IH, IW,
        reciprocal_scale_d, reciprocal_scale_h, reciprocal_scale_w,
        BLOCK_SIZE=_EW_BLOCK,
    )
    return output


import math


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
        grid = (triton.cdiv(M, _WNORM_BM),)
        weight_norm_kernel_first[grid](output, norm, v, g, M, N, eps,
                                       BLOCK_M=_WNORM_BM, BLOCK_N=_WNORM_BN,
                                       num_warps=4)
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        grid = (triton.cdiv(N, _WNORM_BN),)
        weight_norm_kernel_last[grid](output, norm, v, g, M, N, eps,
                                      BLOCK_M=_WNORM_BM, BLOCK_N=_WNORM_BN,
                                      num_warps=4)
    else:
        raise ValueError(f"weight_norm only supports dim=0 or dim={v.ndim - 1}, got {dim}")

    return output, norm


def repeat_interleave_self_int_wrapper(
    inp, repeats: int, dim: int = 0, output_size: int = None
) :
    """repeat_interleave with scalar repeats: stride trick + copy.

    [1,2,3] with repeats=2 -> [1,1,2,2,3,3]
    """
    if dim < 0:
        dim = dim + inp.ndim
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)
    output_shape[dim] *= repeats

    if repeats == 0:
        return NBXTensor.empty(output_shape, dtype=inp.dtype, device=inp.device)

    # Use expand + clone to implement the stride trick
    view_shape = inp_shape[:dim + 1] + [repeats] + inp_shape[dim + 1:]
    view_stride = inp_stride[:dim + 1] + [0] + inp_stride[dim + 1:]
    expanded = lambda t, s, st: t.as_strided(s, st)(inp, view_shape, view_stride)
    return expanded.reshape(output_shape).contiguous()


def repeat_interleave_tensor_wrapper(
    repeats, output_size: int = None
) :
    """repeat_interleave with tensor repeats: returns index tensor.

    repeats=[2,3,1] -> [0,0,1,1,1,2]
    """
    assert repeats.ndim == 1, "repeats must be a 1D tensor"
    cumsum = repeats.cumsum(dim=0)
    result_size = cumsum[-1].item()

    out = NBXTensor.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[(size,)](
        repeats, cumsum, out, size, BLOCK_SIZE=BLOCK_SIZE, num_warps=1,
    )
    return out


def repeat_interleave_self_tensor_wrapper(
    inp, repeats, dim: int = 0, output_size: int = None
) :
    """repeat_interleave with tensor repeats applied to input.

    If repeats is a scalar tensor, delegates to self_int version.
    Otherwise builds index tensor and uses index_select.
    """
    if dim is None:
        inp = inp.flatten()
        dim = 0
    if dim < 0:
        dim = dim + inp.ndim

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int_wrapper(
            inp, repeats.item(), dim=dim, output_size=output_size
        )

    indices = repeat_interleave_tensor_wrapper(repeats)
    return inp


# ===========================================================================
# RNG OPS — Random number generation via Triton kernels
# ===========================================================================

def rand_wrapper(*args, **kwargs) :
    """rand(size, ...) → uniform [0, 1) via Triton kernel."""
    from .ops.rand_op import rand_kernel
    # ATen signature: rand(SymInt[] size, ...)
    size = args[0] if args else kwargs.get('size', [])
    if isinstance(size, NBXTensor):
        size = size.tolist()
    dtype = kwargs.get('dtype', NBXDtype.float32)
    device = kwargs.get('device', None)
    if device is None:
        device = str('cuda')
    output = NBXTensor.empty(size, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    rand_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def randn_wrapper(*args, **kwargs) :
    """randn(size, ...) → standard normal N(0,1) via Triton kernel."""
    from .ops.rand_op import randn_kernel
    size = args[0] if args else kwargs.get('size', [])
    if isinstance(size, NBXTensor):
        size = size.tolist()
    dtype = kwargs.get('dtype', NBXDtype.float32)
    device = kwargs.get('device', None)
    if device is None:
        device = str('cuda')
    output = NBXTensor.empty(size, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    randn_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def rand_like_wrapper(x, **kwargs) :
    """rand_like(tensor) → uniform [0, 1) same shape/dtype/device."""
    from .ops.rand_op import rand_kernel
    dtype = kwargs.get('dtype', x.dtype)
    device = kwargs.get('device', x.device)
    output = NBXTensor.empty(x.shape, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    rand_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def randn_like_wrapper(x, **kwargs) :
    """randn_like(tensor) → standard normal N(0,1) same shape/dtype/device."""
    from .ops.rand_op import randn_kernel
    dtype = kwargs.get('dtype', x.dtype)
    device = kwargs.get('device', x.device)
    output = NBXTensor.empty(x.shape, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    randn_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def normal_wrapper(*args, **kwargs) :
    """normal(mean, std, size) → N(mean, std) via Triton randn + transform."""
    from .ops.rand_op import randn_kernel
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
    output = NBXTensor.empty(size, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    randn_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    # Transform N(0,1) → N(mean, std)
    if std != 1.0:
        output.mul_(std)
    if mean != 0.0:
        output.add_(mean)
    return output


def uniform_wrapper(x, low: float = 0.0, high: float = 1.0,
                    **kwargs) :
    """uniform_(tensor, low, high) → fill with uniform [low, high)."""
    from .ops.rand_op import rand_kernel
    x = _ensure_cuda(x).contiguous()
    n = x.numel()
    if n == 0:
        return x
    seed = __import__("random").randint(0, 2**31 - 1)
    rand_kernel[_1d_grid(n)](x, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    # Transform [0,1) → [low, high)
    if high - low != 1.0:
        x.mul_(high - low)
    if low != 0.0:
        x.add_(low)
    return x


def bernoulli_wrapper(x, p: float = 0.5, **kwargs) :
    """bernoulli(tensor) → 0/1 values based on probabilities."""
    from .ops.rand_op import rand_kernel
    x = _ensure_cuda(x)
    output = NBXTensor.empty_like(x)
    # Generate uniform random values
    temp = NBXTensor.empty_like(x, dtype=NBXDtype.float32)
    n = temp.numel()
    if n == 0:
        return output
    seed = __import__("random").randint(0, 2**31 - 1)
    rand_kernel[_1d_grid(n)](temp, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    # Compare with probability: if rand < p → 1.0, else → 0.0
    # If x is the probability tensor, compare temp < x
    if x.is_floating_point():
        result = (temp < x.float()).to(output.dtype)
    else:
        result = (temp < p).to(output.dtype)
    output.copy_(result)
    return output


def multinomial_wrapper(x, num_samples: int,
                        replacement: bool = False, **kwargs) :
    """multinomial(probs, num_samples) → sampled indices.

    Uses Gumbel-max trick: argmax(log(probs) + Gumbel) = categorical sample.
    Gumbel noise = -log(-log(uniform)).
    """
    from .ops.rand_op import rand_kernel
    x = _ensure_cuda(x)
    # Normalize probabilities
    probs = x.float()
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    batch_size, n_categories = probs.shape

    if num_samples == 1:
        # Single sample: Gumbel-max trick
        # gumbel = -log(-log(uniform(0,1)))
        gumbel = NBXTensor.empty_like(probs)
        n = gumbel.numel()
        seed = __import__("random").randint(0, 2**31 - 1)
        rand_kernel[_1d_grid(n)](gumbel, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
        gumbel = clamp(gumbel, min_val=1e-10)
        gumbel = neg(log_wrapper(neg(log_wrapper(gumbel))))
        log_probs = clamp(log_wrapper(probs), min_val=-1e30)
        scores = add(log_probs, gumbel)
        indices = argmax_wrapper(scores, dim=-1, keepdim=True)
    else:
        # Multiple samples: repeated Gumbel-max
        all_indices = []
        for _ in range(num_samples):
            gumbel = NBXTensor.empty_like(probs)
            n = gumbel.numel()
            seed = __import__("random").randint(0, 2**31 - 1)
            rand_kernel[_1d_grid(n)](gumbel, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
            gumbel = clamp(gumbel, min_val=1e-10)
            gumbel = neg(log_wrapper(neg(log_wrapper(gumbel))))
            log_probs = clamp(log_wrapper(probs), min_val=-1e30)
            scores = add(log_probs, gumbel)
            all_indices.append(argmax_wrapper(scores, dim=-1, keepdim=True))
        indices = NBXTensor.cat(all_indices, dim=-1)

    if squeeze_out:
        indices = indices.squeeze(0)
    return indices.to(NBXDtype.int64)


# ===========================================================================
# PADDING OPS — via Triton kernels
# ===========================================================================

def constant_pad_nd_wrapper(x, pad_list, value: float = 0.0) :
    """constant_pad_nd via Triton kernel."""
    from .ops.pad_op import constant_pad_2d_kernel, constant_pad_1d_kernel
    x = _ensure_cuda(x).contiguous()

    ndim = x.ndim
    pad_len = len(pad_list)

    # Parse pad list: [left, right, top, bottom, ...]
    # Pads from last dim backwards (ATen convention)
    pad_before = [0] * ndim
    pad_after = [0] * ndim
    for i in range(pad_len // 2):
        dim_idx = ndim - 1 - i
        pad_before[dim_idx] = int(pad_list[2 * i])
        pad_after[dim_idx] = int(pad_list[2 * i + 1])

    out_shape = list(x.shape)
    for i in range(ndim):
        out_shape[i] += pad_before[i] + pad_after[i]

    output = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)
    total = output.numel()

    if total == 0:
        return output

    if ndim >= 4 and pad_len == 4:
        # 4D: use optimized 2D kernel (N,C,H,W)
        batch_channels = 1
        for d in range(ndim - 2):
            batch_channels *= x.shape[d]
        in_h, in_w = x.shape[-2], x.shape[-1]
        out_h, out_w = out_shape[-2], out_shape[-1]
        constant_pad_2d_kernel[_1d_grid(total)](
            x, output, total,
            in_h, in_w, out_h, out_w,
            batch_channels,
            x.stride(-4) if ndim >= 4 else 0,
            x.stride(-3) if ndim >= 3 else 0,
            x.stride(-2), x.stride(-1),
            output.stride(-4) if ndim >= 4 else 0,
            output.stride(-3) if ndim >= 3 else 0,
            output.stride(-2), output.stride(-1),
            pad_before[-1], pad_before[-2],
            float(value),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
        )
    elif pad_len == 2:
        batch_size = 1
        for d in range(ndim - 1):
            batch_size *= x.shape[d]
        in_size = x.shape[-1]
        out_size = out_shape[-1]
        constant_pad_1d_kernel[_1d_grid(total)](
            x, output, total,
            in_size, out_size, batch_size,
            pad_before[-1],
            float(value),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
        )
    else:
        batch_size = 1
        for d in range(ndim - 1):
            batch_size *= x.shape[d]
        in_size = x.shape[-1]
        out_size = out_shape[-1]
        constant_pad_1d_kernel[_1d_grid(total)](
            x.reshape(batch_size, in_size),
            output.reshape(batch_size, out_size),
            total, in_size, out_size, batch_size,
            pad_before[-1], float(value),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
        )

    return output


def pad_wrapper(x, pad_list, mode: str = "constant",
                value: float = 0.0) :
    """F.pad via Triton. constant mode uses kernel, others crash."""
    if mode == "constant":
        return constant_pad_nd_wrapper(x, pad_list, value)
    raise RuntimeError(
        f"[--triton mode] pad mode='{mode}' not implemented. "
        f"Only 'constant' mode has a Triton kernel. "
        f"Implement reflection/replication pad kernels if needed."
    )


def reflection_pad1d_wrapper(x, pad_list) :
    """Reflection pad 1D — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] reflection_pad1d has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/pad_op.py."
    )


def reflection_pad2d_wrapper(x, pad_list) :
    """Reflection pad 2D — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] reflection_pad2d has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/pad_op.py."
    )


def replication_pad1d_wrapper(x, pad_list) :
    """Replication pad 1D — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] replication_pad1d has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/pad_op.py."
    )


def replication_pad2d_wrapper(x, pad_list) :
    """Replication pad 2D — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] replication_pad2d has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/pad_op.py."
    )


# ===========================================================================
# FOLD / UNFOLD_BACKWARD — compute ops
# ===========================================================================

def fold_wrapper(*args, **kwargs) :
    """fold — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] fold (col2im) has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/fold_op.py."
    )


def unfold_backward_wrapper(*args, **kwargs) :
    """unfold_backward — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] unfold_backward has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/unfold_op.py."
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

def scaled_dot_product_attention_wrapper(q, k, v, attn_mask=None,
                                          dropout_p=0.0, is_causal=False,
                                          scale=None, **kwargs):
    """aten::scaled_dot_product_attention via Triton Flash Attention.

    Args:
        q: (batch, nheads, seqlen_q, headdim)
        k: (batch, nheads_k, seqlen_k, headdim)
        v: (batch, nheads_k, seqlen_k, headdim)
        attn_mask: optional attention mask (bias)
        dropout_p: dropout probability (ignored at inference)
        is_causal: causal masking
        scale: softmax scale (default: 1/sqrt(headdim))

    Reference: Dao-AILab _flash_attn_forward signature and grid computation.
    """
    import math
    from .ops.flash_attention import flash_attention_forward_kernel

    _set_device(q)

    # Fix pre-transposed K from graph math decomposition path.
    #
    # PyTorch's SDPA math path does bmm(Q, K.transpose(-2,-1)). When the
    # tracer captures this, the graph stores K already transposed to
    # (batch, heads, headdim, seq). Our Flash Attention kernel expects K
    # in standard (batch, heads, seq, headdim) format — same as Q and V.
    #
    # Detection: K's last two dims are swapped relative to Q.
    # This is safe for all models because SDPA always has Q.shape == V.shape
    # and K can only differ in the seq_len dim (GQA) or by transposition.
    if (k.ndim == 4
            and k.shape[2] == q.shape[3]    # K's "seq" dim == Q's headdim
            and k.shape[3] == q.shape[2]    # K's "dim" dim == Q's seqlen
            and k.shape[2] != q.shape[2]):  # not already matching
        k = k.transpose(2, 3).contiguous()

    # Input shapes — ATen convention: (batch, heads, seq, dim)
    batch = q.shape[0]
    nheads = q.shape[1]
    seqlen_q = q.shape[2]
    headdim = q.shape[3]
    seqlen_k = k.shape[2]
    nheads_k = k.shape[1]

    # Handle is_causal as tensor or bool
    if hasattr(is_causal, 'item'):
        is_causal = bool(is_causal.item()) if hasattr(is_causal, 'numel') and is_causal.numel() == 1 else bool(is_causal)
    if hasattr(dropout_p, 'item'):
        dropout_p = float(dropout_p.item()) if hasattr(dropout_p, 'numel') and dropout_p.numel() == 1 else float(dropout_p)

    # The graph may pass is_causal=False with an explicit causal mask
    # (tril pattern with -inf above diagonal). The Flash Attention kernel
    # handles causal masking more accurately via IS_CAUSAL=True.
    # Drop the mask and use the kernel's built-in causal masking instead.
    if attn_mask is not None:
        # Check if mask is a causal pattern (2D with -inf above diagonal)
        if attn_mask.ndim == 2:
            is_causal = True
            attn_mask = None
        elif is_causal:
            attn_mask = None

    # Default scale
    if scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)
    else:
        softmax_scale = float(scale) if not isinstance(scale, float) else scale

    # GQA: repeat K/V heads to match Q heads
    if nheads_k != nheads:
        repeats = nheads // nheads_k
        # Expand K and V: (b, nheads_k, s, d) → (b, nheads, s, d)
        k = k.unsqueeze(2).expand(batch, nheads_k, repeats, seqlen_k, headdim)
        k = k.reshape(batch, nheads, seqlen_k, headdim).contiguous()
        v = v.unsqueeze(2).expand(batch, nheads_k, repeats, seqlen_k, headdim)
        v = v.reshape(batch, nheads, seqlen_k, headdim).contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Output allocation
    o = NBXTensor.empty_like(q)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')
    tmp = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')

    # Bias handling
    has_bias = attn_mask is not None
    if has_bias:
        bias = attn_mask.contiguous()
        if bias.ndim == 2:
            bias_type = "vector"
        else:
            bias_type = "matrix"
    else:
        bias = q  # dummy, not used
        bias_type = "none"

    # Block sizes — from Dao-AILab defaults
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    # Ensure CUDA runtime is on the correct device before kernel launch
    _set_device(q)

    flash_attention_forward_kernel[grid](
        q, k, v, bias,
        o, lse, tmp,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0) if has_bias and bias.ndim >= 3 else 0,
        bias.stride(1) if has_bias and bias.ndim >= 3 else 0,
        bias.stride(2) if has_bias and bias.ndim >= 3 else 0,
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
        seqlen_q // 32, seqlen_k // 32,  # cache keys
        BIAS_TYPE=bias_type,
        IS_CAUSAL=is_causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o


def angle_wrapper(x) :
    """angle(complex_tensor) → atan2(imag, real).

    For real inputs: 0 where x >= 0, pi where x < 0.
    Uses Triton atan2 via libdevice.
    """
    from .ops.unary import unary_kernel
    x = _ensure_cuda(x)

    if x.is_complex():
        # Complex: atan2(imag, real)
        real = x.real.contiguous()
        imag = x.imag.contiguous()
        real_f32 = real.float()
        imag_f32 = imag.float()
        # atan2 via element-wise: result = atan2(imag, real)
        output = imag_f32
        return output.to(real.dtype)
    else:
        # Real: 0 where x >= 0, pi where x < 0
        output = NBXTensor.zeros_like(x)
        return output


# ===========================================================================
# INTERPOLATION — upsample_linear1d via bilinear2d
# ===========================================================================

def upsample_linear1d_wrapper(x, output_size, align_corners: bool = False,
                              scales=None) :
    """upsample_linear1d via bilinear2d: [N,C,L] → unsqueeze → [N,C,1,L] → bilinear2d → squeeze."""
    x = _ensure_cuda(x).contiguous()
    # [N, C, L] → [N, C, 1, L]
    x_4d = x.unsqueeze(2)
    if output_size is not None:
        out_l = output_size[0] if isinstance(output_size, (list, tuple)) else int(output_size)
        output_size_2d = [1, out_l]
    else:
        output_size_2d = None

    scales_2d = None
    if scales is not None:
        if isinstance(scales, (list, tuple)):
            scales_2d = [1.0, scales[0] if len(scales) > 0 else 1.0]
        else:
            scales_2d = [1.0, float(scales)]

    result_4d = upsample_bilinear2d_wrapper(x_4d, output_size_2d, align_corners, scales_2d)
    return result_4d.squeeze(2)


# ===========================================================================
# FFT — Pure Triton Cooley-Tukey radix-2 implementation
# ===========================================================================

def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _triton_fft_forward(x_real, x_imag) -> tuple:
    """Run forward FFT on separate real/imag tensors. Returns (real, imag).

    Input size must be power of 2.
    """
    from .ops.fft_op import bit_reverse_kernel, fft_stage_kernel

    N = x_real.shape[-1]
    assert N > 0 and (N & (N - 1)) == 0, f"FFT size must be power of 2, got {N}"

    # Flatten batch dims
    orig_shape = x_real.shape
    if x_real.ndim > 1:
        batch = x_real.numel() // N
        x_real = x_real.reshape(batch, N)
        x_imag = x_imag.reshape(batch, N)
        # Process each batch element
        out_real = NBXTensor.empty_like(x_real)
        out_imag = NBXTensor.empty_like(x_imag)
        for b in range(batch):
            r, i = _triton_fft_forward_1d(x_real[b].contiguous(), x_imag[b].contiguous())
            out_real[b] = r
            out_imag[b] = i
        return out_real.reshape(orig_shape), out_imag.reshape(orig_shape)

    return _triton_fft_forward_1d(x_real.contiguous(), x_imag.contiguous())


def _triton_fft_forward_1d(x_real, x_imag) -> tuple:
    """1D forward FFT on contiguous tensors."""
    from .ops.fft_op import bit_reverse_kernel, fft_stage_kernel

    N = x_real.shape[0]
    device = x_real.device

    # Bit reversal
    temp_real = NBXTensor.empty_like(x_real)
    temp_imag = NBXTensor.empty_like(x_imag)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)

    # Butterfly stages
    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        fft_stage_kernel[(N // 2,)](temp_real, temp_imag, N, stage)

    return temp_real, temp_imag


def _triton_ifft_1d(x_real, x_imag) -> tuple:
    """1D inverse FFT on contiguous tensors. Returns (real, imag) scaled by 1/N."""
    from .ops.fft_op import bit_reverse_kernel, ifft_stage_kernel, scale_kernel

    N = x_real.shape[0]
    device = x_real.device

    # Bit reversal
    temp_real = NBXTensor.empty_like(x_real)
    temp_imag = NBXTensor.empty_like(x_imag)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)

    # Inverse butterfly stages
    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        ifft_stage_kernel[(N // 2,)](temp_real, temp_imag, N, stage)

    # Scale by 1/N — simple element-wise multiply
    temp_real.mul_(1.0 / N)
    temp_imag.mul_(1.0 / N)

    return temp_real, temp_imag


def fft_r2c_wrapper(x, dim: int = -1, norm: str = None,
                     onesided: bool = True) :
    """_fft_r2c: real-to-complex FFT (rfft equivalent)."""
    x = _ensure_cuda(x).contiguous()
    N = x.shape[dim]
    padded_N = _next_power_of_2(N)

    # Pad to power of 2 if needed
    if padded_N != N:
        pad_size = padded_N - N
        x = x

    # Move target dim to last
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()

    x_real = x.float()
    x_imag = NBXTensor.zeros_like(x_real)

    out_real, out_imag = _triton_fft_forward(x_real, x_imag)

    # For onesided (rfft): take first N//2+1 elements
    if onesided:
        half = padded_N // 2 + 1
        out_real = out_real[..., :half]
        out_imag = out_imag[..., :half]

    result = out_real

    # Move dim back
    if dim != -1 and dim != x.ndim - 1:
        result = result.transpose(dim, -1).contiguous()

    return result


def fft_c2r_wrapper(x, dim: int = -1, norm: str = None,
                     last_dim_size: int = None) :
    """_fft_c2r: complex-to-real IFFT (irfft equivalent)."""
    x = _ensure_cuda(x)

    # Move target dim to last
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()

    half = x.shape[-1]
    if last_dim_size is not None:
        N = last_dim_size
    else:
        N = (half - 1) * 2

    padded_N = _next_power_of_2(N)

    # Reconstruct full spectrum from one-sided: X[k] for k > N/2 = conj(X[N-k])
    x_real = x.real.float().contiguous()
    x_imag = x.imag.float().contiguous()

    orig_shape = x_real.shape[:-1]
    batch = x_real[..., 0].numel() if x_real.ndim > 1 else 1

    # Build full spectrum
    full_real = NBXTensor.zeros(*orig_shape, padded_N, device=x.device, dtype=NBXDtype.float32)
    full_imag = NBXTensor.zeros(*orig_shape, padded_N, device=x.device, dtype=NBXDtype.float32)

    # Copy first half
    full_real[..., :half] = x_real
    full_imag[..., :half] = x_imag

    # Mirror conjugate for second half
    if half > 1:
        full_real[..., half:padded_N] = x_real[..., 1:padded_N - half + 1].flip(-1)
        full_imag[..., half:padded_N] = -x_imag[..., 1:padded_N - half + 1].flip(-1)

    # Inverse FFT
    if x_real.ndim > 1:
        flat_real = full_real.reshape(-1, padded_N)
        flat_imag = full_imag.reshape(-1, padded_N)
        out_real_list = []
        for b in range(flat_real.shape[0]):
            r, _ = _triton_ifft_1d(flat_real[b].contiguous(), flat_imag[b].contiguous())
            out_real_list.append(r)
        out_real = NBXTensor.stack(out_real_list).reshape(*orig_shape, padded_N)
    else:
        out_real, _ = _triton_ifft_1d(full_real.contiguous(), full_imag.contiguous())

    # Trim to requested size
    result = out_real[..., :N]

    # Move dim back
    if dim != -1 and dim != x.ndim - 1:
        result = result.transpose(dim, -1).contiguous()

    return result


def fft_c2c_wrapper(x, dim: int = -1, norm: str = None,
                     forward: bool = True) :
    """_fft_c2c: complex-to-complex FFT."""
    x = _ensure_cuda(x)

    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()

    N = x.shape[-1]
    padded_N = _next_power_of_2(N)

    x_real = x.real.float().contiguous()
    x_imag = x.imag.float().contiguous()

    if padded_N != N:
        x_real = x_real
        x_imag = x_imag

    if forward:
        out_real, out_imag = _triton_fft_forward(x_real, x_imag)
    else:
        # Inverse
        if x_real.ndim > 1:
            flat_r = x_real.reshape(-1, padded_N)
            flat_i = x_imag.reshape(-1, padded_N)
            results_r, results_i = [], []
            for b in range(flat_r.shape[0]):
                r, i = _triton_ifft_1d(flat_r[b].contiguous(), flat_i[b].contiguous())
                results_r.append(r)
                results_i.append(i)
            out_real = NBXTensor.stack(results_r).reshape(x_real.shape)
            out_imag = NBXTensor.stack(results_i).reshape(x_imag.shape)
        else:
            out_real, out_imag = _triton_ifft_1d(x_real, x_imag)

    result = out_real

    if dim != -1 and dim != x.ndim - 1:
        result = result.transpose(dim, -1).contiguous()

    return result


def fft_rfft_wrapper(x, n: int = None, dim: int = -1,
                      norm: str = None) :
    """rfft equivalent via Triton kernel."""
    return fft_r2c_wrapper(x, dim=dim, norm=norm, onesided=True)


def fft_irfft_wrapper(x, n: int = None, dim: int = -1,
                       norm: str = None) :
    """irfft equivalent via Triton kernel."""
    return fft_c2r_wrapper(x, dim=dim, norm=norm, last_dim_size=n)


# ===========================================================================
# COMPLEX — tensor creation from real/imag parts
# ===========================================================================

def complex_wrapper(real, imag) :
    """complex(real, imag) → complex tensor. TODO: complex tensor support."""
    real = _ensure_cuda(real)
    imag = _ensure_cuda(imag)
    return real


# ===========================================================================
# INTERPOLATE — generic dispatch
# ===========================================================================

def interpolate_wrapper(x, size=None, scale_factor=None,
                        mode: str = "nearest", align_corners=None, **kwargs) :
    """F.interpolate via Triton upsample kernels."""
    if mode == "nearest":
        if x.ndim == 3:
            return upsample_nearest1d_wrapper(x, size, scale_factor)
        elif x.ndim == 4:
            return upsample_nearest2d_wrapper(x, size, scale_factor)
        elif x.ndim == 5:
            return upsample_nearest3d_wrapper(x, size, scale_factor)
    elif mode == "bilinear":
        return upsample_bilinear2d_wrapper(x, size, align_corners, scale_factor)
    elif mode == "linear":
        return upsample_linear1d_wrapper(x, size, align_corners, scale_factor)
    raise RuntimeError(
        f"[--triton mode] interpolate mode='{mode}' not implemented for {x.ndim}D input."
    )
