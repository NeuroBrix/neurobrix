"""Triton Kernel Wrappers — triton mode execution layer.

Each wrapper:
  1. Receives NBXTensor
  2. Allocates output via NBXTensor.empty (cudaMalloc)
  3. Launches @triton.jit kernel
  4. Returns NBXTensor

Dependencies: triton, NBXTensor. Used exclusively by dispatch.py.
"""

import os
import triton

from .nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, _broadcast_shapes, _set_device, dtype_size

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

from .ops.add import add_forward_kernel, add_scalar_kernel, add_bias_broadcast_kernel
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
from .ops._autotune_policy import NBX_FORCE_FP32_ACCUM

# === Fused MoE ===

from .ops.fused_moe import fused_moe_kernel, silu_and_mul_kernel, silu_mul_split_kernel
from .ops.dtype_convert import bf16_to_fp16_kernel

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
from .ops.depthwise_conv2d import depthwise_conv2d_kernel
from .ops.batch_norm import batch_norm_forward_kernel
from .ops.cumsum import scan_part_sum_kernel, add_base_sum_kernel
from .ops.scatter_op import scatter_kernel, scatter_add_kernel, scatter_reduce_amax_kernel, scatter_reduce_amin_kernel
from .ops.gather_op import gather_kernel
from .ops.avg_pool2d import avg_pool2d_forward_kernel
from .ops.max_pool2d import max_pool2d_forward_kernel
from .ops.max_pool3d import max_pool3d_forward_kernel
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
from .ops.floor import floor_forward_kernel, ceil_forward_kernel, round_forward_kernel, trunc_forward_kernel

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
from .ops.index_add import index_add_gather_kernel
from .ops.index_put_op import index_put_kernel
from .ops.sort_op import radix_sort_histogram_kernel, radix_sort_sweep_kernel

# === Phase 5: RoPE, spatial, RNG, remaining ===

from .ops.rope import rope_forward_kernel
from .ops.pixel_shuffle import (
    pixel_shuffle_kernel,
    pixel_shuffle_broadcast_aware_kernel,
)
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


# ---------------------------------------------------------------------------
# Hardware capability surface — single source of truth for the Triton side.
#
# Default is True (modern hardware). The CLI / serving layer calls
# set_hardware_profile(profile) once at executor construction to override
# with the actual PrismProfile.has_native_bf16. Kernel wrappers then read
# _NBX_HAS_NATIVE_BF16 for runtime dtype decisions (e.g. forcing fp32
# output on fp16 mm/bmm/addmm when bf16 is unavailable).
#
# Rationale: NeuroBrix is universal-data-driven. The fp16 overflow
# protection mirrors what the native DtypeEngine's AMP rules already do
# on pre-Ampere GPUs — so it's an existing policy, not a V100 hardcode.
# On Ampere+ (has_native_bf16=True) the protection path is a no-op and
# all kernel wrappers behave identically to before this was introduced.
# ---------------------------------------------------------------------------
_NBX_HAS_NATIVE_BF16 = True
_NBX_HW_PROFILE = None


def set_hardware_profile(profile) -> None:
    """Configure Triton kernel wrappers from a PrismProfile.

    Called once per process, typically by the CLI and serving entry
    points right before constructing RuntimeExecutor. Plumbs the
    hardware capability into a module-level flag so kernel wrappers
    can make dtype-safety decisions without carrying a profile through
    every call site. The full profile is also stashed so the weight
    loader can query per-device VRAM when deciding whether to
    bind-time upcast fp16 weights to fp32 on pre-Ampere hardware.

    Accepts any object with a `has_native_bf16` attribute (duck-typed
    so tests can pass a mock without importing the full Prism stack).
    """
    global _NBX_HAS_NATIVE_BF16, _NBX_HW_PROFILE
    if profile is None:
        return
    _NBX_HW_PROFILE = profile
    _NBX_HAS_NATIVE_BF16 = bool(getattr(profile, "has_native_bf16", True))


def has_native_bf16() -> bool:
    """Expose the cached hardware capability flag for non-kernel callers."""
    return _NBX_HAS_NATIVE_BF16


def get_hardware_profile():
    """Return the PrismProfile stashed by set_hardware_profile, or None."""
    return _NBX_HW_PROFILE


# ---------------------------------------------------------------------------
# Per-component runtime context: compute_dtype + activations_fp16_safe flag.
# These globals are set by TritonSequence.run() at hot-loop entry and
# restored at exit (try/finally). Single-threaded NeuroBrix runtime by
# convention (sequence.py:325 docstring), so module-level state is safe.
#
# - _NBX_COMPUTE_DTYPE: the per-component compute dtype Prism allocated
#   for this sequence. Read by self-managed wrappers (conv2d_wrapper)
#   to decide output dtype, mirroring what cuDNN does in compiled mode.
# - _NBX_ACTIVATIONS_FP16_SAFE: per-component opt-in flag from
#   forge/config/model_registry.yml. When True, ops in AMP_FP32_OPS
#   that produce fp32 internally cast their output back to compute_dtype
#   (rms_norm, div, etc.) — VRAM-preserving for models whose activations
#   are confirmed within fp16 range by measure_activation_ranges. Default
#   False keeps the conservative fp32-output behavior.
# ---------------------------------------------------------------------------
_NBX_COMPUTE_DTYPE = None
_NBX_ACTIVATIONS_FP16_SAFE: bool = False


def set_compute_dtype(dt) -> None:
    """Set the active per-component compute dtype. Called by TritonSequence.run."""
    global _NBX_COMPUTE_DTYPE
    _NBX_COMPUTE_DTYPE = dt


def get_compute_dtype():
    """Return the active per-component compute dtype, or None if not set."""
    return _NBX_COMPUTE_DTYPE


def set_activations_fp16_safe(safe: bool) -> None:
    """Set the activations-fp16-safe opt-in flag for the active component."""
    global _NBX_ACTIVATIONS_FP16_SAFE
    _NBX_ACTIVATIONS_FP16_SAFE = bool(safe)


def get_activations_fp16_safe() -> bool:
    """Return the activations-fp16-safe flag for the active component."""
    return _NBX_ACTIVATIONS_FP16_SAFE


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


def _transfer_to_device(tensor, target_device_idx: int):
    """Transfer NBXTensor to a different GPU via cudaMemcpy D2D."""
    if not hasattr(tensor, '_device_idx') or tensor._device_idx == target_device_idx:
        return tensor
    src = tensor.contiguous()
    dst = NBXTensor.empty(src._shape, src._dtype, f"cuda:{target_device_idx}")
    DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), src._nbytes, kind=3)
    return dst


# Dtype promotion order: wider dtype wins.
# Maps (dtype_a, dtype_b) → result dtype following PyTorch semantics.
_DTYPE_PRIORITY = {
    NBXDtype.bool_: 0,
    NBXDtype.uint8: 1,
    NBXDtype.int8: 2,
    NBXDtype.int16: 3,
    NBXDtype.int32: 4,
    NBXDtype.int64: 5,
    NBXDtype.float16: 6,
    NBXDtype.bfloat16: 6,
    NBXDtype.float32: 7,
    NBXDtype.float64: 8,
    NBXDtype.complex64: 9,
    NBXDtype.complex128: 10,
}


def _wider_dtype(a_dtype, b_dtype):
    """Return the wider of two NBXDtype values for binary op promotion."""
    if a_dtype == b_dtype:
        return a_dtype
    pa, pb = _DTYPE_PRIORITY.get(a_dtype, 7), _DTYPE_PRIORITY.get(b_dtype, 7)
    if pa >= pb:
        return a_dtype
    return b_dtype


def _prepare_unary(x):
    """Prepare a single tensor for kernel launch."""
    x = x.contiguous()
    _set_device(x)
    return x, x.numel(), None


def _prepare_binary(a, b):
    """Prepare two tensors for a binary element-wise kernel.

    UNIVERSAL CONTRACT:
      - Scalar position normalized: tensor ALWAYS in `a`, scalar ALWAYS in `b`
      - Dtype aligned: both upcast to wider dtype
      - Device aligned: both on same GPU
      - Shape broadcast: expanded to common shape
      - Contiguous: both made contiguous before kernel launch

    Returns: (a, b_or_scalar, output, n_elements, device_ctx, is_scalar)
    When is_scalar=True: a is tensor, b is Python scalar (int/float/bool).
    """
    # --- Scalar path: normalize so tensor=a, scalar=b ---
    if _is_scalar(b):
        a = a.contiguous()
        _set_device(a)
        output = NBXTensor.empty_like(a)
        return a, _to_scalar(b), output, a.numel(), None, True

    if _is_scalar(a):
        # Swap: tensor goes to `a`, scalar goes to `b`
        b = b.contiguous()
        _set_device(b)
        output = NBXTensor.empty_like(b)
        return b, _to_scalar(a), output, b.numel(), None, True

    # --- Two-tensor path ---

    # 1) Align dtypes (upcast to wider)
    common_dtype = _wider_dtype(a._dtype, b._dtype)
    if a._dtype != common_dtype:
        a = a.to(common_dtype)
    if b._dtype != common_dtype:
        b = b.to(common_dtype)

    # 2) Align devices
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _transfer_to_device(b, a._device_idx)

    # 3) Broadcast shapes
    if a.shape != b.shape:
        out_shape = _broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    _set_device(a)
    output = NBXTensor.empty_like(a)
    return a, b, output, a.numel(), None, False


def _prepare_comparison(a, b):
    """Prepare two tensors for a comparison kernel (output is bool).

    Same guards as _prepare_binary (dtype align, device align, broadcast,
    contiguous) but output tensor is always bool dtype.

    CONTRACT: when is_scalar=True, tensor is in `a`, scalar is in `b`.
    """
    if _is_scalar(b):
        a = a.contiguous()
        _set_device(a)
        output = NBXTensor.empty(a.shape, NBXDtype.bool_, device=a.device)
        return a, _to_scalar(b), output, a.numel(), None, True

    if _is_scalar(a):
        # Swap: tensor goes to `a`, scalar goes to `b`
        b = b.contiguous()
        _set_device(b)
        output = NBXTensor.empty(b.shape, NBXDtype.bool_, device=b.device)
        return b, _to_scalar(a), output, b.numel(), None, True

    # Align dtypes for comparison (both must be same type for the kernel)
    common_dtype = _wider_dtype(a._dtype, b._dtype)
    if a._dtype != common_dtype:
        a = a.to(common_dtype)
    if b._dtype != common_dtype:
        b = b.to(common_dtype)

    # Align devices
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _transfer_to_device(b, a._device_idx)

    # Broadcast
    if a.shape != b.shape:
        out_shape = _broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    _set_device(a)
    output = NBXTensor.empty(a.shape, NBXDtype.bool_, device=a.device)
    return a, b, output, a.numel(), None, False


# ===========================================================================
# ACTIVATION WRAPPERS
# ===========================================================================

def relu(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    relu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def silu(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    silu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def gelu(x, approximate: str = 'none') :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    gelu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), approximate=(approximate == 'tanh'),
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sigmoid_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    sigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def tanh_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    tanh_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def hardsigmoid(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    hardsigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def hardswish(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    hardswish_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def leaky_relu(x, negative_slope: float = 0.01) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    leaky_relu_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), negative_slope,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def elu(x, alpha: float = 1.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    elu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def mish(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    mish_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def selu_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    selu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# UNARY ELEMENT-WISE WRAPPERS
# ===========================================================================

def neg(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    neg_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


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
    exp_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sin(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    sin_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def cos(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    cos_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def _promote_int_unary(x):
    """Float-math unary ops (rsqrt/sqrt/log/exp/reciprocal/...) promote integer
    or bool input to float32, matching PyTorch type promotion. Without it the
    kernel computes in the integer dtype AND NBXTensor.empty_like keeps the int
    output — e.g. rsqrt(int64 2) truncates 0.7071 to 0, which silently collapsed
    the Kokoro AdainResBlk1d residual `(h + sc) * rsqrt(2)` to all-zeros (the
    rsqrt(2) constant was traced as an int64 scalar; torch auto-promotes it).
    No-op for tensors already in a floating dtype."""
    if hasattr(x, "is_floating_point") and not x.is_floating_point():
        return x.to(NBXDtype.float32)
    return x


def rsqrt(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    rsqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def sqrt_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    sqrt_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
    abs_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def log_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    log_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def reciprocal(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    reciprocal_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
        pow_forward_kernel[_1d_grid(x.numel())](
            x, output, x.numel(), float(exponent),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def erf(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    erf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def floor_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    floor_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def ceil_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    ceil_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def round_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    round_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def trunc_wrapper(x):
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    trunc_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def copy_to(x, dtype: object) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x, dtype=dtype)
    _set_device(x)
    copy_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
    if (not _is_scalar(b)
            and hasattr(a, '_dtype') and hasattr(b, '_dtype')
            and a.ndim >= 1 and b.ndim == 1
            and a.shape[-1] == b.shape[0]
            and a.is_contiguous()):
        # Dtype align (mirror _prepare_binary).
        common_dtype = _wider_dtype(a._dtype, b._dtype)
        if a._dtype != common_dtype:
            a = a.to(common_dtype)
        if b._dtype != common_dtype:
            b = b.to(common_dtype)
        # Device align.
        if (hasattr(a, '_device_idx') and hasattr(b, '_device_idx')
                and a._device_idx != b._device_idx):
            b = _transfer_to_device(b, a._device_idx)
        n = a.numel()
        feat_dim = b.shape[0]
        output = NBXTensor.empty_like(a)
        _set_device(a)
        add_bias_broadcast_kernel[_1d_grid(n)](
            a, b, output, n, feat_dim, alpha,
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
        return output

    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        add_scalar_kernel[_1d_grid(n)](a, output, n, float(b) * alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        add_forward_kernel[_1d_grid(n)](a, b, output, n, alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
        other = _transfer_to_device(other, target._device_idx)
    # Dtype alignment. In-place semantics require writing into target's
    # buffer, so target's dtype is the result dtype. If `other` is at a
    # different precision, cast it (the cast allocates a transient sized
    # like other — same VRAM cost as a non-in-place add only when other
    # < target dtype). If target is NARROWER than other, in-place would
    # lose precision: fall back to the standard non-in-place `add` so the
    # standard `_wider_dtype` upcast path kicks in.
    if target._dtype != other._dtype:
        if _wider_dtype(target._dtype, other._dtype) == target._dtype:
            other = other.to(target._dtype)
        else:
            return add(target, other, alpha=alpha)
    other = other.contiguous()
    n = target.numel()
    _set_device(target)
    add_forward_kernel[_1d_grid(n)](
        target, other, target, n, alpha,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        # Contract: tensor is always `a`, scalar is always `b`
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
    if (isinstance(a, NBXTensor) and a.is_complex()) or (isinstance(b, NBXTensor) and b.is_complex()):
        return _complex_addsub(a, b, alpha, is_sub=True)
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
    _set_device(cond)
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
    _set_device(x)
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
    _set_device(x_2d)
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


def layer_norm_wrapper(x, normalized_shape, weight=None, bias=None, eps=1e-5,
                       cudnn_enable=True) :
    """aten::layer_norm — the high-level op returns ONLY the normalized output
    (native_layer_norm also returns mean/rstd). The triton runtime stores a
    tuple result as out_0=the-whole-tuple when the graph declares a single
    output, so a downstream op (dropout) then receives a tuple. Return just the
    output tensor here."""
    out = native_layer_norm(x, normalized_shape, weight, bias, eps)
    return out[0] if isinstance(out, (tuple, list)) else out


def rms_norm(x, weight, eps=1e-6, epsilon=None):
    """RMSNorm wrapper.

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
    x = _ensure_cuda(x)
    weight = _ensure_cuda(weight)
    feat_dim = x.shape[-1]
    batch_dim = x.numel() // feat_dim

    x_contig = x.contiguous()
    x_2d = x_contig.view(batch_dim, feat_dim)
    if x_contig is not x:
        # contiguous() allocated a fresh buffer with no other holder —
        # write rms_norm output directly into it (in-place) instead of
        # paying for a second 8 GiB allocation.
        output_2d = x_2d
    else:
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
    _set_device(x_2d)
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
    _set_device(x_2d)
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

def _matmul_out_dtype(a, M: int = 1, force_fp32: bool = False):
    """Output dtype for mm/bmm/addmm/mv/addmv.

    Kernels accumulate in fp32. Three overlapping policies decide the
    *store* dtype:

    1. **Hardware gate — fp16 on pre-Ampere (no native bf16)**: force
       fp32 output unconditionally for fp16 inputs. Mirrors what the
       native DtypeEngine AMP already does on V100/Turing — fp16
       accumulation over K can exceed 65,504 even when per-element
       magnitudes look reasonable, and the store saturates to ±Inf
       which cascades through residual adds (openaudio DualAR
       Apr-2026, Qwen3-30B earlier). On Ampere+ / any bf16-capable
       card this path is a no-op and the legacy M-gated rule applies.
       The flag comes from `PrismProfile.has_native_bf16` via
       `set_hardware_profile()` — ZERO V100 hardcode, data-driven
       hardware capability. See the NeuroBrix universal-engine rule
       in CLAUDE.md §5.

    2. **M <= 4 (decode path)**: store fp32 even with bf16 inputs or
       on bf16-capable hardware. Tiny tensors (1 row × N), negligible
       memory cost, catches rare decode-loop drift.

    3. **M > 4 (prefill / spatial path), bf16-capable hardware**:
       store in input dtype. Keeps diffusion spatial mm (PixArt-Sigma
       M=4096 × CFG=2) within VRAM budget. Prefill-magnitude
       activations stay within the dtype range because a single
       forward pass doesn't chain accumulations like a deep decode.

    `force_fp32=True` overrides everything (used by bmm for diffusion
    attention on all hardware).

    **Phase 1.5 attempt (2026-05) — REVERTED.** A bypass of hardware
    gate (1) when `activations_fp16_safe=True` was prototyped on the
    intuition that fp16-safe activations would unlock HMMA-fp16 via
    Triton. Empirical measurement on Sana 1024 (M=2048×2240) refuted
    this: pure fp16 path was 6.52 ms vs 2.82 ms current path (2.3×
    WORSE). Triton fp16×fp16 → fp32 accumulator on Volta does NOT
    lower to HMMA-fp16 in this kernel — it falls back to fp32
    emulation. cuBLAS reaches 0.295 ms via its own dedicated HMMA-fp16
    code path that we cannot replicate by toggling a flag. The
    force_fp32 hardware gate stays unconditional. Real speedup
    requires either a HMMA-tuned Triton kernel OR adopting
    FlagGems / similar — see Phase 1.5 follow-up.
    """
    dt = a.nbx_dtype if hasattr(a, 'nbx_dtype') else a.dtype
    is_fp16 = (dt == NBXDtype.float16)
    is_bf16 = (dt == NBXDtype.bfloat16)
    is_half = is_fp16 or is_bf16

    # NBX_FORCE_FP32_ACCUM diagnostic: always store fp32 for any half
    # input. Pairs with the input upcast in mm/bmm/addmm wrappers so
    # the entire matmul chain operates in fp32 - isolates the dtype
    # variable from the P-SANA-4KPX-RUNTIME bug hunt.
    if NBX_FORCE_FP32_ACCUM and is_half:
        return NBXDtype.float32

    # (1) Hardware gate: fp16 on hardware without native bf16 gets fp32.
    if is_fp16 and not _NBX_HAS_NATIVE_BF16:
        return NBXDtype.float32

    # (2) + (3) legacy M gate + force_fp32 escape hatch.
    if is_half and (M <= 4 or force_fp32):
        return NBXDtype.float32

    return dt


def mm(a, b) :
    """Matrix multiplication: C = A @ B.
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles fp16/bf16 overflow protection externally.

    Small-M fast path (M <= 4): route to mv_wrapper per-row. matmul_kernel
    tiles on BLOCK_M=64 and wastes 63/64 threads when M=1 (decode). mv_kernel
    parallelises on N only and is ~10–15× faster in that regime. Universal:
    works for LLM decode (M=1), short-sequence audio, small-batch inference.
    For M > 4, stays on matmul_kernel which is optimal for larger GEMM
    (diffusion spatial mm with M in the thousands).
    """
    a = _ensure_cuda(a)
    b = _ensure_cuda(b)
    # Multi-device: align b to a's device
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _transfer_to_device(b, a._device_idx)

    # Hardware-gated fp16 overflow protection. On GPUs without native bf16
    # (Volta/Turing/older), fp16 inputs whose upstream producers had large
    # magnitudes get saturated to ±Inf when a graph-level _to_copy cast
    # them back to fp16 — the next mm then propagates Inf through HMMA →
    # NaN cascade. Mirror the native DtypeEngine AMP path: upcast the
    # ACTIVATION to fp32. On bf16-capable hardware this branch is a no-op.
    # NBXTensor.dtype returns triton.language.dtype (for kernel dispatch);
    # NBXTensor.nbx_dtype returns the NBXDtype enum used by these guards.
    #
    # NOTE (Phase 1.5 attempt, reverted): an activations_fp16_safe opt-in
    # bypass was prototyped but empirically slower on Sana 1024 (6.5 vs
    # 2.8 ms — Triton fp16×fp16 does NOT lower to HMMA on Volta in this
    # kernel; cuBLAS reaches 0.295 ms via dedicated HMMA path we cannot
    # match by flag-flipping). Real speedup needs a HMMA-tuned Triton
    # kernel or FlagGems adoption — see Phase 1.5 follow-up.
    a_nbx = a.nbx_dtype
    b_nbx = b.nbx_dtype
    # NBX_FORCE_FP32_ACCUM diagnostic: upcast BOTH inputs to fp32 on any
    # hardware when the env var is set. Sacrifices VRAM/perf to isolate
    # the dtype-intermediate hypothesis in P-SANA-4KPX-RUNTIME. The
    # existing fp16-only Volta gate below missed bf16 inputs, which on
    # Volta have no native HMMA support and may degrade through Triton's
    # tl.dot lowering.
    if NBX_FORCE_FP32_ACCUM and a_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32
    if NBX_FORCE_FP32_ACCUM and b_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        b = b.to(NBXDtype.float32)
        b_nbx = NBXDtype.float32
    if not _NBX_HAS_NATIVE_BF16 and a_nbx == NBXDtype.float16:
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32

    # Dtype alignment. Three situations:
    #   1. Same dtype  → no-op.
    #   2. fp32 act × fp16 weight on pre-Ampere → this is the common case
    #      after step 2. Keep the weight fp16 in memory; the kernel
    #      promotes the b tile to fp32 inline via PROMOTE_B. No heap alloc.
    #   3. Anything else (fp32 × bf16, bf16 × fp16, etc.) → widen to the
    #      common dtype. Rare; typically a downstream force_fp32 bmm feeding
    #      the next matmul.
    promote_b = (not _NBX_HAS_NATIVE_BF16
                 and a_nbx == NBXDtype.float32
                 and b_nbx == NBXDtype.float16)
    if a_nbx != b_nbx and not promote_b:
        _order = (NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.float16)
        widest = next(d for d in _order if d in (a_nbx, b_nbx))
        if a_nbx != widest:
            a = a.to(widest)
        if b_nbx != widest:
            b = b.to(widest)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {K} vs {K2}"

    # === TEMP DIAGNOSTIC: record M,N,K per call ===
    import os as _os
    if _os.environ.get("NBX_MM_SHAPES") == "1":
        import collections, atexit
        if "_rec" not in mm.__dict__:
            mm.__dict__["_rec"] = collections.Counter()
            mm.__dict__["_call_idx"] = [0]

            def _dump_shapes():
                rec = mm.__dict__["_rec"]
                print("\n[NBX_MM_SHAPES] unique (M,N,K) × count:")
                for (m_, n_, k_), c in sorted(rec.items(),
                                              key=lambda x: -x[1]):
                    print(f"  M={m_:5d} N={n_:5d} K={k_:5d}  × {c}")
                print(f"[NBX_MM_SHAPES] total calls: {sum(rec.values())}")
            atexit.register(_dump_shapes)
        mm.__dict__["_rec"][(M, N, K)] += 1
    # ================================================

    # Small-M GEMV path. Do NOT call b.contiguous() on the full weight:
    # if b is a pre-transposed stride-view (from _eliminate_weight_transpose_ops),
    # b.t() restores the original contiguous layout for free. Calling
    # b.contiguous() first would copy the entire weight — defeating the point.
    if M <= 4:
        _set_device(a)
        a = a.contiguous()
        bt = b.t()  # (N, K); contiguous if b came from pre-transpose
        dtype_out = _matmul_out_dtype(a, M)
        dev_str = (f"cuda:{a._device_idx}" if hasattr(a, '_device_idx')
                   else 'cuda')
        c = NBXTensor.empty((M, N), device=dev_str, dtype=dtype_out)
        for i in range(M):
            c[i] = mv_wrapper(bt, a[i])
        return c

    a = a.contiguous()
    b = b.contiguous()
    out_dtype = _matmul_out_dtype(a, M)
    c = NBXTensor.empty((M, N), device=f"cuda:{a._device_idx}" if hasattr(a, '_device_idx') else 'cuda',
                        dtype=out_dtype)
    # IEEE mode: force strict fp32 tl.dot on pre-Ampere when we've promoted
    # inputs to fp32 for overflow protection. Otherwise tl.dot silently
    # casts fp32 → fp16 HMMA → saturates at ±65504 → NaN/Inf cascade.
    ieee = (not _NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)
    # Phase 1.5 autotune: BLOCK_*/GROUP_M/num_warps/num_stages chosen
    # adaptively per (M, N, K, IEEE_PRECISION, PROMOTE_B). Grid uses
    # META lambda so autotune-selected blocks drive the launch shape.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _set_device(a)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        IEEE_PRECISION=ieee,
        PROMOTE_B=promote_b,
    )
    return c


def bmm(a, b) :
    """Batched matrix multiplication: C[i] = A[i] @ B[i].
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally.

    Small-M fast path (M <= 4): per-batch mv loop. Applies to decode-path
    bmm (attention Q@K.T with seq=1) and similar short-sequence patterns.
    """
    a = _ensure_cuda(a)
    b = _ensure_cuda(b)
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _transfer_to_device(b, a._device_idx)

    # Hardware-gated fp16→fp32 input upcast — same rationale as mm().
    # Use nbx_dtype for guard comparisons; .dtype returns triton.language.dtype.
    a_nbx = a.nbx_dtype
    b_nbx = b.nbx_dtype
    # NBX_FORCE_FP32_ACCUM diagnostic — see mm() comment.
    if NBX_FORCE_FP32_ACCUM and a_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32
    if NBX_FORCE_FP32_ACCUM and b_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        b = b.to(NBXDtype.float32)
        b_nbx = NBXDtype.float32
    if not _NBX_HAS_NATIVE_BF16 and a_nbx == NBXDtype.float16:
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32

    # Dtype alignment — see mm() for the full rationale. Fast path: when
    # activation is fp32 (after step 2) and weight is fp16 on pre-Ampere,
    # leave the weight fp16 and let the kernel promote its tile via
    # PROMOTE_B. All other mismatches fall back to the widening path.
    # mv_wrapper (M≤4 decode path below) already upcasts both operands
    # internally via `.to(tl.float32)` in mv_kernel, so it is also safe
    # to pass mixed dtypes there.
    promote_b = (not _NBX_HAS_NATIVE_BF16
                 and a_nbx == NBXDtype.float32
                 and b_nbx == NBXDtype.float16)
    if a_nbx != b_nbx and not promote_b:
        _order = (NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.float16)
        widest = next(d for d in _order if d in (a_nbx, b_nbx))
        if a_nbx != widest:
            a = a.to(widest)
        if b_nbx != widest:
            b = b.to(widest)

    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, f"bmm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"

    if M <= 4:
        a = a.contiguous()
        c = NBXTensor.empty((B, M, N), device=a.device,
                            dtype=_matmul_out_dtype(a, M, force_fp32=True))
        for bi in range(B):
            bt = b[bi].t()
            a_bi = a[bi]
            c_bi = c[bi]
            for i in range(M):
                c_bi[i] = mv_wrapper(bt, a_bi[i])
        return c

    a = a.contiguous()
    b = b.contiguous()
    out_dtype = _matmul_out_dtype(a, M, force_fp32=True)
    c = NBXTensor.empty((B, M, N), device=a.device, dtype=out_dtype)
    ieee = (not _NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Match mm() — sync Triton driver to a's device immediately before launch.
    # NBXTensor.empty(...) above can cudaSetDevice without updating Triton's
    # driver active device, and dtype alignment paths may also have touched
    # it. Without this, batched kernels on non-zero devices hit
    # "Pointer argument (at 0) cannot be accessed from Triton".
    _set_device(a)
    for i in range(B):
        matmul_kernel[grid](
            a[i], b[i], c[i],
            M, N, K,
            a[i].stride(0), a[i].stride(1),
            b[i].stride(0), b[i].stride(1),
            c[i].stride(0), c[i].stride(1),
            IEEE_PRECISION=ieee,
            PROMOTE_B=promote_b,
        )
    return c


def matmul_wrapper(a, b):
    """General matmul dispatcher: routes to mm, bmm, or mv based on input dims.

    Matches torch.matmul semantics:
    - 2D × 2D → mm
    - 3D × 3D → bmm
    - 2D × 1D → mv
    - ND × 2D → reshape to batched mm
    """
    if a.ndim == 2 and b.ndim == 2:
        return mm(a, b)
    if a.ndim == 3 and b.ndim == 3:
        return bmm(a, b)
    if a.ndim == 2 and b.ndim == 1:
        return mv_wrapper(a, b)
    if a.ndim >= 3 and b.ndim == 2:
        # Batched: reshape a to (batch, M, K), mm each, reshape back
        orig_shape = a.shape
        M, K = orig_shape[-2], orig_shape[-1]
        batch = a.numel() // (M * K)
        a_3d = a.contiguous().view(batch, M, K)
        result = bmm(a_3d, b.unsqueeze(0).expand(batch, K, b.shape[1]))
        return result.view(*orig_shape[:-1], b.shape[1])
    if a.ndim >= 3 and b.ndim >= 3:
        # General batched matmul. bmm is strictly 3D, so collapse the leading
        # batch dims into one, bmm, then restore the batch shape. Passing raw 4-D
        # tensors straight to bmm unpacked a 3-tuple → "too many values".
        a_c = a.contiguous(); b_c = b.contiguous()
        M, K = a_c.shape[-2], a_c.shape[-1]
        Kb, N = b_c.shape[-2], b_c.shape[-1]
        batch_a = a_c.numel() // (M * K)
        batch_b = b_c.numel() // (Kb * N)
        a_3d = a_c.view(batch_a, M, K)
        b_3d = b_c.view(batch_b, Kb, N)
        lead = list(a_c.shape[:-2])
        if batch_b == 1 and batch_a != 1:
            b_3d = b_3d.expand(batch_a, Kb, N)
        elif batch_a == 1 and batch_b != 1:
            a_3d = a_3d.expand(batch_b, M, K)
            lead = list(b_c.shape[:-2])
        result = bmm(a_3d, b_3d)
        return result.view(*lead, M, N)
    raise RuntimeError(f"matmul: unsupported shapes {a.shape} × {b.shape}")


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
    elements_np = _nbx_to_numpy(elements.contiguous())
    test_np = _nbx_to_numpy(test_elements.contiguous())
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


def _nbx_to_numpy(t):
    """Generic NBXTensor → numpy (D2H). Zero torch."""
    import numpy as np
    nb_dtype_to_np = {
        NBXDtype.float32: np.float32,
        NBXDtype.float16: np.float16,
        NBXDtype.int32:   np.int32,
        NBXDtype.int64:   np.int64,
    }
    dt = t.dtype
    if dt == NBXDtype.bfloat16:
        t = t.to(NBXDtype.float32)
        dt = NBXDtype.float32
    np_dtype = nb_dtype_to_np.get(dt)
    if np_dtype is None:
        t = t.to(NBXDtype.float32)
        np_dtype = np.float32
    arr = np.empty(t.shape, dtype=np_dtype)
    import ctypes as _ct  # noqa: F401 — force DeviceAllocator path
    DeviceAllocator.memcpy(arr.ctypes.data, t.data_ptr(), arr.nbytes, kind=2)
    return arr


def is_nonzero_wrapper(x):
    """aten::is_nonzero — returns the scalar truthiness of a 1-element tensor.

    PyTorch's native impl returns a Python bool, but traced graphs feed
    the output back through tensor ops, so we return a 0-d NBXTensor
    (uint8) to match the graph's expected dtype surface.
    """
    import numpy as np
    val = x.item()
    return NBXTensor.from_numpy(np.array(bool(val), dtype=np.uint8))


def linear_wrapper(input, weight, bias=None):
    """aten::linear — y = input @ weight.T + bias.

    Signature matches torch.nn.functional.linear:
      input : (..., in_features)
      weight: (out_features, in_features)   — NOT pre-transposed
      bias  : (out_features,) | None
      output: (..., out_features)

    Routes through the existing mm / bmm / addmm primitives with an
    implicit transpose on `weight`. No new kernel launches beyond what
    these helpers already do.
    """
    w_t = weight.t()
    if bias is None:
        return matmul_wrapper(input, w_t)
    # matmul then broadcast-add — matches torch's handling for ND inputs.
    # For pure 2D we could use addmm directly, but matmul_wrapper already
    # takes the mm path and `add` broadcasts cleanly.
    out = matmul_wrapper(input, w_t)
    return add(out, bias)


def addmm(bias, a, b,
          beta: float = 1.0, alpha: float = 1.0) :
    """C = beta * bias + alpha * (A @ B).
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally.

    Small-M fast path (M <= 4): per-row addmv. Bias is broadcast the same
    way torch.addmm does: (N,), (1, N), or (M, N).
    """
    a = _ensure_cuda(a)
    b = _ensure_cuda(b)
    bias = _ensure_cuda(bias)

    # Hardware-gated fp16→fp32 input upcast — same rationale as mm().
    # Use nbx_dtype for guard comparisons; .dtype returns triton.language.dtype.
    a_nbx = a.nbx_dtype
    b_nbx = b.nbx_dtype
    # NBX_FORCE_FP32_ACCUM diagnostic — see mm() comment.
    if NBX_FORCE_FP32_ACCUM and a_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32
    if NBX_FORCE_FP32_ACCUM and b_nbx in (NBXDtype.float16, NBXDtype.bfloat16):
        b = b.to(NBXDtype.float32)
        b_nbx = NBXDtype.float32
    if not _NBX_HAS_NATIVE_BF16 and a_nbx == NBXDtype.float16:
        a = a.to(NBXDtype.float32)
        a_nbx = NBXDtype.float32

    # Dtype alignment (see mm() for rationale). Same two branches:
    # promote_b keeps fp16 weight fp16 + kernel casts tile inline;
    # otherwise fall back to full widening.
    promote_b = (not _NBX_HAS_NATIVE_BF16
                 and a_nbx == NBXDtype.float32
                 and b_nbx == NBXDtype.float16)
    if a_nbx != b_nbx and not promote_b:
        _order = (NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.float16)
        widest = next(d for d in _order if d in (a_nbx, b_nbx))
        if a_nbx != widest:
            a = a.to(widest)
        if b_nbx != widest:
            b = b.to(widest)
    # Bias tracks the accumulator dtype — it is added inside the kernel
    # after tl.dot, so bias must match the activation dtype, not the
    # (possibly-fp16) weight.
    if bias.nbx_dtype != a.nbx_dtype:
        bias = bias.to(a.nbx_dtype)

    # N-D activation: addmm is strictly 2-D. Flatten the leading dims of `a`
    # ([..., M, K] @ [K, N]), addmm in 2-D, restore the leading shape. Mirror
    # of the matmul() ND×2D path; raw N-D `a` unpacked a 2-tuple → "too many
    # values to unpack" (Flex DiT feeds a 3-D activation to addmm). Bias is the
    # Linear bias ([N] or [1, N]) and broadcasts over the flattened rows.
    if a.ndim > 2:
        orig_lead = a.shape[:-1]
        K_ = a.shape[-1]
        batch = a.numel() // K_
        a2d = a.contiguous().view(batch, K_)
        out2d = addmm(bias, a2d, b, beta=beta, alpha=alpha)
        return out2d.view(*orig_lead, out2d.shape[-1])

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    if M <= 4:
        a = a.contiguous()
        bias = bias.contiguous()
        bt = b.t()
        c = NBXTensor.empty((M, N), device=a.device,
                            dtype=_matmul_out_dtype(a, M))
        bias_per_row = (bias.ndim == 2 and bias.shape[0] == M)
        if bias.ndim == 2 and bias.shape[0] == 1:
            bias_shared = bias[0]
        elif bias.ndim == 1:
            bias_shared = bias
        else:
            bias_shared = None
        for i in range(M):
            br = bias[i] if bias_per_row else bias_shared
            c[i] = addmv_wrapper(br, bt, a[i], beta=beta, alpha=alpha)
        return c

    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out_dtype = _matmul_out_dtype(a, M)
    c = NBXTensor.empty((M, N), device=a.device, dtype=out_dtype)
    ieee = (not _NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    addmm_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        alpha, beta,
        IEEE_PRECISION=ieee,
        PROMOTE_B=promote_b,
    )
    return c


# ===========================================================================
# FUSED MOE WRAPPER
# ===========================================================================

# Block sizes for fused MoE kernel (tuned for decode batch=1)
_MOE_BM = 16
_MOE_BN = 64
_MOE_BK = 32
_MOE_GROUP = 8


def invoke_fused_moe(
    hidden_states,            # NBXTensor [M, K] — activations
    expert_ptrs,              # NBXTensor [E] int64 — absolute expert weight pointers
    output,                   # NBXTensor [M * top_k, N] — pre-allocated output
    topk_weights,             # NBXTensor [M * top_k] — flat routing scores
    sorted_token_ids,         # NBXTensor [num_tokens_post_padded] — sorted indices
    expert_ids,               # NBXTensor [num_blocks] — expert per BLOCK_M group
    num_tokens_post_padded,   # NBXTensor [1] — total sorted entries
    N, K,                     # output dim, reduction dim
    stride_bk, stride_bn,    # weight strides (shared by all experts)
    top_k,                    # int
    mul_routed_weight=False,
    topk_divide=True,
):
    """Launch fused MoE grouped GEMM kernel (absolute pointer table).

    One kernel launch handles ALL experts. Each expert's weight pointer is
    stored as an absolute int64 in the table — no offset arithmetic.
    """
    M = hidden_states.shape[0]
    num_valid_tokens = M * top_k
    EM = sorted_token_ids.shape[0]

    if M < _MOE_BM:
        EM = min(EM, M * top_k * _MOE_BM)

    grid = (triton.cdiv(EM, _MOE_BM) * triton.cdiv(N, _MOE_BN),)
    _set_device(hidden_states)

    fused_moe_kernel[grid](
        hidden_states, expert_ptrs, output,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        hidden_states.stride(0), hidden_states.stride(1),
        stride_bk, stride_bn,
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=_MOE_BM, BLOCK_SIZE_N=_MOE_BN, BLOCK_SIZE_K=_MOE_BK,
        GROUP_SIZE_M=_MOE_GROUP,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=hidden_states.dtype,
        TOPK_DIVIDE=topk_divide,
        num_warps=4, num_stages=2,
    )


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

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    orig_shape = gate_c.shape
    N = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    gate2d = gate_c.view(M, N)
    up2d = up_c.view(M, N)
    output = NBXTensor.empty((M, N), dtype=gate_c._dtype,
                             device=f"cuda:{gate_c._device_idx}")

    BLOCK_M = 16
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _set_device(gate_c)

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

    # Align cos/sin dtype with q/k (kernel computes in cos/sin dtype).
    target_dt = q_raw.dtype
    if cos.dtype != target_dt:
        cos = cos.to(target_dt)
    if sin.dtype != target_dt:
        sin = sin.to(target_dt)
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

def bf16_to_fp16_gpu(src: NBXTensor) -> NBXTensor:
    """Convert bf16 (stored as uint16) → fp16 on GPU. Zero CPU work.

    Args:
        src: NBXTensor with dtype uint16 containing raw bf16 bits

    Returns:
        NBXTensor with dtype float16, same shape
    """
    N = src.numel()
    dst = NBXTensor.empty(src._shape, NBXDtype.float16,
                          device=f"cuda:{src._device_idx}")
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _set_device(src)
    bf16_to_fp16_kernel[grid](src, dst, N, BLOCK=BLOCK, num_warps=4)
    return dst


# ===========================================================================
# EMBEDDING WRAPPER
# ===========================================================================

def embedding(weight, indices, padding_idx=-1, **kwargs) :
    """Embedding lookup: output[i] = weight[indices[i]].

    Indices must be integer. Diffusion timestep embeddings sometimes arrive
    as fp32 (timestep is a scalar float in the scheduler); cast to int64 so
    the Triton kernel's pointer arithmetic (`weight_ptr += row_idx * N`)
    sees an integer operand.
    """
    if indices.dtype not in (NBXDtype.int64, NBXDtype.int32, NBXDtype.int16,
                             NBXDtype.int8, NBXDtype.uint8, NBXDtype.bool_):
        indices = indices.to(NBXDtype.int64)
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
        DeviceAllocator.memcpy(new_indices.data_ptr(), indices.data_ptr(), indices._nbytes)
        indices = new_indices
    output = NBXTensor.empty((*indices.shape, N), dtype=weight.nbx_dtype if hasattr(weight, 'nbx_dtype') else weight.dtype, device=dev)
    _set_device(weight)
    embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)
    return output


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


def mean_wrapper(x, dim=None, keepdim=False) :
    """Mean reduction. Reduces over one dim, a 1-elem list, or (via
    _reduce_over_dims) an arbitrary list of dims."""
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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    _set_device(x_2d)
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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    _set_device(x_2d)
    sum_forward_kernel[grid](
        x_2d, output,
        batch_dim, feat_dim,
        x_2d.stride(0), x_2d.stride(1),
        num_warps=4,
    )

    if dim is None:
        return output.squeeze(0).to(out_dtype)

    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return output.to(out_dtype).view(shape)


def norm_wrapper(x, p=2, dim=None, keepdim=False):
    """aten::norm(self, p, dim, keepdim) — Lp norm reduction over ``dim``.

    The DAC codec's only norm is weight-norm: ``||v|| = sqrt(sum(v*v, dim))``
    with p=2 over a single dim. Composed from existing triton kernels
    (mul + sum + sqrt) — R33-pure, no new @triton.jit needed; sum_wrapper
    accumulates in fp32 so the reduction over a long row does not overflow fp16.
    Other p raise ZERO-FALLBACK (named follow-up) rather than mis-computing.
    """
    pv = _to_scalar(p) if isinstance(p, NBXTensor) else (2 if p is None else p)
    if float(pv) != 2.0:
        raise NotImplementedError(
            f"aten::norm p={pv} unwired (DAC weight-norm uses p=2) — "
            "follow-up P-TRITON-NORM-GENERAL-P.")
    return sqrt_wrapper(sum_wrapper(mul(x, x), dim=dim, keepdim=keepdim))


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
    _bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, _bsb),)
    _set_device(x_2d)
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
    _set_device(x1)
    glu_forward_kernel[_1d_grid(x1.numel())](
        x1, x2, output, x1.numel(), act_func,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ===========================================================================
# PHASE 2 WRAPPERS — Extracted from FlagGems reference
# ===========================================================================

def le(a, b) :
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        le_scalar_kernel[_1d_grid(n)](a, output, n, float(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        le_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def softplus_wrapper(x, beta: float = 1.0, threshold: float = 20.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
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
        _set_device(x)
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
        out_size_2d = [1, output_size[0]] if len(output_size) else [1, None]
    else:
        out_size_2d = [1, output_size]
    # aten::upsample_nearest1d.vec passes scale_factors as a 1-element LIST
    # ([scale]); the 2D wrapper's scalar scales_w path does `float(scales_w)`,
    # which raises on a list. Unwrap to a scalar. When a scale is present the 2D
    # wrapper recomputes OW = IW * scale from the LIVE input — exactly what a
    # variable-length audio vocoder (HiFiGAN/iSTFTNet/DAC/S3Gen) needs so the
    # output tracks the runtime token count rather than the baked trace length.
    if isinstance(scales, (list, tuple)):
        scale_w = scales[0] if len(scales) else None
    else:
        scale_w = scales
    out_2d = upsample_nearest2d_wrapper(x_2d, out_size_2d, scales_h=None, scales_w=scale_w)
    return out_2d.squeeze(2)


def upsample_nearest2d_wrapper(
    x, output_size, scales_h=None, scales_w=None
) :
    """Upsample nearest 2D. Wrapper from FlagGems.

    PyTorch's traced graph stores both `output_size` (concrete trace
    value) and `scales_h/w` (the upsampling ratio). When the runtime
    input size differs from the trace size — Sana 4Kpx: trace 64×64
    latent → runtime 128×128, and every cascading decoder upsample
    inherits the doubling — `output_size` no longer matches
    `IH * scales_h`. Native PyTorch recomputes from the live input
    when scale factors are present; the wrapper must do the same so
    the cascade scales correctly. When `scales_h/w` is None we keep
    the literal `output_size` (path used by audio paths that don't
    rebind spatial dims).
    """
    assert x.ndim == 4
    N, C, IH, IW = x.shape
    if isinstance(output_size, (list, tuple)):
        OH, OW = output_size[0], output_size[1]
    else:
        OH, OW = output_size, output_size

    # Recompute output size from runtime input when scale factors are
    # given. Bit-identical when trace == runtime; corrects the cascade
    # when they differ (Sana 4Kpx and any other graph traced at a
    # smaller spatial size than runtime).
    if scales_h is not None:
        OH = int(IH * float(scales_h))
    if scales_w is not None:
        OW = int(IW * float(scales_w))

    reciprocal_scale_h = (1.0 / scales_h) if scales_h is not None else (IH / OH)
    reciprocal_scale_w = (1.0 / scales_w) if scales_w is not None else (IW / OW)

    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total_threads = OH * OW
    grid = (
        triton.cdiv(total_threads, _EW_BLOCK),
        triton.cdiv(N * C, 4),
    )
    _set_device(output)
    upsample_nearest2d_kernel[grid](
        output, x, N, C, OH, OW, IH, IW,
        reciprocal_scale_h, reciprocal_scale_w,
        BLOCK_SIZE=_EW_BLOCK,
    )
    return output


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
    # BLOCK_SIZE: chunk along hidden_size = group_size*HxW. Cap at 16384
    # (= 16K elements per chunk → 64 KB at fp32, 32 KB at fp16) so the
    # per-program tile stays under SMEM regardless of dtype, and well
    # under Triton's 2^20 numel ceiling. The kernel loops over chunks
    # so any hidden_size works (PixArt VAE 1024×1024: hidden up to ~4M
    # spread across many chunks).
    hidden = group_size * HxW
    block_size = min(16384, triton.next_power_of_2(min(hidden, 16384)))
    group_norm_forward_kernel[grid](
        x, y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        mean, rstd,
        group_size, C, HxW, num_groups, eps,
        scale_by_weight=(weight is not None),
        add_bias=(bias is not None),
        BLOCK_SIZE=block_size,
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


def index_select_wrapper(x, dim: int, index) :
    """Index select along dimension. Wrapper from FlagGems."""
    # aten::index / aten::index_select REQUIRE integer indices. The triton path
    # can present an integer-VALUED index tagged float32 (e.g. the whisper
    # decoder position_ids reaching aten::index via _meta_index). The kernel
    # computes a pointer offset `inp + (rows*N + indices)`; a float `indices`
    # makes that pointer arithmetic illegal (Triton compile error
    # "pointer<fp16> and float32"). Enforce the integer contract at this single
    # choke point both modes funnel through — symmetric across triton /
    # triton_sequential (R30), and matches torch (index_select needs a Long).
    if hasattr(index, "is_floating_point") and index.is_floating_point():
        index = index.to(NBXDtype.int64)
    # Complex tensors: the kernel copies one real value per element, so it
    # mishandles the interleaved real/imag storage of complex64/128 (reads half
    # the bytes / wrong stride -> garbage, e.g. the Wan rotary-embedding freqs
    # constant indexed per grid axis). Route through the real view: view_as_real
    # appends the [.,2] real/imag axis AFTER all original dims, so the select
    # `dim` (resolved on the complex rank) indexes the same axis untouched.
    if x.is_complex():
        ndc = x.ndim
        dd = dim % ndc
        outr = index_select_wrapper(x.view_as_real(), dd, index)
        return outr.view_as_complex()
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
    _set_device(x)
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
        _set_device(x)
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
        _set_device(B)
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
        _set_device(x)
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
        _set_device(B)
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

        _set_device(x.contiguous())
        argmax_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
        _set_device(mid_value)
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
        _set_device(x)
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

        _set_device(x.contiguous())
        argmin_kernel_1[(mid_size, 1, 1)](x.contiguous(), mid_value, mid_index, M, block_size)
        _set_device(mid_value)
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
        _set_device(x)
        argmin_kernel[grid](x, out_index, M, N, K, BLOCK_M, BLOCK_N)
        return out_index


# ===========================================================================
# CONV2D WRAPPER — Extracted from FlagGems
# ===========================================================================

# Spatial band-streaming threshold for conv2d. When the per-launch output
# tensor would exceed this many bytes, conv2d_wrapper transparently splits
# along the H dimension and streams band-by-band. This is the kernel-level
# tiling lever (P-SANA-4KPX-RUNTIME, Étape 1 — internal to the wrapper, not
# a Prism op-level interceptor). 4 GiB default leaves headroom for weights,
# activations and arena overhead on V100 32 GB. Override via env var
# NBX_CONV2D_BAND_BYTES if needed for non-Volta hardware.
_NBX_CONV2D_BAND_BYTES = int(os.environ.get("NBX_CONV2D_BAND_BYTES", str(4 * 1024 * 1024 * 1024)))

# P-TRITON-LIVE-SET-AUDIT (#37) — conv3d-via-conv2d eager-free threshold.
# The temporal decomposition in _conv3d_via_conv2d materialises several full-
# resolution copies per temporal-kernel slice (input contiguous-copy, conv2d
# output, permuted output, accumulator). For large video VAE conv3d (CogVideoX
# 5D decode at (1,256,13,480,720) = 4.4 GB per copy in fp32), 5-6 of these are
# live simultaneously → 26-30 GB → single-GPU OOM at aten.convolution. When a
# per-slice intermediate is at least this many bytes, it is dropped the moment
# it is consumed so the live set falls to input + accumulator + one transient.
# free_cuda is cudaFree, which blocks until the device is idle for the relevant
# stream, so dropping a tensor whose producing/consuming kernel just ran is
# async-safe (no UAF). GATED so small convs (Wan/Mochi/SANA at their validated
# configs) keep the EXACT prior code path — no relocated free, byte- and
# timing-identical. Numerics are unchanged (frees sooner only). 1 GiB default
# fires only on the multi-GB video-VAE convs.
_NBX_CONV3D_EAGER_FREE_BYTES = int(
    os.environ.get("NBX_CONV3D_EAGER_FREE_BYTES", str(1 * 1024 * 1024 * 1024)))

# Conv3d temporal chunk-streaming — kernel-level tiling INSIDE the wrapper,
# same lever as the conv2d band-streaming above (P-SANA-4KPX Étape 1 class).
# Even with the #37 eager frees, the one-shot temporal decomposition needs
# several FULL folded transients simultaneously (temporal pad copy, x2 fold,
# conv2d output incl. band-streaming machinery, permuted copy, accumulator +
# accumulate transient) — ~4-5x the activation bytes. cuDNN in compiled mode
# needs input + output + a bounded workspace for the same op, so a component
# Prism correctly sized as "fits" for compiled can OOM in triton only
# (Allegro-TI2V vae_encoder at 720x1280x12f fp32: 5.66 GB per plane →
# 30.4 GB live at aten.convolution::1 on a 32 GB V100). Gate, two stages:
#   1. deterministic floor — the folded transient must exceed this many
#      bytes, so small/validated convs never even query the driver;
#   2. exact-fit check — the one-shot path's real allocation peak is
#      compared to driver-free bytes; when it fits, the EXACT one-shot
#      path runs (byte-identical behavior for every config that is green
#      today, on any hardware — R23), when it cannot fit (guaranteed OOM)
#      the wrapper streams over temporal output chunks whose per-transient
#      size is bounded by this same value.
# Chunking splits the folded batch axis (B*T frames): spatial conv2d is
# independent per frame and the per-frame kt accumulation order is
# preserved, so the math is unchanged.
_NBX_CONV3D_CHUNK_BYTES = int(
    os.environ.get("NBX_CONV3D_CHUNK_BYTES", str(1 * 1024 * 1024 * 1024)))

# Diagnostic trace — when set, every conv2d_wrapper call prints the
# (in_shape, out_shape, kernel, groups, output_MB) so we can see the actual
# spatial workload arriving in triton mode. Used by P-SANA-4KPX-RUNTIME
# Étape 1 verification on Sana 4Kpx (was the wrapper-internal band-streaming
# triggered? what shapes are dominant?).
_NBX_CONV2D_TRACE = os.environ.get("NBX_CONV2D_TRACE", "0") == "1"


def _conv2d_should_band_stream(N, out_c, out_h, out_w, dtype_bytes):
    """Return True when the conv2d output alone would exceed the spatial
    band-streaming threshold. Output is the dominant transient because the
    @triton.jit kernel accumulates in fp32 internally but writes the final
    output at compute_dtype. Weights are residence-cost (already in arena),
    not per-launch transients."""
    out_bytes = N * out_c * out_h * out_w * dtype_bytes
    return out_bytes > _NBX_CONV2D_BAND_BYTES


def conv_transpose_wrapper(
    x, weight, bias=None,
    stride=1, padding=0, dilation=1, output_padding=0, groups=1,
):
    """Transposed convolution — aten::convolution(transposed=True).

    Handles 1D (3D tensors, via H=1 unsqueeze) and 2D (4D tensors). Weight layout
    matches PyTorch ConvTranspose: (C_in, C_out/groups, *K). Scatter-based
    @triton.jit kernel (kernels/ops/conv_transpose2d.py), fp32 accumulation.
    R33-pure: NBXTensor + Triton only.

    A regular strided conv DOWNSAMPLES (out = in // stride); a transposed conv
    UPSAMPLES: out = (in-1)*stride - 2*pad + dil*(k-1) + output_padding + 1. The
    Kokoro F0/N ProsodyPredictor pool is a depthwise ConvTranspose1d (stride 2)
    that doubles the frame axis (62->124). conv2d_wrapper routes here when
    transposed=True (which it previously dropped when delegating 1D convs).
    """
    is_1d = (weight.ndim == 3)
    if is_1d:
        x = x.unsqueeze(2)            # [N, C_in, 1, L]
        weight = weight.unsqueeze(2)  # [C_in, C_out/g, 1, K]
        def _s(v):
            return v[0] if isinstance(v, (list, tuple)) else v
        sh, sw = 1, _s(stride)
        ph, pw = 0, _s(padding)
        dh, dw = 1, _s(dilation)
        oph, opw = 0, _s(output_padding)
    else:
        def _hw(v):
            if isinstance(v, (list, tuple)):
                return (v[0], v[1] if len(v) > 1 else v[0])
            return (v, v)
        sh, sw = _hw(stride); ph, pw = _hw(padding)
        dh, dw = _hw(dilation); oph, opw = _hw(output_padding)

    # dtype alignment NARROWING (mirror conv2d_wrapper; fp32 accum in-kernel)
    x_nbx = x.nbx_dtype if hasattr(x, 'nbx_dtype') else x.dtype
    w_nbx = weight.nbx_dtype if hasattr(weight, 'nbx_dtype') else weight.dtype
    if x_nbx != w_nbx:
        _order = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32)
        narrowest = next(d for d in _order if d in (x_nbx, w_nbx))
        if x_nbx != narrowest:
            x = x.to(narrowest)
        if w_nbx != narrowest:
            weight = weight.to(narrowest)

    x_c = x.contiguous()
    w_c = weight.contiguous()
    out_dtype = _NBX_COMPUTE_DTYPE if _NBX_COMPUTE_DTYPE is not None else x.dtype

    N, C_in, IH, IW = x_c.shape
    _, C_out_per_g, KH, KW = w_c.shape
    C_out = C_out_per_g * groups
    C_in_per_g = C_in // groups
    OH = (IH - 1) * sh - 2 * ph + dh * (KH - 1) + oph + 1
    OW = (IW - 1) * sw - 2 * pw + dw * (KW - 1) + opw + 1

    output = NBXTensor.empty((N, C_out, OH, OW), device=x_c.device, dtype=out_dtype)
    _set_device(x_c)
    BLOCK = 256
    grid = (N * C_out, triton.cdiv(OH * OW, BLOCK))
    conv_transpose2d_kernel[grid](
        x_c, w_c, output,
        N, C_in, IH, IW, C_out, KH, KW, OH, OW,
        sh, sw, ph, pw, dh, dw, C_in_per_g, C_out_per_g,
        BLOCK_SIZE=BLOCK,
    )
    if bias is not None:
        output = add(output, bias.view(1, C_out, 1, 1))
    if is_1d:
        output = output.squeeze(2)
    return output


def _conv3d_via_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """conv3d via temporal decomposition: conv3d(x, w) = sum over the temporal
    kernel positions kt of conv2d on the kt-th temporally-shifted frames. The
    spatial (H, W) convolution is delegated to conv2d_wrapper (this same triton
    kernel) per temporal slice; the temporal axis is handled by slicing +
    accumulation. Validated bit-close to torch F.conv3d (max|diff| ~3e-5) across
    kt=3 / patch-embed kt=1 / strided / temporal-only configs. R33-pure
    (NBXTensor slice/permute/reshape/add + conv2d_wrapper, no torch).
    """
    def _triple(v):
        return (v[0], v[1], v[2]) if isinstance(v, (list, tuple)) else (v, v, v)
    st, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)
    B, Cin, T, H, W = x.shape
    Cout, Cin_g, kt, kh, kw = weight.shape

    # Temporal chunk-streaming gate (see _NBX_CONV3D_CHUNK_BYTES). Evaluated
    # BEFORE the temporal pad so the pad copy counts toward the one-shot
    # path's allocation peak. Fires ONLY when the folded transient exceeds
    # the deterministic floor AND the one-shot peak cannot fit the driver-
    # free bytes — every run that fits keeps the exact path below.
    # NOTE: x.device returns the tensor itself (NBX device-context pattern);
    # the raw device string lives in _device.
    if str(getattr(x, '_device', '')).startswith('cuda'):
        T_out_est = (T + 2 * pt - dt * (kt - 1) - 1) // st + 1
        if T_out_est > 0:
            ds_in = dtype_size(x.nbx_dtype if hasattr(x, 'nbx_dtype') else x._dtype)
            ds_out = (dtype_size(_NBX_COMPUTE_DTYPE)
                      if _NBX_COMPUTE_DTYPE is not None else ds_in)
            oh_est = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            ow_est = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
            fold_in = B * T_out_est * Cin * H * W * ds_in
            fold_out = B * T_out_est * Cout * oh_est * ow_est * ds_out
            if max(fold_in, fold_out) > _NBX_CONV3D_CHUNK_BYTES:
                # One-shot allocation peak (mirrors the loop below with the
                # #37 eager frees active — all transients are >= 1 GiB here):
                #  - during the folded conv2d at kti>=1:
                #      pad + accumulator + x2 + y2 (+ band machinery)
                #  - during the accumulate add at kti>=1:
                #      pad + out_old + y + out_new
                pad_bytes = (B * Cin * (T + 2 * pt) * H * W * ds_in
                             if pt > 0 else 0)
                band_extra = (_NBX_CONV2D_BAND_BYTES
                              if fold_out > _NBX_CONV2D_BAND_BYTES else 0)
                acc_extra = fold_out if kt >= 2 else 0
                need = pad_bytes + max(
                    fold_out + fold_in + band_extra + acc_extra,
                    2 * fold_out + acc_extra,
                ) + 256 * 1024 * 1024
                free = DeviceAllocator.device_free_bytes(
                    getattr(x, '_device_idx', None))
                if free >= 0 and need > free:
                    if _NBX_CONV2D_TRACE:
                        print(f"[CONV3D] CHUNK-STREAM in=({B},{Cin},{T},{H},{W}) "
                              f"kt={kt} need={need/1e9:.2f}GB free={free/1e9:.2f}GB",
                              flush=True)
                    return _conv3d_via_conv2d_chunked(
                        x, weight, bias, st, sh, sw, pt, ph, pw, dt, dh, dw,
                        groups,
                        max(fold_in, fold_out) // max(1, T_out_est))

    # Pad the temporal axis only (constant_pad_nd pads from the last dim:
    # [W_l,W_r, H_l,H_r, T_l,T_r]); H/W padding is applied by conv2d.
    if pt > 0:
        x = constant_pad_nd_wrapper(x, [0, 0, 0, 0, pt, pt], 0.0)
    Tp = x.shape[2]
    T_out = (Tp - dt * (kt - 1) - 1) // st + 1
    out = None
    _ef = _NBX_CONV3D_EAGER_FREE_BYTES  # #37 size-gated eager-free threshold
    for kti in range(kt):
        start = kti * dt
        if st == 1:
            xs = x[:, :, start:start + T_out]                  # [B,Cin,T_out,H,W]
        else:
            import numpy as np
            idx = NBXTensor.from_numpy(
                (start + st * np.arange(T_out)).astype(np.int64))
            xs = index_select_wrapper(x, 2, idx)
        x2 = xs.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T_out, Cin, H, W)
        # .contiguous() before reshape: weight[:, :, kti:kti+1] is a non-contiguous
        # dim-2 slice; a flat-indexed reshape would read wrong memory for kti>0
        # (contiguous-guard pattern).
        w2 = weight[:, :, kti:kti + 1].contiguous().reshape(Cout, Cin_g, kh, kw)
        y2 = conv2d_wrapper(x2, w2, None, (sh, sw), (ph, pw), (dh, dw),
                            False, 0, groups)
        # #37: conv2d has produced y2, so the input contiguous-copy x2 is dead.
        # Drop it now for large (video-VAE) slices so the live set does not
        # accumulate input + x2 + y2 + y + out simultaneously (CogVideoX 5D
        # decode OOM at aten.convolution). Gated → small convs keep the exact
        # prior path. cudaFree blocks until the conv2d kernel is idle (async-safe).
        if x2._nbytes >= _ef:
            x2 = None
        Ho, Wo = y2.shape[2], y2.shape[3]
        y = y2.reshape(B, T_out, Cout, Ho, Wo).permute(0, 2, 1, 3, 4).contiguous()
        if y2._nbytes >= _ef:
            y2 = None  # conv output pre-permute copy is dead once y is materialised
        if out is None:
            out = y
        else:
            out = out + y  # old accumulator auto-frees on reassignment
            if y._nbytes >= _ef:
                y = None   # addend is dead once out+y is allocated
    if bias is not None:
        out = out + bias.reshape(1, Cout, 1, 1, 1)
    return out


def _conv3d_via_conv2d_chunked(x, weight, bias, st, sh, sw, pt, ph, pw,
                               dt, dh, dw, groups, frame_bytes):
    """Temporal chunk-streaming variant of _conv3d_via_conv2d — identical
    math, bounded live set (see _NBX_CONV3D_CHUNK_BYTES for the gate).

    Streams the folded batch axis (B*T_out frames) in chunks sized so each
    transient stays under _NBX_CONV3D_CHUNK_BYTES. Per output frame the
    spatial conv2d is independent and the kt accumulation runs in the same
    kti order with the same operand values as the one-shot path, so the
    result is numerically equivalent (launch shapes differ, which may pick
    different autotune configs — same class of variation as the conv2d
    band-streaming). Live set: temporal pad copy + final 5D output +
    ~3 chunk-sized transients, vs ~4-5 full folds for the one-shot path.
    R33-pure (NBXTensor slice/permute/reshape/add + conv2d_wrapper +
    strided-scatter __setitem__, no torch).

    `frame_bytes` = max folded bytes per output frame (input-side fold vs
    conv2d output), computed by the caller's gate.
    """
    B, Cin, T, H, W = x.shape
    Cout, Cin_g, kt, kh, kw = weight.shape
    if pt > 0:
        x = constant_pad_nd_wrapper(x, [0, 0, 0, 0, pt, pt], 0.0)
    Tp = x.shape[2]
    T_out = (Tp - dt * (kt - 1) - 1) // st + 1
    tc = max(1, int(_NBX_CONV3D_CHUNK_BYTES // max(1, frame_bytes)))
    # Per-kt 2D weight slices, materialised once (tiny; contiguous-guard on
    # the non-contiguous dim-2 slice, same as the one-shot path).
    w2s = [weight[:, :, kti:kti + 1].contiguous().reshape(Cout, Cin_g, kh, kw)
           for kti in range(kt)]
    bias4 = bias.reshape(1, Cout, 1, 1) if bias is not None else None
    out = None
    for t0 in range(0, T_out, tc):
        n = min(tc, T_out - t0)
        acc = None
        for kti in range(kt):
            start = t0 * st + kti * dt
            if st == 1:
                xs = x[:, :, start:start + n]              # [B,Cin,n,H,W] view
            else:
                import numpy as np
                idx = NBXTensor.from_numpy(
                    (start + st * np.arange(n)).astype(np.int64))
                xs = index_select_wrapper(x, 2, idx)
            x2 = xs.permute(0, 2, 1, 3, 4).contiguous().reshape(
                B * n, Cin, H, W)
            y2 = conv2d_wrapper(x2, w2s[kti], None, (sh, sw), (ph, pw),
                                (dh, dw), False, 0, groups)
            x2 = None  # fold copy dead once conv2d returned
            acc = y2 if acc is None else acc + y2
            y2 = None
        if bias4 is not None:
            # Elementwise-identical to the one-shot path's 5D bias add
            # (broadcast over the folded batch axis), chunk-sized transient.
            acc = acc + bias4
        Ho, Wo = acc.shape[2], acc.shape[3]
        y = acc.reshape(B, n, Cout, Ho, Wo).permute(0, 2, 1, 3, 4).contiguous()
        acc = None
        if out is None:
            out_dt = y.nbx_dtype if hasattr(y, 'nbx_dtype') else y.dtype
            out = NBXTensor.empty((B, Cout, T_out, Ho, Wo),
                                  device=y.device, dtype=out_dt)
        out[:, :, t0:t0 + n] = y  # strided-scatter into the temporal slice
        y = None
    return out


def conv2d_wrapper(
    x, weight, bias=None,
    stride=1, padding=0, dilation=1,
    transposed=False, output_padding=0, groups=1,
) :
    """Convolution forward. Handles both 1D (3D tensors) and 2D (4D tensors).
    Routes to conv1d_wrapper for 3D inputs, and to conv_transpose_wrapper when
    transposed=True (1D or 2D).

    Spatial band-streaming (P-SANA-4KPX-RUNTIME Étape 1): when the output
    tensor would exceed _NBX_CONV2D_BAND_BYTES (default 4 GiB), the wrapper
    splits along the H output dimension and streams band-by-band. Each band
    re-enters this same wrapper with a smaller H, so the recursion bottoms
    out automatically. The full output stays allocated for downstream
    consumers."""
    # Transposed convolutions (1D or 2D) — route BEFORE the 1D delegation, which
    # historically dropped the transposed/output_padding flags (Kokoro F0/N
    # ConvTranspose1d was computed as a regular strided conv → halved instead of
    # doubled the frame axis).
    if transposed:
        return conv_transpose_wrapper(x, weight, bias, stride, padding, dilation,
                                      output_padding, groups)

    # Route 1D convolutions to conv1d_wrapper
    if weight.ndim == 3:
        return conv1d_wrapper(x, weight, bias, stride, padding, dilation, groups)

    # 5D (video) conv3d -> temporal decomposition into conv2d (R33-pure: reuses
    # this same conv2d kernel per temporal-kernel slice). Wan video models use
    # Conv3D (transformer patch_embed + VAE temporal convs).
    if weight.ndim == 5:
        return _conv3d_via_conv2d(x, weight, bias, stride, padding, dilation, groups)

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

    # Self-managed dtype doctrine — VRAM-preserving (different from the
    # mm/bmm/addmm doctrine which is accumulation-overflow-protected).
    #
    # mm/bmm/addmm: deep matmul stacks risk fp16 accumulation overflow.
    #   Step 1 upcasts input fp16 → fp32 on pre-Ampere, output forced to
    #   fp32 (force_fp32 in _matmul_out_dtype). Cible précision.
    #
    # conv2d: spatial activation chain (VAE/UNet). The kernel already
    #   accumulates in fp32 internally (conv2d.py line 47:
    #   `accum = tl.zeros(..., dtype=tl.float32)`), so there is no
    #   overflow risk requiring an input fp16→fp32 upcast. The risk here
    #   is VRAM (4096×4096 spatial tensors). We therefore:
    #     - skip Step 1 (no input upcast)
    #     - Step 2 narrows mismatched input/weight to the lowest-precision
    #       common dtype (opposite of mm widening — VRAM over precision)
    #     - Step 4 sets output dtype = compute_dtype from
    #       _NBX_COMPUTE_DTYPE (the per-component Prism dtype set by
    #       TritonSequence.run), mirroring what cuDNN does silently in
    #       compiled mode. Falls back to x.dtype if not set (run outside
    #       a TritonSequence context).
    # Universal hardware: no _NBX_HAS_NATIVE_BF16 gate — same behavior on
    # Volta and Ampere+ since it operates on dtype tags, not hardware
    # capability. R33-pure: NBXTensor.to uses a @triton.jit copy kernel.

    # Step 0: ensure cuda + cross-device alignment (identical to mm)
    # (already implicit; weight transfer handled by upstream callers / arena)

    # Step 2: dtype alignment NARROWING (opposite of mm widening)
    x_nbx = x.nbx_dtype if hasattr(x, 'nbx_dtype') else x.dtype
    w_nbx = weight.nbx_dtype if hasattr(weight, 'nbx_dtype') else weight.dtype
    if x_nbx != w_nbx:
        _order = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32)
        narrowest = next(d for d in _order if d in (x_nbx, w_nbx))
        if x_nbx != narrowest:
            x = x.to(narrowest)
        if w_nbx != narrowest:
            weight = weight.to(narrowest)

    x_c = x.contiguous()
    w_c = weight.contiguous()

    # Step 4: output dtype = compute_dtype from per-component Prism context.
    # Falls back to x.dtype when no TritonSequence is active.
    out_dtype = _NBX_COMPUTE_DTYPE if _NBX_COMPUTE_DTYPE is not None else x.dtype

    # P-SANA-4KPX-RUNTIME Étape 1 — kernel-level spatial band-streaming.
    # When the output tensor would exceed the per-launch threshold (default
    # 4 GiB), split along H and stream band-by-band. Halo handling mirrors
    # _tiled_conv2d_spatial_nbx: each band re-enters this same wrapper with
    # a smaller H, so the recursion bottoms out on its own. The full output
    # remains allocated for downstream consumers in the DAG.
    # x.dtype returns a triton.language dtype; use x.nbx_dtype for the
    # NBXDtype used by dtype_size().
    out_nbx_dtype = x.nbx_dtype if hasattr(x, 'nbx_dtype') else x._dtype
    out_dtype_bytes = dtype_size(out_nbx_dtype)
    if _NBX_CONV2D_TRACE:
        out_mb = N * out_c * out_h * out_w * out_dtype_bytes / 1024 / 1024
        print(f"[CONV2D] in=({N},{in_c},{in_h},{in_w}) out=({N},{out_c},{out_h},{out_w}) "
              f"k=({kh},{kw}) g={groups} out={out_mb:.1f}MB", flush=True)

    # P-SANA-4KPX-RUNTIME Étape 3 — depthwise specialization. The generic
    # im2col conv2d_forward_kernel is structurally inefficient for the
    # depthwise pattern (groups == in_c == out_c, weight (C,1,kh,kw)) on
    # Sana 4Kpx VAE: ~4.8 s per call vs cuDNN dedicated path ~2.6 ms,
    # ~1800x gap. Route to the dedicated stencil kernel instead.
    if (os.environ.get("NBX_DEPTHWISE_DISABLE", "0") != "1") and groups == in_c and groups == out_c and dil_h == 1 and dil_w == 1:
        if _NBX_CONV2D_TRACE:
            print(f"[CONV2D] DEPTHWISE path (g={groups})", flush=True)
        return _depthwise_conv2d_dispatch(
            x_c, w_c, bias,
            N, in_c, in_h, in_w, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w,
            out_dtype,
        )

    if _conv2d_should_band_stream(N, out_c, out_h, out_w, out_dtype_bytes):
        if _NBX_CONV2D_TRACE:
            print(f"[CONV2D] BAND-STREAM triggered (out > {_NBX_CONV2D_BAND_BYTES/1024/1024/1024:.1f}GiB)", flush=True)
        return _conv2d_band_streamed(
            x_c, w_c, bias,
            N, in_c, in_h, in_w, out_c, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, out_dtype, out_dtype_bytes,
        )

    output = NBXTensor.empty((N, out_c, out_h, out_w), device=x.device, dtype=out_dtype)

    fp16 = x.dtype == NBXDtype.float16

    # Phase 1.5 conv2d autotune: BLOCK_SIZE_*/num_warps/num_stages chosen
    # adaptively per shape signature, persisted via cache_results=True.
    grid = lambda META: (
        triton.cdiv(N * out_h * out_w, META['BLOCK_SIZE_BHW']),
        triton.cdiv(out_c // groups, META['BLOCK_SIZE_OUTF']),
        groups,
    )
    _set_device(x_c)
    conv2d_forward_kernel[grid](
        x_c, w_c, output,
        N, in_c, in_h, in_w,
        out_c, out_h, out_w,
        *x_c.stride(), *w_c.stride(), *output.stride(),
        kernel_height=kh, kernel_width=kw,
        stride_height=stride_h, stride_width=stride_w,
        padding_height=pad_h, padding_width=pad_w,
        dilation_height=dil_h, dilation_width=dil_w,
        groups=groups, fp16=fp16,
    )

    if bias is not None:
        output = add(output, bias.view(1, -1, 1, 1))

    return output


def _depthwise_conv2d_dispatch(
    x_c, w_c, bias,
    N, C, IH, IW, out_h, out_w,
    kh, kw, stride_h, stride_w, pad_h, pad_w,
    out_dtype,
):
    """Launch the depthwise stencil kernel. Caller has already verified
    the depthwise signature (groups == in_c == out_c, dilation == 1) and
    done dtype narrowing + contiguous(). Bias is added separately so the
    kernel stays signature-clean."""
    output = NBXTensor.empty((N, C, out_h, out_w), device=x_c.device, dtype=out_dtype)
    x_nbx_dt = x_c.nbx_dtype if hasattr(x_c, 'nbx_dtype') else x_c._dtype
    fp16 = x_nbx_dt == NBXDtype.float16

    grid = lambda META: (
        N,
        triton.cdiv(C, META['BLOCK_C']),
        triton.cdiv(out_h * out_w, META['BLOCK_HW']),
    )
    _set_device(x_c)
    depthwise_conv2d_kernel[grid](
        x_c, w_c, output,
        N, C,
        IH, IW, out_h, out_w,
        *x_c.stride(), *w_c.stride()[:1], *w_c.stride()[2:],
        *output.stride(),
        kh=kh, kw=kw,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        fp16=fp16,
    )

    if bias is not None:
        output = add(output, bias.view(1, -1, 1, 1))

    return output


def _conv2d_band_streamed(
    x_c, w_c, bias,
    N, in_c, IH, IW, out_c, out_h, out_w,
    kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    groups, out_dtype, out_dtype_bytes,
):
    """Band-streaming inner path for conv2d_wrapper. The full output tensor
    is allocated up front (downstream consumers expect it whole); per band
    we slice the input H, recurse into conv2d_wrapper for the band (which
    falls back to the single-launch path because the band's output is now
    small), and write the band slice back into the full output.

    Halo on internal frontiers: each band's input slice is extended by
    pad_h on the read side (clamped to image edges); the recursive
    conv2d_wrapper call uses the original padding=(pad_h, pad_w), which
    inserts pad_h zeros on internal frontiers — same engineering trade
    used by `_tiled_conv2d_spatial_nbx` for compiled mode (faint seam at
    band frontiers, accepted as a follow-up halo-correctness chantier).
    """
    # Choose tile_factor so each band's output bytes <= half the threshold;
    # the headroom covers transient input slice + kernel intermediate.
    band_target_bytes = max(1, _NBX_CONV2D_BAND_BYTES // 2)
    row_bytes = N * out_c * out_w * out_dtype_bytes
    rows_per_band = max(1, band_target_bytes // max(1, row_bytes))
    tile_factor = max(1, (out_h + rows_per_band - 1) // rows_per_band)
    band_oh = (out_h + tile_factor - 1) // tile_factor

    output = NBXTensor.empty((N, out_c, out_h, out_w), device=x_c.device, dtype=out_dtype)

    for oh_start in range(0, out_h, band_oh):
        oh_end = min(oh_start + band_oh, out_h)
        ih_start = max(0, oh_start * stride_h - pad_h)
        ih_end = min(IH, (oh_end - 1) * stride_h + dil_h * (kh - 1) + 1 - pad_h)
        if ih_end <= ih_start:
            continue
        in_band = x_c[:, :, ih_start:ih_end, :]
        # Recurse — band's out_h is now small enough to fall through to the
        # single-launch path. bias is applied once at the end here (not
        # per-band) to avoid double-add when the band path returns.
        conv_band = conv2d_wrapper(
            in_band, w_c, None,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dil_h, dil_w),
            groups=groups,
        )
        actual_band_h = oh_end - oh_start
        band_first_oh = ih_start // stride_h
        local_offset = max(0, oh_start - band_first_oh)
        actual_band_h = min(conv_band.shape[2] - local_offset, actual_band_h)
        if actual_band_h <= 0:
            continue
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, local_offset:local_offset + actual_band_h, :]

    if bias is not None:
        output = add(output, bias.view(1, -1, 1, 1))

    return output


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

        _set_device(row_in)
        scan_part_sum_kernel[(part_num,)](
            row_in, row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

        if part_num > 1:
            for i in range(1, part_num):
                partial_sum[i] += partial_sum[i - 1]
            _set_device(row_out)
            add_base_sum_kernel[(part_num,)](
                row_out, partial_sum, scan_size, part_num, BLOCK_SIZE=BLOCK_SIZE)

    return output_2d.view_as(x_perm).movedim(-1, dim)


# ===========================================================================
# PHASE 3 — Simple element-wise wrappers
# ===========================================================================

def exp2_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    exp2_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def tan_wrapper(x) :
    x = _promote_int_unary(x).contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    tan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def celu_wrapper(x, alpha: float = 1.0) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    celu_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), alpha, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def log_sigmoid_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    log_sigmoid_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isfinite_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isfinite_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isinf_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isinf_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def isnan_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
    isnan_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def nan_to_num_wrapper(x, nan: float = 0.0,
                       posinf: float = 3.4028235e+38,
                       neginf: float = -3.4028235e+38) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
    nan_to_num_forward_kernel[_1d_grid(x.numel())](
        x, output, x.numel(), nan, posinf, neginf,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def threshold_wrapper(x, threshold: float, value: float) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(x)
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
    _set_device(input)
    addcdiv_forward_kernel[_1d_grid(input.numel())](
        input, tensor1, tensor2, output, input.numel(), value,
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def addcmul_wrapper(input, tensor1,
                    tensor2, value: float = 1.0) :
    """result = input + value * tensor1 * tensor2"""
    input, tensor1, tensor2 = input.contiguous(), tensor1.contiguous(), tensor2.contiguous()
    output = NBXTensor.empty_like(input)
    _set_device(input)
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
        _set_device(x)
        all_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
        _set_device(mid)
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
        _set_device(x_perm)
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
        _set_device(x)
        any_kernel_mid[(mid_size,)](x, mid, M, mid_size, BLOCK_SIZE=BLOCK_SIZE)
        _set_device(mid)
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
        _set_device(x_perm)
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
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        # 0-dim / python-scalar operand (e.g. the traced `mask.all() & qside`
        # opening the T5 extended-mask chain): materialize to a full tensor of
        # a's dtype and run the tensor kernel. Cold path, exact for every
        # dtype. The former code fed the 0-dim tensor straight to the flat
        # kernel, reading a.numel() elements past its 1-element allocation
        # (out-of-bounds, undefined values).
        b = NBXTensor.empty_like(a).fill_(b)
    bitwise_and_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_or_wrapper(a, b) :
    # Same universal binary contract as bitwise_and_wrapper (broadcast fix).
    a, b, output, n, dev_ctx, scalar = _prepare_binary(a, b)
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    bitwise_or_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
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
    bitwise_not_forward_kernel[_1d_grid(x.numel())](x, output, x.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


# ---------------------------------------------------------------------------
# Logical ops (binary: AND/OR → bool, unary: NOT → bool)
# ---------------------------------------------------------------------------

def logical_and_wrapper(a, b) :
    # _prepare_comparison = the universal binary contract with a bool output
    # (dtype align, device align, BROADCAST, contiguous). Same broadcast fix
    # as bitwise_and_wrapper — the former form silently ignored broadcasting.
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    logical_and_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def logical_or_wrapper(a, b) :
    # Same universal binary contract as logical_and_wrapper (broadcast fix).
    a, b, output, n, dev_ctx, scalar = _prepare_comparison(a, b)
    if scalar:
        # Scalar operand: materialize (see bitwise_and_wrapper).
        b = NBXTensor.empty_like(a).fill_(b)
    logical_or_forward_kernel[_1d_grid(n)](a, b, output, n, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def logical_not_wrapper(x) :
    x = x.contiguous()
    output = NBXTensor.empty(x.shape, dtype=NBXDtype.bool_, device=x.device)
    _set_device(x)
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


def conv_transpose1d_wrapper(x, weight, bias=None, stride=1, padding=0,
                             output_padding=0, groups=1, dilation=1) :
    """aten::conv_transpose1d — 1D transposed convolution. conv_transpose_wrapper
    already handles the 1D case (weight.ndim == 3, H=1 unsqueeze); this thin
    adapter only reorders the positional args from the aten signature
    (stride, padding, output_padding, groups, dilation) to the wrapper's keyword
    order. Used by the chatterbox/CosyVoice s3gen HiFiGAN-style upsampler."""
    return conv_transpose_wrapper(
        x, weight, bias,
        stride=stride, padding=padding, dilation=dilation,
        output_padding=output_padding, groups=groups)


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
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    elif abs(weight) < 0.5:
        _set_device(input)
        lerp_scalar_head_kernel[_1d_grid(input.numel())](
            input, end, output, input.numel(), float(weight),
            BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        _set_device(input)
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
        _set_device(x)
        dot_kernel_small[(1,)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
        return out.to(x.dtype)
    else:
        mid_size = triton.cdiv(N, BLOCK_SIZE)
        mid = NBXTensor.empty(mid_size, dtype=NBXDtype.float32, device=x.device)
        _set_device(x)
        dot_kernel_partial[(mid_size,)](x, y, mid, N, BLOCK_SIZE=BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)
        out = NBXTensor.empty([], dtype=NBXDtype.float32, device=x.device)
        _set_device(mid)
        dot_kernel_reduce[(1,)](mid, out, mid_size, BLOCK_MID=BLOCK_MID)
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# MV (matrix-vector multiply)
# ---------------------------------------------------------------------------

def mv_wrapper(mat, vec) :
    """Matrix-vector multiply: out[i] = sum_j(mat[i, j] * vec[j])."""
    mat, vec = mat.contiguous(), vec.contiguous()
    N, M = mat.shape
    out = NBXTensor.empty(N, device=mat.device, dtype=_matmul_out_dtype(mat))
    grid = (triton.cdiv(N, _MV_BN),)
    _set_device(mat)
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

def roll_wrapper(x, shifts, dims=None):
    """Circular shift along given dim(s) — `torch.roll` semantics.

    R33-pure: decomposed into `narrow` + `NBXTensor.cat`, no new
    Triton kernel. A roll by `s` on a dim of size `n` is
    `cat([x[..., n-s:, ...], x[..., :n-s, ...]], dim)`. Used by
    Swin-attention models (Swin2SR / SwinIR / HAT) for the cyclic
    window shift.
    """
    # Normalise shifts/dims to parallel lists (torch allows int or
    # tuple for both; dims=None means roll over the flattened tensor).
    if dims is None:
        flat = x.contiguous().view(-1)
        n = flat.shape[0]
        s = int(shifts if isinstance(shifts, int) else shifts[0]) % n
        if s == 0:
            return x
        rolled = NBXTensor.cat(
            [flat.narrow(0, n - s, s), flat.narrow(0, 0, n - s)], dim=0)
        return rolled.view(*x.shape)

    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    result = x.contiguous()
    for shift, dim in zip(shifts, dims):
        dim = dim % result.ndim
        n = result.shape[dim]
        s = int(shift) % n
        if s == 0:
            continue
        tail = result.narrow(dim, n - s, s)
        head = result.narrow(dim, 0, n - s)
        result = NBXTensor.cat([tail, head], dim=dim).contiguous()
    return result


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
            _set_device(result)
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
            _set_device(result)
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
        _set_device(x.view(-1))
        prod_kernel_mid[(mid_size,)](x.view(-1), mid, M, BLOCK_SIZE=BLOCK_SIZE)
        _set_device(mid)
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
        _set_device(x_perm)
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
        grid = (triton.cdiv(M, _RED_BM),)
        _set_device(x_perm)
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
        grid = (triton.cdiv(M, _RED_BM),)
        _set_device(x_perm)
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

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        1,
        B,
    )
    _set_device(batch1)
    baddbmm_kernel[grid](
        batch1, batch2, output, input,
        alpha, beta,
        M, N, K,
        batch1.stride(0), batch1.stride(1), batch1.stride(2),
        batch2.stride(0), batch2.stride(1), batch2.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        bias_batch_stride, bias_m_stride, bias_n_stride,
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
    out = NBXTensor.empty(N, device=mat.device, dtype=_matmul_out_dtype(mat))
    grid = (triton.cdiv(N, _MV_BN),)
    _set_device(mat)
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
        _set_device(input)
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
    _set_device(input)
    mse_loss_partial_kernel[(mid_size,)](
        input, target, mid, M, BLOCK_SIZE=BLOCK_SIZE, reduction=reduction)
    _set_device(mid)
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

    _set_device(input)
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

    _set_device(input)
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
        _set_device(x.view(-1))
        std_map_kernel[(num_blocks,)](x.view(-1), tmp_sum, tmp_sum_sq, N, BLOCK_N=BLOCK_N)
        BLOCK_SIZE = triton.next_power_of_2(num_blocks)
        _set_device(tmp_sum)
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
        _set_device(x_2d)
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
        _set_device(x.view(-1))
        var_kernel_1[(num_blocks,)](x.view(-1), acc, average, count, N, BLOCK_N=BLOCK_N)
        BLOCK_NUM = triton.next_power_of_2(num_blocks)
        _set_device(acc)
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
        _set_device(x_2d)
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

def broadcast_tensors_wrapper(*tensors):
    """aten::broadcast_tensors(Tensor[] tensors) -> Tensor[].

    The TensorList arrives as a single list/tuple positional arg. Broadcast every
    input to their common (numpy-style, right-aligned) shape and return them
    UNPACKED — one output tensor per input — so the executor fills out_0, out_1,
    ... A bare ``lambda *t: t`` returned the nested list into out_0, so a
    downstream consumer (lt / where / stack) received a Python list instead of an
    NBXTensor (``'list' object has no attribute '_dtype'``).
    """
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if len(tensors) <= 1:
        return tuple(tensors)
    ndim = max(t.ndim for t in tensors)
    out_shape = [1] * ndim
    for t in tensors:
        sh = (1,) * (ndim - t.ndim) + tuple(t.shape)
        for i, s in enumerate(sh):
            if s != 1:
                if out_shape[i] != 1 and out_shape[i] != s:
                    raise RuntimeError(
                        f"broadcast_tensors: incompatible shapes at dim {i}: "
                        f"{out_shape[i]} vs {s}")
                out_shape[i] = s
    out_shape = tuple(out_shape)
    return tuple(t if tuple(t.shape) == out_shape else t.expand(*out_shape)
                 for t in tensors)


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

    # Pre-compute strides for the kernel.
    # gather_kernel's outer_idx is a ROW-MAJOR FLATTENING of every dim before
    # `dim` (range [0, outer_size)). For a contiguous tensor that whole outer
    # block collapses to a single stride = product of the sizes from `dim`
    # onward = shape[dim] * inner_size (== stride(dim-1)). Using stride(0) is the
    # stride of dim 0 = prod(shape[1:]); it equals the correct value ONLY when
    # there is exactly one outer dim (dim == 1). With >= 2 outer dims it
    # over-counts by prod(shape[1:dim]) and drives `outer_idx * stride` past the
    # buffer end -> destructive OOB store/load -> CUDA 700 at the next sync
    # (e.g. chatterbox s3gen T5 rel-pos gather [1,8,180,359] idx dim=3). input,
    # index and out are all contiguous here, so the flat collapse is exact.
    inp_dim_stride = input.stride(dim)
    idx_stride_outer = dim_size * inner_size if dim > 0 else 0
    idx_stride_dim = index.stride(dim)
    idx_stride_inner = index.stride(-1) if dim < input.ndim - 1 else 0
    inp_stride_outer = input.shape[dim] * inner_size if dim > 0 else 0
    inp_stride_inner = input.stride(-1) if dim < input.ndim - 1 else 0
    out_stride_outer = dim_size * inner_size if dim > 0 else 0
    out_stride_dim = out.stride(dim)
    out_stride_inner = out.stride(-1) if dim < input.ndim - 1 else 0

    _set_device(input)
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

    _set_device(src)
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

    _set_device(src)
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

    _set_device(out)
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

    outer_size = 1
    for i in range(dim):
        outer_size *= source.shape[i]
    inner_size = 1
    for i in range(dim + 1, source.ndim):
        inner_size *= source.shape[i]
    dim_size = source.shape[dim]
    inp_shape_dim = x.shape[dim]

    if x.numel() == 0 or dim_size == 0:     # nothing to add -> out = x
        return x.clone()

    out = NBXTensor.empty_like(x)

    src_stride_outer = source.stride(0) if dim > 0 else 0
    src_stride_dim = source.stride(dim)
    src_stride_inner = source.stride(-1) if dim < source.ndim - 1 else 0

    # Deterministic output-owner gather: one program per
    # (outer, dest, inner-tile) output cell, ascending-j sequential
    # fp32 fold, zero atomics, no sort dependency. See
    # ops/index_add.py for the rationale and state-of-the-art notes.
    n_inner_tiles = (inner_size + _EW_BLOCK - 1) // _EW_BLOCK
    grid = (outer_size * inp_shape_dim * n_inner_tiles,)
    _set_device(x)
    index_add_gather_kernel[grid](
        x, source, index, out,
        dim_size, inp_shape_dim, inner_size, alpha,
        src_stride_outer, src_stride_dim, src_stride_inner,
        INNER_BLOCK=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


# ---------------------------------------------------------------------------
# Index Put
# ---------------------------------------------------------------------------

def index_put_wrapper(x, indices, values, accumulate: bool = False):
    """aten::index_put / index_put_ — scatter-write `values` into `x`.

    Functional: returns a fresh tensor (the executor reassigns the op's
    output slot, exactly like `index_add_wrapper`; the `_` in-place
    variant is realised by the graph, not by mutating the input here).

    Scope = the decomposed-ATen norm: exactly ONE non-None *integer*
    index tensor on the leading dim (covers MoE-v2 expert aggregation,
    KV-cache indexed writes, post-`aten::nonzero` masked scatter).
    k>=2 advanced indices, non-leading-dim indexing, boolean masks, and
    un-broadcastable `values` raise NotImplementedError with a named
    follow-up rather than silently mis-scattering (ZERO-FALLBACK — a
    visible crash beats the previous identity-lambda silent data loss).
    """
    from .nbx_tensor import NBXDtype

    if not isinstance(indices, (list, tuple)):
        indices = [indices]
    non_none = [(d, ix) for d, ix in enumerate(indices) if ix is not None]
    if len(non_none) != 1 or non_none[0][0] != 0:
        raise NotImplementedError(
            "aten::index_put supports exactly one leading-dim index "
            f"tensor; got non-None at positions "
            f"{[d for d, _ in non_none]} of {len(indices)} — k>=2 / "
            "non-leading advanced indexing is unwired (follow-up "
            "P-INDEX-PUT-ADVANCED-GENERAL).")
    idx = non_none[0][1]
    # Boolean-mask index_put = masked_fill semantics: out[mask] = scalar, the
    # mask broadcasting over x's TRAILING dims. The attention masked_fill
    # (scores.masked_fill(~mask, value)) functionalises to exactly this form —
    # a 1-byte mask whose shape is a leading-dim PREFIX of x, with a scalar value.
    # Route it to the masked_fill kernel instead of the 1-D integer-index path
    # (which read the mask as garbage indices and wrote far out of bounds → the
    # OpenAudio DualAR-decode triton CUDA-700). Match the mask dtype by NAME,
    # robust to whether the index was typed NBXDtype.{bool_,uint8} or left as a
    # raw triton dtype (tl.uint8) by an upstream bitwise_not/isin. The shape-prefix
    # + scalar-value test keeps genuine 1-D integer indices on the integer path.
    _idx_dtn = str(idx.dtype).lower()
    if (("uint8" in _idx_dtn or "bool" in _idx_dtn)
            and idx.ndim < x.ndim
            and list(idx.shape) == list(x.shape[:idx.ndim])
            and values.numel() == 1):
        mask_b = idx.reshape([*list(idx.shape),
                              *([1] * (x.ndim - idx.ndim))])
        fill_val = _to_scalar(values) if isinstance(values, NBXTensor) else float(values)
        return masked_fill(x, mask_b, fill_val)
    if idx.dtype == NBXDtype.bool_:
        raise NotImplementedError(
            "aten::index_put with a boolean-mask index is unwired (no "
            "nonzero kernel in the catalogue) — follow-up "
            "P-INDEX-PUT-ADVANCED-GENERAL.")
    if idx.dtype in (NBXDtype.float16, NBXDtype.bfloat16,
                     NBXDtype.float32, NBXDtype.float64):
        raise NotImplementedError(
            f"aten::index_put index tensor must be integer, got "
            f"{idx.dtype} — follow-up P-INDEX-PUT-ADVANCED-GENERAL.")

    x = x.contiguous()
    idx = idx.contiguous()
    out = x.clone()

    T = 1
    for i in range(1, x.ndim):
        T *= x.shape[i]
    Sn = idx.numel()
    N = Sn * T
    if N == 0:
        return out

    vnumel = values.numel()
    if vnumel == 1:
        val_scalar = True
        vbuf = values.contiguous()
    elif vnumel == N:
        val_scalar = False
        vbuf = values.contiguous()
    else:
        raise NotImplementedError(
            f"aten::index_put values numel {vnumel} != idx*tail {N} "
            "and not scalar — value broadcasting unwired (follow-up "
            "P-INDEX-PUT-ADVANCED-GENERAL).")

    _set_device(out)
    index_put_kernel[_1d_grid(N)](
        out, idx, vbuf,
        T, N,
        VAL_SCALAR=val_scalar,
        ACCUMULATE=bool(accumulate),
        BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS,
    )
    return out


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
    _set_device(x.contiguous())
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
    _set_device(x.contiguous())
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


def max_pool3d_with_indices_wrapper(
    x, kernel_size, stride=None, padding=0,
    dilation=1, ceil_mode=False,
):
    """Max pooling 3D forward — aten::max_pool3d_with_indices.

    Returns (values, indices); indices are int64 flattened (T, H, W)
    offsets per (N, C) plane (PyTorch convention). Dedicated
    @triton.jit kernel (kernels/ops/max_pool3d.py), temporal extension
    of the 2D pool kernel. Video conditioning masks (e.g. Allegro-TI2V
    state_video_mask downsampled to the latent grid) trace to this op.
    """
    assert x.ndim == 5
    N, C, T, H, W = x.shape

    def _t3(v, default):
        if v is None:
            return default
        if isinstance(v, int):
            return (v, v, v)
        return (v[0], v[1], v[2]) if len(v) >= 3 else (v[0], v[0], v[0])

    kt, kh, kw = _t3(kernel_size, None)
    st_, sh, sw = _t3(stride, (kt, kh, kw)) if stride not in (None, [], ()) \
        else (kt, kh, kw)
    pt, ph, pw = _t3(padding, (0, 0, 0))
    dt_, dh, dw = _t3(dilation, (1, 1, 1))

    def _odim(inp, k, s, p, d):
        if ceil_mode:
            return (inp + 2 * p - d * (k - 1) - 1 + s - 1) // s + 1
        return (inp + 2 * p - d * (k - 1) - 1) // s + 1

    ot = _odim(T, kt, st_, pt, dt_)
    oh = _odim(H, kh, sh, ph, dh)
    ow = _odim(W, kw, sw, pw, dw)

    x_c = x.contiguous()
    output = NBXTensor.empty((N, C, ot, oh, ow), device=x_c.device, dtype=x_c.dtype)
    indices = NBXTensor.empty((N, C, ot, oh, ow), device=x_c.device,
                              dtype=NBXDtype.int64)

    grid = (
        N * C * ot,
        triton.cdiv(oh, _POOL_BH) * triton.cdiv(ow, _POOL_BW),
    )
    _set_device(x_c)
    max_pool3d_forward_kernel[grid](
        x_c, output, indices,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3), x_c.stride(4),
        C, T, H, W,
        ot, oh, ow,
        kt, kh, kw, st_, sh, sw, pt, ph, pw, dt_, dh, dw,
        BLOCK_H=_POOL_BH, BLOCK_W=_POOL_BW,
        num_warps=4,
    )
    return output, indices


def max_pool3d_wrapper(
    x, kernel_size, stride=None, padding=0,
    dilation=1, ceil_mode=False,
):
    """aten::max_pool3d — values-only variant."""
    return max_pool3d_with_indices_wrapper(
        x, kernel_size, stride, padding, dilation, ceil_mode)[0]


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

    grid_r = triton.cdiv(num_bins, TILE_R)
    SWEEP_TILE = 2048
    sweep_grid_n = triton.cdiv(n, SWEEP_TILE)
    status = NBXTensor.empty((m, num_bins, sweep_grid_n), device=x.device, dtype=NBXDtype.int32)

    # Stage 2: sweep per radix pass
    for i in range(n_passes):
        status.zero_()
        _set_device(arr_in)
        radix_sort_sweep_kernel[(m * sweep_grid_n, grid_r)](
            arr_in, indices_in, arr_out, indices_out,
            ex_cumsum, status,
            n_passes, i, i * k_bits, m, n, n,
            SWEEP_TILE, TILE_R, k_bits, descending)
        arr_in, arr_out = arr_out, arr_in
        indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


# ===========================================================================
# PHASE 5 — spatial, RNG, remaining wrappers
# ===========================================================================

def pixel_shuffle_wrapper(x, upscale_factor: int) :
    """Pixel shuffle: [N,C*r*r,H,W] -> [N,C,H*r,W*r].

    If `x` is a `BroadcastClonePyroxy` (returned by an upstream intercepted
    aten.clone whose input is an NBX expand stride-0 view), reads through
    the unmaterialized broadcast view via a stride-aware kernel and
    eliminates the 8 GiB clone copy. P-SANA-4KPX-RUNTIME Fix F2 (Approche C).
    """
    from .ops.fused_upsample_conv import BroadcastClonePyroxy

    if isinstance(x, BroadcastClonePyroxy):
        return _pixel_shuffle_broadcast_aware(x, upscale_factor)

    N, C_in, H, W = x.shape
    r = upscale_factor
    C_out = C_in // (r * r)
    OH, OW = H * r, W * r
    x = x.contiguous()
    output = NBXTensor.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C_out * OH * OW
    grid = (triton.cdiv(total, _EW_BLOCK),)
    _set_device(x)
    pixel_shuffle_kernel[grid](
        x, output, total,
        C_out, H, W, r, OH, OW,
        *x.stride(), *output.stride(),
        BLOCK_SIZE=_EW_BLOCK)
    return output


def _pixel_shuffle_broadcast_aware(proxy, upscale_factor: int):
    """Read directly through an NBX expand stride-0 view, allocate clean
    4D contiguous output, no clone materialization.

    `proxy.bcast_view` is the expand output (5D NBXTensor with stride 0
    on `proxy.bcast_dim`). Same data_ptr as the pre-expand tensor — NBX
    expand is metadata-only (kernels/nbx_tensor.py:1581). The kernel reads
    via 5D strides; the stride-0 dim aliases broadcast indices for free.

    Layout assumption (matches DC-AE / Sana 4Kpx VAE pattern after
    `unsqueeze(dim=2) -> expand(dim=2, B)`): the 5D view is laid out
    (N, C_pre, B, H, W). bcast_dim is dim 2.

    Output shape: (N, C_out, H*r, W*r) where C_out = (C_pre * B) // (r*r).
    """
    view = proxy.bcast_view
    bcast = proxy.bcast_factor
    bcast_dim = proxy.bcast_dim
    r = int(upscale_factor)

    if view.ndim != 5 or bcast_dim != 2:
        raise NotImplementedError(
            f"_pixel_shuffle_broadcast_aware: only 5D layout (N,C,B,H,W) "
            f"with bcast_dim=2 supported (matches NBX unsqueeze+expand "
            f"DC-AE pattern). Got ndim={view.ndim} bcast_dim={bcast_dim}.")

    N, C_pre, B_size, H, W = view.shape
    if B_size != bcast:
        raise AssertionError(
            f"_pixel_shuffle_broadcast_aware: view dim 2 size {B_size} != "
            f"bcast factor {bcast} from static DAG analysis")
    C_in_view = C_pre * bcast
    if C_in_view % (r * r) != 0:
        raise AssertionError(
            f"_pixel_shuffle_broadcast_aware: post-view channels {C_in_view} "
            f"not divisible by r*r={r*r}")
    C_out = C_in_view // (r * r)
    OH, OW = H * r, W * r

    output = NBXTensor.empty((N, C_out, OH, OW),
                             device=view.device, dtype=view.dtype)
    total = N * C_out * OH * OW
    grid = (triton.cdiv(total, _EW_BLOCK),)
    _set_device(view)
    pixel_shuffle_broadcast_aware_kernel[grid](
        view, output, total,
        C_out, OH, OW, r, bcast,
        *view.stride(), *output.stride(),
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
    _set_device(x)
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
    # Grid must match the kernel's axis use: program_id(0)→ow (OW, BLOCK_X),
    # program_id(1)→oh (OH, BLOCK_Y). The OH/OW axes were swapped here, which is
    # invisible for square outputs (OH==OW, the image-upscaler case) but drops
    # most output columns when OH != OW — e.g. the 1-D-as-2-D path used by
    # upsample_linear1d (H=1), which the Kokoro iSTFT SineGen relies on.
    grid = (
        triton.cdiv(OW, _BILINEAR_BX),
        triton.cdiv(OH, _BILINEAR_BY),
        N * C,
    )
    _set_device(x.contiguous())
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
    _set_device(x_c)
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
    _set_device(x.contiguous())
    conv_depthwise2d_kernel[grid](
        x.contiguous(), weight.contiguous(), output,
        N, C, IH, IW, C, KH, KW, OH, OW,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        1,  # ch_mult
        BLOCK_X=_BILINEAR_BX, BLOCK_Y=_BILINEAR_BY,
        num_warps=4)
    if bias is not None:
        output = add(output, bias.view(1, -1, 1, 1))
    return output


def logical_xor_wrapper(a, b) :
    a, b = a.contiguous(), b.contiguous()
    output = NBXTensor.empty(a.shape, dtype=NBXDtype.bool_, device=a.device)
    _set_device(a)
    logical_xor_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_left_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        _set_device(a)
        bitwise_left_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        from .ops.bitwise_left_shift import bitwise_left_shift_scalar_kernel
        _set_device(a)
        bitwise_left_shift_scalar_kernel[_1d_grid(a.numel())](a, output, a.numel(), int(b), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def bitwise_right_shift_wrapper(a, b) :
    a = a.contiguous()
    output = NBXTensor.empty_like(a)
    if isinstance(b, NBXTensor):
        b = b.contiguous()
        _set_device(a)
        bitwise_right_shift_kernel[_1d_grid(a.numel())](a, b, output, a.numel(), BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    else:
        from .ops.bitwise_right_shift import bitwise_right_shift_scalar_kernel
        _set_device(a)
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
        _set_device(x_2d)
        l2_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == float('inf'):
        _set_device(x_2d)
        linf_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == 0 or ord == 0.0:
        _set_device(x_2d)
        l0_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    elif ord == 1 or ord == 1.0:
        _set_device(x_2d)
        l1_norm_kernel[grid](x_2d, out, M, N, BLOCK_M=_RED_BM, BLOCK_N=_RED_BN, num_warps=4)
    else:
        _set_device(x_2d)
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

    chunk_num = triton.cdiv(topk_n, chunk_size)

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
    _set_device(output)
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
        _set_device(output)
        weight_norm_kernel_first[grid](output, norm, v, g, M, N, eps,
                                       BLOCK_M=_WNORM_BM, BLOCK_N=_WNORM_BN,
                                       num_warps=4)
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        grid = (triton.cdiv(N, _WNORM_BN),)
        _set_device(output)
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
    # NBXTensor has no .cumsum()/[-1] — use the cumsum wrapper + select/item
    # (the wrapper was never exercised in triton before Kokoro's alignment).
    cumsum = cumsum_wrapper(repeats, 0)
    result_size = int(cumsum.select(0, -1).item())

    out = NBXTensor.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    BLOCK_SIZE = 32
    _set_device(repeats)
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


def repeat_interleave_wrapper(*args, **kwargs):
    """aten::repeat_interleave dispatcher — routes by args, since the bare
    op_type cannot distinguish the overloads:
      repeat_interleave(repeats: Tensor)            1 arg        -> .Tensor   (index build)
      repeat_interleave(self, repeats: int, dim)    int repeats  -> .self_int
      repeat_interleave(self, repeats: Tensor, dim) Tensor reps  -> .self_Tensor
    Kokoro's predictor alignment uses the 1-arg Tensor variant
    (repeat_interleave(pred_dur) -> interleaved indices, length = sum(pred_dur));
    the dispatch previously hardcoded self_int and dropped the `repeats` arg.
    """
    if len(args) == 1:
        return repeat_interleave_tensor_wrapper(args[0], **kwargs)
    if isinstance(args[1], NBXTensor):
        return repeat_interleave_self_tensor_wrapper(*args, **kwargs)
    return repeat_interleave_self_int_wrapper(*args, **kwargs)


# ===========================================================================
# RNG OPS — Random number generation via Triton kernels
# ===========================================================================

def _draw_seed() -> int:
    """Per-draw Philox kernel seed for the random wrappers.

    Run-scoped stream when armed (reproducible, --seed-driven; the Triton
    mirror of the compiled engine's run-scoped generator — see
    kernels/rng_stream.py), else the historical unseeded fallback. The
    NBX_FORCE_RAND_SEED pin (rng_pin) is checked by the callers BEFORE any
    kernel draw and keeps precedence over this stream.
    """
    from . import rng_stream
    if rng_stream.active():
        return rng_stream.next_seed()
    return __import__("random").randint(0, 2**31 - 1)


def rand_wrapper(*args, **kwargs) :
    """rand(size, ...) → uniform [0, 1) via Triton kernel."""
    from .ops.rand_op import rand_kernel
    from .rng_pin import pinned_seed, pinned_uniform
    # ATen signature: rand(SymInt[] size, ...)
    size = args[0] if args else kwargs.get('size', [])
    if isinstance(size, NBXTensor):
        size = size.tolist()
    dtype = kwargs.get('dtype', NBXDtype.float32)
    device = kwargs.get('device', None)
    if device is None:
        device = str('cuda')
    if pinned_seed() is not None:
        return NBXTensor.from_numpy(pinned_uniform(size)).to(dtype)
    output = NBXTensor.empty(size, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = _draw_seed()
    _set_device(output)
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
    seed = _draw_seed()
    _set_device(output)
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
    seed = _draw_seed()
    _set_device(output)
    rand_kernel[_1d_grid(n)](output, n, seed, BLOCK_SIZE=_EW_BLOCK, num_warps=_EW_WARPS)
    return output


def randn_like_wrapper(x, **kwargs) :
    """randn_like(tensor) → standard normal N(0,1) same shape/dtype/device."""
    from .ops.rand_op import randn_kernel
    from .rng_pin import pinned_seed, pinned_normal
    dtype = kwargs.get('dtype', x.dtype)
    device = kwargs.get('device', x.device)
    if pinned_seed() is not None:
        return NBXTensor.from_numpy(pinned_normal(x.shape)).to(dtype)
    output = NBXTensor.empty(x.shape, dtype=dtype, device=device)
    n = output.numel()
    if n == 0:
        return output
    seed = _draw_seed()
    _set_device(output)
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
    seed = _draw_seed()
    _set_device(output)
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
    seed = _draw_seed()
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
    seed = _draw_seed()
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
        seed = _draw_seed()
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
            seed = _draw_seed()
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
        # General N-D constant pad (pads any combination of dims, e.g. a middle
        # temporal dim in a 5D [B,C,T,H,W] conv3d). The optimized 1D/2D kernels
        # above only touch the last 1/2 dims; this branch handled everything
        # else but ONLY ever padded the last dim (silent garbage when padding a
        # middle dim — pad_before[-1] is 0 for e.g. [0,0,0,0,pt,pt]). Decompose
        # into cat-of-constant-blocks per padded dim: R33-pure (NBXTensor.zeros
        # /fill_ + NBXTensor.cat, which dispatches cat_copy_kernel for dim>0).
        # Every padded region is `value`, so corners (padded on 2+ dims) come
        # out correct regardless of dim order.
        def _const_block(width, dim_idx, ref):
            shp = list(ref.shape); shp[dim_idx] = width
            blk = NBXTensor.zeros(tuple(shp), dtype=ref.dtype, device=ref.device)
            if float(value) != 0.0:
                blk.fill_(float(value))
            return blk

        result = x
        for d in range(ndim):
            b, a = pad_before[d], pad_after[d]
            if b == 0 and a == 0:
                continue
            parts = []
            if b > 0:
                parts.append(_const_block(b, d, result))
            parts.append(result)
            if a > 0:
                parts.append(_const_block(a, d, result))
            result = NBXTensor.cat(parts, dim=d)
        return result.contiguous()

    return output


def pad_wrapper(x, pad_list, mode: str = "constant",
                value: float = 0.0) :
    """F.pad via Triton — routes by mode + padded-dim count (len(pad_list)//2).

    `aten::pad` is the generic functional pad; PyTorch carries the mode as an
    arg ('constant'/'reflect'/'replicate') rather than lowering to the
    `aten::reflection_padNd` op, so the per-rank kernels must be reached from
    here too. The reflection/replication wrappers below are R33-pure
    (flip + narrow + NBXTensor.cat — no new kernel), so reflect/replicate are
    real Triton paths, not torch fallbacks. The padded-dim count is the last-
    dim-first PyTorch convention: 2 entries → 1 dim, 4 → 2 dims.
    """
    if mode == "constant":
        return constant_pad_nd_wrapper(x, pad_list, value)
    ndim_padded = len(pad_list) // 2
    if mode in ("reflect", "reflection"):
        if ndim_padded == 1:
            return reflection_pad1d_wrapper(x, pad_list)
        if ndim_padded == 2:
            return reflection_pad2d_wrapper(x, pad_list)
        if ndim_padded == 3:
            return reflection_pad3d_wrapper(x, pad_list)
        raise RuntimeError(
            f"[--triton mode] reflection pad over {ndim_padded} dims has no "
            f"Triton wrapper (1D/2D/3D only). pad_list={pad_list}."
        )
    if mode in ("replicate", "replication"):
        if ndim_padded == 1:
            return replication_pad1d_wrapper(x, pad_list)
        if ndim_padded == 2:
            return replication_pad2d_wrapper(x, pad_list)
        if ndim_padded == 3:
            return replication_pad3d_wrapper(x, pad_list)
        raise RuntimeError(
            f"[--triton mode] replication pad over {ndim_padded} dims has no "
            f"Triton wrapper (1D/2D/3D only). pad_list={pad_list}."
        )
    raise RuntimeError(
        f"[--triton mode] pad mode='{mode}' not implemented. "
        f"Supported: constant, reflect (1D/2D), replicate (1D/2D)."
    )


def reflection_pad1d_wrapper(x, pad_list) :
    """Reflection pad 1D — `F.pad(mode='reflect')` on the last dim.

    R33-pure: narrow + flip + cat, the width-axis case of
    reflection_pad2d_wrapper (no new kernel — flip is a Triton kernel, cat and
    narrow are Triton-backed). Reflection excludes the edge element (PyTorch
    `reflect`): a left pad of p mirrors elements [1 .. p] reversed, a right pad
    mirrors [L-1-p .. L-2] reversed. Used by vocoder conv/iSTFT pre-pad.
    """
    pad_list = [int(v) for v in pad_list]
    left, right = (pad_list + [0, 0])[:2]
    out = x.contiguous()
    L = out.shape[-1]
    parts = []
    if left > 0:
        parts.append(flip_wrapper(out.narrow(-1, 1, left), [-1]))
    parts.append(out)
    if right > 0:
        parts.append(flip_wrapper(out.narrow(-1, L - 1 - right, right), [-1]))
    if len(parts) > 1:
        out = NBXTensor.cat(parts, dim=-1).contiguous()
    return out


def reflection_pad2d_wrapper(x, pad_list) :
    """Reflection pad 2D — `F.pad(mode='reflect')` semantics.

    R33-pure: decomposed into `narrow` + `flip` + `NBXTensor.cat`,
    no new Triton kernel (flip is already a Triton kernel; cat and
    narrow are Triton-backed). Reflection excludes the edge pixel
    itself (PyTorch `reflect`, not `replicate`/`symmetric`): a left
    pad of `p` mirrors columns `[1 .. p]` reversed, a right pad
    mirrors columns `[W-1-p .. W-2]` reversed. Same on the H axis,
    applied to the width-padded intermediate.

    Used by Swin-attention SR models (Swin2SR / SwinIR / HAT) for
    the input feature reflection pad.
    """
    pad_list = [int(v) for v in pad_list]
    # torch convention: last-dim-first → [W_left, W_right, H_top, H_bot]
    left, right, top, bottom = (pad_list + [0, 0, 0, 0])[:4]
    out = x.contiguous()

    # --- Width axis (dim=-1) ---
    W = out.shape[-1]
    parts = []
    if left > 0:
        parts.append(flip_wrapper(out.narrow(-1, 1, left), [-1]))
    parts.append(out)
    if right > 0:
        parts.append(flip_wrapper(out.narrow(-1, W - 1 - right, right), [-1]))
    if len(parts) > 1:
        out = NBXTensor.cat(parts, dim=-1).contiguous()

    # --- Height axis (dim=-2), on the width-padded tensor ---
    H = out.shape[-2]
    parts = []
    if top > 0:
        parts.append(flip_wrapper(out.narrow(-2, 1, top), [-2]))
    parts.append(out)
    if bottom > 0:
        parts.append(flip_wrapper(out.narrow(-2, H - 1 - bottom, bottom), [-2]))
    if len(parts) > 1:
        out = NBXTensor.cat(parts, dim=-2).contiguous()

    return out


def _reflect_pad_axis(out, axis, before, after):
    """Reflect (mirror, excluding the edge element — PyTorch `reflect`)
    `before`/`after` slices along `axis` and cat. R33-pure: narrow + flip +
    NBXTensor.cat (no new kernel), the per-axis core of reflection_pad2d_wrapper.
    """
    if before <= 0 and after <= 0:
        return out
    L = out.shape[axis]
    parts = []
    if before > 0:
        parts.append(flip_wrapper(out.narrow(axis, 1, before), [axis]))
    parts.append(out)
    if after > 0:
        parts.append(flip_wrapper(out.narrow(axis, L - 1 - after, after), [axis]))
    return NBXTensor.cat(parts, dim=axis).contiguous()


def reflection_pad3d_wrapper(x, pad_list) :
    """Reflection pad 3D (last three dims) — F.pad(mode='reflect'). R33-pure
    narrow+flip+cat (no new kernel). Used by the SANA-Video / 5D-video VAE
    reflection pad on [B, C, D, H, W]; previously mis-routed to the 2D wrapper
    (correct only when the depth pad is 0)."""
    pad_list = [int(v) for v in pad_list]
    # torch convention: [W_left, W_right, H_top, H_bot, D_front, D_back]
    wl, wr, ht, hb, df, db = (pad_list + [0, 0, 0, 0, 0, 0])[:6]
    out = _reflect_pad_axis(x.contiguous(), -1, wl, wr)
    out = _reflect_pad_axis(out, -2, ht, hb)
    return _reflect_pad_axis(out, -3, df, db)


def _replicate_pad_axis(out, axis, before, after):
    """Replicate the edge slice `before`/`after` times along `axis` and cat.

    R33-pure: narrow + expand + NBXTensor.cat — no new Triton kernel (same
    decomposition discipline as reflection_pad2d_wrapper). Replication repeats
    the EDGE element (PyTorch `replicate`), unlike reflection which mirrors the
    interior — so a left pad of p is the first slice broadcast p times, a right
    pad of p is the last slice broadcast p times.
    """
    if before <= 0 and after <= 0:
        return out
    L = out.shape[axis]
    parts = []
    if before > 0:
        shp = list(out.shape); shp[axis] = before
        parts.append(out.narrow(axis, 0, 1).expand(*shp).contiguous())
    parts.append(out)
    if after > 0:
        shp = list(out.shape); shp[axis] = after
        parts.append(out.narrow(axis, L - 1, 1).expand(*shp).contiguous())
    return NBXTensor.cat(parts, dim=axis).contiguous()


def replication_pad1d_wrapper(x, pad_list) :
    """Replication pad 1D (last dim) — F.pad(mode='replicate'). R33-pure
    narrow+expand+cat (no new kernel)."""
    pad_list = [int(v) for v in pad_list]
    left, right = (pad_list + [0, 0])[:2]
    return _replicate_pad_axis(x.contiguous(), -1, left, right)


def replication_pad2d_wrapper(x, pad_list) :
    """Replication pad 2D (last two dims) — F.pad(mode='replicate'). R33-pure."""
    pad_list = [int(v) for v in pad_list]
    # torch convention: last-dim-first → [W_left, W_right, H_top, H_bot]
    wl, wr, ht, hb = (pad_list + [0, 0, 0, 0])[:4]
    out = _replicate_pad_axis(x.contiguous(), -1, wl, wr)
    return _replicate_pad_axis(out, -2, ht, hb)


def replication_pad3d_wrapper(x, pad_list) :
    """Replication pad 3D (last three dims) — F.pad(mode='replicate'). R33-pure
    narrow+expand+cat (no new kernel). Used by the Mochi / 5D-video VAE decoder
    edge pad on [B, C, D, H, W]."""
    pad_list = [int(v) for v in pad_list]
    # torch convention: [W_left, W_right, H_top, H_bot, D_front, D_back]
    wl, wr, ht, hb, df, db = (pad_list + [0, 0, 0, 0, 0, 0])[:6]
    out = _replicate_pad_axis(x.contiguous(), -1, wl, wr)
    out = _replicate_pad_axis(out, -2, ht, hb)
    return _replicate_pad_axis(out, -3, df, db)


# ===========================================================================
# FOLD / UNFOLD_BACKWARD — compute ops
# ===========================================================================

def fold_wrapper(*args, **kwargs) :
    """fold — NOT YET IMPLEMENTED in Triton."""
    raise RuntimeError(
        "[--triton mode] fold (col2im) has no Triton kernel. "
        "Implement in src/neurobrix/kernels/ops/fold_op.py."
    )


def unfold_backward_wrapper(grad_in, input_sizes, dim, size, step) :
    """unfold_backward (overlap-add): scatter-add the unfolded frames
    [..lead.., N_frames, size] back into the original signal [..lead.., L] (L on
    `dim`), summing overlapping positions. This is the iSTFT overlap-add.
    R33-pure (one @triton.jit scatter-add kernel)."""
    from .ops.unfold_op import unfold_backward_1d_kernel
    grad_in = _ensure_cuda(grad_in).contiguous()
    if isinstance(dim, (list, tuple)):
        dim = dim[0]
    if isinstance(size, (list, tuple)):
        size = size[0]
    if isinstance(step, (list, tuple)):
        step = step[0]
    out_shape = list(input_sizes)
    dim = dim % len(out_shape)
    L = out_shape[dim]
    # grad_in: [*out_shape[:dim], N_frames, size] (the unfold replaced dim by
    # (N_frames, size)). Collapse the leading dims into a batch.
    gshape = list(grad_in.shape)
    N_frames = gshape[dim]
    sz = gshape[-1]
    B = 1
    for d in out_shape[:dim]:
        B *= d
    g2 = grad_in.reshape(B, N_frames, sz)
    out = NBXTensor.zeros((B, L), dtype=grad_in.dtype, device=grad_in.device)
    _set_device(g2)
    max_overlap = (sz // step) + 2
    BLOCK = 256
    grid = (B, triton.cdiv(L, BLOCK))
    unfold_backward_1d_kernel[grid](
        g2, out, N_frames, sz, step, L,
        MAX_OVERLAP=max_overlap, BLOCK_SIZE=BLOCK)
    return out.reshape(*out_shape)


def im2col_wrapper(x, kernel_size, dilation, padding, stride):
    """aten::im2col / nn.Unfold — sliding local blocks of an NCHW tensor.

    x [N, C, IH, IW] -> [N, C*KH*KW, L], L = OH*OW, in PyTorch's channel-major
    column order (c*KH*KW + kh*KW + kw). R33-pure: one @triton.jit kernel that
    reads the padding halo via boundary-masked tl.load (no F.pad). Drives HAT's
    OCAB overlapping-window cross-attention (P-TRITON-IM2COL-KERNEL).
    """
    from .ops.im2col import im2col_kernel

    def _pair(v):
        return tuple(v) if isinstance(v, (list, tuple)) else (v, v)

    x = _ensure_cuda(x).contiguous()
    N, C, IH, IW = x.shape
    KH, KW = _pair(kernel_size)
    dil_h, dil_w = _pair(dilation)
    pad_h, pad_w = _pair(padding)
    stride_h, stride_w = _pair(stride)
    OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    L = OH * OW
    out = NBXTensor.empty((N, C * KH * KW, L), dtype=x.dtype, device=x.device)
    in_sn, in_sc, in_sh, in_sw = C * IH * IW, IH * IW, IW, 1
    out_sn, out_sc, out_sl = C * KH * KW * L, L, 1
    BLOCK_L = min(1024, triton.next_power_of_2(L))
    grid = (N * C * KH * KW,)
    _set_device(x)
    im2col_kernel[grid](
        x, out, C, IH, IW, OW, L, KH, KW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        in_sn, in_sc, in_sh, in_sw, out_sn, out_sc, out_sl,
        BLOCK_L=BLOCK_L)
    return out


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
    tril_mask = tril_wrapper(ones, diagonal=offset)
    inverted = eq(tril_mask, 0.0)
    bias = masked_fill(base, inverted, float('-inf'))
    _causal_bias_cache[key] = bias
    return bias


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _sdpa_math_scores_budget_bytes() -> int:
    """Pre-Ampere scores-tensor budget for routing SDPA to the
    deterministic `_math_attention` path (P-TRITON-MOE-DETERMINISM-
    RESIDUAL, option B). Data-driven from
    config/vendors/<vendor>/<arch>.yml `memory.sdpa_math_max_scores_bytes`
    (R10/R23 — no hardcoded hardware param, no driver query in the hot
    path; the value is read from the YAML keyed by the detected
    profile). Returns 0 (→ pow2 head_dim never routes on size, i.e.
    flash unchanged) when the profile or the key is unavailable, so an
    unconfigured arch never silently changes the attention path."""
    prof = get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return 0
    try:
        from neurobrix.core.config.loader import get_vendor_config
        # The vendor-config key is the GPU brand ("nvidia"/"amd"), i.e.
        # the DEVICE brand — NOT PrismProfile.vendor (that is the
        # machine maker, e.g. "dell", and would mis-resolve the path
        # config/vendors/<vendor>/<arch>.yml).
        _brand = devices[0].brand
        _vendor = getattr(_brand, "value", _brand)
        cfg = get_vendor_config(_vendor, devices[0].architecture)
        return int(cfg.get("memory", {}).get(
            "sdpa_math_max_scores_bytes", 0))
    except Exception:
        return 0


def _math_attention(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """Math-decomposed attention — deterministic for any head_dim.

    Used for non-power-of-2 head_dim (PixArt 72, Sana 112) where the
    flash kernel's masked-load path is non-deterministic on Volta SIMT.

    Shapes: q [B, H, T_q, D], k/v [B, H_k, T_k, D]. Returns [B, H, T_q, D].

    Algorithm:
        scores = Q @ K^T * scale       (bmm)
        scores = scores + mask         (additive bias when needed)
        p = softmax(scores, dim=-1)    (per-row softmax)
        out = p @ V                    (bmm)

    GQA: when nheads != nheads_k, K and V are repeated to match Q heads
    via expand+reshape (zero-copy view). Same pattern flash uses
    internally via GQA_GROUPS, but materialized here since math path
    has no constexpr GQA hook.
    """
    import math as _math
    B, H, T_q, D = q.shape
    H_k = k.shape[1]
    T_k = k.shape[2]

    if scale is None:
        scale = 1.0 / _math.sqrt(D)

    # GQA broadcast — repeat K/V heads to match Q. For plain MHA
    # (H == H_k) this is a no-op view.
    if H != H_k:
        groups = H // H_k
        assert H_k * groups == H
        # [B, H_k, T_k, D] -> [B, H_k, 1, T_k, D] -> [B, H_k, groups, T_k, D] -> [B, H, T_k, D]
        k = k.unsqueeze(2).expand(B, H_k, groups, T_k, D).reshape(B, H, T_k, D).contiguous()
        v = v.unsqueeze(2).expand(B, H_k, groups, T_k, D).reshape(B, H, T_k, D).contiguous()

    # Reshape to 3D for bmm
    q_3d = q.reshape(B * H, T_q, D)
    # Transpose K's last two dims so bmm computes Q @ K^T
    k_3d_t = k.transpose(-2, -1).contiguous().reshape(B * H, D, T_k)
    v_3d = v.reshape(B * H, T_k, D)

    # scores = q @ k^T (bmm returns fp32 on V100 via force_fp32)
    scores = bmm(q_3d, k_3d_t)  # [B*H, T_q, T_k]
    scores = mul(scores, float(scale))

    # Causal/explicit mask handling — additive bias on scores.
    # Built via existing tril-based path; skipped for non-causal no-mask.
    if is_causal and attn_mask is None:
        device_idx = q._device_idx if hasattr(q, '_device_idx') else 0
        bias = _get_causal_bias(device_idx, T_q, T_k, scores.nbx_dtype)
        # bias shape [T_q, T_k] -> broadcast to [B*H, T_q, T_k]
        scores = add(scores, bias.unsqueeze(0).expand(B * H, T_q, T_k))
    elif attn_mask is not None:
        # attn_mask may be [T_q, T_k], [B, T_q, T_k], or [B, H, T_q, T_k].
        if attn_mask.ndim == 2:
            mask_3d = attn_mask.unsqueeze(0).expand(B * H, T_q, T_k)
        elif attn_mask.ndim == 3:
            if attn_mask.shape[0] == 1 and B > 1:
                attn_mask = attn_mask.expand(B, T_q, T_k)
            mask_3d = attn_mask.unsqueeze(1).expand(B, H, T_q, T_k).reshape(B * H, T_q, T_k)
        elif attn_mask.ndim == 4:
            mask_3d = attn_mask.expand(B, H, T_q, T_k).reshape(B * H, T_q, T_k)
        else:
            raise RuntimeError(f"unsupported attn_mask.ndim={attn_mask.ndim}")
        if mask_3d.nbx_dtype != scores.nbx_dtype:
            mask_3d = mask_3d.to(scores.nbx_dtype)
        scores = add(scores, mask_3d.contiguous())

    # softmax along K dim — kernel handles fp16/fp32 internally,
    # returns same dtype as input. Score is fp32 from bmm so softmax
    # output is fp32.
    p = softmax(scores, dim=-1)
    # Fully-masked-row guard — parity with PyTorch's fused SDPA backends
    # (memory-efficient / flash), which NeuroBrix compiled mode uses as the
    # numerical oracle. A query row whose every key is masked (additive bias
    # = -inf on ALL keys — e.g. the Open-Sora HunyuanVAE mid-block attention
    # passes an all-(-inf) bias) makes softmax(all -inf) = NaN in this
    # plain-softmax math path, whereas the efficient backend emits 0 for such
    # a row. Scrub NaN -> 0 so triton math attention matches compiled and the
    # NaN does not blacken the whole VAE. In this fp32 score path (bmm is
    # fp32-accumulated on V100, softmax of any finite row is finite) a NaN can
    # ONLY originate from a fully-masked row, so this is inert — bit-identical
    # — for every attention that has no fully-masked row. Mirror of the flash
    # kernel's fully-masked-row guard in kernels/ops/flash_attention.py (the
    # two SDPA paths, math and flash, stay symmetric under R30).
    p = nan_to_num_wrapper(p, nan=0.0)
    # Cast back to value dtype for the second bmm
    if p.nbx_dtype != v_3d.nbx_dtype:
        p = p.to(v_3d.nbx_dtype)

    # out = p @ v (bmm again returns fp32 on V100)
    out_3d = bmm(p, v_3d)  # [B*H, T_q, D]
    if out_3d.nbx_dtype != q.nbx_dtype:
        out_3d = out_3d.to(q.nbx_dtype)

    return out_3d.reshape(B, H, T_q, D)


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

    # When is_causal=True with an explicit attn_mask, drop the mask —
    # the causal bias materializer below covers the causal semantics
    # uniformly through memory.
    if is_causal and attn_mask is not None:
        attn_mask = None
    # The former heuristic "2D square mask with seqlen_q == seqlen_k →
    # assume causal tril" has been removed. The kernel is now strictly
    # bias-driven (no IS_CAUSAL constexpr path), and any causal pattern
    # arrives via memory through the cache below.

    # Default scale
    if scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)
    else:
        softmax_scale = float(scale) if not isinstance(scale, float) else scale

    # GQA is handled inside the kernel via GQA_GROUPS — no materialization.
    # off_h_kv = off_h // GQA_GROUPS picks the right K/V head for each Q head,
    # so K/V keep their native (b, nheads_k, s, d) layout. For plain MHA
    # (nheads == nheads_k), gqa_groups=1 and off_h_kv == off_h (zero cost).
    # Avoids the per-call expand+reshape+contiguous that copied K and V in
    # full every decode step (~7 MB/step on TinyLlama, >>100 MB at 2k ctx).
    gqa_groups = nheads // nheads_k
    assert nheads_k * gqa_groups == nheads, (
        f"nheads ({nheads}) must be a multiple of nheads_k ({nheads_k})")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # tl.dot on V100 TensorCores requires matching operand dtypes; our
    # conditional fp32-output matmul (M<=4 decode) can leave q/k/v mixed
    # (Sana diffusion hit fp32 Q vs fp16 K here). If they disagree, cast
    # to fp32. For the common LLM case where all three match, this is a
    # no-op — zero overhead.
    if not (q.dtype == k.dtype == v.dtype):
        q, k, v = q.to(NBXDtype.float32), k.to(NBXDtype.float32), v.to(NBXDtype.float32)

    # Deterministic-attention routing (P-TRITON-MOE-DETERMINISM-RESIDUAL,
    # Hocine scope decision = option B: hardware + memory-budget, ZERO
    # model/family knowledge — R34 strict; the SDPA wrapper is a
    # universal runtime primitive).
    #
    # The Dao-AILab flash Triton kernel is non-deterministic on
    # pre-Ampere SIMT (Volta sm_70 / Turing sm_75):
    #   - documented for the masked-load path (EVEN_HEADDIM=False,
    #     head_dim < BLOCK_HEADDIM, e.g. PixArt 72 / Sana 112): "race
    #     conditions on non-64/128 head dimensions"; 5 consecutive
    #     calls yield 5 different outputs (DiT h=126,127 banding).
    #   - EMPIRICALLY confirmed for power-of-2 head_dim=128 too
    #     (Qwen3-30B-A3B GQA/fp32: §5.8 op-by-op full-hash differential
    #     — pos 0..74 byte-identical across 3 runs, first divergence
    #     SDPA::0 with bit-identical inputs; 3 runs → 3 distinct
    #     outputs). So `_is_power_of_2(headdim)` was an INCOMPLETE
    #     guard — pow2 hd=128 is not safe on Volta.
    #
    # `_math_attention` (Q@K^T→softmax→@V; plain bmm+softmax, no
    # online-softmax accumulation, no MMA reorder across K-tiles) is
    # deterministic by construction. Counterfactual proof:
    # NBX_FORCE_MATH_ATTENTION=1 → Qwen3-30B::triton 3 runs
    # BYTE-IDENTICAL. Cost: a [B*H,Tq,Tk] scores tensor — negligible
    # for LLM small-T, ~1 GiB+ for image-diffusion large-T.
    #
    # Routing (universal technical signals only):
    #   - Ampere+ (sm_80+, has_native_bf16): flash unchanged — flash is
    #     deterministic there (R23: zero non-Volta regression).
    #   - non-pow2 head_dim: _math_attention, UNCONDITIONAL on ALL
    #     hardware — EXACT prior behaviour preserved (the old guard was
    #     `if not _is_power_of_2(headdim): return _math_attention`,
    #     hardware-independent). R23 strict: zero non-Volta regression,
    #     PixArt hd=72 / Sana hd=112 path byte-unchanged everywhere.
    #   - pow2 head_dim + scores_bytes ≤ vendor-yml budget:
    #     _math_attention — NEW (Qwen3 LLM fix), the ONLY added
    #     routing. The hardware gate IS the data-driven per-arch
    #     budget: only config/vendors/nvidia/volta.yml defines
    #     `memory.sdpa_math_max_scores_bytes`; ampere.yml / hopper.yml
    #     / cdna.yml have no such key → budget 0 → scores_bytes ≤ 0 is
    #     False → flash unchanged (R23: flash is deterministic on
    #     Ampere+; zero non-Volta regression, no ambiguous capability
    #     proxy). Budget-gated so a pow2 + large-T tensor never OOMs:
    #     it stays on flash, residual non-determinism documented +
    #     trackable via NBX_OP_FINGERPRINT.
    # NBX_FORCE_MATH_ATTENTION=1 — retained diagnostic (same class as
    # NBX_DISABLE_AUTOTUNE / NBX_DUMP_TIDS): force the deterministic
    # path regardless of hardware/headdim.
    import os as _os_fma
    _use_math = _os_fma.environ.get("NBX_FORCE_MATH_ATTENTION") == "1"
    if not _use_math:
        _scores_bytes = batch * nheads * seqlen_q * seqlen_k * 4
        if not _is_power_of_2(headdim):
            # Non-pow2 head_dim: math WHILE the fp32 scores tensor fits a
            # sane memory bound; FLASH beyond it. The flash kernel fully
            # supports non-pow2 head dims (BLOCK_HEADDIM = next_power_of_2 +
            # masked d-axis loads `offs_d < headdim, other=0.0` — the Dao
            # EVEN_HEADDIM pattern), so the old UNCONDITIONAL math guard was
            # conservatism, and at video-native scale it is fatal: Allegro
            # 720p×88f = 79200 tokens, hd=96 → math scores [2,24,79200²]
            # fp32 = 1.204 TB (guaranteed OOM — no behavior existed to
            # regress on that branch). Bound = the per-arch vendor-yml
            # budget where defined (Volta 128 MB), else a universal 2 GiB
            # sanity floor — PixArt hd=72 / Sana hd=112 image-scale scores
            # stay ≤ the floor on every arch → their math path is
            # byte-unchanged (R23).
            _bound = _sdpa_math_scores_budget_bytes() or (2 << 30)
            _use_math = _scores_bytes <= _bound
        else:
            # _math_attention materialises an fp32 [B*H,Tq,Tk] scores
            # tensor (bmm returns fp32 on V100) — 4 bytes/elem is the
            # true memory cost, independent of q's dtype. Budget is 0
            # on non-Volta (no yml key) → this never fires there.
            _use_math = _scores_bytes <= _sdpa_math_scores_budget_bytes()
    if _os_fma.environ.get("NBX_SDPA_ROUTE_DIAG") == "1" and not getattr(
            scaled_dot_product_attention_wrapper, "_route_diag_done", False):
        scaled_dot_product_attention_wrapper._route_diag_done = True
        _pr = get_hardware_profile()
        _dv = getattr(_pr, "devices", None) if _pr is not None else None
        print(f"[NBX_SDPA_ROUTE_DIAG] hd={headdim} pow2={_is_power_of_2(headdim)} "
              f"B={batch} H={nheads} Tq={seqlen_q} Tk={seqlen_k} "
              f"scores_bytes={batch*nheads*seqlen_q*seqlen_k*4} "
              f"budget={_sdpa_math_scores_budget_bytes()} "
              f"profile={'None' if _pr is None else getattr(_pr,'vendor','?')+'/'+(getattr(_dv[0],'architecture','?') if _dv else '?')} "
              f"use_math={_use_math}", flush=True)
    if _use_math:
        return _math_attention(q, k, v, attn_mask=attn_mask,
                                is_causal=is_causal, scale=softmax_scale)

    # Adaptive BLOCK_M (Phase 1): decode-path seqlen_q is typically 1–4. Using
    # BLOCK_M=128 wastes >99% of the Q tile. 16 is the floor enforced by
    # tl.dot (TensorCore tile minimum) — below that the kernel crashes.
    # Block sizes — adapted from Dao-AILab defaults.
    # Reduce BLOCK_M for large headdim to stay within shared memory limits.
    # V100 has 96KB shared memory. With headdim=256: 128×256×2×3 = 192KB > 96KB.
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    if seqlen_q <= 16:
        BLOCK_M = 16
        BLOCK_N = 64 if BLOCK_HEADDIM < 128 else (64 if BLOCK_HEADDIM < 256 else 32)
    elif BLOCK_HEADDIM >= 512:
        # PixArt VAE on V100. (32,32) for h=512 needs 131KB SMEM > 96KB
        # opt-in. (16,16) measured 49KB → fits with margin. Inlined for
        # zero Python overhead vs Layer 6's data-driven approach.
        BLOCK_M = 16
        BLOCK_N = 16
    elif BLOCK_HEADDIM >= 256:
        BLOCK_M = 32
        BLOCK_N = 32
    elif BLOCK_HEADDIM >= 128:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        # head_dim < 128 (the common case: hd=64 LLMs, hd=64 video DiTs).
        # BLOCK_M=128 is the Dao A100 default; on pre-Ampere (Volta sm_70 /
        # Turing sm_75: 96 KB SMEM, fewer registers) a 128-row Q tile spills
        # registers and serialises — arch-gate to a smaller tile. Data-driven
        # via the hardware capability flag (same single source of truth as
        # the dtype-protection path); Ampere+ keeps 128. NBX_FLASH_BLOCK_M_VOLTA
        # overrides the Volta value (diagnostic / per-arch tuning).
        if not _NBX_HAS_NATIVE_BF16:
            # Measured on V100 (sm_70), seq=8192 hd=64: BLOCK_M 128 -> 9.2 s,
            # 64 -> 6.1 s, 32 -> 0.42 s. The 128-row Q tile spills registers to
            # local memory (DRAM); 64 cuts that ~1.5x and stays NUMERICALLY
            # CORRECT (CogVideoX-2b/5b triton produce a coherent fox). 32 is
            # ~14x faster but WRONG — it diverges (max|diff| 0.035 vs torch
            # SDPA, vs 0.0002 at 64/128) and the error compounds across the
            # diffusion loop into pure noise. So 64 is the floor of
            # correctness on Volta. NBX_FLASH_BLOCK_M_VOLTA overrides (do not
            # set below 64 for accuracy-sensitive / iterative models).
            BLOCK_M = int(os.environ.get("NBX_FLASH_BLOCK_M_VOLTA", "64"))
        else:
            BLOCK_M = 128
        BLOCK_N = 64

    # Output allocation. seqlen_q_rounded must align with actual BLOCK_M.
    o = NBXTensor.empty_like(q)
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')
    tmp = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')

    # Bias handling — always memory-resident.
    # The kernel only accepts BIAS_TYPE in {"vector", "matrix"} and the
    # IS_CAUSAL constexpr is gone. Every tensor reaching tl.dot must
    # arrive via tl.load to guarantee Triton compiles a single MMA path.
    # Three sub-cases below all converge to BIAS_TYPE="matrix" with a
    # memory-resident bias.
    device_idx = q._device_idx if hasattr(q, '_device_idx') else 0
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(
                batch, nheads, seqlen_q, seqlen_k
            )
        elif attn_mask.ndim == 3:
            if attn_mask.shape[0] == 1 and batch > 1:
                attn_mask = attn_mask.expand(batch, seqlen_q, seqlen_k)
            attn_mask = attn_mask.unsqueeze(1).expand(
                batch, nheads, seqlen_q, seqlen_k
            )
        elif attn_mask.ndim == 4:
            attn_mask = attn_mask.expand(batch, nheads, seqlen_q, seqlen_k)
        # Pass the bias as a BROADCAST view: keep stride-0 on every axis the
        # mask is constant over (H and/or the query axis for a key-padding
        # mask). The kernel reads the bias through its strides, so a key-pad
        # mask stays a few-MB [.., Sk] buffer (stride_bm=0) instead of a
        # materialized [B,H,Sq,Sk] matrix — for video self-attention that
        # matrix is tens of GB (Mochi [2,24,~16k,~16k] ≈ 25 GB). The previous
        # .contiguous() on a [B,1,1,Sk] mask kept unit-size H/Sq dims with
        # NON-ZERO contiguous strides (=Sk), which the kernel then indexed
        # PAST at off_h<nheads / offs_m<seqlen_q → out-of-bounds illegal
        # access (Mochi triton "error 700"). The bias is still memory-resident
        # (a real tl.load, bit-equivalent — only the strides change). Only
        # materialize if the key axis is not unit-stride (kernel reads offs_n
        # with implicit stride 1).
        bias = attn_mask if attn_mask.stride(-1) == 1 else attn_mask.contiguous()
    elif is_causal:
        causal_base = _get_causal_bias(device_idx, seqlen_q, seqlen_k, q.dtype)
        bias = causal_base.unsqueeze(0).unsqueeze(0).expand(
            batch, nheads, seqlen_q, seqlen_k
        )
    else:
        zero_base = _get_zero_bias(device_idx, seqlen_q, seqlen_k, q.dtype)
        bias = zero_base.unsqueeze(0).unsqueeze(0).expand(
            batch, nheads, seqlen_q, seqlen_k
        )
    bias_type = "matrix"
    # Causal semantics now live entirely in the materialized bias above —
    # the kernel's IS_CAUSAL path has been removed. Force False here so
    # any caller-provided is_causal flag does not reach the kernel
    # signature (it no longer accepts one anyway, but we keep the local
    # variable for clarity).
    is_causal = False

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    # Launch meta (num_warps / num_stages). Triton auto-selects when not
    # given; on Volta the auto-pick for a small Q tile under-parallelises the
    # softmax reduction (the BLOCK_M=32 wrongness probe). NBX_FLASH_NUM_WARPS
    # / NBX_FLASH_NUM_STAGES let us pin them for diagnosis / per-arch tuning.
    _flash_launch_meta = {}
    _fnw = os.environ.get("NBX_FLASH_NUM_WARPS")
    if _fnw:
        _flash_launch_meta["num_warps"] = int(_fnw)
    _fns = os.environ.get("NBX_FLASH_NUM_STAGES")
    if _fns:
        _flash_launch_meta["num_stages"] = int(_fns)

    # Ensure CUDA runtime is on the correct device before kernel launch
    _set_device(q)

    flash_attention_forward_kernel[grid](
        q, k, v, bias,
        o, lse, tmp,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0), bias.stride(1), bias.stride(-2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
        seqlen_q // 32, seqlen_k // 32,  # cache keys
        BIAS_TYPE=bias_type,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GQA_GROUPS=gqa_groups,
        **_flash_launch_meta,
    )
    # Fully-masked-row guard at the SDPA-op level — parity with PyTorch's fused
    # SDPA backends (which NeuroBrix compiled mode uses as the numerical oracle).
    # A query row whose every key is masked (an additive attn_mask of -inf on ALL
    # keys — e.g. the Open-Sora-v2 VAE mid-block self-attention) has no finite
    # score, so the online-softmax flash kernel emits NaN/-inf for that row while
    # the efficient backend emits 0. Scrub it -> 0 HERE at the wrapper, NOT inside
    # the kernel: editing the @triton.jit body perturbs Triton register
    # allocation and changes the bit-exact result of EVERY attention (measured:
    # TinyLlama greedy output drifted with an in-kernel guard, byte-identical with
    # this one). Only an explicit additive mask can fully-mask a row (causal /
    # no-mask attention never does), so gate on attn_mask: inert (skipped) for
    # causal + plain self-attention, and a value-identity (nan_to_num on a finite
    # tensor) for any masked attention with no fully-masked row. posinf/neginf ->
    # 0 so a fully-masked row that resolves to -inf rather than NaN is also
    # zeroed. Behaviour-symmetric (R30) with the math path guard in
    # _math_attention.
    if attn_mask is not None:
        o = nan_to_num_wrapper(o, nan=0.0, posinf=0.0, neginf=0.0)
    return o


def angle_wrapper(x) :
    """angle(complex_tensor) → atan2(imag, real).

    For real inputs: 0 where x >= 0, pi where x < 0.
    Uses Triton atan2 via libdevice.
    """
    x = _ensure_cuda(x)

    if x.is_complex():
        # Complex: atan2(imag, real) element-wise (libdevice, fp32).
        from .ops.atan2 import atan2_kernel
        real = x.real.contiguous()
        imag = x.imag.contiguous()
        out = NBXTensor.empty(tuple(real.shape), dtype=NBXDtype.float32,
                              device=real.device)
        _set_device(real)
        n = real.numel()
        atan2_kernel[_1d_grid(n)](imag, real, out, n, BLOCK_SIZE=_EW_BLOCK)
        return out
    else:
        # Real input: angle is 0 where x >= 0, pi where x < 0. Not on the
        # complex-FFT path (kept minimal: zeros for the x>=0 common case).
        return NBXTensor.zeros_like(x)


# ===========================================================================
# INTERPOLATION — upsample_linear1d via bilinear2d
# ===========================================================================

def upsample_linear1d_wrapper(x, output_size, align_corners: bool = False,
                              scales=None) :
    """upsample_linear1d via bilinear2d: [N,C,L] → unsqueeze → [N,C,1,L] → bilinear2d → squeeze.

    Like upsample_nearest2d_wrapper: the traced graph stores BOTH the concrete
    trace output_size AND the scale factor. When the runtime input length differs
    from trace (variable-length audio decoders — Kokoro iSTFT source generator
    runs at 124 vs the 256 trace), the baked output_size is stale; PyTorch
    recomputes from the live input when a scale is present, so we do the same:
    out_l = round(runtime_input_len * scale). Bit-identical when trace == runtime
    (input_len * scale == baked output_size). Without this, a linear upsample
    would desync from its sibling nearest upsample (which already scale-recomputes).
    """
    x = _ensure_cuda(x).contiguous()
    x_4d = x.unsqueeze(2)  # [N, C, 1, L]

    scale_val = None
    if scales is not None:
        scale_val = scales[0] if isinstance(scales, (list, tuple)) else float(scales)

    if scale_val is not None:
        out_l = int(round(x.shape[-1] * scale_val))
    elif output_size is not None:
        out_l = output_size[0] if isinstance(output_size, (list, tuple)) else int(output_size)
    else:
        out_l = None
    output_size_2d = [1, out_l] if out_l is not None else None

    # Pass output_size only; bilinear2d derives its interpolation scale from the
    # (input, output) shapes + align_corners.
    result_4d = upsample_bilinear2d_wrapper(x_4d, output_size_2d, align_corners)
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
    _set_device(x_real)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)

    # Butterfly stages
    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        _set_device(temp_real)
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
    _set_device(x_real)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)

    # Inverse butterfly stages
    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        _set_device(temp_real)
        ifft_stage_kernel[(N // 2,)](temp_real, temp_imag, N, stage)

    # Scale by 1/N — simple element-wise multiply
    temp_real.mul_(1.0 / N)
    temp_imag.mul_(1.0 / N)

    return temp_real, temp_imag


def _dft_rfft_matrices(N, N_bins, ref):
    """cos[k,n] and -sin[k,n] rfft basis matrices [N_bins, N] (R33-pure kernel).
    Allocated on `ref`'s device."""
    import math
    from .ops.dft import dft_rfft_matrix_kernel
    _set_device(ref)
    cos_mat = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    nsin_mat = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    total = N_bins * N
    dft_rfft_matrix_kernel[_1d_grid(total)](
        cos_mat, nsin_mat, N_bins, N, 2.0 * math.pi / N, BLOCK_SIZE=_EW_BLOCK)
    return cos_mat, nsin_mat


def _idft_c2r_matrices(N, N_bins, ref):
    """irfft basis matrices [N_bins, N]: real_out = X_real @ C + X_imag @ S."""
    import math
    from .ops.dft import idft_c2r_matrix_kernel
    _set_device(ref)
    C = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    S = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    total = N_bins * N
    idft_c2r_matrix_kernel[_1d_grid(total)](
        C, S, N_bins, N, 2.0 * math.pi / N, 1.0 / N,
        1 if (N % 2 == 0) else 0, BLOCK_SIZE=_EW_BLOCK)
    return C, S


def _dft_r2c(x, onesided):
    """Real frames [.., N] → (X_real, X_imag) [.., N_bins] via DFT matmul."""
    N = x.shape[-1]
    N_bins = N // 2 + 1 if onesided else N
    cos_mat, nsin_mat = _dft_rfft_matrices(N, N_bins, x)
    lead = tuple(x.shape[:-1])
    M = 1
    for d in lead:
        M *= d
    x2 = x.reshape(M, N)
    if x2.nbx_dtype != NBXDtype.float32:
        x2 = x2.to(NBXDtype.float32)
    Xr = matmul_wrapper(x2, cos_mat.transpose(0, 1).contiguous())   # [M, N_bins]
    Xi = matmul_wrapper(x2, nsin_mat.transpose(0, 1).contiguous())  # [M, N_bins]
    return Xr.reshape(*lead, N_bins), Xi.reshape(*lead, N_bins)


def _dft_c2r(x_complex, N):
    """Complex spectrum [.., N_bins] → real [.., N] via inverse-DFT matmul."""
    N_bins = x_complex.shape[-1]
    Xr = x_complex.real.contiguous()
    Xi = x_complex.imag.contiguous()
    C, S = _idft_c2r_matrices(N, N_bins, x_complex)
    lead = tuple(Xr.shape[:-1])
    M = 1
    for d in lead:
        M *= d
    Xr2 = Xr.reshape(M, N_bins)
    Xi2 = Xi.reshape(M, N_bins)
    real_out = add(matmul_wrapper(Xr2, C), matmul_wrapper(Xi2, S))   # [M, N]
    return real_out.reshape(*lead, N)


def fft_r2c_wrapper(x, dim=-1, norm: str = None,
                     onesided: bool = True) :
    """_fft_r2c: real-to-complex rfft → complex64 NBXTensor.

    pow2 N: radix-2 butterfly (_triton_fft_forward), returning the FULL complex
    pair (the old code dropped the imaginary part). non-pow2 N (e.g. iSTFT
    n_fft=20): DFT-via-matmul (the standard small-N path). Model-agnostic — routes
    on N, never on model identity. R33-pure."""
    if isinstance(dim, (list, tuple)):
        dim = dim[0] if dim else -1
    x = _ensure_cuda(x).contiguous()
    moved = (dim != -1 and dim != x.ndim - 1)
    if moved:
        x = x.transpose(dim, -1).contiguous()
    N = x.shape[-1]
    if N > 1 and (N & (N - 1)) == 0:
        # power of 2 → radix-2 butterfly
        x_real = x.to(NBXDtype.float32) if x.nbx_dtype != NBXDtype.float32 else x
        x_imag = NBXTensor.zeros_like(x_real)
        out_real, out_imag = _triton_fft_forward(x_real, x_imag)
        if onesided:
            half = N // 2 + 1
            out_real = out_real.narrow(-1, 0, half).contiguous()
            out_imag = out_imag.narrow(-1, 0, half).contiguous()
    else:
        # non-power-of-2 → DFT-via-matmul
        out_real, out_imag = _dft_r2c(x, onesided)
    result = complex_wrapper(out_real, out_imag)
    if moved:
        result = result.transpose(dim, -1).contiguous()
    return result


def fft_c2r_wrapper(x, dim: int = -1, norm: str = None,
                     last_dim_size: int = None) :
    """_fft_c2r: complex-to-real irfft → real, via inverse-DFT-via-matmul.

    Hermitian-symmetric inverse rfft, correct for any N; the small-N / non-pow2
    case (Kokoro iSTFT n_fft=20) is the primary use. Unified on the DFT-matmul
    path: the prior pow2 radix-2 _triton_ifft path was broken (it splatted the
    NBXTensor.zeros shape and called a non-existent NBXTensor.flip) and was never
    reachable for the audio decoders, so no working path is lost. Model-agnostic,
    R33-pure. fft_r2c keeps its working pow2 butterfly (+ non-pow2 DFT)."""
    # aten::_fft_c2r passes dim as int[] and last_dim_size as a (Sym)int; take ints.
    if isinstance(dim, (list, tuple)):
        dim = dim[0] if dim else -1
    if isinstance(last_dim_size, (list, tuple)):
        last_dim_size = last_dim_size[0] if last_dim_size else None
    x = _ensure_cuda(x)
    moved = (dim != -1 and dim != x.ndim - 1)
    if moved:
        x = x.transpose(dim, -1).contiguous()
    half = x.shape[-1]
    N = last_dim_size if last_dim_size is not None else (half - 1) * 2
    result = _dft_c2r(x, N)
    if moved:
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


def stft_wrapper(x, n_fft, hop_length=None, win_length=None, window=None,
                 normalized=False, onesided=None, return_complex=True,
                 center=False, pad_mode="reflect") :
    """aten::stft — framing + window + rfft → complex spectrogram
    [.., n_fft//2+1, num_frames]. Built on the existing rfft (fft_r2c_wrapper,
    which already covers pow2 and non-pow2 N); framing via NBXTensor.unfold.
    `center` is applied upstream by the traced graph (an explicit pad op), so the
    op-level stft frames without re-padding. Model-agnostic, R33-pure.

    Mirrors torch.stft: real input → onesided complex spectrogram. Used by the
    chatterbox/CosyVoice s3gen vocoder (n_fft=16, hop=4)."""
    if isinstance(n_fft, (list, tuple)):
        n_fft = n_fft[0]
    hop_length = (n_fft // 4) if hop_length in (None, 0) else hop_length
    win_length = n_fft if win_length in (None, 0) else win_length
    onesided = True if onesided is None else bool(onesided)

    x = _ensure_cuda(x).contiguous()
    squeezed = (x.ndim == 1)
    if squeezed:
        x = x.unsqueeze(0)                                 # [1, signal]

    # frame: [..., num_frames, win_length]
    frames = x.unfold(-1, win_length, hop_length).contiguous()
    if window is not None:
        frames = mul(frames, window)                       # broadcast [win_length]
    # centre a short window inside the n_fft buffer (uncommon; here win == n_fft)
    if win_length < n_fft:
        pad_total = n_fft - win_length
        left = pad_total // 2
        frames = constant_pad_nd(frames, [left, pad_total - left], 0.0)

    spec = fft_r2c_wrapper(frames, dim=-1, onesided=onesided)  # [..., num_frames, freq]
    spec = spec.transpose(-1, -2).contiguous()             # [..., freq, num_frames]
    if squeezed:
        spec = spec.squeeze(0)
    return spec


def istft_wrapper(x, n_fft, hop_length=None, win_length=None, window=None,
                  center=True, normalized=False, onesided=None, length=None,
                  return_complex=False) :
    """aten::istft — inverse STFT: irfft each frame + windowed overlap-add +
    window-envelope normalisation. Built on fft_c2r_wrapper (irfft) and
    unfold_backward_wrapper (the overlap-add scatter). Model-agnostic, R33-pure.
    Used by the chatterbox/CosyVoice s3gen vocoder (n_fft=16, hop=4)."""
    if isinstance(n_fft, (list, tuple)):
        n_fft = n_fft[0]
    hop_length = (n_fft // 4) if hop_length in (None, 0) else hop_length
    win_length = n_fft if win_length in (None, 0) else win_length

    x = _ensure_cuda(x)
    squeezed = (x.ndim == 2)
    if squeezed:
        x = x.unsqueeze(0)                                 # [1, freq, frames]
    # [batch, freq, frames] -> [batch, frames, freq]
    xt = x.transpose(-1, -2).contiguous()
    n_frames = xt.shape[-2]
    batch = xt.shape[0]
    # irfft each frame -> real [batch, frames, n_fft]
    frames = fft_c2r_wrapper(xt, dim=-1, last_dim_size=n_fft)
    if window is not None:
        frames = mul(frames, window)                       # synthesis window

    sig_len = (n_frames - 1) * hop_length + win_length
    # overlap-add the windowed frames back into the signal
    signal = unfold_backward_wrapper(frames, [batch, sig_len], 1, win_length, hop_length)
    # window-energy envelope (sum of squared windows at each position) — torch.istft
    # divides by this so overlapping windows reconstruct unit gain.
    if window is not None:
        win_sq = mul(window, window)
    else:
        win_sq = NBXTensor.ones((win_length,), dtype=frames.nbx_dtype, device=frames.device)
    win_sq_b = win_sq.reshape([1, 1, win_length])
    win_sq_frames = mul(NBXTensor.ones((batch, n_frames, win_length),
                                       dtype=frames.nbx_dtype, device=frames.device),
                        win_sq_b)
    win_env = unfold_backward_wrapper(win_sq_frames, [batch, sig_len], 1, win_length, hop_length)
    signal = div(signal, clamp(win_env, 1e-11, None))

    if center:
        pad = n_fft // 2
        signal = signal.narrow(-1, pad, sig_len - 2 * pad)
    if length is not None:
        cur = signal.shape[-1]
        if length <= cur:
            signal = signal.narrow(-1, 0, length)
        else:
            signal = constant_pad_nd(signal, [0, length - cur], 0.0)
    if squeezed:
        signal = signal.squeeze(0)
    return signal


# ===========================================================================
# COMPLEX — tensor creation from real/imag parts
# ===========================================================================

def complex_wrapper(real, imag) :
    """complex(real, imag) → complex64 tensor (interleaved [real, imag] pairs).

    Builds the NBX complex64 representation: stack real/imag on a new last dim
    → [..,N,2] → reinterpret as complex64 [..,N] (NBXTensor.view_as_complex).
    R33-pure (cat + view; no torch)."""
    real = _ensure_cuda(real)
    imag = _ensure_cuda(imag)
    if real.nbx_dtype != NBXDtype.float32:
        real = real.to(NBXDtype.float32)
    if imag.nbx_dtype != NBXDtype.float32:
        imag = imag.to(NBXDtype.float32)
    stacked = NBXTensor.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1).contiguous()
    return stacked.view_as_complex()


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
    """
    T = x.shape[1]
    H = hidden
    Wx = matmul_wrapper(x, w_ih_t)                       # [B, T, 4H]
    if b_ih is not None:
        Wx = add(Wx, b_ih)                                # broadcast [4H]
    outs = [None] * T
    step_range = range(T - 1, -1, -1) if reverse else range(T)
    for t in step_range:
        gates = add(Wx.select(1, t), matmul_wrapper(h, w_hh_t))   # [B, 4H]
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
