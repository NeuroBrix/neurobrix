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
# The bucket of a request-dependent key dimension (kernels/autotune_bucket.py): the key
# carries the bucket's top, the kernel runs the true size (the owner's decision, 2026-09-21).
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

# Route this module's kernel[grid] sites through the engine's launcher.
#
# `kernels/dispatch.py` already calls install(), and for a full engine run
# that is enough. It is not enough for anything that reaches a wrapper
# WITHOUT going through dispatch — a unit test, a tool, a probe — which then
# silently runs upstream Triton's launch path instead of the engine's, and
# on a backend whose upstream driver cannot load our artifacts it simply
# fails. Worse, whether it happened depended on whether some earlier import
# in the same process had pulled dispatch: nine kernels tests changed state
# on 2026-09-06 because an unrelated import reordered.
#
# install() is idempotent and honours NBX_LAUNCHER=triton, so the
# differential arm is unaffected. The launch sites live in this module; the
# routing of those sites belongs with them.
from nbxtensor.backend.launcher import install as _install_launcher

_install_launcher()


_FLOOR_CONFLICTS_SAID = set()


def _announce_floor_over_ceiling(clamped, taken, floor) -> None:
    """Say, once per (clamped, floor) pair, that a profile row lost to the floor.

    Once per pair and not per call: attention wrappers run thousands of times
    a decode, and a line repeated thousands of times is a line the next
    reader has learned to skip — the same rule as the autotune exclusions.
    """
    token = (clamped, floor)
    if token in _FLOOR_CONFLICTS_SAID:
        return
    _FLOOR_CONFLICTS_SAID.add(token)
    print(f"[SDPA_TILE] the profile ceiling proposed {clamped} but this "
          f"backend's attention lowering refuses tile dimensions under "
          f"{floor}; using {taken}. The profile row promises a tile the "
          f"backend will not lower — the row should be raised or scoped to "
          f"the template path that measured it.", flush=True)


def _autotune_headroom_guard(launch):
    """Wrap an autotuned kernel launch (`kernel[grid]`) against the ONE
    device allocation the engine does not own: the Triton autotuner's
    benchmark buffer (its L2-flush cache, a 256 MiB torch allocation,
    taken once per never-seen key). On an edge row the allocator pool
    and the deferred dead-tensor queues may hold the headroom that
    allocation needs — the pool flushes on OUR malloc failures only.
    On an out-of-memory failure of the launch, return the pool to the
    driver, drain the registered reclaimers, say so once, and retry the
    launch ONCE; a second failure raises unchanged. Zero cost on the
    cached-key path (one try/except around the call). 2026-09-02,
    xlong int4 row: the first attention chunk's autotune bench found
    121 MiB free on a 32 GB card."""
    def _run(*args, **kwargs):
        try:
            return launch(*args, **kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            freed = DeviceAllocator.empty_cache_pool()
            reclaimed = 0
            try:
                dev = DeviceAllocator.get_device()
            except Exception:
                dev = None
            for hook in list(getattr(DeviceAllocator, "_oom_reclaim_hooks", ())):
                try:
                    reclaimed += int(hook(dev, 0) or 0)
                except Exception as hook_err:
                    print(f"[AUTOTUNE_HEADROOM] reclaim hook error: {hook_err}", flush=True)
            print(f"[AUTOTUNE_HEADROOM] the autotune benchmark buffer was refused "
                  f"({str(exc).splitlines()[0][:90]}); pool flushed {freed / 2**20:.0f}MB, "
                  f"reclaimers freed {reclaimed / 2**20:.0f}MB — retrying the launch once",
                  flush=True)
            return launch(*args, **kwargs)
    return _run

# === Activations ===

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

# === Unary element-wise ===

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

# === Binary element-wise ===

from nbxtensor.ops.add import add_forward_kernel, add_scalar_kernel, add_scalar_dev_kernel, add_bias_broadcast_kernel
from nbxtensor.ops.binary_strided import add_strided_nd_kernel, mul_strided_nd_kernel, sub_strided_nd_kernel, div_strided_nd_kernel, add_scalar_strided_nd_kernel, mul_scalar_strided_nd_kernel
from nbxtensor.ops.mul import mul_forward_kernel, mul_scalar_kernel, mul_scalar_dev_kernel
from nbxtensor.ops.div import div_forward_kernel, div_scalar_kernel, div_scalar_dev_kernel, int_div_kernel
from nbxtensor.ops.sub import sub_forward_kernel, rsub_forward_kernel
from nbxtensor.ops.where import where_forward_kernel
from nbxtensor.ops.maximum import maximum_forward_kernel
from nbxtensor.ops.minimum import minimum_forward_kernel
from nbxtensor.ops.masked_fill import masked_fill_forward_kernel

# === Comparisons ===

from nbxtensor.ops.gt import gt_forward_kernel, gt_scalar_kernel
from nbxtensor.ops.ge import ge_forward_kernel, ge_scalar_kernel
from nbxtensor.ops.lt import lt_forward_kernel, lt_scalar_kernel
from nbxtensor.ops.eq import eq_forward_kernel, eq_scalar_kernel
from nbxtensor.ops.ne import ne_forward_kernel, ne_scalar_kernel

# === Normalization ===

from nbxtensor.ops.layernorm import layer_norm_forward_kernel
from nbxtensor.ops.rmsnorm import rms_norm_forward_kernel

# === Softmax ===

from nbxtensor.ops.softmax import softmax_forward_kernel, softmax_forward_looped_kernel

# === Matmul ===

from nbxtensor.ops.matmul import matmul_kernel, addmm_kernel
from nbxtensor.ops._autotune_policy import NBX_FORCE_FP32_ACCUM

# === Fused MoE ===

from nbxtensor.ops.fused_moe import fused_moe_kernel, silu_and_mul_kernel, silu_mul_split_kernel
from nbxtensor.ops.dtype_convert import bf16_to_fp16_kernel

# === Embedding ===

from nbxtensor.ops.embedding import embedding_kernel, EMBEDDING_OOB

# === Reductions ===

from nbxtensor.ops.sum import sum_forward_kernel
from nbxtensor.ops.mean import mean_forward_kernel
from nbxtensor.ops.amax import amax_forward_kernel

# === GLU ===

from nbxtensor.ops.glu import glu_forward_kernel

# === New kernels (Phase 2) ===

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

# === Simple element-wise (Phase 3) ===

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

# === Phase 4: Remaining kernels ===

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

# === Phase 5: RoPE, spatial, RNG, remaining ===

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

# === Phase 6: Video + Audio ops ===

from nbxtensor.ops.upsample_nearest3d import upsample_nearest3d_kernel
from nbxtensor.ops.weight_norm import weight_norm_kernel_first, weight_norm_kernel_last
from nbxtensor.ops.repeat_interleave import repeat_interleave_tensor_kernel

# An element-wise launch's tile and warps: its element count's class, certified (`ops/elementwise_launch.py`).
from nbxtensor.ops.elementwise_launch import ew_meta as _ew_meta, ew_grid as _ew_grid


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

    Accepts any object with `has_native_bf16` and `devices` (each with `brand` and
    `architecture`), duck-typed so tests can pass a mock without importing the full Prism
    stack; the kernels' float64 capability is the backend's fact, declared to the capabilities
    (`set_backend_has_fp64`).
    """
    global _NBX_HAS_NATIVE_BF16, _NBX_HW_PROFILE
    if profile is None:
        return
    _NBX_HW_PROFILE = profile
    _NBX_HAS_NATIVE_BF16 = bool(getattr(profile, "has_native_bf16", True))
    # The kernels' float64 capability, the backend's own fact (`device_facts.kernels_carry_fp64`), declared to the
    # capabilities the integer-division widening and the cast door ask; an application that states it otherwise sets
    # it after this call (`capabilities.set_backend_has_fp64`, NeuroBrix: `triton.dtype.declare_hardware_profile`).
    from nbxtensor.backend.device_facts import kernels_carry_fp64
    from nbxtensor.backend.capabilities import set_backend_has_fp64
    set_backend_has_fp64(kernels_carry_fp64())


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
# - _NBX_ACTIVATIONS_FP16_SAFE: per-component flag of the precision
#   contract (calibration record). Read under an fp16 compute dtype only:
#   when True, ops in AMP_FP32_OPS that produce fp32 internally cast their
#   output back to compute_dtype (rms_norm, div, etc.) — VRAM-preserving for
#   models whose activations are confirmed within fp16 range by the
#   calibration record (neurobrix calibrate); False keeps the conservative
#   fp32 output. Under bf16 the output is cast back whatever the flag says
#   (triton/dtype.py `amp_fp32_output_dtype`).
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


# --- Fixed block sizes for kernels without @triton.autotune ---
_MM_BM, _MM_BN, _MM_BK = 64, 64, 32      # matmul / addmm / baddbmm
_MM_GROUP = 8                              # matmul GROUP_M
_CONV_BHW, _CONV_OUTF, _CONV_INF = 64, 64, 32  # conv2d
_BILINEAR_BX, _BILINEAR_BY = 16, 16       # upsample_bilinear2d


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
        if os.environ.get("NBX_SCALAR_DIAG") == "1":
            import traceback as _tb
            site = next((f"{_f.name}@{_f.lineno}" for _f in
                         reversed(_tb.extract_stack()[:-1])
                         if "wrappers.py" in _f.filename), "?")
            print(f"[SCALAR_DIAG] device-0d .item() via {site} "
                  f"dtype={getattr(x, 'nbx_dtype', '?')}")
        return x.item()
    if hasattr(x, 'numel') and x.numel() == 1:
        if os.environ.get("NBX_SCALAR_DIAG") == "1":
            import traceback as _tb
            site = next((f"{_f.name}@{_f.lineno}" for _f in
                         reversed(_tb.extract_stack()[:-1])
                         if "wrappers.py" in _f.filename), "?")
            print(f"[SCALAR_DIAG] device-1el .item() via {site} "
                  f"shape={getattr(x, 'shape', '?')}")
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
    set_device(t)
    return t


def _transfer_to_device(tensor, target_device_idx: int):
    """Transfer NBXTensor to a different GPU via cudaMemcpy D2D."""
    if not hasattr(tensor, '_device_idx') or tensor._device_idx == target_device_idx:
        return tensor
    src = tensor.contiguous()
    dst = NBXTensor.empty(src._shape, src._dtype, f"cuda:{target_device_idx}")
    DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), src._nbytes, kind=3)
    return dst


def _wider_dtype(a_dtype, b_dtype):
    """The dtype a binary element-wise op computes in: nbxtensor's one rule (`dtype.promotion.binary_result`)."""
    if a_dtype == b_dtype:
        return a_dtype
    return parse_dtype(binary_result(canonical_name(a_dtype), canonical_name(b_dtype)))


def _prepare_unary(x):
    """Prepare a single tensor for kernel launch."""
    x = x.contiguous()
    set_device(x)
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
    # Device-resident 0-d operands stay ON DEVICE (is_scalar="dev"):
    # the host .item() here was 1,196 syncs per Ming request and made
    # the denoiser bucket replay-ineligible (per-step timestep baked
    # into launch scalars). The *_scalar_dev kernels load the scalar
    # in-register with bit-exact host-path arithmetic (2026-08-15).
    def _dev_scalar(x):
        # NBXTensor.device returns self (its own device context) — the
        # canonical device string lives in _device ('cuda' / 'cpu').
        return (isinstance(x, NBXTensor) and getattr(x, 'ndim', None) == 0
                and str(getattr(x, '_device', 'cpu')).startswith('cuda'))

    if _is_scalar(b):
        a = a.contiguous()
        set_device(a)
        output = NBXTensor.empty_like(a)
        if _dev_scalar(b):
            return a, b, output, a.numel(), None, "dev"
        return a, _to_scalar(b), output, a.numel(), None, True

    if _is_scalar(a):
        # Swap: tensor goes to `a`, scalar goes to `b`
        b = b.contiguous()
        set_device(b)
        output = NBXTensor.empty_like(b)
        if _dev_scalar(a):
            return b, a, output, b.numel(), None, "dev"
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
        out_shape = broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    set_device(a)
    output = NBXTensor.empty_like(a)
    return a, b, output, a.numel(), None, False


def _prepare_binary_strided(a, b):
    """The two-tensor case read by strides: (a, b, output, n, shape_buf, a_strides_buf,
    b_strides_buf, NDIM) with both operands broadcast to the common shape as VIEWS —
    no `expand(...).contiguous()` transient. Returns (None, a, b) when the flat path
    applies (both contiguous at the same shape: the flat kernel's indexing is cheaper),
    with a and b ALREADY aligned in dtype and device so the caller's `_prepare_binary`
    casts nothing twice (the first form did: 45 casts a token on TinyLlama, 2026-09-08).
    The dtype cast stays a copy (an in-kernel widen moves the compiler's contraction —
    the RoPE lesson, 2026-09-07)."""
    if _is_scalar(a) or _is_scalar(b) or not (hasattr(a, "_dtype") and hasattr(b, "_dtype")):
        return None, a, b
    common_dtype = _wider_dtype(a._dtype, b._dtype)
    if a._dtype != common_dtype:
        a = a.to(common_dtype)
    if b._dtype != common_dtype:
        b = b.to(common_dtype)
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _transfer_to_device(b, a._device_idx)
    if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
        return None, a, b
    out_shape = broadcast_shapes(a.shape, b.shape) if a.shape != b.shape else tuple(a.shape)
    if tuple(a.shape) != tuple(out_shape):
        a = a.expand(*out_shape)
    if tuple(b.shape) != tuple(out_shape):
        b = b.expand(*out_shape)
    ndim = max(len(out_shape), 1)
    if ndim > _NBX_MAX_NDIM:
        return None, a, b
    shape = tuple(out_shape) or (1,)
    a_st = tuple(a._strides) or (0,)
    b_st = tuple(b._strides) or (0,)
    set_device(a)
    output = NBXTensor.empty(tuple(out_shape), a._dtype, f"cuda:{a._device_idx}")
    n = output.numel()
    return (a, b, output, n,
            _nbx_upload_int64(shape, a._device_idx), _nbx_upload_int64(a_st, a._device_idx),
            _nbx_upload_int64(b_st, a._device_idx), ndim)


def _prepare_scalar_strided(a, b):
    """The tensor-and-host-scalar case with a strided tensor: (tensor, scalar, output, n,
    shape_buf, strides_buf, NDIM), the tensor read in place by the strided scalar kernel —
    or None when the flat path applies (a contiguous tensor, or a device-resident 0-d
    scalar, whose kernel reads it in-register). A chunk of an adaLN vector, `1 + scale` on a
    (2, 1, 1152) slice of (2, 6, 1152), used to be copied contiguous per modulation."""
    if _is_scalar(a) and not _is_scalar(b):
        a, b = b, a
    if not (_is_scalar(b) and hasattr(a, "_strides")) or a.is_contiguous():
        return None
    if isinstance(b, NBXTensor):          # a device-resident 0-d scalar: the *_scalar_dev kernels' path
        return None
    ndim = max(a.ndim, 1)
    if ndim > _NBX_MAX_NDIM:
        return None
    shape = tuple(a.shape) or (1,)
    st = tuple(a._strides) or (0,)
    set_device(a)
    output = NBXTensor.empty(tuple(a.shape), a._dtype, f"cuda:{a._device_idx}")
    return (a, _to_scalar(b), output, output.numel(),
            _nbx_upload_int64(shape, a._device_idx), _nbx_upload_int64(st, a._device_idx), ndim)


def _prepare_comparison(a, b):
    """Prepare two tensors for a comparison kernel (output is bool).

    Same guards as _prepare_binary (dtype align, device align, broadcast,
    contiguous) but output tensor is always bool dtype.

    CONTRACT: when is_scalar=True, tensor is in `a`, scalar is in `b`.
    """
    if _is_scalar(b):
        a = a.contiguous()
        set_device(a)
        output = NBXTensor.empty(a.shape, NBXDtype.bool_, device=a.device)
        return a, _to_scalar(b), output, a.numel(), None, True

    if _is_scalar(a):
        # Swap: tensor goes to `a`, scalar goes to `b`
        b = b.contiguous()
        set_device(b)
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
        out_shape = broadcast_shapes(a.shape, b.shape)
        a = a.expand(out_shape).contiguous()
        b = b.expand(out_shape).contiguous()
    else:
        a = a.contiguous()
        b = b.contiguous()

    set_device(a)
    output = NBXTensor.empty(a.shape, NBXDtype.bool_, device=a.device)
    return a, b, output, a.numel(), None, False


def _is_integer_operand(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    dt = getattr(x, "dtype", None)
    return hasattr(dt, "is_floating") and not dt.is_floating() and dt != NBXDtype.bool_


def _int_division_widening_dtype(a, b):
    """The float dtype an integer division widens through: float64 where the backend has it
    (exact below 2^53), float32 where it has none (Metal; exact below 2^24, and REFUSED by name
    above, because a float64 element there is misread, not computed — `backend_has_fp64`,
    2026-09-28). The magnitude check is one reduction per integer division, a rare op."""
    from nbxtensor.backend.capabilities import backend_has_fp64
    if backend_has_fp64():
        return NBXDtype.float64
    limit = 1 << 24
    for t in (a, b):
        if hasattr(t, "to") and hasattr(t, "numel") and t.numel() > 0:
            # measured through float32 (exact below 2^24, and anything at or above it reads as such)
            m = int(_red.amax_wrapper(_ew.abs_wrapper(t.to(NBXDtype.float32))).item())
            if m >= limit:
                raise RuntimeError(
                    f"ZERO FALLBACK: an integer division with |values| up to {m} on a backend with no "
                    f"float64 ('{__import__('nbxtensor.backend.runtime', fromlist=['gpu_backend']).gpu_backend()}'): "
                    f"the float32 widening is exact only below 2^24, and float64 is misread there. "
                    f"An integer division kernel is the extension that lifts this.")
    return NBXDtype.float32


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


def _reduction_dtype(op, dtype):
    """The `dtype` keyword of aten::sum / aten::mean as an NBXDtype (None when absent). ATen casts the INPUT to it
    before reducing, and the result carries it (`sum(x, -1, keepdim=True, dtype=torch.float32)` over a bf16 router's
    top-k weights, Ming-Lite-Omni-1.5's language model, 2026-10-14). A string the NBX dtypes do not know is refused."""
    if dtype is None or isinstance(dtype, NBXDtype):
        return dtype
    if isinstance(dtype, dict) and dtype.get("type") == "dtype":
        dtype = dtype.get("value")
    name = str(dtype).replace("torch.", "")
    nbx = getattr(NBXDtype, name, None)
    if nbx is None:
        raise RuntimeError(f"ZERO FALLBACK: {op} dtype {dtype!r} is not a known NBXDtype.")
    return nbx


# ===========================================================================
# CONV2D WRAPPER — Extracted from FlagGems
# ===========================================================================

# Spatial band-streaming threshold for conv2d. When the per-launch output
# tensor would exceed `tiling_sizes.conv2d_band_bytes()` (`conv2d.band_bytes` in
# nbxtensor/ops/tiling.yml), conv2d_wrapper transparently splits along the H dimension
# and streams band-by-band. This is the kernel-level tiling lever
# (P-SANA-4KPX-RUNTIME, Step 1 — internal to the wrapper, not a Prism op-level
# interceptor). The TilingEngine's sizing half is the one definition Prism, the
# launch keys and this wrapper share.
from nbxtensor.ops import tiling_sizes as _ts


import math


def _arch_param(section: str, key: str, default):
    """One `<section>.<key>` value from config/vendors/<vendor>/<arch>.yml
    of the executing hardware profile; `default` when there is no profile,
    no key, or the loader fails (R23: hardware params come from the yml,
    never from code, and never from a driver query in a hot path).

    Shared reader for every per-architecture knob — the memory policy
    (`memory.sdpa_math_min_chunk_rows`, `memory.alloc_pool_parked_cap_fraction`,
    …) and the pipelining budget (`pipelining.max_num_stages`)."""
    prof = get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return default
    try:
        from nbxtensor.backend.device_files import param as vendor_param   # one key of the device, both halves
        _brand = devices[0].brand
        _vendor = getattr(_brand, "value", _brand)
        return vendor_param(_vendor, devices[0].architecture, section, key, default)
    except Exception:
        return default


def _arch_dispatch_param(key: str, default):
    """One `dispatch.<key>` value for the executing hardware profile.

    The launch path's own knobs live here — how many command buffers a queue
    may hold in flight, and anything else about how work reaches the device.
    They belong to the device, so they are read from its profile rather than
    written into the driver."""
    return _arch_param("dispatch", key, default)


def _arch_memory_param(key: str, default):
    """One `memory.<key>` value for the executing hardware profile.
    Thin alias of `_arch_param("memory", ...)`, kept because the memory
    knobs are read from many call sites."""
    return _arch_param("memory", key, default)

# the families cut out of this module (M2), reached through their module
from nbxtensor.ops import elementwise_wrappers as _ew  # noqa: E402 — the cut's families (M2)
from nbxtensor.ops import reduction_wrappers as _red  # noqa: E402 — the cut's families (M2)


#: the family modules of the wrappers, in the order a name is looked up (`function`)
_FAMILY_MODULES = ("elementwise_wrappers", "shape_wrappers", "conv_wrappers", "linalg_wrappers", "reduction_wrappers",
                   "attention_wrappers", "spectral_wrappers", "index_wrappers", "norm_wrappers", "random_wrappers",
                   "loss_wrappers", "moe_wrappers")
_MISSING = object()


def function(name: str, default=_MISSING):
    """The wrapper called `name`, wherever the split put it (this module or one of its families): the lookup for a
    wrapper NAMED BY DATA (an op table, an in-place map, a candidate list). A name no module defines is refused by name,
    unless a `default` is given."""
    import importlib
    if name in globals() and callable(globals()[name]):
        return globals()[name]
    for fam in _FAMILY_MODULES:
        mod = importlib.import_module(f"nbxtensor.ops.{fam}")
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    if default is not _MISSING:
        return default
    raise AttributeError(f"no wrapper named {name!r} in nbxtensor.ops.wrappers or its families ({', '.join(_FAMILY_MODULES)})")
