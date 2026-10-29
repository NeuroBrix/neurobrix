"""Attention wrappers: softmax, scaled dot-product attention (math, chunked, flash, the matrix-unit flash), paged and split-KV decode. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import linalg_wrappers as _la
from nbxtensor.ops import wrappers as _core


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_TL_DTYPE = {NBXDtype.float16: triton.language.float16, NBXDtype.bfloat16: triton.language.bfloat16,
             NBXDtype.float32: triton.language.float32, NBXDtype.float64: triton.language.float64}


# ===========================================================================
# SOFTMAX WRAPPER
# ===========================================================================

# Above this feat size the single-tile kernel's next_power_of_2(feat) tile
# spills to CUDA local memory, whose pool the driver allocates DEVICE-WIDE
# at launch (frame × max resident threads — ~2 GiB measured at vocab
# 151936 on V100, D9 2026-07-27). Rows larger than the cap route to the
# bounded-tile looped kernel; rows at or under it keep the proven
# single-tile kernel byte-for-byte. Same capped-tile idiom as
# argmax_wrapper's TILE_N = min(next_power_of_2(N), 4096) (FlagGems).
_SOFTMAX_MAX_FUSED_FEAT = _lk.SOFTMAX_FUSED_MAX_FEAT     # one value with the census (launch_keys)
def _launch_looped_softmax(x_2d, output_2d, batch_dim, feat_dim, log: bool):
    """Launch the bounded-tile online softmax — one program per row; its tile and warps the certified configuration
    of this key (chantier 1), the key the derived census forms (`launch_keys.softmax_launches`)."""
    softmax_forward_looped_kernel[(batch_dim,)](
        x_2d, output_2d,
        feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
        x_2d.stride(0), x_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        log=log,
    )


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

    _set_device(x_2d)
    if feat_dim > _SOFTMAX_MAX_FUSED_FEAT:
        _launch_looped_softmax(x_2d, output_2d, batch_dim, feat_dim, log=False)
    else:
        # rows per program and warps: the certified configuration of this key (workstream 1), the row's bucket from
        # the launcher's own `bucket_of` — the key the derived census forms (`launch_keys.softmax_launches`)
        grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_forward_kernel[grid](
            x_2d, output_2d,
            batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
            x_2d.stride(0), x_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            log=False,
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

    _set_device(x_2d)
    if feat_dim > _SOFTMAX_MAX_FUSED_FEAT:
        _launch_looped_softmax(x_2d, output_2d, batch_dim, feat_dim, log=True)
    else:
        # rows per program and warps: the certified configuration of this key (workstream 1), the row's bucket from
        # the launcher's own `bucket_of` — the key the derived census forms (`launch_keys.softmax_launches`)
        grid = lambda META: (triton.cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_forward_kernel[grid](
            x_2d, output_2d,
            batch_dim, feat_dim, _lk.bucket_of("M", batch_dim), _lk.softmax_feat_bucket(feat_dim),
            x_2d.stride(0), x_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            log=True,
        )
    return output_2d.view_as(x_perm).movedim(-1, dim)


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
    prof = _core.get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return 0
    try:
        from nbxtensor.host import need as _need   # the application's core.config.loader (nbxtensor.host)
        get_vendor_config = _need("vendor_config")
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


def _sdpa_math_scores_device_fraction() -> float:
    """Optional per-DEVICE cap on the math-route scores budget, as a
    fraction of the EXECUTING device's memory. Data-driven from
    config/vendors/<vendor>/<arch>.yml
    `memory.sdpa_math_scores_device_fraction` (R23 — the 2026-08-31
    xlong-prefill OOM proved the per-arch byte budget alone is wrong on
    heterogeneous rigs: 2 GiB of scores fit a 32 GB card's headroom but
    not a 16 GB card's). 0.0 (absent key) = no device cap, prior
    behaviour byte-unchanged."""
    prof = _core.get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return 0.0
    try:
        from nbxtensor.host import need as _need   # the application's core.config.loader (nbxtensor.host)
        get_vendor_config = _need("vendor_config")
        _brand = devices[0].brand
        _vendor = getattr(_brand, "value", _brand)
        cfg = get_vendor_config(_vendor, devices[0].architecture)
        return float(cfg.get("memory", {}).get(
            "sdpa_math_scores_device_fraction", 0.0))
    except Exception:
        return 0.0


def _sdpa_math_scores_budget_bytes_for(device_idx) -> int:
    """Scores budget for the device that will execute this SDPA:
    min(per-arch yml byte budget, device_fraction x THIS device's
    memory_mb from the Prism profile — a static profile read, never a
    driver query in the hot path). Absence (no fraction key, no
    profile, unknown index) is handled by EXPLICIT conditions and
    returns the plain per-arch budget; there is no broad except — a
    real profile-read failure must surface, not silently restore the
    16G OOM this cap exists to prevent."""
    base = _sdpa_math_scores_budget_bytes()
    if not base:
        return 0
    frac = _sdpa_math_scores_device_fraction()
    if not frac or device_idx is None:
        return base
    prof = _core.get_hardware_profile()
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return base
    dev = next((d for d in devices
                if getattr(d, "index", None) == int(device_idx)), None)
    if dev is None:
        return base
    return _ts.sdpa_device_scores_budget(base, frac, dev.memory_mb)


def _sdpa_math_min_chunk_rows() -> int:
    """Row alignment AND floor of the chunked deterministic math prefill,
    from config/vendors/<vendor>/<arch>.yml `memory.sdpa_math_min_chunk_rows`
    (the kernel's row block); 0 (absent key / no profile) = the chunked
    route never fires, like `sdpa_math_max_chunks` (R23)."""
    try:
        return int(_core._arch_memory_param("sdpa_math_min_chunk_rows", 0))
    except Exception:
        return 0


def _sdpa_math_max_chunks() -> int:
    """Chunk-count ceiling for the CHUNKED deterministic math prefill
    (P-NONDET-LONG-ROW fix). Data-driven from
    config/vendors/<vendor>/<arch>.yml `memory.sdpa_math_max_chunks`;
    0 (absent key / no profile) = the chunked route never fires, so an
    unconfigured arch keeps its attention path byte-unchanged (R23).
    The ceiling is the determinism tax bound: LLM-class prefills chunk
    a handful of times under the 2 GiB scores budget (4k ctx → 2,
    16k → 16); video-scale prefills would chunk 30-500+ times at a
    prohibitive cost and stay on their existing path, REGISTERED as
    the residual non-deterministic class on this hardware."""
    try:
        return int(_core._arch_memory_param("sdpa_math_max_chunks", 0))
    except Exception:
        return 0


def _math_attention(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """Math-decomposed attention — deterministic for any head_dim.

    Used for non-power-of-2 head_dim (PixArt 72, Sana 112) where the
    flash kernel's masked-load path is non-deterministic on Volta SIMT,
    and for asymmetric head_dim attention (MLA class — DeepSeek-V2
    qk_head_dim 192 / v_head_dim 128), which the flash kernel cannot
    express (single BLOCK_HEADDIM constexpr, q-shaped output).

    Shapes: q [B, H, T_q, D], k [B, H_k, T_k, D], v [B, H_k, T_k, D_v]
    (D_v may differ from D). Returns [B, H, T_q, D_v].

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
    # Value head_dim is carried INDEPENDENTLY of q/k head_dim. MLA-class
    # attention (DeepSeek-V2: qk_head_dim 192 = 128 nope + 64 rope,
    # v_head_dim 128) is asymmetric — reusing q's D for the v reshape
    # built a numel-mismatched view (NBXTensor.view re-strides without
    # numel validation → silent out-of-bounds read) and returned a
    # q-shaped [B, H, T_q, 192] output that the baked downstream views
    # re-inferred into heads*192 = 3072 columns → aten.mm
    # "Incompatible dimensions: 3072 vs 2048"
    # (DeepSeek-Coder-V2-Lite aten.mm::3, P-TRITON-MLA).
    D_v = v.shape[3]
    # q @ k^T requires matching q/k head dims. Since NBXTensor.view /
    # reshape never validate numel, a mismatch here would be a SILENT
    # out-of-bounds read — fail loudly instead.
    assert k.shape[3] == D, (
        f"math attention: q head_dim {D} != k head_dim {k.shape[3]}")

    if scale is None:
        scale = 1.0 / _math.sqrt(D)

    # GQA WITHOUT BROADCASTING K/V.
    #
    # The old form expanded K and V up to Q's head count and
    # materialised the result — the whole KV cache, per layer, per
    # token. Measured at a 4,164-token context that was 34 MB per call,
    # 4.9 GB per decode step for K and the same again for V, and it is
    # the dominant term in our context scaling: `strided_copy` fitted at
    # 11.9 us per context token against `baddbmm`'s 3.8.
    #
    # The algebra never needed it. Q's heads can be grouped INSIDE each
    # KV head instead: with H = H_k * groups the flat order of
    # [B, H, T_q, D] is already b, h_k, g, t, d, so viewing it as
    # [B*H_k, groups*T_q, D] is a pure view, and batched against
    # [B*H_k, T_k, D] it computes exactly the same dot products.
    #
    # For plain MHA groups == 1, BH == B*H and M == T_q, so every
    # expression below is literally the previous one — MHA is unchanged
    # by construction, not by luck.
    groups = H // H_k
    assert H_k * groups == H, f"GQA: H={H} not a multiple of H_k={H_k}"
    BH = B * H_k
    M = groups * T_q

    def _mask_to_grouped(m2d):
        """[T_q, T_k] -> [BH, M, T_k], row (g, t) taking mask row t."""
        if groups == 1:
            return m2d.unsqueeze(0).expand(BH, M, T_k)
        return (m2d.unsqueeze(0).expand(groups, T_q, T_k)
                .reshape(M, T_k).unsqueeze(0).expand(BH, M, T_k))

    # Reshape to 3D for bmm
    q_3d = q.reshape(BH, M, D)
    # Q @ K^T without materialising K^T.
    #
    # This used to be `k.transpose(-2,-1).contiguous().reshape(...)`,
    # which copied the whole K tile on every layer of every decode step
    # to serve a single-token query — on the canonical 30B row, 48 x
    # [32, 256, 128] per token. `baddbmm_kernel` takes B's strides and
    # can walk the transposed view directly. Measured on that exact
    # shape: BIT-IDENTICAL result, 0.519 ms -> 0.122 ms (x4.26), which
    # over 48 layers is ~19 ms per token.
    #
    # Order matters: reshape FIRST on the contiguous k (merging B and H,
    # a pure view), THEN transpose. Transposing first and reshaping
    # after would send `reshape` through its own materialisation and put
    # the copy straight back.
    # merge_dims keeps the (B, H) merge a PURE VIEW even when K/V are
    # the bucketed cache's prefix-slice views (non-contiguous in the T
    # dim — a plain reshape materialised them here, copying the whole
    # padded K and V per layer per decode step; copies-at-source census
    # 2026-08-23). Both bmms below walk B's strides.
    k_3d_t = k.merge_dims(0, 1).transpose(-2, -1)
    v_3d = v.merge_dims(0, 1)

    # scores = q @ k^T (bmm returns fp32 on V100 via force_fp32)
    scores = _la.bmm(q_3d, k_3d_t, allow_strided_b=True)  # [BH, M, T_k]
    scores = _ew.mul(scores, float(scale))

    # Causal/explicit mask handling — additive bias on scores.
    # Built via existing tril-based path; skipped for non-causal no-mask.
    if is_causal and attn_mask is None:
        device_idx = q._device_idx if hasattr(q, '_device_idx') else 0
        bias = _ew._get_causal_bias(device_idx, T_q, T_k, scores.nbx_dtype)
        # bias [T_q, T_k] -> [BH, M, T_k], grouped like the scores
        scores = _ew.add(scores, _mask_to_grouped(bias))
    elif attn_mask is not None:
        # attn_mask may be [T_q, T_k], [B, T_q, T_k], or [B, H, T_q, T_k].
        if attn_mask.ndim == 2:
            mask_3d = _mask_to_grouped(attn_mask)
        elif attn_mask.ndim == 3:
            if attn_mask.shape[0] == 1 and B > 1:
                attn_mask = attn_mask.expand(B, T_q, T_k)
            mask_3d = (attn_mask.unsqueeze(1).unsqueeze(1)
                       .expand(B, H_k, groups, T_q, T_k).reshape(BH, M, T_k))
        elif attn_mask.ndim == 4:
            mask_3d = (attn_mask.expand(B, H, T_q, T_k)
                       .reshape(B, H_k, groups, T_q, T_k).reshape(BH, M, T_k))
        else:
            raise RuntimeError(f"unsupported attn_mask.ndim={attn_mask.ndim}")
        if mask_3d.nbx_dtype != scores.nbx_dtype:
            mask_3d = mask_3d.to(scores.nbx_dtype)
        scores = _ew.add(scores, mask_3d.contiguous())

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
    p = _ew.nan_to_num_wrapper(p, nan=0.0)
    # Cast back to value dtype for the second bmm
    if p.nbx_dtype != v_3d.nbx_dtype:
        p = p.to(v_3d.nbx_dtype)

    # out = p @ v (bmm again returns fp32 on V100). Strided B: v_3d may
    # be the merged bucketed-cache view (row pitch = buffer capacity).
    out_3d = _la.bmm(p, v_3d, allow_strided_b=True)  # [BH, M, D_v]
    if out_3d.nbx_dtype != q.nbx_dtype:
        out_3d = out_3d.to(q.nbx_dtype)

    return out_3d.reshape(B, H, T_q, D_v)


def _math_attention_chunked(q, k, v, attn_mask, is_causal, scale,
                            chunk_rows):
    """Deterministic math attention for scores tensors OVER the budget —
    the P-NONDET-LONG-ROW fix (2026-08-23).

    The long-context prefill routed to the flash kernel (its full fp32
    scores exceed `sdpa_math_max_scores_bytes`), and on Volta the flash
    kernel's masked-load specialisation — which the exact D>=128 pad-
    by-one detour lands in — is NON-DETERMINISTIC (kernel-level: 3
    distinct outputs in 5 identical calls at the 4,164-token prefill
    shape; end-to-end: the first replayed-step logits differed across
    every pair of runs, and greedy near-ties downstream made the row's
    output bimodal). Forcing math made 4/4 runs byte-identical at
    every step.

    This runs the SAME `_math_attention` (deterministic by
    construction) over query-row chunks, so no chunk's scores tensor
    exceeds the budget. Causality is preserved exactly: the full
    causal bias is built once and each chunk receives its row slice as
    an explicit additive mask — the same values `_math_attention`
    would have added row-by-row unchunked.
    """
    B, H, T_q, _D = q.shape
    T_k = k.shape[2]
    D_v = v.shape[3]
    device_idx = q._device_idx if hasattr(q, '_device_idx') else 0
    out = NBXTensor.empty((B, H, T_q, D_v), dtype=q.nbx_dtype,
                          device=f"cuda:{device_idx}")
    bias_full = None
    if is_causal and attn_mask is None:
        bias_full = _ew._get_causal_bias(device_idx, T_q, T_k,
                                     NBXDtype.float32)
    # ZERO FALLBACK: `chunk_rows` is the TilingEngine's size
    # (`tiling_sizes.sdpa_chunk_rows`), fixed BEFORE the launch. An allocation
    # the driver refuses is an error naming the key (op, shape, rows, bytes
    # asked) and is re-raised — never a smaller retry, which would run a chunk
    # count the plan and the launch key never saw.
    q0 = 0
    while q0 < T_q:
        c = min(chunk_rows, T_q - q0)
        q_c = q[:, :, q0:q0 + c, :].contiguous()
        if bias_full is not None:
            mask_c = bias_full[q0:q0 + c, :]
        elif attn_mask is None:
            mask_c = None
        elif attn_mask.ndim == 2:
            mask_c = attn_mask[q0:q0 + c, :]
        elif attn_mask.ndim == 3:
            mask_c = (attn_mask if attn_mask.shape[1] == 1
                      else attn_mask[:, q0:q0 + c, :])
        else:
            mask_c = (attn_mask if attn_mask.shape[2] == 1
                      else attn_mask[:, :, q0:q0 + c, :])
        try:
            o = _math_attention(q_c, k, v, attn_mask=mask_c,
                                is_causal=False, scale=scale)
        except DeviceOOMError as _oom:
            _asked = _ts.sdpa_scores_bytes(B, H, c, T_k)
            raise DeviceOOMError(
                f"chunked SDPA: q [{B}, {H}, {T_q}, {_D}] k/v [{B}, {k.shape[1]}, {T_k}, {D_v}], "
                f"{chunk_rows} rows a chunk (tiling_sizes.sdpa_chunk_rows), chunk at row {q0}: "
                f"{_asked} bytes of fp32 scores asked on cuda:{device_idx}, refused by the "
                f"allocator ({_oom}). The row count is the plan's; it is not retried smaller.",
                requested=_oom.requested, device_idx=_oom.device_idx, live=_oom.live,
                pool_cached=_oom.pool_cached, pool_blocks=_oom.pool_blocks,
                driver_free=_oom.driver_free, driver_total=_oom.driver_total) from _oom
        out[:, :, q0:q0 + c, :] = o
        q0 += c
    return out


def paged_kv_append(new, pool, slot_mapping):
    """Write new positions into a layer's paged pool (state engine step 2, ops/paged_decode_attn.py).

    new [N_tok, H_kv, D] (any float dtype, last dim contiguous) · pool [num_pages, H_kv, page_tokens, D] · slot_mapping
    [N_tok] int32 (core/state/pages.PageTable.slots: page * page_tokens + position in page). Cast to the pool's dtype on
    store. Returns the pool."""
    from nbxtensor.ops.paged_decode_attn import paged_kv_append_kernel
    n_tok, h_kv, d = (int(x) for x in new.shape)
    if tuple(pool.shape[1:2]) != (h_kv,) or int(pool.shape[3]) != d:
        raise ValueError(f"paged_kv_append: new {list(new.shape)} does not fit the pool {list(pool.shape)}")
    if slot_mapping.nbx_dtype != NBXDtype.int32 or int(slot_mapping.numel()) != n_tok:
        raise ValueError(f"paged_kv_append: slot_mapping must be int32 [{n_tok}], got {slot_mapping.nbx_dtype} "
                         f"{list(slot_mapping.shape)}")
    if new.stride(2) != 1 or pool.stride(3) != 1:
        raise ValueError("paged_kv_append: the head dim must be contiguous in the new positions and in the pool")
    if n_tok == 0:
        return pool
    _set_device(pool)
    paged_kv_append_kernel[(n_tok, h_kv)](
        new, pool, slot_mapping, int(pool.shape[2]),
        new.stride(0), new.stride(1),
        pool.stride(0), pool.stride(1), pool.stride(2),
        BLOCK_D=max(triton.next_power_of_2(d), _fa_min_tile()), D=d,
        num_warps=1)
    return pool


def paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens, softmax_scale, max_seq_len, block_n,
                           q_round=None):
    """Decode attention (one query position) read through the page table — the vector split-KV kernel of
    ops/decode_attn_vec.py on paged K / V, merged by the fixed-order `flash_decode_reduce_kernel`.

    q [B, H, 1, D] contiguous · k_pool / v_pool [num_pages, H_kv, page_tokens, D] · block_table [B, max_pages] int32
    (-1 past each sequence) · seq_lens [B] int32 · max_seq_len: the longest of seq_lens (sizes the splits; read by the
    caller from its page tables, never from the device) · block_n: the position tile, one of
    PAGED_DECODE_TILE_CANDIDATES dividing page_tokens · q_round: the dtype Q is read in (the cache's), rounded on load
    as `_decode_attn_vec` does; the output is written in it. Output [B, H, 1, D_v] in q_round, else q's dtype."""
    from nbxtensor.ops.paged_decode_attn import (paged_decode_attn_split_kernel, PAGED_DECODE_TILE_CANDIDATES,
                                        PAGED_DECODE_SPLIT_POSITIONS, PAGED_DECODE_MAX_SPLITS)
    from nbxtensor.ops.flash_decode import flash_decode_reduce_kernel
    batch, nheads, tq, headdim = (int(x) for x in q.shape)
    num_pages, nheads_k, page_tokens, d_k = (int(x) for x in k_pool.shape)
    d_v = int(v_pool.shape[3])
    if tq != 1 or d_k != headdim or nheads % nheads_k:
        raise ValueError(f"paged_decode_attention: q {list(q.shape)} against the pool {list(k_pool.shape)}")
    if block_n not in PAGED_DECODE_TILE_CANDIDATES or page_tokens % block_n:
        raise ValueError(f"ZERO FALLBACK: tile {block_n} is not a declared candidate {PAGED_DECODE_TILE_CANDIDATES} "
                         f"dividing the page of {page_tokens} positions — a tile would straddle two pages")
    for name, t in (("block_table", block_table), ("seq_lens", seq_lens)):
        if t.nbx_dtype != NBXDtype.int32:
            raise ValueError(f"paged_decode_attention: {name} must be int32, got {t.nbx_dtype}")
    if int(block_table.shape[0]) != batch or int(seq_lens.numel()) != batch:
        raise ValueError(f"paged_decode_attention: block_table {list(block_table.shape)} / seq_lens "
                         f"{list(seq_lens.shape)} for a batch of {batch}")
    groups = nheads // nheads_k
    bhq = batch * nheads
    BLOCK_D = max(triton.next_power_of_2(max(headdim, d_v)), _fa_min_tile())
    # The vector decode kernel's split shape (ops/paged_decode_attn.py), each split a whole number of tiles so a split
    # never starts inside a tile.
    longest = max(int(max_seq_len), 1)
    split = max(1, min(int(triton.cdiv(longest, PAGED_DECODE_SPLIT_POSITIONS)), PAGED_DECODE_MAX_SPLITS))

    q2 = q.reshape(bhq, headdim)
    dev = f"cuda:{q._device_idx}"
    opart = NBXTensor.empty((bhq, split, d_v), dtype=NBXDtype.float32, device=dev)
    mpart = NBXTensor.empty((bhq, split), dtype=NBXDtype.float32, device=dev)
    lpart = NBXTensor.empty((bhq, split), dtype=NBXDtype.float32, device=dev)
    out_dtype = q_round if q_round is not None else q.nbx_dtype
    q_to = _TL_DTYPE[q_round] if q_round is not None else None
    q_saturate = (q_round == NBXDtype.float16
                  and q.nbx_dtype in (NBXDtype.float32, NBXDtype.float64, NBXDtype.bfloat16))
    out = NBXTensor.empty((bhq, 1, d_v), dtype=out_dtype, device=dev)
    _set_device(q)
    paged_decode_attn_split_kernel[(bhq, split)](
        q2, k_pool, v_pool, block_table, seq_lens,
        opart, mpart, lpart,
        batch, page_tokens, float(softmax_scale), nheads,
        q2.stride(0),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_table.stride(0),
        opart.stride(0), opart.stride(1),
        mpart.stride(0),
        GQA_GROUPS=groups, BLOCK_N=block_n, BLOCK_D=BLOCK_D, D=headdim, D_V=d_v,
        Q_TO=q_to, Q_SATURATE=q_saturate,
        num_warps=4, num_stages=1)
    flash_decode_reduce_kernel[(bhq,)](
        opart, mpart, lpart, out,
        opart.stride(0), opart.stride(1), d_v,
        mpart.stride(0), mpart.stride(1),
        out.stride(0), out.stride(1),
        SPLIT=split, GROUPS=1, BLOCK_G=max(16, _fa_min_tile()),
        BLOCK_D=BLOCK_D, D=d_v,
        num_warps=4)
    return out.reshape(batch, nheads, 1, d_v)


def _decode_attn_vec(q, k, v, bias, softmax_scale,
                     batch, nheads, nheads_k, seqlen_k, headdim, q_round=None):
    """Vector (SIMT) decode attention launch pair — see
    ops/decode_attn_vec.py for the sourced structure (R16 2026-08-23:
    FasterTransformer mmha / vLLM paged v2 / llama.cpp fattn-vec /
    lightllm — no tl.dot on sm_70, one query row per program, split-KV
    with the existing fixed-order reduce).

    q [B, H, 1, D] fp16 CONTIGUOUS (wrapper entry) · k/v in native
    cache layout, consumed BY THEIR OWN STRIDES (the bucketed cache's
    prefix-slice views flow in unmaterialised — the strided-KV lot's
    contract) · bias flat [T_k] or None.
    """
    import os as _os_dv
    from nbxtensor.ops.decode_attn_vec import decode_attn_vec_split_kernel
    from nbxtensor.ops.flash_decode import flash_decode_reduce_kernel
    groups = nheads // nheads_k
    BHq = batch * nheads
    D_v = v.shape[3]
    # The floor is the backend capability, not 16: a dot tile dimension the
    # attention lowering refuses is not a tile this path can use. Inert on
    # cuda and hip, where it reads 16.
    BLOCK_D = max(triton.next_power_of_2(max(headdim, D_v)), _fa_min_tile())
    # Defaults = the judged config (locked pinned protocol, 2026-08-23:
    # seg 256 / BN 32 / 4 warps won the row at all three contexts).
    BLOCK_N = int(_os_dv.environ.get("NBX_DV_BLOCK_N", "32"))
    seg = int(_os_dv.environ.get("NBX_DV_SEG", "256"))
    split = max(1, min(int(triton.cdiv(seqlen_k, seg)), 64))
    seg_len = int(triton.cdiv(seqlen_k, split))
    warps = int(_os_dv.environ.get("NBX_DV_WARPS", "4"))

    q2 = q.reshape(BHq, headdim)   # contiguous -> pure view
    # `q_round`: the dtype the caller asked Q to be read in (the KV cache's);
    # the kernel rounds Q to it on load and the output is written in it — the
    # bytes of a cast copy of Q, without the copy.
    out_dtype = q_round if q_round is not None else q.nbx_dtype
    q_to = _TL_DTYPE[q_round] if q_round is not None else None
    q_saturate = (q_round == NBXDtype.float16
                  and q.nbx_dtype in (NBXDtype.float32, NBXDtype.float64, NBXDtype.bfloat16))

    dev = f"cuda:{q._device_idx}"
    opart = NBXTensor.empty((BHq, split, D_v), dtype=NBXDtype.float32,
                            device=dev)
    mpart = NBXTensor.empty((BHq, split), dtype=NBXDtype.float32,
                            device=dev)
    lpart = NBXTensor.empty((BHq, split), dtype=NBXDtype.float32,
                            device=dev)
    out = NBXTensor.empty((BHq, 1, D_v), dtype=out_dtype, device=dev)

    _set_device(q)
    if _os_dv.environ.get("NBX_DV_GROUPED", "0") == "1":
        # One KV head per program: K/V loaded once, reused across the
        # GQA group from registers (see the grouped kernel's docstring).
        from nbxtensor.ops.decode_attn_vec import decode_attn_vec_grouped_kernel
        decode_attn_vec_grouped_kernel[(batch * nheads_k, split)](
            q2, k, v,
            bias if bias is not None else q2,
            opart, mpart, lpart,
            seqlen_k, seg_len, float(softmax_scale),
            q2.stride(0),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            opart.stride(0), opart.stride(1),
            mpart.stride(0),
            GQA_GROUPS=groups, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            D=headdim, D_V=D_v,
            HAS_BIAS=bias is not None,
            Q_TO=q_to, Q_SATURATE=q_saturate,
            num_warps=warps, num_stages=1)
    else:
        decode_attn_vec_split_kernel[(BHq, split)](
            q2, k, v,
            bias if bias is not None else q2,   # dead pointer when HAS_BIAS=0
            opart, mpart, lpart,
            seqlen_k, seg_len, float(softmax_scale),
            q2.stride(0),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            opart.stride(0), opart.stride(1),
            mpart.stride(0),
            GQA_GROUPS=groups, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            D=headdim, D_V=D_v,
            HAS_BIAS=bias is not None,
            Q_TO=q_to, Q_SATURATE=q_saturate,
            num_warps=warps, num_stages=1)
    # Fixed-order deterministic combine — the EXISTING flash_decode
    # reduce kernel viewed at GROUPS=1 (partials [BHq, split, 1, D_v]).
    flash_decode_reduce_kernel[(BHq,)](
        opart, mpart, lpart, out,
        opart.stride(0), opart.stride(1), D_v,
        mpart.stride(0), mpart.stride(1),
        out.stride(0), out.stride(1),
        SPLIT=split, GROUPS=1, BLOCK_G=max(16, _fa_min_tile()),
        BLOCK_D=BLOCK_D, D=D_v,
        num_warps=4)
    return out.reshape(batch, nheads, 1, D_v)


def _flash_decode(q, k, v, bias, softmax_scale,
                  batch, nheads, nheads_k, seqlen_k, headdim):
    """Split-KV flash-decoding launch pair. See ops/flash_decode.py.

    q [B, H, 1, D] fp16 · k/v [B, H_kv, T_k, D] fp16 (native cache
    layout) · bias flat [T_k] additive fp32/fp16 or None.

    GQA by grouping Q inside each KV head: [B, H, 1, D] viewed as
    [B*H_kv, GROUPS, D] is a pure view (flat order b, h_kv, g, d).

    The split count targets SM occupancy: enough programs that
    B*H_kv*SPLIT covers the 80 SMs, capped so each split still has a
    few BLOCK_N chunks to amortise its prologue.
    """
    import os as _os_fd2
    from nbxtensor.ops.flash_decode import (flash_decode_split_kernel,
                                   flash_decode_reduce_kernel)
    groups = nheads // nheads_k
    BH = batch * nheads_k
    # Same capability as the prefill path: a decode step at one query row
    # still forms a `tl.dot` whose tile dimension must be one the backend
    # computes correctly. `seqlen_q=1, headdim=64` reached the refusal here
    # after the prefill floor was already raised -- the small dimension was
    # BLOCK_G, not BLOCK_M, and the refusal names neither.
    BLOCK_G = max(triton.next_power_of_2(groups), _fa_min_tile())
    BLOCK_D = max(triton.next_power_of_2(headdim), _fa_min_tile())
    # K and V tiles are [BLOCK_N, BLOCK_D] fp16, double-buffered by the
    # pipeliner: BLOCK_N * BLOCK_D * 2 B * ~4 buffers must fit Volta's
    # 96 KB. At BLOCK_D<=128, BLOCK_N=128 uses ~
    # 128*128*2*4 = 128 KB requested pre-pipelining but measured to fit;
    # at BLOCK_D=256 the same BLOCK_N requests 147 KB and refuses to
    # launch (caught by the day-one oracle at D=129). Halve until it
    # fits the 96 KB budget.
    # Defaults = the measured best of the 2026-08-23 10-config sweep
    # (BN=64/SEG=512: 0.47 ms/call at T_k=4,352 vs 2.0 ms at the former
    # BN=128 default, whose 128x128 double-buffered tile chokes Volta
    # smem/occupancy). The kernel remains opt-in and remains SLOWER
    # than the math path on the row even at this config (see the flag
    # site) — but whoever arms it gets its best self.
    BLOCK_N = int(_os_fd2.environ.get("NBX_FD_BLOCK_N", "64")) \
        if BLOCK_D <= 128 else (32 if BLOCK_D <= 256 else 16)
    split = max(1, min(int(triton.cdiv(
        seqlen_k, int(_os_fd2.environ.get("NBX_FD_SEG", "512")))), 64))
    seg_len = int(triton.cdiv(seqlen_k, split))

    qg = q.reshape(BH, groups, headdim)
    kf = k.reshape(BH, seqlen_k, headdim)
    vf = v.reshape(BH, seqlen_k, headdim)

    dev = f"cuda:{q._device_idx}"
    opart = NBXTensor.empty((BH, split, groups, headdim),
                            dtype=NBXDtype.float32, device=dev)
    mpart = NBXTensor.empty((BH, split, groups), dtype=NBXDtype.float32,
                            device=dev)
    lpart = NBXTensor.empty((BH, split, groups), dtype=NBXDtype.float32,
                            device=dev)
    out = NBXTensor.empty((BH, groups, headdim), dtype=q.nbx_dtype,
                          device=dev)

    _set_device(q)
    flash_decode_split_kernel[(BH, split)](
        qg, kf, vf,
        bias if bias is not None else qg,   # dead pointer when HAS_BIAS=0
        opart, mpart, lpart,
        seqlen_k, seg_len, float(softmax_scale),
        qg.stride(0), qg.stride(1),
        kf.stride(0), kf.stride(1),
        opart.stride(0), opart.stride(1), opart.stride(2),
        mpart.stride(0), mpart.stride(1),
        GROUPS=groups, BLOCK_G=BLOCK_G, BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D, D=headdim,
        HAS_BIAS=bias is not None,
        num_warps=4, num_stages=2)
    flash_decode_reduce_kernel[(BH,)](
        opart, mpart, lpart, out,
        opart.stride(0), opart.stride(1), opart.stride(2),
        mpart.stride(0), mpart.stride(1),
        out.stride(0), out.stride(1),
        SPLIT=split, GROUPS=groups, BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D, D=headdim,
        num_warps=4)
    return out.reshape(batch, nheads, 1, headdim)


def _flash_m8n8k4(q, k, v, bias, softmax_scale, gqa_groups, tile, has_bias, split=False):
    """Flash forward on the m8n8k4 matrix unit (ops/flash_attention_m8n8k4.py); `bias` is the wrapper's
    memory-resident [B, H, Sq, Sk] bias view (unit stride on the key axis), not read when `has_bias` is False (the
    zero bias of a call with neither mask nor causality). `split`: the operands carried as hi + lo (the DtypeEngine's
    "split" representation, `launch_keys.unit_flash`), `tile` then a `matrix_unit.fp32_split.flash` row whose
    `block_k` head-dim columns the unit sums before each round-to-nearest add."""
    from nbxtensor.ops.flash_attention_m8n8k4 import flash_attention_m8n8k4_kernel, head_parts
    from nbxtensor.ops.mma_m8n8k4 import UNIT
    from nbxtensor.host import need as _need   # the application's triton.dtype (nbxtensor.host)
    matrix_unit_split_scale = _need("matrix_unit_split_scale")
    batch, nheads, seqlen_q, headdim = q.shape
    seqlen_k = k.shape[2]
    DA, DB = head_parts(headdim)
    hi_exp, lo_shift = matrix_unit_split_scale(_matrix_unit())
    ksa = min(tile["block_k"], DA) // UNIT["k"] if split else 1       # instructions per round-to-nearest add
    ksb = min(tile["block_k"], DB) // UNIT["k"] if split and DB else 1
    o = NBXTensor.empty_like(q)
    # the LSE rows rounded to the largest query tile a candidate takes (the tile is the certified one of this key)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')
    _set_device(q)
    grid = lambda META: (triton.cdiv(seqlen_q, 16 * META["MB"]), batch * nheads)
    flash_attention_m8n8k4_kernel[grid](
        q, k, v, bias, o, lse, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0), bias.stride(1), bias.stride(-2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
        _lk.bucket_of("M", seqlen_q), _lk.bucket_of("N", seqlen_k), DA=DA, DB=DB, GQA_GROUPS=gqa_groups, HAS_BIAS=bool(has_bias),
        SPLIT=bool(split), HI_EXP=hi_exp, LO_SHIFT=lo_shift, KSA=ksa, KSB=ksb)
    return o


def scaled_dot_product_attention_wrapper(q, k, v, attn_mask=None,
                                          dropout_p=0.0, is_causal=False,
                                          scale=None, k_pre_transposed=None,
                                          **kwargs):
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
    from nbxtensor.ops.flash_attention import flash_attention_forward_kernel

    _set_device(q)

    # Fix pre-transposed K from graph math decomposition path.
    #
    # PyTorch's SDPA math path does bmm(Q, K.transpose(-2,-1)). When the
    # tracer captures this, the graph stores K already transposed to
    # (batch, heads, headdim, seq). Our Flash Attention kernel expects K
    # in standard (batch, heads, seq, headdim) format — same as Q and V.
    #
    # `k_pre_transposed` is the graph's answer, recorded at load by
    # GraphExecutor._mark_sdpa_k_layout and passed down by the dispatcher.
    # It exists because the shape test below cannot decide the square case:
    # when the sequence length equals the head dimension both layouts have
    # the same shape, and the test then declined to transpose and attention
    # ran with K's axes crossed. Measured 2026-09-07 on Apple — TinyLlama,
    # head_dim 64, a 64-token prompt: argmax 29892 at logit 6.41 where the
    # float64 oracle says 3864 at 22.36, |delta| 23.27 across the vocabulary,
    # with 63 and 65 tokens exact to 0.02. The correction is inert wherever
    # the graph is unambiguous, which is every other length.
    if k.ndim == 4 and q.ndim == 4:
        if k_pre_transposed is not None:
            if k_pre_transposed:
                k = k.transpose(2, 3).contiguous()
        elif (k.shape[2] == q.shape[3] and k.shape[3] == q.shape[2]
                and k.shape[2] == q.shape[2]):
            raise RuntimeError(
                f"scaled_dot_product_attention: K is square "
                f"{tuple(k.shape)} with seq_len == head_dim, so its layout "
                f"cannot be read from its shape, and no k_pre_transposed was "
                f"passed. Refusing to guess.")
        elif (k.shape[2] == q.shape[3]      # K's "seq" dim == Q's headdim
                and k.shape[3] == q.shape[2]):  # K's "dim" dim == Q's seqlen
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

    # Q is materialised unconditionally (tiny at decode; graph-contiguous
    # at prefill, so free). K and V are NOT: the bucketed KV cache hands
    # out prefix-slice views `buffer[:, :, :padded, :]` that are
    # non-contiguous in the T dim, and the unconditional `.contiguous()`
    # here copied the ENTIRE padded K and V per layer per decode step —
    # 24 MB/token at bucket 256, scaling with context (copies-at-source
    # census, 2026-08-23). The math path (decode's route) now consumes
    # the views directly (`merge_dims` + strided-B bmm); the flash paths
    # materialise at their own entry below, exactly as before.
    q = q.contiguous()

    # ---- FLASH-DECODING (T_q == 1) ------------------------------------
    #
    # Decode used to fall through to the math budget below, which at
    # T_q=1 always chose `_math_attention` — bmm/softmax/bmm with
    # materialised scores, the largest single kernel of the captured
    # step at a working context (9.3 ms/token at 4,164, linear in T_k).
    #
    # This routes single-query attention to the split-KV flash-decoding
    # pair instead: the KV sequence is split across programs so the SMs
    # fill even at batch 1, each split combines under online softmax,
    # and a fixed-order second pass merges the splits — deterministic by
    # construction (replay verifies plans byte-equal, so run-to-run
    # variance would be disqualifying).
    #
    # Routing is correctness-preserving: any shape or mask this path
    # does not model falls through to the existing paths unchanged. The
    # decode mask from the bucketed KV cache is additive [1,1,1,T_k];
    # anything not flattenable to [T_k] falls through.
    # DEFAULT OFF, by measurement (2026-08-22): correct and deterministic
    # (oracle green at D=127/128/129), but SLOWER than the math path on
    # this card at real bucket sizes — paired, machine exclusive:
    #   256-bucket   9.074 -> 8.167 tok/s  (-10.0%)
    #   4,352-bucket 7.568 -> 6.512        (-14.0%)  best tuning 7.419
    #   8,448-bucket 2.747 -> 2.841        (+3.4%, n=1)
    # VERDICT CORRECTED (2026-08-23, supervisor): the math path is not
    # "full", it is SLOW — at a 4,164 context it reads ~409 MB of KV in
    # 9.3 ms = 44 GB/s = 5% of the 900 GB/s HBM. The split pair lost to
    # it anyway (its own overheads are larger still at these buckets).
    # RE-JUDGED under the pinned protocol (2026-08-23, bench_row.py,
    # interleaved arms, n=5, published ffn-only artifact, at the 93%-GPU
    # step where a GPU-side change is no longer masked by host time —
    # HEAD 5cc7362):
    #   default cfg (BN=128): 4,164 ctx x0.560 (-44.0%) · short x0.754
    #     — BLOCK_N=128 is the pathology (the 128x128 double-buffered
    #     fp16 tile chokes Volta smem/occupancy: 2-6 ms/call isolated).
    #   best cfg (BN=64, SEG=512, from a 10-config isolated sweep):
    #     4,164 ctx x0.857 (-14.3%) · short x0.968 (-3.2%), no overlap.
    # The kernel REPLAYS and graph-captures fine (plan frozen, verified
    # byte-equal), and its ROW OUTPUTS were byte-equal to the math
    # path's (every armed rep reproduced the math arm's majority sha
    # over 60 greedy tokens) — the loss is its own GPU time: ~0.47
    # ms/call = ~22.7 ms/token vs the in-engine autotuned math band's
    # ~14 ms. The
    # isolated math figure (1.47 ms/call) was 5x SLOWER than in-engine
    # — eager wrapper overhead + cold autotune; the FD isolated figures
    # transferred. Third refusal, first two protocol-grade.
    # THE BAR STANDS: ~0.5 ms per token at 4k context (the KV bytes at
    # ~900 GB/s). This split-KV shape cannot reach it on sm_70 (fp32
    # partials + reduce pass + 16-row G-tile carrying 4 real rows, no
    # cp.async pipeline) — reaching the bar needs a different kernel,
    # a named follow-up, not a tuning of this one.
    import os as _os_fd
    # NBX_DECODE_VEC three-state contract: "1" = ARMED here regardless
    # of the math routing below (diagnostics + the oracle, which runs
    # in profile-less processes where the budget is 0); unset = the
    # kernel is the DEFAULT Tq=1 refinement INSIDE the math route
    # below; "0" = kill switch everywhere.
    # The KV cache asks for Q in the cache's dtype (`q_dtype_of_kv`): the vector
    # decode kernel rounds Q to it on load (the stored cast's protected
    # conversion, then the same fp32 math), so the per-layer cast copy of Q
    # (TinyLlama: 22 a token) is gone on the decode route; every other route
    # casts once, here below, where it diverges from the vec kernel.
    q_round = None                       # an NBXDtype (`.dtype` is the Triton element type)
    if kwargs.get("q_dtype_of_kv") and q._dtype != k._dtype:
        q_round = k._dtype

    def _q_cast(t):
        return t.to(q_round) if (q_round is not None and t._dtype != q_round) else t

    if (seqlen_q == 1 and q._device != "cpu"
            and _os_fd.environ.get("NBX_DECODE_VEC", "") == "1"):
        _dv_out = _ew._try_decode_vec(q, k, v, attn_mask, softmax_scale,
                                  batch, nheads, nheads_k, seqlen_k,
                                  headdim, q_round=q_round)
        if _dv_out is not None:
            return _dv_out
    if (seqlen_q == 1 and q._device != "cpu"
            and _os_fd.environ.get("NBX_FLASH_DECODE", "0") == "1"
            and headdim == v.shape[3]):
        _fd_bias = None
        _fd_ok = True
        if attn_mask is not None:
            if attn_mask.numel() == seqlen_k:
                _fd_bias = attn_mask.reshape(seqlen_k)
            else:
                _fd_ok = False
        if _fd_ok:
            # This opt-in path keeps its measured behavior: flat layout.
            return _flash_decode(_q_cast(q), k.contiguous(), v.contiguous(),
                                 _fd_bias, softmax_scale,
                                 batch, nheads, nheads_k, seqlen_k, headdim)

    # The kernels require matching operand dtypes; the fp32-stored GEMMs (Volta's store policy,
    # M<=4 decode) can leave q/k/v mixed (Sana: fp32 Q vs fp16 K; Allegro: fp32 V). If they
    # disagree, all three take the narrowest — the compiled engine's alignment (R30). For the
    # common case where all three match, this is a no-op — zero overhead.
    _qd, _kd, _vd, q_round = _lk.sdpa_operand_dtypes(q._dtype, k._dtype, v._dtype, q_round)
    if _qd != q._dtype or _kd != k._dtype or _vd != v._dtype:
        q, k, v = q.to(_qd), k.to(_kd), v.to(_vd)

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
    _use_math_forced = _os_fma.environ.get("NBX_FORCE_MATH_ATTENTION") == "1"
    headdim_v = v.shape[3]
    # The route is `launch_keys.sdpa_route` — the function the derived census keys with: math
    # when forced or when the value head dim differs; else by the fp32 scores' size against this
    # device's budget (a non-power-of-two head dim routes on `sdpa.non_pow2_head_scores_bytes`
    # when the arch declares no budget); over it, chunked when the rows fit the arch's ceiling; otherwise flash.
    _q_dev_idx = getattr(q, "_device_idx", None)
    _route, _chunk_rows = _lk.sdpa_route(
        batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v,
        _sdpa_math_scores_budget_bytes_for(_q_dev_idx), _sdpa_math_min_chunk_rows(),
        _sdpa_math_max_chunks(), force_math=_use_math_forced,
        unit_flash=_lk.unit_flash_takes(headdim, q._dtype if q_round is None else q_round, k._dtype, v._dtype))
    _use_math = {"math": True, "chunked": "chunked", "flash": False}[_route]
    if _os_fma.environ.get("NBX_SDPA_ROUTE_DIAG") == "1":
        # Print once PER DISTINCT EXECUTING DEVICE, not once per process:
        # on heterogeneous rigs the budget is per-device, so a single line
        # (which lands on whichever device runs the first SDPA) cannot
        # show the routing on the other memory class. dev_idx is the
        # activation instrument — None here means the call site failed to
        # extract the device index and the per-device cap silently fell
        # back to the per-arch base budget.
        _diag_idx = getattr(q, "_device_idx", None)
        _seen = getattr(scaled_dot_product_attention_wrapper,
                        "_route_diag_seen", None)
        if _seen is None:
            _seen = set()
            scaled_dot_product_attention_wrapper._route_diag_seen = _seen
        if _diag_idx not in _seen:
            _seen.add(_diag_idx)
            _pr = _core.get_hardware_profile()
            _dv = getattr(_pr, "devices", None) if _pr is not None else None
            print(f"[NBX_SDPA_ROUTE_DIAG] hd={headdim} hdv={headdim_v} pow2={_ew._is_power_of_2(headdim)} "
                  f"B={batch} H={nheads} Tq={seqlen_q} Tk={seqlen_k} "
                  f"scores_bytes={batch*nheads*seqlen_q*seqlen_k*4} "
                  f"dev_idx={_diag_idx} qtype={type(q).__name__} "
                  f"budget={_sdpa_math_scores_budget_bytes_for(_diag_idx)} "
                  f"profile={'None' if _pr is None else getattr(_pr,'vendor','?')+'/'+(getattr(_dv[0],'architecture','?') if _dv else '?')} "
                  f"use_math={_use_math}", flush=True)
    if _use_math == "chunked":
        return _math_attention_chunked(_q_cast(q), k, v, attn_mask, is_causal,
                                       softmax_scale, _chunk_rows)
    if _use_math:
        # ---- VECTOR (SIMT) DECODE ATTENTION — DEFAULT on the math
        # route at Tq == 1 (ADOPTED 2026-08-23, three-outcome verdict
        # under the locked pinned protocol, interleaved arms, n=5, no
        # overlap, outputs byte-identical to the math chain at all
        # three contexts):
        #   short (256 bucket)   21.538 -> 22.759 tok/s  (+5.7%)
        #   4,164-token context  17.033 -> 19.868        (+16.6%)
        #   ~8,200-token context 13.043 -> 16.972        (+30.1%)
        # The gain grows with T, as the KV-streaming structure predicts
        # (see ops/decode_attn_vec.py for the sourced design). Scoped
        # INSIDE the math route: archs whose vendor yml defines no math
        # budget never reach here at pow2 dims (R23 — non-Volta
        # attention unchanged). NBX_FORCE_MATH_ATTENTION=1 keeps its
        # exact meaning (the pure bmm+softmax chain) and bypasses this.
        # NBX_DECODE_VEC=0 is the kill switch.
        if (seqlen_q == 1 and not _use_math_forced
                and _os_fd.environ.get("NBX_DECODE_VEC", "") != "0"):
            _dv_out = _ew._try_decode_vec(q, k, v, attn_mask, softmax_scale,
                                      batch, nheads, nheads_k,
                                      seqlen_k, headdim, q_round=q_round)
            if _dv_out is not None:
                return _dv_out
        return _math_attention(_q_cast(q), k, v, attn_mask=attn_mask,
                                is_causal=is_causal, scale=softmax_scale)

    # Flash path from here on: flat-indexed kernel — materialise the K/V
    # views that the math path above consumes strided (see the entry
    # comment). Contiguous tensors short-circuit at zero cost.
    q = _q_cast(q)
    k = k.contiguous()
    v = v.contiguous()

    # ---- P-FLASH-D128-CORRECTNESS: exact detour around a wrong kernel ----
    #
    # The Triton flash kernel returns WRONG answers when headdim is a
    # power of two >= 128 — measured against a float64 reference:
    #
    #     headdim 127 -> 5.0e-04     headdim 128 -> 4.2e-01  (208x bound)
    #     headdim 129 -> 3.1e-04     headdim 256 -> 4.1e-01
    #
    # 127 and 128 compile the SAME kernel source with the SAME tile and
    # the SAME block sizes; the divergence is in Triton's codegen for
    # that specialisation, and it is a known upstream class (Dao-AILab's
    # own flash_attn_triton.py: "Triton bug ... we get the wrong
    # output!", tested only on A100 — we run sm_70). Root cause needs a
    # version bisect / PTX diff and is tracked separately; it must not
    # keep a wrong kernel in production in the meantime.
    #
    # The detour is ZERO-PADDING the head dimension by one. Padded
    # components contribute exactly nothing to a dot product, so
    # Q·K is unchanged, the softmax is unchanged, and the extra output
    # columns are zero and discarded: the algebra is EXACT, not
    # approximate. Measured 3.35e-01 -> 5.95e-04, i.e. inside the bound
    # and equal to the math path's own accuracy.
    #
    # NOTE ON THE SCALE, which is the trap here: `softmax_scale` is
    # computed from the TRUE headdim above and is passed through
    # unchanged. Letting the kernel derive 1/sqrt(headdim) from the
    # padded value would silently rescale every score.
    #
    # A half-and-half split of the head dimension was evaluated first and
    # is NOT viable: the softmax needs the COMPLETE score, so two
    # 64-wide flash calls produce softmaxes of half dot products at the
    # wrong scale, which no later operation can recombine. Splitting only
    # works if the scores are materialised first, which is the math path.
    # The condition is `launch_keys.flash_headdim_detour` — the function the derived census
    # re-routes the padded call with, so the keys it derives are this detour's.
    _hd_broken = _lk.flash_headdim_detour(
        headdim, _os_fd.environ.get("NBX_D128_DETOUR", "1") != "0") != headdim
    # NBX_D128_DETOUR=0 disables the zero-pad detour — DIAGNOSTIC ONLY:
    # it re-exposes the wrong kernel. Exists to measure the detour's true
    # cost (TTFT at long prefill) and for the day the upstream codegen
    # defect is fixed, when the correctness oracle's _KNOWN_BROKEN
    # mechanism will demand this whole block be removed.
    if _hd_broken:
        _pad = _lk.flash_headdim_detour(headdim) - headdim
        def _pad_hd(t):
            z = NBXTensor.zeros(tuple(t.shape[:-1]) + (_pad,),
                                dtype=t._dtype, device=f"cuda:{t._device_idx}")
            return NBXTensor.cat([t, z], dim=-1).contiguous()
        _dv = v.shape[-1]
        _kw = {k_: v_ for k_, v_ in kwargs.items() if k_ != "k_pre_transposed"}
        out = scaled_dot_product_attention_wrapper(
            _pad_hd(q), _pad_hd(k), _pad_hd(v), attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal,
            scale=softmax_scale,
            # K was normalised above; saying so keeps the inner call from
            # re-reading a layout off a shape it can no longer interpret.
            k_pre_transposed=False, **_kw)
        return out.narrow(-1, 0, _dv).contiguous()

    # Adaptive BLOCK_M (Phase 1): decode-path seqlen_q is typically 1–4. Using
    # BLOCK_M=128 wastes >99% of the Q tile. 16 is the floor enforced by
    # tl.dot (TensorCore tile minimum) — below that the kernel crashes.
    # Block sizes — adapted from Dao-AILab defaults.
    # Reduce BLOCK_M for large headdim to stay within shared memory limits.
    # V100 has 96KB shared memory. With headdim=256: 128×256×2×3 = 192KB > 96KB.
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    # The floor is a BACKEND CAPABILITY, not 16. A backend whose generic
    # attention lowering mis-computes below 32 refuses the kernel, and a
    # refusal wastes the whole model where a padded tile only wastes half a Q
    # tile at seqlen_q=1. On cuda and hip this reads 16 and nothing moves.
    _fa_floor = _fa_min_tile()
    if seqlen_q <= 16:
        BLOCK_M = max(16, _fa_floor)
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
        if not _core._NBX_HAS_NATIVE_BF16:
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

    # Clamp to what the HARDWARE PROFILE allows for this shape.
    #
    # The cascade above proposes; the profile is the ceiling. It stays a
    # clamp and never an instruction because the cascade carries measurements
    # the profiles do not — BLOCK_M=64 is the floor of CORRECTNESS on Volta
    # at head_dim<128, where volta.yml declares 128 — and a ceiling cannot
    # undo that.
    #
    # Why it is needed at all: every branch above decides on
    # `_NBX_HAS_NATIVE_BF16`, which separates Volta from Ampere and means
    # nothing off CUDA. On an Apple GPU the cascade therefore took the
    # Ampere branch and asked for a 128-row Q tile — four times the 32 that
    # apple_silicon.yml has declared since it was written, over the 32 KB
    # threadgroup budget in the same file, and past what the Metal backend
    # will lower at all. The profile had the answer; nothing read it.
    #
    # MEASURED 2026-09-06: no NVIDIA tile changes (min(64,128)=64 on Volta,
    # min(128,128)=128 on Hopper), so the CUDA rows are untouched.
    _ceil_m, _ceil_n = _sdpa_block_ceiling(seqlen_q, headdim)
    if _ceil_m:
        BLOCK_M = min(BLOCK_M, int(_ceil_m))
    if _ceil_n:
        BLOCK_N = min(BLOCK_N, int(_ceil_n))
    # The ceiling clamps, the floor holds — and the floor is applied LAST.
    #
    # The ceiling is a RESOURCE bound read from the profile; the floor is a
    # CORRECTNESS bound read from the backend, below which the attention
    # lowering refuses (or, worse, would mis-compute). min(32, 16) = 16 is
    # exactly what happened on whisper's decode probe: the wrapper's floor
    # raised BLOCK_M to 32 and the Apple profile's `block_m: 16` decode row
    # clamped it straight back down into the refusal. Below the floor there
    # is nothing to run, so a profile row under it is a CONFLICT — obeyed
    # nowhere, and said once, because a profile that promises a tile the
    # backend refuses should be heard about and fixed, not silently outrun.
    if BLOCK_M < _fa_floor or BLOCK_N < _fa_floor:
        _clamped = (BLOCK_M, BLOCK_N)
        BLOCK_M = max(BLOCK_M, _fa_floor)
        BLOCK_N = max(BLOCK_N, _fa_floor)
        _core._announce_floor_over_ceiling(_clamped, (BLOCK_M, BLOCK_N), _fa_floor)

    # Bias handling — always memory-resident.
    # The kernel only accepts BIAS_TYPE in {"vector", "matrix"} and the
    # IS_CAUSAL constexpr is gone. Every tensor reaching tl.dot must
    # arrive via tl.load to guarantee Triton compiles a single MMA path.
    # Three sub-cases below all converge to BIAS_TYPE="matrix" with a
    # memory-resident bias.
    device_idx = q._device_idx if hasattr(q, '_device_idx') else 0
    _bias_read = attn_mask is not None or bool(is_causal)     # else the zero base below: nothing to add
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
        causal_base = _ew._get_causal_bias(device_idx, seqlen_q, seqlen_k, q.dtype)
        bias = causal_base.unsqueeze(0).unsqueeze(0).expand(
            batch, nheads, seqlen_q, seqlen_k
        )
    else:
        zero_base = _ew._get_zero_bias(device_idx, seqlen_q, seqlen_k, q.dtype)
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

    # The arch's matrix unit, where the profile declares one Triton does not lower `tl.dot` onto
    # (`matrix_unit`, volta.yml): the flash kernel built on that unit, for the operands the DtypeEngine carries
    # on it (its operand dtype natively, an `fp32_split` operand as hi + lo) and the head dims its measured rows
    # cover. Same contract as the kernel below (bias, GQA, LSE, tails).
    # Taken before the tl.dot kernel's own output, LSE and tile buffers are allocated.
    _mu_flash = _lk.unit_flash(headdim, q._dtype, k._dtype, v._dtype)
    if _mu_flash is not None:
        o = _flash_m8n8k4(q, k, v, bias, softmax_scale, gqa_groups, _mu_flash[1], _bias_read,
                          split=_mu_flash[0] == "split")
        if attn_mask is not None:
            o = _ew.nan_to_num_wrapper(o, nan=0.0, posinf=0.0, neginf=0.0)
        return o

    # The tl.dot flash with its tile CERTIFIED per key (workstream 1, roadmap C3) where the profile declares no hand
    # table of tiles (`launch_keys.flash_tiles_certified`): the autotuned handle of the same kernel, keyed on the
    # buckets of the query rows and the keys and on the head dim (`launch_keys.flash_launches`, the census's key);
    # the log-sum-exp and its scratch are sized for the tallest candidate tile, so every candidate's rows fit.
    if _lk.flash_tiles_certified():
        from nbxtensor.ops.flash_attention import flash_attention_certified_kernel, flash_configs
        o = NBXTensor.empty_like(q)
        _rows_top = max(c.kwargs["BLOCK_M"] for c in flash_configs())
        seqlen_q_rounded = math.ceil(seqlen_q / _rows_top) * _rows_top
        _dev = f"cuda:{q._device_idx}" if hasattr(q, "_device_idx") else "cuda"
        lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32, device=_dev)
        tmp = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32, device=_dev)
        _set_device(q)
        flash_attention_certified_kernel[lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)](
            q, k, v, bias,
            o, lse, tmp,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            bias.stride(0), bias.stride(1), bias.stride(-2),
            o.stride(0), o.stride(1), o.stride(2),
            nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
            _bucket_of("M", seqlen_q), _bucket_of("N", seqlen_k),
            BIAS_TYPE=bias_type,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            GQA_GROUPS=gqa_groups,
        )
        if attn_mask is not None:
            o = _ew.nan_to_num_wrapper(o, nan=0.0, posinf=0.0, neginf=0.0)
        return o

    # Output allocation. seqlen_q_rounded must align with actual BLOCK_M.
    o = NBXTensor.empty_like(q)
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')
    tmp = NBXTensor.empty((batch, nheads, seqlen_q_rounded), dtype=NBXDtype.float32,
                          device=f"cuda:{q._device_idx}" if hasattr(q, '_device_idx') else 'cuda')

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    # Launch meta (num_warps / num_stages). Triton auto-selects when not
    # given; on Volta the auto-pick for a small Q tile under-parallelises the
    # softmax reduction (the BLOCK_M=32 wrongness probe). NBX_FLASH_NUM_WARPS
    # / NBX_FLASH_NUM_STAGES let us pin them for diagnosis / per-arch tuning.
    _flash_launch_meta = {}
    # The arch profile's row for this shape: its stated warps (a row without them is an error), and on an FMA-path row (Volta/Turing
    # under Triton >= 3.3) the head-dim chunk the QK^T contraction is done in. No chunk: held Q.
    _fa_meta = _sdpa_launch_meta(seqlen_q, headdim)
    QK_CHUNK = min(_fa_meta["qk_chunk"], BLOCK_HEADDIM) if "qk_chunk" in _fa_meta else 0
    if _fa_meta:
        _flash_launch_meta["num_warps"] = _fa_meta["num_warps"]
    _fnw = os.environ.get("NBX_FLASH_NUM_WARPS")
    if _fnw:
        _flash_launch_meta["num_warps"] = int(_fnw)
    _fns = os.environ.get("NBX_FLASH_NUM_STAGES")
    if _fns:
        _flash_launch_meta["num_stages"] = int(_fns)

    # Ensure CUDA runtime is on the correct device before kernel launch
    _set_device(q)

    # The tile is offered only if the DEVICE says it can hold it.
    #
    # Everything above proposes on `_NBX_HAS_NATIVE_BF16`, which separates Volta
    # from Ampere and says nothing else — so sm_86 takes the branch written for
    # sm_80 and asks a 128-row Q tile of a card declaring 99 KB where sm_80
    # declares 163. That is what stopped every language model on an A40: 164 352
    # bytes requested against 101 376 available, the same number in both Triton
    # modes and unmoved by sequence length. The clamp above could not catch it
    # either: `sdpa_thresholds` exists only in the seven profiles we ship, so for
    # any card whose profile we do not ship it returns (None, None) and clamps
    # nothing.
    #
    # The cost is READ FROM THE COMPILER, never estimated: `prepare()` returns
    # the shared memory Triton computed for this exact specialisation. An
    # analytic formula is a guess — one written while diagnosing this said
    # 112 KB for a Volta tile that runs in 96, and pruning on that would have
    # deleted a configuration that works. One compile per candidate, which the
    # autotuner already pays, and a dict lookup on every launch after.
    _smem_budget = _nbx_launcher.max_shared_memory_per_block()
    if _smem_budget is not None:
        _probe_args = (
            q, k, v, bias, o, lse, tmp, softmax_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            bias.stride(0), bias.stride(1), bias.stride(-2),
            o.stride(0), o.stride(1), o.stride(2),
            nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
            seqlen_q // 32, seqlen_k // 32,
        )
        _probe_kw = dict(BIAS_TYPE=bias_type, BLOCK_HEADDIM=BLOCK_HEADDIM,
                         GQA_GROUPS=gqa_groups, QK_CHUNK=QK_CHUNK, **_flash_launch_meta)

        # `prepare` binds a JITFunction; the kernel is wrapped in
        # @triton.heuristics, whose only job is to add constexprs. Unwrap to
        # the function and apply those heuristics exactly as Heuristics.run
        # does, so the probe compiles the SAME specialisation the launch will.
        _inner, _heur = flash_attention_forward_kernel, {}
        while hasattr(_inner, "values") and hasattr(_inner, "fn"):
            _heur.update(_inner.values)
            _inner = _inner.fn

        def _tile_cost(_t):
            _kw = dict(_probe_kw, BLOCK_M=_t[0], BLOCK_N=_t[1])
            if _heur:
                _named = dict(zip(_inner.arg_names, _probe_args))
                for _v, _h in _heur.items():
                    _kw[_v] = _h({**_named, **_kw})
            _prep, _ = _nbx_launcher.prepare(_inner, _probe_args, _kw)
            return int(_prep.shared)

        # The shrink stops at the backend's floor, not at 16: a candidate the
        # backend refuses is not a candidate.
        _cands, _m = [(BLOCK_M, BLOCK_N)], BLOCK_M
        while _m > max(16, _fa_min_tile()):
            _m //= 2
            _cands.append((_m, BLOCK_N))
        try:
            _sel = _largest_tile_within_smem(_cands, _tile_cost, _smem_budget)
        except Exception as _e:
            # A probe that cannot run changes nothing — but it SAYS SO. A guard
            # that degrades in silence is the defect this whole change is about.
            print(f"[flash] shared-memory probe unavailable ({type(_e).__name__}: "
                  f"{_e}); the tile is not checked against the device", flush=True)
            _sel = (BLOCK_M, BLOCK_N)
        if _sel != (BLOCK_M, BLOCK_N):
            print(f"[flash] tile {(BLOCK_M, BLOCK_N)} needs "
                  f"{_tile_cost((BLOCK_M, BLOCK_N))} bytes of shared memory and this "
                  f"device declares {_smem_budget} — using {_sel}", flush=True)
            BLOCK_M, BLOCK_N = _sel
            # Everything downstream that was derived from the proposed tile.
            seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
            _dev = f"cuda:{q._device_idx}" if hasattr(q, "_device_idx") else "cuda"
            lse = NBXTensor.empty((batch, nheads, seqlen_q_rounded),
                                  dtype=NBXDtype.float32, device=_dev)
            tmp = NBXTensor.empty((batch, nheads, seqlen_q_rounded),
                                  dtype=NBXDtype.float32, device=_dev)
            grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

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
        QK_CHUNK=QK_CHUNK,
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
        o = _ew.nan_to_num_wrapper(o, nan=0.0, posinf=0.0, neginf=0.0)
    return o
