"""Linear-algebra wrappers: matmul, bmm, addmm, baddbmm, addmv, mv, dot, gemv. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import wrappers as _core

# gridDim.z architectural bound — 65535 on both CUDA and HIP (an API
# limit shared by every supported arch, not a tunable hardware param).
# Batched matmul launches put the batch on grid axis 2 and chunk past it.
_GRID_Z_MAX = 65535


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
    # The decision is `launch_keys.matmul_out_dtype` — the one the derived census computes keys
    # with (NBX_FORCE_FP32_ACCUM first, then the hardware gate, then the M / force_fp32 gate).
    return _lk.matmul_out_dtype(dt, M, force_fp32, _core._NBX_HAS_NATIVE_BF16, NBX_FORCE_FP32_ACCUM,
                                contract_half=bool(_core._NBX_ACTIVATIONS_FP16_SAFE))


def _apply_matmul_epilogue(c, epilogue_code: int):
    """Unfused epilogue application for matmul fallback paths (decode
    M <= 4 mv/addmv routing). Calls the SAME standalone wrappers the
    unfused graph would dispatch, in the same order — byte-identical to
    the unfused pair by construction, zero-risk on the decode path.
    Codes mirror kernels/ops/matmul.py EPILOGUE_*."""
    if epilogue_code == 0:
        return c
    if epilogue_code == 1:
        return _ew.silu(c)
    return _ew.gelu(c, approximate=('tanh' if epilogue_code == 3 else 'none'))


_DEQUANT_GEMV_MIN_PROGRAMS = None  # lazy: vendor-yml read once per process


def _dequant_gemv_block_n(N: int) -> int:
    """Occupancy-adapted BLOCK_N for the M=1 dequant-GEMV grid.

    The tier's pinned parity BLOCK_N under-fills the device at small N
    (the MoE router mv is N=128 -> a 2-program grid on 80 SMs, ~38 us
    latency-bound). Halve BLOCK_N (pow2, floor 1) until the grid
    reaches `block_sizes.dequant_gemv.min_programs` from the vendor
    arch YAML (R7/R24). BLOCK_N only partitions OUTPUT columns; the
    parity oracle runs the SAME adapted config, so fused == ref by
    construction at every width. Key absent / 0 / any config error ->
    the pinned BLOCK_N (prior behavior). Large-N grids that already
    meet the target are untouched.
    """
    from nbxtensor.ops.dequant_gemv import BLOCK_N
    global _DEQUANT_GEMV_MIN_PROGRAMS
    if _DEQUANT_GEMV_MIN_PROGRAMS is None:
        target = 0
        if os.environ.get("NBX_DGEMV_ADAPT", "1") == "0":
            # Kill switch: pinned width everywhere (pre-adaptation
            # behavior). Cached like the resolved target.
            _DEQUANT_GEMV_MIN_PROGRAMS = 0
            return BLOCK_N
        try:
            prof = _core.get_hardware_profile()
            devices = getattr(prof, "devices", None) if prof else None
            if devices:
                from nbxtensor.host import need as _need   # the application's core.config.loader (nbxtensor.host)
                get_vendor_config = _need("vendor_config")
                _brand = devices[0].brand
                _vendor = getattr(_brand, "value", _brand)
                cfg = get_vendor_config(_vendor, devices[0].architecture)
                target = int(cfg.get("block_sizes", {})
                             .get("dequant_gemv", {})
                             .get("min_programs", 0))
        except Exception:
            target = 0
        _DEQUANT_GEMV_MIN_PROGRAMS = max(0, target)
    bn = BLOCK_N
    target = _DEQUANT_GEMV_MIN_PROGRAMS
    if target:
        while bn > 1 and -(-N // bn) < target:
            bn //= 2
    return bn


def _mm_quantized(a, qt, _epilogue: int = 0):
    """mm where B is an int4-g128-asym QuantizedTensor (weight triplet).

    The graph's `t(weight) -> mm` marks the triplet transposed — which
    IS the packed layout's natural [K, N] orientation. Decode (M == 1)
    runs the fused byte-gated dequant-GEMV with the family's pinned
    parity config; prefill (M > 1) dequantizes to a transient dense
    tensor via the family's standalone kernel (the byte oracle's own
    dequant) and re-enters the normal mm path. Output dtype mirrors
    the dense path (a's dtype).
    """
    import triton as _tr
    from nbxtensor.ops.dequant_gemv import (
        BLOCK_K, BLOCK_N, NUM_WARPS, GROUP_SIZE, PACK,
        dequant_gemv_int4_kernel, dequant_int4_kernel)
    if not qt.transposed:
        raise NotImplementedError(
            "quantized mm without the graph-side transpose marker is "
            "outside the v1 contract (the packed layout is [K, N]; "
            "the traced linear always transposes) — named follow-up.")
    a = _core._ensure_cuda(a)
    K, N = qt.shape
    M = a.shape[0] if a.ndim == 2 else 1
    wq, sc, mn = qt.qweight, qt.scales, qt.qmins
    if hasattr(a, '_device_idx') and a._device_idx != wq._device_idx:
        a = _core._transfer_to_device(a, wq._device_idx)
    _set_device(wq)

    if a.ndim == 2 and M == 1 or a.ndim == 1:
        x = a.contiguous().view(-1)
        out32 = NBXTensor.empty((N,), dtype=NBXDtype.float32,
                                device=a.device)
        bn = _dequant_gemv_block_n(N)
        dequant_gemv_int4_kernel[(_tr.cdiv(N, bn),)](
            x, wq, sc, mn, out32, K, N,
            wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
            BLOCK_K_C=BLOCK_K, BLOCK_N_C=bn,
            GROUP_C=GROUP_SIZE, PACK_C=PACK,
            num_warps=NUM_WARPS, num_stages=2)
        out = out32.to(a.nbx_dtype) if a.nbx_dtype != NBXDtype.float32 \
            else out32
        return out.view(1, N) if a.ndim == 2 else out

    # Prefill: transient dense via the family's standalone dequant.
    dense32 = NBXTensor.empty((K, N), dtype=NBXDtype.float32,
                              device=a.device)
    dequant_int4_kernel[(_tr.cdiv(K, BLOCK_K), _tr.cdiv(N, BLOCK_N))](
        wq, sc, mn, dense32, K, N,
        wq.stride(0), wq.stride(1), sc.stride(0), sc.stride(1),
        dense32.stride(0), dense32.stride(1),
        BLOCK_K_C=BLOCK_K, BLOCK_N_C=BLOCK_N,
        GROUP_C=GROUP_SIZE, PACK_C=PACK,
        num_warps=NUM_WARPS, num_stages=2)
    dense = dense32.to(a.nbx_dtype) if a.nbx_dtype != NBXDtype.float32 \
        else dense32
    return mm(a, dense, _epilogue)


def _gemm_unit_tile(a, b, M: int, N: int):
    """The matrix unit's tile for this GEMM, with how it carries each operand (`split_a` / `split_b`: the
    DtypeEngine's representation, `launch_keys.matrix_unit_operands`), else None. Two native operands take a
    `matrix_unit.mm` row (their products are exact in the fp32 accumulator, as on the tl.dot path); a split one
    takes a `matrix_unit.fp32_split.mm` row (its three terms hold three accumulators)."""
    reps = _lk.matrix_unit_operands(a.nbx_dtype, b.nbx_dtype, M)   # the census keys with the same function
    if reps is None:
        return None
    mu = _matrix_unit()
    split = "split" in reps
    tile = _matrix_unit_mm_tile(M, N, mu["fp32_split"] if split else mu)
    tile["split_a"], tile["split_b"] = reps[0] == "split", reps[1] == "split"
    return tile


def _gemm_m8n8k4(a, b, bias, c, alpha, beta, promote_bias, epilogue, tile, bias_strides=None):
    """C = A @ B (+ beta * bias, alpha scaling) on the m8n8k4 matrix unit (ops/matmul_m8n8k4.py). 2-D or batched
    3-D operands: `a` row-major per batch, `b` walked by its strides, `bias` None or broadcast by `bias_strides`
    (batch, row, column; default a [N] / [1, N] row). The batch runs on grid axis 1, chunked at _GRID_Z_MAX."""
    from nbxtensor.ops.matmul_m8n8k4 import matmul_m8n8k4_kernel, mm_layouts
    batched = a.ndim == 3
    Z = a.shape[0] if batched else 1
    M, K = a.shape[-2:]
    N = b.shape[-1]
    sa = (a.stride(0) if batched else 0, a.stride(-2), a.stride(-1))
    sb = (b.stride(0) if batched else 0, b.stride(-2), b.stride(-1))
    sc = (c.stride(0) if batched else 0, c.stride(-2), c.stride(-1))
    sx = (0, 0, bias.stride(-1)) if bias is not None and bias_strides is None else (bias_strides or (0, 0, 0))
    MB, NB = tile["block_m"] // 16, tile["block_n"] // 16
    b_k_contig = sb[1] == 1 and N > 1
    lay = mm_layouts(MB, NB, tile["warps_m"], tile["warps_n"], tile["block_k"], b_k_contig)
    from nbxtensor.dtype.operands import matrix_unit_split_scale   # the library's operand rules
    hi_exp, lo_shift = matrix_unit_split_scale(_matrix_unit())
    _set_device(a)
    for z0 in range(0, Z, _GRID_Z_MAX):
        zb = min(_GRID_Z_MAX, Z - z0)
        grid = (triton.cdiv(M, tile["block_m"]) * triton.cdiv(N, tile["block_n"]), zb)
        x = c if bias is None else bias
        matmul_m8n8k4_kernel[grid](
            a if z0 == 0 else a.narrow(0, z0, zb), b if z0 == 0 else b.narrow(0, z0, zb),
            x if z0 == 0 or sx[0] == 0 else x.narrow(0, z0, zb), c if z0 == 0 else c.narrow(0, z0, zb),
            M, N, K, *sa, *sb, *sc, *sx,
            float(alpha), float(beta),
            HAS_BIAS=bias is not None, PROMOTE_BIAS=bool(promote_bias), EPILOGUE=int(epilogue),
            B_K_CONTIG=b_k_contig, SPLIT_A=tile["split_a"], SPLIT_B=tile["split_b"], HI_EXP=hi_exp,
            LO_SHIFT=lo_shift, MB=MB, NB=NB, BK=tile["block_k"], GROUP_M=tile["group_m"],
            num_warps=tile["warps_m"] * tile["warps_n"], **lay)


def mm(a, b, _epilogue: int = 0) :
    """Matrix multiplication: C = A @ B.
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles fp16/bf16 overflow protection externally.

    `_epilogue` (fusion_vertical pass): fused elementwise epilogue code
    (0 none, 1 silu, 2 gelu exact, 3 gelu tanh) executed in-kernel with
    per-stage rounding emulation — see kernels/ops/matmul.py. The
    M <= 4 decode path stays UNFUSED (mv routing + standalone epilogue
    wrapper) so decode bytes are untouched by construction.

    Small-M fast path (M <= 4): route to mv_wrapper per-row. matmul_kernel
    tiles on BLOCK_M=64 and wastes 63/64 threads when M=1 (decode). mv_kernel
    parallelises on N only and is ~10–15× faster in that regime. Universal:
    works for LLM decode (M=1), short-sequence audio, small-batch inference.
    For M > 4, stays on matmul_kernel which is optimal for larger GEMM
    (diffusion spatial mm with M in the thousands).
    """
    from nbxtensor.quant.quantized_tensor import QuantizedTensor
    if isinstance(b, QuantizedTensor):
        return _mm_quantized(a, b, _epilogue)
    a = _core._ensure_cuda(a)
    b = _core._ensure_cuda(b)
    # Multi-device: align b to a's device
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _core._transfer_to_device(b, a._device_idx)

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
    # The operand decisions are `launch_keys.mm_dtypes` — the function the derived census keys
    # with; the wrapper only performs the casts it names. NBX_FORCE_FP32_ACCUM (a diagnostic)
    # widens both half inputs first.
    #
    # The fp16 activation is NOT materialised as fp32 any more: the kernels widen it in
    # registers — matmul_kernel through PROMOTE_A, the GEMV kernels on every load — the same
    # numbers the copy produced (an exact widening), without the copy per matmul. The store
    # dtype is decided by _matmul_out_dtype from the hardware gate, as before (fp32 here).
    a_to, b_to, promote_a, promote_b = _lk.mm_dtypes(a.nbx_dtype, b.nbx_dtype,
                                                     _core._NBX_HAS_NATIVE_BF16, NBX_FORCE_FP32_ACCUM)
    if a.nbx_dtype != a_to:
        a = a.to(a_to)
    if b.nbx_dtype != b_to:
        b = b.to(b_to)

    # Dtype alignment (decided in `launch_keys.mm_dtypes`). Three situations:
    #   1. Same dtype  → no-op.
    #   2. fp32 act × fp16 weight on pre-Ampere → this is the common case
    #      (the activation widened in-kernel). Keep the weight fp16 in memory; the kernel
    #      promotes the b tile to fp32 inline via PROMOTE_B. No heap alloc.
    #   3. Anything else (fp32 × bf16, bf16 × fp16, etc.) → widen to the
    #      common dtype. Rare; typically a downstream force_fp32 bmm feeding
    #      the next matmul.
    # A NARROW WEIGHT IS PROMOTED IN THE KERNEL, NOT MATERIALISED.
    #
    # `PROMOTE_B` makes the kernel do `b = b.to(a.dtype)` on the loaded tile —
    # it is dtype-agnostic (matmul.py:243) and has always been able to take
    # bf16. The `not _NBX_HAS_NATIVE_BF16` guard is a pre-Ampere CUDA
    # condition, and it excluded every machine that HAS native bf16 from a path
    # that has nothing to do with bf16 arithmetic: here `a_eff` is fp32, so the
    # math is fp32 either way and the only question is whether the weight is
    # widened in memory first.
    #
    # It was being widened, on EVERY call, for a weight that never changes.
    # Measured on an M4 Pro (TinyLlama-1.1B, 120 tokens, 2026-09-18):
    # `strided_copy_nd_kernel` was 1408 calls and 11.86 s of a 19.74 s decode —
    # 60% — and its four dominant shapes are weight-sized with transposing
    # strides, e.g. shape (2048, 5632) src (1, 2048) -> dst (5632, 1) at
    # ~5.5 GB/s against the 80.97 GB/s a contiguous copy reaches. Every one came
    # from this line via `b.to(widest)`.
    #
    # bf16 -> fp32 is lossless, so promoting per tile yields the same fp32
    # values the materialised cast did; the outputs are expected to be
    # bit-identical and that is checked, not assumed.
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
        return _apply_matmul_epilogue(c, _epilogue)

    a = a.contiguous()
    # The weight is walked by its own strides: matmul_kernel receives (stride_bk, stride_bn)
    # and loads the B tile through them, so a pre-transposed weight — the (K, N) stride view of
    # a (N, K) row-major buffer, stride_bk == 1 — is read in place. Materialising it here copied
    # the whole model once per prefill (TinyLlama: 154 weights, 2.2 GB, per request — the copy
    # census of 2026-09-07). Only a broadcast (a zero stride) is materialised: the kernel's
    # integer analysis assumes every stride positive.
    if any(st == 0 for st in b.stride()):
        b = b.contiguous()
    out_dtype = _matmul_out_dtype(a, M, force_fp32=promote_a)
    c = NBXTensor.empty((M, N), device=f"cuda:{a._device_idx}" if hasattr(a, '_device_idx') else 'cuda',
                        dtype=out_dtype)
    # IEEE mode: force strict fp32 tl.dot on pre-Ampere when we've promoted
    # inputs to fp32 for overflow protection. Otherwise tl.dot silently
    # casts fp32 → fp16 HMMA → saturates at ±65504 → NaN/Inf cascade.
    ieee = (not _core._NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)
    # Phase 1.5 autotune: BLOCK_*/GROUP_M/num_warps/num_stages chosen
    # adaptively per (M, N, K, IEEE_PRECISION, PROMOTE_B). Grid uses
    # META lambda so autotune-selected blocks drive the launch shape.
    _set_device(a)
    # ROW BANDING above 2^31 OUTPUT ELEMENTS. A launch whose C holds more than 2^31 elements
    # writes nothing past row `2**31 // N` on the Metal backend: the offsets are promoted to
    # int64 in the kernel (matmul.py:257) and CUDA is exact, but the Metal lowering does not
    # honour it (2026-09-23; tests/unit/kernels/test_a_gemm_beyond_two_billion_elements.py).
    # Three kernel-side spellings were tried and none helped, so the launch is split instead:
    # each band's own C is under the boundary, and the arithmetic per band is untouched.
    #
    # The AUTOTUNE KEY IS PRESERVED. `M_BUCKET` is a separate argument from `M` and the key is
    # keyed on the bucket, so every band is served the configuration certified for the whole
    # shape — banding changes how many launches happen, never which entry answers for them.
    bands = [(0, M)]
    if N > 0 and M * N > _NBX_MM_MAX_OUTPUT_ELEMS:
        per = max(1, _NBX_MM_MAX_OUTPUT_ELEMS // N)
        bands = [(r, min(r + per, M)) for r in range(0, M, per)]
    m_bucket = _bucket_of("M", M)
    _mu_tile = _gemm_unit_tile(a, b, M, N)
    for r0, r1 in bands:
        rows = r1 - r0
        a_b = a[r0:r1] if len(bands) > 1 else a
        c_b = c[r0:r1] if len(bands) > 1 else c
        if _mu_tile is not None:
            _gemm_m8n8k4(a_b, b, None, c_b, 1.0, 0.0, False, _epilogue, _mu_tile)
            continue
        grid = (lambda META, _m=rows: (triton.cdiv(_m, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),))
        _core._autotune_headroom_guard(matmul_kernel[grid])(
            a_b, b, c_b,
            rows, N, K, m_bucket,
            a_b.stride(0), a_b.stride(1),
            b.stride(0), b.stride(1),
            c_b.stride(0), c_b.stride(1),
            IEEE_PRECISION=ieee,
            PROMOTE_B=promote_b,
            PROMOTE_A=promote_a,
            EPILOGUE=_epilogue,
        )
    return c


def bmm(a, b, allow_strided_b: bool = False) :
    """Batched matrix multiplication: C[i] = A[i] @ B[i].

    `allow_strided_b` lets the caller hand in a non-contiguous B and have
    the kernel walk it by its own strides. `baddbmm_kernel` already
    receives all three of B's strides; the unconditional `.contiguous()`
    below simply threw that generality away. Off by default — a badly
    strided operand can cost more in uncoalesced loads than the copy it
    saves, so it is opt-in per call site with its measurement, never a
    blanket change. See `_math_attention`, the first caller.
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally.

    One batched 3D-grid launch via `baddbmm_kernel` (HAS_BIAS=False) for
    every M — P-KERNEL-LAUNCH-HYGIENE B1. The former per-batch
    matmul_kernel loop (prefill) and per-batch × per-row mv loop (decode
    M<=4) issued B (resp. B·M) launches per call; on the deterministic
    decode path (`_math_attention`, batch=B·H, M=1) that was thousands of
    launches per generated token, pure launch-bound. The batched kernel
    body is a line-for-line mirror of matmul_kernel, so results are
    value-identical to the former prefill loop under the same autotune
    config; the decode path leaves the mv reduction order for tl.dot
    tiles (fp reassociation-level difference only — products are exact in
    the fp32 accumulator for half inputs).
    """
    a = _core._ensure_cuda(a)
    b = _core._ensure_cuda(b)
    if hasattr(a, '_device_idx') and hasattr(b, '_device_idx') and a._device_idx != b._device_idx:
        b = _core._transfer_to_device(b, a._device_idx)

    # The matrix unit takes two operands of its operand dtype as they are in memory: the widening below exists
    # because the tl.dot kernel has no PROMOTE_A, and an fp16 value is exact in fp32 with fp16 x fp16 products
    # exact in the fp32 accumulator, so the unit computes the same sums (the store dtype is unchanged). Rows that
    # do not fill the unit's smallest tile (decode) keep the batched kernel (launch_keys.matrix_unit_operands).
    _mu_tile = _gemm_unit_tile(a, b, a.shape[1], b.shape[2])
    if _mu_tile is not None:
        a = a.contiguous()
        if not allow_strided_b:
            b = b.contiguous()
        c = NBXTensor.empty((a.shape[0], a.shape[1], b.shape[2]), device=a.device,
                            dtype=_matmul_out_dtype(a, a.shape[1], force_fp32=True))
        _gemm_m8n8k4(a, b, None, c, 1.0, 0.0, False, 0, _mu_tile)
        return c

    # Hardware-gated fp16→fp32 input upcast — same rationale as mm().
    # Use nbx_dtype for guard comparisons; .dtype returns triton.language.dtype.
    # The operand decisions are `launch_keys.bmm_dtypes` — the function the derived census keys
    # with (NBX_FORCE_FP32_ACCUM first; an fp16 activation widened in memory on hardware without
    # native bf16 — bmm has no PROMOTE_A; an fp16 weight under fp32 widened in the kernel there;
    # any other mismatch widened to the widest). The wrapper only casts.
    a_to, b_to, promote_b = _lk.bmm_dtypes(a.nbx_dtype, b.nbx_dtype,
                                           _core._NBX_HAS_NATIVE_BF16, NBX_FORCE_FP32_ACCUM)
    if a.nbx_dtype != a_to:
        a = a.to(a_to)
    if b.nbx_dtype != b_to:
        b = b.to(b_to)

    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, f"bmm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"

    a = a.contiguous()
    if not allow_strided_b:
        b = b.contiguous()
    out_dtype = _matmul_out_dtype(a, M, force_fp32=True)
    c = NBXTensor.empty((B, M, N), device=a.device, dtype=out_dtype)
    ieee = (not _core._NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)

    # Match mm() — sync Triton driver to a's device immediately before launch.
    # NBXTensor.empty(...) above can cudaSetDevice without updating Triton's
    # driver active device, and dtype alignment paths may also have touched
    # it. Without this, batched kernels on non-zero devices hit
    # "Pointer argument (at 0) cannot be accessed from Triton".
    _set_device(a)
    # Batched launch — grid axis 2 walks the batch inside the kernel
    # (baddbmm_kernel HAS_BIAS=False). `c` doubles as the (dead) bias
    # pointer: with HAS_BIAS=False the bias load is compiled out, so no
    # uninitialized read; alpha/beta are likewise unused. gridDim.z caps
    # at _GRID_Z_MAX: matmul_wrapper collapses ALL leading dims into B, so
    # an extreme collapsed batch chunks into z-sized launches (one launch
    # per 65535 batches — the loop body runs exactly once below the cap,
    # with no narrow() on that fast path).
    for z0 in range(0, B, _GRID_Z_MAX):
        zb = min(_GRID_Z_MAX, B - z0)
        a_z = a if zb == B else a.narrow(0, z0, zb)
        b_z = b if zb == B else b.narrow(0, z0, zb)
        c_z = c if zb == B else c.narrow(0, z0, zb)
        grid = lambda META, _zb=zb: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
            1,
            _zb,
        )
        _core._autotune_headroom_guard(baddbmm_kernel[grid])(
            a_z, b_z, c_z, c_z,
            1.0, 0.0,
            M, N, K, _bucket_of("M", M), _bucket_of("N", N), _bucket_of("K", K),
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            0, 0, 0,
            IEEE_PRECISION=ieee,
            PROMOTE_B=promote_b,
            HAS_BIAS=False,
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
    route = _lk.matmul_route(a.ndim, b.ndim)      # the route the derived census keys with
    if route == "mm":
        return mm(a, b)
    if route == "bmm":
        return bmm(a, b)
    if route == "mv":
        return mv_wrapper(a, b)
    if route == "batched":
        # Batched: reshape a to (batch, M, K), mm each, reshape back
        orig_shape = a.shape
        M, K = orig_shape[-2], orig_shape[-1]
        batch = a.numel() // (M * K)
        a_3d = a.contiguous().view(batch, M, K)
        result = bmm(a_3d, b.unsqueeze(0).expand(batch, K, b.shape[1]))
        return result.view(*orig_shape[:-1], b.shape[1])
    if route == "general":
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
    return _ew.add(out, bias)


def addmm(bias, a, b,
          beta: float = 1.0, alpha: float = 1.0, _epilogue: int = 0) :
    """C = beta * bias + alpha * (A @ B).
    Kernel accumulates in fp32 (hardware). Output matches input dtype.
    DtypeEngine handles overflow protection externally.

    Small-M fast path (M <= 4): per-row addmv. Bias is broadcast the same
    way torch.addmm does: (N,), (1, N), or (M, N).

    `_epilogue` (fusion_vertical pass): fused elementwise epilogue code
    (0 none, 1 silu, 2 gelu exact, 3 gelu tanh) with per-stage rounding
    emulation in-kernel; M <= 4 decode path stays unfused (see mm()).
    """
    a = _core._ensure_cuda(a)
    b = _core._ensure_cuda(b)
    bias = _core._ensure_cuda(bias)

    # Hardware-gated fp16→fp32 input upcast — same rationale as mm().
    # Use nbx_dtype for guard comparisons; .dtype returns triton.language.dtype.
    # The operand decisions are `launch_keys.addmm_dtypes` — the function the derived census
    # keys with (NBX_FORCE_FP32_ACCUM first; PROMOTE_A / PROMOTE_B in-kernel widenings on hardware
    # without native bf16; any other mismatch widened to the widest; the bias widened in-kernel
    # under an fp32 activation, else cast to the activation's dtype). The wrapper only casts.
    a_to, b_to, bias_to, promote_a, promote_b, promote_bias = _lk.addmm_dtypes(
        a.nbx_dtype, b.nbx_dtype, bias.nbx_dtype, _core._NBX_HAS_NATIVE_BF16, NBX_FORCE_FP32_ACCUM)
    if a.nbx_dtype != a_to:
        a = a.to(a_to)
    if b.nbx_dtype != b_to:
        b = b.to(b_to)
    if bias.nbx_dtype != bias_to:
        bias = bias.to(bias_to)

    if a.ndim > 2:
        orig_lead = a.shape[:-1]
        K_ = a.shape[-1]
        batch = a.numel() // K_
        a2d = a.contiguous().view(batch, K_)
        out2d = addmm(bias, a2d, b, beta=beta, alpha=alpha, _epilogue=_epilogue)
        return out2d.view(*orig_lead, out2d.shape[-1])

    M, K = a.shape
    K2, N = b.shape
    # Named, not bare: a shape mismatch here reaches the census as
    # "AssertionError: " with nothing in it, and a failure that carries no
    # numbers costs a whole reproduction run to say what the raise could
    # have said itself (2026-09-23, the two PixArt -MS containers).
    if K != K2:
        raise AssertionError(
            f"addmm shape mismatch: ({M}, {K}) @ ({K2}, {N}) — the contracted "
            f"dimension differs, {K} against {K2}")

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
        return _apply_matmul_epilogue(c, _epilogue)

    if bias.ndim == 2 and bias.shape[0] != 1:
        # The kernel adds ONE bias row [N] to every output row: a per-row bias above the decode rows would be computed
        # with its first row only. Refused by name, never computed wrong (no catalogue graph forms one: its 10 478
        # addmm carry a bias [N], read 2026-10-29).
        raise ValueError(f"addmm: a per-row bias {tuple(bias.shape)} is served at M <= 4 only (here M = {M}); "
                         f"the kernel adds one bias row [N] to every row")
    a = a.contiguous()
    # A pre-transposed weight is walked by its strides (as mm); only a broadcast
    # (stride 0) operand is materialised, the kernel's tile loads need a real address.
    if any(st == 0 for st in b.stride()):
        b = b.contiguous()
    bias = bias.contiguous()
    out_dtype = _matmul_out_dtype(a, M, force_fp32=promote_a)
    c = NBXTensor.empty((M, N), device=a.device, dtype=out_dtype)
    ieee = (not _core._NBX_HAS_NATIVE_BF16) and (out_dtype == NBXDtype.float32)
    # Row banding above 2^31 output elements, exactly as `mm` — see the comment there. The
    # bias is indexed by COLUMN and is shared by every band, so it is passed whole.
    bands = [(0, M)]
    if N > 0 and M * N > _NBX_MM_MAX_OUTPUT_ELEMS:
        per = max(1, _NBX_MM_MAX_OUTPUT_ELEMS // N)
        bands = [(r, min(r + per, M)) for r in range(0, M, per)]
    m_bucket = _bucket_of("M", M)
    _mu_tile = _gemm_unit_tile(a, b, M, N) if bias.ndim == 1 or (bias.ndim == 2 and bias.shape[0] == 1) else None
    for r0, r1 in bands:
        rows = r1 - r0
        a_b = a[r0:r1] if len(bands) > 1 else a
        c_b = c[r0:r1] if len(bands) > 1 else c
        if _mu_tile is not None:
            _gemm_m8n8k4(a_b, b, bias, c_b, alpha, beta, promote_bias, _epilogue, _mu_tile)
            continue
        grid = (lambda META, _m=rows: (triton.cdiv(_m, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),))
        _core._autotune_headroom_guard(addmm_kernel[grid])(
            a_b, b, bias, c_b,
            rows, N, K, m_bucket,
            a_b.stride(0), a_b.stride(1),
            b.stride(0), b.stride(1),
            c_b.stride(0), c_b.stride(1),
            alpha, beta,
            IEEE_PRECISION=ieee,
            PROMOTE_B=promote_b,
            PROMOTE_A=promote_a,
            PROMOTE_BIAS=promote_bias,
            EPILOGUE=_epilogue,
        )
    return c


# Epilogue string -> kernel code mapping (fusion_vertical pass). Kept in
# ONE place so the emitter, the wrappers, and the kernel constexpr codes
# cannot drift apart.
_MATMUL_EPILOGUE_CODES = {
    ("silu", "none"): 1,
    ("silu", "tanh"): 1,   # approximate is silu-irrelevant; tolerate
    ("gelu", "none"): 2,
    ("gelu", "tanh"): 3,
}


def mm_epilogue_wrapper(a, b, epilogue: str = "silu",
                        approximate: str = "none"):
    """custom::mm_epilogue — aten::mm fused with a unary elementwise
    epilogue (fusion_vertical pass, EXACT gate). One kernel launch for
    the pair; byte-identical to the unfused pair via per-stage rounding
    emulation + shared autotune config (same kernel function)."""
    return mm(a, b, _epilogue=_MATMUL_EPILOGUE_CODES[(epilogue, approximate)])


def addmm_epilogue_wrapper(bias, a, b, beta: float = 1.0,
                           alpha: float = 1.0, epilogue: str = "silu",
                           approximate: str = "none"):
    """custom::addmm_epilogue — aten::addmm fused with a unary
    elementwise epilogue (fusion_vertical pass, EXACT gate)."""
    return addmm(bias, a, b, beta=beta, alpha=alpha,
                 _epilogue=_MATMUL_EPILOGUE_CODES[(epilogue, approximate)])
#: A matmul launch whose OUTPUT holds more than this many elements is split along M. The
#: boundary is 2^31 because that is where an int32 element offset wraps; the margin below it
#: is deliberate, so a tile that straddles the boundary is never the last one.
_NBX_MM_MAX_OUTPUT_ELEMS = int(os.environ.get("NBX_MM_MAX_OUTPUT_ELEMS", str(2 ** 31 - 2 ** 20)))


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
    """Matrix-vector multiply: out[i] = sum_j(mat[i, j] * vec[j]).

    The SIMT gemv_vec kernel (ops/gemv_vec.py, adopted 2026-08-24 over the FlagGems mv kernel: its (BLOCK_N,)
    per-chunk-reduced accumulator, all of K inside one program in fixed order — no split-K atomics). Its tile —
    rows per program, K chunk, warps, stages — is the certified configuration of the key (workstream 1, small-M
    GEMM): the N and K buckets from the launcher's own `bucket_of` and the dtypes of its tensors, the key the derived
    census forms (`launch_keys.mv_launches`).
    """
    # A wrapper that copies whatever it is given cannot honour a caller that took care not to
    # give it a copy. The gemv kernels take the matrix's two strides and the vector's one; what
    # they need is K-contiguous ROWS (stride(1) == 1: the pre-transposed weight layout, where
    # the loads vectorise) — a matrix whose rows are strided is copied once here and the line
    # says why; a vector is read through its stride, a broadcast (stride 0) materialised.
    if mat.stride(1) != 1:
        mat = mat.contiguous()          # justified: the kernel's rows must be K-contiguous
    if vec.stride(0) == 0:
        vec = vec.contiguous()          # justified: a broadcast vector has no stride to walk
    N, K = mat.shape
    out = NBXTensor.empty(N, device=mat.device, dtype=_matmul_out_dtype(mat))
    _set_device(mat)
    from nbxtensor.ops.gemv_vec import gemv_vec_kernel
    gemv_vec_kernel[lambda META: (triton.cdiv(N, META['BLOCK_N']),)](
        mat, vec, out,
        N, K, _bucket_of("N", N), _bucket_of("K", K),
        mat.stride(0), mat.stride(1),
        vec.stride(0),
        out.stride(0),
    )
    return out


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
    if K != K2:
        raise AssertionError(
            f"baddbmm shape mismatch: ({B}, {M}, {K}) @ (..., {K2}, {N}) — the "
            f"contracted dimension differs, {K} against {K2}")

    output = NBXTensor.empty((B, M, N), device=batch1.device, dtype=batch1.dtype)

    # Handle bias broadcasting: input may be [M, N], [B, M, N], [1, M, N],
    # [B, 1, N], [M, 1], etc. Any size-1 dim broadcasts with stride 0 —
    # keeping the memory stride of a size-1 dim would walk PAST the bias
    # row/column (wrong values, OOB on the last batch).
    if input.ndim == 2:
        bias_batch_stride = 0
        bias_m_stride = input.stride(0) if input.shape[0] > 1 else 0
        bias_n_stride = input.stride(1) if input.shape[1] > 1 else 0
    else:
        bias_batch_stride = input.stride(0) if input.shape[0] > 1 else 0
        bias_m_stride = input.stride(-2) if input.shape[-2] > 1 else 0
        bias_n_stride = input.stride(-1) if input.shape[-1] > 1 else 0

    _mu_tile = _gemm_unit_tile(batch1, batch2, M, N)
    if _mu_tile is not None:           # the matrix unit (same epilogue: alpha * acc + beta * bias, cast to output)
        _gemm_m8n8k4(batch1, batch2, input, output, alpha, beta, False, 0, _mu_tile,
                     bias_strides=(bias_batch_stride, bias_m_stride, bias_n_stride))
        return output

    _set_device(batch1)
    # IEEE_PRECISION=True preserves the kernel's historical
    # `allow_tf32=False` dot semantics for fp32 inputs (ignored by tl.dot
    # for half inputs). HAS_BIAS=True keeps the beta*bias epilogue.
    # gridDim.z caps at _GRID_Z_MAX — chunk the batch past it (the loop
    # body runs exactly once below the cap, no narrow() on the fast path).
    for z0 in range(0, B, _GRID_Z_MAX):
        zb = min(_GRID_Z_MAX, B - z0)
        b1_z = batch1 if zb == B else batch1.narrow(0, z0, zb)
        b2_z = batch2 if zb == B else batch2.narrow(0, z0, zb)
        out_z = output if zb == B else output.narrow(0, z0, zb)
        bias_z = (input if (zb == B or bias_batch_stride == 0)
                  else input.narrow(0, z0, zb))
        grid = lambda META, _zb=zb: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
            1,
            _zb,
        )
        _core._autotune_headroom_guard(baddbmm_kernel[grid])(
            b1_z, b2_z, out_z, bias_z,
            alpha, beta,
            M, N, K, _bucket_of("M", M), _bucket_of("N", N), _bucket_of("K", K),
            batch1.stride(0), batch1.stride(1), batch1.stride(2),
            batch2.stride(0), batch2.stride(1), batch2.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            bias_batch_stride, bias_m_stride, bias_n_stride,
            IEEE_PRECISION=True,
            PROMOTE_B=False,
            HAS_BIAS=True,
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
    _set_device(mat)
    # rows, reduction tile, warps: the certified configuration of the key (workstream 1, small-M GEMM) — the N and M
    # buckets and the dtypes of its tensors, the key the derived census forms (`launch_keys.addmv_launches`)
    addmv_kernel[lambda META: (triton.cdiv(N, META['BLOCK_N']),)](
        mat, vec, input, out,
        N, M, _bucket_of("N", N), _bucket_of("M", M),
        alpha, beta,
        mat.stride(0), mat.stride(1),
        vec.stride(0),
        input.stride(0),
        out.stride(0),
    )
    return out


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
        triton.cdiv(OW, _core._BILINEAR_BX),
        triton.cdiv(OH, _core._BILINEAR_BY),
        N * C,
    )
    _set_device(x.contiguous())
    upsample_bilinear2d_kernel[grid](
        x.contiguous(), output,
        N, C, IH, IW, OH, OW,
        scale_h, scale_w, offset,
        BLOCK_X=_core._BILINEAR_BX, BLOCK_Y=_core._BILINEAR_BY,
        num_warps=4)
    return output


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
    x = _core._ensure_cuda(x).contiguous()
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


def mm_cost(a, b, epilogue: int = 0, native_bf16=None, force_accum=None, contract_half=None):
    """`mm` on `a` [M, K] and `b` [K, N] (`_cost.Spec`s), in the wrapper's order: the operands cast to
    `launch_keys.mm_dtypes` (a dense copy each); then, at M <= 4, `a` made contiguous, the output [M, N] and, per row,
    the GEMV's own output (N at `matmul_out_dtype` for one row) — and a full copy of the weight when its rows are not
    K-contiguous (`mv_wrapper`: `b.t()` must have stride 1 along K; a pre-transposed weight view has it, a row-major
    one does not) — each freed once copied into the output's row, then an epilogue (SiLU, GELU) makes its output while
    the GEMM's is alive; above, `a` made contiguous, `b` only when broadcast (a zero stride), the output at
    `matmul_out_dtype` (fp32 when `a` was promoted), the matrix unit or the kernel allocating nothing more (row bands
    are views, the epilogue runs in the kernel). A local rebound frees its old buffer after the new one exists; the
    rest die at the return."""
    from nbxtensor.ops import _cost as C
    from nbxtensor.shape import is_contiguous
    native_bf16 = bool(_core._NBX_HAS_NATIVE_BF16) if native_bf16 is None else native_bf16
    force_accum = bool(NBX_FORCE_FP32_ACCUM) if force_accum is None else force_accum
    contract_half = bool(_core._NBX_ACTIVATIONS_FP16_SAFE) if contract_half is None else contract_half
    st = C._Steps()
    M, K = (int(d) for d in a.shape)
    N = int(b.shape[1])
    flops = 2 * M * N * K
    held = {}

    def rebind(name, spec, label):
        i = st.alloc(label, spec.nbytes)
        old = held.get(name)
        held[name] = i
        if old is not None:
            st.free(old)
        return spec

    a_to, b_to, promote_a, _promote_b = _lk.mm_dtypes(NBXDtype[a.dtype], NBXDtype[b.dtype], native_bf16, force_accum)
    if NBXDtype[a.dtype] != a_to:
        a = rebind("a", C.Spec.contiguous(a.shape, a_to.name), "a cast")
    if NBXDtype[b.dtype] != b_to:
        b = rebind("b", C.Spec.contiguous(b.shape, b_to.name), "b cast")
    if not is_contiguous(a.shape, a.strides):
        a = rebind("a", C.Spec.contiguous(a.shape, a.dtype), "a made contiguous")
    if M <= 4:
        out = _lk.matmul_out_dtype(NBXDtype[a.dtype], M, False, native_bf16, force_accum, contract_half)
        c = st.alloc("output", M * N * dtype_size(out), returned=epilogue == 0)
        row_out = _lk.matmul_out_dtype(NBXDtype[b.dtype], 1, False, native_bf16, force_accum, contract_half)
        k_contiguous_rows = int(b.strides[0]) == 1          # b.t() has stride 1 along K
        for _ in range(M):
            copy = None if k_contiguous_rows else st.alloc("the weight made K-contiguous (mv)", b.nbytes)
            r = st.alloc("a row's GEMV output", N * dtype_size(row_out))
            if copy is not None:
                st.free(copy)
            st.free(r)
    else:
        if any(int(x) == 0 for x in b.strides):
            b = rebind("b", C.Spec.contiguous(b.shape, b.dtype), "b broadcast made dense")
        out = _lk.matmul_out_dtype(NBXDtype[a.dtype], M, bool(promote_a), native_bf16, force_accum, contract_half)
        st.alloc("output", M * N * dtype_size(out), returned=True)
        return st.done(flops)                                # an epilogue runs inside the kernel here (EPILOGUE)
    if epilogue:                                             # the GEMV rows: the epilogue after, its own output
        for i in st.nest(C.elementwise_unary(C.Spec.contiguous((M, N), out.name)), "epilogue"):
            st.buffers[i][4] = True                          # the epilogue's output is the one returned
    return st.done(flops)


def addmm_cost(bias, a, b, epilogue: int = 0, native_bf16=None, force_accum=None, contract_half=None):
    """`addmm` on `bias` ([N], [1, N] or [M, N]), `a` [..., K] and `b` [K, N] (`_cost.Spec`s), in the wrapper's order:
    the three operands cast to `launch_keys.addmm_dtypes` (a dense copy each); an `a` of more than two dims made
    contiguous and flattened, the product computed on the flat rows (its view returned); then, at M <= 4, `a` and the
    bias made contiguous, the output [M, N] at `matmul_out_dtype` and per row the GEMV's own output (N for one row) —
    and the whole weight copied when `b.t()` is not contiguous (`addmv_wrapper` makes its matrix contiguous: a row-major
    weight; a pre-transposed weight view is) — each freed once copied into the output's row, then an epilogue's output
    beside the GEMM's; above, `a` and the bias made contiguous, `b` only when broadcast, the output at
    `matmul_out_dtype` (fp32 when `a` was promoted), the epilogue in the kernel. A bias of another shape is refused by
    name (the wrapper's own rows: shared [N] / [1, N], per-row [M, N] at M <= 4 only, as the wrapper)."""
    from nbxtensor.ops import _cost as C
    from nbxtensor.shape import is_contiguous
    native_bf16 = bool(_core._NBX_HAS_NATIVE_BF16) if native_bf16 is None else native_bf16
    force_accum = bool(NBX_FORCE_FP32_ACCUM) if force_accum is None else force_accum
    contract_half = bool(_core._NBX_ACTIVATIONS_FP16_SAFE) if contract_half is None else contract_half
    st = C._Steps()
    held = {}

    def rebind(name, spec, label):
        i = st.alloc(label, spec.nbytes)
        old = held.get(name)
        held[name] = i
        if old is not None:
            st.free(old)
        return spec

    a_to, b_to, bias_to, promote_a, _pb, _pbias = _lk.addmm_dtypes(
        NBXDtype[a.dtype], NBXDtype[b.dtype], NBXDtype[bias.dtype], native_bf16, force_accum)
    if NBXDtype[a.dtype] != a_to:
        a = rebind("a", C.Spec.contiguous(a.shape, a_to.name), "a cast")
    if NBXDtype[b.dtype] != b_to:
        b = rebind("b", C.Spec.contiguous(b.shape, b_to.name), "b cast")
    if NBXDtype[bias.dtype] != bias_to:
        bias = rebind("bias", C.Spec.contiguous(bias.shape, bias_to.name), "bias cast")
    if len(a.shape) > 2:
        if not is_contiguous(a.shape, a.strides):
            a = rebind("a", C.Spec.contiguous(a.shape, a.dtype), "a made contiguous (flattened)")
        rows = 1
        for d in a.shape[:-1]:
            rows *= int(d)
        a = C.Spec.contiguous((rows, int(a.shape[-1])), a.dtype)
    M, K = (int(d) for d in a.shape)
    N = int(b.shape[1])
    if not (len(bias.shape) == 1 or (len(bias.shape) == 2 and int(bias.shape[0]) in (1, M))):
        raise ValueError(f"addmm_cost: a bias of shape {tuple(bias.shape)} is none of the wrapper's ([N], [1, N], [M, N])")
    flops = 2 * M * N * K
    if not is_contiguous(a.shape, a.strides):
        a = rebind("a", C.Spec.contiguous(a.shape, a.dtype), "a made contiguous")
    if not is_contiguous(bias.shape, bias.strides):
        bias = rebind("bias", C.Spec.contiguous(bias.shape, bias.dtype), "bias made contiguous")
    if M <= 4:
        out = _lk.matmul_out_dtype(NBXDtype[a.dtype], M, False, native_bf16, force_accum, contract_half)
        st.alloc("output", M * N * dtype_size(out), returned=epilogue == 0)
        row_out = _lk.matmul_out_dtype(NBXDtype[b.dtype], 1, False, native_bf16, force_accum, contract_half)
        bt_contiguous = tuple(int(x) for x in b.strides) == (1, K)       # b.t() is (N, K) row-major
        for _ in range(M):
            copy = None if bt_contiguous else st.alloc("the weight made contiguous (addmv)", b.nbytes)
            r = st.alloc("a row's GEMV output", N * dtype_size(row_out))
            if copy is not None:
                st.free(copy)
            st.free(r)
        if epilogue:
            for i in st.nest(C.elementwise_unary(C.Spec.contiguous((M, N), out.name)), "epilogue"):
                st.buffers[i][4] = True
        return st.done(flops)
    if len(bias.shape) == 2 and int(bias.shape[0]) != 1:
        raise ValueError(f"addmm_cost: a per-row bias {tuple(bias.shape)} is served at M <= 4 only (here M = {M}), "
                         f"as the wrapper refuses it")
    if any(int(x) == 0 for x in b.strides):
        b = rebind("b", C.Spec.contiguous(b.shape, b.dtype), "b broadcast made dense")
    out = _lk.matmul_out_dtype(NBXDtype[a.dtype], M, bool(promote_a), native_bf16, force_accum, contract_half)
    st.alloc("output", M * N * dtype_size(out), returned=True)
    return st.done(flops)


def bmm_cost(a, b, allow_strided_b: bool = False, native_bf16=None, force_accum=None, contract_half=None,
             unit=None):
    """`bmm` on `a` [B, M, K] and `b` [B, K, N] (`_cost.Spec`s): on the matrix unit (`launch_keys.matrix_unit_operands`)
    `a` made contiguous and `b` unless the caller lets it be walked by its strides, then the output; otherwise the
    operands at `launch_keys.bmm_dtypes` (a cast each, dense), made contiguous as above, then the output at
    `launch_keys.matmul_out_dtype` (fp32 for half inputs on hardware without native bf16 unless the component's precision
    contract holds). The facts the wrapper reads from the run (`native_bf16`, `force_accum`, `contract_half`) are the
    run's in force when None; `unit` the profile's matrix unit when None. A local rebound frees its old buffer after
    the new one exists; the rest die at the return."""
    from nbxtensor.ops import _cost as C
    from nbxtensor.shape import is_contiguous
    native_bf16 = bool(_core._NBX_HAS_NATIVE_BF16) if native_bf16 is None else native_bf16
    force_accum = bool(NBX_FORCE_FP32_ACCUM) if force_accum is None else force_accum
    contract_half = bool(_core._NBX_ACTIVATIONS_FP16_SAFE) if contract_half is None else contract_half
    st = C._Steps()
    Bz, M, K = (int(d) for d in a.shape)
    N = int(b.shape[2])
    flops = 2 * Bz * M * N * K
    ad, bd = NBXDtype[a.dtype], NBXDtype[b.dtype]
    held = {}

    def rebind(name, spec, label):
        i = st.alloc(label, spec.nbytes)
        old = held.get(name)
        held[name] = i
        if old is not None:
            st.free(old)
        return spec

    if _lk.matrix_unit_operands(ad, bd, M, unit) is not None:
        if not is_contiguous(a.shape, a.strides):
            a = rebind("a", C.Spec.contiguous(a.shape, a.dtype), "a made contiguous")
        if not allow_strided_b and not is_contiguous(b.shape, b.strides):
            b = rebind("b", C.Spec.contiguous(b.shape, b.dtype), "b made contiguous")
        out = _lk.matmul_out_dtype(ad, M, True, native_bf16, force_accum, contract_half)
        st.alloc("output", Bz * M * N * dtype_size(out), returned=True)
        return st.done(flops)
    a_to, b_to, _promote = _lk.bmm_dtypes(ad, bd, native_bf16, force_accum)
    if ad != a_to:
        a = rebind("a", C.Spec.contiguous(a.shape, a_to.name), "a cast")
    if bd != b_to:
        b = rebind("b", C.Spec.contiguous(b.shape, b_to.name), "b cast")
    if not is_contiguous(a.shape, a.strides):
        a = rebind("a", C.Spec.contiguous(a.shape, a.dtype), "a made contiguous")
    if not allow_strided_b and not is_contiguous(b.shape, b.strides):
        b = rebind("b", C.Spec.contiguous(b.shape, b.dtype), "b made contiguous")
    out = _lk.matmul_out_dtype(NBXDtype[a.dtype], M, True, native_bf16, force_accum, contract_half)
    st.alloc("output", Bz * M * N * dtype_size(out), returned=True)
    return st.done(flops)
