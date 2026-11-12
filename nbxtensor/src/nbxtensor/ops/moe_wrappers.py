"""Mixture-of-experts wrappers: the fused MoE launches (dense and wna16). Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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



# ===========================================================================
# FUSED MOE WRAPPER
# ===========================================================================

# Block sizes for fused MoE kernel (tuned for decode batch=1)
_MOE_BM = 16
#: the row block the grouped kernel runs at — the block a caller aligns the routing by (`moe_align_block_size`)
MOE_BLOCK_M = _MOE_BM
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
    # num_valid_tokens MUST be the true assignment count (tokens × top_k)
    # — topk_weights is [M_tokens * top_k] on every pass. Deriving it
    # from hidden_states.shape[0] was correct for the gate/up passes
    # (input [M_tokens, K]) but WRONG for the down pass (input
    # `activated` is [padded, N]): the inflated bound turned the
    # sentinel lanes' token_mask TRUE, and the moe_weight load
    # (mul_routed_weight, down pass only) read past the [n] routing
    # buffer — a layout-dependent Warp Illegal Address, named by the
    # CUDA coredump at fused_moe.py:107 on the Qwen3-Omni audio leg
    # (immune to every sync barrier, hidden under CUDA_LAUNCH_BLOCKING
    # because the heap layout — not timing — decided whether the OOB
    # page was mapped).
    grid, args, kwargs = _fused_moe_launch(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids,
                                           expert_ids, num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k,
                                           mul_routed_weight, topk_divide)
    set_device(hidden_states)
    fused_moe_kernel[grid](*args, **kwargs)


def _fused_moe_launch(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids,
                      num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k, mul_routed_weight, topk_divide):
    """(grid, args, kwargs) of `fused_moe_kernel` for this launch — one definition for the launch and for the key its
    route reads (`fused_moe_unit_tile`)."""
    M = hidden_states.shape[0]
    num_valid_tokens = topk_weights.shape[0]
    EM = sorted_token_ids.shape[0]
    if M < _MOE_BM:
        EM = min(EM, M * top_k * _MOE_BM)
    # the tile beyond the alignment's row block is the key's certified configuration (`fused_moe._FUSED_MOE_CONFIGS`)
    grid = lambda META: (triton.cdiv(EM, _MOE_BM) * triton.cdiv(N, META['BLOCK_SIZE_N']),)    # noqa: E731
    args = (hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, EM, num_valid_tokens, _lk.bucket_of("M", num_valid_tokens), expert_ptrs.numel(),
            hidden_states.stride(0), hidden_states.stride(1), stride_bk, stride_bn, output.stride(0), output.stride(1))
    kwargs = dict(BLOCK_SIZE_M=_MOE_BM, MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
                  compute_type=hidden_states.dtype, TOPK_DIVIDE=topk_divide)
    return grid, args, kwargs


def fused_moe_unit_tile(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k, mul_routed_weight=False,
                        topk_divide=True):
    """The matrix unit's tile (a `matrix_unit.mm` row) a fused MoE launch runs at when its certificate measured the
    unit faster than the FMA kernel (`autotune_certified.unit_choice`, the FMA launch's key), else None. The unit takes
    experts of its operand dtype only (no split), and a launch with no certificate keeps the FMA kernel; a census
    shadow and the certifier's sweep take the FMA kernel (the unit's tiles are measured beside it,
    `matrix_unit_runs`). The arguments are `invoke_fused_moe`'s at the FMA alignment."""
    from nbxtensor.ops._configs import matrix_unit
    from nbxtensor.ops import autotune_certified as _cert
    from nbxtensor.ops.launch_observer import launch_observer
    if hidden_states.nbx_dtype != NBXDtype.float16 or not (matrix_unit() or {}).get("mm"):
        return None
    if launch_observer().active() or _cert.certifying():
        return None
    _grid, args, kwargs = _fused_moe_launch(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids,
                                            expert_ids, num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k,
                                            mul_routed_weight, topk_divide)
    chosen = _cert.unit_choice(fused_moe_kernel, args, kwargs)
    return chosen[1] if chosen is not None and chosen[0] == "unit" else None


def fused_moe_unit_tiles(hidden_states):
    """The matrix-unit tiles a fused MoE launch on these activations can run at (`matrix_unit.mm` of the profile in
    force, the unit's operand dtype only): the candidates its certification measures beside the FMA kernel."""
    from nbxtensor.ops._configs import matrix_unit
    from nbxtensor.ops.autotune_certified import _UNIT_TILE_FIELDS
    if hidden_states.nbx_dtype != NBXDtype.float16:
        return []
    return [{f: int(row[f]) for f in _UNIT_TILE_FIELDS} for row in ((matrix_unit() or {}).get("mm") or [])]


def invoke_fused_moe_m8n8k4(hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, N, K, stride_bk, stride_bn, top_k, tile, mul_routed_weight=False,
                            topk_divide=True):
    """`invoke_fused_moe` on the m8n8k4 matrix unit (`fused_moe_m8n8k4_kernel`) at `tile`; the routing aligned at the
    tile's rows (`moe_align_block_size(..., tile["block_m"], ...)`)."""
    from nbxtensor.ops.fused_moe_m8n8k4 import fused_moe_m8n8k4_kernel
    from nbxtensor.ops.matmul_m8n8k4 import mm_layouts
    BM, BN, BK = int(tile["block_m"]), int(tile["block_n"]), int(tile["block_k"])
    WM, WN = int(tile["warps_m"]), int(tile["warps_n"])
    EM = sorted_token_ids.shape[0]
    b_k_contig = stride_bk == 1 and N > 1
    set_device(hidden_states)
    fused_moe_m8n8k4_kernel[(triton.cdiv(EM, BM) * triton.cdiv(N, BN),)](
        hidden_states, expert_ptrs, output, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, topk_weights.shape[0],
        hidden_states.stride(0), hidden_states.stride(1), stride_bk, stride_bn, output.stride(0), output.stride(1),
        MUL_ROUTED_WEIGHT=bool(mul_routed_weight), TOP_K=int(top_k), TOPK_DIVIDE=bool(topk_divide),
        B_K_CONTIG=b_k_contig, MB=BM // 16, NB=BN // 16, BK=BK, GROUP_M=int(tile["group_m"]),
        num_warps=WM * WN, **mm_layouts(BM // 16, BN // 16, WM, WN, BK, b_k_contig))


def invoke_fused_moe_wna16(
    hidden_states,            # NBXTensor [M, K] — activations
    qw_ptrs, sc_ptrs, mn_ptrs,  # NBXTensor [E] int64 — triplet tables
    output,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    N, K,
    stride_qk, stride_qn,     # packed int32 strides
    stride_sg, stride_sn,     # scales/mins strides
    top_k,
    mul_routed_weight=False,
    topk_divide=True,
):
    """Launch the W4 fused MoE grouped GEMM (int4-g128-asym experts,
    in-register dequant — byte-gated vs fused_moe_fp32b_kernel).
    Same launch contract as invoke_fused_moe; _MOE_BK (32) divides the
    quant group (128) so a K-tile never crosses a group boundary."""
    from nbxtensor.ops.fused_moe import fused_moe_wna16_kernel
    M = hidden_states.shape[0]
    num_valid_tokens = topk_weights.shape[0]
    EM = sorted_token_ids.shape[0]
    if M < _MOE_BM:
        EM = min(EM, M * top_k * _MOE_BM)
    grid = (triton.cdiv(EM, _MOE_BM) * triton.cdiv(N, _MOE_BN),)
    set_device(hidden_states)

    fused_moe_wna16_kernel[grid](
        hidden_states, qw_ptrs, sc_ptrs, mn_ptrs, output,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        hidden_states.stride(0), hidden_states.stride(1),
        stride_qk, stride_qn,
        stride_sg, stride_sn,
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=_MOE_BM, BLOCK_SIZE_N=_MOE_BN, BLOCK_SIZE_K=_MOE_BK,
        GROUP_SIZE_M=_MOE_GROUP,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=hidden_states.dtype,
        QGROUP=128, QPACK=8,
        TOPK_DIVIDE=topk_divide,
        num_warps=4, num_stages=2,
    )


# ===========================================================================
# The routing's alignment and the one-token half band (launchers)
# ===========================================================================

def moe_align_block_size(topk_ids_flat, block_size, num_experts, device_idx, pairs=None):
    """Sort tokens by expert, pad to block_size alignment — ON DEVICE.

    Three constant-tuple Triton kernels (ops/moe_align.py) replace a host path (D2H of topk_ids -> numpy sort -> H2D
    of tables): the routing tables are recomputed from the CURRENT router output at every launch, which removes one
    D2H sync per MoE layer per step AND makes the MoE band replay-eligible (the host D2H was a structural plan-breaker
    at recording). The deterministic pairwise rank in stage 3 preserves the host sort's exact token order — the swap
    was byte-gated against the removed implementation.

    Outputs are sized on the shape-derived worst case min(n*BS, n + E*(BS-1)) rounded up to BS; the TRUE total lives
    only in the device scalar (every fused variant early-exits on it — the launch grids depend on host shapes alone,
    never on routing data).

    Args:
        topk_ids_flat: NBXTensor [M * top_k] — flat expert indices
        block_size: BLOCK_SIZE_M for the fused kernel
        num_experts: total number of experts
        device_idx: GPU device for output tensors
        pairs: the layer's (token, expert) pairs, the key's length class (`launch_keys.moe_align_launches`) — the ids'
            count unless this is one card's share of a layer split across cards

    Returns:
        sorted_token_ids: NBXTensor [max_total] — token indices sorted by expert (sentinel n beyond each expert's
            true count)
        expert_ids: NBXTensor [max_blocks] — expert id per block (-1 pad)
        num_tokens_post_padded: NBXTensor [1] — total entries after padding
    """
    from nbxtensor.ops.moe_align import moe_align_stage1_kernel, moe_align_stage2_kernel, moe_align_stage3_kernel
    from nbxtensor.ops.launch_keys import moe_align_classes

    n = topk_ids_flat.numel()
    bs = int(block_size)
    bound = min(n * bs, n + num_experts * (bs - 1))
    max_total = ((bound + bs - 1) // bs) * bs
    max_blocks = max_total // bs

    DeviceAllocator.set_device(device_idx)
    dev = f"cuda:{device_idx}"
    sorted_ids = NBXTensor.empty((max_total,), dtype=NBXDtype.int64, device=dev)
    expert_ids = NBXTensor.empty((max_blocks,), dtype=NBXDtype.int64, device=dev)
    num_post_pad = NBXTensor.empty((1,), dtype=NBXDtype.int64, device=dev)
    offsets_ws = NBXTensor.empty((num_experts,), dtype=NBXDtype.int64, device=dev)
    padded_ws = NBXTensor.empty((num_experts,), dtype=NBXDtype.int64, device=dev)

    ids = topk_ids_flat.contiguous()
    set_device(sorted_ids)
    # the chunk widths and warps of each stage are its key's certified configuration (ops/moe_align.py), keyed by
    # the layer's pairs (`launch_keys.moe_align_launches`)
    BE, nb = moe_align_classes(num_experts, n if pairs is None else pairs)
    moe_align_stage1_kernel[(1,)](
        ids, offsets_ws, padded_ws, num_post_pad, n, nb, BS=bs, BE=BE, E=num_experts)
    moe_align_stage2_kernel[lambda META: (triton.cdiv(max_total, META["BLK"]),)](
        offsets_ws, padded_ws, sorted_ids, expert_ids, n, max_total, nb, BS=bs, BE=BE, E=num_experts)
    moe_align_stage3_kernel[lambda META: (triton.cdiv(n, META["BLKT"]),)](
        ids, offsets_ws, sorted_ids, n, nb, E=num_experts)

    return sorted_ids, expert_ids, num_post_pad


def moe_decode_vec_half_pass(hidden_states, tables, dev, flat_indices, flat_scores, top_k, K, N_gate, bf16):
    """The SIMT decode band for half-precision experts (`moe_decode_vec` half kernels): gateup with SwiGLU, the down
    projection with the expert axis on the grid writing router-weighted partials, their fixed-order sum. hidden [1, K]
    -> [1, K] in hidden's dtype. `tables`: the experts' pointer tables and strides per device (`gate_ptrs[dev]`,
    `up_ptrs[dev]`, `down_ptrs[dev]`, `gate_stride_bk/bn[dev]`, `down_stride_bk/bn[dev]`), indexed by what
    `flat_indices` holds — an expert's number, or its slot."""
    from nbxtensor.ops.moe_decode_vec import (
        moe_gateup_vec_half_kernel, moe_down_split_vec_half_kernel, moe_part_reduce_kernel)

    dt = hidden_states.nbx_dtype
    x = hidden_states.reshape(K)
    h = NBXTensor.empty((top_k, N_gate), dtype=NBXDtype.float32, device=f"cuda:{dev}")
    part = NBXTensor.empty((top_k, K), dtype=NBXDtype.float32, device=f"cuda:{dev}")
    out32 = NBXTensor.empty((K,), dtype=NBXDtype.float32, device=f"cuda:{dev}")
    set_device(h)
    # the tiles are each key's certified configuration (`moe_decode_vec._HALF_VEC_CONFIGS`, `_PART_REDUCE_CONFIGS`)
    moe_gateup_vec_half_kernel[lambda META: (top_k, triton.cdiv(N_gate, META["BLOCK_N"]))](
        x, flat_indices, tables.gate_ptrs[dev], tables.up_ptrs[dev], h, K, N_gate,
        tables.gate_stride_bk[dev], tables.gate_stride_bn[dev], top_k, W_BF16=bool(bf16))
    moe_down_split_vec_half_kernel[lambda META: (top_k, triton.cdiv(K, META["BLOCK_N"]))](
        h, flat_indices, flat_scores, tables.down_ptrs[dev], part, N_gate, K,
        tables.down_stride_bk[dev], tables.down_stride_bn[dev], top_k, W_BF16=bool(bf16))
    moe_part_reduce_kernel[lambda META: (triton.cdiv(K, META["BLOCK_N"]),)](part, out32, K, TOP_K=top_k)
    return out32.to(dt).reshape(1, K)
