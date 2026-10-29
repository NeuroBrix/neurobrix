"""Indexing wrappers: gather, scatter, index_select/add/put, embedding, masked fill, where, nonzero, bucketize. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import reduction_wrappers as _red
from nbxtensor.ops import wrappers as _core


def where_wrapper(cond, x, y) :
    # Broadcast all three to same shape
    if cond.shape != x.shape or x.shape != y.shape:
        out_shape = broadcast_shapes(cond.shape, x.shape, y.shape)
        cond = cond.expand(out_shape).contiguous()
        x = x.expand(out_shape).contiguous()
        y = y.expand(out_shape).contiguous()
    else:
        cond, x, y = cond.contiguous(), x.contiguous(), y.contiguous()
    output = NBXTensor.empty_like(x)
    _set_device(cond)
    where_forward_kernel[_ew._1d_grid(x.numel())](cond, x, y, output, x.numel(), **_ew_meta(x.numel()))
    return output


def masked_fill(x, mask, value: float) :
    # Broadcast mask to match x shape
    if x.shape != mask.shape:
        out_shape = broadcast_shapes(x.shape, mask.shape)
        x = x.expand(out_shape).contiguous()
        mask = mask.expand(out_shape).contiguous()
    else:
        x, mask = x.contiguous(), mask.contiguous()
    value = _core._to_scalar(value) if isinstance(value, NBXTensor) else value
    output = NBXTensor.empty_like(x)
    _set_device(x)
    masked_fill_forward_kernel[_ew._1d_grid(x.numel())](
    x, mask, output, x.numel(), float(value),
    **_ew_meta(x.numel()))
    return output


def is_nonzero_wrapper(x):
    """aten::is_nonzero — returns the scalar truthiness of a 1-element tensor.

    PyTorch's native impl returns a Python bool, but traced graphs feed
    the output back through tensor ops, so we return a 0-d NBXTensor
    (uint8) to match the graph's expected dtype surface.
    """
    import numpy as np
    val = x.item()
    return NBXTensor.from_numpy(np.array(bool(val), dtype=np.uint8))


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
    # weight.shape[0] = the id bound the kernel traps on (OOB parity with torch).
    embedding_kernel[M,](
        output, indices, weight, weight.shape[0],
        *fault_channel(EMBEDDING_OOB, output),
        N, BLOCK_SIZE)
    return output


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

    # A gather along a middle axis of a contiguous input reads it as (outer, N, inner) and
    # writes the output in its final layout: no movedim copy, no permute copy (the copy lever,
    # 2026-09-07). A non-contiguous input is materialised once first — the line that justifies
    # it: the kernels index a flat buffer.
    if dim != x.ndim - 1:
        x = x.contiguous()
        N = inp_shape[dim]
        outer = 1
        for d_ in inp_shape[:dim]:
            outer *= d_
        inner = 1
        for d_ in inp_shape[dim + 1:]:
            inner *= d_
        out_shape = list(inp_shape); out_shape[dim] = index_len
        out = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)
        total = outer * index_len * inner
        if total > 0:
            from nbxtensor.ops.index_select import index_select_mid_kernel
            _set_device(x)
            _m = _ew_meta(total)
            index_select_mid_kernel[_ew_grid(total, _m)](
                x, out, outer, N, inner, index, index_len,
                *fault_channel(INDEX_SELECT_OOB, out),
                BLOCK=_m['BLOCK_SIZE'], num_warps=_m['num_warps'])
        return out
    x = x.contiguous()
    N = inp_shape[dim]
    M = x.numel() // N
    out_shape = list(x.shape)
    out_shape[-1] = index_len
    out = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)
    # Diagnostic (NBX_INDEX_SELECT_SENTINEL=1, §8): pre-fill the output with a
    # sentinel and count what the kernel leaves unwritten — the kernel stores
    # under `index_valid_mask`, so an out-of-range index leaves its output
    # element as whatever the allocation held (pool gate 2026-09-02).
    _sentinel = os.environ.get("NBX_INDEX_SELECT_SENTINEL") == "1"
    if _sentinel:
        out.fill_(float("nan") if x.dtype in (NBXDtype.float16, NBXDtype.float32, NBXDtype.bfloat16) else -7777)

    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(index_len))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(index_len, BLOCK_N))
    _set_device(x)
    index_select_kernel[grid](
        x, out, M, N, index, index_len,
        *fault_channel(INDEX_SELECT_OOB, out),
        BLOCK_M, BLOCK_N)
    if _sentinel:
        import numpy as _np
        _o = out.numpy()
        _unwritten = int(_np.isnan(_o).sum()) if _o.dtype.kind == "f" else int((_o == -7777).sum())
        if _unwritten:
            _ix = index.numpy()
            print(f"[INDEX_SELECT_SENTINEL] {_unwritten}/{_o.size} output elements UNWRITTEN "
                  f"(index dtype {index.dtype} len {index_len} min {int(_ix.min())} max {int(_ix.max())} N={N} M={M})", flush=True)

    return out


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
    gather_kernel[_ew._1d_grid(N)](
        input, index, out,
        N,
        inp_dim_stride,
        idx_stride_outer, idx_stride_dim, idx_stride_inner,
        inp_stride_outer, inp_stride_inner,
        out_stride_outer, out_stride_dim, out_stride_inner,
        outer_size, dim_size, inner_size,
        **_ew_meta(N),
    )
    return out


def masked_scatter_wrapper(x, mask, source) :
    """aten::masked_scatter — fill mask-True elements of `x` from the
    flat `source` buffer in row-major order (torch semantics).

    R33-pure composition over existing Triton bricks (no dedicated scan
    kernel): the running index of each True element is the inclusive
    cumsum of the flat mask minus one; False lanes gather a clamped
    dummy and are discarded by the final `where`.

        idx      = cumsum(mask.float32) - 1      (exclusive True count)
        gathered = source_flat[clamp(idx, 0, n_src-1).int64]
        out      = where(mask, gathered, x)

    The count arithmetic runs in float32 (the well-trodden elementwise
    path) — exact for counts < 2^24, guarded below; a dedicated scan
    kernel is the extension past that ceiling (and the optimisation path
    if this op ever lands on a hot loop beyond the twice-per-forward
    splice — the flat single-row cumsum walks its partial-sum chain
    host-side, numel/4096 launches).

    First consumer: the MiniCPM-o llm.model graph (in-graph bool-mask
    modality splice — expanded [B, S, H] mask, flattened [n, H] source).
    v1 contract: `mask` must match `x`'s shape exactly (the traced form);
    broadcasting masks raise a named error.
    """
    if tuple(mask.shape) != tuple(x.shape):
        raise RuntimeError(
            "ZERO FALLBACK: masked_scatter v1 requires mask.shape == "
            f"self.shape (got mask {tuple(mask.shape)} vs self "
            f"{tuple(x.shape)}); broadcast-mask support lands with its "
            "first consumer graph.")
    n = x.numel()
    n_src = source.numel()
    x_flat = x.contiguous().reshape(n)
    if n_src == 0:
        # torch semantics: valid only when the mask has no True element;
        # the splice flows feed a 1-row stub instead, so an empty source
        # here means an all-False mask — the input passes through.
        return x_flat.reshape(*x.shape)
    if n >= (1 << 24):
        raise RuntimeError(
            "ZERO FALLBACK: masked_scatter float32 count path is exact "
            f"only below 2^24 elements (got {n}); a dedicated int scan "
            "kernel is the extension for larger splices.")
    mask_flat = mask.contiguous().reshape(n)
    src_flat = source.contiguous().reshape(n_src)
    if src_flat.dtype != x_flat.dtype:
        src_flat = src_flat.to(x_flat.dtype)
    idx = _red.cumsum_wrapper(mask_flat, dim=0, dtype=NBXDtype.float32)
    idx = _ew.sub(idx, 1.0)
    idx = _ew.clamp(idx, min_val=0.0, max_val=float(n_src - 1))
    gathered = gather_wrapper(src_flat, 0, idx.to(NBXDtype.int64))
    out = where_wrapper(mask_flat, gathered, x_flat)
    return out.reshape(*x.shape)


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
    scatter_kernel[_ew._1d_grid(N)](
        src, index, out,
        N,
        out.stride(dim),
        src.stride(0) if dim > 0 else 0, src.stride(dim), src.stride(-1) if dim < src.ndim - 1 else 0,
        index.stride(0) if dim > 0 else 0, index.stride(dim), index.stride(-1) if dim < index.ndim - 1 else 0,
        out.stride(0) if dim > 0 else 0, out.stride(-1) if dim < out.ndim - 1 else 0,
        outer_size, dim_size, inner_size,
        **_ew_meta(N),
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
    scatter_add_kernel[_ew._1d_grid(N)](
        src, index, out,
        N,
        out.stride(dim),
        src.stride(0) if dim > 0 else 0, src.stride(dim), src.stride(-1) if dim < src.ndim - 1 else 0,
        index.stride(0) if dim > 0 else 0, index.stride(dim), index.stride(-1) if dim < index.ndim - 1 else 0,
        out.stride(0) if dim > 0 else 0, out.stride(-1) if dim < out.ndim - 1 else 0,
        outer_size, dim_size, inner_size,
        **_ew_meta(N),
    )
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
    _inner = _ew_meta(inner_size)
    n_inner_tiles = (inner_size + _inner['BLOCK_SIZE'] - 1) // _inner['BLOCK_SIZE']
    grid = (outer_size * inp_shape_dim * n_inner_tiles,)
    _set_device(x)
    index_add_gather_kernel[grid](
        x, source, index, out,
        dim_size, inp_shape_dim, inner_size, alpha,
        src_stride_outer, src_stride_dim, src_stride_inner,
        INNER_BLOCK=_inner['BLOCK_SIZE'], num_warps=_inner['num_warps'],
    )
    return out


# ---------------------------------------------------------------------------
# Index Put
# ---------------------------------------------------------------------------

def nonzero_1d_wrapper(mask, expected=None) -> "NBXTensor":
    """Ascending int64 indices of the True elements of a 1-D mask.

    Deterministic single-program compaction (kernels/ops/nonzero.py) —
    no atomics, indices come out in order, bit-reproducible. The output
    length is data-dependent: worst-case buffer + one count readback
    (the host-boundary sync every data-dependent-shape op requires),
    narrowed to [n] (a leading-dim narrow of a contiguous buffer stays
    contiguous — no guard needed).

    `expected`: the count the caller's own contract states (a masked scatter's values have exactly one row per True
    element). A census shadow launches nothing and reads no value, so it takes that count as the extent — the read it
    replaced returned a count the shadow never computed (Qwen3-VL's DeepStack write, `values` [196, 2048], formed one
    row: every Qwen3-VL coverage cell rc 1 since the shadow existed). A real run reads the count and refuses by name a
    mask that disagrees with the caller's contract."""
    from nbxtensor.ops.nonzero import nonzero_1d_kernel
    assert mask.ndim == 1, f"nonzero_1d expects 1-D, got {mask.ndim}-D"
    m = mask.contiguous()
    n = int(m.shape[0])
    if expected is not None:
        from nbxtensor.host import need as _need   # the application's launch census (nbxtensor.host)
        if _need("census")().active():
            return NBXTensor.empty([int(expected)], dtype=NBXDtype.int64, device=m.device)
    out = NBXTensor.empty([max(n, 1)], dtype=NBXDtype.int64, device=m.device)
    count = NBXTensor.zeros([1], dtype=NBXDtype.int32, device=m.device)
    if n > 0:
        BLOCK = min(1024, triton.next_power_of_2(n))
        _set_device(m)
        nonzero_1d_kernel[(1,)](m, out, count, n, BLOCK)
        k = int(count.numpy()[0])
    else:
        k = 0
    if expected is not None and k != int(expected):
        raise RuntimeError(f"nonzero: the mask holds {k} True elements and the caller's contract states {int(expected)} "
                           f"(a masked scatter's values have one row per True element)")
    return out[:k]


def bucketize_wrapper(x, boundaries, out_int32: bool = False,
                      right: bool = False) -> "NBXTensor":
    """aten::bucketize.Tensor — per-element boundary index (linear scan
    kernel, kernels/ops/bucketize.py). Boundary vectors on this path
    are decode-control scale (one per temporal frame / packed segment),
    so the O(n_boundaries) scan is exact, branch-free and
    bit-reproducible."""
    from nbxtensor.ops.bucketize import bucketize_kernel
    xc = x.contiguous()
    bc = boundaries.contiguous()
    n = 1
    for d in xc.shape:
        n *= int(d)
    nb = int(bc.shape[0]) if bc.ndim >= 1 else 1
    if nb == 0:
        # No boundary is below any element: every index is 0, and the empty
        # boundary vector has no buffer to bind (a one-image Qwen3-VL request
        # buckets over cu_seqlens[1:-1], which is empty).
        out = NBXTensor.zeros(list(xc.shape) or [1], dtype=NBXDtype.int64,
                              device=xc.device)
        return out.to(NBXDtype.int32) if out_int32 else out
    out = NBXTensor.empty(list(xc.shape) or [1], dtype=NBXDtype.int64,
                          device=xc.device)
    if n > 0:
        BLOCK = min(1024, triton.next_power_of_2(n))
        grid = ((n + BLOCK - 1) // BLOCK,)
        _set_device(xc)
        bucketize_kernel[grid](xc, bc, out, n, nb,
                               RIGHT=bool(right), BLOCK_SIZE=BLOCK)
    if out_int32:
        out = out.to(NBXDtype.int32)
    return out


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
    from nbxtensor.dtype.nbx_dtype import NBXDtype

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
        fill_val = _core._to_scalar(values) if isinstance(values, NBXTensor) else float(values)
        return masked_fill(x, mask_b, fill_val)
    if (("bool" in _idx_dtn or "uint8" in _idx_dtn)
            and list(idx.shape) == list(x.shape[:idx.ndim])):
        # General boolean-mask scatter with tensor values: x[mask] = vals
        # where mask is a leading-dim PREFIX of x and vals is [n, ...]
        # (n = mask True-count) — the DeepStack injection write
        # (hidden[visual_pos_masks, :] = local_this). NBX stores bool
        # masks as uint8 (bool-repr doctrine), hence the same name+
        # shape-prefix conjunction as the masked_fill route above.
        # Lower to the integer path: flatten the masked leading dims,
        # compact the mask to ascending row indices (deterministic
        # nonzero), scatter through the proven single-index route,
        # restore the shape.
        xc = x.contiguous()
        flat_rows = 1
        for _s in idx.shape:
            flat_rows *= int(_s)
        tail = list(x.shape[idx.ndim:])
        x_flat = xc.reshape([flat_rows] + tail)
        # x[mask] = vals: vals holds one row per True element (its leading dim) unless it broadcasts a single row
        expected = (int(values.shape[0]) if isinstance(values, NBXTensor) and values.ndim == 1 + len(tail)
                    and int(values.shape[0]) != 1 else None)
        rows = nonzero_1d_wrapper(idx.reshape([flat_rows]), expected=expected)
        if int(rows.shape[0]) == 0:
            # All-False mask (text-mode empty-visual stubs): zero writes —
            # a fresh copy per the functional contract, same as the
            # integer path's clone-then-scatter with an empty scatter.
            return x.clone()
        out_flat = index_put_wrapper(x_flat, [rows], values, accumulate)
        return out_flat.reshape(list(x.shape))
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
    index_put_kernel[_ew._1d_grid(N)](
        out, idx, vbuf,
        T, N, out.shape[0],  # R: the row bound the kernel traps on (OOB parity with torch)
        *fault_channel(INDEX_PUT_OOB, out),
        VAL_SCALAR=val_scalar,
        ACCUMULATE=bool(accumulate),
        **_ew_meta(N),
    )
    return out
