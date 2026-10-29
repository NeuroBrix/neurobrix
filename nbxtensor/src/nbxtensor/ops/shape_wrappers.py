"""Shape and copy wrappers: concatenation, stacking, padding, flips, rolls, repeats, triangles, pixel (un)shuffle, upsampling, broadcasting. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import linalg_wrappers as _la
from nbxtensor.ops import reduction_wrappers as _red
from nbxtensor.ops import wrappers as _core
_TRIU_MBS, _TRIU_NBS = 64, 512            # triu / tril 2D
_TRIU_BATCH_BS, _TRIU_MN_BS = 32, 1024    # triu / tril batch


def grid_sampler_2d_wrapper(inp, grid, interpolation_mode=0,
                            padding_mode=0, align_corners=False):
    """aten::grid_sampler_2d — pure Triton 2D grid sampler.

    ATen integer mode encoding: interpolation 0=bilinear, 1=nearest,
    2=bicubic; padding 0=zeros, 1=border, 2=reflection. Implemented:
    bilinear+bicubic x zeros+border x both align flags (the GLM-4.1V /
    Qwen-VL pos-embed adaptation uses bicubic+border+align_corners=False);
    ZERO FALLBACK on the rest. fp32 coordinate math and accumulation
    (ATen opmath), output cast back to the input dtype."""
    from nbxtensor.ops.grid_sampler import grid_sampler_2d_kernel

    interpolation_mode = int(interpolation_mode)
    padding_mode = int(padding_mode)
    if interpolation_mode not in (0, 2):
        raise RuntimeError(
            "ZERO FALLBACK: grid_sampler_2d interpolation_mode="
            f"{interpolation_mode} not implemented (0=bilinear, 2=bicubic).")
    if padding_mode not in (0, 1):
        raise RuntimeError(
            "ZERO FALLBACK: grid_sampler_2d padding_mode="
            f"{padding_mode} not implemented (0=zeros, 1=border).")

    inp = inp.contiguous()
    grid = grid.contiguous()
    N, C, H, W = inp.shape
    gN, oH, oW, two = grid.shape
    out_spatial = oH * oW
    out32 = NBXTensor.empty((N, C, oH, oW), dtype=NBXDtype.float32,
                            device=inp.device)
    inp32 = inp.to(NBXDtype.float32)
    grid32 = grid.to(NBXDtype.float32)

    BLOCK_C = min(triton.next_power_of_2(max(C, 1)), 256)
    launch_grid = (out_spatial, triton.cdiv(C, BLOCK_C), N)
    set_device(inp32)
    grid_sampler_2d_kernel[launch_grid](
        inp32, grid32, out32,
        C, H, W, out_spatial,
        C * H * W, H * W, W, 1,
        out_spatial * 2, 2, 1,
        C * out_spatial, out_spatial, 1,
        MODE=interpolation_mode,
        PAD_BORDER=padding_mode,
        ALIGN=1 if align_corners else 0,
        BLOCK_C=BLOCK_C,
    )
    return out32.to(inp.dtype) if inp.dtype != NBXDtype.float32 else out32


def copy_to(x, dtype: object) :
    x = x.contiguous()
    output = NBXTensor.empty_like(x, dtype=dtype)
    set_device(x)
    copy_forward_kernel[_ew._1d_grid(x.numel())](x, output, x.numel(), **_ew_meta(x.numel()))
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
        triton.cdiv(total_threads, _ew_meta(total_threads)['BLOCK_SIZE']),
        triton.cdiv(N * C, 4),
    )
    set_device(output)
    s_n, s_c, s_h, s_w = x.stride()
    upsample_nearest2d_kernel[grid](
        output, x, N, C, OH, OW, IH, IW, s_n, s_c, s_h, s_w,
        reciprocal_scale_h, reciprocal_scale_w,
        **_ew_meta(total_threads),
    )
    return output


def triu_wrapper(x, diagonal: int = 0) :
    """Upper triangular. Wrapper from FlagGems."""
    x = x.contiguous()
    out = NBXTensor.empty_like(x)
    assert x.ndim >= 2
    M, N = x.shape[-2:]

    if x.ndim == 2:
        grid = (triton.cdiv(M, _TRIU_MBS),)
        set_device(x)
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
        set_device(B)
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
        set_device(x)
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
        set_device(B)
        tril_batch_kernel[grid](B, out.reshape(batch, M * N), batch, M * N, N, diagonal,
                                BATCH_BLOCK_SIZE=_TRIU_BATCH_BS, MN_BLOCK_SIZE=_TRIU_MN_BS,
                                num_warps=4)
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

    # Every rolled dim at once, written into ONE output: per dim the tail [n-s, n) lands at [0, s) and the head
    # [0, n-s) at [s, n), so the result is 2^k blocks copied from x (read by its strides, `copy_`) into their places.
    # Rolling one dim after the other made a whole intermediate per dim beside its input and output — HAT-S at 2048
    # px on 16 GB: the window shift's second dim held three 2 304 MB maps where the op's price holds two (2026-10-22).
    from itertools import product
    moves = []
    for shift, dim in zip(shifts, dims):
        dim = dim % x.ndim
        n = int(x.shape[dim])
        s = int(shift) % n
        if s:
            moves.append([(dim, n - s, 0, s), (dim, 0, s, n - s)])     # (dim, from, to, length)
    if not moves:
        return x.contiguous()
    out = NBXTensor.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
    for blocks in product(*moves):
        src, dst = x, out
        for dim, frm, to, length in blocks:
            src = src.narrow(dim, frm, length)
            dst = dst.narrow(dim, to, length)
        dst.copy_(src)
    return out


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
            set_device(result)
            flip_1d_kernel[_ew._1d_grid(result.numel())](
                result, output, result.numel(),
                **_ew_meta(result.numel()))
            result = output
        else:
            output = NBXTensor.empty_like(result)
            n_elements = result.numel()
            dim_size = result.shape[dim]
            # stride_dim = product of dims after flip dim
            stride_dim = 1
            for i in range(dim + 1, result.ndim):
                stride_dim *= result.shape[i]
            grid = _ew._1d_grid(n_elements)
            set_device(result)
            flip_strided_kernel[grid](
                result, output, n_elements,
                dim_size, stride_dim, stride_dim,
                **_ew_meta(n_elements))
            result = output
    return result


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
    from nbxtensor.ops.fused_upsample_conv import BroadcastClonePyroxy

    if isinstance(x, BroadcastClonePyroxy):
        return _pixel_shuffle_broadcast_aware(x, upscale_factor)

    N, C_in, H, W = x.shape
    r = upscale_factor
    C_out = C_in // (r * r)
    OH, OW = H * r, W * r
    x = x.contiguous()
    output = NBXTensor.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C_out * OH * OW
    grid = _ew._1d_grid(total)
    set_device(x)
    pixel_shuffle_kernel[grid](
        x, output, total,
        C_out, H, W, r, OH, OW,
        *x.stride(), *output.stride(),
        **_ew_meta(total))
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
    grid = _ew._1d_grid(total)
    set_device(view)
    pixel_shuffle_broadcast_aware_kernel[grid](
        view, output, total,
        C_out, OH, OW, r, bcast,
        *view.stride(), *output.stride(),
        **_ew_meta(total))
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
    grid = _ew._1d_grid(total)
    set_device(x)
    pixel_unshuffle_kernel[grid](
        x, output, total,
        C, IH, IW, r, C_out, OH, OW,
        *x.stride(), *output.stride(),
        **_ew_meta(total))
    return output


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
        triton.cdiv(total_threads, _ew_meta(total_threads)['BLOCK_SIZE']),
        triton.cdiv(N * C, 4),
    )
    set_device(output)
    upsample_nearest3d_kernel[grid](
        output, x, N, C, OD, OH, OW, ID, IH, IW,
        reciprocal_scale_d, reciprocal_scale_h, reciprocal_scale_w,
        **_ew_meta(total_threads),
    )
    return output


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
    cumsum = _red.cumsum_wrapper(repeats, 0)
    result_size = int(cumsum.select(0, -1).item())

    out = NBXTensor.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    BLOCK_SIZE = 32
    set_device(repeats)
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
# PADDING OPS — via Triton kernels
# ===========================================================================

def constant_pad_nd_wrapper(x, pad_list, value: float = 0.0) :
    """constant_pad_nd via Triton kernel."""
    from nbxtensor.ops.pad_op import constant_pad_2d_kernel, constant_pad_1d_kernel, constant_pad_nd_kernel
    x = _core._ensure_cuda(x).contiguous()

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

    if ndim > 5:
        raise NotImplementedError(f"constant_pad_nd: {ndim}-D input; the one-pass kernel takes up to 5 dims")
    output = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)
    total = output.numel()

    if total == 0:
        return output

    from nbxtensor.tensor import _index_wide
    _wide = _index_wide(total, max_linear_offset(x.shape, x.stride()), max_linear_offset(output.shape, output.stride()))

    if ndim >= 4 and pad_len == 4:
        # 4D: use optimized 2D kernel (N,C,H,W)
        batch_channels = 1
        for d in range(ndim - 2):
            batch_channels *= x.shape[d]
        in_h, in_w = x.shape[-2], x.shape[-1]
        out_h, out_w = out_shape[-2], out_shape[-1]
        constant_pad_2d_kernel[_ew._1d_grid(total)](
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
            **_ew_meta(total), WIDE=_wide,
        )
    elif pad_len == 2:
        batch_size = 1
        for d in range(ndim - 1):
            batch_size *= x.shape[d]
        in_size = x.shape[-1]
        out_size = out_shape[-1]
        constant_pad_1d_kernel[_ew._1d_grid(total)](
            x, output, total,
            in_size, out_size, batch_size,
            pad_before[-1],
            float(value),
            **_ew_meta(total), WIDE=_wide,
        )
    else:
        # General N-D constant pad in ONE pass (any dims, a middle temporal dim of a 5-D [B,C,T,H,W] causal pad
        # included): every output element reads its input element or the value. The former path padded one dim at
        # a time through `cat`, holding one output-sized copy per padded dim beside an unused output buffer: 3 extra
        # outputs on Wan2.1's causal pad (6,182 MB beside a 2,060 MB result, 2026-10-17), unpriced by any plan.
        dims_in = [1] * (5 - ndim) + list(x.shape)
        dims_out = [1] * (5 - ndim) + list(out_shape)
        strides = [0] * (5 - ndim) + [x.stride(i) for i in range(ndim)]
        before = [0] * (5 - ndim) + list(pad_before)
        constant_pad_nd_kernel[_ew._1d_grid(total)](
            x, output, total,
            *dims_in, *dims_out, *strides, *before,
            float(value),
            **_ew_meta(total), WIDE=_wide,
        )

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


def _replication_pad_last(x, pad_list, n_axes: int):
    """Replication pad (PyTorch `replicate`) of the last `n_axes` dims, `pad_list` in torch's last-dim-first order
    ([W_left, W_right, H_top, H_bot, D_front, D_back]), in ONE pass over the output with the input read through its
    strides (`replication_pad_nd_kernel`). The former narrow + expand + cat held a contiguous copy of the input and one
    output-sized tensor per padded axis: on mochi-1-preview's VAE (3 GB activations) up to three of them beside the
    result, priced by no plan (2026-10-28)."""
    from nbxtensor.ops.pad_op import replication_pad_nd_kernel
    from nbxtensor.tensor import _index_wide
    pads = ([int(v) for v in pad_list] + [0] * (2 * n_axes))[:2 * n_axes]
    ndim = len(x.shape)
    if ndim > 5 or ndim < n_axes:
        raise NotImplementedError(f"replication pad of {n_axes} dims on a {ndim}-D input: the one-pass kernel takes "
                                  f"{n_axes} to 5 dims")
    before = [0] * ndim
    for k in range(n_axes):
        before[ndim - 1 - k] = pads[2 * k]
    out_shape = list(x.shape)
    for k in range(n_axes):
        out_shape[ndim - 1 - k] += pads[2 * k] + pads[2 * k + 1]
    output = NBXTensor.empty(out_shape, dtype=x.dtype, device=x.device)
    total = output.numel()
    if total == 0:
        return output
    _wide = _index_wide(total, max_linear_offset(x.shape, x.stride()),
                        max_linear_offset(output.shape, output.stride()))
    pad5 = 5 - ndim
    set_device(x)
    replication_pad_nd_kernel[_ew._1d_grid(total)](
        x, output, total,
        *([1] * pad5 + list(x.shape)), *([1] * pad5 + out_shape),
        *([0] * pad5 + [x.stride(i) for i in range(ndim)]), *([0] * pad5 + before),
        **_ew_meta(total), WIDE=_wide,
    )
    return output


def replication_pad1d_wrapper(x, pad_list) :
    """Replication pad 1D (last dim) — F.pad(mode='replicate'), one pass (`_replication_pad_last`)."""
    return _replication_pad_last(x, pad_list, 1)


def replication_pad2d_wrapper(x, pad_list) :
    """Replication pad 2D (last two dims) — F.pad(mode='replicate'), one pass (`_replication_pad_last`)."""
    return _replication_pad_last(x, pad_list, 2)


def replication_pad3d_wrapper(x, pad_list) :
    """Replication pad 3D (last three dims) — F.pad(mode='replicate'), one pass (`_replication_pad_last`). Used by
    the Mochi / 5D-video VAE decoder edge pad on [B, C, D, H, W]."""
    return _replication_pad_last(x, pad_list, 3)
    _wide = _index_wide(total, max_linear_offset(x.shape, x.stride()),
                        max_linear_offset(output.shape, output.stride()))
    pad5 = 5 - ndim
    set_device(x)
    replication_pad_nd_kernel[_ew._1d_grid(total)](
        x, output, total,
        *([1] * pad5 + list(x.shape)), *([1] * pad5 + out_shape),
        *([0] * pad5 + [x.stride(i) for i in range(ndim)]), *([0] * pad5 + before),
        **_ew_meta(total), WIDE=_wide,
    )
    return output


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
        return _la.upsample_bilinear2d_wrapper(x, size, align_corners, scale_factor)
    elif mode == "linear":
        return _la.upsample_linear1d_wrapper(x, size, align_corners, scale_factor)
    raise RuntimeError(
        f"[--triton mode] interpolate mode='{mode}' not implemented for {x.ndim}D input."
    )
