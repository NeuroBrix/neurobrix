"""Convolution and pooling wrappers: conv1d/2d/3d (strided, depthwise, transposed, chunked), im2col, pooling. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import index_wrappers as _idx
from nbxtensor.ops import shape_wrappers as _shp
from nbxtensor.ops import wrappers as _core
_POOL_BH, _POOL_BW = 4, 16                # avg_pool2d / max_pool2d

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
# same lever as the conv2d band-streaming above (P-SANA-4KPX Step 1 class).
# Even with the #37 eager frees, the one-shot temporal decomposition needs
# several FULL folded transients simultaneously (temporal pad copy, x2 fold,
# conv2d output incl. band-streaming machinery, permuted copy, accumulator +
# accumulate transient) — ~4-5x the activation bytes. cuDNN in compiled mode
# needs input + output + a bounded workspace for the same op, so a component
# Prism correctly sized as "fits" for compiled can OOM in triton only
# (Allegro-TI2V vae_encoder at 720x1280x12f fp32: 5.66 GB per plane →
# 30.4 GB live at aten.convolution::1 on a 32 GB V100). Gate, two stages:
#   1. deterministic floor — the folded transient must exceed
#      `tiling_sizes.conv3d_chunk_bytes()` (`conv3d.chunk_bytes`), so
#      small/validated convs never even query the driver;
#   2. exact-fit check — the one-shot path's real allocation peak is
#      compared to driver-free bytes; when it fits, the EXACT one-shot
#      path runs (byte-identical behavior for every config that is green
#      today, on any hardware — R23), when it cannot fit (guaranteed OOM)
#      the wrapper streams over temporal output chunks whose per-transient
#      size is bounded by this same value. (Stage 2 is now Prism's PLAN
#      decision, `tiling_sizes.conv3d_need` against the op budget.)
# Chunking splits the folded batch axis (B*T frames): spatial conv2d is
# independent per frame and the per-frame kt accumulation order is
# preserved, so the math is unchanged.

# Diagnostic trace — when set, every conv2d_wrapper call prints the
# (in_shape, out_shape, kernel, groups, output_MB) so we can see the actual
# spatial workload arriving in triton mode. Used by P-SANA-4KPX-RUNTIME
# Step 1 verification on Sana 4Kpx (was the wrapper-internal band-streaming
# triggered? what shapes are dominant?).
_NBX_CONV2D_TRACE = os.environ.get("NBX_CONV2D_TRACE", "0") == "1"


def _conv_width_key(in_h, out_h, in_w, out_w, kw, stride_w, pad_w, dil_w):
    """The width a convolution's autotune key carries. A convolution whose spatial extent is
    ONE ROW is a 1-D convolution over a sequence — a vocoder's frames, a mel spectrogram's
    time axis — and that extent is the request's (a speech of 1 716 tokens and one of 1 856
    gave chatterbox's vocoder two different exact keys for every layer, 2026-09-21): it keys
    on the profile's ladder like a GEMM's M, the output width derived from the input's top by
    the convolution's own arithmetic so a certifier synthesising at the top forms the same
    key. A 2-D convolution keeps its exact extents (bounded by resolutions and tile edges;
    the ladder measured 37–40 % loss where the conv optimum flips)."""
    return _lk.conv_width_key(in_h, out_h, in_w, out_w, kw, stride_w, pad_w, dil_w)


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
    out_dtype = _core._NBX_COMPUTE_DTYPE if _core._NBX_COMPUTE_DTYPE is not None else x.dtype

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
        output = _conv_bias_inplace(output, bias)
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
    _t_out = (T + 2 * pt - dt * (kt - 1) - 1) // st + 1
    if _t_out <= 0:
        from nbxtensor.memory.allocator import ImpossibleExtentError
        raise ImpossibleExtentError(
            f"convolution produces no frame: input {tuple(x.shape)}, weight "
            f"{tuple(weight.shape)}, temporal stride {st}, padding {pt}, dilation {dt} -> "
            f"T_out={_t_out}")

    # The chunked variant is a PLAN decision (`core/module/tiling_sizes.py`; the supervisor's
    # decision 2, 2026-09-29): Prism registers `conv3d_chunked_wrapper` as this op's interceptor
    # when the one-shot peak exceeds the op's planned budget. It read the driver's free bytes here,
    # so a key a live run formed depended on the memory free at that instant.
    if pt > 0:
        x = _shp.constant_pad_nd_wrapper(x, [0, 0, 0, 0, pt, pt], 0.0)
    Tp = x.shape[2]
    T_out = (Tp - dt * (kt - 1) - 1) // st + 1
    if st == 1:
        # Every temporal tap read and written IN PLACE: the tap's frames x[b, :, start:start+T_out] permuted to
        # [T_out, Cin, H, W] are a strided view the conv kernel reads through its strides, and its result goes through
        # a [T_out, Cout, Ho, Wo] view of the [B, Cout, T_out, Ho, Wo] output. The two permute copies per tap were 65 %
        # of a VAE conv3d (1,128,9,242,362: 525 of 810 ms; 9 ms each once the copy kernel indexed in 32 bits, Mac
        # 2026-10-21). The first tap writes the output; each later one writes a tap buffer added to it.
        Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        out_dt = _core._NBX_COMPUTE_DTYPE if _core._NBX_COMPUTE_DTYPE is not None else x.dtype
        out = None
        for kti in range(kt):
            start = kti * dt
            w2 = weight[:, :, kti:kti + 1].contiguous().reshape(Cout, Cin_g, kh, kw)
            tap = NBXTensor.empty((B, Cout, T_out, Ho, Wo), device=x.device, dtype=out_dt)
            for b in range(B):
                x2 = x[b][:, start:start + T_out].permute(1, 0, 2, 3)
                conv2d_wrapper(x2, w2, None, (sh, sw), (ph, pw), (dh, dw), False, 0, groups,
                               out=tap[b].permute(1, 0, 2, 3))
            # each later tap added INTO the sum (the same add kernel, the same order: the same bits): `out + tap` held
            # the old sum, the tap and the new sum at once — three output-sized buffers where the plan priced one
            # (mochi-1-preview's VAE at the 8 GB rung, 1.5 GB each, 2026-10-28)
            out = tap if out is None else _ew.add_inplace_nbx(out, tap)
            tap = None
        if bias is not None:
            out = out + bias.reshape(1, Cout, 1, 1, 1)
        return out

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
            xs = _idx.index_select_wrapper(x, 2, idx)
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


def conv3d_chunked_wrapper(x, weight, bias=None, stride=1, padding=0, dilation=1,
                           transposed=False, output_padding=0, groups=1, *args, **kwargs):
    """A 3-D convolution Prism planned to chunk (op-level tiling, `conv3d_chunks`): the chunked
    path, the chunk sized from the folded bytes per output frame (`tiling_sizes.chunk_frames`)."""
    def _triple(v):
        return (v[0], v[1], v[2]) if isinstance(v, (list, tuple)) else (v, v, v)
    st, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)
    ds_in = dtype_size(x.nbx_dtype if hasattr(x, 'nbx_dtype') else x._dtype)
    ds_out = dtype_size(_core._NBX_COMPUTE_DTYPE) if _core._NBX_COMPUTE_DTYPE is not None else ds_in
    _need, frame = _ts.conv3d_need(x.shape, weight.shape, (st, sh, sw), (pt, ph, pw), (dt, dh, dw),
                                   ds_in, ds_out)
    return _conv3d_via_conv2d_chunked(x, weight, bias, st, sh, sw, pt, ph, pw, dt, dh, dw, groups, frame)


def _conv3d_via_conv2d_chunked(x, weight, bias, st, sh, sw, pt, ph, pw,
                               dt, dh, dw, groups, frame_bytes):
    """Temporal chunk-streaming variant of _conv3d_via_conv2d — identical
    math, bounded live set (`tiling_sizes.conv3d_need` is the gate).

    Streams the folded batch axis (B*T_out frames) in chunks sized so each
    transient stays under `tiling_sizes.conv3d_chunk_bytes()`. Per output frame the
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
        x = _shp.constant_pad_nd_wrapper(x, [0, 0, 0, 0, pt, pt], 0.0)
    Tp = x.shape[2]
    T_out = (Tp - dt * (kt - 1) - 1) // st + 1
    tc = _ts.chunk_frames(frame_bytes)
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
                xs = _idx.index_select_wrapper(x, 2, idx)
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


def _conv_bias_inplace(output, bias):
    """Per-channel conv bias, IN PLACE on the wrapper's own fresh output.

    Every conv-family wrapper ended with `add(output, bias.view(1,-1,1,1))`
    — an out-of-place broadcast allocating a SECOND full-size map per
    biased convolution. Measured cost: 2450 MB of the 8008 MB tile peak on
    real-esrgan-x8 @1024 (malloc trace, 2026-09-21). The in-place contract
    holds by construction at all four call sites: `output` is the
    wrapper's own fresh contiguous NCHW allocation with no other consumer.
    The bias is cast to the output's dtype (the standard add's wider-dtype
    rule would upcast the full map instead — the very copy this removes).
    Fixed launch config: no autotune key, nothing for any census.
    """
    from nbxtensor.ops.conv2d import conv2d_bias_inplace_kernel
    bias_c = bias if bias._dtype == output._dtype else bias.to(output._dtype)
    n = output.numel()
    hw = int(output.shape[-2]) * int(output.shape[-1])
    c = int(output.shape[1])
    _set_device(output)
    _m = _ew_meta(n)
    conv2d_bias_inplace_kernel[_ew_grid(n, _m)](
        output, bias_c.contiguous(), n, hw, c, BLOCK=_m['BLOCK_SIZE'], num_warps=_m['num_warps'])
    return output


def _conv_offsets_wide(*tensors) -> bool:
    """64-bit offsets in the convolution kernel when an operand's reach (its last element's offset, over its strides)
    is within a factor of two of 2^31: a masked lane computes an address past the extent (the grid's last block,
    the padding rows) before its mask drops it, and the factor covers that overshoot. Below it 32-bit offsets are
    exact and faster (17 % on Apple's ALUs, Mac 2026-10-20)."""
    reach = max(sum((int(n) - 1) * int(st) for n, st in zip(t.shape, t.stride())) + 1 for t in tensors)
    return reach >= (1 << 30)


def _conv_reads_strided(x) -> bool:
    """True when the implicit-GEMM conv kernel can read `x` through its strides in place: four non-negative strides
    with the width innermost (a permuted or sliced view of a channel-plane layout). Anything else is copied."""
    st = x.stride()
    return len(st) == 4 and all(int(v) >= 0 for v in st) and (int(st[3]) == 1 or int(x.shape[3]) == 1)


def conv2d_wrapper(
    x, weight, bias=None,
    stride=1, padding=0, dilation=1,
    transposed=False, output_padding=0, groups=1, out=None,
) :
    """Convolution forward. Handles both 1D (3D tensors) and 2D (4D tensors).
    Routes to conv1d_wrapper for 3D inputs, and to conv_transpose_wrapper when
    transposed=True (1D or 2D).

    Spatial band-streaming (P-SANA-4KPX-RUNTIME Step 1): when the output
    tensor would exceed `tiling_sizes.conv2d_band_bytes()` (`conv2d.band_bytes`), the wrapper
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

    # ATen broadcasts single-element stride/padding/dilation lists over
    # both spatial dims (padding="valid" convs record [0] — the
    # MiniCPM-o NaViT patch_embedding hit exactly that as an
    # IndexError here; same contract as the tracer-side estimator).
    if isinstance(stride, (list, tuple)):
        stride_h = stride[0]
        stride_w = stride[1] if len(stride) > 1 else stride_h
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        pad_h = padding[0] if len(padding) > 0 else 0
        pad_w = padding[1] if len(padding) > 1 else pad_h
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (list, tuple)):
        dil_h = dilation[0]
        dil_w = dilation[1] if len(dilation) > 1 else dil_h
    else:
        dil_h = dil_w = dilation

    N, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = weight.shape
    out_h = (in_h + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        from nbxtensor.memory.allocator import ImpossibleExtentError
        raise ImpossibleExtentError(
            f"convolution produces no output: input {tuple(x.shape)}, weight "
            f"{tuple(weight.shape)}, stride ({stride_h}, {stride_w}), padding ({pad_h}, "
            f"{pad_w}), dilation ({dil_h}, {dil_w}) -> out_h={out_h}, out_w={out_w}")

    # Self-managed dtype doctrine — VRAM-preserving (different from the
    # mm/bmm/addmm doctrine which is accumulation-overflow-protected).
    #
    # mm/bmm/addmm: deep matmul stacks risk fp16 accumulation overflow.
    #   Step 1 upcasts input fp16 → fp32 on pre-Ampere, output forced to
    #   fp32 (force_fp32 in _matmul_out_dtype). Precision target.
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
    # `launch_keys.conv_dtypes`: the pair narrowed to the narrowest when they differ — the
    # function the derived census keys with.
    x_to, w_to = _lk.conv_dtypes(x_nbx, w_nbx)
    if x_nbx != x_to:
        x = x.to(x_to)
    if w_nbx != w_to:
        weight = weight.to(w_to)

    # The implicit GEMM reads the input through its four strides, so a strided view goes in where it lies (a conv3d's
    # temporal slice permuted to [T, C, H, W]: its copy was 85 ms of a 235 ms VAE conv3d, Mac 2026-10-21); the
    # depthwise, band and matrix-unit routes take a contiguous copy, as before.
    x_c = x.contiguous() if not x.is_contiguous() and not _conv_reads_strided(x) else x
    # A weight the loader stored KRSC (`conv.weight_layout`) is read where it lies: the kernel
    # takes the weight's four strides; copied back to KCRS it would lose what the layout gains.
    w_c = weight if _lk.is_krsc_weight(weight) else weight.contiguous()

    # Step 4: output dtype = compute_dtype from per-component Prism context.
    # Falls back to x.dtype when no TritonSequence is active.
    out_dtype = _core._NBX_COMPUTE_DTYPE if _core._NBX_COMPUTE_DTYPE is not None else x.dtype

    # P-SANA-4KPX-RUNTIME Step 1 — kernel-level spatial band-streaming.
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

    # P-SANA-4KPX-RUNTIME Step 3 — depthwise specialization. The generic
    # im2col conv2d_forward_kernel is structurally inefficient for the
    # depthwise pattern (groups == in_c == out_c, weight (C,1,kh,kw)) on
    # Sana 4Kpx VAE: ~4.8 s per call vs cuDNN dedicated path ~2.6 ms,
    # ~1800x gap. Route to the dedicated stencil kernel instead.
    _route = _lk.conv2d_route(N, in_c, out_c, out_h, out_w, dil_h, dil_w, groups, out_nbx_dtype,
                              _ts.conv2d_band_bytes(),
                              depthwise_enabled=os.environ.get("NBX_DEPTHWISE_DISABLE", "0") != "1")
    if out is not None and _route in ("depthwise", "band"):
        raise RuntimeError(f"conv2d out=: the {_route} route writes its own result — refused")
    if _route == "depthwise":
        if _NBX_CONV2D_TRACE:
            print(f"[CONV2D] DEPTHWISE path (g={groups})", flush=True)
        return _depthwise_conv2d_dispatch(
            x_c.contiguous(), w_c, bias,
            N, in_c, in_h, in_w, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w,
            out_dtype,
        )

    if _route == "band":
        if _NBX_CONV2D_TRACE:
            print(f"[CONV2D] BAND-STREAM triggered (out > {_ts.conv2d_band_bytes()/1024/1024/1024:.1f}GiB)", flush=True)
        return _conv2d_band_streamed(
            x_c.contiguous(), w_c, bias,
            N, in_c, in_h, in_w, out_c, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, out_dtype, out_dtype_bytes,
        )

    if out is not None:
        # the caller's own [N, out_c, out_h, out_w] view (any strides) receives the result: a conv3d writes each temporal
        # tap straight into its [B, C, T, H, W] layout instead of a contiguous result permuted after
        if (tuple(out.shape) != (N, out_c, out_h, out_w) or bias is not None
                or (out._dtype != out_dtype and out.dtype != out_dtype)):
            raise RuntimeError(f"conv2d out=: a {tuple(out.shape)} {out._dtype} view for a ({N}, {out_c}, {out_h}, "
                               f"{out_w}) {out_dtype} result{' with a bias' if bias is not None else ''} — refused")
        output = out
    else:
        output = NBXTensor.empty((N, out_c, out_h, out_w), device=x.device, dtype=out_dtype)

    if _lk.matrix_unit_native(x_c.nbx_dtype, w_c.nbx_dtype, N * out_h * out_w):
        # The implicit GEMM on the profile's matrix unit (ops/conv2d_m8n8k4.py), bias fused; no autotune key.
        x_c = x_c.contiguous()
        # The matrix-unit kernel writes a contiguous result: a caller's strided view (a conv3d tap's frames in their
        # [B, C, T, H, W] layout) receives it by one copy. Refused instead, every CUDA video model whose conv3d runs
        # its taps through here failed at its first convolution (6 gate cells, 2026-10-22).
        dest = output if out is None else NBXTensor.empty((N, out_c, out_h, out_w), device=x.device, dtype=out_dtype)
        _conv2d_m8n8k4(x_c, w_c, bias, dest, N, in_c, in_h, in_w, out_c, out_h, out_w, kh, kw,
                       stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups)
        if out is not None:
            out.copy_(dest)
        return output

    # The key's `fp16` flag names an fp16 INPUT. It compared `x.dtype` — the Triton element type —
    # to an NBXDtype member and was False for every launch (5 378 directory entries carried it).
    # The kernel's only use of the flag casts the loaded blocks to fp16 before tl.dot, a no-op on
    # an fp16 input: TTIR, TTGIR, LLIR and PTX are identical with the flag right (sm_70, three
    # shapes, 2026-09-29), so the certified entries were MIGRATED to the right key, not re-proven
    # (tools/migrate_conv_fp16_key.py).
    fp16 = (x.nbx_dtype if hasattr(x, 'nbx_dtype') else x._dtype) == NBXDtype.float16

    # Phase 1.5 conv2d autotune: BLOCK_SIZE_*/num_warps/num_stages chosen
    # adaptively per shape signature, persisted via cache_results=True.
    grid = lambda META: (
        triton.cdiv(N * out_h * out_w, META['BLOCK_SIZE_BHW']),
        triton.cdiv(out_c // groups, META['BLOCK_SIZE_OUTF']),
        groups,
    )
    _set_device(x_c)
    _in_w_key, _out_w_key = _conv_width_key(in_h, out_h, in_w, out_w, kw, stride_w, pad_w, dil_w)
    _core._autotune_headroom_guard(conv2d_forward_kernel[grid])(
        x_c, w_c, output,
        N, in_c, in_h, in_w,
        out_c, out_h, out_w,
        _in_w_key, _out_w_key,
        *x_c.stride(), *w_c.stride(), *output.stride(),
        kernel_height=kh, kernel_width=kw,
        stride_height=stride_h, stride_width=stride_w,
        padding_height=pad_h, padding_width=pad_w,
        dilation_height=dil_h, dilation_width=dil_w,
        groups=groups, fp16=fp16, WIDE_OFFSETS=_conv_offsets_wide(x_c, w_c, output),
    )

    if bias is not None:
        output = _conv_bias_inplace(output, bias)

    return output


def _conv2d_m8n8k4(x, w, bias, out, N, in_c, in_h, in_w, out_c, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw,
                   groups):
    """conv2d as an implicit GEMM on the m8n8k4 matrix unit: M = N*OH*OW, N = OC/groups, K = (IC/groups)*KH*KW in
    the contiguous weight's own order (read in place); groups on grid axis 1; the profile's `mm` tile."""
    from nbxtensor.ops.conv2d_m8n8k4 import conv2d_m8n8k4_kernel, conv_layouts
    M, OG, CG = N * out_h * out_w, out_c // groups, in_c // groups
    tile = _matrix_unit_mm_tile(M, OG, _matrix_unit())
    MB, NB = tile["block_m"] // 16, tile["block_n"] // 16
    lay = conv_layouts(MB, NB, tile["warps_m"], tile["warps_n"], tile["block_k"])
    _set_device(x)
    grid = (triton.cdiv(M, tile["block_m"]) * triton.cdiv(OG, tile["block_n"]), groups)
    conv2d_m8n8k4_kernel[grid](
        x, w, out if bias is None else bias.contiguous(), out,
        M, OG, CG * kh * kw, in_h, in_w, out_h, out_w,
        *x.stride(), w.stride(0), out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        CG, OG,
        KH=kh, KW=kw, SH_=sh, SW_=sw, PH=ph, PW=pw, DH=dh, DW=dw, HAS_BIAS=bias is not None,
        MB=MB, NB=NB, BK=tile["block_k"], GROUP_M=tile["group_m"],
        num_warps=tile["warps_m"] * tile["warps_n"], **lay)


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
    _iw_key, _ow_key = _conv_width_key(IH, out_h, IW, out_w, kw, stride_w, pad_w, 1)
    _core._autotune_headroom_guard(depthwise_conv2d_kernel[grid])(
        x_c, w_c, output,
        N, C,
        IH, IW, out_h, out_w,
        _iw_key, _ow_key,
        *x_c.stride(), *w_c.stride()[:1], *w_c.stride()[2:],
        *output.stride(),
        kh=kh, kw=kw,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        fp16=fp16, WIDE_OFFSETS=_conv_offsets_wide(x_c, w_c, output),
    )

    if bias is not None:
        output = _conv_bias_inplace(output, bias)

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
    band frontiers, accepted as a follow-up halo-correctness workstream).
    """
    # The TilingEngine sizes each band (`conv2d.band_budget_divisor` of the threshold);
    # the headroom covers transient input slice + kernel intermediate.
    band_oh = _lk.conv2d_band_rows(N, out_c, out_h, out_w, out_dtype_bytes, _ts.conv2d_band_bytes())

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
        output = _conv_bias_inplace(output, bias)

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
# Avg Pool 2D
# ---------------------------------------------------------------------------

def avg_pool2d_wrapper(
    x, kernel_size, stride=None, padding=0,
    ceil_mode=False, count_include_pad=True, divisor_override=None,
) :
    """Average pooling 2D forward."""
    assert x.ndim == 4
    N, C, H, W = x.shape

    # ATen broadcast for single-element lists (same contract as
    # conv2d_wrapper — the latent IndexError class).
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh = kernel_size[0]
        kw = kernel_size[1] if len(kernel_size) > 1 else kh

    if stride is None or (isinstance(stride, (list, tuple)) and not stride):
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh = stride[0]
        sw = stride[1] if len(stride) > 1 else sh

    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph = padding[0] if len(padding) > 0 else 0
        pw = padding[1] if len(padding) > 1 else ph

    if ceil_mode:
        oh = (H + 2 * ph - kh + sh - 1) // sh + 1
        ow = (W + 2 * pw - kw + sw - 1) // sw + 1
    else:
        oh = (H + 2 * ph - kh) // sh + 1
        ow = (W + 2 * pw - kw) // sw + 1

    output = NBXTensor.empty((N, C, oh, ow), device=x.device, dtype=x.dtype)

    # Contiguous-guard (manifesto pattern): materialize ONCE and launch
    # with THAT tensor's strides. The previous form passed
    # x.contiguous() as data but x.stride(...) of the ORIGINAL — a
    # non-contiguous input (transpose→unsqueeze chains, e.g. the
    # MiniCPM-o audio pooler) made the kernel read wrong addresses
    # (141% relative error at the first diverging op of the audio
    # drift diff). Contiguous inputs short-circuit at zero cost.
    xc = x.contiguous()
    grid = (
        N * C,
        triton.cdiv(oh, _POOL_BH) * triton.cdiv(ow, _POOL_BW),
    )
    _set_device(xc)
    avg_pool2d_forward_kernel[grid](
        xc, output,
        xc.stride(0), xc.stride(1), xc.stride(2), xc.stride(3),
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

    # ATen broadcast for single-element lists (same contract as
    # conv2d_wrapper/avg_pool2d_wrapper).
    if isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]
    if isinstance(stride, (list, tuple)) and len(stride) == 1:
        stride = [stride[0], stride[0]]
    if isinstance(padding, (list, tuple)) and len(padding) == 1:
        padding = [padding[0], padding[0]]
    if isinstance(dilation, (list, tuple)) and len(dilation) == 1:
        dilation = [dilation[0], dilation[0]]

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

    # Contiguous-guard: same fix as avg_pool2d_wrapper — one
    # materialization, ITS strides.
    xc = x.contiguous()
    grid = (
        N * C,
        triton.cdiv(oh, _POOL_BH) * triton.cdiv(ow, _POOL_BW),
    )
    _set_device(xc)
    max_pool2d_forward_kernel[grid](
        xc, output, indices,
        xc.stride(0), xc.stride(1), xc.stride(2), xc.stride(3),
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


def adaptive_avg_pool2d_wrapper(x, output_size) :
    """Adaptive average pooling 2D."""
    N, C, IH, IW = x.shape
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH, OW = output_size
    output = NBXTensor.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = _ew._1d_grid(total)
    x_c = x.contiguous()
    _set_device(x_c)
    adaptive_avg_pool2d_kernel[grid](
        x_c, output, total,
        IH, IW, OH, OW,
        *x_c.stride(),
        **_ew_meta(total))
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
        triton.cdiv(OW, _core._BILINEAR_BX),
        triton.cdiv(OH, _core._BILINEAR_BY),
        N * C,
    )
    _set_device(x.contiguous())
    conv_depthwise2d_kernel[grid](
        x.contiguous(), weight.contiguous(), output,
        N, C, IH, IW, C, KH, KW, OH, OW,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        1,  # ch_mult
        BLOCK_X=_core._BILINEAR_BX, BLOCK_Y=_core._BILINEAR_BY,
        num_warps=4)
    if bias is not None:
        output = _conv_bias_inplace(output, bias)
    return output


def unfold_backward_wrapper(grad_in, input_sizes, dim, size, step) :
    """unfold_backward (overlap-add): scatter-add the unfolded frames
    [..lead.., N_frames, size] back into the original signal [..lead.., L] (L on
    `dim`), summing overlapping positions. This is the iSTFT overlap-add.
    R33-pure (one @triton.jit scatter-add kernel)."""
    from nbxtensor.ops.unfold_op import unfold_backward_1d_kernel
    grad_in = _core._ensure_cuda(grad_in).contiguous()
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
    from nbxtensor.ops.im2col import im2col_kernel

    def _pair(v):
        return tuple(v) if isinstance(v, (list, tuple)) else (v, v)

    x = _core._ensure_cuda(x).contiguous()
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


# ---------------------------------------------------------------------------
# The cost each convolution path declares (M3, docs/nbxtensor/COST.md): what it allocates, in its order, read from the
# same decisions the wrappers take (`launch_keys.conv_dtypes`, `conv2d_route`, `matrix_unit_native`, the strided read
# rule) — a planner prices the convolution by it instead of re-deriving it. Operands are `_cost.Spec`s; a returned
# buffer is the op's output.
# ---------------------------------------------------------------------------

def _reads_strided(shape, strides) -> bool:
    """`_conv_reads_strided` on a shape and its strides: four non-negative strides, the width innermost."""
    return len(strides) == 4 and all(int(v) >= 0 for v in strides) and (int(strides[3]) == 1 or int(shape[3]) == 1)


def _pair(v):
    return (v[0], v[1] if len(v) > 1 else v[0]) if isinstance(v, (list, tuple)) else (v, v)


def conv2d_cost(x, w, stride=1, padding=0, dilation=1, groups=1, bias=False, out_dtype=None, out_given=False,
                krsc=False, unit=None, depthwise_enabled=True):
    """`conv2d_wrapper` on an input `x` and a weight `w` (`_cost.Spec`s): the operands narrowed to the narrowest of the
    pair when they differ (a copy each), the input made contiguous unless the implicit GEMM reads it where it lies, a
    weight not stored KRSC made contiguous, then the route — depthwise (a contiguous input, its output), band (the whole
    output, each band's conv beside it), or the plain launch: its output (none when the caller hands `out_given`), and
    on the matrix unit the input made contiguous and, for a caller's view, a dense result copied into it. The bias is
    added in place. `out_dtype`: the component's compute dtype the wrapper writes (`_NBX_COMPUTE_DTYPE`), the input's
    when None. `unit`: the matrix unit to judge by (a planner's), the profile in force's when None."""
    from nbxtensor.ops import _cost as C
    from nbxtensor.ops import tiling_sizes as _tsz
    from nbxtensor.shape import contiguous_strides, is_contiguous
    st = C._Steps()
    N, in_c, in_h, in_w = (int(d) for d in x.shape)
    out_c, _cg, kh, kw = (int(d) for d in w.shape)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    out_h, out_w = _lk.conv_out_hw(in_h, in_w, kh, kw, sh, sw, ph, pw, dh, dw)
    x_to, w_to = _lk.conv_dtypes(NBXDtype[x.dtype], NBXDtype[w.dtype])
    if x_to.name != x.dtype:
        x = C.Spec.contiguous(x.shape, x_to.name)
        st.alloc("input narrowed", x.nbytes)
    if w_to.name != w.dtype:
        w = C.Spec.contiguous(w.shape, w_to.name)
        st.alloc("weight narrowed", w.nbytes)
    if not is_contiguous(x.shape, x.strides) and not _reads_strided(x.shape, x.strides):
        x = C.Spec.contiguous(x.shape, x.dtype)
        st.alloc("input made contiguous", x.nbytes)
    if not krsc and not is_contiguous(w.shape, w.strides):
        st.alloc("weight made contiguous", w.nbytes)
    out_dt = out_dtype or x.dtype
    out_bytes = N * out_c * out_h * out_w * dtype_size(NBXDtype[out_dt])
    route = _lk.conv2d_route(N, in_c, out_c, out_h, out_w, dh, dw, groups, NBXDtype[x.dtype],
                             _tsz.conv2d_band_bytes(), depthwise_enabled=depthwise_enabled)
    if route == "depthwise":
        if not is_contiguous(x.shape, x.strides):
            st.alloc("input made contiguous (depthwise)", x.nbytes)
        st.alloc("output", out_bytes, returned=True)
        return st.done(2 * N * out_c * out_h * out_w * kh * kw)
    if route == "band":
        if not is_contiguous(x.shape, x.strides):
            x = C.Spec.contiguous(x.shape, x.dtype)
            st.alloc("input made contiguous (bands)", x.nbytes)
        st.alloc("output", out_bytes, returned=True)
        band_oh = _lk.conv2d_band_rows(N, out_c, out_h, out_w, dtype_size(NBXDtype[x.dtype]), _tsz.conv2d_band_bytes())
        worst = None
        for oh_start in range(0, out_h, band_oh):
            oh_end = min(oh_start + band_oh, out_h)
            ih_start = max(0, oh_start * sh - ph)
            ih_end = min(in_h, (oh_end - 1) * sh + dh * (kh - 1) + 1 - ph)
            if ih_end <= ih_start:
                continue
            band = C.Spec((N, in_c, ih_end - ih_start, in_w), x.strides, x.dtype)
            c = conv2d_cost(band, w, stride, padding, dilation, groups, False, out_dt, False, True, unit,
                            depthwise_enabled)
            if worst is None or c.peak_bytes > worst.peak_bytes:
                worst = c
        if worst is not None:
            for i in st.nest(worst, "the largest band"):
                st.free(i)                       # each band's result copied into the output, then dropped
        return st.done(2 * N * out_c * out_h * out_w * kh * kw * (in_c // max(1, groups)))
    if not out_given:
        st.alloc("output", out_bytes, returned=True)
    if _lk.matrix_unit_native(NBXDtype[x.dtype], NBXDtype[w.dtype], N * out_h * out_w, unit):
        if not is_contiguous(x.shape, x.strides):
            i = st.alloc("input made contiguous (matrix unit)", x.nbytes)
        if out_given:
            st.alloc("dense result for the caller's view (matrix unit)", out_bytes)
    return st.done(2 * N * out_c * out_h * out_w * kh * kw * (in_c // max(1, groups)))


def conv3d_cost(x, w, stride=1, padding=0, dilation=1, groups=1, bias=False, out_dtype=None, unit=None):
    """`_conv3d_via_conv2d` on an input `x` [B, Cin, T, H, W] and a weight `w` [Cout, Cin_g, kt, kh, kw]: the temporal
    pad (a padded copy, alive to the return), then — at temporal stride 1 — each tap's 2-D weight slice, its tap buffer
    and its per-batch conv2d writing into the tap's view (`conv2d_cost(out_given=True)`: on the matrix unit the
    frames made contiguous and a dense result), the first tap kept as the output and each later one added into it and
    dropped; at a larger stride each tap's frames gathered and folded (a copy), its conv2d output, its result permuted
    back (a copy) and the running sum (a new buffer per tap). The bias adds a new output. `out_dtype`: the compute
    dtype the wrapper writes, the input's when None."""
    from nbxtensor.ops import _cost as C
    from nbxtensor.shape import contiguous_strides

    def _triple(v):
        return (v[0], v[1], v[2]) if isinstance(v, (list, tuple)) else (v, v, v)
    st_, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)
    B, Cin, T, H, W = (int(d) for d in x.shape)
    Cout, Cin_g, kt, kh, kw = (int(d) for d in w.shape)
    out_dt = out_dtype or x.dtype
    st = C._Steps()
    if pt > 0:
        x = C.Spec.contiguous((B, Cin, T + 2 * pt, H, W), x.dtype)
        st.alloc("temporal pad", x.nbytes)
    Tp = x.shape[2]
    T_out = (Tp - dt * (kt - 1) - 1) // st_ + 1
    Ho, Wo = _lk.conv_out_hw(H, W, kh, kw, sh, sw, ph, pw, dh, dw)
    out_bytes = B * Cout * T_out * Ho * Wo * dtype_size(NBXDtype[out_dt])
    w2 = C.Spec.contiguous((Cout, Cin_g, kh, kw), w.dtype)
    flops = 2 * B * Cout * T_out * Ho * Wo * kt * kh * kw * Cin_g
    # the wrapper's locals, as Python holds them: a name rebound frees its old buffer only after the new one exists, a
    # buffer at least `_NBX_CONV3D_EAGER_FREE_BYTES` is dropped as soon as it is dead (`x2 = None`, ...), the rest die
    # at the return
    held = {}

    def bind(name, index):
        old = held.get(name)
        held[name] = index
        if old is not None and old not in held.values():
            st.free(old)

    def drop(name):
        old = held.pop(name, None)
        if old is not None and old not in held.values():
            st.free(old)

    if st_ == 1:
        xs = x.strides
        frames = C.Spec((T_out, Cin, H, W), (xs[2], xs[1], xs[3], xs[4]), x.dtype)
        for kti in range(kt):
            bind("w2", st.alloc("tap weight slice", w2.nbytes))
            bind("tap", st.alloc("tap", out_bytes))
            for _b in range(B):
                for i in st.nest(conv2d_cost(frames, w2, (sh, sw), (ph, pw), (dh, dw), groups, False, out_dt,
                                             True, False, unit), f"tap {kti} conv2d"):
                    st.free(i)
            if "out" not in held:
                held["out"] = held["tap"]       # the first tap is the sum
            drop("tap")                         # a later one, added into the sum in place, dropped (`tap = None`)
    else:
        fold_bytes = B * T_out * Cin * H * W * dtype_size(NBXDtype[x.dtype])
        for kti in range(kt):
            bind("idx", st.alloc("frame indices", T_out * 8))
            bind("xs", st.alloc("tap frames gathered", fold_bytes))
            bind("x2", st.alloc("tap frames folded", fold_bytes))
            bind("w2", st.alloc("tap weight slice", w2.nbytes))
            fold = C.Spec.contiguous((B * T_out, Cin, H, W), x.dtype)
            y2, = st.nest(conv2d_cost(fold, w2, (sh, sw), (ph, pw), (dh, dw), groups, False, out_dt, False, False,
                                      unit), f"tap {kti} conv2d")
            bind("y2", y2)
            if fold_bytes >= _NBX_CONV3D_EAGER_FREE_BYTES:
                drop("x2")
            bind("y", st.alloc("tap result permuted back", out_bytes))
            if out_bytes >= _NBX_CONV3D_EAGER_FREE_BYTES:
                drop("y2")
            if "out" not in held:
                held["out"] = held["y"]
            else:
                bind("out", st.alloc("running sum", out_bytes))
                if out_bytes >= _NBX_CONV3D_EAGER_FREE_BYTES:
                    drop("y")
    if bias:
        bind("out", st.alloc("output with the bias", out_bytes))
    out = held.pop("out")
    st.buffers[out][4] = True                   # the result
    st.buffers[out][3] = None
    return st.done(flops)                       # every other local dies at the return
