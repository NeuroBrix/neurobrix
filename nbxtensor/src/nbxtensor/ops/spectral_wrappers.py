"""Spectral wrappers: FFTs (r2c, c2r, c2c), DFT matrices, STFT and its inverse. Cut out of wrappers.py by family (M2, 2026-10-29), unchanged; shared launch layer and run context stay in wrappers.py."""

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
from nbxtensor.ops import conv_wrappers as _cnv
from nbxtensor.ops import elementwise_wrappers as _ew
from nbxtensor.ops import linalg_wrappers as _la
from nbxtensor.ops import wrappers as _core


def _triton_fft_rows(x_real, x_imag, inverse: bool) -> tuple:
    """The radix-2 FFT of every row along the last dim, all rows in ONE launch per step:
    one bit-reversal, log2(N) butterfly stages and, for the inverse, one 1/N scale.
    Returns (real, imag) in the input's shape. N must be a power of 2.

    The rows were launched one at a time from the host — an STFT of a spoken sentence
    is thousands of frames, so thousands of launches per stage: chatterbox's census
    walk spent every py-spy sample inside that loop (2026-09-28), and every live
    Triton run through `aten::stft` paid the same per-frame cost."""
    from nbxtensor.ops.fft_op import bit_reverse_rows_kernel, fft_stage_rows_kernel, scale_kernel

    N = x_real.shape[-1]
    assert N > 0 and (N & (N - 1)) == 0, f"FFT size must be power of 2, got {N}"
    orig_shape = x_real.shape
    rows = x_real.numel() // N
    xr = x_real.reshape(rows, N).contiguous()
    xi = x_imag.reshape(rows, N).contiguous()
    out_real = NBXTensor.empty_like(xr)
    out_imag = NBXTensor.empty_like(xi)
    log2n = N.bit_length() - 1
    total = rows * N
    set_device(xr)
    bit_reverse_rows_kernel[_ew._1d_grid(total)](xr, xi, out_real, out_imag, N, total, log2n,
                                             **_ew_meta(total))
    pairs = rows * (N // 2)
    for stage in range(1, log2n + 1):
        fft_stage_rows_kernel[_ew._1d_grid(pairs)](out_real, out_imag, N, pairs, stage,
                                               INVERSE=inverse, **_ew_meta(pairs))
    if inverse:
        # 1/N through the house scale kernel (an NBXTensor has no torch-style in-place
        # multiply — found by the 2026-09-05 bit gate).
        scale_kernel[_ew._1d_grid(total)](out_real, out_imag, total, 1.0 / N, **_ew_meta(total))
    return out_real.reshape(orig_shape), out_imag.reshape(orig_shape)


def _triton_fft_forward(x_real, x_imag) -> tuple:
    """Forward FFT of every row along the last dim. Returns (real, imag). N a power of 2."""
    return _triton_fft_rows(x_real, x_imag, inverse=False)


def _triton_ifft(x_real, x_imag) -> tuple:
    """Inverse FFT of every row along the last dim, scaled by 1/N. Returns (real, imag)."""
    return _triton_fft_rows(x_real, x_imag, inverse=True)


def _dft_rfft_matrices(N, N_bins, ref):
    """cos[k,n] and -sin[k,n] rfft basis matrices [N_bins, N] (R33-pure kernel).
    Allocated on `ref`'s device."""
    import math
    from nbxtensor.ops.dft import dft_rfft_matrix_kernel
    set_device(ref)
    cos_mat = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    nsin_mat = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    total = N_bins * N
    dft_rfft_matrix_kernel[_ew._1d_grid(total)](
        cos_mat, nsin_mat, N_bins, N, 2.0 * math.pi / N, **_ew_meta(total))
    return cos_mat, nsin_mat


def _idft_c2r_matrices(N, N_bins, ref):
    """irfft basis matrices [N_bins, N]: real_out = X_real @ C + X_imag @ S."""
    import math
    from nbxtensor.ops.dft import idft_c2r_matrix_kernel
    set_device(ref)
    C = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    S = NBXTensor.empty((N_bins, N), dtype=NBXDtype.float32, device=ref.device)
    total = N_bins * N
    idft_c2r_matrix_kernel[_ew._1d_grid(total)](
        C, S, N_bins, N, 2.0 * math.pi / N, 1.0 / N,
        1 if (N % 2 == 0) else 0, **_ew_meta(total))
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
    Xr = _la.matmul_wrapper(x2, cos_mat.transpose(0, 1).contiguous())   # [M, N_bins]
    Xi = _la.matmul_wrapper(x2, nsin_mat.transpose(0, 1).contiguous())  # [M, N_bins]
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
    real_out = _ew.add(_la.matmul_wrapper(Xr2, C), _la.matmul_wrapper(Xi2, S))   # [M, N]
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
    x = _core._ensure_cuda(x).contiguous()
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
    result = _ew.complex_wrapper(out_real, out_imag)
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
    x = _core._ensure_cuda(x)
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
    x = _core._ensure_cuda(x)

    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()

    N = x.shape[-1]
    padded_N = _ew._next_power_of_2(N)

    x_real = x.real.float().contiguous()
    x_imag = x.imag.float().contiguous()

    if padded_N != N:
        x_real = x_real
        x_imag = x_imag

    if forward:
        out_real, out_imag = _triton_fft_forward(x_real, x_imag)
    else:
        out_real, out_imag = _triton_ifft(x_real, x_imag)

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

    x = _core._ensure_cuda(x).contiguous()
    squeezed = (x.ndim == 1)
    if squeezed:
        x = x.unsqueeze(0)                                 # [1, signal]

    # frame: [..., num_frames, win_length]
    frames = x.unfold(-1, win_length, hop_length).contiguous()
    if window is not None:
        frames = _ew.mul(frames, window)                       # broadcast [win_length]
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

    x = _core._ensure_cuda(x)
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
        frames = _ew.mul(frames, window)                       # synthesis window

    sig_len = (n_frames - 1) * hop_length + win_length
    # overlap-add the windowed frames back into the signal
    signal = _cnv.unfold_backward_wrapper(frames, [batch, sig_len], 1, win_length, hop_length)
    # window-energy envelope (sum of squared windows at each position) — torch.istft
    # divides by this so overlapping windows reconstruct unit gain.
    if window is not None:
        win_sq = _ew.mul(window, window)
    else:
        win_sq = NBXTensor.ones((win_length,), dtype=frames.nbx_dtype, device=frames.device)
    win_sq_b = win_sq.reshape([1, 1, win_length])
    win_sq_frames = _ew.mul(NBXTensor.ones((batch, n_frames, win_length),
                                       dtype=frames.nbx_dtype, device=frames.device),
                        win_sq_b)
    win_env = _cnv.unfold_backward_wrapper(win_sq_frames, [batch, sig_len], 1, win_length, hop_length)
    signal = _ew.div(signal, _ew.clamp(win_env, 1e-11, None))

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
