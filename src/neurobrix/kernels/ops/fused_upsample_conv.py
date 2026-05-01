"""Fused upsample_nearest2d + convolution — band-streaming wrapper.

Eliminates the intermediate upsampled tensor that would otherwise sit in
VRAM between an upsample op and the convolution that consumes it. For
Sana 4Kpx VAE: the upsample produces [1, 256, 4096, 4096] fp32 = 16 GB
that the conv consumes immediately. Materializing it OOMs on V100 32 GB
even though the conv output is "only" 8 GB.

Algorithm (band-by-band along output H):
    1. Allocate the FINAL conv output (band-by-band would need a
       streaming consumer downstream — for now the conv output is
       materialized once because subsequent ops in the DAG consume it
       as a whole tensor).
    2. For each output band of size band_h:
       a. Compute the pre-upsample input band needed (with halo for
          the conv kernel: kh-1 extra rows on each side).
       b. Slice that band from pre_upsample_input.
       c. Upsample only that band (small allocation, freed immediately).
       d. Apply the conv on the upsampled band.
       e. Write the band into the conv output at the correct offset.

The intermediate upsampled tensor for one band is band_h/tile_factor of
the full 16 GB → e.g. tile_factor=4 → 4 GB transient (acceptable on V100).

Mode-agnostic: detects torch.Tensor vs NBXTensor and routes to the
appropriate backend. Used by the op-level tiling engine (tiling_engine.py)
which wires this as an op_uid interceptor on aten.convolution::N when
the preceding aten.upsample_nearest2d::M was tagged for fusion by Prism.
"""

from typing import Any, Optional


class FusionUpsampleProxy:
    """Sentinel returned by an intercepted upsample op so the next conv
    can find the pre-upsample input + scales without the upsample tensor
    ever being materialized.

    Holds: pre_upsample input tensor, scales_h, scales_w, intended output
    shape. Stored in the CompiledSequence arena slot that would normally
    hold the upsample output tensor. Only the paired conv interceptor reads
    this slot — verified at Prism planning time (single consumer).
    """

    __slots__ = ("pre_input", "scales_h", "scales_w", "output_shape", "dtype", "device")

    def __init__(self, pre_input, scales_h, scales_w, output_shape):
        self.pre_input = pre_input
        self.scales_h = float(scales_h)
        self.scales_w = float(scales_w)
        self.output_shape = tuple(output_shape)
        self.dtype = getattr(pre_input, "dtype", None)
        self.device = getattr(pre_input, "device", None)

    @property
    def shape(self):
        return self.output_shape

    @property
    def ndim(self):
        return len(self.output_shape)

    def __repr__(self):
        return (
            f"FusionUpsampleProxy(pre={tuple(self.pre_input.shape)}, "
            f"scales=({self.scales_h},{self.scales_w}), "
            f"out_shape={self.output_shape})"
        )


def make_upsample_proxy_interceptor():
    """Return an interceptor that turns an upsample into a FusionUpsampleProxy
    (no compute, no allocation of the full upsampled tensor)."""

    def _proxy(input_tensor, output_size=None, scales_h=None, scales_w=None,
               *args, **kwargs):
        # Resolve scales: prefer explicit scales args (Sana 4Kpx graph stores
        # both output_size and scales). Fall back to ratio if scales missing.
        ih = input_tensor.shape[2]
        iw = input_tensor.shape[3]
        if scales_h is None or scales_w is None:
            if isinstance(output_size, (list, tuple)):
                oh, ow = int(output_size[0]), int(output_size[1])
            else:
                oh = ow = int(output_size)
            sh = oh / ih
            sw = ow / iw
        else:
            sh = float(scales_h)
            sw = float(scales_w)
            oh = int(ih * sh)
            ow = int(iw * sw)
        n, c = input_tensor.shape[0], input_tensor.shape[1]
        return FusionUpsampleProxy(input_tensor, sh, sw, (n, c, oh, ow))

    return _proxy


def fused_upsample_conv2d(
    proxy_or_tensor: Any,
    weight: Any,
    bias: Optional[Any] = None,
    stride: Any = 1,
    padding: Any = 1,
    dilation: Any = 1,
    transposed: bool = False,
    output_padding: Any = 0,
    groups: int = 1,
    tile_factor: int = 4,
):
    """Conv that absorbs a preceding nearest upsample when the input is a
    FusionUpsampleProxy. Falls back to a standard conv if the input is a
    materialized tensor (no upsample to fuse with — interceptor was a no-op
    fusion target).

    Streams the (upsample, conv) pair band-by-band along output H, never
    materializing the full upsampled tensor. The conv output IS materialized
    in full because downstream consumers in the DAG read it as a whole.
    """
    # Detect backend by tensor type
    import torch
    use_nbx = False
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
        ref = proxy_or_tensor.pre_input if isinstance(proxy_or_tensor, FusionUpsampleProxy) else proxy_or_tensor
        use_nbx = isinstance(ref, NBXTensor)
    except ImportError:
        pass

    if use_nbx:
        return _fused_upsample_conv2d_nbx(
            proxy_or_tensor, weight, bias, stride, padding, dilation,
            transposed, output_padding, groups, tile_factor,
        )
    return _fused_upsample_conv2d_torch(
        proxy_or_tensor, weight, bias, stride, padding, dilation,
        transposed, output_padding, groups, tile_factor,
    )


# ---------------------------------------------------------------------------
# PyTorch (compiled / sequential modes) — uses F.interpolate + F.conv2d
# ---------------------------------------------------------------------------


def _fused_upsample_conv2d_torch(
    proxy_or_tensor, weight, bias, stride, padding, dilation,
    transposed, output_padding, groups, tile_factor,
):
    import torch
    import torch.nn.functional as F

    # Normalize stride / padding / dilation to (h, w) tuples
    sh_st, sw_st = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dh, dw = _pair(dilation)

    if isinstance(proxy_or_tensor, FusionUpsampleProxy):
        proxy = proxy_or_tensor
        pre_input = proxy.pre_input
        up_sh, up_sw = proxy.scales_h, proxy.scales_w
        N, _, OH, OW = proxy.output_shape  # output of upsample
        in_c = pre_input.shape[1]
    else:
        # Plain conv path — fallback when the upsample interceptor wasn't
        # active (e.g. compile-time wiring missed). Run a standard conv
        # without tiling (caller will OOM if input is too big — same
        # behaviour as before this interceptor existed).
        return F.conv2d(
            proxy_or_tensor, weight, bias=bias,
            stride=(sh_st, sw_st), padding=(pad_h, pad_w),
            dilation=(dh, dw), groups=groups,
        )

    out_c, _, kh, kw = weight.shape
    # Output spatial of the conv (after the absorbed upsample)
    conv_out_h = (OH + 2 * pad_h - dh * (kh - 1) - 1) // sh_st + 1
    conv_out_w = (OW + 2 * pad_w - dw * (kw - 1) - 1) // sw_st + 1

    # Allocate full conv output — downstream consumers expect a whole tensor.
    out_dtype = weight.dtype
    output = torch.empty(
        (N, out_c, conv_out_h, conv_out_w),
        dtype=out_dtype, device=pre_input.device,
    )

    # Tile along output H. tile_factor is a hint from Prism; clamp to >= 1.
    tile_factor = max(1, int(tile_factor))
    band_oh = (conv_out_h + tile_factor - 1) // tile_factor
    if band_oh < 1:
        band_oh = conv_out_h

    pre_h = pre_input.shape[2]

    # Halo: each output row reads (kh) input rows post-upsample.
    # Pre-upsample row index for upsample-output row r is floor(r / up_sh).
    halo_pre = max(1, int((kh - 1) // 2 + 1))

    for oh_start in range(0, conv_out_h, band_oh):
        oh_end = min(oh_start + band_oh, conv_out_h)

        # Upsample-output rows that this conv band reads (with conv padding
        # reaching into the previous band by kh//2):
        up_start = max(0, oh_start * sh_st - pad_h)
        up_end = min(OH, (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h)
        if up_end <= up_start:
            continue

        # Pre-upsample rows that produce up_start..up_end (nearest mapping):
        # ih = floor(uh / up_sh); we add halo_pre on each side for safety.
        pre_start = max(0, int(up_start // up_sh) - halo_pre)
        pre_end = min(pre_h, int((up_end - 1) // up_sh) + 1 + halo_pre)

        pre_band = pre_input[:, :, pre_start:pre_end, :]
        # Cast pre_band to weight dtype if needed (mixed precision safety)
        if pre_band.dtype != out_dtype:
            pre_band = pre_band.to(out_dtype)
        pre_band = pre_band.contiguous()

        # Upsample the band only (small allocation freed at end of iteration)
        # Use the SAME nearest mapping as the original op: explicit scale
        # factor to avoid output_size rounding mismatch.
        up_band = F.interpolate(
            pre_band, scale_factor=(up_sh, up_sw), mode="nearest"
        )

        # The up_band's first row corresponds to upsample-output row
        # pre_start * up_sh. Slice to get exactly up_start..up_end.
        up_offset_local = up_start - int(pre_start * up_sh)
        up_band_local_h = up_end - up_start
        up_band = up_band[:, :, up_offset_local:up_offset_local + up_band_local_h, :]
        up_band = up_band.contiguous()

        # Conv with adjusted padding: zero on internal band edges (the
        # adjacent band covers them), original padding on outer edges.
        pad_top = pad_h if oh_start == 0 else 0
        pad_bot = pad_h if oh_end == conv_out_h else 0
        if pad_top != pad_bot:
            # Asymmetric pad — apply manually then conv with padding 0 on H.
            up_band = F.pad(up_band, (0, 0, pad_top, pad_bot), mode="constant", value=0.0)
            band_pad_h = 0
        else:
            band_pad_h = pad_top

        conv_band = F.conv2d(
            up_band, weight, bias=None,
            stride=(sh_st, sw_st),
            padding=(band_pad_h, pad_w),
            dilation=(dh, dw),
            groups=groups,
        )

        # The conv_band first output row corresponds to up_start (after
        # accounting for the symmetric padding that was applied). Slice the
        # exact (oh_end - oh_start) rows we want.
        actual_band_h = oh_end - oh_start
        # When padding was symmetric (band_pad_h > 0), the conv preserves H
        # for stride=1: conv_band.shape[2] == up_band_local_h.
        # When asymmetric (manual pad), same property holds.
        band_first_oh = up_start // sh_st
        local_offset = oh_start - band_first_oh
        if local_offset < 0:
            local_offset = 0
        conv_band = conv_band[:, :, local_offset:local_offset + actual_band_h, :]
        if conv_band.shape[2] != actual_band_h:
            # Defensive: clip / pad to match exactly. Should not happen for
            # stride=1 conv with standard padding but covers edge rounding.
            actual_band_h = min(conv_band.shape[2], actual_band_h)
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, :actual_band_h, :]

    if bias is not None:
        # Bias add as separate broadcast — in-place to save one alloc.
        output += bias.view(1, -1, 1, 1)

    return output


def _pair(v):
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ---------------------------------------------------------------------------
# Standalone tiled conv (no upsample to fuse) — for conv ops whose input is
# already materialized but whose cuDNN workspace would OOM. Tiles output H
# into bands; each band materializes only its input slice + cuDNN workspace
# at band size, while the full output stays allocated for downstream use.
# ---------------------------------------------------------------------------


def tiled_conv2d_spatial(
    input_tensor,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed: bool = False,
    output_padding=0,
    groups: int = 1,
    tile_factor: int = 4,
    *args,
    **kwargs,
):
    """Conv2d that tiles the output along H to bound cuDNN workspace.

    Input is a real tensor (no FusionUpsampleProxy here — that's
    fused_upsample_conv2d's job). Each band's transient memory:
        input_band  = (band_h × stride + halo) × W × C × dtype_bytes / tile_factor
        cudnn_ws    ~ workspace_full / tile_factor   (linear in band_h)
        conv_band   = output_bytes / tile_factor

    Mode-agnostic: torch.Tensor → F.conv2d, NBXTensor → conv2d_wrapper.
    """
    sh_st, sw_st = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dh, dw = _pair(dilation)

    # Detect backend
    use_nbx = False
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
        use_nbx = isinstance(input_tensor, NBXTensor)
    except ImportError:
        pass

    if use_nbx:
        return _tiled_conv2d_spatial_nbx(
            input_tensor, weight, bias, sh_st, sw_st, pad_h, pad_w,
            dh, dw, groups, tile_factor,
        )
    return _tiled_conv2d_spatial_torch(
        input_tensor, weight, bias, sh_st, sw_st, pad_h, pad_w,
        dh, dw, groups, tile_factor,
    )


def _tiled_conv2d_spatial_torch(
    input_tensor, weight, bias, sh_st, sw_st, pad_h, pad_w,
    dh, dw, groups, tile_factor,
):
    import torch
    import torch.nn.functional as F

    N, in_c, IH, IW = input_tensor.shape
    out_c, _, kh, kw = weight.shape
    out_h = (IH + 2 * pad_h - dh * (kh - 1) - 1) // sh_st + 1
    out_w = (IW + 2 * pad_w - dw * (kw - 1) - 1) // sw_st + 1

    out_dtype = weight.dtype
    output = torch.empty(
        (N, out_c, out_h, out_w), dtype=out_dtype, device=input_tensor.device
    )

    tile_factor = max(1, int(tile_factor))
    band_oh = (out_h + tile_factor - 1) // tile_factor
    if band_oh < 1:
        band_oh = out_h

    for oh_start in range(0, out_h, band_oh):
        oh_end = min(oh_start + band_oh, out_h)
        # Input rows that produce this output band
        ih_start = max(0, oh_start * sh_st - pad_h)
        ih_end = min(IH, (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h)
        if ih_end <= ih_start:
            continue
        in_band = input_tensor[:, :, ih_start:ih_end, :]
        if in_band.dtype != out_dtype:
            in_band = in_band.to(out_dtype)
        in_band = in_band.contiguous()

        pad_top = pad_h if oh_start == 0 else 0
        pad_bot = pad_h if oh_end == out_h else 0
        if pad_top != pad_bot:
            in_band = F.pad(in_band, (0, 0, pad_top, pad_bot), mode="constant", value=0.0)
            band_pad_h = 0
        else:
            band_pad_h = pad_top

        conv_band = F.conv2d(
            in_band, weight, bias=None,
            stride=(sh_st, sw_st),
            padding=(band_pad_h, pad_w),
            dilation=(dh, dw),
            groups=groups,
        )
        actual_band_h = oh_end - oh_start
        # First output row of conv_band corresponds to ih_start // sh_st
        band_first_oh = ih_start // sh_st
        local_offset = max(0, oh_start - band_first_oh)
        conv_band = conv_band[:, :, local_offset:local_offset + actual_band_h, :]
        actual_band_h = min(conv_band.shape[2], actual_band_h)
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, :actual_band_h, :]

    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output


def tiled_rms_norm_spatial(
    input_tensor,
    weight=None,
    eps: float = 1e-6,
    tile_factor: int = 4,
    *args,
    **kwargs,
):
    """Tile RMS norm along H bands.

    RMS norm computes y = x / sqrt(mean(x^2, dim=last) + eps) * weight.
    Normalization axis is the LAST dim (channels), so each spatial row is
    independent — perfectly tilable along H without any halo or
    cross-band stats. Cuts the transient memory by tile_factor.

    Used by Sana 4Kpx VAE where 4 successive rms_norm ops produce
    [1, H, W, C] fp32 8 GB tensors that accumulate live until the next
    block consumes them.
    """
    import torch

    # Resolve the canonical signature. Accept legacy positional or keyword.
    # Standard SDPA-style sig: (x, weight, eps).
    if not isinstance(input_tensor, torch.Tensor):
        # NBXTensor path — defer to the standard wrapper, no tiling
        # (single rms_norm rarely overflows alone in triton mode where
        # kernels stream).
        from neurobrix.kernels.wrappers import rms_norm_wrapper  # type: ignore
        return rms_norm_wrapper(input_tensor, weight, eps)

    x = input_tensor
    # Channels is the LAST dim
    H = x.shape[-3] if x.ndim >= 3 else 1
    if H < 2 or tile_factor <= 1:
        # Fall back to direct compute (no tiling worth it)
        return _rms_norm_direct(x, weight, eps)

    tile_factor = min(int(tile_factor), H)
    band_h = (H + tile_factor - 1) // tile_factor

    output = torch.empty_like(x)
    for h_start in range(0, H, band_h):
        h_end = min(h_start + band_h, H)
        # Slice along the H dim (axis -3 for [B, H, W, C])
        slicer = [slice(None)] * x.ndim
        slicer[-3] = slice(h_start, h_end)
        s = tuple(slicer)
        x_band = x[s]
        out_band = _rms_norm_direct(x_band, weight, eps)
        output[s] = out_band
    return output


def _rms_norm_direct(x, weight, eps):
    """Reference rms_norm — used by each band."""
    import torch
    # mean of squares along last dim
    var = (x.float() ** 2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps).to(x.dtype)
    if weight is not None:
        y = y * weight
    return y


def _tiled_conv2d_spatial_nbx(
    input_tensor, weight, bias, sh_st, sw_st, pad_h, pad_w,
    dh, dw, groups, tile_factor,
):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import conv2d_wrapper, add as nbx_add

    N, in_c, IH, IW = input_tensor.shape
    out_c, _, kh, kw = weight.shape
    out_h = (IH + 2 * pad_h - dh * (kh - 1) - 1) // sh_st + 1
    out_w = (IW + 2 * pad_w - dw * (kw - 1) - 1) // sw_st + 1

    output = NBXTensor.empty(
        (N, out_c, out_h, out_w),
        device=input_tensor.device, dtype=input_tensor.dtype,
    )
    tile_factor = max(1, int(tile_factor))
    band_oh = (out_h + tile_factor - 1) // tile_factor

    for oh_start in range(0, out_h, band_oh):
        oh_end = min(oh_start + band_oh, out_h)
        ih_start = max(0, oh_start * sh_st - pad_h)
        ih_end = min(IH, (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h)
        if ih_end <= ih_start:
            continue
        in_band = input_tensor[:, :, ih_start:ih_end, :]
        band_pad_h = pad_h
        conv_band = conv2d_wrapper(
            in_band, weight, None,
            stride=(sh_st, sw_st), padding=(band_pad_h, pad_w),
            dilation=(dh, dw), groups=groups,
        )
        actual_band_h = oh_end - oh_start
        band_first_oh = ih_start // sh_st
        local_offset = max(0, oh_start - band_first_oh)
        conv_band = conv_band[:, :, local_offset:local_offset + actual_band_h, :]
        actual_band_h = min(conv_band.shape[2], actual_band_h)
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, :actual_band_h, :]
    if bias is not None:
        output = nbx_add(output, bias.view(1, -1, 1, 1))
    return output


# ---------------------------------------------------------------------------
# NBX / Triton mode — same algorithm via Triton wrappers
# ---------------------------------------------------------------------------


def _fused_upsample_conv2d_nbx(
    proxy_or_tensor, weight, bias, stride, padding, dilation,
    transposed, output_padding, groups, tile_factor,
):
    """NBXTensor flavor — uses the existing Triton upsample + conv2d
    wrappers per band. Identical algorithm to the torch path; the only
    differences are the backend ops and explicit NBXTensor allocations.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import (
        upsample_nearest2d_wrapper, conv2d_wrapper, add as nbx_add,
    )

    sh_st, sw_st = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dh, dw = _pair(dilation)

    if not isinstance(proxy_or_tensor, FusionUpsampleProxy):
        return conv2d_wrapper(
            proxy_or_tensor, weight, bias,
            stride=(sh_st, sw_st), padding=(pad_h, pad_w),
            dilation=(dh, dw), groups=groups,
        )

    proxy = proxy_or_tensor
    pre_input = proxy.pre_input
    up_sh, up_sw = proxy.scales_h, proxy.scales_w
    N, _, OH, OW = proxy.output_shape

    out_c, _, kh, kw = weight.shape
    conv_out_h = (OH + 2 * pad_h - dh * (kh - 1) - 1) // sh_st + 1
    conv_out_w = (OW + 2 * pad_w - dw * (kw - 1) - 1) // sw_st + 1

    output = NBXTensor.empty(
        (N, out_c, conv_out_h, conv_out_w),
        device=pre_input.device, dtype=pre_input.dtype,
    )

    tile_factor = max(1, int(tile_factor))
    band_oh = (conv_out_h + tile_factor - 1) // tile_factor
    pre_h = pre_input.shape[2]
    halo_pre = max(1, (kh - 1) // 2 + 1)

    for oh_start in range(0, conv_out_h, band_oh):
        oh_end = min(oh_start + band_oh, conv_out_h)
        up_start = max(0, oh_start * sh_st - pad_h)
        up_end = min(OH, (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h)
        if up_end <= up_start:
            continue
        pre_start = max(0, int(up_start // up_sh) - halo_pre)
        pre_end = min(pre_h, int((up_end - 1) // up_sh) + 1 + halo_pre)

        pre_band = pre_input[:, :, pre_start:pre_end, :]
        # Upsample band via the existing Triton wrapper
        up_band = upsample_nearest2d_wrapper(
            pre_band,
            output_size=[int(pre_band.shape[2] * up_sh), int(pre_band.shape[3] * up_sw)],
            scales_h=up_sh, scales_w=up_sw,
        )
        up_offset_local = up_start - int(pre_start * up_sh)
        up_band_local_h = up_end - up_start
        up_band = up_band[:, :, up_offset_local:up_offset_local + up_band_local_h, :]

        # Conv via the existing Triton wrapper. NBX wrapper does NOT support
        # asymmetric padding; we handle external boundaries by full padding
        # (the existing wrapper's padding semantic) and rely on the slice
        # below to extract the correct rows.
        band_pad_h = pad_h if (oh_start == 0 or oh_end == conv_out_h) else pad_h
        conv_band = conv2d_wrapper(
            up_band, weight, None,
            stride=(sh_st, sw_st), padding=(band_pad_h, pad_w),
            dilation=(dh, dw), groups=groups,
        )

        actual_band_h = oh_end - oh_start
        band_first_oh = up_start // sh_st
        local_offset = max(0, oh_start - band_first_oh)
        conv_band = conv_band[:, :, local_offset:local_offset + actual_band_h, :]
        actual_band_h = min(conv_band.shape[2], actual_band_h)
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, :actual_band_h, :]

    if bias is not None:
        output = nbx_add(output, bias.view(1, -1, 1, 1))
    return output
