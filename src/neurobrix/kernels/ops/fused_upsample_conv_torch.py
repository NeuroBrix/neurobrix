"""Fused upsample_nearest2d + convolution and the spatially tiled convolution — the PyTorch branch's versions, and the
dispatchers both engines call, which pick a branch by tensor type (NBXTensor -> nbxtensor.ops.fused_upsample_conv).

Mode-agnostic: used by the op-level tiling engine (core/module/tiling_engine.py), which wires these as op_uid
interceptors on aten.convolution::N when the preceding aten.upsample_nearest2d::M was tagged for fusion by Prism.
Split from kernels/ops/fused_upsample_conv.py when that file entered nbxtensor (M1, 2026-10-29), unchanged.
"""

from typing import Any, Optional

from nbxtensor.ops.fused_upsample_conv import (FusionUpsampleProxy, _fused_upsample_conv2d_nbx, _pair,
                                               _tiled_conv2d_spatial_nbx)



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
    # Detect backend by tensor type. Testing for NBXTensor rather than for
    # torch.Tensor is what keeps torch out of this dispatcher entirely — the
    # `import torch` that used to sit here was already unused.
    use_nbx = False
    try:
        from nbxtensor.tensor import NBXTensor
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
    """Band-streaming fused upsample+conv with REAL halo (no zero-pad seams).

    Per band, we build the upsample-output slice that covers
    [up_start - halo, up_end + halo) where halo = (kh-1)*dh // 2. The conv
    runs with padding=(0, pad_w) — the halo provides the rows the kernel
    needs internally, so no zero-pad is inserted at internal band frontiers.
    Image borders (band 0 top, last band bottom) get F.pad with the original
    pad_h. Bit-identical to a non-tiled conv when the halo is correct.
    """
    import torch
    import torch.nn.functional as F

    sh_st, sw_st = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dh, dw = _pair(dilation)

    if not isinstance(proxy_or_tensor, FusionUpsampleProxy):
        return F.conv2d(
            proxy_or_tensor, weight, bias=bias,
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

    out_dtype = weight.dtype
    output = torch.empty(
        (N, out_c, conv_out_h, conv_out_w),
        dtype=out_dtype, device=pre_input.device,
    )

    tile_factor = max(1, int(tile_factor))
    band_oh = (conv_out_h + tile_factor - 1) // tile_factor
    pre_h = pre_input.shape[2]

    # Halo on the upsample-output side: kernel needs (kh-1)*dh // 2 rows
    # of REAL pixels above and below each band. Pre-upsample halo derived
    # by dividing through the nearest-upsample ratio (one pre row covers
    # `up_sh` upsample rows, so halo_post / up_sh pre rows is enough).
    # Rounded up to whole strides; the rows skipped are halo // stride (2026-09-29, the stride-2
    # misalignment — the NBX twin reads the same cut from launch_keys.tiled_conv2d_bands).
    halo_post = -(-((kh - 1) * dh // 2) // sh_st) * sh_st

    for oh_start in range(0, conv_out_h, band_oh):
        oh_end = min(oh_start + band_oh, conv_out_h)

        # Upsample-output rows the conv band reads: stride / dilation aware.
        # First row read = oh_start*sh - pad_h - halo_post (clamp 0)
        # Last row read  = (oh_end-1)*sh + halo_post - pad_h (clamp OH-1)
        # We over-read by halo_post on each side so the conv internal padding
        # is zero (no synthetic zero rows at band frontiers).
        # NOTE: when oh_start == 0 we INCLUDE the natural pad_h at top via
        # F.pad below; idem for oh_end == conv_out_h at bottom.
        up_inner_start = oh_start * sh_st - pad_h
        up_inner_end = (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h

        # Add halo for the conv kernel internal frontier.
        is_top_band = (oh_start == 0)
        is_bot_band = (oh_end == conv_out_h)
        halo_top = 0 if is_top_band else halo_post
        halo_bot = 0 if is_bot_band else halo_post

        up_read_start = up_inner_start - halo_top
        up_read_end = up_inner_end + halo_bot

        # Clamp to valid upsample-output range; track how much each side was
        # cut so we can apply F.pad to recover the boundary halo with zeros.
        up_clamped_start = max(0, up_read_start)
        up_clamped_end = min(OH, up_read_end)
        # P-NBX-TILED-CONV2D-SMALL-SCALE 2026-05-14: previous formula
        # `+ (pad_h if is_top_band else 0)` double-counted edge padding.
        # `up_inner_start = oh_start*sh_st - pad_h`, so for is_top_band
        # (oh_start=0) we have up_inner_start = -pad_h and up_read_start
        # = -pad_h (halo_top=0). Then max(0, -up_read_start) = pad_h
        # already provides the image-top zero padding. Adding pad_h
        # again over-padded by one row, shifting band-1 output up by
        # pad_h rows. Symmetric on bot_band via in_read_end formula.
        # Validated by microtest scripts/microtest_tiled_conv2d_small_scale.py
        # which sweeps (kh, pad_h) at 1024^2 — pad_h=0 already PASS
        # pre-fix, pad_h>0 FAIL pre-fix, all PASS post-fix.
        pad_top_real = max(0, -up_read_start)
        pad_bot_real = max(0, up_read_end - OH)

        if up_clamped_end <= up_clamped_start:
            continue

        # Pre-upsample rows that produce [up_clamped_start, up_clamped_end).
        # nearest mapping: pre_row = floor(up_row / up_sh).
        pre_start = max(0, int(up_clamped_start // up_sh))
        pre_end = min(pre_h, int((up_clamped_end - 1) // up_sh) + 1)

        pre_band = pre_input[:, :, pre_start:pre_end, :]
        if pre_band.dtype != out_dtype:
            pre_band = pre_band.to(out_dtype)
        pre_band = pre_band.contiguous()

        # Upsample only the needed strip (small transient allocation).
        up_band = F.interpolate(
            pre_band, scale_factor=(up_sh, up_sw), mode="nearest"
        )

        # Slice the upsample-band to exactly the rows we wanted to read.
        up_offset_local = up_clamped_start - int(pre_start * up_sh)
        up_local_h = up_clamped_end - up_clamped_start
        up_band = up_band[:, :, up_offset_local:up_offset_local + up_local_h, :]
        up_band = up_band.contiguous()

        # Apply boundary padding (image-edge halo + clipped over-read).
        if pad_top_real > 0 or pad_bot_real > 0 or pad_w > 0:
            up_band = F.pad(
                up_band,
                (pad_w, pad_w, pad_top_real, pad_bot_real),
                mode="constant", value=0.0,
            )

        # Conv with padding=(0,0) on H (halo already provided in up_band).
        conv_band = F.conv2d(
            up_band, weight, bias=None,
            stride=(sh_st, sw_st),
            padding=(0, 0),
            dilation=(dh, dw),
            groups=groups,
        )

        # conv_band has exactly (oh_end - oh_start) rows by construction
        # (halo accounts for kernel borrow; F.pad accounts for image edges).
        actual_band_h = oh_end - oh_start
        # P-NBX-TILED-CONV2D-SMALL-SCALE 2026-05-14: when halo_top>0
        # (internal band frontier), conv_band starts at the halo row
        # before the band's first useful output row. The previous slice
        # `conv_band[:actual_band_h]` skipped the halo offset, shifting
        # all internal-frontier output rows by halo_top toward the top.
        # Cap actual_band_h by available conv_band height after the
        # halo offset, not by raw conv_band.shape[2].
        avail_after_halo = conv_band.shape[2] - halo_top // sh_st
        if avail_after_halo < actual_band_h:
            actual_band_h = avail_after_halo
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, halo_top // sh_st:halo_top // sh_st + actual_band_h, :]
        del pre_band, up_band, conv_band   # released before the next band allocates its own (as the NBX twin)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output




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
        from nbxtensor.tensor import NBXTensor
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
    """Band-streaming standalone conv with REAL halo (no zero-pad seams).

    Same algorithm as the fused variant minus the upsample. For each output
    band we slice the input with halo = (kh-1)*dh//2 rows above and below;
    the conv runs with padding=(0,0) on H. Image-edge halos are filled by
    F.pad with the original pad_h. Bit-identical to non-tiled conv2d.
    """
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
    # The halo rounded up to whole strides, the leading rows skipped = halo // stride: at stride 2
    # a 1-row halo misaligned every internal band by half an output row (2026-09-29; the NBX twin
    # reads the same cut from launch_keys.tiled_conv2d_bands).
    halo_h = -(-((kh - 1) * dh // 2) // sh_st) * sh_st

    for oh_start in range(0, out_h, band_oh):
        oh_end = min(oh_start + band_oh, out_h)
        is_top_band = (oh_start == 0)
        is_bot_band = (oh_end == out_h)

        # Input rows the kernel needs to produce [oh_start, oh_end):
        # core range minus pad_h, then extend by halo_h for internal frontiers.
        in_inner_start = oh_start * sh_st - pad_h
        in_inner_end = (oh_end - 1) * sh_st + dh * (kh - 1) + 1 - pad_h
        halo_top = 0 if is_top_band else halo_h
        halo_bot = 0 if is_bot_band else halo_h
        in_read_start = in_inner_start - halo_top
        in_read_end = in_inner_end + halo_bot

        in_clamped_start = max(0, in_read_start)
        in_clamped_end = min(IH, in_read_end)
        # P-NBX-TILED-CONV2D-SMALL-SCALE 2026-05-14: removed redundant
        # `+ (pad_h if is_top_band/is_bot_band else 0)`. `in_inner_start
        # = oh_start*sh_st - pad_h` already shifts by -pad_h, so the
        # max(0, -in_read_start) term ALREADY provides image-edge
        # padding for the top/bot band. Adding it twice shifted band-1
        # output up by pad_h rows on top and similarly for bot — caused
        # cos near 0 vs F.conv2d at any (kh>=3, pad_h>=1) signature
        # (microtest scripts/microtest_tiled_conv2d_small_scale.py).
        pad_top_real = max(0, -in_read_start)
        pad_bot_real = max(0, in_read_end - IH)

        if in_clamped_end <= in_clamped_start:
            continue

        view = input_tensor[:, :, in_clamped_start:in_clamped_end, :]
        if pad_top_real > 0 or pad_bot_real > 0 or pad_w > 0:
            # ONE band buffer, padded: the slice copied (and cast) straight into its interior. A contiguous copy then
            # F.pad held two input bands where the band's price holds one (HAT-L at 2048 px on 16 GB, 2026-10-22).
            hb, wb = int(view.shape[2]), int(view.shape[3])
            in_band = view.new_zeros((view.shape[0], view.shape[1], hb + pad_top_real + pad_bot_real, wb + 2 * pad_w),
                                     dtype=out_dtype)
            in_band[:, :, pad_top_real:pad_top_real + hb, pad_w:pad_w + wb].copy_(view)
        else:
            in_band = view.to(out_dtype).contiguous()
        del view

        conv_band = F.conv2d(
            in_band, weight, bias=None,
            stride=(sh_st, sw_st),
            padding=(0, 0),
            dilation=(dh, dw),
            groups=groups,
        )

        actual_band_h = oh_end - oh_start
        # P-NBX-TILED-CONV2D-SMALL-SCALE 2026-05-14: when halo_top>0
        # (internal band frontier), conv_band starts at the halo row
        # before the band's first useful output row. The previous slice
        # `conv_band[:actual_band_h]` skipped the halo offset, shifting
        # all internal-frontier output rows by halo_top toward the top.
        # Cap actual_band_h by available conv_band height after the
        # halo offset, not by raw conv_band.shape[2].
        avail_after_halo = conv_band.shape[2] - halo_top // sh_st
        if avail_after_halo < actual_band_h:
            actual_band_h = avail_after_halo
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, halo_top // sh_st:halo_top // sh_st + actual_band_h, :]
        del in_band, conv_band       # released before the next band allocates its own (as the NBX twin)

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

    # Resolve the canonical signature. Accept legacy positional or keyword.
    # Standard SDPA-style sig: (x, weight, eps).
    # Test for OUR type, not torch's. This entry is registered as an
    # interceptor in BOTH modes, so testing `isinstance(x, torch.Tensor)`
    # imported torch on the triton path just to discover it was not needed
    # (R33 inventory, 2026-09-03). Asking whether it is an NBXTensor answers
    # the same question and imports nothing.
    from nbxtensor.tensor import NBXTensor as _NBXTensor
    if isinstance(input_tensor, _NBXTensor):
        # NBXTensor path — defer to the standard wrapper, no tiling
        # (single rms_norm rarely overflows alone in triton mode where
        # kernels stream). Function is named `rms_norm` (no `_wrapper`
        # suffix) in wrappers.py.
        from nbxtensor.ops.norm_wrappers import rms_norm as nbx_rms_norm
        return nbx_rms_norm(input_tensor, weight, eps)

    # Torch branch only — never reached on the triton path.
    import torch

    x = input_tensor
    # Channels is the LAST dim
    H = x.shape[-3] if x.ndim >= 3 else 1
    if H < 2 or tile_factor <= 1:
        # Fall back to direct compute (no tiling worth it)
        return _rms_norm_direct(x, weight, eps)

    tile_factor = min(int(tile_factor), H)
    band_h = (H + tile_factor - 1) // tile_factor

    # PyTorch `empty_like(x, memory_format=contiguous_format)` has known
    # edge cases on non-contig sources (pytorch/pytorch issues #62027,
    # #79813, #158022) where the output stride pattern silently inherits
    # the source's permuted layout instead of true C-contiguous. The
    # symptom on Sana 4Kpx 16g VAE was a fully-black PNG: rms_norm::27
    # receives a non-contig NHWC view of NCHW storage; band slicing at
    # `band_h=256` (tile_factor=16) and `band_h=512` (tile_factor=8) both
    # fired the same compute, but the indexed assignment
    # `output[s] = out_band` wrote to a stride pattern that didn't match
    # the dispatcher's downstream reader.
    # Fix: allocate `output` via explicit `torch.empty(shape, dtype=,
    # device=)` (default C-contig, unambiguous) + force `.contiguous()`
    # on `x_band` before the math so the slice math is on packed
    # storage. Mirrors the repo's "Contiguous-guard pattern" doctrine
    # for tiled wrappers (CLAUDE.md).
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for h_start in range(0, H, band_h):
        h_end = min(h_start + band_h, H)
        slicer = [slice(None)] * x.ndim
        slicer[-3] = slice(h_start, h_end)
        s = tuple(slicer)
        x_band = x[s].contiguous()
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


# Self-managed dtype (the DtypeEngine does not wrap these): fused_upsample_conv2d may receive a FusionUpsampleProxy as
# arg[0], and the public dispatchers (mode-agnostic, the tiling engine's op_uid interceptors) delegate to the
# _torch/_nbx helpers, so the dtype self-management is the chosen path's (the _nbx ones are marked in nbxtensor).
fused_upsample_conv2d.self_manages_dtype = True
tiled_conv2d_spatial.self_manages_dtype = True
tiled_rms_norm_spatial.self_manages_dtype = True
