"""Fused upsample_nearest2d + convolution, and the spatially tiled convolution — band-streaming, Triton side.

Eliminates the intermediate upsampled tensor that would otherwise sit in VRAM between an upsample op and the
convolution that consumes it (Sana 4Kpx VAE: a [1, 256, 4096, 4096] fp32 upsample, 16 GB, that the conv consumes at
once). Band by band along output H: the pre-upsample input band with the conv's halo, upsampled alone, convolved, written
into the conv output at its offset; the full conv output is materialised once. The proxies stand for the upsample that
is not materialised (`FusionUpsampleProxy`) and for a broadcast clone (`BroadcastClonePyroxy`).

The PyTorch branch's versions and the dispatchers that pick a branch by tensor type are the engine's
(neurobrix/kernels/ops/fused_upsample_conv_torch.py); this module imports no torch.
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

    # ── observability ──────────────────────────────────────────────────────
    # THE RULE: a brick that replaces a call site preserves the observability
    # of that site. This proxy stands in for an upsample op, and the per-op
    # recorder drops anything that is not a tensor — so two of the three
    # spatial upsamples of the Wan VAE produced no record at all, and every
    # cross-engine walk was blind exactly there. The proxy therefore produces
    # the record itself.
    #
    # It costs nothing and it is EXACT, because a nearest upsample only
    # replicates: every input value appears scales_h x scales_w times, so
    #     l2(up(x)) == sqrt(sh*sw) * l2(x)
    # and the first row of the result is the first row of the input with each
    # element repeated sw times. No tensor is materialized — which is the
    # whole reason this proxy exists.

    def nbx_observable_summary(self) -> dict:
        """What the replaced upsample would have reported, without materializing it."""
        import math
        x = self.pre_input
        sh = max(1, int(round(float(self.scales_h))))
        sw = max(1, int(round(float(self.scales_w))))
        # `self.dtype` is a __slots__ field the interceptor does not always set,
        # and `getattr(x, "dtype", self.dtype)` would evaluate it EAGERLY as the
        # default — raising AttributeError before the getattr ever runs.
        dt = getattr(x, "dtype", None)
        if dt is None:
            dt = getattr(self, "dtype", None)
        out = {"shape": list(self.output_shape), "dtype": str(dt),
               "synthetic": True, "replaced_by": "fused_upsample_conv2d"}
        try:
            flat = x.reshape(-1)
            head_src = flat[:10]
            head_src = (head_src.tolist() if hasattr(head_src, "tolist")
                        else list(head_src))
            head = [float(v) for v in head_src for _ in range(sw)][:10]
            out["head10"] = head
            if hasattr(x, "detach"):               # a torch tensor (the compiled engine): its own fp32 norm, torch never imported
                l2 = float(x.detach().reshape(-1).float().norm().item())
            else:                                  # NBXTensor -> numpy
                import numpy as _np
                l2 = float(_np.linalg.norm(
                    _np.asarray(x.numpy(), dtype=_np.float64).ravel()))
            out["l2_norm"] = l2 * math.sqrt(sh * sw)
        except Exception as exc:                   # a summary must never break a run
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

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

    # Sentinel: FusionUpsampleProxy holds no GPU allocation of its own
    # (it just references the pre-upsample tensor + scales). The triton
    # sequence's deferred-free accounting (sequence.py:_run_single_device)
    # reads `_nbytes` to size the drain budget; expose 0 here so the
    # proxy is a no-op in that accounting path. The actual pre-upsample
    # tensor lives in its own arena slot and is freed when that slot is
    # killed independently.
    @property
    def _nbytes(self):
        return 0

    def __repr__(self):
        return (
            f"FusionUpsampleProxy(pre={tuple(self.pre_input.shape)}, "
            f"scales=({self.scales_h},{self.scales_w}), "
            f"out_shape={self.output_shape})"
        )


class BroadcastClonePyroxy:
    """Sentinel returned by an intercepted aten.clone op whose input is
    a broadcast-view (stride=0 on some dim from a preceding aten.expand)
    and whose consumer chain is `aten.view -> aten.pixel_shuffle`. Carries
    the broadcast-view tensor itself (NBX expand returns a stride-0 view,
    not a materialized tensor — see NBXTensor.expand line 1581) plus the
    static broadcast metadata so pixel_shuffle can read through the view
    directly (Triton tl.load with stride 0 aliases broadcast indices),
    eliminating the clone materialization entirely.

    Same architectural pattern as FusionUpsampleProxy: zero compute,
    zero allocation in the interceptor; the downstream pixel_shuffle
    consumes the proxy and performs the fused work.

    `bcast_view`: the expand output (5D NBXTensor with stride 0 on
    `bcast_dim`). Same data_ptr as the pre-expand tensor (NBX expand
    is metadata-only).
    `bcast_dim` / `bcast_factor`: extracted statically at register
    time from the expand op's input/output shape diff.
    `post_clone_shape` / `post_view_shape`: kept for the view
    interceptor's dim arithmetic compatibility.
    """

    __slots__ = ("bcast_view", "bcast_dim", "bcast_factor",
                 "post_clone_shape", "post_view_shape", "dtype", "device")

    def __init__(self, bcast_view, bcast_dim, bcast_factor,
                 post_clone_shape, post_view_shape):
        self.bcast_view = bcast_view
        self.bcast_dim = int(bcast_dim)
        self.bcast_factor = int(bcast_factor)
        self.post_clone_shape = tuple(post_clone_shape)
        self.post_view_shape = tuple(post_view_shape)
        self.dtype = getattr(bcast_view, "dtype", None)
        self.device = getattr(bcast_view, "device", None)

    @property
    def shape(self):
        return self.post_clone_shape

    @property
    def ndim(self):
        return len(self.post_clone_shape)

    @property
    def _nbytes(self):
        return 0

    def __repr__(self):
        return (
            f"BroadcastClonePyroxy(view={tuple(self.bcast_view.shape)}, "
            f"bcast_dim={self.bcast_dim}, bcast={self.bcast_factor}, "
            f"post_clone={self.post_clone_shape})"
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


def _pair(v):
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _tiled_conv2d_spatial_nbx(
    input_tensor, weight, bias, sh_st, sw_st, pad_h, pad_w,
    dh, dw, groups, tile_factor,
):
    """Band-streaming standalone conv with REAL halo (no zero-pad seams).

    POINT 5 FIX (P-SANA-4KPX-RUNTIME): mirrors `_tiled_conv2d_spatial_torch`
    exactly. The previous version applied `padding=(pad_h, pad_w)` on every
    band — at internal band frontiers this filled zeros where the kernel
    should have read from the NEXT band's rows (real halo). The result was
    bit-divergent from the non-tiled path at every band boundary, by ~50 in
    max_abs at Sana 4Kpx (98%+ of elements affected). The torch backend has
    always used the halo-based algorithm; the NBX backend now matches it.

    Per band we slice the input with halo = (kh-1)*dh//2 rows above and
    below; conv runs with `padding=(0,0)`. Image-edge halos are filled by
    `constant_pad_nd_wrapper` with the original `pad_h` only on the top of
    the first band and the bottom of the last band. Bit-identical to the
    non-tiled `conv2d_wrapper` + `_torch` backend.
    """
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.wrappers import conv2d_wrapper, constant_pad_nd_wrapper, _conv_bias_inplace

    N, in_c, IH, IW = input_tensor.shape
    out_c, _, kh, kw = weight.shape
    out_h = (IH + 2 * pad_h - dh * (kh - 1) - 1) // sh_st + 1
    out_w = (IW + 2 * pad_w - dw * (kw - 1) - 1) // sw_st + 1

    output = NBXTensor.empty(
        (N, out_c, out_h, out_w),
        device=input_tensor.device, dtype=input_tensor.dtype,
    )
    # The bands are `launch_keys.tiled_conv2d_bands` — the cut the derived census keys with.
    from nbxtensor.ops.launch_keys import tiled_conv2d_bands
    for (oh_start, oh_end, in_clamped_start, in_clamped_end,
         pad_top_real, pad_bot_real, skip) in tiled_conv2d_bands(
            IH, out_h, kh, sh_st, dh, pad_h, tile_factor):
        # POINT 6 FIX (P-SANA-4KPX-RUNTIME): see `_fused_upsample_conv2d_nbx`
        # for the full rationale — H-slice on an NCHW tensor produces a
        # non-contiguous view; the downstream conv/pad Triton wrappers use
        # flat indexing that assumes contiguous and read wrong memory
        # otherwise. The torch backend (`_tiled_conv2d_spatial_torch`)
        # already enforces contiguity at this point.
        view = input_tensor[:, :, in_clamped_start:in_clamped_end, :]
        if pad_top_real > 0 or pad_bot_real > 0 or pad_w > 0:
            # ONE band buffer, padded: the slice copied by its strides straight into its interior (`copy_`). A
            # contiguous copy then constant_pad_nd held two input bands where the band's price holds one (HAT-L at
            # 2048 px on 16 GB, 2026-10-22).
            hb, wb = int(view.shape[2]), int(view.shape[3])
            in_band = NBXTensor.zeros((int(view.shape[0]), int(view.shape[1]), hb + pad_top_real + pad_bot_real,
                                       wb + 2 * pad_w), dtype=view.dtype, device=view.device)
            in_band.narrow(2, pad_top_real, hb).narrow(3, pad_w, wb).copy_(view)
        else:
            in_band = view.contiguous()
        del view

        conv_band = conv2d_wrapper(
            in_band, weight, None,
            stride=(sh_st, sw_st), padding=(0, 0),
            dilation=(dh, dw), groups=groups,
        )

        actual_band_h = oh_end - oh_start
        # P-NBX-TILED-CONV2D-SMALL-SCALE step 2: same halo offset fix
        # as the torch path. When halo_top > 0 (internal band frontier),
        # conv_band[0] is the convolution at the halo row, not at the
        # band's first useful output row. Skip the halo rows on the
        # read side.
        avail_after_halo = conv_band.shape[2] - skip
        if avail_after_halo < actual_band_h:
            actual_band_h = avail_after_halo
        # Bias add inside band — see _fused_upsample_conv2d_nbx for rationale
        # (avoids 8 GiB bias broadcast materialization on Sana 4Kpx).
        if bias is not None:
            # In place, as every conv wrapper adds its bias (`_conv_bias_inplace`): an out-of-place add held a
            # second band beside the first, which Prism's band price does not hold.
            _conv_bias_inplace(conv_band, bias)
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, skip:skip + actual_band_h, :]
        # Released before the next band allocates its own: held, the previous band's input and output stood beside
        # the next band's (HAT-S at 2048 px on 16 GB: the second 1 153 MB band of conv::2 found no room, 2026-10-22).
        del in_band, conv_band
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
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.wrappers import upsample_nearest2d_wrapper, conv2d_wrapper, _conv_bias_inplace

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

    pre_h = pre_input.shape[2]
    from nbxtensor.ops.wrappers import constant_pad_nd_wrapper
    # The bands over the UPSAMPLED extent are `launch_keys.tiled_conv2d_bands` — one cut with the
    # standalone tiled conv and the derived census (the halo rounded to whole strides).
    from nbxtensor.ops.launch_keys import tiled_conv2d_bands
    for (oh_start, oh_end, up_clamped_start, up_clamped_end,
         pad_top_real, pad_bot_real, skip) in tiled_conv2d_bands(
            OH, conv_out_h, kh, sh_st, dh, pad_h, tile_factor):

        # Pre-upsample rows that produce [up_clamped_start, up_clamped_end).
        # nearest mapping: pre_row = floor(up_row / up_sh).
        pre_start = max(0, int(up_clamped_start // up_sh))
        pre_end = min(pre_h, int((up_clamped_end - 1) // up_sh) + 1)

        # POINT 6 FIX (P-SANA-4KPX-RUNTIME): slicing on the H dimension of
        # an NCHW tensor produces a non-contiguous view (the C stride keeps
        # its original H-based value while the slice covers a smaller H).
        # The downstream Triton wrappers (`upsample_nearest2d_wrapper`,
        # `conv2d_wrapper`) use flat 1D indexing that ASSUMES contiguous
        # input — fed a strided view they read wrong memory and produce
        # garbage (94% of elements differ by >1.0 vs torch reference,
        # measured at conv::55 on Sana 4Kpx). The torch path
        # (`_fused_upsample_conv2d_torch`) calls `.contiguous()` at the
        # same two points; mirror that here. Bands where the slice covers
        # the full H (first band when `tile_factor==1`) hit the early-
        # return path in `NBXTensor.contiguous()` at zero cost.
        pre_band = pre_input[:, :, pre_start:pre_end, :].contiguous()
        up_band = upsample_nearest2d_wrapper(
            pre_band,
            output_size=[int(pre_band.shape[2] * up_sh), int(pre_band.shape[3] * up_sw)],
            scales_h=up_sh, scales_w=up_sw,
        )
        up_offset_local = up_clamped_start - int(pre_start * up_sh)
        up_local_h = up_clamped_end - up_clamped_start
        up_band = up_band[:, :, up_offset_local:up_offset_local + up_local_h, :].contiguous()

        # Apply boundary padding (image-edge halo + clipped over-read).
        # Internal band frontiers get pad_top/bot_real == 0 because halo
        # already provides real rows.
        if pad_top_real > 0 or pad_bot_real > 0 or pad_w > 0:
            up_band = constant_pad_nd_wrapper(
                up_band, [pad_w, pad_w, pad_top_real, pad_bot_real], value=0.0,
            )

        # Conv with padding=(0,0) — halo + image-edge padding already in up_band.
        conv_band = conv2d_wrapper(
            up_band, weight, None,
            stride=(sh_st, sw_st), padding=(0, 0),
            dilation=(dh, dw), groups=groups,
        )

        actual_band_h = oh_end - oh_start
        # P-NBX-TILED-CONV2D-SMALL-SCALE step 2: halo_top offset on the
        # conv_band read side. Internal band frontiers (halo_top > 0)
        # write F.conv2d output row `oh_start + halo_top - 1` at
        # output row `oh_start` without this offset.
        avail_after_halo = conv_band.shape[2] - skip
        if avail_after_halo < actual_band_h:
            actual_band_h = avail_after_halo
        # Bias add inside band — keeps the broadcast bounded to one band's
        # (N, out_c, actual_band_h, conv_out_w) instead of the full output
        # (8 GiB broadcast on Sana 4Kpx 4096×4096 fp16). Mathematically
        # identical (bias is per-channel, applies element-wise on H,W).
        if bias is not None:
            _conv_bias_inplace(conv_band, bias)     # in place, as every conv wrapper adds its bias
        output[:, :, oh_start:oh_start + actual_band_h, :] = conv_band[:, :, skip:skip + actual_band_h, :]
        del pre_band, up_band, conv_band           # released before the next band allocates its own

    return output


# Phase 1 opt-in cleanup: mark op-level tiling NBX interceptors as
# self-managing dtype. TritonSequence._compile_op checks this flag to
# skip the DtypeEngine wrap for these ops. Justified because:
#   - fused_upsample_conv2d may receive a FusionUpsampleProxy as arg[0]
#     (no nbx_dtype attribute → DtypeEngine wrap would no-op anyway)
#   - the nested per-band calls delegate to conv2d_wrapper which is
#     itself self-managed (narrowing + output = compute_dtype)
#   - rms_norm tiled wrapper delegates to the rms_norm wrapper which
#     follows AMP_FP32_OPS via DtypeEngine (or the opt-in cast-back
#     variant when activations_fp16_safe is enabled)
_fused_upsample_conv2d_nbx.self_manages_dtype = True
_tiled_conv2d_spatial_nbx.self_manages_dtype = True
# (the public dispatchers, mode-agnostic, are marked in the engine's fused_upsample_conv_torch.py)
