"""
S5 — DC-AE residual chain band-streamed execution.

Implements the runtime wrapper that absorbs a long residual chain
(detected by `OpLevelTilingEngine._detect_residual_chains`) into a
single band-streamed compute. Reduces per-band intermediate footprint
from 3× full-resolution buffers to 1× band-sized buffer.

The chain pattern (validated on Sana 4Kpx VAE DC-AE blocks):

    T_base (fork output)
      |
      |--- (chain): conv -> silu -> conv -> permute -> rms_norm -> add -> permute
      |
      +--- merge_add: chain_output + T_base = output

For each band of H rows:
  1. Slice T_base [h_in_start:h_in_end] with halo for conv kernels
  2. Run the chain ops on the band (PyTorch ATen)
  3. Trim halo from chain output
  4. Add T_base band → write to output[h_start:h_end]

Live tensor footprint per band: T_base (full) + output (full, fill in
place) + ~1 band intermediate (small) → fits 16 GiB on Sana 4Kpx VAE
4096² peak.

R34 conformant: pattern detection is structural (`fork → linear≥3 →
merge`), no model-name lookup. R30 dual-branch: compiled path uses
PyTorch ATen here; triton path port will add NBX equivalent.

S5 P-PRISM-NEVER-REFUSE v2 2026-05-13.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class ChainSentinel:
    """Zero-allocation sentinel returned by chain intermediate
    interceptors. Carries no real data — the merge interceptor
    reads the precomputed merge output from the chain registry.
    """

    __slots__ = ("chain_id", "shape", "dtype", "device")

    def __init__(self, chain_id, shape, dtype, device):
        self.chain_id = chain_id
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def _nbytes(self):
        return 0

    def __repr__(self):
        return f"ChainSentinel(chain_id={self.chain_id}, shape={self.shape})"


def resolve_chain_weights(
    chain_spec: Dict[str, Any],
    dag: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Look up the weight tensors each chain op needs.

    Parses the chain ops' `input_tensor_ids` from the DAG, identifies
    `param::*` references, and pulls the matching entries from the
    executor's `_weights` dict.

    Returns a dict shaped for `band_streamed_chain_torch`:
        {
          "conv1_weight": tensor, "conv1_bias": tensor or None,
          "conv1_padding": [1, 1], "conv1_stride": [1, 1],
          "conv1_dilation": [1, 1], "conv1_groups": 1,
          "conv2_weight": tensor, "conv2_bias": tensor or None,
          "conv2_padding": ..., (same fields)
          "norm_weight": tensor, "norm_eps": 1e-6,
          "post_norm_bias": tensor or None,
          "permute_forward": [0, 2, 3, 1],
          "permute_backward": [0, 3, 1, 2],
        }

    Returns None if any required weight cannot be resolved.
    """
    ops = dag.get("ops", {})
    chain_uids = chain_spec["chain_uids"]
    out: Dict[str, Any] = {
        "conv1_weight": None, "conv1_bias": None,
        "conv1_padding": [1, 1], "conv1_stride": [1, 1],
        "conv1_dilation": [1, 1], "conv1_groups": 1,
        "conv2_weight": None, "conv2_bias": None,
        "conv2_padding": [1, 1], "conv2_stride": [1, 1],
        "conv2_dilation": [1, 1], "conv2_groups": 1,
        "norm_weight": None, "norm_eps": 1e-6,
        "post_norm_bias": None,
        "permute_forward": [0, 2, 3, 1],
        "permute_backward": [0, 3, 1, 2],
    }

    conv_idx = 0
    for uid in chain_uids:
        op = ops.get(uid, {})
        op_type = op.get("op_type", "")
        in_tids = op.get("input_tensor_ids", [])
        if op_type == "aten::convolution":
            # input_tensor_ids: [activation, weight, optional bias]
            if len(in_tids) >= 2:
                w_tid = in_tids[1]
                if isinstance(w_tid, str) and w_tid.startswith("param::"):
                    w = weights.get(w_tid[len("param::"):])
                    if w is None:
                        return {}
                    out[f"conv{conv_idx + 1}_weight"] = w
            if len(in_tids) >= 3:
                b_tid = in_tids[2]
                if isinstance(b_tid, str) and b_tid.startswith("param::"):
                    out[f"conv{conv_idx + 1}_bias"] = weights.get(
                        b_tid[len("param::"):])
            # Conv attrs from args: [act, w, b, stride, padding, dil,
            # transposed, output_padding, groups]
            attrs = op.get("attributes", {})
            args = attrs.get("args", [])

            def _get_list(arg):
                if isinstance(arg, dict) and arg.get("type") == "list":
                    return [int(v) if isinstance(v, int) else int(
                        v.get("value", 1)) for v in arg.get("value", [])]
                return None

            def _get_scalar(arg):
                if isinstance(arg, dict) and arg.get("type") == "scalar":
                    return arg.get("value")
                return None

            if len(args) >= 4:
                v = _get_list(args[3])
                if v:
                    out[f"conv{conv_idx + 1}_stride"] = v
            if len(args) >= 5:
                v = _get_list(args[4])
                if v:
                    out[f"conv{conv_idx + 1}_padding"] = v
            if len(args) >= 6:
                v = _get_list(args[5])
                if v:
                    out[f"conv{conv_idx + 1}_dilation"] = v
            if len(args) >= 9:
                v = _get_scalar(args[8])
                if isinstance(v, int):
                    out[f"conv{conv_idx + 1}_groups"] = v
            conv_idx += 1
        elif op_type == "custom::rms_norm":
            if len(in_tids) >= 2:
                w_tid = in_tids[1]
                if isinstance(w_tid, str) and w_tid.startswith("param::"):
                    w = weights.get(w_tid[len("param::"):])
                    if w is None:
                        return {}
                    out["norm_weight"] = w
            attrs = op.get("attributes", {})
            for arg in attrs.get("args", []):
                if isinstance(arg, dict) and arg.get("type") == "scalar":
                    v = arg.get("value")
                    if isinstance(v, float):
                        out["norm_eps"] = v
                        break
        elif op_type == "aten::add":
            # Post-rms_norm bias-add: a learnable shift.
            # input_tensor_ids: [activation, bias_param]
            if len(in_tids) >= 2:
                b_tid = in_tids[1]
                if isinstance(b_tid, str) and b_tid.startswith("param::"):
                    out["post_norm_bias"] = weights.get(
                        b_tid[len("param::"):])
        elif op_type == "aten::permute":
            attrs = op.get("attributes", {})
            args = attrs.get("args", [])
            if len(args) >= 2 and isinstance(args[1], dict) \
                    and args[1].get("type") == "list":
                perm = [int(v.get("value", 0)) if isinstance(v, dict)
                        else int(v) for v in args[1].get("value", [])]
                if perm == [0, 2, 3, 1]:
                    out["permute_forward"] = perm
                elif perm == [0, 3, 1, 2]:
                    out["permute_backward"] = perm

    if out["conv1_weight"] is None or out["conv2_weight"] is None:
        return {}
    return out


def band_streamed_chain_torch(
    t_base: torch.Tensor,
    chain_weights: Dict[str, Any],
    tile_factor: int,
    halo: int,
) -> torch.Tensor:
    """Execute the residual chain band-by-band, write result IN-PLACE
    into T_base's buffer, return T_base.

    t_base: [N, C, H, W] NCHW, the fork tensor (residual base). MODIFIED
        in place to hold the merge output. Caller's view of T_base is
        invalidated; the returned tensor is the merge output.
    chain_weights: dict from `resolve_chain_weights`.
    tile_factor: number of bands along H.
    halo: rows of halo on each side per band (sum of conv halo radii).

    The chain is hard-coded to the validated pattern:
      conv1 → silu → conv2 → permute(NCHW→NHWC) → rms_norm → bias_add →
      permute(NHWC→NCHW)
    Then add to T_base IN PLACE = output.

    Correctness invariant: a `halo_carry` buffer holds the rows the
    NEXT band needs as its top halo, captured before the current band
    overwrites them. The first band uses T_base directly (no top halo
    has been overwritten yet).

    Memory: T_base full (in-place), halo_carry tiny (halo × W × C × dt
    bytes ≪ 1 MiB for 4Kpx scales), band transient ~ 1 / tile_factor of
    the full intermediate size. No full-output allocation.

    Returns: t_base (modified in place).
    """
    N, C, H, W = t_base.shape
    band_h = (H + tile_factor - 1) // tile_factor  # ceil division

    halo_carry: Optional[torch.Tensor] = None

    for i in range(tile_factor):
        h_start = i * band_h
        h_end = min((i + 1) * band_h, H)
        if h_start >= h_end:
            break

        h_in_start = max(0, h_start - halo)
        h_in_end = min(H, h_end + halo)

        # Build the input band: top halo from halo_carry if any rows in
        # the top-halo region have been overwritten by previous bands;
        # rest from T_base. `.contiguous()` so the conv reads a packed
        # layout.
        if halo_carry is not None and h_in_start < h_start:
            top_size = h_start - h_in_start
            # halo_carry has the most recent `halo` rows of original T_base
            # right before they were overwritten. We need the LAST top_size
            # of them.
            top_band = halo_carry[:, :, -top_size:, :]
            rest_band = t_base[:, :, h_start:h_in_end, :]
            band = torch.cat([top_band, rest_band], dim=2).contiguous()
            del top_band, rest_band
        else:
            band = t_base[:, :, h_in_start:h_in_end, :].contiguous()

        # Save original rows that the NEXT band will need as its top halo,
        # BEFORE we overwrite them. The save is the last `halo` rows of
        # [h_start:h_end]; clone to detach from t_base storage.
        if i + 1 < tile_factor:
            halo_save_size = min(halo, h_end - h_start)
            halo_carry = t_base[:, :,
                                h_end - halo_save_size:h_end, :].clone()

        # conv1 (cast weights once per band for consistency with band's
        # dtype; PyTorch caches weight conversions in practice but the
        # wrapper stays explicit to avoid silent dtype drift).
        w1 = chain_weights["conv1_weight"]
        b1 = chain_weights["conv1_bias"]
        if w1.dtype != band.dtype:
            w1 = w1.to(band.dtype)
        if b1 is not None and b1.dtype != band.dtype:
            b1 = b1.to(band.dtype)
        new_band = F.conv2d(
            band, w1, b1,
            stride=tuple(chain_weights["conv1_stride"]),
            padding=tuple(chain_weights["conv1_padding"]),
            dilation=tuple(chain_weights["conv1_dilation"]),
            groups=chain_weights["conv1_groups"],
        )
        del band  # free conv1 input as soon as F.conv2d has it captured
        band = new_band

        # silu in-place
        band = F.silu(band, inplace=True)

        # conv2
        w2 = chain_weights["conv2_weight"]
        b2 = chain_weights["conv2_bias"]
        if w2.dtype != band.dtype:
            w2 = w2.to(band.dtype)
        if b2 is not None and b2.dtype != band.dtype:
            b2 = b2.to(band.dtype)
        new_band = F.conv2d(
            band, w2, b2,
            stride=tuple(chain_weights["conv2_stride"]),
            padding=tuple(chain_weights["conv2_padding"]),
            dilation=tuple(chain_weights["conv2_dilation"]),
            groups=chain_weights["conv2_groups"],
        )
        del band
        band = new_band

        # permute NCHW → NHWC (contiguous needed for the last-dim reduction)
        band = band.permute(*chain_weights["permute_forward"]).contiguous()

        # rms_norm along last dim (feature dim, per-pixel) — stay in
        # band's dtype throughout. PyTorch's bf16 rms is good enough for
        # diffusion VAE (the reference 32g run uses bf16 end-to-end and
        # produces coherent images). Avoiding the fp32 cast saves a
        # full intermediate per band.
        norm_w = chain_weights["norm_weight"]
        eps = float(chain_weights.get("norm_eps", 1e-6))
        if norm_w.dtype != band.dtype:
            norm_w = norm_w.to(band.dtype)
        rms = band.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        band = band.mul(rms).mul_(norm_w)
        del rms

        # bias add (post-norm shift), in-place when possible
        post_bias = chain_weights.get("post_norm_bias")
        if post_bias is not None:
            if post_bias.dtype != band.dtype:
                post_bias = post_bias.to(band.dtype)
            band = band.add_(post_bias)

        # permute NHWC → NCHW
        band = band.permute(*chain_weights["permute_backward"]).contiguous()

        # Trim halo and merge IN PLACE into T_base rows [h_start:h_end].
        # `t_base[h_start:h_end]` is the LAST consumer of original T_base
        # in this region (top halo for band i+1 was saved into halo_carry
        # above), so writing back here is correctness-safe.
        trim_top = h_start - h_in_start
        trim_h = h_end - h_start
        band_out = band[:, :, trim_top:trim_top + trim_h, :]
        t_base[:, :, h_start:h_end, :].add_(band_out)
        del band, band_out

    return t_base


# ---------------------------------------------------------------------------
# NBX-PURE TRITON variant — R33 zero-torch port of band_streamed_chain_torch.
# ---------------------------------------------------------------------------
# Same band-streaming algorithm, same correctness invariant (halo_carry),
# same memory shape. The only difference is the backend: every ATen op
# is replaced by its NBX wrapper from `neurobrix.kernels.wrappers`.
#
# P-TRITON-LIVE-WATERMARK-AUDIT 2026-05-14 L4: closes the R30 chain wrapper
# gap left open by commit c9d2581 ("future band_streamed_chain_nbx would
# close this gap"). On Sana 4Kpx 16g triton the L2 LIVE_DUMP showed two
# 4 GiB tensors at (1, 128, 4096, 4096) co-resident at add::89 because
# the chain wrapper was SKIP'd on triton — natural NBX dispatch keeps
# all chain intermediates fully materialized. With this wrapper the
# chain runs band-by-band on the triton arena tensors, peak transient
# drops to 1/tile_factor of the chain intermediate.


def band_streamed_chain_nbx(
    t_base: "Any",
    chain_weights: Dict[str, Any],
    tile_factor: int,
    halo: int,
) -> "Any":
    """NBXTensor band-streamed residual chain (R33-pure mirror of
    `band_streamed_chain_torch`).

    Args:
      t_base: NBXTensor, the fork tensor (residual base). Mutated in
        place to hold the merge output. Returned for caller convenience.
      chain_weights: dict from `resolve_chain_weights` — same shape as
        for the torch variant, but weight values are NBXTensor when
        the executor is in triton mode.
      tile_factor: number of bands along H.
      halo: rows of halo on each side per band.

    Returns: t_base (mutated).

    R33: no `torch.*`, no `F.*`. All ops via
    `neurobrix.kernels.wrappers` and NBXTensor methods.

    Correctness invariant: halo_carry is a clone of the rows the NEXT
    band needs as its top halo, captured BEFORE the current band's
    `add_` overwrites them. The first band uses t_base directly because
    no top halo has been overwritten yet.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import (
        conv2d_wrapper, silu as silu_w, rms_norm as rms_norm_w,
        add as add_w,
    )

    N, C, H, W = t_base.shape
    band_h = (H + tile_factor - 1) // tile_factor

    halo_carry = None  # type: Optional[NBXTensor]

    for i in range(tile_factor):
        h_start = i * band_h
        h_end = min((i + 1) * band_h, H)
        if h_start >= h_end:
            break

        h_in_start = max(0, h_start - halo)
        h_in_end = min(H, h_end + halo)

        # Build the input band.
        # If we have a halo_carry AND the top-halo region [h_in_start,
        # h_start) has been overwritten by the previous band's add_,
        # take the top from halo_carry and the rest from t_base.
        if halo_carry is not None and h_in_start < h_start:
            top_size = h_start - h_in_start
            top_band = halo_carry[:, :, -top_size:, :]
            rest_band = t_base[:, :, h_start:h_in_end, :]
            band = NBXTensor.cat([top_band, rest_band], dim=2).contiguous()
        else:
            band = t_base[:, :, h_in_start:h_in_end, :].contiguous()

        # Save halo for next band BEFORE overwriting (clone detaches
        # from t_base storage).
        if i + 1 < tile_factor:
            halo_save_size = min(halo, h_end - h_start)
            halo_carry = t_base[:, :,
                                h_end - halo_save_size:h_end, :].clone()

        # conv1
        w1 = chain_weights["conv1_weight"]
        b1 = chain_weights["conv1_bias"]
        band = conv2d_wrapper(
            band, w1, b1,
            stride=tuple(chain_weights["conv1_stride"]),
            padding=tuple(chain_weights["conv1_padding"]),
            dilation=tuple(chain_weights["conv1_dilation"]),
            groups=chain_weights["conv1_groups"],
        )

        # silu
        band = silu_w(band)

        # conv2
        w2 = chain_weights["conv2_weight"]
        b2 = chain_weights["conv2_bias"]
        band = conv2d_wrapper(
            band, w2, b2,
            stride=tuple(chain_weights["conv2_stride"]),
            padding=tuple(chain_weights["conv2_padding"]),
            dilation=tuple(chain_weights["conv2_dilation"]),
            groups=chain_weights["conv2_groups"],
        )

        # permute NCHW → NHWC (R33-pure NBXTensor.permute)
        band = band.permute(*chain_weights["permute_forward"]).contiguous()

        # rms_norm over last (feature) dim
        norm_w = chain_weights["norm_weight"]
        eps = float(chain_weights.get("norm_eps", 1e-6))
        band = rms_norm_w(band, norm_w, eps)

        # post-norm bias shift (if any)
        post_bias = chain_weights.get("post_norm_bias")
        if post_bias is not None:
            band = add_w(band, post_bias)

        # permute NHWC → NCHW
        band = band.permute(*chain_weights["permute_backward"]).contiguous()

        # Trim halo from chain output, then add to t_base in place via
        # __setitem__ + add. NBX add_inplace_nbx requires identical
        # shapes AND contiguous target, but t_base[:, :, h_start:h_end, :]
        # is a non-contig view (slicing on dim 2) — so we use the
        # `add_w` + setitem pattern: materialize the band-sized sum,
        # write it back. The setitem path uses _strided_scatter for
        # the non-contig dst and a single contiguous src copy.
        trim_top = h_start - h_in_start
        trim_h = h_end - h_start
        band_out = band[:, :, trim_top:trim_top + trim_h, :]
        t_base_slice = t_base[:, :, h_start:h_end, :]
        # add returns a new contiguous tensor; setitem writes back.
        t_base[:, :, h_start:h_end, :] = add_w(t_base_slice, band_out)

    return t_base

