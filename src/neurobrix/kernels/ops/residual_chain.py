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
    """Execute the residual chain band-by-band, return the merge output.

    t_base: [N, C, H, W] NCHW, the fork tensor (residual base).
    chain_weights: dict from `resolve_chain_weights`.
    tile_factor: number of bands along H.
    halo: rows of halo on each side per band (sum of conv halo radii).

    The chain is hard-coded to the validated pattern:
      conv1 → silu → conv2 → permute(NCHW→NHWC) → rms_norm → bias_add →
      permute(NHWC→NCHW)
    Then add to T_base = output.

    Returns: [N, C, H, W] tensor on the same device/dtype as t_base.
    """
    N, C, H, W = t_base.shape
    band_h = (H + tile_factor - 1) // tile_factor  # ceil division

    output = torch.empty_like(t_base)

    for i in range(tile_factor):
        h_start = i * band_h
        h_end = min((i + 1) * band_h, H)
        if h_start >= h_end:
            break

        h_in_start = max(0, h_start - halo)
        h_in_end = min(H, h_end + halo)

        # Band input from T_base — `.contiguous()` so the conv's flat
        # indexing reads a packed layout. The slice is non-contig along
        # the H dim's stride which still uses the original full H.
        band = t_base[:, :, h_in_start:h_in_end, :].contiguous()

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

        # Trim halo and merge with T_base into output
        trim_top = h_start - h_in_start
        trim_h = h_end - h_start
        band_out = band[:, :, trim_top:trim_top + trim_h, :]
        t_base_band = t_base[:, :, h_start:h_end, :]
        torch.add(t_base_band, band_out,
                  out=output[:, :, h_start:h_end, :])
        del band
        del band_out

    return output
