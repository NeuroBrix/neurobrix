"""Band-streamed residual chain — the COMPILED (torch) half.

Split out of `residual_chain.py` on 2026-09-03. The two halves — this one and
the R33-pure `band_streamed_chain_nbx` — lived in one module, whose
module-level `import torch` therefore reached the triton branch through every
consumer of the NBX variant. Dormant on the paths measured, but the import was
unconditional, and dormant is not absent.

Nothing here is called from the triton path: `tiling_engine` selects the NBX
variant in triton and triton_sequential modes and only imports this module in
the compiled branch.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


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
