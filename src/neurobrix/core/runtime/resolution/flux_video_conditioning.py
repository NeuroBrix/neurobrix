"""FLUX-video (Open-Sora-v2) positional-id + cond synthesis (compiled mode).

Open-Sora-v2's MMDiT is a FLUX-style **packed-latent video** denoiser. The vendor
pipeline prepares, OUTSIDE the traced graph:
  * a packed image latent  img    [B, num_tokens, C*p^2]
  * FLUX 3-axis positional  img_ids [B, num_tokens, 3] over the (T, H/p, W/p) grid
  * text positional ids     txt_ids [B, txt_seq, 3]  (all zeros)
  * a channel-concat cond   cond   [B, num_tokens, (C+1)*p^2]
For pure T2V the cond is all-zeros (mask + masked-reference are both empty — the
vendor `prepare_inference_condition` t2v branch leaves them zero). These tensors
are not produced by any traced component, so NeuroBrix synthesizes them at runtime.

DATA-DRIVEN + gated: only fires for a loop denoiser that declares an ``img_ids``
input (FLUX-family only — CogVideoX / Wan / Mochi do not). Inert for every other
model. Mirror: ``triton/flux_video_conditioning.py`` (NBXTensor, R33-pure).
"""

import torch
from typing import Any, List, Optional

IMG_IDS_VAR = "global.img_ids"
TXT_IDS_VAR = "global.txt_ids"
COND_VAR = "global.cond"


def is_flux_video(ctx: Any, components: List[str]) -> bool:
    """True iff a loop denoiser declares an ``img_ids`` input (FLUX-family)."""
    comps = ctx.pkg.topology.get("components", {})
    for c in components:
        inputs = comps.get(c, {}).get("interface", {}).get("inputs", []) or []
        if "img_ids" in inputs:
            return True
    return False


def _resolve_txt(ctx: Any) -> Optional[torch.Tensor]:
    """The T5 text embedding (drives txt_ids length)."""
    for k in ("text_encoder.last_hidden_state", "text_encoder.output_0",
              "global.encoder_hidden_states"):
        v = ctx.variable_resolver.resolved.get(k)
        if v is None:
            try:
                v = ctx.variable_resolver.get(k)
            except Exception:
                v = None
        if isinstance(v, torch.Tensor):
            return v
    return None


def prepare(ctx: Any, packed_state: torch.Tensor, packing_info: dict) -> None:
    """Synthesize img_ids / txt_ids / cond into the variable resolver (T2V).

    Args:
      packed_state: [B, num_tokens, C*p^2] (after the 5D pack).
      packing_info: {channels, frames, height, width, ndim:5} from the 5D pack.
    """
    device = packed_state.device
    dtype = packed_state.dtype
    b, num_tokens, packed_dim = packed_state.shape
    c = int(packing_info["channels"])
    t = int(packing_info["frames"])
    # patch side from the packing (C*p^2 = packed_dim -> p = sqrt(packed_dim/C))
    p = int(round((packed_dim / c) ** 0.5)) or 1
    lh = int(packing_info["height"]) // p
    lw = int(packing_info["width"]) // p

    # img_ids: FLUX 3-axis grid (frame, row, col) over (t, lh, lw).
    ids = torch.zeros(t, lh, lw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = torch.arange(t, device=device, dtype=torch.float32)[:, None, None]
    ids[..., 1] = torch.arange(lh, device=device, dtype=torch.float32)[None, :, None]
    ids[..., 2] = torch.arange(lw, device=device, dtype=torch.float32)[None, None, :]
    img_ids = ids.reshape(1, t * lh * lw, 3).repeat(b, 1, 1).to(dtype)

    # txt_ids: zeros [B, txt_seq, 3] — txt_seq from the T5 embedding.
    txt = _resolve_txt(ctx)
    txt_seq = int(txt.shape[1]) if txt is not None else 0
    txt_ids = torch.zeros(b, txt_seq, 3, device=device, dtype=dtype)

    # cond: zeros [B, num_tokens, (C+1)*p^2] — T2V mask + masked-ref both empty.
    cond_dim = (c + 1) * (p * p)
    cond = torch.zeros(b, num_tokens, cond_dim, device=device, dtype=dtype)

    vr = ctx.variable_resolver
    for name, val in ((IMG_IDS_VAR, img_ids), (TXT_IDS_VAR, txt_ids), (COND_VAR, cond)):
        vr.set(name, val)
        vr.resolved[name] = val
