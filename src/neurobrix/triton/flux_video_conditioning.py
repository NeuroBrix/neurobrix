"""FLUX-video (Open-Sora-v2) positional-id + cond synthesis (triton mode).

R33/R34-pure mirror of ``core/runtime/resolution/flux_video_conditioning.py``.
Open-Sora-v2's MMDiT is a FLUX-style packed-latent video denoiser; the vendor
pipeline prepares, OUTSIDE the traced graph:
  * FLUX 3-axis positional  img_ids [B, num_tokens, 3] over the (T, H/p, W/p) grid
  * text positional ids     txt_ids [B, txt_seq, 3]  (all zeros)
  * a channel-concat cond   cond    [B, num_tokens, (C+1)*p^2]  (all zeros for T2V)
None are produced by any traced component, so NeuroBrix synthesizes them at
runtime. ``img_ids`` is the FLUX positional grid that EmbedND turns into the
rotary cos/sin — it MUST bit-match compiled or the (correct) half-split rotary
rotates by the wrong positions.

R34: the positional ids are deterministic index generation, not neural compute —
built with numpy CPU glue (the same allowance used by ``i2v_conditioning.py``),
then materialized as NBXTensor on the packed-state device. Zero torch, zero
vendor import.
"""

import numpy as np
from typing import Any, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

IMG_IDS_VAR = "global.img_ids"
TXT_IDS_VAR = "global.txt_ids"
COND_VAR = "global.cond"


def is_flux_family(ctx: Any, components: List[str]) -> bool:
    """True iff a loop denoiser declares an ``img_ids`` input (FLUX-family)."""
    comps = ctx.pkg.topology.get("components", {})
    for c in components:
        inputs = comps.get(c, {}).get("interface", {}).get("inputs", []) or []
        if "img_ids" in inputs:
            return True
    return False


def _nbx_on(arr: np.ndarray, dev_idx: int) -> NBXTensor:
    """from_numpy onto a specific device (from_numpy uses the current device)."""
    prev = DeviceAllocator.get_device()
    DeviceAllocator.set_device(dev_idx)
    try:
        return NBXTensor.from_numpy(np.ascontiguousarray(arr))
    finally:
        DeviceAllocator.set_device(prev)


def _resolve_txt(ctx: Any) -> Optional[NBXTensor]:
    """The T5 text embedding (drives txt_ids length)."""
    res = ctx.variable_resolver.resolved
    for k in ("text_encoder.last_hidden_state", "text_encoder.output_0",
              "global.encoder_hidden_states"):
        v = res.get(k)
        if v is None:
            try:
                v = ctx.variable_resolver.get(k)
            except Exception:
                v = None
        if isinstance(v, NBXTensor):
            return v
    return None


def conditioning_shapes(b: int, num_tokens: int, packed_dim: int, channels: int, frames: int,
                        height: int, width: int, txt_seq: int) -> dict:
    """The shapes `prepare` synthesizes, from the packed state and its packing: the patch side
    p = sqrt(packed_dim / C); img_ids [b, t*(h/p)*(w/p), 3]; txt_ids [b, txt_seq, 3]; cond
    [b, num_tokens, (C+1)*p^2]. `prepare` builds with them; the derived census binds with them."""
    p = int(round((packed_dim / channels) ** 0.5)) or 1
    lh, lw = height // p, width // p
    return {"img_ids": [b, frames * lh * lw, 3], "txt_ids": [b, txt_seq, 3],
            "cond": [b, num_tokens, (channels + 1) * p * p], "p": p, "lh": lh, "lw": lw}


def prepare(ctx: Any, packed_state: NBXTensor, packing_info: dict) -> None:
    """Synthesize img_ids / txt_ids / cond into the variable resolver (T2V).

    Args:
      packed_state: [B, num_tokens, C*p^2] (after the 5D pack).
      packing_info: {channels, frames, height, width, ndim:5} from the 5D pack.
    """
    dev_idx = packed_state._device_idx
    dtype = packed_state.dtype
    b, num_tokens, packed_dim = packed_state.shape
    c = int(packing_info["channels"])
    t = int(packing_info["frames"])
    txt = _resolve_txt(ctx)
    txt_seq = int(txt.shape[1]) if txt is not None else 0
    # patch side from the packing (C*p^2 = packed_dim -> p = sqrt(packed_dim/C)) — the one shape rule
    shp = conditioning_shapes(b, num_tokens, packed_dim, c, t, int(packing_info["height"]),
                              int(packing_info["width"]), txt_seq)
    p, lh, lw = shp["p"], shp["lh"], shp["lw"]

    # img_ids: FLUX 3-axis grid (frame, row, col) over (t, lh, lw). Built in
    # float32 then cast to the packed dtype — bit-mirror of the compiled path.
    ids = np.zeros((t, lh, lw, 3), dtype=np.float32)
    ids[..., 0] = np.arange(t, dtype=np.float32)[:, None, None]
    ids[..., 1] = np.arange(lh, dtype=np.float32)[None, :, None]
    ids[..., 2] = np.arange(lw, dtype=np.float32)[None, None, :]
    ids = ids.reshape(1, t * lh * lw, 3)
    ids = np.broadcast_to(ids, (b, t * lh * lw, 3))
    img_ids = _nbx_on(ids, dev_idx).to(dtype)

    # txt_ids: zeros [B, txt_seq, 3] — txt_seq from the T5 embedding.
    txt_ids = NBXTensor.zeros(tuple(shp["txt_ids"]), dtype, f"cuda:{dev_idx}")

    # cond: zeros [B, num_tokens, (C+1)*p^2] — T2V mask + masked-ref both empty.
    cond = NBXTensor.zeros(tuple(shp["cond"]), dtype, f"cuda:{dev_idx}")

    vr = ctx.variable_resolver
    for name, val in ((IMG_IDS_VAR, img_ids), (TXT_IDS_VAR, txt_ids), (COND_VAR, cond)):
        vr.set(name, val)
        vr.resolved[name] = val
