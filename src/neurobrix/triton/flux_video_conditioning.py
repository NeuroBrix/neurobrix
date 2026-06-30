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


def is_flux_video(ctx: Any, components: List[str]) -> bool:
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
    # patch side from the packing (C*p^2 = packed_dim -> p = sqrt(packed_dim/C))
    p = int(round((packed_dim / c) ** 0.5)) or 1
    lh = int(packing_info["height"]) // p
    lw = int(packing_info["width"]) // p

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
    txt = _resolve_txt(ctx)
    txt_seq = int(txt.shape[1]) if txt is not None else 0
    txt_ids = NBXTensor.zeros((b, txt_seq, 3), dtype, f"cuda:{dev_idx}")

    # cond: zeros [B, num_tokens, (C+1)*p^2] — T2V mask + masked-ref both empty.
    cond_dim = (c + 1) * (p * p)
    cond = NBXTensor.zeros((b, num_tokens, cond_dim), dtype, f"cuda:{dev_idx}")

    vr = ctx.variable_resolver
    for name, val in ((IMG_IDS_VAR, img_ids), (TXT_IDS_VAR, txt_ids), (COND_VAR, cond)):
        vr.set(name, val)
        vr.resolved[name] = val
