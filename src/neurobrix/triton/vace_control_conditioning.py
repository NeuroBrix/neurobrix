"""Triton-branch VACE control conditioning (NBXTensor, R33-pure).

TRITON-side mirror of the compiled brick
`core/runtime/resolution/vace_control_conditioning.py`. Same SEMANTICS, but it
lives entirely in the triton branch and computes end-to-end with NBXTensor — no
torch, no `F.*`, no reach into `core/` compute (Hocine's architecture invariant:
the two branches are separate; this is the R30 mirror).

WanVACE feeds the denoiser two extra inputs each step:
  * control_hidden_states  = cat([inactive_latent(16), reactive_latent(16),
    reshaped_mask(64)], dim=1) — 96 channels.
  * control_hidden_states_scale = ones(len(vace_layers)).

All-generate path (zeros control clip, all-white mask): inactive == reactive ==
encode(0) and the reshaped mask is all-ones, so one vae_encoder pass suffices and
both 16-channel halves are equal. Built once (step-invariant); the triton flow
stores them as globals the InputResolver binds; the triton CFG engine repeats the
control to batch=2 (the scale is batch-invariant).

Allowed imports: NBXTensor + triton device glue (R33), numpy for host-side
per-channel stats / mask (R34 CPU glue), and the shared registry-flag / profile
readers (data-driven config, not compute).
"""

import json
from typing import Any, Optional

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator, NBXDtype
from neurobrix.core.runtime.registry_flags import get_component_flag

CONTROL_VAR = "global.control_hidden_states"
SCALE_VAR = "global.control_hidden_states_scale"


def conditioning_spec(ctx: Any, loop_comp: str) -> Optional[dict]:
    """Return the normalized vace_control_conditioning spec, or None (inert)."""
    model_name = ctx.pkg.manifest.get("model_name")
    flag = get_component_flag(model_name, loop_comp, "vace_control_conditioning",
                              default=None)
    if not flag:
        return None
    spec = dict(flag) if isinstance(flag, dict) else {}
    spec.setdefault("condition_component", "vae_encoder")
    spec.setdefault("mask_channels", 64)
    spec.setdefault("vace_layers", 15)
    spec.setdefault("z_dim", 16)
    return spec


def _vae_latent_stats(ctx: Any) -> tuple:
    """Read (latents_mean, latents_std, latent_channels) from the vae profile."""
    prof_path = ctx.pkg.cache_path / "components" / "vae" / "profile.json"
    prof = json.loads(prof_path.read_text())
    cfg = prof.get("config") if isinstance(prof.get("config"), dict) else prof
    mean = cfg.get("latents_mean") or prof.get("latents_mean")
    std = cfg.get("latents_std") or prof.get("latents_std")
    ch = (cfg.get("latent_channels") or prof.get("latent_channels") or cfg.get("z_dim")
          or (len(mean) if mean else 16))
    return mean, std, int(ch)


def _to_channels_first(latent: NBXTensor, latent_channels: int) -> NBXTensor:
    """Arrange a 5D NBXTensor latent to [B, C, T, H, W] (channels at dim 1)."""
    if latent.dim() != 5:
        return latent
    if latent.shape[1] == latent_channels:
        return latent
    if latent.shape[2] == latent_channels:
        return latent.permute(0, 2, 1, 3, 4).contiguous()
    return latent


def _nbx_on(arr: np.ndarray, dev_idx: int) -> NBXTensor:
    """from_numpy onto a specific device (from_numpy uses the current device)."""
    prev = DeviceAllocator.get_device()
    DeviceAllocator.set_device(dev_idx)
    try:
        return NBXTensor.from_numpy(arr)
    finally:
        DeviceAllocator.set_device(prev)


def build_control(ctx: Any, spec: dict) -> Optional[NBXTensor]:
    """Build control_hidden_states = cat([inactive, reactive, mask], dim=1) NBX.

    All-generate path: inactive == reactive == encode(0); mask all-ones. Returns
    None if the vae_encoder output is not yet resolved.
    """
    cond_comp = spec["condition_component"]
    res = ctx.variable_resolver.resolved
    latent = res.get(f"{cond_comp}.output_0")
    if latent is None:
        latent = res.get(f"{cond_comp}.output")
    if not isinstance(latent, NBXTensor):
        return None

    mean, std, latent_channels = _vae_latent_stats(ctx)
    latent = _to_channels_first(latent, latent_channels).float()
    dev_idx = latent._device_idx
    b, _c, lt, lh, lw = latent.shape

    # Normalize as the vendor: (x - mean) * (1/std). Pre-broadcast the per-channel
    # stats to the full latent shape in numpy so the NBXTensor elementwise kernels
    # run same-shape (no broadcast dependency) — same discipline as the i2v brick.
    if mean is not None and std is not None:
        shape = (b, latent_channels, lt, lh, lw)
        mean_full = np.ascontiguousarray(
            np.broadcast_to(np.asarray(mean, np.float32).reshape(1, -1, 1, 1, 1), shape))
        inv_std_full = np.ascontiguousarray(
            np.broadcast_to((1.0 / np.asarray(std, np.float32)).reshape(1, -1, 1, 1, 1), shape))
        latent = (latent - _nbx_on(mean_full, dev_idx)) * _nbx_on(inv_std_full, dev_idx)

    # inactive == reactive == encode(0) for the zeros control clip; channel-cat.
    video_latents = NBXTensor.cat([latent, latent], dim=1)  # [B, 2*z_dim, T, H, W]
    mask_np = np.ones((b, int(spec["mask_channels"]), lt, lh, lw), dtype=np.float32)
    mask_t = _nbx_on(mask_np, dev_idx)
    control = NBXTensor.cat([video_latents, mask_t], dim=1)  # [B, 2*z_dim+mask_ch, ...]

    import os as _os
    if _os.environ.get("NBX_DIAG_VACE") == "1":
        _ln = latent.to(NBXDtype.float32) if latent.nbx_dtype != NBXDtype.float32 else latent
        _ln_np = _ln.numpy()
        print(f"   [NBX-DIAG-VACE-TRITON] vae_latent shape={list(latent.shape)} "
              f"norm mean={float(_ln_np.mean()):.3f} std={float(_ln_np.std()):.3f} | "
              f"control shape={list(control.shape)} "
              f"(expect C={2 * latent_channels + int(spec['mask_channels'])})")
    return control


def build_scale(spec: dict, dev_idx: int) -> NBXTensor:
    """control_hidden_states_scale = ones(len(vace_layers)) as NBXTensor."""
    return _nbx_on(np.ones((int(spec["vace_layers"]),), dtype=np.float32), dev_idx)


def batch_for_cfg(control: NBXTensor) -> NBXTensor:
    """Repeat the batch-1 control to batch=2 for the batched CFG forward.

    Shared across CFG cond/uncond (the vendor passes the SAME tensor to both), so
    in batched mode it is repeated [c, c] to match the batch=2 hidden_states.
    """
    if control.shape[0] == 1:
        return NBXTensor.cat([control, control], dim=0)
    return control
