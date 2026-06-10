"""Name-driven latent axis alignment at component boundaries.

Video models do not share one 5D latent layout: Wan's transformer and VAE both
take [B, C, T, H, W], while CogVideoX's transformer takes [B, T, C, H, W] and
its VAE [B, C, T, H, W] — the vendor pipeline permutes between the denoising
loop and the decode. That permute lives in pipeline code, outside every traced
component, so the runtime must reproduce it at the boundary.

This module derives the required permutation purely from EXISTING contract
data — no new fields, no model names:

- The state tensor's axis roles come from `runtime/variables.json`
  `global.latents.resolver.shape_source` (each axis maps to a runtime
  quantity: batch_size / latent_frames / latent_height / latent_width, or the
  component channel attribute), which the builder derives from the traced
  transformer's named symbol dims.
- The consumer's expected axis roles come from its graph input tensor's
  symbolic dims (named symbols batch/time/height/width; the concrete 5D dim
  is the channel axis).

When both sides resolve to a complete 5-role bijection, the permutation that
maps the state layout onto the consumer layout is returned; identical layouts
(Wan) yield None (no-op). Any ambiguity or missing data also yields None —
the boundary then behaves exactly as before this module existed.
"""
from typing import Any, Dict, List, Optional, Tuple

_ROLES = ("batch", "channels", "time", "height", "width")

# shape_source value → axis role
_SHAPE_SOURCE_ROLE = {
    "runtime.batch_size": "batch",
    "runtime.latent_frames": "time",
    "runtime.latent_height": "height",
    "runtime.latent_width": "width",
}


def latent_axis_roles_from_variables(variables: Dict[str, Any]) -> Optional[List[str]]:
    """Axis roles of global.latents from its shape_source (5D only)."""
    spec = (variables or {}).get("global.latents") or {}
    src = ((spec.get("resolver") or {}).get("shape_source")) or {}
    if len(src) != 5:
        return None
    roles: List[str] = []
    for i in range(5):
        v = src.get(f"axis_{i}")
        if v in _SHAPE_SOURCE_ROLE:
            roles.append(_SHAPE_SOURCE_ROLE[v])
        elif isinstance(v, str) and v.endswith(".state_channels"):
            roles.append("channels")
        else:
            return None
    if sorted(roles) != sorted(_ROLES):
        return None
    return roles


def component_latent_input_roles(dag: Dict[str, Any]) -> Optional[List[str]]:
    """Axis roles of the consumer's single 5D graph input (named symbol dims;
    the one concrete dim is the channel axis). None when not exactly one 5D
    input or the roles are incomplete/ambiguous."""
    tensors = (dag or {}).get("tensors") or {}
    symbols = ((dag or {}).get("symbolic_context") or {}).get("symbols") or {}
    candidates = []
    for tid in (dag or {}).get("input_tensor_ids") or []:
        t = tensors.get(tid) or {}
        dims = (t.get("symbolic_shape") or {}).get("dims") or []
        if len(dims) == 5:
            candidates.append(dims)
    if len(candidates) != 1:
        return None
    roles: List[str] = []
    concrete_slots = 0
    for d in candidates[0]:
        name = None
        if isinstance(d, dict) and d.get("type") == "symbol":
            name = (symbols.get(d.get("id")) or {}).get("name")
        if name in ("batch", "time", "height", "width"):
            roles.append(name)
        elif isinstance(d, int):
            roles.append("channels")
            concrete_slots += 1
        else:
            return None
    if concrete_slots != 1 or sorted(roles) != sorted(_ROLES):
        return None
    return roles


def latent_permutation_for(variables: Dict[str, Any],
                           dag: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Permutation mapping the state layout onto the consumer layout, or None
    when identical / underdetermined."""
    src = latent_axis_roles_from_variables(variables)
    dst = component_latent_input_roles(dag)
    if src is None or dst is None or src == dst:
        return None
    perm = tuple(src.index(role) for role in dst)
    return perm
