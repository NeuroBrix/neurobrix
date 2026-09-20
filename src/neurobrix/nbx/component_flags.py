"""Per-component flags the CONTAINER carries, for every reader of a runtime flag.

A runtime decision that depended on a file only the build toolchain has was
measured on 2026-09-20: `zero_pad_embeddings` was read at runtime through the
toolchain's registry, reachable only in a checkout carrying the gitignored
`.nbx_registry` pointer, so every worktree and every installed engine ran
Wan2.1-T2V-1.3B without it and rendered a lattice of 16-px cells where the
developer's checkout rendered the sailboat (`docs/reference/release-decisions.md`).
Six flags are read that way (`zero_pad_embeddings`, `i2v_latent_conditioning`,
`vace_control_conditioning`, `pad_image_to_num_frames`, `requires_fp32_compute`,
`fp16_conv_cascade_safe`); each is a container that behaves differently for
its author and for its users until the flag rides in the `.nbx`.

The build writes each declared flag into the container's `extracted_values`
under the component it is declared on (topology.json, an existing free-form
per-component dict of build-time values — no new field, R18). This module is
the table those values land in when a container is opened, and the place the
flag reader (`core.runtime.registry_flags`) consults after the developer's
registry and before the documented default. It imports nothing from the
engine so the container format stays the lower layer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Flag names the engine reads per component. A registry annotation outside
# this set is not a runtime flag and is not carried.
RUNTIME_COMPONENT_FLAGS = (
    "zero_pad_embeddings",
    "i2v_latent_conditioning",
    "vace_control_conditioning",
    "pad_image_to_num_frames",
    "requires_fp32_compute",
    "fp16_conv_cascade_safe",
)

_ABSENT = object()
_TABLE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def register(model_name: Optional[str], extracted_values: Optional[Dict[str, Any]]) -> int:
    """Record the flags a container carries; returns how many were recorded.

    `extracted_values` is topology.json's top-level per-component dict. Only the
    names in RUNTIME_COMPONENT_FLAGS are taken, so a config value that happens
    to share a component's dict is never mistaken for a flag.
    """
    if not model_name or not isinstance(extracted_values, dict):
        return 0
    table: Dict[str, Dict[str, Any]] = {}
    for comp_name, vals in extracted_values.items():
        if not isinstance(vals, dict):
            continue
        carried = {k: vals[k] for k in RUNTIME_COMPONENT_FLAGS if k in vals}
        if carried:
            table[str(comp_name)] = carried
    _TABLE[str(model_name)] = table
    return sum(len(v) for v in table.values())


def get(model_name: Optional[str], component_name: Optional[str], flag_name: str,
        default: Any = None) -> Any:
    """The flag the container declares on this component, else `default`."""
    if not model_name or not component_name:
        return default
    comp = _TABLE.get(str(model_name), {}).get(str(component_name))
    if comp is None:
        return default
    return comp.get(flag_name, default)


def registered(model_name: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """What the table holds for a model (a copy), for diagnostics and tests."""
    return {c: dict(v) for c, v in _TABLE.get(str(model_name), {}).items()}


def clear() -> None:
    _TABLE.clear()
