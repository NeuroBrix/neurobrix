"""The CFG cascade's guidance scale and threshold — the ONE reader for the CFG engines and for Prism.

Torch-free (R33): Prism counts the sequences a guided decode keeps in its cache on the Triton branch too
(`core/runtime/lm_facts.decode_sequences`), and the cascade sat in `core/cfg/engine.py`, which imports torch — a
`--triton` chatterbox run loaded torch at its plan, and the shared unload then synchronised an NBX device through
`torch.cuda` (Mac, 2026-10-23: "Torch not compiled with CUDA enabled").
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple


def guidance_cascade(family: str, topology: Mapping[str, Any], defaults: Mapping[str, Any],
                     resolved: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[float], float]:
    """(guidance_scale, cfg_threshold) of the CFG cascade — the ONE reader for the engine and for Prism,
    which counts the sequences a guided decode keeps in its cache (`lm_facts.decode_sequences`):
    1. the request's `global.guidance_scale`; 2. `topology.extracted_values._global.guidance_scale`;
    3. `defaults.guidance_scale`, None when the container declares none (CFG disabled, cond-only).
    The threshold is the family's `cfg_threshold`, else its diffusion defaults', else 1.0."""
    from neurobrix.core.config import get_family_defaults, get_family_config
    family_defaults = get_family_defaults(family)
    diffusion_defaults = get_family_config(family).get("diffusion", {}).get("defaults", {})
    scale = (resolved or {}).get("global.guidance_scale")
    if scale is None:
        scale = ((topology.get("extracted_values") or {}).get("_global") or {}).get("guidance_scale")
    if scale is None:
        scale = defaults.get("guidance_scale")
    threshold = family_defaults.get("cfg_threshold")
    if threshold is None:
        threshold = diffusion_defaults.get("cfg_threshold", 1.0)
    return (None if scale is None else float(scale)), float(threshold)
