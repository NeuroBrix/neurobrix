"""The container's own spatial answers — ONE authority for the executor and for the plan.

Two questions a request may leave open, answered from what the container declares and
from nothing else:

* `vae_scale_factor(manifest, defaults, components)` — the spatial compression between
  pixels and latents, in the two spellings vendors use (`vae_scale_factor`,
  `spatial_compression_ratio`), else derived from a traced latent extent against the
  manifest's trace resolution, else from the latent channel count.
* `container_output_size(manifest, defaults, components)` — (height, width) in pixels:
  the last two extents of a traced LATENT input (backbone first, VAE second) times the
  VAE scale. A decoder's input extents are latent extents by definition.

Why one module (2026-09-21, Wan2.1-T2V-1.3B): the executor answered these for the FLOW
(`_container_output_size`, and rendered at 480x832 from the backbone's traced latent),
while the request the PLAN was budgeted under carried height=None and width=None, so
Prism bound the VAE's spatial symbols to nothing, fell back to the trace extent (112x176)
and estimated 1.74 GiB for a decode whose first conv input alone is 24.84 GB at 480x832
(81 frames, 192 channels, fp32). A plan is budgeted under the request the flow executes:
both sides now read this module. Neither function ever guesses a resolution — a family
constant was the last resort here once and was removed for cause (video 512², colour
bands on every arm, 2026-09-05); None is a legitimate answer for a container that
declares no spatial latent anywhere.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: Where a spatial latent is declared, in the order it is consulted: the backbone's own
#: input first (it is what the flow allocates), the decoder's second (Open-Sora-v2's
#: backbone takes a FLATTENED latent and cannot answer; its VAE can).
_LATENT_SITES = (
    ("transformer", ("hidden_states", "sample", "latents", "x", "latent_model_input")),
    ("unet", ("hidden_states", "sample", "latents", "x", "latent_model_input")),
    ("dit", ("hidden_states", "sample", "latents", "x", "latent_model_input")),
    ("transformer_2", ("hidden_states", "sample", "latents", "x", "latent_model_input")),
    ("vae", ("z", "latents", "sample", "hidden_states")),
    ("vae_decoder", ("z", "latents", "sample", "hidden_states")),
)


def vae_scale_factor(manifest: Dict[str, Any], defaults: Dict[str, Any],
                     components: Dict[str, Any]) -> Optional[int]:
    """The container's declaration first, in both the names it uses; then the trace."""
    manifest_scale = (manifest or {}).get("vae_scale_factor")
    if manifest_scale is not None:
        return int(manifest_scale)
    for key in ("vae_scale_factor", "spatial_compression_ratio"):
        declared = (defaults or {}).get(key)
        if declared:
            return int(declared)
    transformer_attrs = ((components or {}).get("transformer") or {}).get("attributes") or {}
    trace_latent_extent = transformer_attrs.get("state_extent_0")
    if trace_latent_extent:
        trace_pixel_res = (manifest or {}).get("trace_resolution")
        if trace_pixel_res:
            return int(trace_pixel_res) // int(trace_latent_extent)
        state_channels = transformer_attrs.get("state_channels", 4)
        if state_channels >= 32:
            return 32
        return 8
    return None


def container_output_size(manifest: Dict[str, Any], defaults: Dict[str, Any],
                          components: Dict[str, Any],
                          comp_configs: Optional[Dict[str, Any]] = None) -> Optional[Tuple[int, int]]:
    """(height, width) in pixels the container itself implies, or None.

    `components` is the TOPOLOGY's component table (it carries each component's traced
    `shapes`); `comp_configs` is the loaded package's per-component configuration (it
    carries the profile's `attributes`, the trace fallback of the scale) — the executor
    holds both, the CLI holds the topology and passes it for both."""
    scale = vae_scale_factor(manifest, defaults, comp_configs if comp_configs is not None else components)
    if not scale:
        return None
    for name, keys in _LATENT_SITES:
        shapes = ((components or {}).get(name) or {}).get("shapes") or {}
        for key in keys:
            shape = shapes.get(key)
            if (isinstance(shape, (list, tuple)) and len(shape) in (4, 5)
                    and all(isinstance(v, int) for v in shape[-2:])):
                h, w = int(shape[-2]), int(shape[-1])
                if h > 0 and w > 0:
                    return h * int(scale), w * int(scale)
    return None
