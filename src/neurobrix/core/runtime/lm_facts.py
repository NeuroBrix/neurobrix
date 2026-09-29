"""The facts of the LM a flow decodes with — one reader for the plan and for both engines' sessions.

Pure Python (no torch, no Triton): Prism reads it to price the decode cache, and the decode
sessions of both engines (`core/flow/autoregressive.py`, `triton/flow/autoregressive.py`) read
it to build that cache. Written once because it was written three times and disagreed: Prism
read the LM facts from the package's `lm_config` only, the two sessions fell back to the LM
component's extracted values, so a flow whose package carries no `lm_config` (VibeVoice's
next-token diffusion) ran a KV-cached decoder the plan had not priced (2026-09-29).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def lm_config_of(defaults: Optional[Mapping[str, Any]], topology: Optional[Mapping[str, Any]],
                 lm_name: Optional[str]) -> Dict[str, Any]:
    """The LM facts a decode cache is sized from: the package's `lm_config`, else the LM
    component's extracted values (the build records them per component)."""
    lm_config = dict((defaults or {}).get("lm_config") or {})
    if lm_config:
        return lm_config
    extracted = (((topology or {}).get("extracted_values") or {}).get(lm_name) or {}) if lm_name else {}
    return {
        "num_layers": extracted.get("num_hidden_layers") or extracted.get("num_layers"),
        "num_heads": extracted.get("num_attention_heads") or extracted.get("num_heads"),
        "hidden_size": extracted.get("hidden_size"),
        "num_kv_heads": extracted.get("num_key_value_heads") or extracted.get("num_kv_heads"),
        "head_dim": extracted.get("head_dim"),
        # the window is load-bearing for the prompt-aware KV ceiling
        "max_position_embeddings": extracted.get("max_position_embeddings"),
    }


def session_lm_name(gen_info: Mapping[str, Any], component_names: Iterable[str]) -> str:
    """The component a decode session runs as the LM: the generation's `lm_component` when it is
    a component, else the first component that is neither the text head nor a codec."""
    names = list(component_names)
    lm_name = gen_info.get("lm_component", "language_model")
    if lm_name not in names:
        for name in names:
            if name not in ("lm_head", "codec.decoder"):
                return name
    return lm_name


def decode_lm_component(topology: Mapping[str, Any], component_names: Iterable[str]) -> Optional[str]:
    """The component a flow runs as a KV-cached decoder, or None when its flow opens no decode
    session: the autoregressive flow's session LM; the next-token-diffusion LM — the component
    its diffusion stage is conditioned from."""
    flow = topology.get("flow") or {}
    kind = flow.get("type")
    if kind == "autoregressive_generation":
        return session_lm_name(flow.get("generation") or {}, component_names)
    if kind == "next_token_diffusion":
        stages = flow.get("stages") or (flow.get("audio") or {}).get("stages") or []
        for st in stages:
            src = (st.get("diffusion") or {}).get("condition_from")
            if src:
                return src
    return None


def image_guidance_weight(resolved: Optional[Mapping[str, Any]], defaults: Mapping[str, Any]) -> float:
    """An image-AR request's guidance weight: the CLI's, else the package's. Above 1 the flow runs
    the LM on [cond, uncond] (batch 2) and the head once per branch."""
    cli = (resolved or {}).get("global.guidance_scale")
    weight = float(cli) if cli is not None else defaults.get("guidance_scale")
    if weight is None:
        raise RuntimeError(
            "guidance_scale missing from defaults.json for "
            "autoregressive_image. Set in the model registry at import.")
    return float(weight)


def decode_sequences(topology: Mapping[str, Any], defaults: Mapping[str, Any],
                     resolved: Optional[Mapping[str, Any]] = None) -> int:
    """How many sequences the decode cache holds at once: an image-AR generation under guidance
    decodes [cond, uncond] as a batch of 2; a next-token diffusion under guidance keeps a second,
    negative context beside the prompt's (a second branch of the same cache). One otherwise."""
    flow = topology.get("flow") or {}
    kind = flow.get("type")
    if kind == "autoregressive_generation" and \
            (flow.get("generation") or {}).get("type") == "autoregressive_image":
        return 2 if image_guidance_weight(resolved, defaults) > 1.0 else 1
    if kind == "next_token_diffusion":
        cli = (resolved or {}).get("global.guidance_scale")
        scale = float(cli) if cli is not None else defaults.get("cfg_scale")
        if scale is None:
            raise RuntimeError("ZERO FALLBACK: 'cfg_scale' missing from defaults.json — "
                               "next_token_diffusion refuses to invent a value.")
        return 2 if float(scale) != 1.0 else 1
    return 1
