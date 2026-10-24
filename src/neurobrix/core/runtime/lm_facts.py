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
    component's extracted values (the build records them per component), else the KV state the
    build declares for that decoder (topology `state`, read from its graph: layers, key-value heads,
    head dims, window — the record a staged decoder whose vendor config names no architecture
    carries; canary-qwen's llm and chatterbox's t3_cfg, 2026-10-18). Query heads and hidden size
    are then the extracted values' or None: a reader that needs them refuses by name."""
    lm_config = dict((defaults or {}).get("lm_config") or {})
    if lm_config:
        return lm_config
    extracted = (((topology or {}).get("extracted_values") or {}).get(lm_name) or {}) if lm_name else {}
    missing = [k for k, alts in (("num_layers", ("num_hidden_layers", "num_layers")),
                                 ("num_heads", ("num_attention_heads", "num_heads")),
                                 ("hidden_size", ("hidden_size",)))
               if not any(extracted.get(a) for a in alts)]
    if missing:
        kv = next((e for e in (((topology or {}).get("state") or {}).get(lm_name) or []) if e.get("kind") == "kv"), None)
        if kv is None:
            raise RuntimeError(
                f"ZERO FALLBACK: the decoder '{lm_name}' has no lm_config, its extracted values lack "
                f"{', '.join(missing)} and the container declares no KV state for it — its decode cache cannot "
                f"be sized; the build must record them.")
        return {
            "num_layers": int(kv["layers"]),
            "num_heads": extracted.get("num_attention_heads") or extracted.get("num_heads"),
            "hidden_size": extracted.get("hidden_size"),
            "num_kv_heads": int(kv["kv_heads"]),
            "head_dim": int(kv["k_dim"]),
            "max_position_embeddings": kv.get("window") or extracted.get("max_position_embeddings"),
            "source": "state",
        }
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
    its diffusion stage is conditioned from; the encoder-decoder flow's autoregressive stage (the
    decoder conditioned on the encoder's states, `cross_attention_from`)."""
    flow = topology.get("flow") or {}
    kind = flow.get("type")
    names = set(component_names)
    stages = flow.get("stages") or (flow.get("audio") or {}).get("stages") or []
    if kind == "autoregressive_generation":
        return session_lm_name(flow.get("generation") or {}, component_names)
    if kind == "encoder_decoder":
        for st in stages:
            if st.get("execution") == "autoregressive" and st.get("cross_attention_from"):
                return st.get("component")
        return None
    if kind == "next_token_diffusion":
        for st in stages:
            src = (st.get("diffusion") or {}).get("condition_from")
            if src:
                return src
    # A flow that stages its decoder (audio_llm, tts_llm, dual_ar): the stage it runs token by token
    # is the KV-cached decoder — one rule for every staged flow, read from the stage's `execution`
    # (state engine step 3, 2026-10-18: these flows re-forwarded their whole context every token and
    # the plan priced no cache for them).
    for st in stages:
        if st.get("execution") in ("autoregressive", "dual_ar") and st.get("component") in names:
            return st["component"]
    # A flow that names its decoder outright (the VLM flow's `lm_component`: the thinker of an omni
    # model, the language model of a VLM) — the KV-cached decoder is that component. Without it the
    # plan charged Qwen3-Omni-30B's KV cache to no component and the expert store's grant took its
    # room (32g triton, 2026-10-13).
    for section in ("vlm", "multimodal", "audio_llm"):
        named = (flow.get(section) or {}).get("lm_component")
        if named and named in set(component_names):
            return named
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
                     resolved: Optional[Mapping[str, Any]] = None, family: Optional[str] = None) -> int:
    """How many sequences the decode cache holds at once: an image-AR generation under guidance
    decodes [cond, uncond] as a batch of 2; a next-token diffusion under guidance keeps a second,
    negative context beside the prompt's (a second branch of the same cache); a speech LM under
    guidance (tts_llm) keeps its text-zeroed context beside the conditioned one, read through the
    CFG engine's own cascade (`family` names its defaults). One otherwise."""
    flow = topology.get("flow") or {}
    kind = flow.get("type")
    if kind == "tts_llm":
        from neurobrix.core.guidance_cascade import guidance_cascade
        if family is None:
            raise RuntimeError("ZERO FALLBACK: a tts_llm decode counts its sequences through the CFG cascade, "
                               "which needs the container's family (manifest) — none was given.")
        scale, threshold = guidance_cascade(family, topology, defaults, resolved)
        return 2 if scale is not None and scale > threshold else 1
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


def decoder_cache_facts(dag: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """A decoder's KV cache geometry, read from its graph by the flows' own reader
    (`core.flow.decoder_kv.decoder_self_attention_plan`): {num_layers, num_heads, head_dim}, or None
    when the flows build no cache — a graph without self-attention, or one whose decoded token nothing
    can place at its position (no position_ids input, no arange, table slice or count to offset; it
    decodes by recompute). One reader for the plan and for both engines' flows: before it, the flows
    sized the cache from the graph and max_tokens, and the plan priced none (2026-09-29)."""
    if not dag:
        return None
    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan, positions_are_offset_by
    plan = decoder_self_attention_plan(dag)
    if plan is None or not positions_are_offset_by(plan):
        return None
    return {"num_layers": int(plan["num_layers"]), "num_heads": int(plan["num_heads"]),
            "head_dim": int(plan["head_dim"]), "kv_heads_at_op": int(plan["kv_heads_at_op"]),
            "v_head_dim": int(plan["v_head_dim"]), "plan": plan}


def decoder_cache_from_plan(kv_plan, facts: Mapping[str, Any], max_tokens: int, decoder: str):
    """The plan's cache for a decoder the flow opens (encoder-decoder, audio_llm, tts_llm, dual_ar,
    vlm), checked against the graph's geometry: refused
    by name when the plan carries none, when its geometry is not the graph's, or when it is shorter
    than the window the flow decodes — never sized in the flow."""
    if kv_plan is None:
        raise RuntimeError(
            f"ZERO FALLBACK: the plan carries no KV cache for the decoder '{decoder}'. Prism prices the "
            f"cache of every decoding flow (core/runtime/lm_facts.decode_lm_component, the encoder-decoder "
            f"stage included); a plan without one did not see this flow — re-plan, never size it here.")
    # The plan's cache keeps the KEY-VALUE heads the build declared (unexpanded); the graph may hand the op
    # its keys grouped (the same count) or already expanded to the query heads (a multiple: the wrappers
    # un-expand by that ratio) — GLM-4.1V's 2 key-value heads under 32 query heads, 2026-10-18.
    got = (int(kv_plan.num_layers), int(kv_plan.k_head_dim), int(kv_plan.v_head_dim))
    want = (facts["num_layers"], facts["head_dim"], facts.get("v_head_dim", facts["head_dim"]))
    kv_heads, at_op, q_heads = int(kv_plan.num_kv_heads), facts.get("kv_heads_at_op", facts["num_heads"]), facts["num_heads"]
    heads_fit = kv_heads == at_op or (at_op == q_heads and kv_heads > 0 and q_heads % kv_heads == 0)
    if got != want or not heads_fit:
        raise RuntimeError(
            f"ZERO FALLBACK: the plan's KV cache for '{decoder}' is {got[0]} layers, {kv_heads} key-value heads, "
            f"k/v head dims {got[1]}/{got[2]}, and the decoder graph's self-attention is {want[0]} layers, "
            f"{at_op} heads at the op ({q_heads} query heads), head dims {want[1]}/{want[2]} — the plan priced "
            f"another decoder.")
    if int(kv_plan.max_cache_len) < int(max_tokens):
        raise RuntimeError(
            f"ZERO FALLBACK: the plan's KV cache for '{decoder}' holds {kv_plan.max_cache_len} positions "
            f"and the flow decodes up to {max_tokens} — re-plan at this request.")
    return kv_plan
