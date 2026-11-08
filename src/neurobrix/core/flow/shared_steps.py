"""Flow steps the compiled and the Triton handlers ran alike, body for body — written once (Hocine Benkelaya 2026-10-29 22:39).

One mixin per flow (VLM, audio, RNN-T, static graph), inherited by both of its handlers. Torch-free at import (the
Triton branch inherits them: R33); the tensors they touch are each branch's own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nbxtensor.tensor import NBXTensor
from neurobrix.nbx.neurotax import normalize_tensor_name_strict
import logging


class VLMSteps:
    """The steps the compiled `VLMEngine` and the Triton `TritonVLMEngine` ran alike (core/flow/vlm.py, triton/flow/vlm.py)."""

    @staticmethod
    def _mrope_segments(seq_len: int, img_span: Tuple[int, int],
                        aud_span: Tuple[int, int],
                        vis_grid_llm: Optional[Tuple[int, int, int]]):
        """Segment layout for the M-RoPE builder (R30 mirror): text runs
        between the modality spans, the visual span as a t/h/w grid, the
        audio span text-like on the three planes."""
        spans = []
        if img_span[1] > img_span[0]:
            if vis_grid_llm is None:
                raise RuntimeError(
                    "ZERO FALLBACK: a visual span needs its llm grid "
                    "(t, h/merge, w/merge) for the M-RoPE builder.")
            spans.append((img_span[0], img_span[1], "visual", vis_grid_llm))
        if aud_span[1] > aud_span[0]:
            spans.append((aud_span[0], aud_span[1], "text", None))
        spans.sort(key=lambda s: s[0])
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int, float]]]] = []
        cursor = 0
        for lo, hi, kind, grid in spans:
            if lo > cursor:
                segments.append(("text", lo - cursor, None))
            if kind == "visual":
                gt, gh, gw = grid
                segments.append(("visual", hi - lo, (gt, gh, gw, 1.0)))
            else:
                segments.append(("text", hi - lo, None))
            cursor = hi
        if cursor < seq_len:
            segments.append(("text", seq_len - cursor, None))
        return segments

    def _get_embed_weight(self, comp_name: str) -> Optional[NBXTensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return None
        for key in executor._weights:
            if ("embed_tokens" in key or "token_embed" in key
                    or key == "embed.weight"):
                return executor._weights[key]
        # Second pass (R30 mirror) — the NeuroTax substring rule
        # ("embed" in key) with the declared vocab_size as the
        # discriminator: a rank-2 weight whose rows equal vocab_size IS
        # the token embedding (e.g. `word_embeddings.weight`). Additive:
        # builds resolved by the exact names above never reach it.
        vocab = (self.ctx.pkg.defaults.get("lm_config", {}) or {}).get(
            "vocab_size")
        if vocab is None:
            return None
        for key, tensor in executor._weights.items():
            if ("embed" in key and tensor is not None
                    and getattr(tensor, "ndim", 0) == 2
                    and int(tensor.shape[0]) == int(vocab)):
                return tensor
        return None


class AudioSteps:
    """The steps the compiled `AudioEngine` and the Triton `TritonAudioEngine` ran alike (core/flow/audio.py, triton/flow/audio.py)."""

    def _preprocess_input(self, input_config: Dict, direction: str, stages: List) -> None:
        """Preprocess input based on modality and preprocessing type."""
        modality = input_config.get("modality", "audio" if direction == "stt" else "text")

        if modality == "audio":
            self._preprocess_audio_input(input_config, stages)
        elif modality == "text":
            self._preprocess_text_input(input_config)

    def _postprocess_output(self, output_config: Dict, direction: str) -> None:
        """Postprocess output based on modality."""
        modality = output_config.get("modality", "text" if direction == "stt" else "audio")

        if modality == "text":
            self._postprocess_text_output(output_config)
        elif modality == "audio":
            self._postprocess_audio_output(output_config)

    def _postprocess_text_output(self, output_config: Dict) -> None:
        """Decode generated token IDs to text (STT)."""
        generated_ids = self.ctx.variable_resolver.resolved.get("global.generated_token_ids")
        if generated_ids is None:
            return

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
        else:
            text = str(generated_ids)

        variable = output_config.get("variable", "global.transcription")
        self.ctx.variable_resolver.resolved[variable] = text
        self.ctx.variable_resolver.resolved["global.transcription"] = text
        logging.getLogger(__name__).debug(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")


# The RNN-T joint's weights, NeuroTax name -> role (both branches' joint).
_JOINT_KEYS = {normalize_tensor_name_strict(vendor): role for vendor, role in (
    ("enc.weight", "enc_weight"), ("enc.bias", "enc_bias"),
    ("pred.weight", "dec_weight"), ("pred.bias", "dec_bias"),
    ("joint_net.2.weight", "out_weight"), ("joint_net.2.bias", "out_bias"))}


class AudioLLMSteps:
    """The steps the compiled `AudioLLMEngine` and the Triton `TritonAudioLLMEngine` ran alike (core/flow/audio_llm.py,
    triton/flow/audio_llm.py)."""

    def _get_embed_weight(self, comp_name: str):
        """The token-embedding weight [vocab, hidden] (each branch's tensor), from the LLM or its embed_tokens component."""
        # 1. Check inside the LLM component
        executor = self.ctx.executors.get(comp_name)
        if executor is not None:
            for key in executor._weights:
                if "embed_tokens" in key or "token_embed" in key:
                    return executor._weights[key]
        # 2. Check separate embed_tokens component
        embed_executor = self.ctx.executors.get("embed_tokens")
        if embed_executor is not None:
            self._ensure_weights_loaded("embed_tokens")
            for key in embed_executor._weights:
                if key == "weight" or "embed" in key:
                    w = embed_executor._weights[key]
                    if w.ndim == 2 and w.shape[0] > w.shape[1]:
                        return w
        return None

    def _get_last_forward_output(self, forward_stages: List[Dict]):
        """Get output from the last forward stage (projector output)."""
        for s in reversed(forward_stages):
            output = self._get_component_output(s["component"])
            if output is not None:
                return output
        return None


class RNNTSteps:
    """The steps the compiled `RNNTEngine` and the Triton `TritonRNNTEngine` ran alike (core/flow/rnnt.py, triton/flow/rnnt.py)."""

    def _get_component_output(self, comp_name: str):
        """Get the primary output tensor from a component."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.output"]:
            if key in resolved:
                return resolved[key]
        executor = self.ctx.executors.get(comp_name)
        if executor and hasattr(executor, '_last_outputs') and executor._last_outputs:
            return executor._last_outputs[0]
        raise RuntimeError(f"ZERO FALLBACK: No output found for {comp_name}")

    def _find_tokenizer_model(self) -> Optional[Path]:
        """Find sentencepiece tokenizer.model in NBX cache."""
        cache_path = self.ctx.pkg.cache_path
        if cache_path is None:
            return None
        sp_paths = list(cache_path.rglob("*.model"))
        return sp_paths[0] if sp_paths else None

    def _bind_window(self, features, length) -> None:
        # Bind under all possible keys the executor might look for (each branch's own tensors)
        for key in ["global.audio_signal", "global.input_features", "audio_signal",
                     "input::audio_signal", "encoder.audio_signal"]:
            self.ctx.variable_resolver.resolved[key] = features
        for key in ["global.length", "length", "input::length", "encoder.length"]:
            self.ctx.variable_resolver.resolved[key] = length


def joint_role(key: str) -> Optional[str]:
    """The joint-network role of a weight key (the key itself, or the key under a component prefix)."""
    for canon, role in _JOINT_KEYS.items():
        if key == canon or key.endswith("." + canon):
            return role
    return None


class StaticGraphSteps:
    """The steps the compiled `StaticGraphHandler` and the Triton `TritonStaticGraphHandler` ran alike (core/flow/static_graph.py, triton/flow/static_graph.py)."""

    def execute(self) -> Dict[str, Any]:
        """
        Execute static graph (no loop).

        Executes all components in order: pre_loop + loop.components + post_loop

        Returns:
            Dict of resolved variables/outputs
        """
        flow = self.ctx.pkg.topology.get("flow", {})

        # Execute all components in topological order
        all_components = (
            flow.get("pre_loop", []) +
            flow.get("loop", {}).get("components", []) +
            flow.get("post_loop", [])
        )

        for comp_name in all_components:
            self._execute_component(comp_name, "static", None)

            # Unload after each component
            self._unload_component(comp_name)

        return self.ctx.variable_resolver.resolve_all()


# The sampling a decode runs with, read alike by every LLM-headed handler of both branches (audio_llm, encoder_decoder,
# tts_llm, vlm; Triton audio_llm, encoder_decoder).
SAMPLING_PARAMS = ("temperature", "top_k", "top_p",
                   "repetition_penalty", "min_p")


def effective_sampling(ctx, defaults):
    """Effective sampling config, and which parts the USER asked for.

    The registry's `defaults` is only half the story: a CLI flag or a
    serving request lands as a `global.*` variable in the resolver, and
    the autoregressive flow's own override table reads it from there.
    A guard that inspected `defaults` alone would miss exactly the case
    it exists to catch — whisper carries no `top_k` at all, so
    `--top-k 20` is invisible in `defaults` and visible only here.

    Returns (config, explicit) where config is defaults overlaid with
    the overrides, and explicit names the overridden parameters.
    """
    config = {k: defaults.get(k) for k in SAMPLING_PARAMS}
    explicit = set()
    resolver = getattr(ctx, "variable_resolver", None)
    if resolver is None:
        return config, explicit
    for name in SAMPLING_PARAMS:
        try:
            val = resolver.get(f"global.{name}", None)
        except Exception:
            continue
        if val is None:
            continue
        config[name] = val
        # EXPLICIT means the value DIFFERS from the registry default,
        # not merely that the resolver holds one. The configuration
        # pipeline populates `global.*` from defaults too, so presence
        # alone classifies every inherited field as a user request — the
        # first version did exactly that and refused MiniCPM-o for a
        # top_k=20 nobody had asked for. Re-requesting the default value
        # lands in the inherited branch, which is harmless: it is the
        # same value that was going to be ignored either way.
        # The default cascade has two rungs: the model's defaults.json and
        # the family's defaults (config/families/<family>.yml). A value
        # equal to EITHER is inherited — the family rung populates
        # `global.*` for a model whose defaults carry no such key
        # (audio_llm top_k=50, 2026-09-05: every model of the family refused).
        inherited = [defaults.get(name), family_default(ctx, name)]
        same = False
        for ref in inherited:
            if ref is None:
                continue
            try:
                same = same or float(val) == float(ref)
            except (TypeError, ValueError):
                same = same or val == ref
        if not same:
            explicit.add(name)
    return config, explicit


def family_default(ctx, name):
    """The family-level default of a sampling parameter, or None."""
    try:
        family = (getattr(getattr(ctx, "pkg", None), "manifest", None) or {}).get("family")
        if not family:
            return None
        from neurobrix.core.config.loader import get_family_config
        return ((get_family_config(family) or {}).get("defaults") or {}).get(name)
    except Exception:
        return None


def require_contract(flow: str, block: Dict[str, Any], key: str, where: str):
    """The value `key` of a topology.flow.<flow> contract block, or a refusal naming it (the image_gen and speech
    legs of both branches)."""
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.{flow} is missing '{key}' "
            f"({where}) — re-emit the {flow} contract from the registry.")
    return val


# next_token_diffusion, both branches
def _require_default(defaults: Dict[str, Any], key: str) -> Any:
    """ZERO FALLBACK read of a semantic key from the .nbx defaults.json (both next_token_diffusion handlers).

    These keys encode model semantics (guidance scale, latent scale/bias,
    scheduler shape); a silently-invented value yields plausible-but-wrong
    audio. Absent key = malformed/incomplete .nbx → crash naming the key
    (engine audit #2 2026-07-05, P-ZERO-FALLBACK-SWEEP — the sibling
    encoder_decoder flow is the reference pattern).
    """
    val = defaults.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: '{key}' missing from defaults.json — "
            f"next_token_diffusion refuses to invent a value; rebuild the "
            f".nbx with '{key}' in its runtime defaults."
        )
    return val


def encode_nopad(tok, text: str, add_special: bool = False) -> List[int]:
    """The tokenizer's ids for `text`, unpadded, whichever encode signature it offers.

    The runtime HFTokenizer pads to model_max_length by default; the prompt is assembled piece by piece, so each
    piece must be encoded unpadded."""
    ids = None
    try:
        ids = tok.encode(text, add_special_tokens=add_special, padding=False)
    except TypeError:
        try:
            ids = tok.encode(text, add_special_tokens=add_special)
        except TypeError:
            ids = tok.encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


# encoder_decoder, both branches
def _maybe_log_topk(row, step: int, engine: str, stage: str) -> None:
    """NBX_DECODE_TOPK=<jsonl>: per-step top-4 ids/values + top-2 margin
    of one logits row (numpy), `stage` = "raw" (model logits) or
    "rules" (after the vendor logit rules). Same record shape as the
    vlm flows so engine records pair by (step, stage). Default-off."""
    import os as _os
    path = _os.environ.get("NBX_DECODE_TOPK")
    if not path:
        return
    import json as _json
    import numpy as _np
    r = _np.asarray(row, dtype=_np.float64).reshape(-1)
    top = _np.argsort(-r)[:4]
    vals = [float(r[i]) for i in top]
    rec = {"engine": engine, "stage": stage, "step": int(step),
           "ids": [int(i) for i in top], "vals": [round(v, 6) for v in vals],
           "margin12": round(vals[0] - vals[1], 6)}
    with open(path, "a") as f:
        _json.dump(rec, f)
        f.write("\n")


# The Triton dual_ar and tts_llm handlers' tensor test (an NBXTensor, or anything shaped and typed).
def _is_tensor(val) -> bool:
    """Check if val is any tensor type."""
    return isinstance(val, NBXTensor) or (hasattr(val, 'shape') and hasattr(val, 'dtype'))
