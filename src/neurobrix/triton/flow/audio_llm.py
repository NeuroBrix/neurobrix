"""Triton AudioLLMEngine — zero-torch audio-conditioned LLM flow.

R33-pure port of core/flow/audio_llm.py (AudioLLMEngine). Pattern:
encoder(audio) → projector → LLM(inputs_embeds) → text.

All compute is NBXTensor + kernel wrappers; ZERO torch on the compute
path. Two sanctioned BOUNDARY helpers are imported from
core.flow.audio_utils — exactly as the other triton audio handlers do
(triton/flow/encoder_decoder.py, dual_ar.py): `preprocess_audio_input`
(mel/FFT, torch/torchaudio internally, output converted to NBXTensor
at the boundary) and `postprocess_text_output` (tokenizer decode of
the generated ids — pure text, no compute tensor). These are the same
torch-boundary subroutines the R33 doctrine explicitly allows for
audio I/O; the model forward, embedding, concat, logits and sampling
are all NBXTensor/Triton.

ZERO SEMANTIC: no model-specific knowledge. ZERO HARDCODE: all
parameters from the NBX container / defaults.json.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional

from neurobrix import decode_progress, decode_stop
from neurobrix.core.flow.shared_steps import effective_sampling as _effective_sampling   # both branches read the sampling alike
from neurobrix.core.flow.shared_steps import AudioLLMSteps   # the steps both branches share
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.memory_pool import release_flow_memory
from nbxtensor.ops import index_wrappers as _nbx_index_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.ops import linalg_wrappers as _nbx_linalg_wrappers  # the family the moved names live in (M2 split)
from neurobrix.core.flow.base import ComponentFlowHandler   # the executor's four component callables, both branches
import logging
_log = logging.getLogger(__name__)


_DTYPE_STR_TO_NBX = {
    "float16": NBXDtype.float16, "fp16": NBXDtype.float16,
    "bfloat16": NBXDtype.bfloat16, "bf16": NBXDtype.bfloat16,
    "float32": NBXDtype.float32, "fp32": NBXDtype.float32,
}


class TritonAudioLLMEngine(AudioLLMSteps, ComponentFlowHandler):
    """Audio-conditioned LLM (R33): encode audio → project → LLM
    autoregressive decode with audio embeddings. Mirror of
    core/flow/audio_llm.py AudioLLMEngine."""

    def _compute_dtype(self) -> NBXDtype:
        """Prism-plan resolved dtype (triton-side resolver), string → NBXDtype."""
        from neurobrix.triton.dtype import resolve_compute_dtype
        s = resolve_compute_dtype(self.ctx)
        return _DTYPE_STR_TO_NBX.get(s, NBXDtype.float16)

    def execute(self) -> Dict[str, Any]:
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        forward_stages: List[Dict] = []
        ar_stage = None
        for s in stages:
            if s.get("execution", "forward") == "autoregressive":
                ar_stage = s
            else:
                forward_stages.append(s)

        if ar_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: audio_llm flow requires at least one "
                "'autoregressive' stage."
            )

        # ── Step 1: Preprocess audio (zero-torch numpy front-end) ──
        from neurobrix.triton.audio_frontend import (
            preprocess_audio_input_np as preprocess_audio_input,
            postprocess_text_output_np as postprocess_text_output,
        )
        preprocess_audio_input(self.ctx, audio_config, stages)

        # ── Step 2: Forward stages (encoder, projector) ──
        # Long-form (D-AUDIOLLM-LONGFORM) — R30 mirror of the compiled flow:
        # every fixed 30 s window on ctx._audio_windows runs the unchanged
        # single-window stages; the last stage's outputs are concatenated
        # along the sequence axis (NBXTensor.cat, R33). One window = the
        # classic path, byte for byte.
        _windows = getattr(self.ctx, "_audio_windows", None) or [None]
        _audio_variable = audio_config.get("input", {}).get(
            "variable", "global.input_features")
        _window_outputs = []
        for _wi, _win in enumerate(_windows):
            if _win is not None:
                for _k in (_audio_variable, _audio_variable.split(".")[-1]):
                    self.ctx.variable_resolver.resolved[_k] = _win
                if len(_windows) > 1:
                    _log.debug(f"   [Audio] window {_wi + 1}/{len(_windows)}")
            for stage in forward_stages:
                comp_name = stage["component"]
                if comp_name not in self.ctx.executors:
                    _log.debug(f"   [{comp_name}] Skipped (not in executors)")
                    continue
                _log.debug(f"   [{comp_name}] Running forward pass...")
                start = time.perf_counter()
                self._ensure_weights_loaded(comp_name)
                self._execute_component(comp_name, "forward", None)
                _log.debug(f"   [{comp_name}] Done in "
                           f"{(time.perf_counter() - start) * 1000:.0f}ms")
                self._store_output(comp_name)
                self._reshape_output_for_connections(comp_name)
                if not self.ctx.persistent_mode and _wi == len(_windows) - 1:
                    self._unload_component_weights(comp_name)
                    release_flow_memory(self.ctx.primary_device)
            if len(_windows) > 1:
                _out = self._get_last_forward_output(forward_stages)
                if _out is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: long-form audio_llm window produced no "
                        "projector output.")
                _window_outputs.append(_out)
        if _window_outputs:
            _cat = NBXTensor.cat(_window_outputs, dim=1)
            _last = forward_stages[-1]["component"]
            self.ctx.variable_resolver.resolved[f"{_last}.output_0"] = _cat
            _log.debug(f"   [Audio] {len(_window_outputs)} windows → audio embeddings "
                       f"{tuple(_cat.shape)}")

        # ── Step 3: Autoregressive LLM decode with audio embeddings ──
        lm_name = ar_stage["component"]
        logits_source = ar_stage.get("logits_source", "lm_head")
        dtype = self._compute_dtype()

        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        # The request's own budget first (global.max_tokens from the CLI /
        # serve request), the container default after — the mirror of the
        # autoregressive generator's resolver cascade (R30). A 600 s
        # recording needs ~2,000 tokens; the default 448 truncated the
        # long-form transcript (2026-09-03 gate).
        _mt = self.ctx.variable_resolver.resolved.get("global.max_tokens")
        if _mt is None:
            _mt = self.ctx.variable_resolver.resolved.get("max_tokens")
        if _mt is None:
            _mt = defaults.get("max_tokens")
        max_tokens = decode_bound(_mt)
        if max_tokens is None:
            raise RuntimeError(
                "ZERO FALLBACK: max_tokens missing from defaults.json.")
        # The decode's sampler: the branch's one CombinedSampler, its parameters resolved the one way (request >
        # container > family file — core/module/sampling_contract.resolve_sampling); the request's --temperature,
        # --top-k, --top-p and --repetition-penalty are read, as every other decode reads them.
        from neurobrix.triton.samplers import CombinedSampler, sample_next_id
        from neurobrix.core.module.sampling_contract import sampler_config
        sampler = CombinedSampler(sampler_config(self.ctx.variable_resolver, defaults, "audio_llm (triton)"))
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError(
                "ZERO FALLBACK: eos_token_id missing from defaults.json.")

        # Sampling-parameter contract: what the sampler does not implement (min_p) is refused when asked, reported
        # when inherited (core/module/sampling_contract).
        from neurobrix.core.module.sampling_contract import (
            enforce_sampling_support, SAMPLER_IMPLEMENTS)
        _samp_cfg, _samp_explicit = _effective_sampling(
            self.ctx, defaults)
        enforce_sampling_support(
            "audio_llm (triton)", SAMPLER_IMPLEMENTS, _samp_cfg, explicit=_samp_explicit)

        self._ensure_weights_loaded(lm_name)
        embed_weight = self._get_embed_weight(lm_name)

        # Inject embed weight for weight-tied models (mirror core).
        if embed_weight is not None:
            executor = self.ctx.executors.get(lm_name)
            if executor is not None and hasattr(executor, "_weights"):
                dag = getattr(executor, "_dag", None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied in ("head.weight",
                                 "model.token_embed.weight"):
                        if (tied not in executor._weights
                                and f"param::{tied}" in tensors):
                            executor._weights[tied] = embed_weight

        audio_embeds = self._get_last_forward_output(forward_stages)
        if audio_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires "
                f"projected audio embeddings from forward stages, but "
                f"none found.")
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires "
                f"embed_tokens weight.")

        audio_embeds = audio_embeds.to(dtype)
        _log.debug(f"   [{lm_name}] Audio-LLM mode: audio_embeds "
                   f"{tuple(audio_embeds.shape)}")

        # Build prompt: prefix_embeds + audio_embeds + suffix_embeds
        prefix_ids = defaults.get("stt_prefix_ids", [1])
        suffix_ids = defaults.get("stt_suffix_ids", [])

        parts: List[NBXTensor] = []
        if prefix_ids:
            parts.append(self._embed_ids(prefix_ids, embed_weight, dtype))
        parts.append(audio_embeds)
        if suffix_ids:
            parts.append(self._embed_ids(suffix_ids, embed_weight, dtype))
        context_embeds = (NBXTensor.cat(parts, dim=1)
                          if len(parts) > 1 else audio_embeds)

        _log.debug(f"   [{lm_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        generated_ids: List[int] = []

        from neurobrix.triton.flow.decoder_cache import DecodeSteps, open_decoder_cache
        _steps = DecodeSteps(open_decoder_cache(self.ctx, lm_name, max_tokens))
        for _step in range(max_tokens):
            seq_len = int(context_embeds.shape[1])
            # A cached step hands the LM the one new token at its absolute position seq_len - 1;
            # the prefill (and the recompute oracle) the whole context (triton/flow/decoder_cache).
            feed_embeds = _steps.feed(context_embeds, context_embeds.narrow(1, seq_len - 1, 1))
            position_ids = NBXTensor.from_numpy(
                np.arange(seq_len - int(feed_embeds.shape[1]), seq_len, dtype=np.int64).reshape(1, -1))

            res = self.ctx.variable_resolver.resolved
            res["global.inputs_embeds"] = feed_embeds
            res["inputs_embeds"] = feed_embeds
            res["global.position_ids"] = position_ids
            res["position_ids"] = position_ids

            _steps.before_run()
            self._execute_component(lm_name, "forward", None)
            _steps.after_run()

            output = self._get_component_output(lm_name)
            if output is None:
                break

            logits = self._compute_logits(
                output, embed_weight, logits_source)
            from neurobrix.triton.flow.decode_topk import log_topk
            log_topk(logits, _step)
            next_token = sample_next_id(sampler, logits, generated_ids)
            generated_ids.append(next_token)
            _done = next_token == eos_token_id or _step + 1 >= max_tokens
            # The stop seam (Studio request 3, decode_stop): ends the decode
            # here as if its length were reached. stdlib-only (R33).
            _stop = not _done and decode_stop.requested(len(generated_ids))
            decode_progress.record(_step, len(generated_ids), next_token, _done or _stop)
            if next_token == eos_token_id or _stop:
                break

            token_embed = self._embed_ids(
                [next_token], embed_weight, dtype)
            context_embeds = NBXTensor.cat(
                [context_embeds, token_embed], dim=1)

        _log.debug(f"   [{lm_name}] Generated {len(generated_ids)} tokens "
                   f"in {(time.perf_counter() - start) * 1000:.0f}ms")

        self.ctx.variable_resolver.resolved[
            "global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

        # ── Step 4: Decode tokens → text (BOUNDARY: tokenizer) ──
        postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers (NBXTensor mirrors of the core helpers) ──────────────

    def _embed_ids(self, ids: List[int], embed_weight: NBXTensor,
                   dtype: NBXDtype) -> NBXTensor:
        """F.embedding(ids, embed_weight) → NBXTensor [1, len(ids), H]."""
        idx = NBXTensor.from_numpy(
            np.array([ids], dtype=np.int64))
        emb = _nbx_index_wrappers.embedding(embed_weight, idx)
        return emb.to(dtype)

    def _get_component_output(self, comp_name: str) -> Optional[NBXTensor]:
        resolved = self.ctx.variable_resolver.resolved
        for key in (f"{comp_name}.output_0",
                    f"{comp_name}.last_hidden_state",
                    f"{comp_name}.output"):
            v = resolved.get(key)
            if isinstance(v, NBXTensor):
                return v
        return None

    def _compute_logits(
            self, hidden_states: NBXTensor,
            embed_weight: Optional[NBXTensor],
            logits_source: str) -> NBXTensor:
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, H]

        def _proj(weight: NBXTensor) -> NBXTensor:
            wt = weight.to(last_hidden.dtype)
            B, S, H = last_hidden.shape
            a2 = last_hidden.reshape(B * S, H)         # [1, H]
            wtT = wt.transpose(0, 1)                   # [H, V]
            out = _nbx_linalg_wrappers.matmul_wrapper(a2, wtT)            # [1, V]
            return out.reshape(B, S, out.shape[-1])    # [1, 1, V]

        if (logits_source == "lm_head"
                and "lm_head" in self.ctx.executors):
            self._ensure_weights_loaded("lm_head")
            executor = self.ctx.executors["lm_head"]
            for _key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    return _proj(tensor)
            return last_hidden

        if logits_source == "embed_weight_tied" and embed_weight is not None:
            return _proj(embed_weight)

        return last_hidden

    def _store_output(self, comp_name: str) -> None:
        resolved = self.ctx.variable_resolver.resolved
        for key in (f"{comp_name}.output_0",
                    f"{comp_name}.last_hidden_state"):
            if key in resolved:
                return

    def _reshape_output_for_connections(self, comp_name: str) -> None:
        """Multimodal frame pooling (DATA-DRIVEN), NBXTensor mirror of
        the core helper: audio_tower [B,T,D] → MMP-expected [B,T/p,D*p]."""
        connections = self.ctx.pkg.topology.get("connections", [])
        resolved = self.ctx.variable_resolver.resolved
        for conn in connections:
            src_key = conn.get("from", "")
            if not src_key.startswith(f"{comp_name}."):
                continue
            src = resolved.get(src_key)
            if not isinstance(src, NBXTensor) or src.ndim != 3:
                continue
            target_str = conn.get("to", "")
            parts = target_str.split(".")
            if len(parts) < 2:
                continue
            target_comp = parts[0]
            target_input = ".".join(parts[1:])
            target_executor = self.ctx.executors.get(target_comp)
            if target_executor is None:
                continue
            dag = getattr(target_executor, "_dag", None)
            if dag is None:
                continue
            target_feat = None
            for _tid, spec in dag.get("tensors", {}).items():
                if spec.get("input_name") == target_input:
                    shape = spec.get("shape", [])
                    if len(shape) >= 3:
                        feat = shape[-1]
                        if isinstance(feat, dict):
                            feat = feat.get("trace_value", feat)
                        if isinstance(feat, int):
                            target_feat = feat
                    break
            if target_feat is None:
                continue
            pooled = pooled_frames_shape(list(src.shape), target_feat)
            if pooled is None:
                continue
            B, T, D = src.shape
            new_T = pooled[1]
            reshaped = src[:, :new_T * (target_feat // D), :].reshape(B, new_T, target_feat)
            resolved[src_key] = reshaped
            _log.debug(f"   [Reshape] {comp_name}: [{B}, {T}, {D}] "
                       f"→ [{B}, {new_T}, {target_feat}]")


def pooled_frames_shape(src_shape, target_feat: int):
    """The frame pooling between an audio tower and a projector that expects `target_feat`
    features: [B, T, D] -> [B, T // p, D * p] with p = target_feat / D (the tail frames that do
    not fill a group dropped), or None when the widths already agree or do not divide. The flow
    reshapes by it; the derived census binds the projector with it."""
    if len(src_shape) != 3:
        return None
    B, T, D = src_shape
    if target_feat == D or not (target_feat > D and target_feat % D == 0):
        return None
    return [B, T // (target_feat // D), target_feat]
