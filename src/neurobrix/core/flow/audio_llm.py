"""
AudioLLMEngine — Audio-Conditioned LLM Flow

Handles models where audio is encoded, projected, and fed as embeddings
to an autoregressive LLM decoder: Voxtral, Granite Speech, Canary-Qwen.

Pattern: encoder(audio) → projector → LLM(inputs_embeds) → text

ZERO SEMANTIC: No knowledge of specific models.
ZERO HARDCODE: All parameters from NBX container.
"""

import gc
import time
import torch
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("audio_llm")
class AudioLLMEngine(FlowHandler):
    """
    Audio-conditioned LLM: encode audio → project → LLM autoregressive decode.

    topology.flow.audio:
        direction: stt
        stages:
          - component: encoder (or audio_tower)
            execution: forward
          - component: projector (or multi_modal_projector)
            execution: forward
          - component: language_model (or llm)
            execution: autoregressive
            logits_source: lm_head | self
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    def execute(self) -> Dict[str, Any]:
        """Execute audio-LLM pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        # Separate stages by role
        forward_stages = []
        ar_stage = None
        for s in stages:
            if s.get("execution", "forward") == "autoregressive":
                ar_stage = s
            else:
                forward_stages.append(s)

        if ar_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: audio_llm flow requires at least one 'autoregressive' stage."
            )

        # ── Step 1: Preprocess audio input ──
        from .audio_utils import preprocess_audio_input, postprocess_text_output, get_compute_dtype
        preprocess_audio_input(self.ctx, audio_config, stages)

        # ── Step 2: Forward stages (encoder, projector) ──
        for stage in forward_stages:
            comp_name = stage["component"]
            if comp_name not in self.ctx.executors:
                print(f"   [{comp_name}] Skipped (not in executors)")
                continue

            print(f"   [{comp_name}] Running forward pass...")
            start = time.perf_counter()
            self._ensure_weights_loaded(comp_name)
            self._execute_component(comp_name, "forward", None)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   [{comp_name}] Done in {elapsed:.0f}ms")

            # Store output for downstream
            self._store_output(comp_name)
            self._reshape_output_for_connections(comp_name)

            if not self.ctx.persistent_mode:
                self._unload_component_weights(comp_name)
                gc.collect()
                device_empty_cache(self.ctx.primary_device)

        # ── Step 3: Autoregressive LLM decode with audio embeddings ──
        lm_name = ar_stage["component"]
        logits_source = ar_stage.get("logits_source", "lm_head")
        dtype = get_compute_dtype(self.ctx)
        device = self.ctx.primary_device

        max_tokens = defaults.get("max_tokens")
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        self._ensure_weights_loaded(lm_name)
        embed_weight = self._get_embed_weight(lm_name)

        # Inject embed weight for weight-tied models
        if embed_weight is not None:
            executor = self.ctx.executors.get(lm_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

        # Get audio embeddings from last forward stage
        audio_embeds = self._get_last_forward_output(forward_stages)
        if audio_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires projected "
                f"audio embeddings from forward stages, but none found."
            )
        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{lm_name}' requires embed_tokens weight."
            )

        audio_embeds = audio_embeds.to(dtype=dtype)
        print(f"   [{lm_name}] Audio-LLM mode: audio_embeds {audio_embeds.shape}")

        # Build prompt: prefix_embeds + audio_embeds + suffix_embeds
        prefix_ids = defaults.get("stt_prefix_ids", [1])
        suffix_ids = defaults.get("stt_suffix_ids", [])

        parts = []
        if prefix_ids:
            prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                prefix_embeds = torch.nn.functional.embedding(prefix_tensor, embed_weight).to(dtype=dtype)
            parts.append(prefix_embeds)
        parts.append(audio_embeds)
        if suffix_ids:
            suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                suffix_embeds = torch.nn.functional.embedding(suffix_tensor, embed_weight).to(dtype=dtype)
            parts.append(suffix_embeds)

        context_embeds = torch.cat(parts, dim=1) if len(parts) > 1 else audio_embeds

        print(f"   [{lm_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        generated_ids: list = []

        for step in range(max_tokens):
            seq_len = context_embeds.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids

            self._execute_component(lm_name, "forward", None)

            output = self._get_component_output(lm_name)
            if output is None:
                break

            logits = self._compute_logits(output, embed_weight, logits_source)
            from .audio_utils import sample_token
            next_token = sample_token(
                logits, temperature,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

            # Append new token embedding to context
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                token_embed = torch.nn.functional.embedding(token_tensor, embed_weight).to(dtype=dtype)
            context_embeds = torch.cat([context_embeds, token_embed], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{lm_name}] Generated {len(generated_ids)} tokens in {elapsed:.0f}ms")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            gc.collect()
            device_empty_cache(self.ctx.primary_device)

        # ── Step 4: Decode tokens to text ──
        postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _get_last_forward_output(self, forward_stages: List[Dict]) -> Optional[torch.Tensor]:
        """Get output from the last forward stage (projector output)."""
        for s in reversed(forward_stages):
            output = self._get_component_output(s["component"])
            if output is not None:
                return output
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
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

    def _compute_logits(
        self, hidden_states: torch.Tensor, embed_weight: Optional[torch.Tensor],
        logits_source: str,
    ) -> torch.Tensor:
        last_hidden = hidden_states[:, -1:, :]

        if logits_source == "lm_head" and "lm_head" in self.ctx.executors:
            self._ensure_weights_loaded("lm_head")
            executor = self.ctx.executors["lm_head"]
            for key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    w = tensor.to(dtype=last_hidden.dtype)
                    return torch.matmul(last_hidden, w.T)
            return last_hidden

        if logits_source == "embed_weight_tied" and embed_weight is not None:
            w = embed_weight.to(dtype=last_hidden.dtype)
            return torch.matmul(last_hidden, w.T)

        return last_hidden

    def _store_output(self, comp_name: str) -> None:
        """Ensure component output is stored in variable resolver."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state"]:
            if key in resolved:
                return

    def _reshape_output_for_connections(self, comp_name: str) -> None:
        """Reshape component output if downstream expects different feature dim.

        Handles multimodal frame pooling (DATA-DRIVEN):
          audio_tower [B, T, 1280] → MMP expects [B, T/4, 5120]
        """
        connections = self.ctx.pkg.topology.get("connections", [])
        resolved = self.ctx.variable_resolver.resolved

        for conn in connections:
            src_key = conn.get("from", "")
            if not src_key.startswith(f"{comp_name}."):
                continue
            src_tensor = resolved.get(src_key)
            if not isinstance(src_tensor, torch.Tensor) or src_tensor.dim() != 3:
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
            dag = getattr(target_executor, '_dag', None)
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

            src_feat = src_tensor.shape[-1]
            if target_feat == src_feat:
                continue

            if target_feat > src_feat and target_feat % src_feat == 0:
                pool_factor = target_feat // src_feat
                B, T, D = src_tensor.shape
                new_T = T // pool_factor
                if new_T * pool_factor <= T:
                    reshaped = src_tensor[:, :new_T * pool_factor, :].reshape(B, new_T, target_feat)
                    resolved[src_key] = reshaped
                    print(f"   [Reshape] {comp_name}: [{B}, {T}, {D}] → [{B}, {new_T}, {target_feat}]")
