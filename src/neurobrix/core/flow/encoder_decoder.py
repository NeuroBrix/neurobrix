"""
EncoderDecoderEngine — Encoder-Decoder with Cross-Attention Flow

Handles models like Whisper: encoder processes input features,
decoder generates tokens autoregressively with cross-attention
from encoder output.

ZERO SEMANTIC: No knowledge of "Whisper" or "speech".
ZERO HARDCODE: All parameters from NBX container.
"""

import time
import torch
from neurobrix.core.device_utils import device_multinomial
from neurobrix.core.memory.manager import release_flow_memory
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("encoder_decoder")
class EncoderDecoderEngine(FlowHandler):
    """
    Encoder-decoder cross-attention flow.

    topology.flow.audio:
        direction: stt
        stages:
          - component: model.encoder
            execution: forward
          - component: model.decoder
            execution: autoregressive
            cross_attention_from: model.encoder
            logits_source: embed_weight_tied | self | lm_head
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
        """Execute encoder-decoder pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        # ── Step 1: Preprocess audio input ──
        from .audio_utils import preprocess_audio_input, postprocess_text_output
        preprocess_audio_input(self.ctx, audio_config, stages)

        # ── Step 2: Forward encoder ──
        encoder_stage = None
        decoder_stage = None
        for s in stages:
            exec_type = s.get("execution", "forward")
            if exec_type == "forward":
                encoder_stage = s
            elif exec_type == "autoregressive":
                decoder_stage = s

        if encoder_stage is None or decoder_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: encoder_decoder flow requires one 'forward' stage "
                "(encoder) and one 'autoregressive' stage (decoder)."
            )

        enc_name = encoder_stage["component"]
        dec_name = decoder_stage["component"]

        print(f"   [{enc_name}] Running encoder...")
        start = time.perf_counter()
        self._ensure_weights_loaded(enc_name)
        self._execute_component(enc_name, "forward", None)
        enc_elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{enc_name}] Done in {enc_elapsed:.0f}ms")

        # Store encoder output for cross-attention
        encoder_output = self._get_component_output(enc_name)
        if encoder_output is not None:
            self.ctx.variable_resolver.resolved[f"{enc_name}.output_0"] = encoder_output
        _prog0 = __import__("os").environ.get("NBX_DECODE_PROGRESS")
        if _prog0 and encoder_output is not None:  # gated diagnostic — compiled encoder ref
            try:
                import numpy as _np
                _eo = encoder_output.detach().float().cpu().numpy()
                with open(_prog0, "w") as _pf:
                    _pf.write(f"ENCODER shape={_eo.shape} l2={float(_np.linalg.norm(_eo)):.3f} "
                              f"mean={float(_eo.mean()):.5f} std={float(_eo.std()):.5f} "
                              f"nan={bool(_np.isnan(_eo).any())} "
                              f"head={_np.round(_eo.flatten()[:6],4).tolist()}\n")
                    _pf.flush()
            except Exception:
                pass

        if not self.ctx.persistent_mode:
            self._unload_component_weights(enc_name)
            release_flow_memory(self.ctx.primary_device)

        # ── Step 3: Autoregressive decode with cross-attention ──
        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        max_tokens = decode_bound(defaults.get("max_tokens"))
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        decoder_start_token_id = defaults.get("decoder_start_token_id", eos_token_id)
        logits_source = decoder_stage.get("logits_source", "embed_weight_tied")
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        # Forced decoder IDs (language/task tokens for Whisper)
        forced_decoder_ids = defaults.get("forced_decoder_ids", [])
        forced_map = {pos: tid for pos, tid in forced_decoder_ids}

        print(f"   [{dec_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        self._ensure_weights_loaded(dec_name)

        # Get embed weight for weight-tied logits
        embed_weight = self._get_embed_weight(dec_name)

        # Inject embed weight for weight-tied models (LoRA breaks data_ptr tying)
        if embed_weight is not None:
            executor = self.ctx.executors.get(dec_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

        device = self.ctx.primary_device
        generated_ids = [decoder_start_token_id]

        for step in range(1, max_tokens):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            self._execute_component(dec_name, "forward", None)

            decoder_output = self._get_component_output(dec_name)
            if decoder_output is None:
                break

            logits = self._compute_logits(decoder_output, embed_weight, logits_source)

            current_pos = len(generated_ids)
            if current_pos in forced_map and forced_map[current_pos] is not None:
                next_token = forced_map[current_pos]
            else:
                next_token = self._sample_token(
                    logits, temperature,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                )

            generated_ids.append(next_token)
            _prog = __import__("os").environ.get("NBX_DECODE_PROGRESS")
            if _prog:  # gated diagnostic — compiled decoder ref trajectory
                _stat = ""
                try:
                    import numpy as _np
                    _do = decoder_output.detach().float().cpu().numpy()
                    _lt = _do.reshape(-1, _do.shape[-1])[-1]
                    _stat = (f" dec_l2={float(_np.linalg.norm(_lt)):.4f}"
                             f" dec_mean={float(_lt.mean()):.5f}"
                             f" dec_head={_np.round(_lt[:4],4).tolist()}")
                except Exception as _e:
                    _stat = f" (dec-stat err: {_e})"
                with open(_prog, "a") as _pf:
                    _pf.write(f"step={step} last={next_token}{_stat}\n")
                    _pf.flush()
            if next_token == eos_token_id:
                break

        dec_elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{dec_name}] Generated {len(generated_ids)} tokens in {dec_elapsed:.0f}ms")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(dec_name)
            release_flow_memory(self.ctx.primary_device)

        # ── Step 4: Decode tokens to text ──
        postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get a component's primary output tensor."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get TOKEN embedding weight for weight-tied logits.

        NeuroTax standard: token_embed.weight (token embeddings).
        Must NOT match embed_positions.weight (positional embeddings)
        which has a different shape and purpose.
        """
        executor = self.ctx.executors.get(comp_name)
        if executor is not None:
            # Priority: exact token_embed match first
            for key in executor._weights:
                if "token_embed" in key:
                    return executor._weights[key]
            # Fallback: largest 2D embed weight (vocab_size > pos_table)
            best = None
            for key in executor._weights:
                if "embed" in key and executor._weights[key].ndim == 2:
                    w = executor._weights[key]
                    if best is None or w.shape[0] > best.shape[0]:
                        best = w
            return best
        return None

    def _compute_logits(
        self, hidden_states: torch.Tensor, embed_weight: Optional[torch.Tensor],
        logits_source: str,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
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

    def _sample_token(
        self, logits: torch.Tensor, temperature: float,
        generated_ids: Optional[List[int]] = None,
        repetition_penalty: float = 1.0,
    ) -> int:
        """Sample next token from logits."""
        last_logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                if last_logits[0, tid] > 0:
                    last_logits[0, tid] /= repetition_penalty
                else:
                    last_logits[0, tid] *= repetition_penalty

        if temperature == 0.0:
            return last_logits.argmax(dim=-1).item()
        probs = torch.softmax(last_logits / temperature, dim=-1)
        return device_multinomial(probs, 1).item()
