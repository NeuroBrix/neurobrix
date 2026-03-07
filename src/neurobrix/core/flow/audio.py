"""
AudioEngine — Universal Audio Flow Handler

Handles ALL audio models (STT + TTS) with ZERO model-specific code.
All behavior driven by topology.json flow.audio section.

Topology schema:
    flow.audio.direction: "stt" | "tts"
    flow.audio.input: {modality, preprocessing, variable}
    flow.audio.output: {modality, variable}
    flow.audio.stages[]: {component, execution, cross_attention_from, logits_source, ...}

ZERO SEMANTIC: No knowledge of "Whisper", "Kokoro", etc.
ZERO HARDCODE: All parameters from NBX container.
"""

import gc
import time
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import FlowHandler, FlowContext, register_flow


@register_flow("audio")
class AudioEngine(FlowHandler):
    """
    Universal audio flow handler for STT and TTS models.

    Reads flow.audio from topology.json and executes stages mechanically:
    1. Input preprocessing (audio→features or text→tokens)
    2. Stage execution (forward or autoregressive per stage)
    3. Output postprocessing (tokens→text or waveform→file)
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Any],
        resolve_inputs_fn: Callable[[str], Dict[str, Any]],
        ensure_weights_fn: Callable[[str], None],
        unload_weights_fn: Callable[[str], None],
    ):
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    def execute(self) -> Dict[str, Any]:
        """Execute the audio pipeline from topology.json flow.audio."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio")
        if audio_config is None:
            raise RuntimeError(
                "ZERO FALLBACK: Audio flow requires topology.flow.audio section.\n"
                "Re-build the model with updated topology schema."
            )

        direction = audio_config.get("direction", "stt")
        stages = audio_config.get("stages", [])
        if not stages:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.stages is empty.\n"
                "At least one stage is required."
            )

        # ── Step 1: Input preprocessing ──
        input_config = audio_config.get("input", {})
        self._preprocess_input(input_config, direction, stages)

        # ── Step 2: Execute stages in order ──
        for stage in stages:
            comp_name = stage["component"]
            execution = stage.get("execution", "forward")

            if comp_name not in self.ctx.executors:
                raise RuntimeError(
                    f"ZERO FALLBACK: Stage component '{comp_name}' not found in executors.\n"
                    f"Available: {list(self.ctx.executors.keys())}"
                )

            if execution == "forward":
                self._execute_forward_stage(stage)
            elif execution == "autoregressive":
                self._execute_autoregressive_stage(stage, audio_config)
            else:
                raise RuntimeError(
                    f"ZERO FALLBACK: Unknown execution type '{execution}' "
                    f"for stage '{comp_name}'. Expected: forward, autoregressive"
                )

        # ── Step 3: Output postprocessing ──
        output_config = audio_config.get("output", {})
        self._postprocess_output(output_config, direction)

        return self.ctx.variable_resolver.resolve_all()

    # ─────────────────────────────────────────────────────────────
    # Input preprocessing
    # ─────────────────────────────────────────────────────────────

    def _preprocess_input(self, input_config: Dict, direction: str, stages: List) -> None:
        """Preprocess input based on modality and preprocessing type."""
        modality = input_config.get("modality", "audio" if direction == "stt" else "text")

        if modality == "audio":
            self._preprocess_audio_input(input_config, stages)
        elif modality == "text":
            self._preprocess_text_input(input_config)

    def _preprocess_audio_input(self, input_config: Dict, stages: List) -> None:
        """Load audio file and extract features. Input shape read from graph (DATA-DRIVEN)."""
        audio_path = self.ctx.variable_resolver.resolved.get("global.audio_path")
        if audio_path is None:
            raise RuntimeError(
                "ZERO FALLBACK: Audio model requires global.audio_path.\n"
                "Use --audio <path> to provide an audio file."
            )

        preprocessing = input_config.get("preprocessing", "mel_spectrogram")
        variable = input_config.get("variable", "global.input_features")
        device = torch.device(self.ctx.primary_device)
        dtype = self._get_compute_dtype()

        # Read expected input shape from first stage's graph (DATA-DRIVEN)
        first_comp = stages[0]["component"] if stages else None
        input_shape = self._get_component_input_shape(first_comp)

        # Auto-correct preprocessing type from graph input shape when topology is wrong
        # [1, mel_bins, frames] with mel_bins in {40,64,80,128} → mel_spectrogram
        # [1, frames, feat_dim] with feat_dim > 80 → conformer
        if input_shape and len(input_shape) >= 3:
            dim1, dim2 = input_shape[1], input_shape[2]
            if preprocessing == "raw_waveform":
                if dim1 in (40, 64, 80, 128) and dim2 > dim1:
                    preprocessing = "mel_spectrogram"
                elif dim2 in (40, 64, 80, 128, 160, 256) and dim1 > dim2:
                    preprocessing = "conformer"

        print(f"   [Audio] Loading: {audio_path}")
        if input_shape:
            print(f"   [Audio] Expected input shape: {input_shape}")

        from neurobrix.core.module.audio.input_processor import AudioInputProcessor
        features = AudioInputProcessor.process(
            preprocessing_type=preprocessing,
            audio_path=str(audio_path),
            model_path=self._find_snapshot_path(),
            device=device,
            dtype=dtype,
            input_shape=input_shape,
        )

        # Pad/truncate to match trace-time dimensions for encoders
        # with non-symbolic position ops (relative PE, repeat, etc.)
        if input_shape and len(input_shape) == len(features.shape) and len(input_shape) >= 3:
            for dim_idx in range(1, len(input_shape)):  # Skip batch dim
                trace_size = input_shape[dim_idx]
                actual_size = features.shape[dim_idx]
                if actual_size != trace_size:
                    if actual_size > trace_size:
                        slices = [slice(None)] * len(features.shape)
                        slices[dim_idx] = slice(None, trace_size)
                        features = features[tuple(slices)]
                    else:
                        pad_shape = list(features.shape)
                        pad_shape[dim_idx] = trace_size - actual_size
                        pad = torch.zeros(pad_shape, device=features.device, dtype=features.dtype)
                        features = torch.cat([features, pad], dim=dim_idx)

        print(f"   [Audio] Features: {features.shape} ({preprocessing})")

        # Bind to variable resolver
        self.ctx.variable_resolver.resolved[variable] = features
        # Also bind without prefix for connection resolution
        short_key = variable.split(".")[-1] if "." in variable else variable
        self.ctx.variable_resolver.resolved[short_key] = features

    def _preprocess_text_input(self, input_config: Dict) -> None:
        """Tokenize text prompt for TTS/LLM-audio models."""
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None:
            raise RuntimeError(
                "ZERO FALLBACK: TTS model requires global.prompt.\n"
                "Use --prompt <text> to provide text input."
            )

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

        # Auto-detect tokenization style from components (DATA-DRIVEN)
        tokenization = input_config.get("tokenization", "auto")
        if tokenization == "auto":
            has_lm = any(
                k in self.ctx.executors
                for k in ["language_model", "model", "lm_head"]
            )
            tokenization = "llm" if has_lm else "diffusion"

        device = self.ctx.primary_device

        if tokenization == "llm":
            # LLM-style: direct encode (Orpheus, Voxtral TTS, etc.)
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
            except TypeError:
                # Custom tokenizer without return_tensors support
                ids = tokenizer.encode(prompt)
                input_ids = torch.tensor([ids], dtype=torch.long)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Diffusion-style: TextProcessor handles padding/truncation
            from neurobrix.core.module.text.processor import TextProcessor
            tp = TextProcessor(
                tokenizer=tokenizer,
                defaults=self.ctx.pkg.defaults,
                topology=self.ctx.pkg.topology,
                variable_resolver=self.ctx.variable_resolver,
            )
            input_ids, attention_mask = tp.tokenize_for_diffusion(prompt, device)

        self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
        self.ctx.variable_resolver.resolved["input_ids"] = input_ids
        if attention_mask is not None:
            self.ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
            self.ctx.variable_resolver.resolved["attention_mask"] = attention_mask

    # ─────────────────────────────────────────────────────────────
    # Stage execution
    # ─────────────────────────────────────────────────────────────

    def _execute_forward_stage(self, stage: Dict) -> None:
        """Execute a single forward-pass stage (encoder, projector, vocoder, etc.)."""
        comp_name = stage["component"]
        print(f"   [{comp_name}] Running forward pass...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)
        self._execute_component(comp_name, "forward", None)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Done in {elapsed:.0f}ms")

        # Store output for downstream stages
        self._store_stage_output(comp_name)

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _execute_autoregressive_stage(self, stage: Dict, audio_config: Dict) -> None:
        """Execute an autoregressive generation stage (decoder, LM)."""
        comp_name = stage["component"]
        device = self.ctx.primary_device
        defaults = self.ctx.pkg.defaults

        max_tokens = defaults.get("max_tokens", 448)
        temperature = defaults.get("temperature", 0.0)

        # Get special tokens from topology extracted_values (DATA-DRIVEN)
        eos_token_id = self._get_extracted_value("eos_token_id", 50257)
        decoder_start_token_id = self._get_extracted_value("decoder_start_token_id", 50258)

        # Detect architecture: does the graph take inputs_embeds or input_ids?
        uses_inputs_embeds = self._component_has_input(comp_name, "inputs_embeds")

        # Cross-attention binding (encoder-decoder models like Whisper)
        cross_from = stage.get("cross_attention_from")
        if cross_from:
            encoder_output = self._get_component_output(cross_from)
            if encoder_output is not None:
                self.ctx.variable_resolver.resolved[f"{cross_from}.output_0"] = encoder_output

        # Logits source
        logits_source = stage.get("logits_source", "self")

        print(f"   [{comp_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)

        # Get embed weight for:
        # 1. weight-tied logits (Whisper: proj_out = embed_tokens.T)
        # 2. audio-LLM embedding merge (need embed_tokens to convert text → embeddings)
        embed_weight = self._get_embed_weight(comp_name)

        if uses_inputs_embeds:
            self._run_audio_llm_autoregressive(
                comp_name, stage, audio_config, embed_weight,
                max_tokens, temperature, eos_token_id, logits_source,
            )
        else:
            self._run_encoder_decoder_autoregressive(
                comp_name, embed_weight, max_tokens, temperature,
                eos_token_id, decoder_start_token_id, logits_source,
            )

        elapsed = (time.perf_counter() - start) * 1000
        generated_ids = self.ctx.variable_resolver.resolved.get("global.generated_token_ids", [])
        print(f"   [{comp_name}] Generated {len(generated_ids)} tokens in {elapsed:.0f}ms")

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _run_encoder_decoder_autoregressive(
        self, comp_name: str, embed_weight: Optional[torch.Tensor],
        max_tokens: int, temperature: float,
        eos_token_id: int, decoder_start_token_id: int, logits_source: str,
    ) -> None:
        """Autoregressive decode for encoder-decoder models (Whisper).
        Graph takes input_ids + cross-attention from encoder output."""
        device = self.ctx.primary_device

        # Forced decoder IDs from generation config
        forced_decoder_ids = self._get_forced_decoder_ids()
        forced_map = {pos: tid for pos, tid in forced_decoder_ids}

        generated_ids = [decoder_start_token_id]

        for step in range(1, max_tokens):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            self._execute_component(comp_name, "forward", None)

            decoder_output = self._get_component_output(comp_name)
            if decoder_output is None:
                break

            logits = self._compute_logits(decoder_output, embed_weight, logits_source)

            current_pos = len(generated_ids)
            if current_pos in forced_map and forced_map[current_pos] is not None:
                next_token = forced_map[current_pos]
            else:
                next_token = self._sample_token(logits, temperature)

            generated_ids.append(next_token)
            if next_token == eos_token_id:
                break

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

    def _run_audio_llm_autoregressive(
        self, comp_name: str, stage: Dict, audio_config: Dict,
        embed_weight: Optional[torch.Tensor],
        max_tokens: int, temperature: float,
        eos_token_id: int, logits_source: str,
    ) -> None:
        """Autoregressive decode for audio-LLM models (Voxtral, Granite Speech).

        Uses growing-context approach: at each step, the full sequence
        [audio_embeds, generated_token_embeds...] is fed through the model.
        No KV cache needed — the graph takes inputs_embeds and outputs logits.
        O(n²) per step but acceptable for STT output lengths (~100-500 tokens).
        """
        device = self.ctx.primary_device
        dtype = self._get_compute_dtype()

        audio_embeds = self._get_previous_stage_output(stage, audio_config)
        if audio_embeds is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{comp_name}' requires projected "
                f"audio embeddings from a previous stage, but none found."
            )

        if embed_weight is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Audio-LLM stage '{comp_name}' requires embed_tokens "
                f"weight for text→embedding conversion, but not found."
            )

        print(f"   [{comp_name}] Audio-LLM mode: audio_embeds {audio_embeds.shape}")

        audio_embeds = audio_embeds.to(dtype=dtype)
        generated_ids: list = []

        # Build prompt: prefix_embeds + audio_embeds + suffix_embeds
        # Prefix/suffix token IDs from defaults (set by builder from model config)
        prefix_ids = self.ctx.pkg.defaults.get("stt_prefix_ids", [1])  # default: [BOS]
        suffix_ids = self.ctx.pkg.defaults.get("stt_suffix_ids", [])

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

        for step in range(max_tokens):
            seq_len = context_embeds.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids

            self._execute_component(comp_name, "forward", None)

            output = self._get_component_output(comp_name)
            if output is None:
                break

            logits = self._compute_logits(output, embed_weight, logits_source)
            next_token = self._sample_token(logits, temperature)
            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

            # Append new token embedding to context for next step
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                token_embed = torch.nn.functional.embedding(token_tensor, embed_weight).to(dtype=dtype)

            context_embeds = torch.cat([context_embeds, token_embed], dim=1)

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

    # ─────────────────────────────────────────────────────────────
    # Output postprocessing
    # ─────────────────────────────────────────────────────────────

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
        print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")

    def _postprocess_audio_output(self, output_config: Dict) -> None:
        """Store waveform output for TTS."""
        variable = output_config.get("variable", "global.output_audio")
        # The final stage's output tensor is the waveform
        # It's already in the variable resolver from stage execution
        if variable not in self.ctx.variable_resolver.resolved:
            # Try to find any tensor output from the last stage
            for key, val in self.ctx.variable_resolver.resolved.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 1:
                    self.ctx.variable_resolver.resolved[variable] = val
                    break

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _store_stage_output(self, comp_name: str) -> None:
        """Store a component's output in the variable resolver for downstream stages."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
            if key in resolved:
                return  # Already stored by executor
        # If not stored, the executor's connection system should have handled it

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get a component's primary output tensor."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [
            f"{comp_name}.output_0",
            f"{comp_name}.last_hidden_state",
            f"{comp_name}.output",
        ]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _get_embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get embedding weight for weight-tied logits or text→embedding conversion."""
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return None
        for key in executor._weights:
            if "embed_tokens" in key or "token_embed" in key:
                return executor._weights[key]
        return None

    def _component_has_input(self, comp_name: str, input_name: str) -> bool:
        """Check if a component's graph declares a specific input name."""
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return False
        dag = getattr(executor, '_dag', None)
        if dag is None:
            return False
        for _tid, spec in dag.get("tensors", {}).items():
            if spec.get("input_name") == input_name:
                return True
        return False

    def _compute_logits(
        self, hidden_states: torch.Tensor, embed_weight: Optional[torch.Tensor],
        logits_source: str,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
        last_hidden = hidden_states[:, -1:, :]
        if logits_source == "embed_weight_tied" and embed_weight is not None:
            w = embed_weight.to(dtype=last_hidden.dtype)
            return torch.matmul(last_hidden, w.T)
        # logits_source == "self": output IS logits
        return last_hidden

    def _sample_token(self, logits: torch.Tensor, temperature: float) -> int:
        """Sample next token from logits."""
        if temperature == 0.0:
            return logits[:, -1, :].argmax(dim=-1).item()
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _get_previous_stage_output(self, stage: Dict, audio_config: Dict) -> Optional[torch.Tensor]:
        """Get output from the stage immediately before this one."""
        stages = audio_config.get("stages", [])
        comp_name = stage["component"]
        for i, s in enumerate(stages):
            if s["component"] == comp_name and i > 0:
                prev_comp = stages[i - 1]["component"]
                return self._get_component_output(prev_comp)
        return None

    def _get_extracted_value(self, key: str, default: Any = None) -> Any:
        """Get a value from topology extracted_values."""
        extracted = self.ctx.pkg.topology.get("extracted_values", {})
        for section in extracted.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        # Also check generation_config in defaults
        return self.ctx.pkg.defaults.get(key, default)

    def _get_forced_decoder_ids(self) -> List:
        """Get forced decoder IDs from generation config in snapshot."""
        snapshot_path = self._find_snapshot_path()
        gen_config_path = snapshot_path / "generation_config.json"
        if gen_config_path.exists():
            import json
            with open(gen_config_path) as f:
                gen = json.load(f)
            forced = gen.get("forced_decoder_ids", [])
            if forced:
                return [(pos, tid) for pos, tid in forced]
        return []

    def _find_snapshot_path(self) -> Path:
        """Find the HF snapshot path for preprocessor/generation config."""
        model_name = self.ctx.pkg.manifest.get("model_name", "")
        snapshot_path = Path(f"/home/mlops/hf_snapshots/{model_name}")
        if snapshot_path.exists():
            return snapshot_path

        nbx_path = Path(self.ctx.nbx_path_str)
        if (nbx_path / "modules" / "tokenizer").exists():
            return nbx_path / "modules" / "tokenizer"

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot find snapshot path for {model_name}. "
            f"Expected at {snapshot_path}"
        )

    def _get_component_input_shape(self, comp_name: Optional[str]) -> Optional[Tuple[int, ...]]:
        """Read first input tensor shape from component's graph.json (DATA-DRIVEN)."""
        if comp_name is None:
            return None
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return None
        dag = getattr(executor, '_dag', None)
        if dag is None:
            return None
        for tid, spec in dag.get("tensors", {}).items():
            # Match input tensors by: type field, input_name field, or tensor ID prefix
            is_input = (
                spec.get("type") == "input"
                or spec.get("input_name") is not None
                or tid.startswith("input::")
            )
            if is_input:
                shape = spec.get("shape", [])
                # Resolve any symbolic dims to trace_value for shape hint
                resolved = []
                for dim in shape:
                    if isinstance(dim, dict):
                        resolved.append(dim.get("trace_value", dim.get("trace", 0)))
                    elif isinstance(dim, int):
                        resolved.append(dim)
                    else:
                        resolved.append(0)
                return tuple(resolved)
        return None

    def _get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype from manifest."""
        dtype_str = self.ctx.pkg.manifest.get("dtype", "float16")
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            dtype_str, torch.float16
        )
