"""
AudioEngine — Orchestrator for Kokoro-82M and VibeVoice-1.5B audio flows.

Dispatches stage execution to model-specific handlers in stages/.
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
from neurobrix.core.device_utils import device_empty_cache


@register_flow("audio")
class AudioEngine(FlowHandler):
    """
    Audio flow orchestrator for Kokoro-82M and VibeVoice-1.5B.

    Reads flow.audio from topology.json and executes stages mechanically:
    1. Input preprocessing (audio->features or text->tokens)
    2. Stage execution (forward, native_kokoro, diffusion, native_acoustic_decoder)
    3. Output postprocessing (tokens->text or waveform->file)
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

        direction = audio_config.get("direction")
        if direction is None:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.direction is required ('stt' or 'tts')."
            )
        stages = audio_config.get("stages", [])
        if not stages:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.stages is empty.\n"
                "At least one stage is required."
            )

        # -- Step 1: Input preprocessing --
        input_config = audio_config.get("input", {})
        self._preprocess_input(input_config, direction, stages)

        # -- Step 2: Execute stages in order --
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
            elif execution == "native_kokoro":
                from .stages.kokoro import execute_native_kokoro
                execute_native_kokoro(self, stage, audio_config)
            elif execution == "diffusion":
                from .stages.vibevoice import execute_diffusion_stage
                execute_diffusion_stage(self, stage, audio_config)
            elif execution == "native_acoustic_decoder":
                from .stages.vibevoice import execute_native_acoustic_decoder
                execute_native_acoustic_decoder(self, stage, audio_config)
            else:
                raise RuntimeError(
                    f"ZERO FALLBACK: Unknown execution type '{execution}' "
                    f"for stage '{comp_name}'. Expected: forward, native_kokoro, "
                    f"diffusion, native_acoustic_decoder"
                )

        # -- Step 3: Output postprocessing --
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

        preprocessing = input_config.get("preprocessing")
        if preprocessing is None:
            raise RuntimeError(
                "ZERO FALLBACK: topology.flow.audio.input.preprocessing is required.\n"
                "Types: mel_spectrogram, nemo_mel, conformer, raw_waveform, dac_codec, text_only"
            )
        variable = input_config.get("variable", "global.input_features")
        device = torch.device(self.ctx.primary_device)
        dtype = self._get_compute_dtype()

        # Read expected input shape from first stage's graph (DATA-DRIVEN)
        first_comp = stages[0]["component"] if stages else None
        input_shape = self._get_component_input_shape(first_comp)

        # Auto-correct preprocessing type from graph input shape when topology is wrong
        # [1, mel_bins, frames] with mel_bins in {40,64,80,128} -> mel_spectrogram
        # [1, frames, feat_dim] with feat_dim > 80 -> conformer
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
            model_path=self._find_model_config_path(),
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

        # Bind audio length for models that need it (NeMo Conformer, etc.)
        # The actual audio length (before padding) is the feature frame count
        actual_frames = features.shape[-1] if preprocessing in ("mel_spectrogram", "nemo_mel") else features.shape[1]
        length_tensor = torch.tensor([actual_frames], dtype=torch.long, device=features.device)
        self.ctx.variable_resolver.resolved["global.audio_signal_length"] = length_tensor
        self.ctx.variable_resolver.resolved["audio_signal_length"] = length_tensor
        self.ctx.variable_resolver.resolved["global.length"] = length_tensor
        self.ctx.variable_resolver.resolved["length"] = length_tensor

    def _preprocess_text_input(self, input_config: Dict) -> None:
        """Tokenize text prompt for TTS/LLM-audio models."""
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None:
            raise RuntimeError(
                "ZERO FALLBACK: TTS model requires global.prompt.\n"
                "Use --prompt <text> to provide text input."
            )

        # Apply TTS prompt template if configured (DATA-DRIVEN from defaults.json)
        tts_template = self.ctx.pkg.defaults.get("tts_prompt_template")
        if tts_template and "{text}" in tts_template:
            prompt = tts_template.format(text=prompt)

        tokenizer = self.ctx.modules.get("tokenizer")

        # Phonemizer path: models like Kokoro use espeak-ng phonemes -> IDs
        # instead of a standard tokenizer. Vocab stored in defaults.json.
        phoneme_vocab = self.ctx.pkg.defaults.get("phoneme_vocab")
        if tokenizer is None and phoneme_vocab:
            from .stages.kokoro import preprocess_phonemizer_input
            preprocess_phonemizer_input(self, prompt, phoneme_vocab)
            return

        if tokenizer is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

        # Auto-detect tokenization style from components (DATA-DRIVEN)
        tokenization = input_config.get("tokenization", "auto")
        if tokenization == "auto":
            has_lm = any(
                k in self.ctx.executors or any(
                    ek.endswith(f".{k}") or ek == k for ek in self.ctx.executors
                )
                for k in ["language_model", "model", "lm_head"]
            )
            # Also detect autoregressive stages -- they use LLM-style tokenization
            if not has_lm:
                flow = self.ctx.pkg.topology.get("flow", {})
                stages = flow.get("audio", {}).get("stages", [])
                has_lm = any(s.get("execution") == "autoregressive" for s in stages)
            tokenization = "llm" if has_lm else "diffusion"

        device = self.ctx.primary_device

        if tokenization == "llm":
            # LLM-style: direct encode (Orpheus, Voxtral TTS, etc.)
            # Use add_special_tokens=False when template already includes BOS
            add_special = tts_template is None
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                             add_special_tokens=add_special)
            except TypeError:
                ids = tokenizer.encode(prompt, add_special_tokens=add_special)
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

        # Check if required inputs are available AND are tensors.
        # TTS pipelines may have optional stages (e.g., voice cloning reference
        # encoder expects audio waveform tensor -- skipped in text-only mode
        # where only string prompt is available).
        has_tensor_input = False
        comp_connections = self.ctx.connections_index.get(comp_name, {})
        if comp_connections:
            for _input_name, sources in comp_connections.items():
                for src in sources:
                    val = self.ctx.variable_resolver.resolved.get(src)
                    if isinstance(val, torch.Tensor):
                        has_tensor_input = True
                        break
                if has_tensor_input:
                    break
        else:
            # No connections defined: check if graph inputs can be resolved
            # from variable resolver (input_name or global.input_name)
            executor = self.ctx.executors.get(comp_name)
            dag = getattr(executor, '_dag', None) if executor else None
            if dag:
                for _tid, spec in dag.get("tensors", {}).items():
                    iname = spec.get("input_name")
                    if iname:
                        for key in [iname, f"global.{iname}", f"{comp_name}.{iname}"]:
                            val = self.ctx.variable_resolver.resolved.get(key)
                            if isinstance(val, torch.Tensor):
                                has_tensor_input = True
                                break
                    if has_tensor_input:
                        break
        if not has_tensor_input:
            print(f"   [{comp_name}] Skipped (no tensor inputs available)")
            return

        print(f"   [{comp_name}] Running forward pass...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)

        # Check if input needs chunking (e.g., codec.decoder expects fixed seq_len)
        chunked = self._try_chunked_forward(comp_name)
        if not chunked:
            self._execute_component(comp_name, "forward", None)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Done in {elapsed:.0f}ms")

        # Store output for downstream stages
        self._store_stage_output(comp_name)

        # Reshape output if next stage expects different feature dim
        # Handles multimodal pooling (e.g., audio_tower [B,T,1280] -> MMP [B,T/4,5120])
        self._reshape_output_for_connections(comp_name)

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            device_empty_cache(self.ctx.primary_device)

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
        """Process TTS output: decode audio tokens or store raw waveform."""
        variable = output_config.get("variable", "global.output_audio")
        device = self.ctx.primary_device

        # Check if output is audio token IDs or raw waveform
        # Detection is DATA-DRIVEN: defaults.json audio_output_type and audio_token_start
        defaults = self.ctx.pkg.defaults
        audio_output_type = defaults.get("audio_output_type")
        generated_ids = self.ctx.variable_resolver.resolved.get("global.generated_token_ids")

        if audio_output_type == "snac_tokens" and generated_ids and isinstance(generated_ids, list):
            audio_token_start = defaults.get("audio_token_start")
            if audio_token_start is None:
                raise RuntimeError(
                    "ZERO FALLBACK: audio_token_start missing from defaults.json.\n"
                    "Required for SNAC token decoding."
                )
            vocab_size = defaults.get("vocab_size")
            if vocab_size is None:
                raise RuntimeError(
                    "ZERO FALLBACK: vocab_size missing from defaults.json.\n"
                    "Required for SNAC token decoding."
                )
            audio_count = sum(1 for t in generated_ids if t >= audio_token_start)
            print(f"   [TTS] Tokens: {len(generated_ids)}, audio: {audio_count}")

            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            waveform = AudioOutputProcessor.decode_snac_tokens(
                generated_ids, vocab_size=vocab_size, device=device,
            )
            if waveform.numel() > 1:
                self.ctx.variable_resolver.resolved[variable] = waveform
                model_name = self.ctx.pkg.manifest.get("model_name", "model")
                sample_rate = self._get_sample_rate()
                output_path = f"output_{model_name}.wav"
                AudioOutputProcessor.save_waveform(waveform, output_path, sample_rate=sample_rate)
                print(f"\n{'='*70}\nSAVED: {output_path}\n{'='*70}")
                return

        # Raw waveform: find output tensor from last stage (Kokoro, VibeVoice, etc.)
        waveform = None
        for key in ["decoder.output_0", "global.output_audio"]:
            val = self.ctx.variable_resolver.resolved.get(key)
            if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[-1] > 1000:
                waveform = val
                break

        if waveform is None:
            for val in self.ctx.variable_resolver.resolved.values():
                if isinstance(val, torch.Tensor) and val.dim() == 3 and val.shape[1] == 1:
                    if val.shape[2] > 1000:
                        waveform = val
                        break

        if waveform is not None:
            self.ctx.variable_resolver.resolved[variable] = waveform
            model_name = self.ctx.pkg.manifest.get("model_name", "model")
            sample_rate = self._get_sample_rate()
            output_path = f"output_{model_name}.wav"
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            AudioOutputProcessor.save_waveform(waveform, output_path, sample_rate=sample_rate)
            print(f"\n{'='*70}\nSAVED: {output_path}\n{'='*70}")

    # ─────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────

    def _try_chunked_forward(self, comp_name: str) -> bool:
        """Run chunked forward if input seq_len exceeds graph's trace-time seq_len.

        Used for codec.decoder: graph expects [1, 1024, 64] but input may be [1, 1024, T_gen].
        Chunks input into 64-frame blocks, runs each, concatenates waveform output.
        Returns True if chunking was performed.
        """
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return False

        dag = getattr(executor, '_dag', None)
        if dag is None:
            return False

        # Find graph input spec
        graph_input_name = None
        graph_seq_len = None
        for _tid, spec in dag.get("tensors", {}).items():
            iname = spec.get("input_name")
            if iname:
                shape = spec.get("shape", [])
                if len(shape) == 3:
                    graph_seq_len = shape[2]
                    if isinstance(graph_seq_len, dict):
                        graph_seq_len = graph_seq_len.get("trace_value", graph_seq_len)
                    graph_input_name = iname
                break

        if graph_input_name is None or not isinstance(graph_seq_len, int):
            return False

        # Find the actual input tensor
        resolved = self.ctx.variable_resolver.resolved
        actual_input = None
        for key in [f"global.{graph_input_name}", graph_input_name]:
            val = resolved.get(key)
            if isinstance(val, torch.Tensor) and val.dim() == 3:
                actual_input = val
                break

        # Also check connections
        if actual_input is None:
            connections = self.ctx.pkg.topology.get("connections", [])
            for conn in connections:
                target = conn.get("to", "")
                if target == f"{comp_name}.{graph_input_name}":
                    src_key = conn.get("from", "")
                    val = resolved.get(src_key)
                    if isinstance(val, torch.Tensor) and val.dim() == 3:
                        actual_input = val
                        break

        if actual_input is None:
            return False

        actual_seq = actual_input.shape[2]
        if actual_seq <= graph_seq_len:
            return False  # Fits in single pass, no chunking needed

        # Chunk and execute
        print(f"   [{comp_name}] Chunked: {actual_seq} frames -> {graph_seq_len}-frame blocks")
        waveform_chunks = []

        for chunk_start in range(0, actual_seq, graph_seq_len):
            chunk_end = min(chunk_start + graph_seq_len, actual_seq)
            chunk = actual_input[:, :, chunk_start:chunk_end]

            # Pad last chunk if needed
            if chunk.shape[2] < graph_seq_len:
                pad_size = graph_seq_len - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            comp_inputs = {graph_input_name: chunk}
            output = executor.run(comp_inputs)

            if isinstance(output, dict):
                out_tensor = next(iter(output.values()))
            elif isinstance(output, torch.Tensor):
                out_tensor = output
            else:
                out_tensor = self._get_component_output(comp_name)

            if out_tensor is not None:
                # If last chunk was padded, trim proportionally
                if chunk_end - chunk_start < graph_seq_len and out_tensor.dim() >= 2:
                    ratio = (chunk_end - chunk_start) / graph_seq_len
                    trim_len = int(out_tensor.shape[-1] * ratio)
                    out_tensor = out_tensor[..., :trim_len]
                waveform_chunks.append(out_tensor)

        if waveform_chunks:
            full_output = torch.cat(waveform_chunks, dim=-1)
            # Store as component output
            self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = full_output
            self.ctx.variable_resolver.resolved["global.output_audio"] = full_output
            print(f"   [{comp_name}] Waveform: {list(full_output.shape)}")

        return True

    def _reshape_output_for_connections(self, comp_name: str) -> None:
        """Reshape component output if downstream stage expects different feature dim.

        Handles multimodal frame pooling (DATA-DRIVEN):
          audio_tower [B, T, 1280] -> MMP expects [B, T/4, 5120]
          pool_factor = target_feat_dim / source_feat_dim

        Detects from topology connections + target graph input shapes.
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

            # Parse target component and input name
            target_str = conn.get("to", "")
            parts = target_str.split(".")
            if len(parts) < 2:
                continue
            target_comp = parts[0]
            target_input = ".".join(parts[1:])

            # Look up expected input shape from target graph
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
                continue  # Dims match, no reshape needed

            if target_feat > src_feat and target_feat % src_feat == 0:
                # Group consecutive frames: [B, T, D] -> [B, T/pool, D*pool]
                pool_factor = target_feat // src_feat
                B, T, D = src_tensor.shape
                new_T = T // pool_factor
                if new_T * pool_factor <= T:
                    reshaped = src_tensor[:, :new_T * pool_factor, :].reshape(B, new_T, target_feat)
                    resolved[src_key] = reshaped
                    print(f"   [Reshape] {comp_name}: [{B}, {T}, {D}] -> [{B}, {new_T}, {target_feat}] (pool={pool_factor})")

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

    def _get_sample_rate(self) -> int:
        """Get sample rate from topology or defaults. ZERO FALLBACK."""
        flow = self.ctx.pkg.topology.get("flow", {})
        sr = flow.get("audio", {}).get("sample_rate")
        if sr is not None:
            return sr
        sr = self.ctx.pkg.defaults.get("sample_rate")
        if sr is not None:
            return sr
        raise RuntimeError(
            "ZERO FALLBACK: sample_rate missing from both topology.flow.audio "
            "and defaults.json."
        )

    def _find_model_config_path(self) -> Path:
        """Find model config path from NBX container."""
        nbx_path = Path(self.ctx.nbx_path_str)
        # NBX container modules/tokenizer or modules/processor
        # (has preprocessor_config.json for audio models).
        for subdir in ["modules/tokenizer", "modules/processor"]:
            candidate = nbx_path / subdir
            if candidate.exists():
                return candidate
        # Removed: legacy absolute-path fallback. See the comment in
        # `core/flow/audio_utils.py:find_model_config_path` for the
        # rationale — the field held a trace-host snapshot location
        # and is no longer present in correctly-built containers.
        raise RuntimeError(
            "Cannot find model config path. Expected `modules/tokenizer/` "
            "or `modules/processor/` inside the .nbx. Re-import the model "
            "with the current builder which embeds these directories."
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
