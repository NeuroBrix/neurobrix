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
            elif execution == "native_kokoro":
                self._execute_native_kokoro_stage(stage, audio_config)
            elif execution == "diffusion":
                self._execute_diffusion_stage(stage, audio_config)
            elif execution == "native_acoustic_decoder":
                self._execute_native_acoustic_decoder(stage, audio_config)
            else:
                raise RuntimeError(
                    f"ZERO FALLBACK: Unknown execution type '{execution}' "
                    f"for stage '{comp_name}'. Expected: forward, autoregressive, native_kokoro, diffusion, native_acoustic_decoder"
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

        # Phonemizer path: models like Kokoro use espeak-ng phonemes → IDs
        # instead of a standard tokenizer. Vocab stored in defaults.json.
        phoneme_vocab = self.ctx.pkg.defaults.get("phoneme_vocab")
        if tokenizer is None and phoneme_vocab:
            self._preprocess_phonemizer_input(prompt, phoneme_vocab)
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

    def _preprocess_phonemizer_input(self, prompt: str, phoneme_vocab: Dict) -> None:
        """Convert text to phoneme IDs using espeak-ng + vocabulary mapping.

        Used by models like Kokoro that take IPA phoneme sequences instead
        of standard text tokens.
        """
        device = self.ctx.primary_device

        # Step 1: Phonemize text → IPA string
        # Try kokoro g2p first (most accurate for Kokoro models),
        # then phonemizer library, then espeak-ng subprocess as fallback.
        phonemes = None
        try:
            from kokoro import KPipeline
            pipe = KPipeline(lang_code=self.ctx.pkg.defaults.get("phoneme_lang", "a"))
            phonemes, _ = pipe.g2p(prompt)
        except Exception:
            pass

        if phonemes is None:
            try:
                from phonemizer import phonemize as ph_fn
                phonemes = ph_fn(
                    prompt, language='en-us', backend='espeak',
                    strip=True, preserve_punctuation=True, with_stress=True,
                )
            except Exception:
                pass

        if phonemes is None:
            # Last resort: subprocess espeak-ng
            import subprocess
            try:
                result = subprocess.run(
                    ['espeak-ng', '-q', '--ipa', prompt],
                    capture_output=True, text=True, timeout=10,
                )
                phonemes = result.stdout.strip()
            except Exception:
                raise RuntimeError(
                    "ZERO FALLBACK: Phonemizer model requires 'kokoro', "
                    "'phonemizer', or 'espeak-ng' CLI."
                )

        # Step 2: Map phonemes to IDs
        ids = [0]  # BOS/padding
        for ch in phonemes:
            if ch in phoneme_vocab:
                ids.append(phoneme_vocab[ch])
        ids.append(0)  # EOS/padding

        # Step 3: Pad/truncate to graph's expected length
        # Get expected length from first component's input shape
        first_stage = self.ctx.pkg.topology.get("flow", {}).get("audio", {}).get("stages", [])
        max_len = None
        if first_stage:
            first_comp = first_stage[0].get("component", "")
            executor = self.ctx.executors.get(first_comp)
            if executor and hasattr(executor, '_dag') and executor._dag:
                for _tid, tinfo in executor._dag.get("tensors", {}).items():
                    if tinfo.get("is_input") and tinfo.get("input_name") in ("input_ids", "input", "inp"):
                        shape = tinfo.get("shape", [])
                        if len(shape) >= 2:
                            max_len = shape[1]
                        break

        if max_len is None:
            raise RuntimeError(
                "ZERO FALLBACK: Cannot determine input_ids length from first stage graph.\n"
                "Expected a graph input named 'input_ids', 'input', or 'inp' with a 2D shape."
            )

        actual_len = len(ids)
        if len(ids) > max_len:
            ids = ids[:max_len]
            actual_len = max_len
        else:
            ids = ids + [0] * (max_len - len(ids))

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
        self.ctx.variable_resolver.resolved["input_ids"] = input_ids
        print(f"   [Phonemizer] '{prompt}' → {len(phonemes)} phonemes → {actual_len} IDs (padded to {max_len})")

        # Bind text length and mask for downstream stages (text_encoder, predictor)
        text_lengths = torch.tensor([actual_len], dtype=torch.long, device=device)
        for key in ["global.text_lengths", "text_lengths", "input_lengths"]:
            self.ctx.variable_resolver.resolved[key] = text_lengths

        text_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
        text_mask[0, :actual_len] = True
        for key in ["global.text_mask", "text_mask", "m"]:
            self.ctx.variable_resolver.resolved[key] = text_mask

        # Load voicepack for TTS models with voice packs
        self._load_voicepack(actual_len, device)

    # ─────────────────────────────────────────────────────────────
    # Stage execution
    # ─────────────────────────────────────────────────────────────

    def _execute_forward_stage(self, stage: Dict) -> None:
        """Execute a single forward-pass stage (encoder, projector, vocoder, etc.)."""
        comp_name = stage["component"]

        # Check if required inputs are available AND are tensors.
        # TTS pipelines may have optional stages (e.g., voice cloning reference
        # encoder expects audio waveform tensor — skipped in text-only mode
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
        self._execute_component(comp_name, "forward", None)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Done in {elapsed:.0f}ms")

        # Store output for downstream stages
        self._store_stage_output(comp_name)

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _execute_diffusion_stage(self, stage: Dict, audio_config: Dict) -> None:
        """Execute a diffusion denoising stage (VibeVoice prediction_head).

        Runs an iterative denoising loop using a scheduler module.
        Condition comes from a previously-executed component's output.
        Reuses existing scheduler infrastructure (DDIM/DPM++ from scheduler factory).

        Stage config expects:
            diffusion.num_inference_steps: int (default from defaults.json)
            diffusion.condition_from: str (component name for condition tensor)
            diffusion.latent_shape: [T, D] (noise shape, from graph input)
        """
        comp_name = stage["component"]
        device = self.ctx.primary_device
        defaults = self.ctx.pkg.defaults

        print(f"   [{comp_name}] Running diffusion denoising...")
        start = time.perf_counter()
        self._ensure_weights_loaded(comp_name)

        # ── Get diffusion config (ALL values from defaults.json, ZERO FALLBACK) ──
        diffusion_cfg = stage.get("diffusion", {})
        num_steps = diffusion_cfg.get("num_inference_steps",
                    defaults.get("ddpm_num_inference_steps"))
        if num_steps is None:
            raise RuntimeError(
                "ZERO FALLBACK: ddpm_num_inference_steps missing from defaults.json.\n"
                "Builder must extract from model's diffusion config."
            )

        # ── Get or create scheduler ──
        scheduler = self.ctx.modules.get("scheduler")
        if scheduler is None:
            from neurobrix.core.module.scheduler.factory import SchedulerFactory
            sched_config = defaults.get("scheduler_config", {})
            # All scheduler params MUST be in defaults.json
            for key, default_key in [
                ("_class_name", "scheduler_type"),
                ("num_train_timesteps", "ddpm_num_steps"),
                ("prediction_type", "prediction_type"),
                ("beta_schedule", "ddpm_beta_schedule"),
            ]:
                if key not in sched_config:
                    val = defaults.get(default_key)
                    if val is None:
                        raise RuntimeError(
                            f"ZERO FALLBACK: '{default_key}' missing from defaults.json.\n"
                            f"Builder must extract diffusion scheduler config from model."
                        )
                    sched_config[key] = val
            scheduler = SchedulerFactory.create(sched_config)

        scheduler.set_timesteps(num_steps, device=torch.device(device))

        # ── Determine latent shape from graph input ──
        executor = self.ctx.executors[comp_name]
        dag = getattr(executor, '_dag', None)
        latent_shape = None
        condition_input_name = None
        noisy_input_name = None
        if dag:
            for _tid, spec in dag.get("tensors", {}).items():
                iname = spec.get("input_name")
                if iname and "noisy" in iname:
                    noisy_input_name = iname
                    latent_shape = spec.get("shape", [])
                elif iname and "condition" in iname:
                    condition_input_name = iname

        if latent_shape is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Cannot determine latent shape from {comp_name} graph.\n"
                f"Expected a graph input with 'noisy' in its name."
            )

        # Resolve symbolic dims to concrete values
        concrete_shape = []
        for s in latent_shape:
            if isinstance(s, dict):
                tv = s.get("trace_value")
                if tv is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Symbolic dim in {comp_name} graph has no trace_value."
                    )
                concrete_shape.append(tv)
            elif isinstance(s, int):
                concrete_shape.append(s)
            else:
                raise RuntimeError(
                    f"ZERO FALLBACK: Unexpected dim type {type(s)} in {comp_name} graph."
                )

        # ── Get condition tensor from previous stage output ──
        condition_from = diffusion_cfg.get("condition_from")
        if condition_from is None:
            raise RuntimeError(
                "ZERO FALLBACK: Diffusion stage requires 'condition_from' in topology.\n"
                "Specifies which component provides the conditioning tensor."
            )
        condition = self._get_component_output(condition_from)
        if condition is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Condition source '{condition_from}' produced no output."
            )

        # Reshape condition to match graph expectation [1, T, hidden]
        if condition is not None and condition.dim() == 2:
            condition = condition.unsqueeze(0)
        # Ensure condition sequence length matches latent sequence length
        if condition is not None and len(concrete_shape) == 3:
            target_seq = concrete_shape[1]
            if condition.shape[1] != target_seq:
                if condition.shape[1] > target_seq:
                    condition = condition[:, :target_seq, :]
                else:
                    pad = torch.zeros(
                        condition.shape[0], target_seq - condition.shape[1], condition.shape[2],
                        device=condition.device, dtype=condition.dtype
                    )
                    condition = torch.cat([condition, pad], dim=1)

        # ── Initialize noise ──
        dtype = self._get_compute_dtype()
        noisy = torch.randn(concrete_shape, device=device, dtype=dtype)

        # ── Bind condition to variable resolver ──
        if condition is not None and condition_input_name:
            self.ctx.variable_resolver.resolved[f"{comp_name}.{condition_input_name}"] = condition.to(device, dtype=dtype)
            self.ctx.variable_resolver.resolved[condition_input_name] = condition.to(device, dtype=dtype)

        # ── Denoising loop ──
        # Call executor.run() directly with all inputs — _execute_component only
        # resolves from topology connections (condition) but noisy_images and
        # timesteps are runtime-generated per step.
        executor = self.ctx.executors[comp_name]
        print(f"   [{comp_name}] Diffusion: {num_steps} steps, latent {concrete_shape}")
        for step_idx, t in enumerate(scheduler.timesteps):
            if isinstance(t, torch.Tensor) and t.dim() == 0:
                t_input = t.unsqueeze(0).to(device)
            else:
                t_input = torch.tensor([t], device=device, dtype=torch.long)

            # Scale model input (identity for DDIM)
            scaled_noisy = scheduler.scale_model_input(noisy, t)

            # Build complete inputs dict for this step
            comp_inputs = {}
            if noisy_input_name:
                comp_inputs[noisy_input_name] = scaled_noisy
            comp_inputs["timesteps"] = t_input
            if condition is not None and condition_input_name:
                comp_inputs[condition_input_name] = condition.to(device, dtype=dtype)

            # Execute prediction head directly
            output = executor.run(comp_inputs)

            # Extract model output
            if isinstance(output, dict):
                model_output = next(iter(output.values()))
            elif isinstance(output, torch.Tensor):
                model_output = output
            else:
                model_output = self._get_component_output(comp_name)
            if model_output is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: Diffusion stage '{comp_name}' produced no output."
                )

            # Scheduler step
            step_result = scheduler.step(model_output, t, noisy)
            if isinstance(step_result, dict):
                noisy = step_result["prev_sample"]
            else:
                noisy = step_result.prev_sample

        # ── Apply speech scaling (VibeVoice-specific, DATA-DRIVEN from defaults) ──
        speech_scaling = defaults.get("speech_scaling_factor")
        speech_bias = defaults.get("speech_bias_factor")
        if speech_scaling is not None and speech_bias is not None:
            noisy = noisy / speech_scaling - speech_bias

        # ── Store denoised output ──
        self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = noisy
        self.ctx.variable_resolver.resolved["global.denoised_latent"] = noisy

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Diffusion done in {elapsed:.0f}ms")

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _execute_autoregressive_stage(self, stage: Dict, audio_config: Dict) -> None:
        """Execute an autoregressive generation stage (decoder, LM)."""
        comp_name = stage["component"]
        device = self.ctx.primary_device
        defaults = self.ctx.pkg.defaults

        max_tokens = defaults.get("max_tokens")
        if max_tokens is None:
            raise RuntimeError(
                "ZERO FALLBACK: max_tokens missing from defaults.json.\n"
                "Builder must set max_tokens from model's generation_config."
            )
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError(
                "ZERO FALLBACK: temperature missing from defaults.json.\n"
                "Builder must set temperature from model's generation_config."
            )

        # Get special tokens (DATA-DRIVEN from defaults.json)
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            eos_token_id = self._get_extracted_value("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError(
                "ZERO FALLBACK: eos_token_id missing from defaults.json.\n"
                "Builder must extract eos_token_id from model config."
            )
        decoder_start_token_id = defaults.get("decoder_start_token_id")
        if decoder_start_token_id is None:
            decoder_start_token_id = self._get_extracted_value("decoder_start_token_id")
        if decoder_start_token_id is None:
            # Not all models need decoder_start_token_id (only encoder-decoder like Whisper)
            decoder_start_token_id = eos_token_id

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

        # Weight tying: if graph expects head.weight or model.token_embed.weight
        # but they're missing from LLM weights (LoRA merge breaks data_ptr tying,
        # or embed_tokens is a separate component), inject the embed weight.
        if embed_weight is not None:
            executor = self.ctx.executors.get(comp_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

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
    # Native Kokoro predictor execution (RNNT pattern)
    # ─────────────────────────────────────────────────────────────

    def _load_voicepack(self, phoneme_count: int, device) -> None:
        """Load voice pack and split into predictor/decoder styles.

        Voicepacks are [N, 256] tensors stored in modules/voices/.
        Index by phoneme count, then split: [:128]=decoder, [128:]=predictor.
        """
        nbx_path = Path(self.ctx.nbx_path_str)
        voices_dir = nbx_path / "modules" / "voices"

        if not voices_dir.exists():
            return

        voice_name = self.ctx.pkg.defaults.get("voice", "af_heart")
        voice_path = voices_dir / f"{voice_name}.pt"

        if not voice_path.exists():
            voice_files = sorted(voices_dir.glob("*.pt"))
            if not voice_files:
                return
            voice_path = voice_files[0]
            voice_name = voice_path.stem

        voicepack = torch.load(voice_path, map_location=device, weights_only=True)

        if voicepack.dim() == 1:
            ref_s = voicepack.unsqueeze(0)
        elif voicepack.dim() == 2:
            idx = min(phoneme_count, voicepack.shape[0] - 1)
            ref_s = voicepack[idx:idx + 1]
        elif voicepack.dim() == 3:
            idx = min(phoneme_count, voicepack.shape[0] - 1)
            ref_s = voicepack[idx]
        else:
            vp_dim = voicepack.shape[-1]
            ref_s = voicepack.reshape(-1, vp_dim)[0:1]

        # Split voicepack: first half = decoder style, second half = predictor style
        # Dimension comes from the voicepack tensor shape itself (DATA-DRIVEN)
        split_at = ref_s.shape[-1] // 2
        style_dec = ref_s[:, :split_at]
        style_pred = ref_s[:, split_at:]

        for key in ["global.decoder_style", "decoder_style"]:
            self.ctx.variable_resolver.resolved[key] = style_dec
        for key in ["global.predictor_style", "predictor_style"]:
            self.ctx.variable_resolver.resolved[key] = style_pred

        print(f"   [Voicepack] Loaded '{voice_name}' (ref_s={ref_s.shape})")

    def _execute_native_kokoro_stage(self, stage: Dict, audio_config: Dict) -> None:
        """Native execution for Kokoro components that can't run through CompiledSequence.

        Handles:
        - text_encoder: embedding → WeightNorm Conv1d × 3 → BiLSTM (pack_padded_sequence)
        - predictor: BiLSTM + duration + F0/N + alignment
        """
        comp_name = stage["component"]

        # Sub-dispatch by topology field, NOT by component name string
        native_subtype = stage.get("native_subtype", "predictor")
        if native_subtype == "text_encoder":
            self._execute_native_text_encoder(comp_name)
            return

        device = self.ctx.primary_device
        dtype = self._get_compute_dtype()

        print(f"   [{comp_name}] Running native Kokoro predictor...")
        start = time.perf_counter()

        # Get inputs from previous stages (DATA-DRIVEN from topology inputs_from)
        inputs_from = stage.get("inputs_from", [])
        if len(inputs_from) < 2:
            raise RuntimeError(
                f"ZERO FALLBACK: native_kokoro predictor requires 'inputs_from' "
                f"with 2 component names in topology stage config."
            )
        bert_enc_out = self._get_component_output(inputs_from[0])
        if bert_enc_out is None:
            raise RuntimeError(f"ZERO FALLBACK: {inputs_from[0]} output not found.")
        d_en = bert_enc_out.transpose(-1, -2).to(device=device, dtype=dtype)  # [B, 512, T]

        text_enc_out = self._get_component_output(inputs_from[1])
        if text_enc_out is None:
            raise RuntimeError(f"ZERO FALLBACK: {inputs_from[1]} output not found.")
        t_en = text_enc_out.to(device=device, dtype=dtype)  # [B, 512, T]

        style_pred = self.ctx.variable_resolver.resolved.get("global.predictor_style")
        style_dec = self.ctx.variable_resolver.resolved.get("global.decoder_style")
        if style_pred is None or style_dec is None:
            raise RuntimeError("ZERO FALLBACK: Kokoro predictor requires voicepack styles.")

        style_pred = style_pred.to(device=device, dtype=dtype)  # [B, 128]
        style_dec = style_dec.to(device=device, dtype=dtype)

        text_mask = self.ctx.variable_resolver.resolved.get("global.text_mask")
        text_lengths = self.ctx.variable_resolver.resolved.get("global.text_lengths")

        self._ensure_weights_loaded(comp_name)
        w = dict(self.ctx.executors[comp_name]._weights)

        target_shapes = self._get_kokoro_decoder_shapes()
        target_asr_frames = target_shapes["asr_frames"]
        target_f0_len = target_shapes["f0_len"]
        target_n_len = target_shapes["n_len"]

        with torch.inference_mode():
            # ── Step 1: DurationEncoder (text_encoder.lstms) ──
            # Vendor: predictor.text_encoder(d_en, s, input_lengths, text_mask)
            T = d_en.shape[2]
            x = d_en.permute(2, 0, 1)  # [T, B, 512]
            s_exp = style_pred.unsqueeze(0).expand(T, -1, -1)  # [T, B, 128]
            x = torch.cat([x, s_exp], dim=-1)  # [T, B, 640]
            if text_mask is not None:
                x.masked_fill_(text_mask.unsqueeze(-1).transpose(0, 1).to(device), 0.0)
            x = x.transpose(0, 1)  # [B, T, 640]
            x = x.transpose(-1, -2)  # [B, 640, T]

            for layer_idx in range(6):
                prefix = f"text_encoder.lstms.{layer_idx}"
                lstm_key = f"{prefix}.weight_ih_l0"
                if lstm_key in w:
                    # LSTM layer
                    x = self._run_kokoro_single_lstm(
                        x, w, prefix, text_lengths, T, device, dtype
                    )
                elif f"{prefix}.proj.weight" in w:
                    # AdaLayerNorm layer
                    x = self._run_kokoro_adaln(
                        x, style_pred, w, prefix, device, dtype
                    )
                    # Re-concat style
                    s_ch = s_exp.permute(1, 2, 0)  # [B, 128, T]
                    if x.shape[2] < s_ch.shape[2]:
                        s_ch = s_ch[:, :, :x.shape[2]]
                    elif x.shape[2] > s_ch.shape[2]:
                        x = x[:, :, :s_ch.shape[2]]
                    x = torch.cat([x, s_ch], dim=1)  # [B, 640, T]
                    if text_mask is not None:
                        x.masked_fill_(text_mask.unsqueeze(1).to(device), 0.0)

            d = x.transpose(-1, -2)  # [B, T, 640] — DurationEncoder output

            # ── Step 2: Duration LSTM + projection ──
            # Vendor: x, _ = self.predictor.lstm(d)
            dur_lstm = self._build_bilstm(w, "lstm", 640, 256, device, dtype)
            d_dur, _ = dur_lstm(d)  # [B, T, 512]

            dur_w = w["dur_proj.linear_layer.weight"].to(device=device, dtype=dtype)
            dur_b = w["dur_proj.linear_layer.bias"].to(device=device, dtype=dtype)
            dur_logits = d_dur @ dur_w.T + dur_b  # [B, T, 50]

            speed = self.ctx.pkg.defaults.get("speed", 1.0)
            raw_durations = torch.sigmoid(dur_logits).sum(dim=-1) / speed  # [B, T]

            if text_mask is not None:
                inv_mask = (~text_mask).float().to(device=device)
                raw_durations = raw_durations * inv_mask

            durations = self._scale_kokoro_durations(raw_durations[0], target_asr_frames)

            # ── Step 3: Build alignment matrix ──
            num_phonemes = durations.shape[0]
            alignment = torch.zeros(1, num_phonemes, target_asr_frames, device=device, dtype=dtype)
            pos = 0
            for i in range(num_phonemes):
                dur_val = int(durations[i].item())
                if dur_val > 0 and pos < target_asr_frames:
                    end = min(pos + dur_val, target_asr_frames)
                    alignment[0, i, pos:end] = 1.0
                    pos = end

            # ── Step 4: Compute en and asr ──
            # Vendor: en = d.transpose(-1, -2) @ pred_aln_trg
            en = d.transpose(-1, -2) @ alignment  # [B, 640, T_frames]
            # Vendor: asr = t_en @ pred_aln_trg
            asr = t_en @ alignment  # [B, 512, T_frames]

            # ── Step 5: F0 and N prediction (F0Ntrain) ──
            # Vendor: self.shared(en.transpose(-1, -2)) → F0/N blocks
            shared_lstm = self._build_bilstm(w, "shared", 640, 256, device, dtype)
            shared_out, _ = shared_lstm(en.transpose(-1, -2))  # [B, T_frames, 512]
            shared_out = shared_out.transpose(-1, -2)  # [B, 512, T_frames]

            F0_raw = self._run_kokoro_f0n_blocks(shared_out, style_pred, w, "F0", device, dtype)
            N_raw = self._run_kokoro_f0n_blocks(shared_out, style_pred, w, "N", device, dtype)

            # Expand F0/N to decoder target shapes via interpolation
            F0_curve = torch.nn.functional.interpolate(
                F0_raw, size=target_f0_len, mode='linear', align_corners=False
            ).squeeze(1)  # [B, target_f0_len]
            N_curve = torch.nn.functional.interpolate(
                N_raw, size=target_n_len, mode='linear', align_corners=False
            ).squeeze(1)  # [B, target_n_len]

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Native done in {elapsed:.0f}ms  "
              f"asr={asr.shape} F0={F0_curve.shape} N={N_curve.shape}")

        # Bind all decoder inputs to variable resolver
        for key in ["global.asr", "asr", f"{comp_name}.asr"]:
            self.ctx.variable_resolver.resolved[key] = asr
        for key in ["global.F0_curve", "F0_curve"]:
            self.ctx.variable_resolver.resolved[key] = F0_curve
        for key in ["global.N", "N"]:
            self.ctx.variable_resolver.resolved[key] = N_curve
        for key in ["global.decoder_style", "decoder_style", "s", "global.s"]:
            self.ctx.variable_resolver.resolved[key] = style_dec

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _execute_native_text_encoder(self, comp_name: str) -> None:
        """Native execution for text_encoder: embedding → CNN → BiLSTM.

        cuDNN RNN internal ops can't be replayed through CompiledSequence
        (pack_padded_sequence/aten::set/cuDNN weight buffer ops).
        Extract weights and run natively using torch.nn modules.

        TODO: Fix CompiledSequence to handle aten::lstm or retrace without
        pack_padded_sequence so this can run as a compiled forward pass.
        """
        device = self.ctx.primary_device
        dtype = self._get_compute_dtype()

        print(f"   [{comp_name}] Running native text encoder...")
        start = time.perf_counter()

        # Get inputs
        input_ids = self.ctx.variable_resolver.resolved.get("global.input_ids")
        input_lengths = self.ctx.variable_resolver.resolved.get("global.text_lengths")
        text_mask = self.ctx.variable_resolver.resolved.get("global.text_mask")

        if input_ids is None:
            raise RuntimeError("ZERO FALLBACK: input_ids not bound for text_encoder")
        if input_lengths is None:
            raise RuntimeError("ZERO FALLBACK: text_lengths not bound for text_encoder")

        # Extract weights
        self._ensure_weights_loaded(comp_name)
        w = dict(self.ctx.executors[comp_name]._weights)

        with torch.inference_mode():
            # Embedding
            embed_w = w["embed.weight"].to(device=device, dtype=dtype)
            x = torch.nn.functional.embedding(input_ids.to(device), embed_w)
            # x: [B, seq, 512]
            x = x.transpose(1, 2)  # [B, 512, seq]

            # 3x (WeightNorm Conv1d + LeakyReLU + LayerNorm)
            for i in range(3):
                # WeightNorm: weight = g * v / ||v||
                wg = w[f"cnn.{i}.0.weight_g"].to(device=device, dtype=dtype)
                wv = w[f"cnn.{i}.0.weight_v"].to(device=device, dtype=dtype)
                bias = w[f"cnn.{i}.0.bias"].to(device=device, dtype=dtype)
                # Compute weight_norm: w = g * v / ||v||_2
                norm = wv.norm(dim=(1, 2), keepdim=True)
                conv_w = wg * wv / (norm + 1e-12)
                x = torch.nn.functional.conv1d(x, conv_w, bias, padding=2)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
                # LayerNorm over last dim (channel dim after transpose)
                gamma = w[f"cnn.{i}.1.gamma"].to(device=device, dtype=dtype)
                beta = w[f"cnn.{i}.1.beta"].to(device=device, dtype=dtype)
                x_t = x.transpose(1, 2)  # [B, seq, C]
                x_t = torch.nn.functional.layer_norm(x_t, [gamma.shape[0]], gamma, beta)
                x = x_t.transpose(1, 2)  # [B, 512, seq]

            # BiLSTM with pack_padded_sequence
            x = x.transpose(1, 2)  # [B, seq, 512]

            # Build nn.LSTM and load weights
            lstm = torch.nn.LSTM(
                input_size=512, hidden_size=256,
                num_layers=1, batch_first=True, bidirectional=True
            ).to(device=device, dtype=dtype)

            lstm.weight_ih_l0.data.copy_(w["lstm.weight_ih_l0"].to(device=device, dtype=dtype))
            lstm.weight_hh_l0.data.copy_(w["lstm.weight_hh_l0"].to(device=device, dtype=dtype))
            lstm.bias_ih_l0.data.copy_(w["lstm.bias_ih_l0"].to(device=device, dtype=dtype))
            lstm.bias_hh_l0.data.copy_(w["lstm.bias_hh_l0"].to(device=device, dtype=dtype))
            lstm.weight_ih_l0_reverse.data.copy_(w["lstm.weight_ih_l0_reverse"].to(device=device, dtype=dtype))
            lstm.weight_hh_l0_reverse.data.copy_(w["lstm.weight_hh_l0_reverse"].to(device=device, dtype=dtype))
            lstm.bias_ih_l0_reverse.data.copy_(w["lstm.bias_ih_l0_reverse"].to(device=device, dtype=dtype))
            lstm.bias_hh_l0_reverse.data.copy_(w["lstm.bias_hh_l0_reverse"].to(device=device, dtype=dtype))

            lengths_cpu = input_lengths.cpu().to(torch.int64)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu.clamp(min=1), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = lstm(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            # output: [B, T', 512] where T' = actual length

            # Transpose to [B, 512, T']
            output = output.transpose(1, 2)

            # Re-pad to full mask length (vendor: x_pad[:, :, :x.shape[-1]] = x)
            mask_len = text_mask.shape[-1] if text_mask is not None else output.shape[2]
            if output.shape[2] < mask_len:
                x_pad = torch.zeros(output.shape[0], output.shape[1], mask_len,
                                    device=device, dtype=dtype)
                x_pad[:, :, :output.shape[2]] = output
                output = x_pad

            # Apply mask (vendor: x.masked_fill_(m, 0.0))
            if text_mask is not None:
                output.masked_fill_(text_mask.unsqueeze(1).to(device), 0.0)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Native done in {elapsed:.0f}ms  output={output.shape}")

        # Store output for predictor stage
        self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = output

    def _build_bilstm(
        self, w: Dict, prefix: str, input_size: int, hidden_size: int,
        device, dtype,
    ) -> torch.nn.LSTM:
        """Build a bidirectional LSTM from extracted weights."""
        lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        with torch.no_grad():
            for pname in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
                          "weight_ih_l0_reverse", "weight_hh_l0_reverse",
                          "bias_ih_l0_reverse", "bias_hh_l0_reverse"]:
                getattr(lstm, pname).copy_(w[f"{prefix}.{pname}"].to(device=device, dtype=dtype))
        lstm.eval()
        return lstm.to(device=device, dtype=dtype)

    def _run_kokoro_single_lstm(
        self, x: torch.Tensor, w: Dict, prefix: str,
        text_lengths: Optional[torch.Tensor], max_len: int, device, dtype,
    ) -> torch.Tensor:
        """Run one BiLSTM layer of the DurationEncoder with pack/pad."""
        wih = w[f"{prefix}.weight_ih_l0"]
        input_size = wih.shape[1]
        hidden_size = w[f"{prefix}.weight_hh_l0"].shape[1]

        lstm = self._build_bilstm(w, prefix, input_size, hidden_size, device, dtype)

        x_in = x.transpose(-1, -2)  # [B, C, T] → [B, T, C]
        if text_lengths is not None:
            lengths_cpu = text_lengths.cpu().to(torch.int64).clamp(min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_in, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            out, _ = lstm(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = lstm(x_in)
        # out: [B, T', 2*hidden] → transpose → [B, 2*hidden, T']
        out = out.transpose(-1, -2)
        # Pad to original mask length
        if out.shape[2] < max_len:
            pad = torch.zeros(out.shape[0], out.shape[1], max_len, device=device, dtype=dtype)
            pad[:, :, :out.shape[2]] = out
            out = pad
        return out  # [B, 2*hidden, T]

    def _run_kokoro_adaln(
        self, x: torch.Tensor, style: torch.Tensor, w: Dict,
        prefix: str, device, dtype,
    ) -> torch.Tensor:
        """Run AdaLayerNorm: FC(style) → gamma/beta → LayerNorm → affine."""
        channels = x.shape[1]
        fc_w = w[f"{prefix}.proj.weight"].to(device=device, dtype=dtype)
        fc_b = w[f"{prefix}.proj.bias"].to(device=device, dtype=dtype)

        # x: [B, C, T], style: [B, 128]
        h = style @ fc_w.T + fc_b  # [B, 2*C]
        h = h.unsqueeze(2)  # [B, 2*C, 1]
        gamma, beta = h.chunk(2, dim=1)  # each [B, C, 1]

        # LayerNorm over channel dim
        x_t = x.transpose(1, 2).transpose(1, -1)  # [B, T, C] → [B, C, T] transposed for LN
        # Actually vendor code: x.transpose(-1,-2) → [B,T,C], then .transpose(1,-1) → [B,C,T]
        # Then layer_norm over last dim (C) → then affine
        x_ln = x.permute(0, 2, 1)  # [B, T, C]
        x_ln = torch.nn.functional.layer_norm(x_ln, (channels,))
        x_ln = x_ln.permute(0, 2, 1)  # [B, C, T]

        return (1 + gamma) * x_ln + beta  # [B, C, T]

    def _run_kokoro_f0n_blocks(
        self, d: torch.Tensor, style: torch.Tensor,
        weights: Dict, block_name: str, device, dtype,
    ) -> torch.Tensor:
        """Run F0 or N prediction: 3 AdainResBlk1d blocks + Conv1d projection.

        Returns [1, 1, T] per-phoneme prediction values.
        """
        x = d.clone()

        for block_idx in range(3):
            x = self._run_kokoro_adain_resblock(
                x, style, weights, f"{block_name}.{block_idx}", device, dtype,
            )

        proj_w = weights[f"{block_name}_proj.weight"].to(device=device, dtype=dtype)
        proj_b = weights[f"{block_name}_proj.bias"].to(device=device, dtype=dtype)
        return torch.nn.functional.conv1d(x, proj_w, proj_b)  # [1, 1, T]

    def _run_kokoro_adain_resblock(
        self, x: torch.Tensor, style: torch.Tensor,
        weights: Dict, prefix: str, device, dtype,
    ) -> torch.Tensor:
        """Run AdainResBlk1d matching vendor istftnet.py exactly.

        Architecture (vendor):
          residual: AdaIN→LeakyReLU→pool→Conv1→AdaIN→LeakyReLU→Conv2
          shortcut: upsample→conv1x1 (if dim_in != dim_out)
          output:   (residual + shortcut) * rsqrt(2)

        Detects upsample and learned_sc from weight presence.
        """
        def get_w(name):
            return weights[f"{prefix}.{name}"].to(device=device, dtype=dtype)

        def has_w(name):
            return f"{prefix}.{name}" in weights

        def weight_norm_conv(wg, wv, bias, h, stride=1, padding=None):
            norm = wv.norm(dim=list(range(1, wv.dim())), keepdim=True).clamp(min=1e-12)
            w = wg * wv / norm
            if padding is None:
                padding = (w.shape[2] - 1) // 2
            return torch.nn.functional.conv1d(h, w, bias, stride=stride, padding=padding)

        def weight_norm_conv_transpose(wg, wv, bias, h):
            norm = wv.norm(dim=list(range(1, wv.dim())), keepdim=True).clamp(min=1e-12)
            w = wg * wv / norm
            groups = w.shape[0]
            return torch.nn.functional.conv_transpose1d(
                h, w, bias, stride=2, padding=1, output_padding=1, groups=groups
            )

        def adain(h, norm_proj_w, norm_proj_b):
            h_norm = torch.nn.functional.instance_norm(h)
            proj = style @ norm_proj_w.T + norm_proj_b  # [B, 2*C]
            gamma, beta = proj.chunk(2, dim=-1)
            return (1 + gamma.unsqueeze(-1)) * h_norm + beta.unsqueeze(-1)

        has_upsample = has_w("pool.weight_g")
        has_learned_sc = has_w("conv1x1.weight_g")

        # ── Residual path ──
        h = adain(x, get_w("norm1.proj.weight"), get_w("norm1.proj.bias"))
        h = torch.nn.functional.leaky_relu(h, 0.2)
        if has_upsample:
            h = weight_norm_conv_transpose(
                get_w("pool.weight_g"), get_w("pool.weight_v"), get_w("pool.bias"), h
            )
        h = weight_norm_conv(get_w("conv1.weight_g"), get_w("conv1.weight_v"), get_w("conv1.bias"), h)
        h = adain(h, get_w("norm2.proj.weight"), get_w("norm2.proj.bias"))
        h = torch.nn.functional.leaky_relu(h, 0.2)
        h = weight_norm_conv(get_w("conv2.weight_g"), get_w("conv2.weight_v"), get_w("conv2.bias"), h)

        # ── Shortcut path ──
        sc = x
        if has_upsample:
            sc = torch.nn.functional.interpolate(sc, scale_factor=2, mode='nearest')
        if has_learned_sc:
            sc = weight_norm_conv(
                get_w("conv1x1.weight_g"), get_w("conv1x1.weight_v"), None, sc, padding=0
            )

        return (h + sc) * torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype))

    def _scale_kokoro_durations(self, raw: torch.Tensor, target: int) -> torch.Tensor:
        """Scale predicted durations so they sum to exactly target frames."""
        durations = torch.round(raw).clamp(min=0)
        active = durations > 0
        if active.sum() == 0:
            result = torch.zeros_like(durations, dtype=torch.long)
            result[0] = target
            return result

        current_sum = durations[active].sum()
        if current_sum > 0:
            scale = target / current_sum.item()
            durations[active] = torch.round(durations[active] * scale).clamp(min=1)

        result = durations.long()
        diff = int(target - result.sum().item())

        active_indices = torch.where(active)[0]
        for i in range(abs(diff)):
            idx = int(active_indices[i % len(active_indices)].item())
            result[idx] += 1 if diff > 0 else -1

        return result.clamp(min=0)

    def _get_kokoro_decoder_shapes(self) -> Dict[str, int]:
        """Read decoder input shapes from graph for exact target dimensions."""
        executor = self.ctx.executors.get("decoder")
        if executor is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'decoder' component not found in executors.\n"
                "Required to determine target shapes for predictor output."
            )
        dag = getattr(executor, '_dag', None)
        if dag is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'decoder' component has no DAG.\n"
                "Cannot determine target shapes for predictor output."
            )

        result: Dict[str, int] = {}
        shape_map = {"asr": ("asr_frames", 2), "F0_curve": ("f0_len", 1), "N": ("n_len", 1)}
        for spec in dag.get("tensors", {}).values():
            name = spec.get("input_name", "")
            if name in shape_map:
                key, dim_idx = shape_map[name]
                shape = spec.get("shape", [])
                if len(shape) > dim_idx:
                    val = shape[dim_idx]
                    if isinstance(val, int):
                        result[key] = val
                    elif isinstance(val, dict) and "trace_value" in val:
                        result[key] = val["trace_value"]
                    else:
                        raise RuntimeError(
                            f"ZERO FALLBACK: Cannot resolve shape dim for '{name}' in decoder graph."
                        )

        for required in ("asr_frames", "f0_len", "n_len"):
            if required not in result:
                raise RuntimeError(
                    f"ZERO FALLBACK: '{required}' shape not found in decoder graph inputs."
                )
        return result

    # ─────────────────────────────────────────────────────────────
    # Native acoustic decoder (VibeVoice)
    # ─────────────────────────────────────────────────────────────

    def _execute_native_acoustic_decoder(self, stage: Dict, audio_config: Dict) -> None:
        """Run acoustic tokenizer decoder natively.

        The traced graph covers the full encode+decode path (waveform→waveform),
        but TTS only needs the decoder (latent→waveform). Extract decoder weights
        and run the ConvNext1d architecture natively.

        Architecture params (decoder_depths, decoder_upsampling_ratios) read from
        defaults.json — set by builder from model's config.json.

        TODO: Retrace only the decoder portion as a separate component so this
        can run through CompiledSequence as a standard forward pass.
        """
        comp_name = stage["component"]
        device = self.ctx.primary_device
        dtype = self._get_compute_dtype()

        # Get denoised latent from diffusion stage
        latent = self.ctx.variable_resolver.resolved.get("global.denoised_latent")
        if latent is None:
            raise RuntimeError(
                "ZERO FALLBACK: native_acoustic_decoder requires global.denoised_latent "
                "from a preceding diffusion stage."
            )

        print(f"   [{comp_name}] Running native acoustic decoder...")
        start = time.perf_counter()

        self._ensure_weights_loaded(comp_name)
        w = self.ctx.executors[comp_name]._weights

        # Diffusion outputs [B, T, latent_dim=64], decoder expects [B, C=64, T]
        x = latent.to(device=device, dtype=dtype)
        if x.shape[1] != 64 or (x.dim() == 3 and x.shape[2] == 64 and x.shape[1] != 64):
            x = x.transpose(1, 2)

        # Decoder config (DATA-DRIVEN from defaults.json)
        defaults = self.ctx.pkg.defaults
        upsample_strides = defaults.get("decoder_upsampling_ratios")
        stage_depths = defaults.get("decoder_depths")
        if upsample_strides is None or stage_depths is None:
            raise RuntimeError(
                "ZERO FALLBACK: decoder_upsampling_ratios and decoder_depths "
                "missing from defaults.json.\n"
                "Builder must extract from model's config.json."
            )

        with torch.inference_mode():
            # ── Stem: CausalConv1d(64→2048, k=7) ──
            x = self._vv_causal_conv1d(x, w,
                "decoder.upsample_layers.0.0.conv.conv", kernel_size=7)

            # ── Stem stage: 8 ConvNext blocks at 2048 ──
            for blk in range(stage_depths[0]):
                x = self._vv_convnext_block(x, w, f"decoder.stages.0.{blk}")

            # ── Upsample stages 1-5 (ConvTranspose1d) ──
            for i, stride in enumerate(upsample_strides[:5]):
                kernel_size = stride * 2
                x = self._vv_causal_conv_transpose1d(
                    x, w, f"decoder.upsample_layers.{i+1}.0.convtr.convtr",
                    stride=stride, kernel_size=kernel_size)
                for blk in range(stage_depths[i + 1]):
                    x = self._vv_convnext_block(x, w, f"decoder.stages.{i+1}.{blk}")

            # ── Last upsample (conv_layers.0, stride=2) ──
            x = self._vv_causal_conv_transpose1d(
                x, w, "decoder.conv_layers.0.convtr.convtr",
                stride=upsample_strides[5], kernel_size=upsample_strides[5] * 2)
            for blk in range(stage_depths[6]):
                x = self._vv_convnext_block(x, w, f"decoder.stages.6.{blk}")

            # ── Head: CausalConv1d(32→1, k=7) ──
            x = self._vv_causal_conv1d(x, w, "decoder.head.conv.conv", kernel_size=7)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{comp_name}] Decoder done in {elapsed:.0f}ms  output={x.shape}")

        # Store waveform output
        self.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = x
        self.ctx.variable_resolver.resolved["global.output_audio"] = x

        if not self.ctx.persistent_mode:
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()

    def _vv_causal_conv1d(
        self, x: torch.Tensor, w: Dict, prefix: str,
        kernel_size: int = 7, stride: int = 1, groups: int = 1,
    ) -> torch.Tensor:
        """CausalConv1d: left-pad then Conv1d."""
        weight = w[f"{prefix}.weight"].to(device=x.device, dtype=x.dtype)
        bias = w.get(f"{prefix}.bias")
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        causal_pad = (kernel_size - 1) - (stride - 1)
        if causal_pad > 0:
            x = torch.nn.functional.pad(x, (causal_pad, 0))
        return torch.nn.functional.conv1d(x, weight, bias, stride=stride, groups=groups)

    def _vv_causal_conv_transpose1d(
        self, x: torch.Tensor, w: Dict, prefix: str,
        stride: int, kernel_size: int,
    ) -> torch.Tensor:
        """CausalConvTranspose1d: ConvTranspose1d then trim right padding."""
        weight = w[f"{prefix}.weight"].to(device=x.device, dtype=x.dtype)
        bias = w.get(f"{prefix}.bias")
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        x = torch.nn.functional.conv_transpose1d(x, weight, bias, stride=stride)
        padding_total = kernel_size - stride
        if padding_total > 0:
            x = x[..., :-padding_total]
        return x

    def _vv_convnext_block(
        self, x: torch.Tensor, w: Dict, prefix: str,
    ) -> torch.Tensor:
        """ConvNext1d block: mixer path (norm→depthwise_conv→gamma) + FFN path.

        Matches VibeVoiceAcousticTokenizerConvNext1dLayer exactly:
          mixer: residual + gamma * causal_depthwise_conv(rms_norm(x))
          ffn:   residual + ffn_gamma * linear2(gelu(linear1(ffn_norm(x))))
        """
        dev, dt = x.device, x.dtype

        def get(name):
            return w[f"{prefix}.{name}"].to(device=dev, dtype=dt)

        # ── Mixer path ──
        residual = x
        channels = x.shape[1]
        # RMSNorm over last dim: transpose [B,C,T]→[B,T,C], norm, transpose back
        h = self._vv_rms_norm(x.transpose(1, 2), get("norm.weight")).transpose(1, 2)
        # Depthwise causal conv (groups=channels, kernel derived from weight)
        mixer_w = get("mixer.conv.conv.conv.weight")
        mixer_b = get("mixer.conv.conv.conv.bias")
        causal_pad = mixer_w.shape[2] - 1
        h = torch.nn.functional.pad(h, (causal_pad, 0))
        h = torch.nn.functional.conv1d(h, mixer_w, mixer_b, groups=channels)
        h = h * get("gamma").unsqueeze(-1)
        x = residual + h

        # ── FFN path ──
        residual = x
        h = self._vv_rms_norm(x.transpose(1, 2), get("ffn_norm.weight"))
        # Linear1 → GELU → Linear2  (operates on [B,T,C])
        h = torch.nn.functional.linear(h, get("ffn.linear1.weight"), get("ffn.linear1.bias"))
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.linear(h, get("ffn.linear2.weight"), get("ffn.linear2.bias"))
        h = h.transpose(1, 2)  # back to [B,C,T]
        h = h * get("ffn_gamma").unsqueeze(-1)
        x = residual + h

        return x

    def _vv_rms_norm(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5,
    ) -> torch.Tensor:
        """RMSNorm: x * rsqrt(mean(x²) + eps) * weight."""
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + eps)
        return weight * x_normed.to(x.dtype)

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
                vocab_size = self._get_extracted_value("vocab_size")
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
        """Get embedding weight for weight-tied logits or text→embedding conversion.

        Search order:
        1. Inside the LLM component's weights (standard HF models)
        2. Separate embed_tokens component (NeMo SpeechLM models)
        """
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
                if "weight" in key or "embed" in key:
                    return embed_executor._weights[key]

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
        """Compute logits from hidden states.

        logits_source values:
          "self"              — output tensor IS logits (graph includes lm_head)
          "embed_weight_tied" — matmul(hidden, embed_tokens.T) for weight-tied models
          "lm_head"           — execute separate lm_head component on hidden states
        """
        last_hidden = hidden_states[:, -1:, :]

        if logits_source == "lm_head" and "lm_head" in self.ctx.executors:
            # Separate lm_head component: extract weight and matmul directly
            # lm_head is typically a single Linear (no bias): logits = hidden @ weight.T
            self._ensure_weights_loaded("lm_head")
            executor = self.ctx.executors["lm_head"]
            lm_weight = None
            for key, tensor in executor._weights.items():
                if tensor is not None and tensor.ndim == 2:
                    lm_weight = tensor
                    break
            if lm_weight is not None:
                w = lm_weight.to(dtype=last_hidden.dtype)
                return torch.matmul(last_hidden, w.T)
            return last_hidden

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

    def _get_extracted_value(self, key: str) -> Any:
        """Get a value from topology extracted_values or defaults. Returns None if missing."""
        extracted = self.ctx.pkg.topology.get("extracted_values", {})
        for section in extracted.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        return self.ctx.pkg.defaults.get(key)

    def _get_forced_decoder_ids(self) -> List:
        """Get forced decoder IDs from defaults.json (set by builder)."""
        forced = self.ctx.pkg.defaults.get("forced_decoder_ids", [])
        if forced:
            return [(pos, tid) for pos, tid in forced]
        return []

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
        """Find model config path from NBX container, then snapshot."""
        nbx_path = Path(self.ctx.nbx_path_str)
        # 1. NBX container modules/tokenizer (has preprocessor_config.json)
        for subdir in ["modules/tokenizer", "modules/processor"]:
            candidate = nbx_path / subdir
            if candidate.exists():
                return candidate
        # 2. Component path from topology (first component with a path)
        for comp_info in self.ctx.pkg.topology.get("components", {}).values():
            comp_path = comp_info.get("path")
            if comp_path and Path(comp_path).exists():
                return Path(comp_path)
        raise RuntimeError(
            "ZERO FALLBACK: Cannot find model config path.\n"
            "Expected modules/tokenizer in NBX container or component path in topology."
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
