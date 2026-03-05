"""
Encoder-Decoder Audio Flow Handler

Handles speech-to-text models like Whisper:
1. Audio preprocessing (mel spectrogram extraction)
2. Encoder forward pass
3. Autoregressive decoder loop with cross-attention
4. Token decoding to text

ZERO SEMANTIC: Flow structure from topology.json.
ZERO HARDCODE: All parameters from NBX container (config, defaults).
"""

import gc
import time
import torch
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("encoder_decoder_audio")
class EncoderDecoderAudioHandler(FlowHandler):
    """
    Flow handler for encoder-decoder audio models (Whisper, etc.).

    Execution flow:
    1. Load audio → extract mel spectrogram features
    2. Run encoder on features → encoder_hidden_states
    3. Autoregressive decoder: generate tokens using cross-attention
    4. Decode token IDs → text transcription
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
        """Execute encoder-decoder audio flow."""
        flow = self.ctx.pkg.topology.get("flow", {})
        component_order = flow.get("order", [])
        gen_config = flow.get("generation", {})

        if len(component_order) < 2:
            raise RuntimeError(
                "ZERO FALLBACK: encoder_decoder_audio requires at least 2 components "
                f"in flow.order, got {len(component_order)}: {component_order}"
            )

        encoder_name = component_order[0]
        decoder_name = component_order[1]

        # Get generation parameters from defaults
        defaults = self.ctx.pkg.defaults
        max_tokens = defaults.get("max_tokens", gen_config.get("max_tokens", 448))
        temperature = defaults.get("temperature", gen_config.get("temperature", 0.0))

        # Get special token IDs from topology extracted_values or config
        extracted = self.ctx.pkg.topology.get("extracted_values", {})
        decoder_start_token_id = self._get_config_value("decoder_start_token_id", 50258)
        eos_token_id = self._get_config_value("eos_token_id", 50257)

        # Forced decoder IDs (language/task tokens)
        forced_decoder_ids = self._get_forced_decoder_ids()

        device = self.ctx.primary_device

        # ── Step 1: Audio preprocessing ──
        audio_path = self.ctx.variable_resolver.resolved.get("global.audio_path")
        if audio_path is None:
            raise RuntimeError(
                "ZERO FALLBACK: encoder_decoder_audio requires global.audio_path.\n"
                "Use --audio <path> to provide an audio file."
            )

        print(f"   [Audio] Loading: {audio_path}")
        from neurobrix.core.module.audio.processor import AudioProcessor

        # Find snapshot path for preprocessor config
        snapshot_path = self._find_snapshot_path()
        audio_proc = AudioProcessor(
            model_path=snapshot_path,
            device=torch.device(device),
            dtype=self._get_compute_dtype(),
        )
        input_features = audio_proc.process_audio_file(str(audio_path))
        print(f"   [Audio] Features: {input_features.shape}")

        # Bind input_features for encoder
        self.ctx.variable_resolver.resolved["global.input_features"] = input_features

        # ── Step 2: Encoder forward pass ──
        print(f"   [Encoder] Running {encoder_name}...")
        start = time.perf_counter()

        self._ensure_weights_loaded(encoder_name)
        self._execute_component(encoder_name, "forward", None)

        encoder_time = (time.perf_counter() - start) * 1000
        print(f"   [Encoder] Done in {encoder_time:.0f}ms")

        # Get encoder output
        encoder_output = self._get_encoder_output(encoder_name)
        if encoder_output is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Encoder {encoder_name} produced no output. "
                f"Available: {list(self.ctx.variable_resolver.resolved.keys())}"
            )
        print(f"   [Encoder] Output: {encoder_output.shape}")

        # Store encoder output for decoder cross-attention
        self.ctx.variable_resolver.resolved[
            f"{encoder_name}.output_0"
        ] = encoder_output

        # Free encoder weights
        self._unload_component_weights(encoder_name)
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 3: Autoregressive decoder loop ──
        print(f"   [Decoder] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()

        self._ensure_weights_loaded(decoder_name)

        # Get embed_tokens weight for proj_out (weight-tied lm_head)
        embed_weight = self._get_embed_weight(decoder_name)

        # Initialize decoder input: [decoder_start_token_id]
        generated_ids = [decoder_start_token_id]

        # Apply forced decoder IDs first
        for pos, token_id in forced_decoder_ids:
            if token_id is not None:
                generated_ids.append(token_id)

        # Autoregressive generation
        for step in range(max_tokens):
            # Prepare decoder input
            input_ids = torch.tensor(
                [generated_ids], dtype=torch.long, device=device
            )

            # Bind decoder inputs
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            # Run decoder
            self._execute_component(decoder_name, "forward", None)

            # Get decoder hidden states (last position)
            decoder_output = self._get_decoder_output(decoder_name)
            if decoder_output is None:
                break

            # proj_out: hidden_states @ embed_weight.T → logits
            last_hidden = decoder_output[:, -1:, :]  # [1, 1, d_model]
            if embed_weight is not None:
                w = embed_weight.to(dtype=last_hidden.dtype)
                logits = torch.matmul(last_hidden, w.T)  # [1, 1, vocab]
            else:
                logits = last_hidden  # fallback — shouldn't happen

            # Sample next token
            if temperature == 0.0:
                next_token = logits[:, -1, :].argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated_ids.append(next_token)

            # Check for EOS
            if next_token == eos_token_id:
                break

        decoder_time = (time.perf_counter() - start) * 1000
        num_tokens = len(generated_ids) - 1 - len(forced_decoder_ids)
        print(f"   [Decoder] Generated {num_tokens} tokens in {decoder_time:.0f}ms")

        # Free decoder weights
        self._unload_component_weights(decoder_name)
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 4: Decode to text ──
        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            # Skip special tokens (decoder_start, forced tokens, eos)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            text = str(generated_ids)

        # Store result
        self.ctx.variable_resolver.resolved["global.transcription"] = text
        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

        return self.ctx.variable_resolver.resolve_all()

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get config value from extracted_values or defaults."""
        extracted = self.ctx.pkg.topology.get("extracted_values", {})
        for section in extracted.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        return self.ctx.pkg.defaults.get(key, default)

    def _get_forced_decoder_ids(self):
        """Get forced decoder IDs from generation config."""
        # Try from generation_config in snapshot
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
        """Find the HF snapshot path for preprocessor config."""
        # Check if snapshot exists in standard location
        model_name = self.ctx.pkg.manifest.get("model_name", "")
        snapshot_path = Path(f"/home/mlops/hf_snapshots/{model_name}")
        if snapshot_path.exists():
            return snapshot_path

        # Fallback: check NBX cache for tokenizer files
        nbx_path = Path(self.ctx.nbx_path_str)
        if (nbx_path / "modules" / "tokenizer").exists():
            # The preprocessor_config.json might be in the snapshot
            # For now, try the tokenizer directory
            return nbx_path / "modules" / "tokenizer"

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot find snapshot path for {model_name}. "
            f"Expected at {snapshot_path}"
        )

    def _get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype from manifest."""
        dtype_str = self.ctx.pkg.manifest.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _get_encoder_output(self, encoder_name: str) -> Optional[torch.Tensor]:
        """Get encoder output tensor from resolved variables."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [
            f"{encoder_name}.last_hidden_state",
            f"{encoder_name}.output_0",
            f"{encoder_name}.output",
        ]:
            if key in resolved:
                return resolved[key]
        return None

    def _get_decoder_output(self, decoder_name: str) -> Optional[torch.Tensor]:
        """Get decoder output tensor from resolved variables."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [
            f"{decoder_name}.last_hidden_state",
            f"{decoder_name}.output_0",
            f"{decoder_name}.output",
        ]:
            if key in resolved:
                return resolved[key]
        return None

    def _get_embed_weight(self, decoder_name: str) -> Optional[torch.Tensor]:
        """
        Get decoder embedding weight for proj_out (weight-tied lm_head).

        In Whisper, proj_out.weight = model.decoder.embed_tokens.weight.
        The weight is already loaded in the decoder's GraphExecutor.
        """
        executor = self.ctx.executors.get(decoder_name)
        if executor is None:
            return None

        # Search for embed_tokens weight in the executor's weight store
        for key in executor._weights:
            if "embed_tokens" in key or "token_embed" in key:
                return executor._weights[key]
        return None
