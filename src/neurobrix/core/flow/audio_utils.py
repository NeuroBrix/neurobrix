"""
Audio Flow Utilities — Shared preprocessing and postprocessing.

Used by encoder_decoder, audio_llm, dual_ar, and audio flows.

ZERO SEMANTIC: No model knowledge. All config from topology/defaults.
"""

import torch
from neurobrix.core.device_utils import device_multinomial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import FlowContext


def preprocess_audio_input(
    ctx: FlowContext, audio_config: Dict, stages: List[Dict],
) -> None:
    """Load audio file and extract features. Input shape read from graph (DATA-DRIVEN)."""
    input_config = audio_config.get("input", {})
    audio_path = ctx.variable_resolver.resolved.get("global.audio_path")
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
    device = torch.device(ctx.primary_device)
    dtype = get_compute_dtype(ctx)

    # Read expected input shape from first stage's graph (DATA-DRIVEN)
    first_comp = stages[0]["component"] if stages else None
    input_shape = get_component_input_shape(ctx, first_comp)

    # Auto-correct preprocessing type from graph input shape
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
        model_path=find_model_config_path(ctx),
        device=device,
        dtype=dtype,
        input_shape=input_shape,
    )

    # Pad/truncate to match trace-time dimensions
    if input_shape and len(input_shape) == len(features.shape) and len(input_shape) >= 3:
        for dim_idx in range(1, len(input_shape)):
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
    ctx.variable_resolver.resolved[variable] = features
    short_key = variable.split(".")[-1] if "." in variable else variable
    ctx.variable_resolver.resolved[short_key] = features

    # Bind audio length for models that need it
    actual_frames = features.shape[-1] if preprocessing in ("mel_spectrogram", "nemo_mel") else features.shape[1]
    length_tensor = torch.tensor([actual_frames], dtype=torch.long, device=features.device)
    for key in ["global.audio_signal_length", "audio_signal_length", "global.length", "length"]:
        ctx.variable_resolver.resolved[key] = length_tensor


def preprocess_text_input(ctx: FlowContext, input_config: Dict = None) -> None:
    """Tokenize text prompt for TTS/LLM-audio models."""
    if input_config is None:
        input_config = {}

    prompt = ctx.variable_resolver.resolved.get("global.prompt")
    if prompt is None:
        raise RuntimeError(
            "ZERO FALLBACK: TTS model requires global.prompt.\n"
            "Use --prompt <text> to provide text input."
        )

    # Apply TTS prompt template if configured
    tts_template = ctx.pkg.defaults.get("tts_prompt_template")
    if tts_template and "{text}" in tts_template:
        prompt = tts_template.format(text=prompt)

    tokenizer = ctx.modules.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

    device = ctx.primary_device

    # LLM-style tokenization
    add_special = tts_template is None
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=add_special)
    except TypeError:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special)
        input_ids = torch.tensor([ids], dtype=torch.long)
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    ctx.variable_resolver.resolved["global.input_ids"] = input_ids
    ctx.variable_resolver.resolved["input_ids"] = input_ids
    ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
    ctx.variable_resolver.resolved["attention_mask"] = attention_mask


def postprocess_text_output(ctx: FlowContext) -> None:
    """Decode generated token IDs to text."""
    generated_ids = ctx.variable_resolver.resolved.get("global.generated_token_ids")
    if generated_ids is None:
        return

    tokenizer = ctx.modules.get("tokenizer")
    if tokenizer is not None:
        from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
        text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
    else:
        text = str(generated_ids)

    ctx.variable_resolver.resolved["global.transcription"] = text
    print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")


def postprocess_audio_output(ctx: FlowContext) -> None:
    """Process TTS output: decode audio tokens or store raw waveform."""
    device = ctx.primary_device
    defaults = ctx.pkg.defaults
    audio_output_type = defaults.get("audio_output_type")
    generated_ids = ctx.variable_resolver.resolved.get("global.generated_token_ids")

    # Read-and-clear the per-request content fraction so a stale ratio cannot
    # crop a later warm-mode request routed through a different model (R30:
    # the kokoro stage sets this in both modes; both must consume it).
    content_ratio = ctx.variable_resolver.resolved.pop("global.audio_content_ratio", None)

    if audio_output_type == "snac_tokens" and generated_ids and isinstance(generated_ids, list):
        audio_token_start = defaults.get("audio_token_start")
        if audio_token_start is None:
            raise RuntimeError("ZERO FALLBACK: audio_token_start missing from defaults.json.")
        vocab_size = defaults.get("vocab_size")
        if vocab_size is None:
            raise RuntimeError("ZERO FALLBACK: vocab_size missing from defaults.json.")

        from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
        waveform = AudioOutputProcessor.decode_snac_tokens(
            generated_ids, vocab_size=vocab_size, device=device,
        )
        if waveform.numel() > 1:
            # Output saving is the CLI's responsibility via
            # `output_dispatch.save_audio` — it reads
            # `global.output_audio` from outputs and writes to
            # args.output / family-aware default. The flow handler
            # must NOT write directly (that produced a stray
            # `output_<model>.wav` in cwd regardless of --output).
            ctx.variable_resolver.resolved["global.output_audio"] = waveform
            return

    # Raw waveform output
    waveform = None
    resolved = ctx.variable_resolver.resolved
    for key in ["decoder.output_0", "global.output_audio"]:
        val = resolved.get(key)
        if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[-1] > 1000:
            waveform = val
            break

    if waveform is None:
        for val in resolved.values():
            if isinstance(val, torch.Tensor) and val.dim() == 3 and val.shape[1] == 1:
                if val.shape[2] > 1000:
                    waveform = val
                    break

    if waveform is not None:
        waveform = crop_waveform_to_content_ratio(waveform, content_ratio)
        # CLI handles the actual save — see comment in SNAC branch above.
        ctx.variable_resolver.resolved["global.output_audio"] = waveform


# ─── Shared helpers ──────────────────────────────────────────────────────────

def crop_waveform_to_content_ratio(waveform, ratio):
    """Crop a synthesised waveform to the spoken-content fraction.

    A stage that fed a fixed-window decoder a partially-filled input (e.g. the
    Kokoro predictor, whose asr/F0/N tail is zero) sets the per-request float
    `global.audio_content_ratio`; this crops the synthesised silent tail to
    `ratio` of the sample length. No-op when `ratio` is absent or >= 1.0.

    Shared by the compiled (`AudioEngine`) and triton output paths so the crop
    is symmetric across modes (R30). Works for both `torch.Tensor` and
    `NBXTensor` — only `.shape` and a trailing-axis slice are used.
    """
    if not isinstance(ratio, (int, float)) or not (0.0 < ratio < 1.0):
        return waveform
    crop_len = max(1, int(waveform.shape[-1] * ratio))
    if crop_len >= waveform.shape[-1]:
        return waveform
    print(f"   [Output] Cropped to spoken content: {crop_len} samples "
          f"(ratio={ratio:.3f})")
    return waveform[..., :crop_len]


def get_compute_dtype(ctx: FlowContext) -> torch.dtype:
    """Get compute dtype from manifest (string→torch.dtype via the dtype engine)."""
    from neurobrix.core.dtype.config import get_torch_dtype
    return get_torch_dtype(ctx.pkg.manifest.get("dtype", "float16"))


def get_sample_rate(ctx: FlowContext) -> int:
    """Get sample rate from topology or defaults."""
    flow = ctx.pkg.topology.get("flow", {})
    sr = flow.get("audio", {}).get("sample_rate")
    if sr is not None:
        return sr
    sr = ctx.pkg.defaults.get("sample_rate")
    if sr is not None:
        return sr
    raise RuntimeError("ZERO FALLBACK: sample_rate missing from topology and defaults.json.")


def find_model_config_path(ctx: FlowContext) -> Path:
    """Find model config path from NBX container.

    Prioritizes modules/processor (has preprocessor_config.json for audio models)
    over modules/tokenizer.
    """
    nbx_path = Path(ctx.nbx_path_str)
    # Processor first (has preprocessor_config.json for mel extraction)
    for subdir in ["modules/processor", "modules/tokenizer"]:
        candidate = nbx_path / subdir
        if candidate.exists():
            return candidate
    # Removed: legacy absolute-path fallback to `topology.components.path`.
    # That field held the trace-host absolute snapshot location and is no
    # longer present in correctly-built containers. Audio models without
    # an embedded `modules/processor` directory must be re-imported with
    # the current builder which embeds `preprocessor_config.json`.
    raise RuntimeError(
        "Cannot find model config path. Expected `modules/processor/` "
        "or `modules/tokenizer/` inside the .nbx. Re-import the model "
        "with the current builder which embeds these directories."
    )


def get_component_input_shape(
    ctx: FlowContext, comp_name: Optional[str],
) -> Optional[Tuple[int, ...]]:
    """Read first input tensor shape from component's graph (DATA-DRIVEN)."""
    if comp_name is None:
        return None
    executor = ctx.executors.get(comp_name)
    if executor is None:
        return None
    dag = getattr(executor, '_dag', None)
    if dag is None:
        return None
    for tid, spec in dag.get("tensors", {}).items():
        is_input = (
            spec.get("type") == "input"
            or spec.get("input_name") is not None
            or tid.startswith("input::")
        )
        if is_input:
            shape = spec.get("shape", [])
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


def get_component_output(
    ctx: FlowContext, comp_name: str,
) -> Optional[torch.Tensor]:
    """Get a component's primary output tensor."""
    resolved = ctx.variable_resolver.resolved
    for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
        if key in resolved and isinstance(resolved[key], torch.Tensor):
            return resolved[key]
    return None


def get_embed_weight(ctx: FlowContext, comp_name: str) -> Optional[torch.Tensor]:
    """Get embedding weight for weight-tied logits or text→embedding conversion."""
    executor = ctx.executors.get(comp_name)
    if executor is not None:
        for key in executor._weights:
            if "token_embed" in key or "embed" in key:  # NeuroTax standard
                return executor._weights[key]
    embed_executor = ctx.executors.get("embed_tokens")
    if embed_executor is not None:
        for key in embed_executor._weights:
            if "weight" in key or "embed" in key:
                return embed_executor._weights[key]
    return None


def sample_token(
    logits: torch.Tensor, temperature: float,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample next token from logits with optional repetition penalty and top-p."""
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

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = sorted_idx[0, device_multinomial(sorted_probs[0], 1)].item()
    else:
        next_token = device_multinomial(probs, 1).squeeze(-1).item()

    return next_token
