"""
Universal output dispatch — data-driven save/extract based on family YAML.

ONE source of truth for both:
- CLI cold path (cli/commands/run.py)
- Serving warm path (serving/engine.py + serving/server.py)

ZERO HARDCODE: All format/extension decisions read from
config/families/<family>.yml output_processing.output_format and
output.default_extension_per_mode.

ZERO FALLBACK: Unknown family or missing output_format raises explicit
error so a model with malformed family can never silently produce a
wrong-modality file (e.g. .png from STT).
"""

from typing import Any, Dict, Optional
from pathlib import Path

from neurobrix.core.config import get_family_config, get_output_processing


# ---------------------------------------------------------------------------
# Family format introspection
# ---------------------------------------------------------------------------


def get_output_format(family: str) -> str:
    """
    Resolve canonical output format string for a family.

    Reads `output_processing.output_format` from the family YAML.

    For families where the format depends on runtime mode (e.g.
    multimodal text vs image), the YAML declares
    `output_format: "mode_dependent"` and the caller must pass `mode`
    to `default_extension_for_mode()` instead.

    Returns:
        One of: "txt", "wav", "png", "mp4", "mode_dependent"

    Raises:
        RuntimeError: family unknown or output_format missing
    """
    op = get_output_processing(family)
    fmt = op.get("output_format")
    if not fmt:
        raise RuntimeError(
            f"ZERO FALLBACK: family '{family}' has no output_processing.output_format. "
            f"Add it to config/families/{family}.yml."
        )
    return fmt


def default_extension_for_mode(family: str, mode: Optional[str] = None) -> str:
    """
    Resolve canonical extension for (family, mode).

    Reads `output.default_extension_per_mode[mode]`. If `mode` is None,
    falls back to `modes.default` then to a derivation from `output_format`.

    Returns:
        Extension WITH leading dot (e.g. ".png", ".wav", ".txt")
    """
    cfg = get_family_config(family)
    ext_map = cfg.get("output", {}).get("default_extension_per_mode", {})
    if mode is None:
        mode = cfg.get("modes", {}).get("default")
    if mode and mode in ext_map:
        return ext_map[mode]
    fmt = get_output_format(family)
    if fmt and fmt != "mode_dependent":
        return f".{fmt}"
    raise RuntimeError(
        f"ZERO FALLBACK: cannot resolve extension for family '{family}' mode '{mode}'. "
        f"Update config/families/{family}.yml output.default_extension_per_mode."
    )


def family_uses_text_warmup(family: str) -> bool:
    """
    Whether a text-only warmup prompt ('warmup', max_tokens=1) is meaningful
    for this family.

    Only LLM benefits from text warmup. Every other family either needs an
    actual modality input (audio for stt/audio_llm, image for vlm/upscaler)
    or has a different runtime path that doesn't respond to a tiny text
    prompt (image/video diffusion, tts, multimodal-strict).
    """
    return family == "llm"


# ---------------------------------------------------------------------------
# Data-driven input validation — replaces hardcoded checks in run.py
# ---------------------------------------------------------------------------


def validate_required_inputs(family: str, args) -> None:
    """
    Validate that all required CLI flags for the family are provided.

    Reads `inputs.required` from the family YAML. Each entry is a CLI flag
    string (e.g. "--audio", "--prompt", "--input-image"). The argparse
    Namespace `args` is checked: dest is derived from the flag (--input-image
    → args.input_image), and the value must be non-None and non-empty.

    Raises:
        RuntimeError: required flag missing
    """
    cfg = get_family_config(family)
    required = cfg.get("inputs", {}).get("required", [])
    for flag in required:
        dest = _flag_to_dest(flag)
        value = getattr(args, dest, None)
        if value is None or value == "":
            raise RuntimeError(
                f"ZERO FALLBACK: family '{family}' requires {flag}. "
                f"Provide it on the command line."
            )


def resolve_mode(family: str, args) -> Optional[str]:
    """
    Resolve runtime mode for the family.

    Cascade:
      1. Explicit --mode (if provided)
      2. Auto-deduce from optional flags via inputs.optional[].triggers_mode
      3. modes.default

    For multimodal_strict families, --mode MUST be explicit (raises if missing).
    """
    cfg = get_family_config(family)
    modes_cfg = cfg.get("modes", {})
    supported = modes_cfg.get("supported", [])
    strict = modes_cfg.get("multimodal_strict", False)
    default = modes_cfg.get("default")

    explicit = getattr(args, "mode", None)
    if explicit and explicit != "auto":
        if supported and explicit not in supported:
            raise RuntimeError(
                f"ZERO FALLBACK: --mode '{explicit}' not supported for family '{family}'. "
                f"Supported: {supported}"
            )
        return explicit

    # Auto-deduce from optional flags
    optional = cfg.get("inputs", {}).get("optional", [])
    for entry in optional:
        if not isinstance(entry, dict):
            continue
        trig = entry.get("triggers_mode")
        if not trig:
            continue
        flag = entry.get("flag")
        if flag and getattr(args, _flag_to_dest(flag), None):
            return trig

    if strict and not default:
        raise RuntimeError(
            f"ZERO FALLBACK: --mode is required for family '{family}'. "
            f"Choices: {supported}."
        )
    return default


def _flag_to_dest(flag: str) -> str:
    """Convert '--input-image' → 'input_image' (argparse dest convention)."""
    return flag.lstrip("-").replace("-", "_")


# ---------------------------------------------------------------------------
# Save dispatch — used by CLI cold path AND serving warm path
# ---------------------------------------------------------------------------


def resolve_output_path(
    user_output: Optional[str],
    model_name: str,
    family: str,
    mode: Optional[str] = None,
    strict_extension: bool = True,
) -> str:
    """
    Resolve final output path with auto-extension and strict mismatch check.

    - If `user_output` is None: auto-name `output_<model>.<ext>`.
    - If `user_output` has the right extension: keep as-is.
    - If `user_output` has a different extension and `strict_extension`:
      raise RuntimeError with a clear message (Q1 doctrine).
    """
    expected_ext = default_extension_for_mode(family, mode)
    if user_output is None:
        return f"output_{model_name}{expected_ext}"

    p = Path(user_output)
    actual_ext = p.suffix.lower()
    if actual_ext and actual_ext != expected_ext.lower():
        if strict_extension:
            raise RuntimeError(
                f"ZERO FALLBACK: output extension '{actual_ext}' incompatible "
                f"with family '{family}' mode '{mode or 'default'}'. "
                f"Expected '{expected_ext}' (or omit --output for auto extension)."
            )
        return str(p.with_suffix(expected_ext))
    if not actual_ext:
        return str(p) + expected_ext
    return user_output


def save_text(
    outputs: Dict[str, Any],
    output_path: str,
    executor,
) -> str:
    """Save text output (LLM, VLM, multimodal-text, audio_llm, stt)."""
    text = _extract_text(outputs, executor)
    if text is None:
        raise RuntimeError(
            f"ZERO FALLBACK: no text output to save (no global.transcription, "
            f"no output_tokens). Available keys: {list(outputs.keys())}"
        )
    if output_path.endswith(".json"):
        import json
        with open(output_path, "w") as f:
            json.dump({"text": text}, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w") as f:
            f.write(text)
    return output_path


def save_audio(
    outputs: Dict[str, Any],
    output_path: str,
    pkg,
) -> str:
    """Save audio waveform (TTS) or transcription text (STT — falls back to save_text)."""
    transcription = outputs.get("global.transcription")
    if transcription:
        return save_text(outputs, output_path, executor=None)

    waveform = outputs.get("global.output_audio")
    if waveform is None:
        raise RuntimeError(
            f"ZERO FALLBACK: no audio output to save (no global.output_audio). "
            f"Available keys: {list(outputs.keys())}"
        )

    from neurobrix.core.module.audio.output_processor import AudioOutputProcessor

    audio_cfg = get_output_processing("tts")
    flow_sr = pkg.topology.get("flow", {}).get("audio", {}).get("sample_rate")
    sample_rate = (
        flow_sr
        or pkg.defaults.get("sample_rate")
        or audio_cfg.get("sample_rate")
        or 24000
    )
    AudioOutputProcessor.save_waveform(waveform, output_path, sample_rate)
    return output_path


def save_image(
    outputs: Dict[str, Any],
    output_path: str,
    family: str,
    executor,
    pkg,
) -> str:
    """Save image tensor (image, multimodal-image, upscaler, vlm-with-image-out)."""
    import numpy as np
    import torch
    from PIL import Image

    from neurobrix.core.module.output_processor import OutputProcessor

    final = executor.get_final_output(outputs)
    if final is None:
        raise RuntimeError(
            f"ZERO FALLBACK: no image tensor output. Available: "
            f"{[k for k, v in outputs.items() if hasattr(v, 'shape')]}"
        )

    output_cfg = get_output_processing(family)
    output_range = pkg.defaults.get("output_range", output_cfg.get("output_range", [-1.0, 1.0]))
    batch_axis = output_cfg.get("batch_axis", 0)
    channel_axis = output_cfg.get("channel_axis", 1)
    valid_channels = output_cfg.get("valid_channels", [1, 3, 4])
    bit_depth = output_cfg.get("bit_depth", 8)
    layout = output_cfg.get("layout", "CHW")

    processor = OutputProcessor.from_package(pkg)
    tensor = torch.select(final, batch_axis, 0).cpu().float()
    tensor = processor.process(tensor, output_range)
    tensor = tensor.clamp(0, 1)

    # save_image only fires when we've already picked image as the output
    # modality, so a "mode_dependent" layout means CHW here.
    if layout in ("CHW", "mode_dependent") and tensor.dim() == 3:
        actual_channel_axis = channel_axis - 1 if batch_axis < channel_axis else channel_axis
        if tensor.shape[actual_channel_axis] in valid_channels:
            tensor = tensor.permute(1, 2, 0)

    if bit_depth == 16:
        img_np = (tensor.numpy() * 65535).astype(np.uint16)
    else:
        img_np = (tensor.numpy() * 255).astype(np.uint8)

    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    Image.fromarray(img_np).save(output_path)
    return output_path


def save_video(
    outputs: Dict[str, Any],
    output_path: str,
    family: str,
    executor,
    pkg,
) -> str:
    """Save video tensor as H.264/mp4."""
    import numpy as np
    import torch

    from neurobrix.core.module.output_processor import OutputProcessor

    final = executor.get_final_output(outputs)
    if final is None:
        raise RuntimeError(
            f"ZERO FALLBACK: no video tensor output. Available: "
            f"{[k for k, v in outputs.items() if hasattr(v, 'shape')]}"
        )

    output_cfg = get_output_processing(family)
    output_range = pkg.defaults.get("output_range", output_cfg.get("output_range", [-1.0, 1.0]))
    batch_axis = output_cfg.get("batch_axis", 0)
    layout = output_cfg.get("layout", "CTHW")
    fps = pkg.defaults.get("fps", output_cfg.get("fps", 24))

    processor = OutputProcessor.from_package(pkg)
    tensor = torch.select(final, batch_axis, 0).cpu().float()
    tensor = processor.process(tensor, output_range)
    tensor = tensor.clamp(0, 1)

    if layout == "CTHW":
        frames = tensor.permute(1, 2, 3, 0).numpy()
    elif layout == "TCHW":
        frames = tensor.permute(0, 2, 3, 1).numpy()
    else:
        frames = tensor.permute(1, 2, 3, 0).numpy()

    frames_uint8 = (frames * 255).astype(np.uint8)
    _write_video_h264(output_path, frames_uint8, fps)
    return output_path


def save_output(
    outputs: Dict[str, Any],
    output_path: str,
    family: str,
    executor,
    pkg,
    mode: Optional[str] = None,
) -> str:
    """
    Universal save entry point. Dispatches based on family output_format.

    Used by both CLI cold path and serving warm path.
    """
    fmt = get_output_format(family)
    if fmt == "mode_dependent":
        ext = default_extension_for_mode(family, mode)
        fmt = ext.lstrip(".")

    if fmt == "txt":
        return save_text(outputs, output_path, executor)
    if fmt == "wav":
        return save_audio(outputs, output_path, pkg)
    if fmt in ("png", "jpg", "jpeg"):
        return save_image(outputs, output_path, family, executor, pkg)
    if fmt == "mp4":
        return save_video(outputs, output_path, family, executor, pkg)

    raise RuntimeError(
        f"ZERO FALLBACK: unknown output_format '{fmt}' for family '{family}'."
    )


# ---------------------------------------------------------------------------
# Result extraction — for serving JSON-RPC response
# ---------------------------------------------------------------------------


def extract_result(
    outputs: Dict[str, Any],
    family: str,
    executor,
) -> Dict[str, Any]:
    """
    Build serialization-friendly result dict from raw executor outputs.

    For text-output families (llm, vlm, multimodal-text, audio_llm, stt):
    returns {text, tokens}. For wav-output families (tts): returns
    {waveform_present: True} (caller then calls save_output to file).
    For image/video: returns {outputs} (caller saves to file).
    """
    fmt = get_output_format(family)
    if fmt == "mode_dependent":
        # multimodal in serving context: needs explicit mode resolution.
        # For now, treat as raw passthrough — CLI / save_output handles it.
        return {"outputs": outputs}

    if fmt == "txt":
        text = _extract_text(outputs, executor)
        tokens = _extract_token_count(outputs)
        return {"text": text or "", "tokens": tokens}
    if fmt == "wav":
        transcription = outputs.get("global.transcription")
        if transcription:
            return {"transcription": transcription}
        waveform = outputs.get("global.output_audio")
        if waveform is not None:
            return {"outputs": outputs, "waveform_present": True}
        return {"outputs": outputs}
    # image, video, etc. — raw passthrough, caller saves
    return {"outputs": outputs}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_tokens(outputs: Dict[str, Any]):
    """Lookup output tokens without booleanising tensors (`or` would fail)."""
    tokens = outputs.get("output_tokens")
    if tokens is None:
        tokens = outputs.get("global.output_tokens")
    return tokens


def _extract_text(outputs: Dict[str, Any], executor) -> Optional[str]:
    """Decode tokens to text via executor.modules['tokenizer'], or pass through transcription."""
    transcription = outputs.get("global.transcription")
    if transcription:
        return transcription

    tokens = _get_tokens(outputs)
    if tokens is None:
        return None

    if executor is not None and "tokenizer" in getattr(executor, "modules", {}):
        tokenizer = executor.modules["tokenizer"]
        if hasattr(tokenizer, "decode"):
            ids = tokens if isinstance(tokens, list) else tokens.flatten().tolist()
            return tokenizer.decode(ids, skip_special_tokens=True)

    return str(tokens if isinstance(tokens, list) else tokens.flatten().tolist())


def _extract_token_count(outputs: Dict[str, Any]) -> int:
    tokens = _get_tokens(outputs)
    if tokens is None:
        return 0
    if isinstance(tokens, list):
        return len(tokens)
    return int(tokens.shape[-1])


def _write_video_h264(output_path: str, frames_uint8, fps: float) -> None:
    """Write video as H.264/mp4 using ffmpeg."""
    import subprocess
    import imageio_ffmpeg

    _, H, W, _ = frames_uint8.shape
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        output_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.communicate(input=frames_uint8.tobytes())
