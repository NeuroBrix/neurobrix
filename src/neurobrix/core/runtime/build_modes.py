"""Whether a container's BUILD serves the generation mode a request asks (the CLI's `run` and the serving door read the
same rule, never a model name).

A multimodal build is made for ONE generation type (`topology.flow.generation.type`): a request asking another mode is
refused by name, never run on the build's own path — Janus-Pro-7B served by chat completions ran its image
autoregression for a text request and answered the image tokens as text (" inqu grav Store European reform...",
both engines, 2026-11-13); `run --mode text` already refused it."""
from typing import Any, Dict, Optional

_MODE_GENERATION = {"text": "autoregressive_text", "image": "autoregressive_image"}


def build_refuses_mode(topology: Dict[str, Any], mode: Optional[str]) -> Optional[str]:
    """The sentence refusing `mode` on this build, or None when the build serves it (or declares no generation type).
    A build declaring `flow.image_gen` serves `image` through its diffusion leg, whatever its generation type."""
    flow = (topology or {}).get("flow") or {}
    built, asked = (flow.get("generation") or {}).get("type", ""), _MODE_GENERATION.get(mode or "")
    if not asked or not built or asked == built or (mode == "image" and flow.get("image_gen")):
        return None
    served = next((m for m, g in _MODE_GENERATION.items() if g == built), built)
    return (f"this build serves only --mode {served} (its generation type is '{built}'); a container built for "
            f"--mode {mode} serves that mode")
