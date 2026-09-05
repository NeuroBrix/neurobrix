"""
Flow Handlers Package

Provides execution flow handlers for different model architectures:
- IterativeProcessHandler: Diffusion models (denoising loop)
- StaticGraphHandler: Single-pass models
- ForwardPassHandler: Sequential transformer models
- AutoregressiveHandler: Token-by-token generation (LLM)
- EncoderDecoderEngine: Encoder-decoder cross-attention (Whisper)
- AudioLLMEngine: Audio-conditioned LLM (Voxtral, Granite, Canary)
- DualAREngine: Fish-Speech DualAR generation
- AudioEngine: Multi-stage audio pipeline (Kokoro, VibeVoice)
- RNNTEngine: RNNT transducer (Parakeet)
- TTSLLMEngine: Speech LM TTS (Chatterbox)

Usage:
    from neurobrix.core.flow import FlowContext, get_flow_handler

    ctx = FlowContext(...)
    handler = get_flow_handler("iterative_process", ctx)
    outputs = handler.execute()
"""

from .base import (
    FlowContext,
    FlowHandler,
    FLOW_REGISTRY,
    register_flow,
    get_flow_handler,
)

# The compiled (ATen) handlers are NOT imported here. They register
# themselves when their module is imported, and `get_flow_handler` imports
# the module of the flow type it is asked for (COMPILED_FLOW_MODULES in
# base.py). R33: a --triton run takes its handlers from neurobrix.triton.flow
# and must never load the ATen branch; the package exports below resolve
# lazily so `from neurobrix.core.flow import AutoregressiveHandler` still
# works for the compiled branch and its tests.
_HANDLER_EXPORTS = {
    "IterativeProcessHandler": "iterative_process",
    "StaticGraphHandler": "static_graph",
    "ForwardPassHandler": "forward_pass",
    "AutoregressiveHandler": "autoregressive",
    "AudioEngine": "audio",
    "EncoderDecoderEngine": "encoder_decoder",
    "AudioLLMEngine": "audio_llm",
    "VLMEngine": "vlm",
    "DualAREngine": "dual_ar",
    "TTSLLMEngine": "tts_llm",
    "NextTokenDiffusionEngine": "next_token_diffusion",
}


def __getattr__(name):
    module = _HANDLER_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


__all__ = [
    "FlowContext",
    "FlowHandler",
    "FLOW_REGISTRY",
    "register_flow",
    "get_flow_handler",
    "IterativeProcessHandler",
    "StaticGraphHandler",
    "ForwardPassHandler",
    "AutoregressiveHandler",
    "AudioEngine",
    "EncoderDecoderEngine",
    "AudioLLMEngine",
    "VLMEngine",
    "DualAREngine",
    "TTSLLMEngine",
    "NextTokenDiffusionEngine",
]
