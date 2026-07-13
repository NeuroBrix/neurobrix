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

# Import flow handlers to register them
from .iterative_process import IterativeProcessHandler
from .static_graph import StaticGraphHandler
from .forward_pass import ForwardPassHandler
from .autoregressive import AutoregressiveHandler
from .audio import AudioEngine
from .encoder_decoder import EncoderDecoderEngine
from .audio_llm import AudioLLMEngine
from .vlm import VLMEngine
from .dual_ar import DualAREngine
from .tts_llm import TTSLLMEngine
from .next_token_diffusion import NextTokenDiffusionEngine

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
