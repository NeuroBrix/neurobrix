"""
Flow Handlers Package

Provides execution flow handlers for different model architectures:
- IterativeProcessHandler: Diffusion models (denoising loop)
- StaticGraphHandler: Single-pass models
- ForwardPassHandler: Sequential transformer models
- AutoregressiveHandler: Token-by-token generation (LLM)

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
]
