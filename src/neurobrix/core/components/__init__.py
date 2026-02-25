"""
NeuroBrix Component Handlers - Data-Driven Component Abstraction Layer

This module provides handlers for different component types (VAE, Transformer, TextEncoder)
that read configuration from profile.json and runtime.json instead of hardcoding.

ZERO HARDCODE: All component-specific constants are derived from config files.
ZERO FALLBACK: Missing configs raise explicit errors.

Usage:
    from neurobrix.core.components import get_component_handler, ComponentConfig

    handler = get_component_handler("vae", "vae", cache_path)
    scaled_inputs = handler.transform_inputs(inputs, "post_loop")
"""

from .base import ComponentConfig, ComponentHandler
from .registry import get_component_handler, register_handler, HANDLER_REGISTRY
from .config_loader import load_component_config

# Import handlers to register them
from .handlers import (
    DefaultComponentHandler,
    VAEComponentHandler,
    TransformerComponentHandler,
    TextEncoderComponentHandler,
)

__all__ = [
    # Core classes
    "ComponentConfig",
    "ComponentHandler",
    # Factory
    "get_component_handler",
    "register_handler",
    "HANDLER_REGISTRY",
    # Config
    "load_component_config",
    # Handlers
    "DefaultComponentHandler",
    "VAEComponentHandler",
    "TransformerComponentHandler",
    "TextEncoderComponentHandler",
]
