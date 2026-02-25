"""
Component Handler Registry

Provides a factory for creating component handlers based on
component type and class name.

Pattern follows the existing SchedulerFactory and FlowHandler registry.

ZERO FALLBACK: Unknown components get DefaultComponentHandler.
"""

from typing import Dict, Type, Optional

from .base import ComponentHandler, ComponentConfig
from .config_loader import load_component_config


# Global registry of component handlers
HANDLER_REGISTRY: Dict[str, Type[ComponentHandler]] = {}


def register_handler(handler_type: str):
    """
    Decorator to register a component handler.

    Usage:
        @register_handler("vae")
        class VAEComponentHandler(ComponentHandler):
            ...

    Args:
        handler_type: The type string to register (e.g., "vae", "transformer")

    Returns:
        Decorator function
    """
    def decorator(cls: Type[ComponentHandler]):
        HANDLER_REGISTRY[handler_type] = cls
        return cls
    return decorator


def get_component_handler(
    component_name: str,
    component_type: str,
    cache_path: str
) -> ComponentHandler:
    """
    Factory to create appropriate handler for a component.

    Strategy:
    1. Load config from cache (profile.json + runtime.json)
    2. Match handler by can_handle() on class_name
    3. Match handler by component_type
    4. Fall back to DefaultComponentHandler

    Args:
        component_name: Name of the component (e.g., "vae", "transformer")
        component_type: Type hint (e.g., "vae", "transformer", "text_encoder")
        cache_path: Path to extracted NBX cache

    Returns:
        Configured ComponentHandler instance

    Raises:
        RuntimeError: If config loading fails (ZERO FALLBACK)
    """
    # Load normalized config from cache
    config = load_component_config(cache_path, component_name)

    # Strategy 1: Check can_handle() for each registered handler
    for handler_class in HANDLER_REGISTRY.values():
        if handler_class.can_handle(config.class_name, config.component_type):
            return handler_class(config, cache_path)

    # Strategy 2: Direct match by component_type
    if component_type in HANDLER_REGISTRY:
        handler_class = HANDLER_REGISTRY[component_type]
        return handler_class(config, cache_path)

    # Strategy 3: Match by inferred component_type from config
    if config.component_type in HANDLER_REGISTRY:
        handler_class = HANDLER_REGISTRY[config.component_type]
        return handler_class(config, cache_path)

    # Strategy 4: Fall back to DefaultComponentHandler
    from .handlers.default_handler import DefaultComponentHandler
    return DefaultComponentHandler(config, cache_path)
