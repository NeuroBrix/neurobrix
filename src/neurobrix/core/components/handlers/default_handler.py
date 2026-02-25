"""
Default Component Handler

A no-op handler used when no specific handler matches the component.
All methods pass through without modification.
"""

from typing import Dict, Any

import torch

from ..base import ComponentHandler, ComponentConfig
from ..registry import register_handler


@register_handler("default")
class DefaultComponentHandler(ComponentHandler):
    """
    Default no-op handler for components without specific behavior.

    This handler passes through all inputs/outputs without modification.
    Used as a fallback when no specialized handler matches.
    """

    @classmethod
    def can_handle(cls, class_name: str, component_type: str) -> bool:
        """
        Default handler can handle anything, but is only used as fallback.

        Returns False so that specialized handlers are preferred.
        """
        return False
