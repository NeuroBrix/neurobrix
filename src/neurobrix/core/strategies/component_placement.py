"""
Component Placement Execution Strategy

Whole components distributed across multiple GPUs.
Each component executes on its assigned GPU with tensor transfers between.

Two variants:
- ComponentPlacementStrategy (component_placement): Eager — weights stay in VRAM permanently.
- ComponentPlacementLazyStrategy (component_placement_lazy): Lazy — weights swap in/out
  between execution phases. Used when total model exceeds combined VRAM but each
  phase (pre_loop, loop, post_loop) fits.

Key features:
- Cross-device tensor transfers (automatic)
- Async transfers when available
- State tracking for efficient transfers
"""

from typing import Dict, Any, Optional, Set
import torch
from neurobrix.core.device_utils import device_empty_cache

from .base import ExecutionStrategy, StrategyContext
from neurobrix.core.strategies.shared_steps import ComponentPlacementLazySteps, ComponentPlacementSteps   # the steps both branches share


class ComponentPlacementStrategy(ComponentPlacementSteps, ExecutionStrategy):
    """
    Component placement across multiple GPUs.

    Each component (text_encoder, transformer, vae) is on a different GPU.
    Tensors are automatically transferred between devices at component boundaries.
    """


    def transfer_model_output(self, model_output: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
        """Transfer model output to same device as state tensor."""
        if model_output.device == state_tensor.device:
            return model_output
        target_device = str(state_tensor.device)
        transferred = self.transfer_tensor(model_output, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred


class ComponentPlacementLazyStrategy(ComponentPlacementLazySteps, ComponentPlacementStrategy):
    """
    Component placement with lazy weight swapping between phases.

    Weights are loaded before each execution phase and unloaded after.
    Allows models where total weights exceed combined VRAM, as long as
    each phase (pre_loop, loop, post_loop) fits.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement_lazy"):
        super().__init__(context, strategy_name)
        self._phase_loaded: Set[str] = set()


