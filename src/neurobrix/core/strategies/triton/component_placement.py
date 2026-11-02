"""Component Placement Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/component_placement.py, rebased on
TritonStrategy. Whole components are distributed across GPUs; activations
move across devices at component boundaries through NBXTensor.to_cuda
(via the TritonStrategy transfer helpers). Two-modes doctrine: same
contract as the pytorch strategy, NBXTensor compute path.
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy
from neurobrix.core.strategies.shared_steps import ComponentPlacementLazySteps, ComponentPlacementSteps   # the steps both branches share


class ComponentPlacementStrategy(ComponentPlacementSteps, TritonStrategy):
    """Component placement across multiple GPUs (triton path)."""


    def transfer_model_output(self, model_output: Any, state_tensor: Any) -> Any:
        """Move the model output (NBXTensor) to the state tensor's device for
        the diffusion loop's residual update. NBXTensor device identity is
        carried by `_device_idx` (no torch `.device`)."""
        src_idx = getattr(model_output, "_device_idx", 0)
        dst_idx = getattr(state_tensor, "_device_idx", 0)
        if src_idx == dst_idx:
            return model_output
        target_device = f"cuda:{dst_idx}"
        transferred = self.transfer_tensor(model_output, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred


class ComponentPlacementLazyStrategy(ComponentPlacementLazySteps, ComponentPlacementStrategy):
    """Component placement with lazy weight swapping between phases (triton)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement_lazy"):
        super().__init__(context, strategy_name)
        self._phase_loaded: Set[str] = set()


