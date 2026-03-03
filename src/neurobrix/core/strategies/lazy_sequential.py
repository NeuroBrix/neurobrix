"""
Lazy Sequential Strategy - One Component at a Time

Loads one component to GPU, executes, unloads, then next.
Peak GPU memory = max(single component), not sum(all).

When a component within lazy_sequential has zero3 sub-strategy
(from Prism recursive cascade), weights stay on CPU and execute
via CompiledSequence multi-device path.

ZERO HARDCODE: Loading mode from Prism plan.
"""

import logging
import torch
from typing import Dict, Optional, Any, Set

from .base import ExecutionStrategy, StrategyContext

logger = logging.getLogger(__name__)


class LazySequentialStrategy(ExecutionStrategy):
    """
    Sequential component execution with lazy weight loading.

    Each component loads weights, executes, then unloads to make room
    for the next. Handles mixed per-component strategies: if Prism's
    recursive cascade assigns zero3 to a component, its weights stay
    on CPU with pinned memory.

    Peak GPU = max(single component weight + activation), not sum(all).
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "lazy_sequential"):
        super().__init__(context, strategy_name)
        self._loaded_components: Set[str] = set()
        self._pinned_components: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute one component, managing memory for sequential use."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        # Load weights if not loaded
        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            self._loaded_components.add(component_name)

        # If this component has zero3 sub-strategy, pin CPU weights
        comp_strategy = self._get_component_strategy(component_name)
        if comp_strategy == "zero3" and component_name not in self._pinned_components:
            self._pin_cpu_weights(component_name, executor)
            self._pinned_components.add(component_name)

        # Prepare inputs on correct device
        if inputs:
            device = self._get_component_device(component_name)
            inputs = self.transfer_dict(inputs, device)

        return executor.run(inputs or {})

    def _get_component_strategy(self, component_name: str) -> str:
        """Get per-component strategy from allocation."""
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('strategy', 'single_gpu')
        return 'single_gpu'

    def _get_component_device(self, component_name: str) -> str:
        """Get device for component."""
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('device', 'cpu')
        elif isinstance(alloc, tuple):
            return alloc[0]
        return 'cpu'

    def _pin_cpu_weights(self, component_name: str, executor: Any) -> None:
        """Pin CPU weights for zero3 sub-components."""
        weights = getattr(executor, '_weights', None)
        if not weights:
            return

        pinned = 0
        total_mb = 0.0
        for name, tensor in list(weights.items()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                total_mb += size_mb
                if not tensor.is_pinned():
                    weights[name] = tensor.contiguous().pin_memory()
                    pinned += 1

        if pinned > 0:
            logger.info(
                f"[LazySeq] {component_name}: pinned {pinned} CPU weights "
                f"({total_mb:.0f}MB)"
            )

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer inputs to component's device."""
        device = self._get_component_device(component_name)
        return self.transfer_dict(inputs, device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transfer outputs if next component is on different device."""
        if target_device:
            return self.transfer_dict(outputs, target_device)
        return outputs

    def unload_weights(self, component_name: str) -> None:
        """Unload component weights (lazy mode — always unload)."""
        super().unload_weights(component_name)
        self._loaded_components.discard(component_name)
        self._pinned_components.discard(component_name)

    def cleanup(self) -> None:
        """Release all resources."""
        self._loaded_components.clear()
        self._pinned_components.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
