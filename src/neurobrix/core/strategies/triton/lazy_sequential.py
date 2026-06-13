"""Lazy Sequential Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/lazy_sequential.py. One component
resident at a time; peak GPU = max(single component), not sum(all). The
zero3 sub-strategy CPU-pinning is handled by the NBX weight loader (host
allocation via cudaMallocHost at load time), so the strategy-level pin is a
no-op here — unlike the torch path which pins via torch.Tensor.pin_memory().
"""

import logging
from typing import Dict, Optional, Any, Set

from ..base import StrategyContext
from .base import TritonStrategy

logger = logging.getLogger(__name__)


class LazySequentialStrategy(TritonStrategy):
    """Sequential component execution with lazy weight loading (triton path)."""

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
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for '{component_name}'")

        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            self._loaded_components.add(component_name)

        # zero3 sub-strategy: NBX weight loader already allocates host
        # weights in pinned memory (cudaMallocHost) for fast DMA — no
        # torch.pin_memory() needed. Mark as handled.
        comp_strategy = self._get_component_strategy(component_name)
        if comp_strategy == "zero3":
            self._pinned_components.add(component_name)

        # Ensure inputs on the component's device (no-op when already there).
        if inputs:
            device = self._get_component_device(component_name)
            inputs = self.transfer_dict(inputs, device)

        return executor.run(inputs or {})

    def _get_component_strategy(self, component_name: str) -> str:
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('strategy', 'single_gpu')
        return 'single_gpu'

    def _get_component_device(self, component_name: str) -> str:
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('device', 'cpu')
        elif isinstance(alloc, tuple):
            return alloc[0]
        return 'cpu'

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = self._get_component_device(component_name)
        return self.transfer_dict(inputs, device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        if target_device:
            return self.transfer_dict(outputs, target_device)
        return outputs

    def unload_weights(self, component_name: str) -> None:
        super().unload_weights(component_name)
        self._loaded_components.discard(component_name)
        self._pinned_components.discard(component_name)

    def cleanup(self) -> None:
        self._loaded_components.clear()
        self._pinned_components.clear()
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.empty_cache_pool()
