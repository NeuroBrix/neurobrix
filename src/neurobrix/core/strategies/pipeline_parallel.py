"""
Pipeline Parallel Execution Strategy

Per-layer sequential fill across multiple GPUs (like Accelerate device_map="auto").
Layers 0..N on GPU0, N+1..M on GPU1, etc. Only N-1 boundary crossings for N GPUs.

This is the optimal multi-GPU strategy for large LLMs: layers are sequential
by nature, so placing them sequentially across GPUs minimizes cross-device
transfers. Each forward pass crosses GPU boundaries only at the layer split points.

Uses the same FGP runtime path (shard_map + multi-device CompiledSequence)
but with sequential fill allocation instead of scattered best-fit.
"""

from typing import Dict, Any, Optional, List, Set
import torch
from neurobrix.core.device_utils import device_empty_cache

from .base import ExecutionStrategy, StrategyContext
from neurobrix.core.strategies.shared_steps import PipelineParallelSteps   # the steps both branches share


class PipelineParallelStrategy(PipelineParallelSteps, ExecutionStrategy):
    """
    Pipeline parallel execution across multiple GPUs.

    Layers distributed sequentially: layers 0-14 on GPU0, 15-27 on GPU1, etc.
    Minimal cross-device transfers — only at layer boundary split points.

    Uses fgp: device prefix and shard_map for weight loading, same as
    BlockScatterStrategy, but the allocation pattern is sequential (greedy fill)
    instead of scattered (best-fit-decreasing).
    """


    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component in pipeline parallel mode."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for '{component_name}'")

        is_pp = self.is_pp_component(component_name)
        devices = self.get_component_devices(component_name)

        if component_name not in self._loaded_components:
            self._load_pp_weights(component_name)
            self._loaded_components.add(component_name)

        if is_pp and hasattr(executor, '_is_fgp_mode'):
            executor._is_fgp_mode = True

        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        return executor.run(inputs or {})


    def unload_weights(self, component_name: str) -> None:
        """Unload weights for PP component."""
        executor = self.context.component_executors.get(component_name)
        if executor is not None:
            weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
            if weights_loaded:
                devices = self.get_component_devices(component_name)
                if hasattr(executor, '_is_fgp_mode'):
                    executor._is_fgp_mode = False
                executor.unload_weights()
                self._loaded_components.discard(component_name)
                for device in devices:
                    self.synchronize_device(device)
                device_empty_cache(device)


