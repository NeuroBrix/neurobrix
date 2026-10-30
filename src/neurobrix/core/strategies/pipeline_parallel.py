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

    def __init__(self, context: StrategyContext, strategy_name: str = "pipeline_parallel"):
        super().__init__(context, strategy_name)

        self.devices: List[str] = context.get_all_devices()
        self.use_async = True

        # Parse PP allocations from context (same format as FGP: fgp: prefix)
        self.pp_components: Dict[str, Dict[str, str]] = {}
        self.component_devices: Dict[str, List[str]] = {}

        for comp_name, alloc in context.allocations.items():
            if isinstance(alloc, dict):
                device_str = alloc.get('device', '')
                shard_map = alloc.get('shard_map', {})
            elif isinstance(alloc, tuple) and len(alloc) >= 2:
                device_str, shard_map = alloc[0], alloc[1]
            else:
                continue

            if isinstance(device_str, str) and device_str.startswith("fgp:"):
                device_list = device_str[4:].split(",")
                self.pp_components[comp_name] = shard_map
                self.component_devices[comp_name] = device_list
            else:
                if isinstance(device_str, str):
                    self.component_devices[comp_name] = [device_str]

        if not self.devices:
            raise RuntimeError("ZERO FALLBACK: No devices assigned by Prism for pipeline parallel")
        self.primary_device = self.devices[0]
        self._loaded_components: Set[str] = set()


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


