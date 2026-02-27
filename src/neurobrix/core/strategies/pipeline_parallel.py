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

from .base import ExecutionStrategy, StrategyContext


class PipelineParallelStrategy(ExecutionStrategy):
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

        self.primary_device = self.devices[0] if self.devices else "cuda:0"
        self._loaded_components: Set[str] = set()

    def is_pp_component(self, component_name: str) -> bool:
        """Check if component is PP-distributed across GPUs."""
        return component_name in self.pp_components

    def get_shard_map(self, component_name: str) -> Dict[str, str]:
        """Get shard map for PP component."""
        return self.pp_components.get(component_name, {})

    def get_component_devices(self, component_name: str) -> List[str]:
        """Get list of devices for component."""
        return self.component_devices.get(component_name, [self.primary_device])

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

    def _load_pp_weights(self, component_name: str) -> None:
        """Load weights for PP component using shard map."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            return

        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if weights_loaded:
            return

        shard_map = self.get_shard_map(component_name)

        nbx_path = None
        if hasattr(self.context, 'runtime_package') and self.context.runtime_package:
            nbx_path = getattr(self.context.runtime_package, 'nbx_path', None)

        if nbx_path:
            if shard_map:
                executor.load_weights(nbx_path, component_name, shard_map)
            else:
                executor.load_weights(nbx_path, component_name)

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs — transfer to first device in the PP chain."""
        devices = self.get_component_devices(component_name)
        target_device = devices[0] if devices else self.primary_device
        return self.transfer_dict(inputs, target_device, self.use_async)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle outputs — transfer from last PP device to target."""
        if target_device:
            source_device = None
            for val in outputs.values():
                if isinstance(val, torch.Tensor):
                    source_device = str(val.device)
                    break
            if source_device and source_device != target_device:
                outputs = self.transfer_dict(outputs, target_device, self.use_async)
                if self.use_async:
                    self.synchronize_device(target_device)
        return outputs

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
                torch.cuda.empty_cache()

    def get_embedding_device(self, component_name: str) -> str:
        """Get device where embedding layer is (first device in PP chain)."""
        devices = self.get_component_devices(component_name)
        return devices[0] if devices else self.primary_device

    def get_head_device(self, component_name: str) -> str:
        """Get device where output head is (last device in PP chain)."""
        devices = self.get_component_devices(component_name)
        return devices[-1] if devices else self.primary_device

    def synchronize_all(self) -> None:
        """Synchronize all devices used in PP execution."""
        for device in self.devices:
            self.synchronize_device(device)
