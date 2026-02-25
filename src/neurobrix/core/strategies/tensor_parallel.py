"""
Tensor Parallel Execution Strategy

Distributes weight files across multiple GPUs via shard_map.
Execution uses CompiledSequence multi-device path with automatic
device alignment at op boundaries.

Functionally equivalent to FGP but selected when component-level
distribution (PP) doesn't work and block-level distribution is needed
at tensor granularity. The solver distributes weight shards evenly
across TP GPUs; CompiledSequence handles cross-device transfers.

Use case: Component too large for single GPU but topology supports
multi-GPU execution (NVLink or PCIe P2P).
"""

from typing import Dict, Any, Optional, List, Set
import torch

from .base import ExecutionStrategy, StrategyContext


class TensorParallelStrategy(ExecutionStrategy):
    """
    Tensor Parallel execution across multiple GPUs.

    Weight files distributed via shard_map (from Prism solver).
    Execution through CompiledSequence multi-device path.
    Device alignment handled by _run_inner_multi_device fast path.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "tp"):
        super().__init__(context, strategy_name)

        # Parse TP vs regular component allocations
        self.tp_components: Dict[str, List[str]] = {}
        self.regular_components: Dict[str, str] = {}

        for comp_name, alloc_info in context.allocations.items():
            if isinstance(alloc_info, dict):
                device_str = alloc_info.get('device', '')
            else:
                device_str = alloc_info[0] if isinstance(alloc_info, tuple) else str(alloc_info)

            if device_str.startswith("tp:"):
                devices = device_str[3:].split(",")
                self.tp_components[comp_name] = devices
            else:
                self.regular_components[comp_name] = device_str

        self._loaded_components: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component — TP or regular."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for component '{component_name}'"
            )

        if component_name in self.tp_components:
            primary_device = self.tp_components[component_name][0]
        elif component_name in self.regular_components:
            primary_device = self.regular_components[component_name]
        else:
            primary_device = str(executor.device)

        # Transfer inputs to primary device
        if inputs:
            inputs = self.transfer_dict(inputs, primary_device, async_transfer=False)

        # Weights already loaded by RuntimeExecutor._ensure_weights_loaded() with shard_map.
        # CompiledSequence multi-device path handles cross-device execution.
        return executor.run(inputs or {})

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer inputs to component's primary device."""
        if component_name in self.tp_components:
            target = self.tp_components[component_name][0]
        elif component_name in self.regular_components:
            target = self.regular_components[component_name]
        else:
            return inputs
        return self.transfer_dict(inputs, target, async_transfer=False)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transfer outputs to target device if needed."""
        if target_device is None:
            return outputs
        return self.transfer_dict(outputs, target_device, async_transfer=False)

    def get_component_device(self, component_name: str) -> str:
        """Get primary device for component."""
        if component_name in self.tp_components:
            return self.tp_components[component_name][0]
        elif component_name in self.regular_components:
            return self.regular_components[component_name]
        raise RuntimeError(f"ZERO FALLBACK: No device for '{component_name}'")

    def cleanup(self) -> None:
        """Release resources."""
        self._loaded_components.clear()
