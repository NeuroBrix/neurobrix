"""
Single GPU Execution Strategy

All components execute on a single GPU.

Loading modes (from Prism plan.loading_mode):
- "lazy": Load component weights, execute, unload (one component in memory at a time)
- "eager": Load all components upfront, keep in memory (faster when GPU has headroom)

No tensor transfers needed (all on same device).
This is the fastest strategy when model fits on one GPU.
"""

from typing import Dict, Any, Optional, Set
import torch

from .base import ExecutionStrategy, StrategyContext


class SingleGPUStrategy(ExecutionStrategy):
    """
    Single GPU execution with configurable loading mode.

    All components are assigned to the same GPU by Prism.

    Loading modes (DATA-DRIVEN from Prism plan.loading_mode):
    - "lazy": Load/unload per component (default, saves memory)
    - "eager": Load all weights upfront, keep in memory (faster, uses more memory)
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "single_gpu"):
        super().__init__(context, strategy_name)

        # Validate: all components should be on same device
        devices = context.get_all_devices()
        self.device = devices[0] if devices else "cuda:0"

        # Loading mode from Prism plan (DATA-DRIVEN)
        # "eager" = load all weights upfront and keep in memory
        self._eager = context.loading_mode == "eager"
        self._loaded_components: Set[str] = set()

        pass

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute component on single GPU.

        Respects loading_mode:
        - "lazy": Load weights, execute (unload handled by flow handler)
        - "persistent": Load once, keep in memory for subsequent calls

        No tensor transfers needed (all on same device).
        """
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        # Load weights if not already loaded
        # In eager mode, track what's loaded to avoid redundant loads
        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            if self._eager:
                self._loaded_components.add(component_name)

        # Prepare inputs (ensure on correct device)
        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        # Execute
        outputs = executor.run(inputs or {})

        return outputs

    def unload_weights(self, component_name: str) -> None:
        """
        Unload weights for component.

        In eager mode, skip unloading (keep weights in memory).
        In lazy mode, delegate to base class to actually unload.
        """
        if self._eager:
            # Skip unload in eager mode
            return

        # Lazy mode: actually unload
        super().unload_weights(component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensure inputs are on the GPU.

        For single GPU, just move any CPU tensors to device.
        """
        return self.transfer_dict(inputs, self.device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle outputs from component.

        For single GPU, outputs are already on the correct device.
        No transfer needed.
        """
        # Single GPU: outputs stay on same device
        return outputs

    def unload_inactive_components(self, keep_component: str) -> None:
        """
        Unload all components except the specified one.

        In eager mode, skip unloading (weights stay in memory).
        In lazy mode, ensures memory is available for the next component.
        """
        if self._eager:
            # Skip unload in eager mode
            return

        for comp_name in self.context.component_executors:
            if comp_name != keep_component:
                self.unload_weights(comp_name)
