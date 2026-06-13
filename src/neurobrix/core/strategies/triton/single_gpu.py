"""Single GPU Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of `core/strategies/single_gpu.py`, rebased on
`TritonStrategy` so transfers/sync are NBXTensor-native. The logic is
identical (all components on one GPU, lazy/eager loading from the Prism
plan); only the transfer base differs (two-modes doctrine).
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy


class SingleGPUStrategy(TritonStrategy):
    """Single GPU execution with configurable loading mode (triton path)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "single_gpu"):
        super().__init__(context, strategy_name)

        devices = context.get_all_devices()
        if not devices:
            raise RuntimeError("ZERO FALLBACK: No devices assigned by Prism")
        self.device = devices[0]

        # Loading mode from Prism plan (DATA-DRIVEN). "eager" keeps all
        # weights resident; "lazy" loads/unloads per component.
        self._eager = context.loading_mode == "eager"
        self._loaded_components: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component on the single assigned GPU (no cross-device
        transfer)."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            if self._eager:
                self._loaded_components.add(component_name)

        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        outputs = executor.run(inputs or {})
        return outputs

    def unload_weights(self, component_name: str) -> None:
        """Lazy mode unloads; eager mode keeps weights resident."""
        if self._eager:
            return
        super().unload_weights(component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ensure inputs are on the assigned GPU (NBXTensor transfer)."""
        return self.transfer_dict(inputs, self.device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Single GPU: outputs already on the correct device."""
        return outputs

    def unload_inactive_components(self, keep_component: str) -> None:
        """Free every component except `keep_component` (lazy mode only)."""
        if self._eager:
            return
        for comp_name in self.context.component_executors:
            if comp_name != keep_component:
                self.unload_weights(comp_name)
