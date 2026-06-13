"""CPU Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/cpu_execution.py. All components
on host. NOTE: NBX Triton kernels are GPU-only; a true Triton-CPU compute
backend is a separate chantier. This class preserves the strategy contract
(device routing to "cpu") torch-free so the triton branch is complete; the
runtime surfaces a clear error if GPU-less NBX compute is actually invoked.
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy


class CPUExecutionStrategy(TritonStrategy):
    """All components execute on the host (triton-path contract mirror)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "cpu_execution"):
        super().__init__(context, strategy_name)
        self.device = "cpu"
        self._eager = context.loading_mode == "eager"
        self._loaded_components: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for '{component_name}'")

        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            if self._eager:
                self._loaded_components.add(component_name)

        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        return executor.run(inputs or {})

    def unload_weights(self, component_name: str) -> None:
        if self._eager:
            return
        super().unload_weights(component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.transfer_dict(inputs, self.device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        return outputs

    def unload_inactive_components(self, keep_component: str) -> None:
        if self._eager:
            return
        for comp_name in self.context.component_executors:
            if comp_name != keep_component:
                self.unload_weights(comp_name)
