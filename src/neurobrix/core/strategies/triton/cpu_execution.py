"""CPU Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/cpu_execution.py. All components
on host. NOTE: NBX Triton kernels are GPU-only; a true Triton-CPU compute
backend is a separate workstream. This class preserves the strategy contract
(device routing to "cpu") torch-free so the triton branch is complete; the
runtime surfaces a clear error if GPU-less NBX compute is actually invoked.
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy
from neurobrix.core.strategies.shared_steps import CPUExecutionSteps   # the steps both branches share


class CPUExecutionStrategy(CPUExecutionSteps, TritonStrategy):
    """All components execute on the host (triton-path contract mirror)."""

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


