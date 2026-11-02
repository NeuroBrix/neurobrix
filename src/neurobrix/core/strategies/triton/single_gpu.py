"""Single GPU Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of `core/strategies/single_gpu.py`, rebased on
`TritonStrategy` so transfers/sync are NBXTensor-native. The logic is
identical (all components on one GPU, lazy/eager loading from the Prism
plan); only the transfer base differs (two-modes doctrine).
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy
from neurobrix.core.strategies.shared_steps import SingleGPUSteps, execute_whole_component   # the steps both branches share


class SingleGPUStrategy(SingleGPUSteps, TritonStrategy):
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

    execute_component = execute_whole_component     # the same execution as the compiled CPU strategy's (one body)

    def unload_weights(self, component_name: str) -> None:
        """Lazy mode unloads; eager mode keeps weights resident."""
        if self._eager:
            return
        super().unload_weights(component_name)


