"""
Static Graph Flow Handler

ZERO SEMANTIC: Executes all components in topological order.
ZERO HARDCODE: All configuration from NBX container.

Handles models without iteration - single pass through all components.
"""

import gc
import torch
from typing import Any, Callable, Dict, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("static_graph")
class StaticGraphHandler(FlowHandler):
    """
    Flow handler for static graph execution (no loop).

    Executes all components in topological order without iteration.
    Used for simple models that don't need denoising loops.

    ZERO SEMANTIC: No domain knowledge.
    ZERO HARDCODE: Component order from topology.
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Any]
    ):
        """
        Initialize static graph handler.

        Args:
            ctx: FlowContext with all shared state
            execute_component_fn: Function to execute a component
        """
        super().__init__(ctx)
        self._execute_component = execute_component_fn

    def execute(self) -> Dict[str, Any]:
        """
        Execute static graph (no loop).

        Executes all components in order: pre_loop + loop.components + post_loop

        Returns:
            Dict of resolved variables/outputs
        """
        flow = self.ctx.pkg.topology.get("flow", {})

        # Execute all components in topological order
        all_components = (
            flow.get("pre_loop", []) +
            flow.get("loop", {}).get("components", []) +
            flow.get("post_loop", [])
        )

        for comp_name in all_components:
            self._execute_component(comp_name, "static", None)

            # Unload after each component
            self._unload_component(comp_name)

        return self.ctx.variable_resolver.resolve_all()

    def _unload_component(self, comp_name: str) -> None:
        """Unload component weights and clear memory."""
        executor = self.ctx.executors.get(comp_name)
        if executor:
            executor.unload_weights()
        gc.collect()
        torch.cuda.empty_cache()
