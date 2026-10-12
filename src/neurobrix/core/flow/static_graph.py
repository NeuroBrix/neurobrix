"""
Static Graph Flow Handler

ZERO SEMANTIC: Executes all components in topological order.
ZERO HARDCODE: All configuration from NBX container.

Handles models without iteration - single pass through all components.
"""

from neurobrix.core.memory.manager import release_flow_memory
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
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Any],
        *,
        unload_weights_fn: Callable[[str], None]
    ):
        """
        Initialize static graph handler.

        Args:
            ctx: FlowContext with all shared state
            execute_component_fn: Function to execute a component
        """
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._unload_component_weights = unload_weights_fn

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
        """Unload component weights and clear memory (skip in serve mode)."""
        if self.ctx.persistent_mode:
            return
        # Through the engine executor's own brick — the strategy's uninstall FIRST (a zero3 ratchet
        # evicts its resident blocks nowhere else), then the executor's unload. Calling
        # `executor.unload_weights()` here left mochi-1-preview's 18 938 MB of resident transformer
        # blocks under the post-loop VAE on a 32 GB V100 (aten.convolution::27, 2026-10-12).
        if comp_name in self.ctx.executors:
            self._unload_component_weights(comp_name)
        release_flow_memory(self.ctx.primary_device)
