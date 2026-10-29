"""Triton Static Graph Flow Handler — zero torch dependency.

Ported from core/flow/static_graph.py. Executes all components in
topological order without iteration (single pass).

No torch imports in this file.
"""

from typing import Any, Callable, Dict, Optional

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.memory_pool import release_flow_memory


class TritonStaticGraphHandler:
    """
    Triton-mode static graph flow handler.

    Executes all components in topological order without iteration.
    Used for simple models that don't need denoising loops.

    ZERO SEMANTIC: No domain knowledge.
    ZERO HARDCODE: Component order from topology.
    """

    def __init__(
        self,
        ctx,
        execute_component_fn: Callable,
        *,
        unload_weights_fn: Callable[[str], None],
    ):
        self.ctx = ctx
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
