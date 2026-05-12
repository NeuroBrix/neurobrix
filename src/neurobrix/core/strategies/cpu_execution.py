"""
CPU Execution Strategy — Doctrine R35 last-resort cascade

All components placed on host RAM. Compute via PyTorch ATen native CPU
dispatcher for the `compiled` and `sequential` modes (branch A). The
`triton` and `triton_sequential` modes route to PyTorch CPU when no
Triton-CPU runtime is integrated yet (branch B Triton-CPU integration
is a separate chantier; until then, those modes fall back to ATen CPU
via the existing dispatcher with a clear log note).

Activation budget: validated by the solver
(`_try_cpu_execution`) at planning time — `sum(component peaks) <=
cpu.ram_mb * 0.7`. The strategy itself is a simple per-component
device-assignment to "cpu"; no cross-device transfer; no dynamic
swap.

Loading modes (DATA-DRIVEN from Prism plan.loading_mode):
- "lazy": Load weights, execute (unload handled by flow handler when
  the next component starts). On CPU RAM is plentiful so eviction is
  cheap; this is the default.
- "eager": Load all components upfront, keep in memory. Faster on
  small models (e.g. TinyLlama) where the whole graph fits with
  headroom.

R34 model-agnostic: no model names anywhere. Discrimination via
hardware profile (`cpu.cores`, `cpu.features`) and component memory
estimates only.

P-PRISM-NEVER-REFUSE v2 — 2026-05-12.
"""

from typing import Dict, Any, Optional, Set

from .base import ExecutionStrategy, StrategyContext


class CPUExecutionStrategy(ExecutionStrategy):
    """All components execute on the host CPU.

    Identical control flow to `SingleGPUStrategy` but the device is
    "cpu" instead of "cuda:N". The PyTorch ATen dispatcher already
    routes ops to MKL/oneDNN automatically for `device='cpu'` tensors;
    threading is pre-configured via `apply_cpu_config` at CLI / serving
    layer entry (reads `cpu.cores` from the hardware profile, mandate
    R34 compliant — no hardcoded thread count).
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "cpu_execution"):
        super().__init__(context, strategy_name)
        # All components share the same logical "cpu" device.
        self.device = "cpu"
        # Loading mode from Prism plan (DATA-DRIVEN)
        self._eager = context.loading_mode == "eager"
        self._loaded_components: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component on CPU.

        Identical pattern to SingleGPUStrategy: load weights (lazy or
        eager), prepare inputs (ensure CPU tensors), execute.
        """
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
        """Unload weights for component.

        In eager mode, skip unloading. In lazy mode, delegate to base
        which frees the underlying tensor storage.
        """
        if self._eager:
            return
        super().unload_weights(component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ensure inputs are on CPU (the only device this strategy uses)."""
        return self.transfer_dict(inputs, self.device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pass-through: outputs are already on CPU."""
        return outputs

    def unload_inactive_components(self, keep_component: str) -> None:
        """Unload all components except the specified one.

        In eager mode, skip. In lazy mode, free RAM held by inactive
        components — useful for very large models where peak RAM
        consumption matters.
        """
        if self._eager:
            return
        for comp_name in self.context.component_executors:
            if comp_name != keep_component:
                self.unload_weights(comp_name)
