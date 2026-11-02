"""
Forward Pass Flow Handler

ZERO SEMANTIC: Executes components in order from topology.flow.order.
ZERO HARDCODE: All configuration from NBX container.

Handles non-iterative models like encoders, decoders, multimodal models.
"""

from neurobrix.core.memory.manager import release_flow_memory
from typing import Any, Dict

from .iterative_steps import IterativeSteps
from .base import ComponentFlowHandler, register_flow


@register_flow("forward_pass")
class ForwardPassHandler(ComponentFlowHandler):
    """
    Flow handler for forward pass execution.

    Executes components in sequence from topology.flow.order.
    Handles preprocessing (tokenization) automatically if needed.

    ZERO SEMANTIC: No domain knowledge - just executes in order.
    ZERO HARDCODE: Component order from topology.
    """

    def execute(self) -> Dict[str, Any]:
        """
        Execute forward_pass flow for non-iterative models.

        1. Preprocess inputs (tokenization if needed)
        2. Execute components in order
        3. Skip components with unavailable inputs

        Returns:
            Dict of resolved variables/outputs
        """
        flow = self.ctx.pkg.topology.get("flow", {})
        component_order = flow.get("order", [])

        if not component_order:
            raise RuntimeError(
                "ZERO FALLBACK: forward_pass requires topology.flow.order.\n"
                "The order array must list components in execution sequence."
            )

        # Preprocess inputs (tokenization)
        self._preprocess_inputs()

        for comp_name in component_order:
            if comp_name not in self.ctx.executors:
                continue

            # Check if inputs are available
            try:
                comp_inputs = self._resolve_component_inputs(comp_name)
            except RuntimeError:
                continue

            # Load weights
            self._ensure_weights_loaded(comp_name)

            try:
                self._execute_component(comp_name, "forward", None)
            except RuntimeError as e:
                if "not found in resolved" in str(e) or "not resolved" in str(e):
                    continue
                raise

            # Unload weights (skip in serve mode — keep resident for next request)
            if not self.ctx.persistent_mode:
                self._unload_component_weights(comp_name)
                release_flow_memory(self.ctx.primary_device)
        return self.ctx.variable_resolver.resolve_all()

    # the prompt tokenized into input_ids / attention_mask: the iterative handlers' step, the same body
    _preprocess_inputs = IterativeSteps._preprocess_inputs
