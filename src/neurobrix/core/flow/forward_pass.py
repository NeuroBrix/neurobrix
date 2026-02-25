"""
Forward Pass Flow Handler

ZERO SEMANTIC: Executes components in order from topology.flow.order.
ZERO HARDCODE: All configuration from NBX container.

Handles non-iterative models like encoders, decoders, multimodal models.
"""

import gc
import torch
from typing import Any, Callable, Dict, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("forward_pass")
class ForwardPassHandler(FlowHandler):
    """
    Flow handler for forward pass execution.

    Executes components in sequence from topology.flow.order.
    Handles preprocessing (tokenization) automatically if needed.

    ZERO SEMANTIC: No domain knowledge - just executes in order.
    ZERO HARDCODE: Component order from topology.
    """

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Any],
        resolve_inputs_fn: Callable[[str], Dict[str, Any]],
        ensure_weights_fn: Callable[[str], None],
        unload_weights_fn: Callable[[str], None]
    ):
        """
        Initialize forward pass handler.

        Args:
            ctx: FlowContext with all shared state
            execute_component_fn: Function to execute a component
            resolve_inputs_fn: Function to resolve component inputs
            ensure_weights_fn: Function to ensure weights are loaded
            unload_weights_fn: Function to unload component weights
        """
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

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

            # Unload weights
            self._unload_component_weights(comp_name)
            gc.collect()
            torch.cuda.empty_cache()
        return self.ctx.variable_resolver.resolve_all()

    def _preprocess_inputs(self) -> None:
        """
        Preprocess user inputs for forward_pass models.

        Delegates to TextProcessor for unified tokenization
        (handles CHI, chat templates, max_length cascade).
        """
        prompt = None
        for key in ["global.prompt", "prompt"]:
            if key in self.ctx.variable_resolver.resolved:
                prompt = self.ctx.variable_resolver.resolved[key]
                break

        if prompt is None:
            return

        if "tokenizer" not in self.ctx.modules:
            return

        from neurobrix.core.module.text.processor import TextProcessor

        tp = TextProcessor(
            tokenizer=self.ctx.modules["tokenizer"],
            defaults=self.ctx.pkg.defaults,
            topology=self.ctx.pkg.topology,
            variable_resolver=self.ctx.variable_resolver,
        )

        device = self.ctx.primary_device

        try:
            input_ids, attention_mask = tp.tokenize_for_diffusion(prompt, device)

            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            if attention_mask is not None:
                self.ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
                self.ctx.variable_resolver.resolved["attention_mask"] = attention_mask

        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}") from e
