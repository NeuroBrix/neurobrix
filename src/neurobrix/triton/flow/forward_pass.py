"""Triton Forward Pass Flow Handler — zero torch dependency.

Ported from core/flow/forward_pass.py. Executes components in order
from topology.flow.order. All tensor ops via NBXTensor + kernel wrappers.

No torch imports in this file.
"""

import gc
import numpy as np
from typing import Any, Callable, Dict, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class TritonForwardPassHandler:
    """
    Triton-mode forward pass flow handler.

    Executes components in sequence from topology.flow.order.
    Handles preprocessing (tokenization) automatically if needed.

    ZERO SEMANTIC: No domain knowledge - just executes in order.
    ZERO HARDCODE: Component order from topology.
    """

    def __init__(
        self,
        ctx,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        self.ctx = ctx
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

            # Unload weights (skip in serve mode)
            if not self.ctx.persistent_mode:
                self._unload_component_weights(comp_name)
                gc.collect()

        return self.ctx.variable_resolver.resolve_all()

    def _preprocess_inputs(self) -> None:
        """
        Preprocess user inputs for forward_pass models.

        Delegates to TextProcessor for unified tokenization.
        Tokenizer output converted to NBXTensor for triton mode.
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
            # TextProcessor returns torch tensors; convert to NBXTensor
            input_ids, attention_mask = tp.tokenize_for_diffusion(prompt, device)

            # Convert torch tensors to NBXTensor if needed
            input_ids = _ensure_nbx_tensor(input_ids)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            if attention_mask is not None:
                attention_mask = _ensure_nbx_tensor(attention_mask)
                self.ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
                self.ctx.variable_resolver.resolved["attention_mask"] = attention_mask

        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}") from e


def _ensure_nbx_tensor(tensor) -> NBXTensor:
    """Convert a tensor to NBXTensor if it isn't already.

    Handles torch.Tensor → numpy → NBXTensor conversion at the boundary.
    """
    if isinstance(tensor, NBXTensor):
        return tensor
    # torch.Tensor boundary: convert via numpy
    if hasattr(tensor, 'cpu'):
        arr = tensor.detach().cpu().numpy()
        return NBXTensor.from_numpy(arr)
    if isinstance(tensor, np.ndarray):
        return NBXTensor.from_numpy(tensor)
    raise TypeError(f"Cannot convert {type(tensor)} to NBXTensor")
