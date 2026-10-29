"""
Output Extractor

ZERO HARDCODE: Uses declared interface.outputs from topology.
ZERO SEMANTIC: Minimal domain knowledge - uses topology connections.

Extracts primary outputs and stores outputs in variable resolver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
from neurobrix.core.runtime.tensor_compat import is_torch_tensor
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Primary-output surface names, in preference order — the SINGLE source of
# truth for "which named output is a component's principal output". Consumed
# by the output_0 alias pass below and by GraphExecutor's hidden-states
# extraction (`_primary_output_tids`). Auxiliary named outputs (e.g. a
# mid-stack hidden tap) must never shadow these.
PRIMARY_OUTPUT_NAMES = ("output", "last_hidden_state", "logits", "sample")

if TYPE_CHECKING:
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver


class OutputExtractor:
    """
    Extracts and routes component outputs.

    This class is responsible for:
    1. Extracting primary output from component output dict
    2. Storing outputs in variable resolver with semantic aliases
    3. Finding encoder hidden states via topology connections
    4. Extracting specific output types (hidden_states, logits, etc.)

    ZERO HARDCODE: Uses topology.interface.outputs for primary output.
    ZERO SEMANTIC: Follows topology connections, minimal domain knowledge.
    """

    def __init__(
        self,
        topology: Dict[str, Any],
        variable_resolver: 'VariableResolver'
    ):
        """
        Initialize output extractor.

        Args:
            topology: Full topology dict with component interfaces
            variable_resolver: VariableResolver for storing outputs
        """
        self._topology = topology
        self._variable_resolver = variable_resolver

    def extract_primary_output(self, comp_name: str, output: Any) -> Any:
        """
        Extract primary output from component output.

        ZERO HEURISTIC: Uses declared interface.outputs or semantic names only.
        No "largest tensor" or dimensional guessing.

        Args:
            comp_name: Component name
            output: Raw output from executor (dict or tensor)

        Returns:
            Primary output tensor or value
        """
        if not isinstance(output, dict):
            return output

        # Get declared outputs from topology
        comp_info = self._topology.get("components", {}).get(comp_name, {})
        interface = comp_info.get("interface", {})
        declared_outputs = interface.get("outputs", [])

        # Strategy 1: Look for semantic output names (not tensor IDs)
        semantic_outputs = {k: v for k, v in output.items() if not k.startswith("T_")}

        # Check if any declared output matches semantic outputs
        for out_name in declared_outputs:
            if out_name in output:
                return output[out_name]
            # Check semantic keys for common patterns
            for sem_key in semantic_outputs:
                if sem_key == out_name:
                    return semantic_outputs[sem_key]

        # If we have semantic outputs but no match with declared, use first semantic
        if semantic_outputs:
            primary_key = next(iter(semantic_outputs.keys()))
            return semantic_outputs[primary_key]

        # ZERO FALLBACK: No heuristics - crash with helpful message
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine primary output for '{comp_name}'.\n"
            f"Declared outputs: {declared_outputs}\n"
            f"Actual keys: {list(output.keys())[:10]}...\n"
            f"Semantic keys (non T_): {list(semantic_outputs.keys())}\n"
            "Fix: Ensure topology.interface.outputs matches actual graph outputs."
        )

    def store_component_outputs(self, comp_name: str, output: Any) -> None:
        """
        Store component outputs in resolved variables.

        EMBEDDED VARIABLE: Stores outputs with both semantic names and output_N aliases.
        Semantic outputs (not T_xxxxxx) are mapped to output_0, output_1, etc.

        Args:
            comp_name: Component name
            output: Raw output from executor
        """
        if isinstance(output, dict):
            # Store all outputs with their original names
            for key, val in output.items():
                self._variable_resolver.resolved[f"{comp_name}.{key}"] = val

            # Find semantic outputs (not tensor IDs)
            semantic_keys = [k for k in output.keys() if not k.startswith("T_")]

            # Positional aliases (output_0, output_1, ...). Two invariants:
            # (1) output_0 mirrors the PRIMARY output — the tracer names a
            #     tuple return's element 0 "output" and dataclass primaries
            #     keep their field names; the gather order is PRODUCER-OP
            #     order, which for multi-output components puts secondary
            #     outputs first (the vision deepstack levels fire before
            #     the merger — aliasing by gather order spliced deepstack_0
            #     as the vision embeds on the first +image run);
            # (2) an alias never clobbers a key that a REAL output owns
            #     (output_1..N are declared names whose tensors must
            #     survive the alias pass).
            # Single-output components and dataclass dicts keep their
            # historical aliases bit-for-bit (primary == first key there).
            if semantic_keys:
                _pref = PRIMARY_OUTPUT_NAMES
                primary = next((p for p in _pref if p in output),
                               semantic_keys[0])
                ordered = [primary] + [k for k in semantic_keys
                                       if k != primary]
                for idx, key in enumerate(ordered):
                    alias = f"output_{idx}"
                    if alias in output and alias != key:
                        continue  # a real output owns this name
                    self._variable_resolver.resolved[
                        f"{comp_name}.{alias}"] = output[key]

            # If no semantic outputs, use first output
            if not semantic_keys and len(output) > 0:
                first_key = next(iter(output))
                self._variable_resolver.resolved[f"{comp_name}.output_0"] = output[first_key]
        else:
            # Single tensor output
            self._variable_resolver.resolved[f"{comp_name}.output_0"] = output

    def produces_encoder_hidden_states(self, comp_name: str) -> bool:
        """
        Check if component produces outputs used as encoder_hidden_states by loop components.

        ZERO HARDCODE: Uses topology connections to determine if this component's outputs
        feed into any loop component's encoder_hidden_states input.

        Args:
            comp_name: Component name

        Returns:
            True if component's outputs are connected to encoder_hidden_states inputs
        """
        # Get loop components
        loop_def = self._topology.get("flow", {}).get("loop", {})
        loop_components = loop_def.get("components", [])

        if not loop_components:
            return False

        # Check connections: does this component's output connect to encoder_hidden_states?
        connections = self._topology.get("connections", [])

        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")

            # Check if from this component to a loop component's encoder input
            if from_port.startswith(f"{comp_name}."):
                if "." in to_port:
                    to_comp, to_input = to_port.split(".", 1)
                    if to_comp in loop_components:
                        ti = to_input.lower()
                        # A FLUX/MMDiT denoiser names its text-condition input `txt`,
                        # not `encoder_hidden_states`, so the to-input check misses
                        # it. Fall back to the from-OUTPUT side: a `*_hidden_state`
                        # output feeding the denoiser is the text condition (it has a
                        # negative). The pooled vector condition (CLIP pooler_output)
                        # has no `hidden_state` in its from-port, so it is excluded.
                        from_out = from_port.split(".", 1)[1].lower()
                        if (("hidden_state" in ti or "hidden_state" in from_out)
                                and "image" not in ti and "image" not in from_out):
                            return True

        return False


    # ========================================================================
    # Semantic Extraction Methods (for autoregressive/LLM flows)
    # ========================================================================


    def extract_logits(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Extract logits from head output.

        Args:
            output: Output dict from head component

        Returns:
            Logits tensor
        """
        for key in ["logits", "output_0", "output"]:
            if key in output and is_torch_tensor(output[key]):
                return output[key]

        for key, val in output.items():
            if is_torch_tensor(val):
                return val

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot extract logits from head output.\n"
            f"Keys: {list(output.keys())}"
        )

    def extract_embedding(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Extract embedding from embed/aligner output.

        Args:
            output: Output dict from embed component

        Returns:
            Embedding tensor
        """
        for key in ["output", "embedding", "output_0", "hidden_states"]:
            if key in output and is_torch_tensor(output[key]):
                return output[key]

        for key, val in output.items():
            if is_torch_tensor(val):
                return val

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot extract embedding from output.\n"
            f"Keys: {list(output.keys())}"
        )

    def extract_image(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Extract image tensor from decoder output.

        Args:
            output: Output dict from decoder component

        Returns:
            Image tensor (4D: [B, C, H, W])
        """
        for key in ["output", "image", "output_0", "sample"]:
            if key in output and is_torch_tensor(output[key]):
                return output[key]

        for key, val in output.items():
            if is_torch_tensor(val) and len(val.shape) == 4:
                return val

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot extract image from decoder output.\n"
            f"Keys: {list(output.keys())}"
        )
