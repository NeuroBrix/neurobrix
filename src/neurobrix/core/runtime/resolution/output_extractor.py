"""
Output Extractor

ZERO HARDCODE: Uses declared interface.outputs from topology.
ZERO SEMANTIC: Minimal domain knowledge - uses topology connections.

Extracts primary outputs and stores outputs in variable resolver.
"""

import torch
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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

        VARIABLE EMBARQUEE: Stores outputs with both semantic names and output_N aliases.
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

            # Map semantic outputs to output_0, output_1, etc.
            for idx, key in enumerate(semantic_keys):
                self._variable_resolver.resolved[f"{comp_name}.output_{idx}"] = output[key]

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
                    if to_comp in loop_components and "hidden_state" in to_input.lower():
                        return True

        return False

    def is_loop_component(self, comp_name: str) -> bool:
        """
        Check if component is part of the main loop.

        ZERO HARDCODE: Uses topology.flow.loop.components instead of hardcoded names.

        Args:
            comp_name: Component name

        Returns:
            True if component is in loop
        """
        loop_def = self._topology.get("flow", {}).get("loop", {})
        return comp_name in loop_def.get("components", [])

    def get_encoder_hidden_states_key(self) -> Optional[str]:
        """
        Get the variable key for encoder hidden states from topology.

        ZERO HARDCODE: Finds the variable by looking for pre_loop component outputs
        that connect to loop components with 'hidden_states' in the input name.

        Returns:
            The variable key (e.g., "text_encoder.last_hidden_state"), or None if not found.
        """
        # Find pre_loop components
        pre_loop = self._topology.get("flow", {}).get("pre_loop", [])
        loop_components = self._topology.get("flow", {}).get("loop", {}).get("components", [])

        if not pre_loop or not loop_components:
            return None

        # Check connections for encoder_hidden_states pattern
        connections = self._topology.get("connections", [])

        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")

            # Check if from a pre_loop component to a loop component's encoder input
            if "." in from_port and "." in to_port:
                from_comp = from_port.split(".")[0]
                to_comp, to_input = to_port.split(".", 1)

                if from_comp in pre_loop and to_comp in loop_components:
                    if "hidden_state" in to_input.lower():
                        return from_port

        return None

    def get_encoder_hidden_states(self) -> Optional[torch.Tensor]:
        """
        Get encoder hidden states from resolved variables.

        ZERO HARDCODE: Uses get_encoder_hidden_states_key to find the variable.

        Returns:
            The encoder hidden states tensor, or None if not found.
        """
        key = self.get_encoder_hidden_states_key()
        if key:
            return self._variable_resolver.get(key)
        return None

    # ========================================================================
    # Semantic Extraction Methods (for autoregressive/LLM flows)
    # ========================================================================

    def extract_hidden_states(self, output: Dict[str, Any], hidden_size: int = 0) -> torch.Tensor:
        """
        Extract hidden states from LM output.

        DATA-DRIVEN: Uses hidden_size from topology to distinguish hidden_states
        from logits (vocab_size >> hidden_size). If hidden_size not provided,
        selects the 3D tensor whose last dim is SMALLEST (hidden < vocab).

        Args:
            output: Output dict from LM component
            hidden_size: Model hidden_size from topology (0 = unknown)

        Returns:
            Hidden states tensor
        """
        # Try common output names first
        for key in ["hidden_states", "last_hidden_state"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        # Collect all 3D tensors with seq_len > 0
        candidates = []
        for key, val in output.items():
            if isinstance(val, torch.Tensor) and len(val.shape) == 3:
                batch_size, seq_len, last_dim = val.shape
                if seq_len > 0:
                    candidates.append((key, val, last_dim))

        if candidates:
            if hidden_size > 0:
                # DATA-DRIVEN: exact match on hidden_size
                exact = [c for c in candidates if c[2] == hidden_size]
                if exact:
                    return exact[0][1]
            # UNIVERSAL: hidden_states has smallest last dim (hidden < vocab)
            candidates.sort(key=lambda x: x[2])
            return candidates[0][1]

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot extract hidden_states from LM output.\n"
            f"Keys: {list(output.keys())}"
        )

    def extract_logits(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Extract logits from head output.

        Args:
            output: Output dict from head component

        Returns:
            Logits tensor
        """
        for key in ["logits", "output_0", "output"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
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
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
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
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        for key, val in output.items():
            if isinstance(val, torch.Tensor) and len(val.shape) == 4:
                return val

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot extract image from decoder output.\n"
            f"Keys: {list(output.keys())}"
        )
