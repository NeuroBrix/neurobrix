"""
Input Resolver

ZERO HARDCODE: All bindings come from topology.json connections.
ZERO SEMANTIC: No knowledge of "text_encoder", "vae" - only connection resolution.

Resolves component inputs by following topology connections from global
variables and component outputs.
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver


class InputResolver:
    """
    Resolves component inputs from topology connections.

    This class is responsible for:
    1. Looking up connections from the pre-indexed connections map
    2. Resolving source chains (trying sources in order until one resolves)
    3. Resolving individual sources (global variables or component outputs)

    ZERO HARDCODE: All bindings from topology.json connections.
    ZERO SEMANTIC: No domain knowledge - just follows connection graph.
    """

    def __init__(
        self,
        variable_resolver: 'VariableResolver',
        connections_index: Dict[str, Dict[str, List[str]]],
        topology: Dict[str, Any],
        loop_id: str
    ):
        """
        Initialize input resolver.

        Args:
            variable_resolver: VariableResolver instance for accessing resolved values
            connections_index: Pre-indexed connections {comp: {input: [sources]}}
            topology: Full topology dict for loop definitions
            loop_id: Current loop identifier for timestep resolution
        """
        self._variable_resolver = variable_resolver
        self._connections_index = connections_index
        self._topology = topology
        self._loop_id = loop_id

    def resolve_component_inputs(self, comp_name: str) -> Dict[str, Any]:
        """
        Resolve component inputs from topology connections.

        ZERO HARDCODE: All bindings from topology.json connections.
        For iterative patterns, tries sources in order (global first, then feedback).

        UNIFIED RESOLUTION: Same pattern for all flow types (diffusers and forward_pass).
        Connections are enriched at import time from graph inputs, so runtime
        uses the same resolve_source_chain() for all models.

        FALLBACK RESOLUTION: If no connections declared but component interface
        specifies inputs, try to resolve from variable_resolver.resolved.

        Args:
            comp_name: Name of component to resolve inputs for

        Returns:
            Dict mapping input names to resolved values
        """
        # Check if component has declared connections
        # UNIVERSAL: Use connections from topology (same pattern for all model types)
        # Connections are generated at trace time (Phase C) for all models including forward_pass
        if comp_name in self._connections_index:
            connections = self._connections_index[comp_name]
            if connections:  # Has actual connection mappings
                resolved = {}
                for input_name, sources in connections.items():
                    # sources is a list - try each in order until one resolves
                    resolved[input_name] = self.resolve_source_chain(sources)
                return resolved

        # FALLBACK: If no connections, try to resolve from variable_resolver.resolved
        # This handles cases where tokenization stores input_ids/attention_mask
        comp_info = self._topology.get("components", {}).get(comp_name, {})
        interface_inputs = comp_info.get("interface", {}).get("inputs", [])

        if interface_inputs:
            resolved = {}
            for input_name in interface_inputs:
                # Try multiple resolution paths
                for key in [input_name, f"global.{input_name}"]:
                    if key in self._variable_resolver.resolved:
                        resolved[input_name] = self._variable_resolver.resolved[key]
                        break
            if resolved:
                return resolved

        # Component has no declared inputs - may use global defaults
        return {}

    def resolve_source_chain(self, sources: List[str]) -> Any:
        """
        Try to resolve sources in order. First available wins.

        This enables iterative patterns:
        global.latents (initial) → transformer.output_0 (feedback)

        Args:
            sources: List of source identifiers to try in order

        Returns:
            First successfully resolved value

        Raises:
            RuntimeError: If no source could be resolved (ZERO FALLBACK)
        """
        last_error = None
        for source in sources:
            try:
                return self.resolve_source(source)
            except (RuntimeError, KeyError) as e:
                last_error = e
                continue

        # None resolved
        raise RuntimeError(
            f"ZERO FALLBACK: None of the sources resolved: {sources}\n"
            f"Last error: {last_error}"
        )

    def resolve_source(self, source: str) -> Any:
        """
        Resolve a source port to its value.

        Sources can be:
        - "global.xxx" - global variable
        - "component.output" - output from another component

        VARIABLE EMBARQUÉE: index_variable from topology.flow.loop, not hardcoded.

        Args:
            source: Source identifier string

        Returns:
            Resolved value from variable resolver

        Raises:
            RuntimeError: If source cannot be resolved (ZERO FALLBACK)
        """
        if source.startswith("global."):
            # VARIABLE EMBARQUÉE: Check if this is the loop index variable (e.g., timestep)
            loop_def = self._topology.get("flow", {}).get("loop", {})
            index_variable = loop_def.get("index_variable")

            if index_variable and source == index_variable:
                return self._variable_resolver.loop_state.get(self._loop_id)
            return self._variable_resolver.get(source)

        # Component output
        if source in self._variable_resolver.resolved:
            return self._variable_resolver.resolved[source]

        raise RuntimeError(
            f"ZERO FALLBACK: Source '{source}' not found in resolved variables.\n"
            f"Available: {list(self._variable_resolver.resolved.keys())}"
        )

    def get_connections(self, comp_name: str) -> Dict[str, List[str]]:
        """
        Get raw connection mapping for a component.

        Args:
            comp_name: Component name

        Returns:
            Dict mapping input names to source lists
        """
        return self._connections_index.get(comp_name, {})

    def update_loop_id(self, loop_id: str) -> None:
        """
        Update the current loop identifier.

        Used when switching between loops or entering iteration.

        Args:
            loop_id: New loop identifier
        """
        self._loop_id = loop_id
