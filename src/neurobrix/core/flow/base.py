"""
Flow Handler Base Classes

ZERO SEMANTIC: Flow handlers execute mechanically based on topology.json.
ZERO HARDCODE: All configuration comes from NBX container.

This module provides:
- FlowContext: Immutable context shared by all flow handlers
- FlowHandler: Abstract base class for execution flows
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.runtime.loader import RuntimePackage
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.strategies import ExecutionStrategy


@dataclass
class FlowContext:
    """
    Immutable context passed to flow handlers.

    Contains all shared state needed for execution without
    requiring flow handlers to access executor internals.
    """
    # Core NBX package (immutable)
    pkg: 'RuntimePackage'

    # Prism execution plan with device allocations
    plan: Any

    # Variable resolver for dynamic binding
    variable_resolver: 'VariableResolver'

    # Component executors (GraphExecutor instances)
    executors: Dict[str, 'GraphExecutor']

    # Loaded modules (scheduler, tokenizer, etc.)
    modules: Dict[str, Any]

    # Execution strategy from Prism
    strategy: 'ExecutionStrategy'

    # Pre-indexed connections: {comp_name: {input_name: [sources]}}
    connections_index: Dict[str, Dict[str, List[str]]]

    # Loop identifier from variables contract
    loop_id: str

    # Path to NBX cache for weight loading
    nbx_path_str: str

    # Execution mode: "compiled" (default), "native" (ATen debug), "triton" (R&D)
    mode: str = "compiled"

    # Persistent mode: when True, GraphExecutors mark themselves _persistent
    # so cleanup() preserves weights in VRAM. Set by serving layer for warm strategies.
    persistent_mode: bool = False

    # Primary device string — ALWAYS derived from Prism allocation (executor.py:_get_primary_device)
    primary_device: str = ""



class FlowHandler(ABC):
    """
    Abstract base class for execution flow handlers.

    Each flow type (iterative_process, static_graph, forward_pass,
    autoregressive_generation) is implemented as a separate FlowHandler.

    ZERO SEMANTIC: FlowHandlers don't know about model semantics.
    They execute the flow mechanically based on topology.
    """

    def __init__(self, ctx: FlowContext):
        """
        Initialize flow handler with context.

        Args:
            ctx: FlowContext containing all execution state
        """
        self.ctx = ctx

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the flow and return final outputs.

        Returns:
            Dict of resolved variables/outputs from execution
        """
        pass



# Flow type registry for factory pattern
FLOW_REGISTRY: Dict[str, type] = {}


def register_flow(flow_type: str):
    """
    Decorator to register flow handler classes.

    Usage:
        @register_flow("iterative_process")
        class IterativeProcessHandler(FlowHandler):
            ...
    """
    def decorator(cls):
        FLOW_REGISTRY[flow_type] = cls
        return cls
    return decorator


def get_flow_handler(flow_type: str, ctx: FlowContext) -> FlowHandler:
    """
    Factory function to create flow handler by type.

    Args:
        flow_type: Flow type string
        ctx: FlowContext for handler

    Returns:
        Instantiated FlowHandler

    Raises:
        RuntimeError: If flow_type not registered (ZERO FALLBACK)
    """
    if flow_type not in FLOW_REGISTRY:
        available = list(FLOW_REGISTRY.keys())
        raise RuntimeError(
            f"ZERO FALLBACK: Unknown flow type '{flow_type}'.\n"
            f"Available: {available}"
        )

    handler_class = FLOW_REGISTRY[flow_type]
    return handler_class(ctx)
