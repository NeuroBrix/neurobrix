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
    import torch
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

    # Execution mode: "compiled" (default), "sequential" (PyTorch eager debug),
    # "triton" (Triton-pure compiled), "triton_sequential" (Triton-pure debug)
    mode: str = "compiled"

    # Persistent mode: when True, GraphExecutors mark themselves _persistent
    # so cleanup() preserves weights in VRAM. Set by serving layer for warm strategies.
    persistent_mode: bool = False

    # Primary device string — ALWAYS derived from Prism allocation (executor.py:_get_primary_device)
    primary_device: str = ""

    def compute_dtype(self, component: str = None) -> 'torch.dtype':
        """Prism-RESOLVED compute dtype for flow-level tensor synthesis.

        SINGLE compiled-side resolver (brick-consolidation E2). The Prism
        plan is the authority for the dtype that actually executes — the
        manifest carries the pre-Prism vendor declaration and is only a
        last-resort fallback when no plan is attached (degraded/isolation
        contexts). Flow handlers that previously read `manifest["dtype"]`
        directly could diverge from the allocation the DtypeEngine runs
        under (e.g. a bf16 manifest resolved to fp32 on non-bf16 hardware).

        Resolution order:
          1. `plan.components[component].dtype` — per-component resolved
             dtype, when the caller names the component it synthesises for;
          2. first allocation carrying a dtype — the model-wide answer for
             single-dtype plans (every allocation agrees);
          3. `plan.target_dtype` — the plan-wide resolved dtype;
          4. `manifest["dtype"]` — no plan attached.

        The triton mirror is `neurobrix.triton.dtype.resolve_compute_dtype`
        (string-dtype boundary, R33) — separate implementation by design.
        """
        from neurobrix.core.dtype.config import get_torch_dtype
        plan = self.plan
        if plan is not None:
            comps = getattr(plan, "components", None)
            if comps:
                alloc = comps.get(component) if component else None
                if alloc is None or not getattr(alloc, "dtype", None):
                    alloc = next((a for a in comps.values()
                                  if getattr(a, "dtype", None)), None)
                if alloc is not None and getattr(alloc, "dtype", None):
                    return get_torch_dtype(alloc.dtype)
            target = getattr(plan, "target_dtype", None)
            if target:
                return get_torch_dtype(target)
        return get_torch_dtype(self.pkg.manifest.get("dtype", "float16"))


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
