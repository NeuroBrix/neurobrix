"""
NeuroBrix Runtime System.

Unified execution layer for neural network inference.

Architecture:
- RuntimeExecutor: High-level orchestration from execution.json
- GraphExecutor: Low-level graph execution
- ExecutorFactory: Creates appropriate executor based on Prism allocation
- WeightLoader: Loads weights from NBX container
- VariableResolver: Resolves variables from runtime contracts

Features:
- Symbolic shape resolution
- Dynamic buffer management
- Semantic tensor IDs
"""

# PEP 562 lazy exports (D-CORE-MODULE-INIT-TORCH, 2026-09-02): the runtime
# package's public names pull the whole compiled executor stack (torch).
# The triton generators import `neurobrix.core.runtime.decode_bound` (a
# torch-free CPU helper) and paid that whole init for it. Names resolve on
# first attribute access; `from neurobrix.core.runtime import X` keeps
# working unchanged.
_LAZY = {
    "NBXRuntimeLoader": ".loader",
    "RuntimePackage": ".loader",
    "RuntimeExecutor": ".executor",
    "VariableResolver": ".resolution.variable_resolver",
    "ExecutorFactory": ".factory",
    "GraphExecutor": ".graph_executor",
    "WeightLoader": "neurobrix.core.io",
    "execute_metadata_op": "neurobrix.kernels.metadata_ops",
    "SymbolicShapeResolver": ".shape_resolver",
    "ShapeResolutionError": ".shape_resolver",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    # Loader
    "NBXRuntimeLoader",
    "RuntimePackage",
    # Orchestration
    "RuntimeExecutor",
    "VariableResolver",
    # Execution
    "ExecutorFactory",
    "GraphExecutor",
    "WeightLoader",
    "execute_metadata_op",
    # Symbolic shapes
    "SymbolicShapeResolver",
    "ShapeResolutionError",
]
