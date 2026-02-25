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

from .loader import NBXRuntimeLoader, RuntimePackage
from .executor import RuntimeExecutor
from .resolution.variable_resolver import VariableResolver
from .factory import ExecutorFactory
from .graph_executor import GraphExecutor
from neurobrix.core.io import WeightLoader
from neurobrix.kernels.metadata_ops import execute_metadata_op
from .shape_resolver import SymbolicShapeResolver, ShapeResolutionError

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
