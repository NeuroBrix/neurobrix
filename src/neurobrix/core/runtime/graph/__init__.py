"""
Graph execution components.

Extracted from GraphExecutor for clarity and testability.

Components:
- ExecutionContext: Shared state dataclass for a single run()
- TensorResolver: Tensor resolution from DAG to live tensors
- CompiledSequence: Pre-compiled execution sequence (GraphVM Lite)

Op dispatch:
- kernels/dispatch.py: Triton kernel dispatch (--triton mode)
- kernels/metadata_ops.py: execute_metadata_op() for PyTorch native
- sequential_dispatcher.py: NativeATenDispatcher (native mode debug)
"""

from .execution_context import ExecutionContext
from .tensor_resolver import TensorResolver
# The compiled sequence is the ATen branch's hot loop; it resolves on
# request so the shared executor imports this package without torch (R33).
_COMPILED = ("CompiledSequence", "CompiledOp", "TensorSlot", "ScalarArg", "ListArg", "DtypeArg")


def __getattr__(name):
    if name in _COMPILED:
        from . import compiled_sequence
        return getattr(compiled_sequence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExecutionContext",
    "TensorResolver",
    "CompiledSequence",
    "CompiledOp",
    "TensorSlot",
    "ScalarArg",
    "ListArg",
    "DtypeArg",
]
