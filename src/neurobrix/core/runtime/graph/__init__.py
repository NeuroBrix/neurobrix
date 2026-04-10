"""
Graph execution components.

Extracted from GraphExecutor for clarity and testability.

Components:
- ExecutionContext: Shared state dataclass for a single run()
- TensorResolver: Tensor resolution from DAG to live tensors
- MemoryPool: GPU memory pooling for tensor reuse
- CompiledSequence: Pre-compiled execution sequence (GraphVM Lite)

Op dispatch:
- kernels/dispatch.py: Triton kernel dispatch (--triton mode)
- kernels/metadata_ops.py: execute_metadata_op() for PyTorch native
- sequential_dispatcher.py: NativeATenDispatcher (native mode debug)
"""

from .execution_context import ExecutionContext
from .tensor_resolver import TensorResolver
from .memory_pool import MemoryPool
from .compiled_sequence import CompiledSequence, CompiledOp, TensorSlot, ScalarArg, ListArg, DtypeArg

__all__ = [
    "ExecutionContext",
    "TensorResolver",
    "MemoryPool",
    "CompiledSequence",
    "CompiledOp",
    "TensorSlot",
    "ScalarArg",
    "ListArg",
    "DtypeArg",
]
