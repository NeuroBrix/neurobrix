"""
Execution Context - Shared state for a single DAG execution.

This dataclass holds all state needed during a single run() call.
Extracted from GraphExecutor to make state management explicit.

ZERO HARDCODE: All values come from DAG or Prism allocation.
ZERO FALLBACK: Missing values raise explicit errors.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver


@dataclass
class ExecutionContext:
    """
    Shared execution state for a single run() call.

    This encapsulates all mutable state needed during DAG execution:
    - tensor_store: Map tensor_id -> live Tensor (inputs, weights, op outputs)
    - Reference to immutable DAG data
    - Device/dtype settings from Prism

    Usage:
        ctx = ExecutionContext(dag=dag, device="cuda:0", dtype=torch.float16, ...)
        resolver = TensorResolver(ctx)
        # Execute ops...
    """

    # Immutable DAG data (from graph.json)
    dag: Dict[str, Any]

    # Hardware settings (from Prism allocation)
    device: str
    dtype: torch.dtype

    # Mutable tensor store (populated during execution)
    tensor_store: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Weight tensors (loaded from ~/.neurobrix/cache/)
    weights: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Input tensors (provided to run())
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Symbolic shape resolver (optional)
    shape_resolver: Optional["SymbolicShapeResolver"] = None

    # Symbolic shapes enabled
    symbolic_shapes_enabled: bool = False

    # Component name (for logging)
    component_name: str = "unknown"

    # Legacy: op_outputs for stats/debugging (maps op_uid -> outputs)
    op_outputs: Dict[str, Any] = field(default_factory=dict)

    def clear_runtime_state(self) -> None:
        """
        Clear state for new execution.

        Called at start of run() to reset per-execution state.
        Weights are preserved (they persist across runs).
        """
        self.tensor_store.clear()
        self.op_outputs.clear()
        self.inputs.clear()

    def store_tensor(self, tensor_id: str, tensor: torch.Tensor) -> None:
        """Store a tensor in the tensor store."""
        self.tensor_store[tensor_id] = tensor

    def get_tensor(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get a tensor from the store, or None if not found."""
        return self.tensor_store.get(tensor_id)

    def has_tensor(self, tensor_id: str) -> bool:
        """Check if a tensor is in the store."""
        return tensor_id in self.tensor_store

    def delete_tensor(self, tensor_id: str) -> None:
        """Delete a tensor from the store (memory management)."""
        if tensor_id in self.tensor_store:
            del self.tensor_store[tensor_id]

    @property
    def tensors_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get tensor metadata from DAG."""
        return self.dag.get("tensors", {})

    @property
    def ops_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get ops metadata from DAG."""
        return self.dag.get("ops", {})

    @property
    def execution_order(self) -> list:
        """Get execution order from DAG."""
        return self.dag.get("execution_order", [])

    @property
    def output_tensor_ids(self) -> list:
        """Get output tensor IDs from DAG."""
        return self.dag.get("output_tensor_ids", [])
