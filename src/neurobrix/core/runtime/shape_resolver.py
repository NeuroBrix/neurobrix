"""
Symbolic Shape Resolver for NeuroBrix Runtime.

Resolves symbolic shapes to concrete values at runtime based on actual input tensors.

Supports SymInt expression trees for automatic propagation.
Handles both SymInt dicts and string expressions for backward compatibility.

ZERO HARDCODE: Shape values come from actual inputs, not config.
ZERO FALLBACK: Missing symbols raise explicit errors.

Usage:
    resolver = SymbolicShapeResolver(dag["symbolic_context"])
    resolver.bind_from_inputs(inputs, dag["tensors"])

    # Resolve a shape (string or SymInt)
    concrete_shape = resolver.resolve(["s0", 4, 128, 128])
    # Returns: [2, 4, 128, 128] if s0=2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import os as _os_shape

# Read once: a per-bind os.environ lookup sat in the warm request's hot path
# (2026-09-05 profile: 16,802 environ reads per whisper request).
_SHAPE_DEBUG = _os_shape.environ.get("NBX_DEBUG") == "1"

from neurobrix.core.runtime import symexpr as _symexpr

# Import SymInt support (from core/runtime, no trace/ dependency)
try:
    from neurobrix.core.runtime.symint import SymInt
except ImportError:
    SymInt = None  # Fallback for when running standalone

logger = logging.getLogger(__name__)


class ShapeResolutionError(Exception):
    """Raised when symbolic shape resolution fails."""
    pass


class SymbolicShapeResolver:
    """
    Resolves symbolic shapes at runtime.

    Workflow:
    1. Initialize with symbolic_context from graph.json
    2. Bind symbol values from actual input tensors
    3. Resolve shapes for any tensor/op

    Example symbolic_context:
    {
        "symbols": {
            "s0": {
                "name": "batch",
                "trace_value": 2,
                "min": 1, "max": 64,
                "source": "input::hidden_states::dim_0"
            }
        },
        "expressions": {
            "e0": "s0 * s1"
        }
    }
    """

    def __init__(
        self,
        symbolic_context: Optional[Dict[str, Any]] = None,
        strict: bool = False
    ):
        """
        Initialize resolver.

        Args:
            symbolic_context: The symbolic_context from graph.json.
                             If None or empty, resolver works in passthrough mode.
            strict: If True, raise error when using trace_value fallback.
                   ZERO FALLBACK: In strict mode, all symbols must be bound explicitly.
        """
        self._context = symbolic_context or {}
        self._symbols = self._context.get("symbols", {})
        self._fallbacks_said = set()
        self._expressions = self._context.get("expressions", {})
        self._strict = strict

        # Runtime values (bound from actual inputs)
        self._runtime_values: Dict[str, int] = {}
        self._bound = False

    @property
    def symbolic_shapes_enabled(self) -> bool:
        """Check if this graph has symbolic shapes."""
        return bool(self._symbols)

    def _get_nested_input(self, inputs: Dict[str, Any], key: str) -> Any:
        """
        Get value from inputs dict, supporting nested keys (dot notation).

        For key="added_cond_kwargs.resolution":
          - First checks if "added_cond_kwargs.resolution" exists directly
          - Then checks if "added_cond_kwargs" exists and has "resolution" key

        Args:
            inputs: Input dict (may contain nested dicts)
            key: Key to look up (may contain dots for nested access)

        Returns:
            The value if found, None otherwise
        """
        # Direct lookup first
        if key in inputs:
            return inputs[key]

        # Nested lookup for dotted names
        if "." in key:
            parts = key.split(".")
            value = inputs
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value

        return None

    def bind_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        tensor_specs: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Bind symbol values from actual input tensors.

        Args:
            inputs: Dict of input_name -> tensor
            tensor_specs: Dict of tensor_id -> tensor spec from graph.json

        Raises:
            ShapeResolutionError: If symbol binding fails
        """
        if not self._symbols:
            self._bound = True
            return

        # Clear stale values — critical for AR loops where input shapes change
        # between iterations (e.g. seq_len grows from 1→2→3...).
        # Without clearing, old symbol values persist even when the source
        # input is no longer present, causing view/reshape to use wrong dims.
        self._runtime_values.clear()

        # For each symbol, find its source and bind
        for symbol_id, symbol_info in self._symbols.items():
            source = symbol_info.get("source", "")

            # Parse source: "input::hidden_states::dim_0"
            if source.startswith("input::"):
                parts = source.split("::")
                if len(parts) >= 3:
                    input_name = parts[1]
                    dim_str = parts[2]

                    # Value-sourced symbol: "input::grid_thw::val_1" binds from
                    # the tensor's DATA at flat index 1, not from a shape dim
                    # (dynamic-resolution grid values, promoted at trace by the
                    # SymValueSources pass).
                    # Negative indices are part of the vocabulary
                    # (e.g. "val_-1" = last element — runtime rope-length
                    # symbols); mirror of the triton binder
                    # (triton/symbols.py bind_from_inputs).
                    val_match = re.match(r"val_(-?\d+)(?:_fd(\d+))?$", dim_str)
                    if val_match:
                        flat_idx = int(val_match.group(1))
                        fdiv = int(val_match.group(2)) if val_match.group(2) else 1
                        tensor = self._get_nested_input(inputs, input_name)
                        if tensor is not None:
                            if hasattr(tensor, "flatten"):
                                flat = tensor.flatten()
                                if -len(flat) <= flat_idx < len(flat):
                                    self._bind_symbol(
                                        symbol_id, int(flat[flat_idx]) // fdiv,
                                        symbol_info)
                                else:
                                    raise ShapeResolutionError(
                                        f"Symbol {symbol_id}: val index {flat_idx} out of "
                                        f"range for input '{input_name}' with "
                                        f"{len(flat)} elements")
                            elif (hasattr(tensor, "__len__")
                                  and -len(tensor) <= flat_idx < len(tensor)):
                                self._bind_symbol(
                                    symbol_id, int(tensor[flat_idx]), symbol_info)
                        else:
                            logger.warning(
                                f"Symbol {symbol_id}: Cannot find input "
                                f"'{input_name}' for value source")
                        continue

                    # Extract dim index
                    dim_match = re.match(r"dim_(\d+)", dim_str)
                    if dim_match:
                        dim_idx = int(dim_match.group(1))

                        # Find the tensor (supports nested dicts like "added_cond_kwargs.resolution")
                        tensor = self._get_nested_input(inputs, input_name)
                        if tensor is not None:
                            if hasattr(tensor, 'ndim') and dim_idx < tensor.ndim:
                                value = tensor.shape[dim_idx]
                                self._bind_symbol(symbol_id, value, symbol_info)
                            elif hasattr(tensor, '__len__') and dim_idx < len(tensor):
                                # Handle list/tuple values (e.g., resolution=[1024, 1024])
                                value = tensor[dim_idx] if isinstance(tensor[dim_idx], int) else len(tensor)
                                self._bind_symbol(symbol_id, value, symbol_info)
                            elif hasattr(tensor, 'ndim'):
                                raise ShapeResolutionError(
                                    f"Symbol {symbol_id}: dim {dim_idx} out of range "
                                    f"for input '{input_name}' with shape {tensor.shape}"
                                )
                        else:
                            # Try to find by tensor_id pattern
                            tensor_id = f"input::{input_name}"
                            for name, t in inputs.items():
                                if name == input_name or tensor_id.endswith(name):
                                    if hasattr(t, 'ndim') and dim_idx < t.ndim:
                                        value = t.shape[dim_idx]
                                        self._bind_symbol(symbol_id, value, symbol_info)
                                        break
                            else:
                                logger.warning(
                                    f"Symbol {symbol_id}: Cannot find input '{input_name}' "
                                    f"in provided inputs: {list(inputs.keys())}"
                                )

        self._bound = True
        logger.debug(f"Bound symbols: {self._runtime_values}")
        if _SHAPE_DEBUG:
            _shapes = {k: tuple(v.shape) for k, v in inputs.items()
                       if hasattr(v, "shape")}
            print(f"[SYMBOLS] bound={self._runtime_values} from inputs={_shapes}",
                  flush=True)

    def _bind_symbol(
        self,
        symbol_id: str,
        value: int,
        symbol_info: Dict[str, Any]
    ) -> None:
        """
        Bind a symbol to a value with constraint validation.

        Args:
            symbol_id: Symbol ID (e.g., "s0")
            value: Runtime value
            symbol_info: Symbol info with constraints

        Raises:
            ShapeResolutionError: If value violates constraints
        """
        # Validate constraints. The tracer NESTS them:
        #   "s1": {"name": "time", "source": "...", "constraints": {"min": 1}}
        # Reading them at the top level (as this did) finds nothing, defaults
        # min to 0, and the check then rejects only a negative — so it had
        # never refused a value in the whole zoo (673 symbols across 182
        # cached components, every one carrying min 1). Wan2.1-VACE bound
        # `time` to 0 from a still image, nothing objected, and sixty ops
        # later the VAE encoder asked the allocator for -4,860,000 bytes at
        # aten.convolution::60 — recorded as an OOM. The top-level read stays
        # as a fallback for a caller holding a flat spec.
        constraints = symbol_info.get("constraints") or {}
        min_val = constraints.get("min", symbol_info.get("min", 0))
        max_val = constraints.get("max", symbol_info.get("max", float('inf')))
        name = symbol_info.get("name", "?")
        source = symbol_info.get("source", "?")

        if value < min_val:
            raise ShapeResolutionError(
                f"Symbol {symbol_id} ({name}) = {value}, below the minimum "
                f"extent the container declares ({min_val}); it binds from "
                f"{source}. A dim at or below zero is not a memory condition — "
                f"left unchecked it becomes a negative allocation further down "
                f"the graph, far from the input that caused it."
            )
        if value > max_val:
            raise ShapeResolutionError(
                f"Symbol {symbol_id} ({name}) = {value}, above the maximum "
                f"extent the container declares ({max_val}); it binds from "
                f"{source}."
            )

        self._runtime_values[symbol_id] = value
        logger.debug(f"Bound {symbol_id}={value} (name={symbol_info.get('name', '?')})")

    def resolve(self, shape: Any) -> Any:
        """
        Resolve a symbolic shape to concrete values.

        Args:
            shape: Can be:
                - List/tuple like ["s0", 4, 128, 128]
                - Single value like "s0" or 4
                - Expression like "e0"
                - Already concrete

        Returns:
            Resolved value (int, list, tuple)

        Raises:
            ShapeResolutionError: If resolution fails
        """
        if shape is None:
            return None

        # List/tuple: resolve each element
        if isinstance(shape, (list, tuple)):
            resolved = [self._resolve_single(dim) for dim in shape]
            return type(shape)(resolved)

        # Single value
        return self._resolve_single(shape)

    def _resolve_single(self, value: Any) -> Any:
        """
        Resolve a single dimension value.

        Handles SymInt objects and dict serialization format.
        Handles string symbol references for backward compatibility.
        """
        # Already concrete int/float
        if isinstance(value, (int, float)):
            return int(value)

        # SymInt object (in-memory)
        if SymInt is not None and isinstance(value, SymInt):
            return value.resolve(self._runtime_values)

        # Dict format (from JSON serialization)
        if isinstance(value, dict):
            return self._resolve_symint_dict(value)

        # String handling
        if isinstance(value, str):
            # Symbol reference (s0, s1, ...)
            if value in self._runtime_values:
                return self._runtime_values[value]

            # A symbol the container declares and the runtime did not bind REFUSES,
            # by name. The trace value is a witnessed extent of one stimulus, not a
            # value; answering it here hid two defects in one day (2026-09-21): a plan
            # bound a VAE's spatial symbols to nothing and estimated 1.74 GiB for a
            # 35 GiB decode; an ATen sequential run evaluated `s1` at 23 on a
            # 213-token context and died a dozen ops later on a view. NBX_STRICT_SYMBOLS
            # is no longer a door: strict is the only behaviour.
            if value in self._symbols:
                info = self._symbols[value]
                raise ShapeResolutionError(
                    f"ZERO FALLBACK: symbol '{value}' ({info.get('name')}, binds from "
                    f"{info.get('source')}) is not bound at runtime; its trace value "
                    f"{info.get('trace_value')} is a witnessed extent, not a value. Bound: "
                    f"{sorted(self._runtime_values)}.")
            return value

        # Unknown type - return as-is
        return value

    def _symbol_value(self, symbol_id: str, node: Any) -> int:
        """The bound value of a symbol, or a refusal naming it (the shared evaluator's callback)."""
        if symbol_id in self._runtime_values:
            return self._runtime_values[symbol_id]
        info = self._symbols.get(symbol_id) or {}
        trace = (node or {}).get("trace", (node or {}).get("trace_value", info.get("trace_value")))
        raise ShapeResolutionError(
            f"ZERO FALLBACK: symbol '{symbol_id}' ({info.get('name')}, binds from "
            f"{info.get('source')}) is not bound at runtime; its trace value "
            f"{trace} is a witnessed extent, not a value. Bound: {sorted(self._runtime_values)}.")

    def _resolve_symint_dict(self, data: Dict[str, Any], slot: str = _symexpr.SHAPE):
        """Evaluate a serialized expression — the ONE vocabulary of `core/runtime/symexpr.py`,
        shared with the Triton resolver. A SHAPE slot (the default: extents, size lists) refuses
        a real by name; a SCALAR slot (an op's scalar argument) takes the int or float."""
        return _symexpr.evaluate(data, self._symbol_value, slot)

    def resolve_scalar(self, value: Any) -> Any:
        """Resolve an op's SCALAR argument: a symbol or an expression evaluates in the scalar
        slot (a `linspace` bound may be real); a dict outside the vocabulary refuses by name;
        a plain number is returned as it stands."""
        if isinstance(value, (dict, str)):
            return self._resolve_symint_dict(value, _symexpr.SCALAR)
        return value

    def resolve_tensor_shape(
        self,
        tensor_spec: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """
        Resolve shape for a tensor spec from graph.json.

        Handles SymInt dict format (expression trees).
        Handles string symbol references.

        Args:
            tensor_spec: Tensor spec with "shape" or "shape_concrete"

        Returns:
            Concrete shape tuple
        """
        if "shape" in tensor_spec:
            shape = tensor_spec["shape"]
            # Check if shape contains symbolic elements
            has_symbolic = False
            for d in shape:
                if isinstance(d, str):
                    has_symbolic = True
                    break
                if isinstance(d, dict):
                    # SymInt dict format
                    has_symbolic = True
                    break

            if has_symbolic:
                return tuple(self.resolve(shape))
            # All concrete ints
            return tuple(shape)

        # Fallback: Use shape_concrete or shape
        if "shape_concrete" in tensor_spec:
            return tuple(tensor_spec["shape_concrete"])

        raise ShapeResolutionError(
            f"Tensor spec missing shape: {tensor_spec.get('tensor_id', '?')}"
        )

    def get_bound_symbols(self) -> Dict[str, int]:
        """Get all bound symbol values."""
        return dict(self._runtime_values)

    def __repr__(self) -> str:
        return (
            f"SymbolicShapeResolver("
            f"symbols={len(self._symbols)}, "
            f"bound={len(self._runtime_values)})"
        )
