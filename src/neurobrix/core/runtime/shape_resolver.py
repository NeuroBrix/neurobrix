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

import torch
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

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

        # For each symbol, find its source and bind
        for symbol_id, symbol_info in self._symbols.items():
            source = symbol_info.get("source", "")

            # Parse source: "input::hidden_states::dim_0"
            if source.startswith("input::"):
                parts = source.split("::")
                if len(parts) >= 3:
                    input_name = parts[1]
                    dim_str = parts[2]

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
        # Validate constraints
        min_val = symbol_info.get("min", 0)
        max_val = symbol_info.get("max", float('inf'))

        if value < min_val:
            raise ShapeResolutionError(
                f"Symbol {symbol_id}={value} violates min constraint ({min_val})"
            )
        if value > max_val:
            raise ShapeResolutionError(
                f"Symbol {symbol_id}={value} violates max constraint ({max_val})"
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

            # Check if it's a symbol we know but haven't bound
            if value in self._symbols:
                # STRICT MODE: No fallback allowed
                if self._strict:
                    raise ShapeResolutionError(
                        f"ZERO FALLBACK: Symbol '{value}' not bound at runtime (strict mode). "
                        f"Ensure all dynamic dimensions are resolved from actual inputs."
                    )

                # Non-strict: Use trace value as fallback (with warning)
                trace_val = self._symbols[value].get("trace_value")
                if trace_val is not None:
                    logger.warning(
                        f"Symbol {value} not bound, using trace_value={trace_val}"
                    )
                    return trace_val
                raise ShapeResolutionError(
                    f"Symbol {value} not bound and no trace_value available"
                )

            # Expression reference (e0, e1, ...)
            if value in self._expressions:
                return self._evaluate_expression(self._expressions[value])

            # Try to parse as int
            try:
                return int(value)
            except ValueError:
                pass

            # Check if it's an inline expression (e.g., "s0 * s1")
            if any(sym_id in value for sym_id in self._runtime_values):
                return self._evaluate_expression(value)

            # Unknown - return as-is (might be a string literal)
            return value

        # Unknown type - return as-is
        return value

    def _resolve_symint_dict(self, data: Dict[str, Any]) -> int:
        """
        Resolve a SymInt serialized as a dict.

        Handles:
        - {"type": "symbol", "id": "s0", "trace": 2}
        - {"type": "mul", "left": {...}, "right": {...}, "trace": 240}
        - etc.
        """
        type_str = data.get("type", "")

        # Constant
        if type_str == "const" or "value" in data:
            return data.get("value", data.get("trace", 0))

        # Symbol reference
        if type_str == "symbol":
            # Handle both formats: (id/trace) and graph format (symbol_id/trace_value)
            symbol_id = data.get("id") or data.get("symbol_id")
            if symbol_id in self._runtime_values:
                return self._runtime_values[symbol_id]
            # Fallback to trace value (both formats)
            trace = data.get("trace") if data.get("trace") is not None else data.get("trace_value")
            if trace is not None:
                if self._strict:
                    raise ShapeResolutionError(
                        f"ZERO FALLBACK: Symbol '{symbol_id}' not bound (strict mode)"
                    )
                logger.warning(f"Symbol {symbol_id} not bound, using trace={trace}")
                return trace
            raise ShapeResolutionError(f"Symbol {symbol_id} not bound")

        # Unary: neg
        if type_str == "neg":
            operand = self._resolve_symint_dict(data["operand"])
            return -operand

        # Binary operations
        if type_str in ("add", "sub", "mul", "floordiv", "mod"):
            left = self._resolve_single(data["left"])
            right = self._resolve_single(data["right"])

            if type_str == "add":
                return left + right
            elif type_str == "sub":
                return left - right
            elif type_str == "mul":
                return left * right
            elif type_str == "floordiv":
                return left // right
            elif type_str == "mod":
                return left % right

        # Product: multiply factors together (e.g., s1 * s2)
        if type_str == "product":
            factors = data.get("factors", [])
            if not factors:
                return data.get("trace_value", 0)
            result = 1
            for factor in factors:
                result *= self._resolve_single(factor)
            return result

        # Scaled product: (scale_h * factors[0]) * (scale_w * factors[1])
        # Used by VAE expand operations that scale height/width
        if type_str == "scaled_product":
            factors = data.get("factors", [])
            scale_h = data.get("scale_h", 1)
            scale_w = data.get("scale_w", 1)

            if len(factors) >= 2:
                h_val = self._resolve_single(factors[0])
                w_val = self._resolve_single(factors[1])
                return (scale_h * h_val) * (scale_w * w_val)
            elif len(factors) == 1:
                val = self._resolve_single(factors[0])
                return scale_h * scale_w * val
            else:
                return data.get("trace_value", 0)

        # Scaled symbol: scale * symbol_value
        if type_str == "scaled_symbol":
            symbol_id = data.get("symbol_id") or data.get("id")
            scale = data.get("scale", 1)
            if symbol_id in self._runtime_values:
                return scale * self._runtime_values[symbol_id]
            # Fallback to trace
            trace = data.get("trace_value") or data.get("trace")
            if trace is not None:
                return trace
            raise ShapeResolutionError(f"Scaled symbol {symbol_id} not bound")

        raise ShapeResolutionError(f"Unknown SymInt type: {type_str}")

    def _evaluate_expression(self, expr: str) -> int:
        """
        Evaluate a symbolic expression.

        Args:
            expr: Expression string like "s0 * s1" or "s1 / 2"

        Returns:
            Evaluated integer result
        """
        # Replace symbols with values
        resolved_expr = expr
        for symbol_id, value in self._runtime_values.items():
            resolved_expr = resolved_expr.replace(symbol_id, str(value))

        # Safe evaluation (only arithmetic)
        try:
            # Only allow safe operations
            allowed = set("0123456789+-*/() ")
            if not all(c in allowed for c in resolved_expr):
                raise ShapeResolutionError(
                    f"Unsafe expression: {expr} -> {resolved_expr}"
                )
            result = eval(resolved_expr)
            return int(result)
        except Exception as e:
            raise ShapeResolutionError(
                f"Failed to evaluate expression '{expr}': {e}"
            )

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
