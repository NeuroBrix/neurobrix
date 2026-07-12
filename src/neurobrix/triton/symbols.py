"""Triton Symbolic Shape Resolution — pure Python math.

Resolves symbolic dimensions (s0=batch, s1=seq_len) from input tensor shapes.
Evaluates expression trees (floordiv, add, mul) for derived dimensions.
Zero torch dependency.
"""

from typing import Dict


class SymbolResolver:
    """Binds symbolic shape variables from actual input tensors."""

    def __init__(self, symbolic_context: dict):
        self._symbols = symbolic_context.get("symbols", {})
        self._bindings: Dict[str, int] = {}

    def bind_from_inputs(self, inputs: dict, input_tensor_ids: list,
                         tensors_meta: dict):
        """Bind symbols from actual input tensor shapes.

        For each symbol, find which input tensor and dimension it maps to,
        then read the actual runtime value from the tensor's shape.
        """
        for sym_id, sym_info in self._symbols.items():
            src = sym_info.get("source", "")

            # Value-sourced symbol: "input::grid_thw::val_1" → the symbol
            # binds from the tensor's DATA at flat index 1, not from a shape
            # dim (dynamic-resolution grid values, promoted at trace by the
            # SymValueSources pass). NBXTensor.numpy() is the R33-pure host
            # read (numpy is allowed CPU glue). Error contract mirrors the
            # compiled binder (shape_resolver.py): out-of-range index =
            # broken build-side value-source output → RAISE, never fall
            # back to the frozen trace grid; missing tensor = warn, leave
            # unbound.
            if isinstance(src, str) and src.startswith("input::") and "::val_" in src:
                parts = src.rsplit("::val_", 1)
                tensor_id = parts[0]
                idx = int(parts[1])
                tensor = inputs.get(tensor_id)
                if tensor is not None and hasattr(tensor, "numpy"):
                    flat = tensor.numpy().reshape(-1)
                    if idx >= flat.shape[0]:
                        raise RuntimeError(
                            f"Symbol {sym_id}: val index {idx} out of range "
                            f"for input '{tensor_id}' with {flat.shape[0]} "
                            f"elements")
                    self._bindings[sym_id] = int(flat[idx])
                elif tensor is not None and hasattr(tensor, "__len__"):
                    if idx >= len(tensor):
                        raise RuntimeError(
                            f"Symbol {sym_id}: val index {idx} out of range "
                            f"for input '{tensor_id}' with {len(tensor)} "
                            f"elements")
                    self._bindings[sym_id] = int(tensor[idx])
                else:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Symbol %s: cannot find input '%s' for value source",
                        sym_id, tensor_id)
                continue

            # Parse source string: "input::input_ids::dim_0" → tensor_id, dim
            if isinstance(src, str) and "::dim_" in src:
                parts = src.rsplit("::dim_", 1)
                tensor_id = parts[0]
                dim = int(parts[1])
                tensor = inputs.get(tensor_id)
                if tensor is not None and hasattr(tensor, 'shape'):
                    val = tensor.shape[dim]
                    self._bindings[sym_id] = val
                else:
                    # Try trace_value as fallback
                    tv = sym_info.get("trace_value")
                    if tv is not None:
                        self._bindings[sym_id] = tv
                continue

            # Dict format: {"tensor_id": "...", "dim": 0}
            if isinstance(src, dict):
                tensor_id = src.get("tensor_id")
                dim = src.get("dim")
                if tensor_id and dim is not None:
                    tensor = inputs.get(tensor_id)
                    if tensor is not None and hasattr(tensor, 'shape'):
                        self._bindings[sym_id] = tensor.shape[dim]

    def resolve(self, val) -> int:
        """Resolve a value that may be symbolic.

        Handles: int, SymDimRef-like dict, expression tree dict.
        """
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, dict):
            return self._eval_expr(val)
        return int(val)

    def _eval_expr(self, expr: dict) -> int:
        """Evaluate an expression tree recursively.

        Graph.json format uses:
        - {"type": "symbol", "id": "s0", "trace": 1}
        - {"type": "mul", "left": {...}, "right": {...}, "trace": 23}
        - {"type": "neg", "operand": {...}}
        - {"type": "product", "factors": [...], "trace_value": N}
        - {"type": "const", "value": N} or {"value": N}

        Ported from shape_resolver._resolve_symint_dict.
        """
        type_str = expr.get("type", "")

        # Constant value
        if type_str == "const" or (type_str == "" and "value" in expr):
            return expr.get("value", expr.get("trace", 0))

        # Symbol reference
        if type_str == "symbol":
            sym_id = expr.get("id") or expr.get("symbol_id")
            if sym_id and sym_id in self._bindings:
                return self._bindings[sym_id] + expr.get("offset", 0)
            trace = expr.get("trace") if expr.get("trace") is not None else expr.get("trace_value")
            return (trace or 0) + expr.get("offset", 0)

        # Unary: neg
        if type_str == "neg":
            operand = expr.get("operand", expr.get("args", [None])[0])
            if operand is not None:
                return -self._eval_expr(operand) if isinstance(operand, dict) else -int(operand)
            return 0

        # Binary operations: left/right format
        if type_str in ("add", "sub", "mul", "floordiv", "mod"):
            left = self._resolve_val(expr.get("left"))
            right = self._resolve_val(expr.get("right"))
            if type_str == "add":
                return left + right
            elif type_str == "sub":
                return left - right
            elif type_str == "mul":
                return left * right
            elif type_str == "floordiv":
                return left // right if right != 0 else 0
            elif type_str == "mod":
                return left % right if right != 0 else 0

        # Product: multiply factors
        if type_str == "product":
            factors = expr.get("factors", [])
            if not factors:
                return expr.get("trace_value", 0)
            result = 1
            for f in factors:
                result *= self._resolve_val(f)
            return result

        # Fallback: trace value
        trace = expr.get("trace") if expr.get("trace") is not None else expr.get("trace_value")
        if trace is not None:
            return trace
        return 0

    def _resolve_val(self, val) -> int:
        """Resolve a single value — int, str symbol ref, or dict expression."""
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            return self._bindings.get(val, 0)
        if isinstance(val, dict):
            return self._eval_expr(val)
        return 0

    def get(self, sym_id: str, default: int = 0) -> int:
        return self._bindings.get(sym_id, default)

    @property
    def bindings(self) -> Dict[str, int]:
        return dict(self._bindings)
