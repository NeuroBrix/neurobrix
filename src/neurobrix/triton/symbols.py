"""Triton Symbolic Shape Resolution — pure Python math.

Resolves symbolic dimensions (s0=batch, s1=seq_len) from input tensor shapes.
Evaluates expression trees through the ONE vocabulary shared with the ATen resolver
(neurobrix.core.runtime.symexpr). Zero torch dependency.
"""

from typing import Dict

from neurobrix.core.runtime import symexpr as _symexpr


class UnboundSymbolError(RuntimeError):
    """A symbol the container declares that the runtime did not bind — refused by name.
    The trace value is a witnessed extent of one stimulus, never a value (2026-09-21)."""


def impossible_extent_context(err, args, resolver) -> str:
    """The suffix an executor appends to an `ImpossibleExtentError`: the op's input shapes and
    every bound symbol with its name, so the refusal says which input extent led there.
    Empty for any other exception (their messages are left exactly as they were)."""
    from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
    if not isinstance(err, ImpossibleExtentError):
        return ""
    def _shape(a):
        if hasattr(a, "shape"):
            return str(tuple(a.shape))
        if isinstance(a, (list, tuple)) and any(hasattr(x, "shape") for x in a):
            return "[" + ", ".join(str(tuple(x.shape)) if hasattr(x, "shape") else repr(x) for x in a) + "]"
        return repr(a) if isinstance(a, (int, float)) else type(a).__name__
    shapes = ", ".join(_shape(a) for a in args)
    if resolver is None:
        syms = "no symbolic context"
    else:
        names = {k: (v or {}).get("name", "") for k, v in (resolver._symbols or {}).items()}
        syms = ", ".join(f"{k}={v}" + (f" ({names[k]})" if names.get(k) else "")
                         for k, v in sorted(resolver._bindings.items())) or "none bound"
    return f" | input shapes: {shapes} | symbols: {syms}"


class SymbolResolver:
    """Binds symbolic shape variables from actual input tensors."""

    def __init__(self, symbolic_context: dict):
        self._symbols = symbolic_context.get("symbols", {})
        self._bindings: Dict[str, int] = {}

    def _bind(self, sym_id: str, value: int) -> None:
        """Record one symbol's runtime extent, refusing what the container
        declares impossible.

        Mirror of the ATen binder's `_bind_symbol` (core/runtime/shape_resolver.py):
        the tracer nests the contract as
        `{"name": "time", "source": "...", "constraints": {"min": 1}}`, and
        every one of the zoo's 673 symbols carries a min of 1 — so this
        refuses a batch, seq_len, height, width or time of zero or below.

        It is the single write site on purpose. A dim at or below zero is not
        a memory condition, but left unchecked that is how it surfaces: a
        negative allocation dozens of ops downstream, naming a convolution
        instead of the input that caused it.
        """
        info = self._symbols.get(sym_id) or {}
        constraints = info.get("constraints") or {}
        lo = constraints.get("min", info.get("min"))
        hi = constraints.get("max", info.get("max"))
        if lo is not None and value < lo:
            raise RuntimeError(
                f"Symbol {sym_id} ({info.get('name', '?')}) = {value}, below "
                f"the minimum extent the container declares ({lo}); it binds "
                f"from {info.get('source', '?')}.")
        if hi is not None and value > hi:
            raise RuntimeError(
                f"Symbol {sym_id} ({info.get('name', '?')}) = {value}, above "
                f"the maximum extent the container declares ({hi}); it binds "
                f"from {info.get('source', '?')}.")
        self._bindings[sym_id] = value

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
                spec = parts[1]
                # optional divisor-annotated source: val_<i>_fd<k> binds
                # int(flat[i]) // k (merge-derived grid values)
                fdiv = 1
                if "_fd" in spec:
                    spec, fd = spec.split("_fd", 1)
                    fdiv = int(fd)
                idx = int(spec)
                tensor = inputs.get(tensor_id)
                if tensor is not None and hasattr(tensor, "numpy"):
                    flat = tensor.numpy().reshape(-1)
                    if idx >= flat.shape[0] or idx < -flat.shape[0]:
                        raise RuntimeError(
                            f"Symbol {sym_id}: val index {idx} out of range "
                            f"for input '{tensor_id}' with {flat.shape[0]} "
                            f"elements")
                    self._bind(sym_id, int(flat[idx]) // fdiv)
                elif tensor is not None and hasattr(tensor, "__len__"):
                    if idx >= len(tensor) or idx < -len(tensor):
                        raise RuntimeError(
                            f"Symbol {sym_id}: val index {idx} out of range "
                            f"for input '{tensor_id}' with {len(tensor)} "
                            f"elements")
                    self._bind(sym_id, int(tensor[idx]) // fdiv)
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
                    self._bind(sym_id, val)
                # An input the flow did not provide binds NOTHING: the symbol stays
                # unbound and refuses by name when it is used (never the trace value).
                continue

            # Dict format: {"tensor_id": "...", "dim": 0}
            if isinstance(src, dict):
                tensor_id = src.get("tensor_id")
                dim = src.get("dim")
                if tensor_id and dim is not None:
                    tensor = inputs.get(tensor_id)
                    if tensor is not None and hasattr(tensor, 'shape'):
                        self._bind(sym_id, tensor.shape[dim])

    def resolve(self, val) -> int:
        """Resolve a value in a SHAPE slot (an extent, a size-list element).

        Handles: int, a symbol id string, an expression tree dict of the shared vocabulary
        (core/runtime/symexpr.py). A real (true division, sqrt) refuses by name: an extent is
        an integer. A bare float literal keeps its integer reading, as before.
        """
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, (dict, str)):
            return _symexpr.evaluate(val, self._symbol_value, _symexpr.SHAPE)
        return int(val)

    def resolve_scalar(self, val):
        """Resolve an op's SCALAR argument: a symbol or an expression evaluates in the scalar
        slot (a `linspace` bound may be real); a dict outside the vocabulary refuses by name;
        a plain number is returned as it stands."""
        if isinstance(val, (dict, str)):
            return _symexpr.evaluate(val, self._symbol_value, _symexpr.SCALAR)
        return val

    def _symbol_value(self, sym_id: str, node) -> int:
        """The bound value of a symbol, or a refusal naming it (the shared evaluator's callback)."""
        if sym_id in self._bindings:
            return self._bindings[sym_id]
        self._refuse(sym_id, node)

    def is_bound(self, sym_id: str) -> bool:
        return sym_id in self._bindings

    def get(self, sym_id: str, default: int = 0) -> int:
        """The bound value, or `default` — callers deciding on boundness use `is_bound`,
        never a sentinel: a symbol legitimately bound to 0 (a cache length at its first
        step) is bound."""
        return self._bindings.get(sym_id, default)

    def _refuse(self, sym_id, expr) -> None:
        info = self._symbols.get(sym_id) or {}
        tv = (expr or {}).get("trace", (expr or {}).get("trace_value", info.get("trace_value")))
        raise UnboundSymbolError(
            f"ZERO FALLBACK: symbol '{sym_id}' ({info.get('name')}, binds from "
            f"{info.get('source')}) is not bound at runtime; its trace value {tv} is a "
            f"witnessed extent, not a value. Bound: {sorted(self._bindings)}.")

    @property
    def bindings(self) -> Dict[str, int]:
        return dict(self._bindings)
