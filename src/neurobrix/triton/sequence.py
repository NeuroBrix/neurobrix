"""Triton Compiled Sequence — zero-overhead execution hot loop.

Ported from compiled_sequence.py. Same proven logic for arg compilation,
closure-based resolution, and liveness analysis. Replaces torch.dtype
with NBXDtype and uses Arena + SymbolResolver for triton mode.

Zero torch dependency in the hot loop.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurobrix.kernels.dispatch import dispatch
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype

from .arena import Arena
from .symbols import SymbolResolver
from .dtype import TritonDtypeEngine


# ============================================================================
# ARGUMENT TYPES — compile-time only, never seen at runtime
# Ported from compiled_sequence.py (pure Python, no torch)
# ============================================================================

@dataclass(frozen=True)
class TensorSlot:
    """Reference to a tensor in the arena by index."""
    slot: int


@dataclass(frozen=True)
class ScalarArg:
    """Literal scalar value (int, float, bool, None)."""
    value: Any


@dataclass(frozen=True)
class ListArg:
    """List of values (may contain TensorSlots or ScalarArgs)."""
    items: Tuple[Any, ...]


@dataclass(frozen=True)
class DtypeArg:
    """Pre-resolved NBXDtype."""
    dtype: NBXDtype


@dataclass(frozen=True)
class SymbolArg:
    """Symbolic dimension resolved at runtime from actual input shapes."""
    symbol_id: str
    trace_value: int
    offset: int = 0


@dataclass(frozen=True)
class ProductArg:
    """Product of symbolic factors (e.g., s0 * s1 * 256)."""
    factors: Tuple[Any, ...]
    trace_value: int


@dataclass(frozen=True)
class ExprArg:
    """Symbolic expression tree (floordiv, add, sub, mul, mod, neg)."""
    expr_dict: dict
    trace_value: int


# ============================================================================
# COMPILED OP
# ============================================================================

_ACCUMULATOR_OPS = frozenset({"add", "mul"})


@dataclass
class CompiledOp:
    """Pre-compiled operation with closure-based argument resolution."""
    op_uid: str
    op_type: str
    func: Callable
    args_resolver: Callable
    kwargs_resolver: Callable
    output_slots: Tuple[int, ...]
    kill_slots: Tuple[int, ...] = ()


# ============================================================================
# TRITON COMPILED SEQUENCE
# ============================================================================

class TritonSequence:
    """Compiled execution sequence for triton mode.

    Ported from CompiledSequence. Same two-phase design:
    1. Compile: parse graph.json → dataclass args → closure resolvers
    2. Run: zero-overhead hot loop (no isinstance checks)
    """

    def __init__(self, dag: dict, device_idx: int = 0,
                 compute_dtype: NBXDtype = NBXDtype.float16):
        self.dag = dag
        self.device_idx = device_idx
        self._ops: List[CompiledOp] = []
        self._arena: Optional[Arena] = None

        # Mappings (built during compile)
        self._tid_to_slot: Dict[str, int] = {}
        self._weight_ids: List[str] = []
        self._input_ids: List[str] = []
        self._output_ids: List[str] = []

        # Slot counts
        self._num_weights: int = 0
        self._num_inputs: int = 0

        # Engines
        self._symbol_resolver: Optional[SymbolResolver] = None
        self._dtype_engine = TritonDtypeEngine(compute_dtype)
        self._compute_dtype = compute_dtype
        self._compiled = False

        # Op interceptors: op_type → callable (for KV cache injection)
        self._op_interceptors: Dict[str, Callable] = {}

    def register_op_interceptor(self, op_type: str, interceptor: Callable):
        """Register an interceptor for a specific op type (e.g., SDPA for KV cache)."""
        self._op_interceptors[op_type] = interceptor
        # Hot-patch already compiled ops if sequence is compiled
        if self._compiled:
            for op in self._ops:
                if op.op_type == op_type:
                    op.func = interceptor

    # ========================================================================
    # COMPILE
    # ========================================================================

    def compile(self):
        """Compile graph.json into op list + arena."""
        tensors = self.dag.get("tensors", {})
        ops_raw = self.dag.get("ops", {})
        exec_order = self.dag.get("execution_order", [])

        # Build op lookup
        if isinstance(ops_raw, list):
            ops_by_uid = {op["op_uid"]: op for op in ops_raw}
        else:
            ops_by_uid = ops_raw

        # Graph-declared outputs
        graph_output_ids = set(self.dag.get("output_tensor_ids", []))

        # Phase 0: Promote trace-time seq_len scalars to symbolic references.
        # Ported from compiled_sequence._promote_seq_len_scalars_to_symbolic.
        # Without this, ops like ones([23,23]) create fixed-size masks instead
        # of dynamically-sized ones matching actual input seq_len.
        self._promote_seq_len_scalars(tensors, ops_by_uid)

        # Phase 1: Categorize tensors and assign slots
        self._categorize_and_assign_slots(tensors, ops_by_uid, graph_output_ids)

        # Phase 2: Symbolic context
        sym_ctx = self.dag.get("symbolic_context", {})
        if sym_ctx:
            self._symbol_resolver = SymbolResolver(sym_ctx)

        # Phase 3: Liveness analysis
        dead_at_op = self._compute_liveness(exec_order, ops_by_uid)

        # Phase 4: Compile each op
        for op_idx, op_uid in enumerate(exec_order):
            op_data = ops_by_uid.get(op_uid)
            if op_data is None:
                continue

            kill_slots = tuple(dead_at_op.get(op_idx, []))
            compiled_op = self._compile_op(op_uid, op_data, tensors, kill_slots)
            self._ops.append(compiled_op)

        # Phase 5: Allocate arena
        total_slots = len(self._tid_to_slot)
        self._arena = Arena(total_slots, self._num_weights, self._num_inputs)

        self._compiled = True

    # ========================================================================
    # SYMBOLIC PROMOTION — ported from compiled_sequence
    # ========================================================================

    def _promote_seq_len_scalars(self, tensors: dict, ops_meta: dict):
        """Promote trace-time seq_len constants to symbolic references.

        Ported from compiled_sequence._promote_seq_len_scalars_to_symbolic.
        Detects concrete scalars matching seq_len trace values and replaces
        them with symbol references for runtime resolution.
        """
        sym_ctx = self.dag.get("symbolic_context", {})
        symbols = sym_ctx.get("symbols", {})

        # Find seq_len symbols
        seq_len_syms: Dict[str, int] = {}
        for sid, sinfo in symbols.items():
            if sinfo.get("name") == "seq_len":
                tv = sinfo.get("trace_value")
                if tv is not None:
                    seq_len_syms[sid] = tv

        if not seq_len_syms:
            return

        # Collision check: skip if trace_value appears in weight dimensions
        weight_dims: set = set()
        for tid, tdata in tensors.items():
            if tid.startswith("param::") or tid.startswith("buffer::"):
                wname = tdata.get("weight_name", "")
                if wname.startswith("constant_T_"):
                    continue
                for d in tdata.get("shape", []):
                    if isinstance(d, int):
                        weight_dims.add(d)

        safe: Dict[str, int] = {}
        for sid, tv in seq_len_syms.items():
            if tv not in weight_dims:
                safe[sid] = tv

        # Handle ambiguous trace values (multiple symbols share same value).
        # If all symbols with the same trace_value have the same name (e.g., all "seq_len"),
        # they resolve to the same runtime value — keep ONE representative.
        tv_counts: Dict[int, list] = {}
        for sid, tv in safe.items():
            tv_counts.setdefault(tv, []).append(sid)
        for tv, sids in tv_counts.items():
            if len(sids) > 1:
                # Check if all symbols with this value have the same name
                names = {symbols.get(s, {}).get("name") for s in sids}
                if len(names) == 1:
                    # Same semantic — keep the first, remove the rest
                    for sid in sids[1:]:
                        del safe[sid]
                else:
                    # Genuinely ambiguous — remove all
                    for sid in sids:
                        if sid in safe:
                            del safe[sid]

        if not safe:
            return

        def _promote_int(val: int):
            """Try to promote a raw int to symbolic ref."""
            for sid, tv in safe.items():
                offset = val - tv
                if 0 <= offset <= 1:
                    return {"type": "symbol", "symbol_id": sid,
                            "trace_value": val, "offset": offset}
            return None

        def _promote_scalar_dict(arg: dict):
            if arg.get("type") != "scalar":
                return None
            val = arg.get("value")
            if isinstance(val, int):
                return _promote_int(val)
            return None

        # Walk all ops and promote matching scalars
        for op_uid, op_data in ops_meta.items():
            op_type = op_data.get("op_type", "")
            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])

            # slice: promote end (index 3)
            if op_type == "aten::slice" and len(args) >= 4:
                if isinstance(args[3], dict):
                    r = _promote_scalar_dict(args[3])
                    if r:
                        args[3] = r

            # arange: promote end (index 0)
            elif op_type == "aten::arange" and len(args) >= 1:
                if isinstance(args[0], dict):
                    r = _promote_scalar_dict(args[0])
                    if r:
                        args[0] = r

            # ones/zeros/full: promote shape elements
            elif op_type in ("aten::full", "aten::zeros", "aten::ones",
                             "aten::new_zeros", "aten::new_ones"):
                shape_idx = 1 if op_type.startswith("aten::new_") else 0
                if len(args) > shape_idx:
                    shape_arg = args[shape_idx]
                    is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                    items = shape_arg.get("value", []) if is_wrapped else shape_arg
                    if isinstance(items, (list, tuple)):
                        items = list(items)
                        changed = False
                        for i, elem in enumerate(items):
                            if isinstance(elem, dict):
                                r = _promote_scalar_dict(elem)
                                if r:
                                    items[i] = r
                                    changed = True
                            elif isinstance(elem, int):
                                r = _promote_int(elem)
                                if r:
                                    items[i] = r
                                    changed = True
                        if changed:
                            if is_wrapped:
                                args[shape_idx] = {"type": "list", "value": items}
                            else:
                                args[shape_idx] = items

            # expand: promote size elements
            elif op_type == "aten::expand" and len(args) >= 2:
                size_arg = args[1]
                if isinstance(size_arg, (list, tuple)):
                    size_list = list(size_arg)
                    changed = False
                    for i, elem in enumerate(size_list):
                        if isinstance(elem, dict):
                            r = _promote_scalar_dict(elem)
                            if r:
                                size_list[i] = r
                                changed = True
                        elif isinstance(elem, int):
                            r = _promote_int(elem)
                            if r:
                                size_list[i] = r
                                changed = True
                    if changed:
                        args[1] = size_list

            # view/reshape/_unsafe_view: promote shape if no tracer symbols
            elif op_type in ("aten::view", "aten::reshape", "aten::_unsafe_view") and len(args) >= 2:
                shape_arg = args[1]
                is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                items = shape_arg.get("value", []) if is_wrapped else shape_arg
                if isinstance(items, (list, tuple)):
                    has_symbols = any(isinstance(e, dict) and e.get("type") == "symbol"
                                     for e in items)
                    if not has_symbols:
                        items = list(items)
                        changed = False
                        for i, elem in enumerate(items):
                            if isinstance(elem, dict):
                                r = _promote_scalar_dict(elem)
                                if r:
                                    items[i] = r
                                    changed = True
                            elif isinstance(elem, int):
                                r = _promote_int(elem)
                                if r:
                                    items[i] = r
                                    changed = True
                        if changed:
                            if is_wrapped:
                                args[1] = {"type": "list", "value": items}
                            else:
                                args[1] = items

            # narrow: promote length (index 3)
            elif op_type == "aten::narrow" and len(args) >= 4:
                if isinstance(args[3], dict):
                    r = _promote_scalar_dict(args[3])
                    if r:
                        args[3] = r

    # ========================================================================
    # SLOT ASSIGNMENT — ported from compiled_sequence._categorize_and_assign_slots
    # ========================================================================

    def _categorize_and_assign_slots(self, tensors: dict, ops_meta: dict,
                                     graph_output_ids: set):
        """Categorize tensors by prefix and assign arena slots.

        Layout: [weights | inputs | intermediates]
        """
        weights = []
        inputs = []
        intermediates = []

        for tid, tdata in tensors.items():
            if tid.startswith("param::") or tid.startswith("buffer::"):
                weights.append(tid)
            elif tid.startswith("input::"):
                inputs.append(tid)
            else:
                intermediates.append(tid)

            if tdata.get("is_output") or tid in graph_output_ids:
                if tid not in self._output_ids:
                    self._output_ids.append(tid)

        # Check ops for additional output tensors
        for _, op_data in ops_meta.items():
            for out_id in op_data.get("output_tensor_ids", []):
                if out_id not in tensors and out_id not in self._tid_to_slot:
                    intermediates.append(out_id)

        # Assign slots in order
        slot = 0
        for tid in weights:
            self._tid_to_slot[tid] = slot
            self._weight_ids.append(tid)
            slot += 1
        self._num_weights = len(weights)

        self._input_ids = list(self.dag.get("input_tensor_ids", []))
        for tid in inputs:
            if tid not in self._tid_to_slot:
                self._tid_to_slot[tid] = slot
                slot += 1
        self._num_inputs = slot - self._num_weights

        for tid in intermediates:
            if tid not in self._tid_to_slot:
                self._tid_to_slot[tid] = slot
                slot += 1

    # ========================================================================
    # LIVENESS ANALYSIS — ported from compiled_sequence._compute_liveness
    # ========================================================================

    def _extract_input_slots(self, op_data: dict) -> List[int]:
        """Extract input tensor slots from op data for liveness analysis."""
        slots = []
        attrs = op_data.get("attributes", {})

        def extract(arg):
            if not isinstance(arg, dict):
                return
            arg_type = arg.get("type")
            if arg_type == "tensor":
                tid = arg.get("tensor_id")
                if tid:
                    s = self._tid_to_slot.get(tid)
                    if s is not None:
                        slots.append(s)
            elif arg_type == "tensor_tuple":
                for tid in arg.get("tensor_ids", []):
                    s = self._tid_to_slot.get(tid)
                    if s is not None:
                        slots.append(s)
            elif arg_type == "list":
                for item in arg.get("value", []):
                    extract(item)

        for arg in attrs.get("args", []):
            extract(arg)
        for arg in attrs.get("kwargs", {}).values():
            extract(arg)
        return slots

    def _compute_liveness(self, exec_order: list, ops_meta: dict) -> Dict[int, List[int]]:
        """Compute dead tensor analysis — O(N) algorithm."""
        # Step 1: Find last usage of each slot
        slot_last_use: Dict[int, int] = {}
        for op_idx, op_uid in enumerate(exec_order):
            op_data = ops_meta.get(op_uid)
            if op_data is None:
                continue
            for s in self._extract_input_slots(op_data):
                slot_last_use[s] = op_idx

        # Step 2: Protected slots (never freed)
        protected = set()
        for i in range(self._num_weights):
            protected.add(i)
        input_start = self._num_weights
        for i in range(self._num_inputs):
            protected.add(input_start + i)
        for tid in self._output_ids:
            s = self._tid_to_slot.get(tid)
            if s is not None:
                protected.add(s)

        # Step 3: Build dead_at_op
        dead_at_op: Dict[int, List[int]] = defaultdict(list)
        for s, last_idx in slot_last_use.items():
            if s not in protected:
                dead_at_op[last_idx].append(s)
        return dict(dead_at_op)

    # ========================================================================
    # OP COMPILATION
    # ========================================================================

    def _compile_op(self, op_uid: str, op_data: dict, tensors: dict,
                    kill_slots: Tuple[int, ...]) -> CompiledOp:
        """Compile a single op with closure resolvers."""
        op_type = op_data.get("op_type", "")
        attrs = op_data.get("attributes", {})
        output_tids = op_data.get("output_tensor_ids", [])

        # Check for registered interceptor (e.g., KV cache for SDPA)
        if op_type in self._op_interceptors:
            func = self._op_interceptors[op_type]
        else:
            func = dispatch(op_type)
            if func is None:
                raise RuntimeError(f"[triton] Missing op: {op_type}")
            # Wrap with AMP dtype rules
            bare_name = op_type.split("::")[-1] if "::" in op_type else op_type
            func = self._dtype_engine.wrap_op(bare_name, func)

        # Compile args → dataclasses
        raw_args = attrs.get("args", [])
        raw_kwargs = attrs.get("kwargs", {})
        compiled_args = self._compile_args(raw_args, tensors)
        compiled_kwargs = self._compile_kwargs(raw_kwargs, tensors)

        # Generate closures
        args_resolver = self._make_args_resolver(compiled_args)
        kwargs_resolver = self._make_kwargs_resolver(compiled_kwargs)

        # Allocate output slots
        output_slots = []
        for tid in output_tids:
            if tid not in self._tid_to_slot:
                s = len(self._tid_to_slot)
                self._tid_to_slot[tid] = s
            output_slots.append(self._tid_to_slot[tid])

        return CompiledOp(
            op_uid=op_uid, op_type=op_type, func=func,
            args_resolver=args_resolver,
            kwargs_resolver=kwargs_resolver,
            output_slots=tuple(output_slots),
            kill_slots=kill_slots,
        )

    # ========================================================================
    # ARG COMPILATION — ported from compiled_sequence._compile_arg
    # ========================================================================

    def _compile_args(self, args: list, tensors: dict) -> Tuple[Any, ...]:
        return tuple(self._compile_arg(arg, tensors) for arg in args)

    def _compile_kwargs(self, kwargs: dict, tensors: dict) -> Dict[str, Any]:
        return {k: self._compile_arg(v, tensors) for k, v in kwargs.items()}

    def _compile_arg(self, arg: Any, tensors: dict) -> Any:
        """Pre-compile a single argument to a typed object.

        Ported from compiled_sequence._compile_arg — same logic,
        NBXDtype instead of torch.dtype.
        """
        if arg is None:
            return ScalarArg(None)

        if isinstance(arg, dict):
            arg_type = arg.get("type")

            if arg_type in ("tensor", "tensor_ref"):
                tid = arg.get("tensor_id")
                if tid in self._tid_to_slot:
                    return TensorSlot(self._tid_to_slot[tid])
                return arg

            if arg_type == "tensor_tuple":
                tids = arg.get("tensor_ids", [])
                slots = []
                for tid in tids:
                    if tid in self._tid_to_slot:
                        slots.append(TensorSlot(self._tid_to_slot[tid]))
                    else:
                        slots.append(tid)
                return ListArg(tuple(slots))

            if arg_type == "dtype":
                dtype_str = arg.get("value", "float32")
                return DtypeArg(self._parse_dtype(dtype_str))

            if arg_type == "device":
                return ScalarArg(f"cuda:{self.device_idx}")

            if arg_type == "list":
                items = arg.get("value", [])
                compiled_items = tuple(self._compile_arg(item, tensors) for item in items)
                return ListArg(compiled_items)

            if arg_type == "scalar":
                return ScalarArg(arg.get("value"))

            if arg_type == "symbol":
                symbol_id = arg.get("symbol_id") or arg.get("id") or arg.get("name")
                if symbol_id is None:
                    raise ValueError(f"symbol arg missing identifier: {arg}")
                trace_value = arg.get("trace_value", arg.get("trace", 0))
                offset = arg.get("offset", 0)
                return SymbolArg(symbol_id=symbol_id, trace_value=trace_value, offset=offset)

            if arg_type == "product":
                factors_raw = arg.get("factors", [])
                trace_value = arg.get("trace_value", 0)
                compiled_factors = []
                for f in factors_raw:
                    if isinstance(f, dict) and f.get("type") == "symbol":
                        compiled_factors.append(f.get("symbol_id") or f.get("id") or f.get("name"))
                    elif isinstance(f, dict):
                        compiled_factors.append(f.get("value", f.get("trace_value", 0)))
                    elif isinstance(f, str):
                        compiled_factors.append(f)
                    else:
                        compiled_factors.append(f)
                return ProductArg(factors=tuple(compiled_factors), trace_value=trace_value)

            if arg_type in ("floordiv", "add", "sub", "mul", "mod", "neg"):
                trace = arg.get("trace", arg.get("trace_value", 0))
                return ExprArg(expr_dict=arg, trace_value=trace)

            if arg_type in ("int", "float", "bool", "none"):
                return ScalarArg(arg.get("value"))

            if arg_type in ("memory_format", "layout"):
                return ScalarArg(None)

            if arg_type == "unknown":
                value = arg.get("value")
                if isinstance(value, str):
                    if value.startswith("torch."):
                        stripped = value.replace("torch.", "")
                        try:
                            return DtypeArg(parse_dtype(stripped))
                        except Exception:
                            pass
                return ScalarArg(value)

            # Dict with value field
            if "value" in arg:
                return ScalarArg(arg["value"])

            # Raw expression (from SymInt.to_json)
            if "symbol_id" in arg or "op" in arg:
                trace = arg.get("trace_value", 0)
                return ExprArg(expr_dict=arg, trace_value=trace)

            return arg

        if isinstance(arg, (int, float, bool)):
            return ScalarArg(arg)

        if isinstance(arg, str):
            if arg in self._tid_to_slot:
                return TensorSlot(self._tid_to_slot[arg])
            return ScalarArg(arg)

        if isinstance(arg, (list, tuple)):
            compiled_items = tuple(self._compile_arg(item, tensors) for item in arg)
            return ListArg(compiled_items)

        return arg

    def _parse_dtype(self, dtype_str: str) -> NBXDtype:
        """Parse dtype string to NBXDtype with Prism remap.

        Remaps bf16↔fp16 based on compute_dtype from Prism:
        - compute_dtype=fp16 (V100): bf16 in graph → fp16
        - compute_dtype=bf16 (A100): fp16 in graph → bf16
        This matches core/dtype/config.parse_dtype(s, compute_dtype=...).
        """
        s = dtype_str.replace("torch.", "")
        parsed = parse_dtype(s)
        # Remap half-precision types to match hardware
        if parsed == NBXDtype.bfloat16 and self._compute_dtype == NBXDtype.float16:
            return NBXDtype.float16
        if parsed == NBXDtype.float16 and self._compute_dtype == NBXDtype.bfloat16:
            return NBXDtype.bfloat16
        return parsed

    # ========================================================================
    # CLOSURE GENERATORS — ported from compiled_sequence._make_*_resolver
    # ========================================================================

    def _make_args_resolver(self, compiled_args: Tuple[Any, ...]) -> Callable:
        """Generate closure that resolves args without isinstance at runtime."""
        resolvers = []
        for arg in compiled_args:
            resolvers.append(self._make_single_resolver(arg))
        resolvers_t = tuple(resolvers)
        return lambda arena: [r(arena) for r in resolvers_t]

    def _make_kwargs_resolver(self, compiled_kwargs: Dict[str, Any]) -> Callable:
        """Generate closure that resolves kwargs without isinstance at runtime."""
        if not compiled_kwargs:
            return lambda _arena: {}
        keys = tuple(compiled_kwargs.keys())
        resolvers = tuple(self._make_single_resolver(v) for v in compiled_kwargs.values())
        return lambda arena: {k: r(arena) for k, r in zip(keys, resolvers)}

    def _make_single_resolver(self, arg: Any) -> Callable:
        """Create a resolver closure for a single compiled arg."""
        if isinstance(arg, TensorSlot):
            s = arg.slot
            return lambda arena, s=s: arena[s]
        elif isinstance(arg, ScalarArg):
            v = arg.value
            return lambda _arena, v=v: v
        elif isinstance(arg, DtypeArg):
            dt = arg.dtype
            return lambda _arena, dt=dt: dt
        elif isinstance(arg, SymbolArg):
            return self._make_symbol_resolver(arg.symbol_id, arg.trace_value, arg.offset)
        elif isinstance(arg, ProductArg):
            return self._make_product_resolver(arg.factors, arg.trace_value)
        elif isinstance(arg, ExprArg):
            return self._make_expr_resolver(arg.expr_dict, arg.trace_value)
        elif isinstance(arg, ListArg):
            return self._make_list_resolver(arg.items)
        else:
            val = arg
            return lambda _arena, val=val: val

    def _make_list_resolver(self, items: Tuple[Any, ...]) -> Callable:
        """Generate resolver for list arguments (recursive)."""
        item_resolvers = tuple(self._make_single_resolver(item) for item in items)
        return lambda arena, rs=item_resolvers: [r(arena) for r in rs]

    # ========================================================================
    # SYMBOL RESOLVERS — ported from compiled_sequence._make_*_resolver
    # ========================================================================

    def _make_symbol_resolver(self, symbol_id: str, trace_value: int,
                              offset: int = 0) -> Callable:
        """Closure that resolves a symbol at runtime via SymbolResolver."""
        def resolve(_arena):
            if self._symbol_resolver is not None:
                v = self._symbol_resolver.get(symbol_id)
                if v > 0:
                    return v + offset
            return trace_value + offset
        return resolve

    def _make_product_resolver(self, factors: Tuple[Any, ...],
                               trace_value: int) -> Callable:
        """Closure that computes product of symbolic factors at runtime."""
        def resolve(_arena):
            if self._symbol_resolver is not None:
                result = 1
                for f in factors:
                    if isinstance(f, str):
                        v = self._symbol_resolver.get(f)
                        if v > 0:
                            result *= v
                        else:
                            return trace_value
                    elif isinstance(f, (int, float)):
                        result *= int(f)
                    else:
                        return trace_value
                return result
            return trace_value
        return resolve

    def _make_expr_resolver(self, expr_dict: dict, trace_value: int) -> Callable:
        """Closure that evaluates expression tree at runtime."""
        def resolve(_arena):
            if self._symbol_resolver is not None:
                try:
                    return self._symbol_resolver.resolve(expr_dict)
                except Exception:
                    return trace_value
            return trace_value
        return resolve

    # ========================================================================
    # BIND / RUN / GATHER
    # ========================================================================

    def bind_weights(self, weights: Dict[str, NBXTensor]):
        """Bind weight tensors to arena slots."""
        for tid in self._weight_ids:
            tdata = self.dag.get("tensors", {}).get(tid, {})
            wname = tdata.get("weight_name", "")
            if wname in weights:
                self._arena[self._tid_to_slot[tid]] = weights[wname]

    def bind_inputs(self, inputs: Dict[str, NBXTensor]):
        """Bind input tensors to arena slots."""
        for tid, tensor in inputs.items():
            slot = self._tid_to_slot.get(tid)
            if slot is not None:
                self._arena[slot] = tensor

    def bind_symbols(self, inputs: dict):
        """Bind symbolic shapes from actual input tensors."""
        if self._symbol_resolver:
            self._symbol_resolver.bind_from_inputs(
                inputs, self._input_ids, self.dag.get("tensors", {}))

    def run(self):
        """Execute all ops. Zero overhead hot loop."""
        DeviceAllocator.ensure_triton_device(self.device_idx)

        arena = self._arena
        for op in self._ops:
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            # NOP propagation (deactivated MoE paths)
            if args and args[0] is None:
                bare = op.op_type.split("::")[-1]
                if bare in _ACCUMULATOR_OPS:
                    result = args[0]
                else:
                    for s in op.output_slots:
                        arena[s] = None
                    for s in op.kill_slots:
                        arena[s] = None
                    continue
            else:
                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed at {op.op_uid} ({op.op_type}): {e}") from e

            # Store outputs
            if len(op.output_slots) == 1:
                arena[op.output_slots[0]] = result
            elif isinstance(result, tuple):
                for i, s in enumerate(op.output_slots):
                    arena[s] = result[i] if i < len(result) else None
            else:
                arena[op.output_slots[0]] = result

            # Kill dead slots
            for s in op.kill_slots:
                arena[s] = None

    def gather_outputs(self, output_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Read output tensors from arena."""
        ids = output_ids or self._output_ids
        outputs = {}
        for tid in ids:
            slot = self._tid_to_slot.get(tid)
            if slot is not None and self._arena[slot] is not None:
                outputs[tid] = self._arena[slot]
        return outputs

    @property
    def num_ops(self) -> int:
        return len(self._ops)
