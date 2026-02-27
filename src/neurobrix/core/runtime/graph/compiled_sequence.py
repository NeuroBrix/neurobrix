"""
CompiledSequence - Zero-Overhead Pre-compiled Execution Sequence

Eliminates ALL Python overhead in the execution loop by:
1. Pre-resolving all tensor lookups to integer indices
2. Pre-binding function references (no string dispatch)
3. Using list-based arena with __slots__ instead of dict-based tensor_store
4. Pre-generating resolver CLOSURES that eliminate isinstance() at runtime

Performance gains vs legacy:
- Legacy: isinstance() check for EVERY arg at runtime (~100ns each)
- CompiledSequence: Pre-compiled closures, zero isinstance() in hot loop

Based on:
- CUDA Graphs (capture/replay pattern)
- JAX JIT (tracing + compilation)
- TorchDynamo (graph mode)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import os
import torch

from neurobrix.core.dtype.config import parse_dtype as _cfg_parse_dtype, DTYPE_MAP as _DTYPE_MAP
from .compiled_ops import CompiledOpResolver

# ============================================================================
# DEBUG FLAGS — read once at import time, not per-step
# ============================================================================
_TRACE_NAN = os.environ.get("NBX_TRACE_NAN", "0") == "1"
_NAN_GUARD = os.environ.get("NBX_NAN_GUARD", "0") == "1"
_NAN_GUARD_VERBOSE = os.environ.get("NBX_NAN_GUARD_VERBOSE", "0") == "1"
_TRACE_ZEROS = os.environ.get("NBX_TRACE_ZEROS", "0") == "1"


# ============================================================================
# ACCUMULATOR OPS — pass through base tensor when source args are None
# Used by NOP propagation for dynamic MoE routing (deactivated expert paths).
# These ops take (base_tensor, dim, indices, source) and accumulate into base.
# When an expert is deactivated, indices/source are None → pass through base.
# ============================================================================
_ACCUMULATOR_OPS = frozenset({
    "aten::scatter_reduce", "aten::scatter_add", "aten::index_add",
    "aten::scatter", "aten::index_put",
})


def _has_none_arg(args: tuple) -> bool:
    """Check if any arg is None, including inside list/tuple args.

    MoE routing produces dynamic-length unbind outputs. When fewer experts
    are active than at trace time, excess slots are None. These None values
    can appear as top-level args OR inside list args (e.g. aten::index(t, [None])).
    """
    for a in args:
        if a is None:
            return True
        if isinstance(a, (list, tuple)):
            if any(item is None for item in a):
                return True
    return False


# ============================================================================
# ARGUMENT TYPES (compile-time only, never seen at runtime)
# ============================================================================

@dataclass(frozen=True)
class TensorSlot:
    """Reference to a tensor in the memory arena by index."""
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
    """Pre-resolved torch dtype."""
    dtype: torch.dtype


@dataclass(frozen=True)
class SymbolArg:
    """
    Reference to a symbolic dimension that must be resolved at runtime.

    Unlike ScalarArg which captures a static value at compile time,
    SymbolArg defers resolution until execution when actual input
    tensor shapes are known.

    Fields:
        symbol_id: Symbol identifier (e.g., "s0", "s1")
        trace_value: Value at trace time (fallback for graphs without symbolic context)
        offset: Constant offset added to resolved value (for seq_len + 1 cases)
    """
    symbol_id: str
    trace_value: int
    offset: int = 0


@dataclass(frozen=True)
class ProductArg:
    """
    Reference to a symbolic product expression (e.g., s0 * s1).

    Factors can be symbol IDs (strings) or concrete integers.
    Resolution multiplies all resolved factor values.

    Fields:
        factors: Tuple of factor references (symbol_ids or ints)
        trace_value: Product value at trace time (fallback)
    """
    factors: Tuple[Any, ...]
    trace_value: int


# ============================================================================
# TENSOR ARENA WITH __slots__ FOR MAXIMUM SPEED
# ============================================================================

class TensorArena:
    """
    Ultra-fast tensor storage using __slots__ for O(1) access.

    Slot ordering: [weights...][inputs...][intermediates...]
    This allows efficient clear_intermediates() by only clearing tail slots.
    """
    __slots__ = ('_memory', '_num_weights', '_num_inputs')

    def __init__(self, size: int, num_weights: int = 0, num_inputs: int = 0):
        self._memory: List[Optional[torch.Tensor]] = [None] * size
        self._num_weights = num_weights
        self._num_inputs = num_inputs

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        return self._memory[idx]

    def __setitem__(self, idx: int, value: torch.Tensor) -> None:
        self._memory[idx] = value

    def clear_intermediates(self) -> None:
        """Clear only intermediate tensors (keep weights and inputs)."""
        start = self._num_weights + self._num_inputs
        for i in range(start, len(self._memory)):
            self._memory[i] = None

    def clear_inputs(self) -> None:
        """Clear input tensors (for next inference)."""
        start = self._num_weights
        end = self._num_weights + self._num_inputs
        for i in range(start, end):
            self._memory[i] = None

    def clear_all(self) -> None:
        """Clear all tensors."""
        for i in range(len(self._memory)):
            self._memory[i] = None


# ============================================================================
# COMPILED OP WITH PRE-COMPILED RESOLVERS
# ============================================================================

@dataclass
class CompiledOp:
    """
    Pre-compiled operation with closure-based argument resolution.

    The resolvers are closures generated at compile time that capture
    slot indices. At runtime, zero isinstance() checks are needed.

    kill_slots: Tensor slots to free AFTER this op executes (liveness analysis).
    """
    op_uid: str                              # For debugging only
    op_type: str                             # For debugging only
    func: Callable                           # Direct function reference
    args_resolver: Callable[[TensorArena], List[Any]]   # Closure: arena -> [args]
    kwargs_resolver: Callable[[TensorArena], Dict[str, Any]]  # Closure: arena -> {kwargs}
    output_slots: Tuple[int, ...]            # Support multi-output ops (split, chunk)
    kill_slots: Tuple[int, ...] = ()         # Slots to free after execution (Dead Tensor Analysis)
    weight_input_slots: Tuple[int, ...] = () # Weight/buffer slots consumed by this op (for FGP device derivation)
    all_input_slots: Tuple[int, ...] = ()    # ALL input slots (weights + activations) for cross-device detection
    device: Optional[torch.device] = None     # torch.device derived from weight placement (set by compute_op_devices)
    needs_transfer: bool = False              # True only for ops at GPU boundary (set by compute_op_devices)


# ============================================================================
# COMPILED SEQUENCE WITH CLOSURE-BASED RESOLUTION
# ============================================================================

class CompiledSequence:
    """
    Pre-compiled execution sequence with zero runtime isinstance() checks.

    Key innovation: Instead of storing TensorSlot/ScalarArg objects and
    checking their type at runtime, we generate CLOSURES at compile time
    that directly access the arena or return constant values.

    Memory Arena Layout:
        [0..W-1]              : Weight tensors (persistent)
        [W..W+I-1]            : Input tensors (per-inference)
        [W+I..N-1]            : Intermediate tensors (cleared between steps)

    Usage:
        seq = CompiledSequence(dag, device, dtype)  # 100% autonomous
        seq.compile()

        # Once at load time
        seq.bind_weights(weights_dict)

        # Per inference
        seq.bind_inputs(inputs_dict)
        seq.run()  # ZERO overhead hot loop
        outputs = seq.gather_outputs()
    """

    __slots__ = (
        'dag', 'op_resolver', 'device', 'dtype',
        '_ops', '_arena',
        '_tensor_id_to_slot', '_slot_to_tensor_id',
        '_weight_tensor_ids', '_input_tensor_ids', '_output_tensor_ids',
        '_num_weights', '_num_inputs', '_num_intermediates',
        '_compiled', '_next_slot',
        '_shape_resolver',  # SymbolicShapeResolver for runtime symbol resolution
        '_is_multi_device',  # FGP: True when weights span multiple devices
        '_persistent_tensor_ids',  # Protected from liveness GC (e.g., hidden states for LLM)
        '_op_interceptors',  # Op interceptors for KV cache (maps op_type -> interceptor)
        '_seq_dependent_constants',  # Constants with trace-time seq_len dim: [(slot, axis, sym_id, trace_val)]
        '_seq_constant_originals',  # Original full-size constants: {slot: tensor} — never narrowed
    )

    def __init__(
        self,
        dag: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize CompiledSequence.

        NOTE: This is 100% AUTONOMOUS - no dependency on NativeATenDispatcher.
        All op resolution is handled by CompiledOpResolver.

        Args:
            dag: The TensorDAG dict from graph.json
            device: The target device (e.g., torch.device("cuda:0"))
            dtype: The target dtype (e.g., torch.float16)
        """
        self.dag = dag
        self.device = device
        self.dtype = dtype

        # Extract graph's traced dtype for AMP policy decision
        graph_dtype_str = dag.get("torch_dtype", "")
        graph_dtype = _cfg_parse_dtype(graph_dtype_str) if graph_dtype_str else None

        # 100% Autonomous op resolution - no sequential_dispatcher dependency
        self.op_resolver = CompiledOpResolver(device, dtype, graph_dtype=graph_dtype)

        # Compilation outputs
        self._ops: List[CompiledOp] = []
        self._arena: Optional[TensorArena] = None

        # Mappings (built during compile)
        self._tensor_id_to_slot: Dict[str, int] = {}
        self._slot_to_tensor_id: Dict[int, str] = {}

        # Tensor categories
        self._weight_tensor_ids: List[str] = []
        self._input_tensor_ids: List[str] = []
        self._output_tensor_ids: List[str] = []

        # Slot counts for arena layout
        self._num_weights: int = 0
        self._num_inputs: int = 0
        self._num_intermediates: int = 0

        # State
        self._compiled: bool = False
        self._next_slot: int = 0

        # Symbolic shape resolver for runtime symbol resolution
        self._shape_resolver = None

        # FGP: Multi-device flag (set by compute_op_devices after bind_weights)
        self._is_multi_device = False

        # Persistent tensor IDs: protected from liveness GC (e.g., hidden states for LLM)
        self._persistent_tensor_ids: set = set()

        # Op interceptors for KV cache injection (maps op_type -> interceptor callable)
        self._op_interceptors: Dict[str, Callable] = {}

        # Seq-dependent constants: RoPE cos/sin with trace-time seq_len dimension.
        # Populated at compile time, sliced at runtime after symbol binding.
        # Each entry: (slot, axis, symbol_id, trace_value)
        self._seq_dependent_constants: List[Tuple[int, int, str, int]] = []

        # Original full-size constants for seq-dependent slots.
        # update_seq_dependent_constants() narrows arena tensors in-place,
        # so we must always narrow from the ORIGINAL, not the previously-narrowed view.
        self._seq_constant_originals: Dict[int, torch.Tensor] = {}

    def register_op_interceptor(self, op_type: str, interceptor: Callable) -> None:
        """
        Register an interceptor for a specific op type.

        Used for KV cache injection in LLM execution. When an interceptor is
        registered, the compiled op will call the interceptor instead of the
        native op function.

        Args:
            op_type: ATen op type (e.g., "aten::scaled_dot_product_attention")
            interceptor: Callable that receives the same args as the native op

        Note: Must be called BEFORE compile() for the interceptor to take effect.
              If called after compile(), call recompile() to pick up changes.
        """
        self._op_interceptors[op_type] = interceptor

    def clear_op_interceptors(self) -> None:
        """
        Clear all registered op interceptors.

        Call recompile() after this if the sequence was already compiled.
        """
        self._op_interceptors.clear()

    def update_op_interceptors(self, interceptors: Dict[str, Callable]) -> None:
        """
        Hot-swap interceptor functions in already-compiled ops.

        Avoids full recompilation when only the interceptor closures change
        (e.g., new KV cache wrapper on a new request). Walks compiled ops
        and patches func references for matching op_types.
        """
        self._op_interceptors.update(interceptors)
        if not self._ops:
            return
        patched = 0
        for op in self._ops:
            if op.op_type in interceptors:
                op.func = interceptors[op.op_type]
                patched += 1

    def compile(self) -> None:
        """
        Compile the DAG into a sequence of CompiledOp with closure resolvers.

        This performs ALL lookups and type resolution once, generating closures
        that can execute without any isinstance() checks.
        """
        if self._compiled:
            return

        tensors = self.dag.get("tensors", {})
        ops_metadata = self.dag.get("ops", {})
        execution_order = self.dag.get("execution_order", [])

        # Phase -1: Eliminate aten::detach ops (identity at inference time — no autograd)
        # DeepSeek: 19,428 detach out of 44,634 total ops (43%)
        self._eliminate_detach_ops(tensors, ops_metadata, execution_order)

        # Phase 0: Promote trace-time seq_len constants to symbolic references
        # UNIVERSAL: Works for all LLMs, safe for diffusion models, collision-checked
        self._promote_seq_len_scalars_to_symbolic(tensors, ops_metadata)

        # Graph-declared outputs MUST be protected from liveness analysis
        graph_output_ids = set(self.dag.get("output_tensor_ids", []))

        # Phase 1: Categorize tensors and assign slots in order: weights, inputs, intermediates
        self._categorize_and_assign_slots(tensors, ops_metadata, graph_output_ids)

        # Phase 1b: Identify constant tensors with trace-time seq_len dimensions
        # These need dynamic slicing at runtime (RoPE cos/sin from CPU-computed rotary_emb)
        self._identify_seq_dependent_constants(tensors)

        # Phase 2: LIVENESS ANALYSIS - Find when each slot becomes dead
        # This is O(N) where N = number of ops
        dead_at_op = self._compute_liveness(execution_order, ops_metadata)

        # Phase 3: Compile each op with closure resolvers and kill_slots
        for op_idx, op_uid in enumerate(execution_order):
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                raise RuntimeError(f"Op '{op_uid}' not found in ops metadata")

            # Get kill_slots for this op (slots to free AFTER this op executes)
            kill_slots = tuple(dead_at_op.get(op_idx, []))

            compiled_op = self._compile_op(op_uid, op_data, tensors, kill_slots)
            self._ops.append(compiled_op)

        # Phase 4: Allocate arena with proper layout
        total_slots = self._num_weights + self._num_inputs + self._num_intermediates
        self._arena = TensorArena(total_slots, self._num_weights, self._num_inputs)

        self._compiled = True

    def _eliminate_detach_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """
        Remove aten::detach ops — identity at inference time (no autograd).

        Detach ops are captured by the tracer for every parameter access.
        They produce a new tensor ID that aliases the input tensor. We rewire
        all references to the output tensor to point to the input tensor instead,
        then remove the op from execution_order.

        Impact: -43% ops for DeepSeek (19,428/44,634), -36% for T5 text encoders.
        """
        # Build rewire map: detach_output_tid -> detach_input_tid
        rewire: Dict[str, str] = {}
        detach_uids: set = set()

        for op_uid in execution_order:
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue
            if op_data.get("op_type") != "aten::detach":
                continue

            # Detach has exactly 1 input tensor and 1 output tensor
            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])
            in_tid = None
            for arg in args:
                if isinstance(arg, dict) and arg.get("type") == "tensor":
                    in_tid = arg.get("tensor_id")
                    break
            # Fallback to input_tensor_ids
            if in_tid is None:
                input_tids = op_data.get("input_tensor_ids", [])
                if input_tids:
                    in_tid = input_tids[0]

            out_tids = op_data.get("output_tensor_ids", [])
            if in_tid is None or not out_tids:
                continue

            out_tid = out_tids[0]

            # Chase rewire chains: if in_tid was itself rewired, follow to root
            while in_tid in rewire:
                in_tid = rewire[in_tid]

            rewire[out_tid] = in_tid
            detach_uids.add(op_uid)

        if not detach_uids:
            return

        # Apply rewire to all tensor references in remaining ops
        def _rewire_arg(arg: Any) -> Any:
            if not isinstance(arg, dict):
                return arg
            arg_type = arg.get("type")
            if arg_type == "tensor":
                tid = arg.get("tensor_id")
                if tid in rewire:
                    arg = dict(arg)
                    arg["tensor_id"] = rewire[tid]
            elif arg_type == "tensor_tuple":
                tids = arg.get("tensor_ids", [])
                new_tids = [rewire.get(t, t) for t in tids]
                if new_tids != tids:
                    arg = dict(arg)
                    arg["tensor_ids"] = new_tids
            elif arg_type == "list":
                items = arg.get("value", [])
                new_items = [_rewire_arg(item) for item in items]
                arg = dict(arg)
                arg["value"] = new_items
            return arg

        for op_uid in execution_order:
            if op_uid in detach_uids:
                continue
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue

            attrs = op_data.get("attributes", {})

            # Rewire args
            args = attrs.get("args", [])
            new_args = [_rewire_arg(a) for a in args]
            if new_args != args:
                attrs["args"] = new_args

            # Rewire kwargs
            kwargs = attrs.get("kwargs", {})
            if kwargs:
                new_kwargs = {k: _rewire_arg(v) for k, v in kwargs.items()}
                if new_kwargs != kwargs:
                    attrs["kwargs"] = new_kwargs

            # Rewire input_tensor_ids
            input_tids = op_data.get("input_tensor_ids", [])
            if input_tids:
                new_input_tids = [rewire.get(t, t) for t in input_tids]
                if new_input_tids != input_tids:
                    op_data["input_tensor_ids"] = new_input_tids

        # Rewire output_tensor_ids of the DAG itself
        dag_outputs = self.dag.get("output_tensor_ids", [])
        if dag_outputs:
            new_dag_outputs = [rewire.get(t, t) for t in dag_outputs]
            if new_dag_outputs != dag_outputs:
                self.dag["output_tensor_ids"] = new_dag_outputs

        # Rewire tensor metadata: if output_tid had is_output=True, transfer to input_tid
        for out_tid, in_tid in rewire.items():
            out_meta = tensors.get(out_tid, {})
            if out_meta.get("is_output"):
                if in_tid in tensors:
                    tensors[in_tid]["is_output"] = True

        # Remove detach ops from execution_order
        new_order = [uid for uid in execution_order if uid not in detach_uids]
        execution_order.clear()
        execution_order.extend(new_order)

    def _promote_seq_len_scalars_to_symbolic(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
    ) -> None:
        """
        Universal symbolic promotion for sequence-length-derived constants.

        When a graph is traced with a specific seq_len (e.g., 96), ops like
        aten::slice capture the seq_len as a concrete scalar. At runtime with
        a different seq_len, these baked values cause shape mismatches.

        This method detects such constants and promotes them to symbolic
        references that resolve dynamically at runtime.

        Safety: Only promotes when the trace seq_len is COLLISION-SAFE
        (doesn't appear in any model weight dimension). This prevents
        false positives where a model constant happens to equal trace_seq_len.

        Works for:
        - All LLMs (DeepSeek-V2, future models traced with collision-safe seq_len)
        - Diffusion models: no seq_len symbols → no promotion → unchanged
        - Old LLMs (traced with collision-prone seq_len): collision detected → skipped
        """
        symbolic_context = self.dag.get("symbolic_context", {})
        symbols = symbolic_context.get("symbols", {})

        # Find seq_len symbols and their trace values
        seq_len_symbols: Dict[str, int] = {}
        for sym_id, sym_info in symbols.items():
            if sym_info.get("name") == "seq_len":
                trace_val = sym_info.get("trace_value")
                if trace_val is not None:
                    seq_len_symbols[sym_id] = trace_val

        if not seq_len_symbols:
            return  # No seq_len symbols (diffusion models, audio, etc.)

        # Collision check: if trace_value appears in any weight/buffer shape,
        # it could be a model constant (head_dim, hidden_size, etc.).
        # Only promote if collision-safe.
        # Exclude constant_T_* tensors — these are trace-time computed buffers
        # (RoPE cos/sin, position embeddings) whose shapes inherently depend on
        # the trace-time seq_len. Their dim matching the trace value is expected,
        # not a collision with a real model dimension.
        weight_dims: set = set()
        for tid, tdata in tensors.items():
            if tid.startswith("param::") or tid.startswith("buffer::"):
                wname = tdata.get("weight_name", "")
                if wname.startswith("constant_T_"):
                    continue  # Skip trace-time computed constants
                shape = tdata.get("shape", [])
                for d in shape:
                    if isinstance(d, int):
                        weight_dims.add(d)

        safe_symbols: Dict[str, int] = {}
        for sym_id, trace_val in seq_len_symbols.items():
            if trace_val not in weight_dims:
                safe_symbols[sym_id] = trace_val

        # Also detect COMBINED seq_len values (sums of pairs).
        # FLUX-style models concatenate img+txt tokens, producing shapes like
        # 768 = 256(img) + 512(txt). The tracer captures these as concrete values.
        seq_len_list = list(seq_len_symbols.items())
        for i, (sid_a, tv_a) in enumerate(seq_len_list):
            for sid_b, tv_b in seq_len_list[i+1:]:
                if tv_a != tv_b:
                    sum_trace = tv_a + tv_b
                    if sum_trace not in weight_dims:
                        sum_id = f"_sum_{sid_a}_{sid_b}"
                        safe_symbols[sum_id] = sum_trace

        if not safe_symbols:
            return  # All seq_len trace values collide with weight dims

        # Promote scalar args in shape-manipulating ops to symbolic references
        promoted = 0

        def _try_promote_scalar(arg: dict) -> Optional[dict]:
            """Try to promote a scalar arg to a symbolic reference. Returns new arg or None."""
            if not isinstance(arg, dict) or arg.get("type") != "scalar":
                return None
            val = arg.get("value")
            if not isinstance(val, int):
                return None
            for sym_id, trace_val in safe_symbols.items():
                offset = val - trace_val
                if 0 <= offset <= 1:
                    return {
                        "type": "symbol",
                        "symbol_id": sym_id,
                        "trace_value": val,
                        "offset": offset,
                    }
            return None

        def _try_promote_raw_int(val) -> Optional[dict]:
            """Try to promote a raw int value (not wrapped in dict) to symbolic."""
            if not isinstance(val, int):
                return None
            for sym_id, trace_val in safe_symbols.items():
                offset = val - trace_val
                if 0 <= offset <= 1:
                    return {
                        "type": "symbol",
                        "symbol_id": sym_id,
                        "trace_value": val,
                        "offset": offset,
                    }
            return None

        for op_uid, op_data in ops_metadata.items():
            op_type = op_data.get("op_type", "")
            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])

            # aten::slice(tensor, dim, start, end) — promote end (index 3)
            # For RoPE table slices: pre-computed cos/sin tables are indexed
            # by absolute position_ids, not by seq_len. Promoting their end to
            # symbolic seq_len would truncate the table during decode (seq_len=1).
            # Instead, set end to the full table size so the slice is a no-op
            # and the full table is always available for absolute position indexing.
            if op_type == "aten::slice" and len(args) >= 4:
                _is_rope_table = False
                input_tids = op_data.get("input_tensor_ids", [])
                if input_tids:
                    first_input = input_tids[0]
                    if isinstance(first_input, str) and first_input.startswith("param::"):
                        wname = first_input[7:]  # Strip "param::"
                        if any(k in wname for k in ("cos_cached", "sin_cached",
                                                     "cos_cache", "sin_cache")):
                            _is_rope_table = True
                if _is_rope_table:
                    # Replace end with full table size → slice becomes identity
                    table_shape = tensors.get(input_tids[0], {}).get("shape", [])
                    dim_arg = args[1] if isinstance(args[1], int) else (
                        args[1].get("value") if isinstance(args[1], dict) else 0
                    )
                    if isinstance(dim_arg, int) and dim_arg < len(table_shape):
                        full_size = table_shape[dim_arg]
                        args[3] = {"type": "scalar", "value": full_size}
                else:
                    result = _try_promote_scalar(args[3])
                    if result:
                        args[3] = result
                        promoted += 1

            # aten::narrow(tensor, dim, start, length) — promote length (index 3)
            elif op_type == "aten::narrow" and len(args) >= 4:
                result = _try_promote_scalar(args[3])
                if result:
                    args[3] = result
                    promoted += 1

            # aten::arange(end, ...) — promote end (index 0)
            # Used for RoPE freq computation, position indices
            elif op_type == "aten::arange" and len(args) >= 1:
                result = _try_promote_scalar(args[0])
                if result:
                    args[0] = result
                    promoted += 1

            # aten::full / aten::zeros / aten::ones / aten::new_zeros / aten::new_ones
            # Shape args contain seq_len — promote matching elements in shape list
            elif op_type in ("aten::full", "aten::zeros", "aten::ones",
                             "aten::new_zeros", "aten::new_ones"):
                # args[0] is shape (list of ints/scalars) for full/zeros/ones
                # For new_zeros/new_ones: args[0] is tensor, args[1] is shape
                shape_idx = 1 if op_type.startswith("aten::new_") else 0
                if len(args) > shape_idx:
                    shape_arg = args[shape_idx]
                    if isinstance(shape_arg, (list, tuple)):
                        shape_list = list(shape_arg)
                        changed = False
                        for i, elem in enumerate(shape_list):
                            if isinstance(elem, dict):
                                result = _try_promote_scalar(elem)
                                if result:
                                    shape_list[i] = result
                                    promoted += 1
                                    changed = True
                            elif isinstance(elem, int):
                                result = _try_promote_raw_int(elem)
                                if result:
                                    shape_list[i] = result
                                    promoted += 1
                                    changed = True
                        if changed:
                            args[shape_idx] = shape_list

            # aten::expand(tensor, size) — promote seq_len in size list
            elif op_type == "aten::expand" and len(args) >= 2:
                size_arg = args[1]
                if isinstance(size_arg, (list, tuple)):
                    size_list = list(size_arg)
                    changed = False
                    for i, elem in enumerate(size_list):
                        if isinstance(elem, dict):
                            result = _try_promote_scalar(elem)
                            if result:
                                size_list[i] = result
                                promoted += 1
                                changed = True
                        elif isinstance(elem, int):
                            result = _try_promote_raw_int(elem)
                            if result:
                                size_list[i] = result
                                promoted += 1
                                changed = True
                    if changed:
                        args[1] = size_list

            # aten::view / aten::reshape / aten::_unsafe_view
            # Shape args may contain seq_len — promote matching elements
            elif op_type in ("aten::view", "aten::reshape", "aten::_unsafe_view") and len(args) >= 2:
                shape_arg = args[1]
                is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                shape_items = shape_arg.get("value", []) if is_wrapped else shape_arg
                if isinstance(shape_items, (list, tuple)):
                    shape_list = list(shape_items)
                    changed = False
                    for i, elem in enumerate(shape_list):
                        if isinstance(elem, dict):
                            result = _try_promote_scalar(elem)
                            if result:
                                shape_list[i] = result
                                promoted += 1
                                changed = True
                        elif isinstance(elem, int):
                            result = _try_promote_raw_int(elem)
                            if result:
                                shape_list[i] = result
                                promoted += 1
                                changed = True
                    if changed:
                        if is_wrapped:
                            args[1] = {"type": "list", "value": shape_list}
                        else:
                            args[1] = shape_list

    def _identify_seq_dependent_constants(self, tensors: Dict[str, Any]) -> None:
        """
        Identify constant tensors whose shape contains the trace-time seq_len.

        RoPE cos/sin embeddings are computed on CPU during tracing and captured
        as constant tensors with shape [1, trace_seq_len, head_dim]. At runtime,
        seq_len varies — these constants must be sliced to [1, runtime_seq_len, head_dim].

        This method tags such constants so they can be dynamically sliced at runtime
        after the symbolic shape resolver binds the actual seq_len value.
        """
        symbolic_context = self.dag.get("symbolic_context", {})
        symbols = symbolic_context.get("symbols", {})

        # Find seq_len symbols and their trace values
        seq_len_info: Dict[str, int] = {}
        for sym_id, sym_info in symbols.items():
            if sym_info.get("name") == "seq_len":
                trace_val = sym_info.get("trace_value")
                if trace_val is not None:
                    seq_len_info[sym_id] = trace_val

        if not seq_len_info:
            return

        # Use the first seq_len symbol (all share the same trace value)
        sym_id, trace_seq_len = next(iter(seq_len_info.items()))

        self._seq_dependent_constants = []

        for tid in self._weight_tensor_ids:
            tdata = tensors.get(tid, {})
            wname = tdata.get("weight_name", "")
            if not wname.startswith("constant_T_"):
                continue
            shape = tdata.get("shape", [])
            for axis, dim in enumerate(shape):
                if dim == trace_seq_len:
                    slot = self._tensor_id_to_slot.get(tid)
                    if slot is not None:
                        self._seq_dependent_constants.append((slot, axis, sym_id, trace_seq_len))
                    break

    def update_seq_dependent_constants(self) -> None:
        """
        Adapt seq-dependent constants to match the runtime seq_len.

        Called after bind_symbols() populates the shape resolver with actual
        seq_len values. bind_weights() has already placed the full-size constants
        in the arena — this method narrows or extends them on the seq_len axis.

        CRITICAL: Always narrow from the ORIGINAL full-size constant stored in
        _seq_constant_originals, NOT from the arena (which may hold a previously
        narrowed view). Without this, repeated calls progressively shrink the
        constant — e.g., RoPE cos/sin degrades to position 0 after decode starts.

        For prefill (seq_len <= trace_seq_len): constant[:, :N, :] gives correct
        RoPE values because positions 0..N-1 are a prefix of the trace-time table.

        For prefill (seq_len > trace_seq_len): recompute from inv_freq stored in
        the arena. RoPE cos/sin are deterministic functions of position and freq.
        """
        if not self._seq_dependent_constants or self._shape_resolver is None:
            return

        runtime_vals = self._shape_resolver.get_bound_symbols()
        arena = self._arena

        for slot, axis, sym_id, trace_val in self._seq_dependent_constants:
            runtime_seq_len = runtime_vals.get(sym_id)
            if runtime_seq_len is None or runtime_seq_len == trace_val:
                # Restore original if it was previously narrowed
                if slot in self._seq_constant_originals:
                    arena[slot] = self._seq_constant_originals[slot]
                continue

            # Save the original full-size constant on first encounter
            if slot not in self._seq_constant_originals:
                current = arena[slot]
                if current is not None:
                    self._seq_constant_originals[slot] = current

            # Always narrow from the ORIGINAL, never from a previously-narrowed view
            original = self._seq_constant_originals.get(slot)
            if original is None:
                continue

            if runtime_seq_len <= original.shape[axis]:
                # Slice: positions 0..N-1 are a prefix of the trace-time table
                arena[slot] = original.narrow(axis, 0, runtime_seq_len)
            else:
                # Extend: recompute RoPE cos/sin from inv_freq for positions 0..N-1
                inv_freq = self._find_inv_freq_in_arena()
                if inv_freq is not None:
                    extended = self._recompute_rope_constant(
                        original, inv_freq, runtime_seq_len, axis
                    )
                    arena[slot] = extended
                    # Update original reference with extended version
                    self._seq_constant_originals[slot] = extended

    def _find_inv_freq_in_arena(self) -> Optional[torch.Tensor]:
        """Find rotary_embed.inv_freq in the arena by tensor_id."""
        tid = "param::rotary_embed.inv_freq"
        slot = self._tensor_id_to_slot.get(tid)
        if slot is not None and self._arena is not None:
            return self._arena[slot]
        return None

    def _recompute_rope_constant(
        self,
        original: torch.Tensor,
        inv_freq: torch.Tensor,
        seq_len: int,
        axis: int,
    ) -> torch.Tensor:
        """
        Recompute RoPE cos or sin for extended positions.

        Uses inv_freq to compute freqs for positions 0..seq_len-1,
        then applies cos or sin based on the original constant's values.

        RoPE formula: freqs = outer(positions, inv_freq)
                      emb = cat(freqs, freqs, dim=-1)
                      cos_table = cos(emb), sin_table = sin(emb)
        """
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        # freqs: [seq_len, head_dim/2]
        freqs = torch.outer(positions, inv_freq)
        # emb: [seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Determine if this is cos or sin by checking position 0 values
        # cos(0) = 1.0, sin(0) = 0.0
        first_val = original.flatten()[0].item()
        if abs(first_val - 1.0) < 0.01:
            result = emb.cos()
        else:
            result = emb.sin()

        # Reshape to match original: [1, seq_len, head_dim]
        result = result.unsqueeze(0).to(original.dtype)
        return result

    def _categorize_and_assign_slots(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        graph_output_ids: Optional[set] = None,
    ) -> None:
        """
        Categorize tensors and assign slots in optimal order.

        Slot layout: [weights...][inputs...][intermediates...]
        This enables efficient clear_intermediates() by only clearing tail slots.

        Tensor categorization is based on tensor_id prefix (NeuroBrix convention):
        - param:: → weight (model parameter)
        - buffer:: → weight (model buffer)
        - input:: → runtime input
        - Everything else → intermediate (computed during execution)
        """
        if graph_output_ids is None:
            graph_output_ids = set()

        # First pass: categorize all tensors by ID prefix
        weights = []
        inputs = []
        intermediates = []

        for tensor_id, tensor_data in tensors.items():
            # Categorize by tensor_id prefix (NeuroBrix convention)
            if tensor_id.startswith("param::") or tensor_id.startswith("buffer::"):
                weights.append(tensor_id)
            elif tensor_id.startswith("input::"):
                inputs.append(tensor_id)
            else:
                intermediates.append(tensor_id)

            # Check for outputs: both is_output flag AND graph-declared outputs
            if tensor_data.get("is_output") or tensor_id in graph_output_ids:
                if tensor_id not in self._output_tensor_ids:
                    self._output_tensor_ids.append(tensor_id)

        # Check ops for additional output tensors (may not be marked in tensors dict)
        for _, op_data in ops_metadata.items():
            for out_id in op_data.get("output_tensor_ids", []):
                if out_id not in tensors and out_id not in self._tensor_id_to_slot:
                    intermediates.append(out_id)

        # Assign slots in order: weights, inputs, intermediates
        slot = 0

        for tid in weights:
            self._tensor_id_to_slot[tid] = slot
            self._slot_to_tensor_id[slot] = tid
            self._weight_tensor_ids.append(tid)
            slot += 1
        self._num_weights = len(weights)

        for tid in inputs:
            self._tensor_id_to_slot[tid] = slot
            self._slot_to_tensor_id[slot] = tid
            self._input_tensor_ids.append(tid)
            slot += 1
        self._num_inputs = len(inputs)

        for tid in intermediates:
            if tid not in self._tensor_id_to_slot:  # Avoid duplicates
                self._tensor_id_to_slot[tid] = slot
                self._slot_to_tensor_id[slot] = tid
                slot += 1
        self._num_intermediates = slot - self._num_weights - self._num_inputs

        self._next_slot = slot

    def _extract_input_slots_from_dag(self, op_data: Dict[str, Any]) -> List[int]:
        """
        Extract input slots from DAG op data for liveness analysis.

        This analyzes the raw DAG structure to find which tensor slots
        are read by this operation. Used for dead tensor analysis.

        Returns list of slot indices that this op reads from.
        """
        slots = []
        attrs = op_data.get("attributes", {})

        def extract_from_arg(arg: Any) -> None:
            """Recursively extract tensor slots from argument structures."""
            if not isinstance(arg, dict):
                return

            arg_type = arg.get("type")
            if arg_type == "tensor":
                tid = arg.get("tensor_id")
                if tid is None:
                    return
                slot = self._tensor_id_to_slot.get(tid)
                if slot is not None:
                    slots.append(slot)
            elif arg_type == "tensor_tuple":
                for tid in arg.get("tensor_ids", []):
                    slot = self._tensor_id_to_slot.get(tid)
                    if slot is not None:
                        slots.append(slot)
            elif arg_type == "list":
                for item in arg.get("value", []):
                    extract_from_arg(item)

        # Process args
        for arg in attrs.get("args", []):
            extract_from_arg(arg)

        # Process kwargs
        for arg in attrs.get("kwargs", {}).values():
            extract_from_arg(arg)

        return slots

    def _compute_liveness(
        self,
        execution_order: List[str],
        ops_metadata: Dict[str, Any]
    ) -> Dict[int, List[int]]:
        """
        Compute liveness analysis to find when each tensor slot becomes dead.

        DEAD TENSOR ANALYSIS - O(N) Algorithm:
        1. Scan all ops to find the last op that uses each slot
        2. Identify protected slots (weights, inputs, outputs) - never freed
        3. Build dead_at_op mapping: op_idx → list of slots to free AFTER op executes

        Returns:
            Dict[int, List[int]]: op_idx → list of slots to free after this op
        """
        # Step 1: Find last usage of each slot (O(N) scan)
        slot_last_use: Dict[int, int] = {}  # slot → last op_idx that reads it

        for op_idx, op_uid in enumerate(execution_order):
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue

            # Get all input slots this op reads
            input_slots = self._extract_input_slots_from_dag(op_data)
            for slot in input_slots:
                slot_last_use[slot] = op_idx  # Update to latest usage

        # Step 2: Define protected slots (never freed)
        # - Weight slots: needed across executions
        # - Input slots: managed externally
        # - Output slots: must be returned to caller
        protected_slots = set()

        # Weights are slots 0..num_weights-1
        for i in range(self._num_weights):
            protected_slots.add(i)

        # Inputs are slots num_weights..num_weights+num_inputs-1
        input_start = self._num_weights
        for i in range(self._num_inputs):
            protected_slots.add(input_start + i)

        # Outputs (marked in DAG)
        for tensor_id in self._output_tensor_ids:
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is not None:
                protected_slots.add(slot)

        # Persistent tensors (e.g., hidden states for LLM extraction)
        for tensor_id in self._persistent_tensor_ids:
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is not None:
                protected_slots.add(slot)

        # Step 3: Build dead_at_op mapping using defaultdict (O(1) append)
        dead_at_op: Dict[int, List[int]] = defaultdict(list)

        for slot, last_op_idx in slot_last_use.items():
            # Skip protected slots
            if slot in protected_slots:
                continue
            # This slot dies AFTER last_op_idx executes
            dead_at_op[last_op_idx].append(slot)

        return dict(dead_at_op)

    def _compile_op(
        self,
        op_uid: str,
        op_data: Dict[str, Any],
        tensors: Dict[str, Any],
        kill_slots: Tuple[int, ...] = ()
    ) -> CompiledOp:
        """
        Compile a single operation with closure-based resolvers.

        100% AUTONOMOUS: Uses CompiledOpResolver, no dependency on NativeATenDispatcher.

        Args:
            kill_slots: Tensor slots to free AFTER this op executes (from liveness analysis)
        """
        op_type = op_data.get("op_type", "")
        attrs = op_data.get("attributes", {})
        output_tensor_ids = op_data.get("output_tensor_ids", [])

        # Include shapes and dtypes in attrs for ops that need them (e.g., lift_fresh, _to_copy)
        # These are at op_data level, not in attributes - mirror what sequential mode does
        if "input_shapes" in op_data:
            attrs = dict(attrs)  # Don't mutate original
            attrs["input_shapes"] = op_data["input_shapes"]
        if "output_shapes" in op_data:
            if not isinstance(attrs, dict) or "input_shapes" not in attrs:
                attrs = dict(attrs)
            attrs["output_shapes"] = op_data["output_shapes"]
        # CRITICAL: Include output_dtypes for _to_copy ops (Sana 1K dtype fix)
        # Graph captures dtype conversions but attrs is None - extract from output_dtypes
        if "output_dtypes" in op_data:
            if not isinstance(attrs, dict):
                attrs = dict(attrs) if attrs else {}
            attrs["output_dtypes"] = op_data["output_dtypes"]
        if "input_dtypes" in op_data:
            if not isinstance(attrs, dict):
                attrs = dict(attrs) if attrs else {}
            attrs["input_dtypes"] = op_data["input_dtypes"]

        # Strip aten:: prefix
        op_name = op_type[6:] if op_type.startswith("aten::") else op_type

        # ================================================================
        # FUSED MoE OP — Custom compilation path
        # ================================================================
        if op_type == "custom::moe_fused":
            return self._compile_moe_fused_op(op_uid, op_data, kill_slots)

        # Check for registered interceptor (KV cache injection for LLM execution)
        if op_type in self._op_interceptors:
            # Use interceptor instead of native op
            func = self._op_interceptors[op_type]
        else:
            # Get function from autonomous op resolver (100% independent from sequential_dispatcher)
            func = self.op_resolver.get_op_func(op_name, attrs)

        # Allocate slots for output tensors not yet assigned
        output_slots = []
        for out_id in output_tensor_ids:
            if out_id not in self._tensor_id_to_slot:
                slot = self._next_slot
                self._tensor_id_to_slot[out_id] = slot
                self._slot_to_tensor_id[slot] = out_id
                self._next_slot += 1
                self._num_intermediates += 1
            output_slots.append(self._tensor_id_to_slot[out_id])

        # Pre-compile args and kwargs to typed objects
        compiled_args = self._compile_args(attrs.get("args", []), tensors)
        compiled_kwargs = self._compile_kwargs(attrs.get("kwargs", {}), tensors)

        # Generate closure resolvers (KEY INNOVATION)
        args_resolver = self._make_args_resolver(compiled_args)
        kwargs_resolver = self._make_kwargs_resolver(compiled_kwargs)

        # Extract input slots for device derivation and cross-device detection
        all_input_slots = self._extract_input_slots_from_dag(op_data)
        weight_slots = []
        for slot_idx in all_input_slots:
            tid = self._slot_to_tensor_id.get(slot_idx)
            if tid and (tid.startswith("param::") or tid.startswith("buffer::")):
                weight_slots.append(slot_idx)

        return CompiledOp(
            op_uid=op_uid,
            op_type=op_type,
            func=func,
            args_resolver=args_resolver,
            kwargs_resolver=kwargs_resolver,
            output_slots=tuple(output_slots),
            kill_slots=kill_slots,
            weight_input_slots=tuple(weight_slots),
            all_input_slots=tuple(all_input_slots),
        )

    # ========================================================================
    # FUSED MoE COMPILATION
    # ========================================================================

    def _compile_moe_fused_op(
        self,
        op_uid: str,
        op_data: Dict[str, Any],
        kill_slots: Tuple[int, ...],
    ) -> CompiledOp:
        """
        Compile a fused MoE dispatch op with custom arena-based weight access.

        The fused op replaces ~893 individual ops per MoE layer with a single
        function that performs dynamic routing + expert FFN + scatter-add.

        All parameters (num_experts, top_k, weight slots) are extracted from
        the DAG attributes set by moe_fusion.py — ZERO HARDCODE.
        """
        import torch.nn.functional as F

        attrs = op_data.get("attributes", {})
        output_tensor_ids = op_data.get("output_tensor_ids", [])

        # Extract parameters from fusion pass attributes
        gate_scores_tid = attrs["gate_scores_tid"]
        hidden_states_tid = attrs["hidden_states_tid"]
        gate_weight_ids = attrs["expert_gate_weight_ids"]
        up_weight_ids = attrs["expert_up_weight_ids"]
        down_weight_ids = attrs["expert_down_weight_ids"]
        top_k = attrs["top_k"]
        num_experts = attrs["num_experts"]
        norm_topk_prob = attrs.get("norm_topk_prob", True)

        # Resolve tensor IDs to arena slots (compile-time)
        gate_scores_slot = self._tensor_id_to_slot[gate_scores_tid]
        hidden_states_slot = self._tensor_id_to_slot[hidden_states_tid]

        # Resolve all expert weight slots (compile-time, zero-copy lists at runtime)
        gate_w_slots = []
        up_w_slots = []
        down_w_slots = []
        all_weight_slots = []

        for i in range(num_experts):
            gs = self._tensor_id_to_slot.get(gate_weight_ids[i])
            us = self._tensor_id_to_slot.get(up_weight_ids[i])
            ds = self._tensor_id_to_slot.get(down_weight_ids[i])
            if gs is None or us is None or ds is None:
                raise RuntimeError(
                    f"[MoE Fusion] Missing weight slot for expert {i} in {op_uid}. "
                    f"gate={gate_weight_ids[i]} up={up_weight_ids[i]} down={down_weight_ids[i]}"
                )
            gate_w_slots.append(gs)
            up_w_slots.append(us)
            down_w_slots.append(ds)
            all_weight_slots.extend([gs, us, ds])

        # Freeze slot lists for closure capture
        gate_w_slots = tuple(gate_w_slots)
        up_w_slots = tuple(up_w_slots)
        down_w_slots = tuple(down_w_slots)

        # Allocate output slot
        output_slots = []
        for out_id in output_tensor_ids:
            if out_id not in self._tensor_id_to_slot:
                slot = self._next_slot
                self._tensor_id_to_slot[out_id] = slot
                self._slot_to_tensor_id[slot] = out_id
                self._next_slot += 1
                self._num_intermediates += 1
            output_slots.append(self._tensor_id_to_slot[out_id])

        # Build the fused dispatch function (closure captures all slots)
        _top_k = top_k
        _num_experts = num_experts
        _norm_topk_prob = norm_topk_prob
        _gate_scores_slot = gate_scores_slot
        _hidden_states_slot = hidden_states_slot
        _gate_w_slots = gate_w_slots
        _up_w_slots = up_w_slots
        _down_w_slots = down_w_slots
        _cached_w_dtype = [None]  # Mutable container for closure — resolved once on first call

        def moe_fused_dispatch(arena):
            """
            Fused MoE dispatch: dynamic routing + expert FFN + scatter-add.

            Replaces ~893 ops with hardcoded slice boundaries.
            All routing computed dynamically from gate_scores.
            """
            gate_scores = arena[_gate_scores_slot]
            hidden_states = arena[_hidden_states_slot]

            if hidden_states is None:
                raise RuntimeError(
                    f"MoE fused: hidden_states is None (slot {_hidden_states_slot}). "
                    f"gate_scores={'None' if gate_scores is None else 'OK'}. "
                    f"Killed by liveness analysis before fused op."
                )

            # Handle 3D tensors [batch, seq, dim] → flatten to 2D [batch*seq, dim]
            orig_shape = hidden_states.shape
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            if gate_scores.dim() == 3:
                gate_scores = gate_scores.reshape(-1, gate_scores.size(-1))

            # DTYPE CONTRACT: resolve weight dtype once, cache for all subsequent calls
            w_dtype = _cached_w_dtype[0]
            if w_dtype is None:
                w_dtype = arena[_gate_w_slots[0]].dtype
                _cached_w_dtype[0] = w_dtype
            if hidden_states.dtype != w_dtype:
                hidden_states = hidden_states.to(w_dtype)

            # ROUTING IN FP32: MoE router precision is critical for expert selection.
            # vLLM PR #14027 documents that bf16/fp16 routing causes quality degradation
            # due to precision loss in softmax/topk/normalization. Gate scores MUST be
            # computed in fp32 regardless of weight dtype.
            gate_scores = gate_scores.float()

            # Ensure ALL routing tensors on hidden_states device
            _compute_dev = hidden_states.device
            if gate_scores.device != _compute_dev:
                gate_scores = gate_scores.to(_compute_dev)

            # Dynamic routing in fp32 (replaces hardcoded topk+sort+bincount+slice)
            scores, indices = gate_scores.topk(_top_k, dim=-1)
            if _norm_topk_prob:
                scores = scores / scores.sum(dim=-1, keepdim=True)

            flat_indices = indices.flatten()
            sorted_expert_ids, perm = flat_indices.sort()
            token_ids = perm // _top_k

            counts = torch.bincount(sorted_expert_ids, minlength=_num_experts)
            boundaries = torch.cumsum(counts, dim=0)

            # CRITICAL: Transfer boundaries to CPU in ONE sync (not 128 .item() calls)
            # Before: 128 × .item() = 128 GPU syncs per layer × 48 layers = 6,144 syncs/token
            # After: 1 .tolist() = 1 GPU sync per layer × 48 layers = 48 syncs/token (128x fewer)
            boundaries_cpu = boundaries.tolist()

            output = torch.zeros_like(hidden_states)
            start = 0

            for expert_id in range(_num_experts):
                end = boundaries_cpu[expert_id]
                if start == end:
                    start = end
                    continue

                expert_token_ids = token_ids[start:end]
                expert_input = hidden_states[expert_token_ids]

                # SwiGLU FFN — weights accessed by slot index (O(1), zero copy)
                gate_w = arena[_gate_w_slots[expert_id]]
                up_w = arena[_up_w_slots[expert_id]]
                down_w = arena[_down_w_slots[expert_id]]

                # Multi-device alignment: move ALL operands to hidden_states device
                _dev = hidden_states.device
                if gate_w.device != _dev:
                    gate_w = gate_w.to(_dev)
                if up_w.device != _dev:
                    up_w = up_w.to(_dev)
                if down_w.device != _dev:
                    down_w = down_w.to(_dev)
                if expert_input.device != _dev:
                    expert_input = expert_input.to(_dev)
                if expert_token_ids.device != _dev:
                    expert_token_ids = expert_token_ids.to(_dev)

                gate = F.silu(expert_input @ gate_w.t())
                up = expert_input @ up_w.t()
                expert_out = (gate * up) @ down_w.t()

                # Weighted scatter-add (scores computed in fp32, cast to w_dtype for accumulation)
                expert_scores = scores.flatten()[perm[start:end]].unsqueeze(-1).to(w_dtype)
                output.index_add_(0, expert_token_ids, expert_out * expert_scores)

                start = end

            # Restore original shape if input was 3D
            if len(orig_shape) == 3:
                output = output.reshape(orig_shape)

            return output

        # Custom args_resolver: passes the arena itself (the function reads slots directly)
        def args_resolver(arena):
            return [arena]

        def kwargs_resolver(_arena: TensorArena) -> Dict[str, Any]:
            return {}

        # The func wrapper unpacks arena from args
        def func_wrapper(arena: TensorArena) -> torch.Tensor:
            return moe_fused_dispatch(arena)

        # All input slots: gate_scores + hidden_states + all expert weights
        moe_all_input_slots = [_gate_scores_slot, _hidden_states_slot] + list(all_weight_slots)

        return CompiledOp(
            op_uid=op_uid,
            op_type="custom::moe_fused",
            func=func_wrapper,
            args_resolver=args_resolver,
            kwargs_resolver=kwargs_resolver,
            output_slots=tuple(output_slots),
            kill_slots=kill_slots,
            weight_input_slots=tuple(all_weight_slots),
            all_input_slots=tuple(moe_all_input_slots),
        )

    # ========================================================================
    # CLOSURE GENERATORS - The key to zero-overhead execution
    # ========================================================================

    def _make_args_resolver(self, compiled_args: Tuple[Any, ...]) -> Callable[[TensorArena], List[Any]]:
        """
        Generate a closure that resolves args WITHOUT isinstance() at runtime.

        At compile time, we check types ONCE and generate lambdas that
        directly access slots or return constant values.
        """
        resolvers = []

        for arg in compiled_args:
            if isinstance(arg, TensorSlot):
                # Capture slot index in closure
                s = arg.slot
                resolvers.append(lambda arena, s=s: arena[s])
            elif isinstance(arg, ScalarArg):
                # Capture value in closure
                v = arg.value
                resolvers.append(lambda _arena, v=v: v)
            elif isinstance(arg, DtypeArg):
                # Capture dtype in closure
                dt = arg.dtype
                resolvers.append(lambda _arena, dt=dt: dt)
            elif isinstance(arg, SymbolArg):
                # Dynamic symbol resolution
                sym_resolver = self._make_symbol_resolver(arg.symbol_id, arg.trace_value, arg.offset)
                resolvers.append(sym_resolver)
            elif isinstance(arg, ProductArg):
                # Dynamic product resolution
                prod_resolver = self._make_product_resolver(arg.factors, arg.trace_value)
                resolvers.append(prod_resolver)
            elif isinstance(arg, ListArg):
                # Recursively create resolver for list items
                item_resolver = self._make_list_resolver(arg.items)
                resolvers.append(item_resolver)
            else:
                # Unknown type - return as constant
                val = arg
                resolvers.append(lambda _arena, val=val: val)

        # Create final resolver that calls all item resolvers
        resolvers_tuple = tuple(resolvers)
        return lambda arena: [r(arena) for r in resolvers_tuple]

    def _make_kwargs_resolver(self, compiled_kwargs: Dict[str, Any]) -> Callable[[TensorArena], Dict[str, Any]]:
        """
        Generate a closure that resolves kwargs WITHOUT isinstance() at runtime.
        """
        if not compiled_kwargs:
            # Fast path: empty kwargs
            return lambda _arena: {}

        keys = tuple(compiled_kwargs.keys())
        resolvers = []

        for key in keys:
            arg = compiled_kwargs[key]
            if isinstance(arg, TensorSlot):
                s = arg.slot
                resolvers.append(lambda arena, s=s: arena[s])
            elif isinstance(arg, ScalarArg):
                v = arg.value
                resolvers.append(lambda _arena, v=v: v)
            elif isinstance(arg, DtypeArg):
                dt = arg.dtype
                resolvers.append(lambda _arena, dt=dt: dt)
            elif isinstance(arg, SymbolArg):
                # Dynamic symbol resolution
                sym_resolver = self._make_symbol_resolver(arg.symbol_id, arg.trace_value, arg.offset)
                resolvers.append(sym_resolver)
            elif isinstance(arg, ProductArg):
                # Dynamic product resolution
                prod_resolver = self._make_product_resolver(arg.factors, arg.trace_value)
                resolvers.append(prod_resolver)
            elif isinstance(arg, ListArg):
                item_resolver = self._make_list_resolver(arg.items)
                resolvers.append(item_resolver)
            else:
                val = arg
                resolvers.append(lambda _arena, val=val: val)

        resolvers_tuple = tuple(resolvers)
        return lambda arena: {k: r(arena) for k, r in zip(keys, resolvers_tuple)}

    def _make_list_resolver(self, items: Tuple[Any, ...]) -> Callable[[TensorArena], List[Any]]:
        """Generate resolver for list arguments."""
        item_resolvers = []

        for item in items:
            if isinstance(item, TensorSlot):
                s = item.slot
                item_resolvers.append(lambda arena, s=s: arena[s])
            elif isinstance(item, ScalarArg):
                v = item.value
                item_resolvers.append(lambda _arena, v=v: v)
            elif isinstance(item, DtypeArg):
                dt = item.dtype
                item_resolvers.append(lambda _arena, dt=dt: dt)
            elif isinstance(item, SymbolArg):
                # Dynamic symbol resolution
                sym_resolver = self._make_symbol_resolver(item.symbol_id, item.trace_value, item.offset)
                item_resolvers.append(sym_resolver)
            elif isinstance(item, ProductArg):
                # Dynamic product resolution
                prod_resolver = self._make_product_resolver(item.factors, item.trace_value)
                item_resolvers.append(prod_resolver)
            elif isinstance(item, ListArg):
                # Nested list - recursive
                nested_resolver = self._make_list_resolver(item.items)
                item_resolvers.append(nested_resolver)
            else:
                val = item
                item_resolvers.append(lambda _arena, val=val: val)

        resolvers_tuple = tuple(item_resolvers)
        return lambda arena: [r(arena) for r in resolvers_tuple]

    def _make_symbol_resolver(self, symbol_id: str, trace_value: int, offset: int = 0) -> Callable[[TensorArena], int]:
        """
        Generate closure for dynamic symbol resolution.

        At compile time, we capture the symbol_id, trace_value, and optional offset.
        At runtime, the closure queries the shape resolver for the actual value.

        Args:
            symbol_id: Symbol identifier (e.g., "s0")
            trace_value: Fallback value for graphs without resolver
            offset: Constant added to resolved value (e.g., 1 for seq_len + 1)

        Returns:
            Closure that resolves symbol at runtime
        """
        def resolve_symbol(_arena: TensorArena) -> int:
            if self._shape_resolver is not None:
                runtime_vals = self._shape_resolver.get_bound_symbols()
                if symbol_id in runtime_vals:
                    return runtime_vals[symbol_id] + offset
            return trace_value
        return resolve_symbol

    def _make_product_resolver(self, factors: Tuple[Any, ...], trace_value: int) -> Callable[[TensorArena], int]:
        """
        Generate closure for dynamic product resolution.

        Handles expressions like s0 * s1 * 256 by multiplying all factors.

        Args:
            factors: Tuple of factor references (symbol_ids or ints)
            trace_value: Fallback value if symbols cannot be resolved

        Returns:
            Closure that computes product at runtime
        """
        def resolve_product(_arena: TensorArena) -> int:
            if self._shape_resolver is not None:
                runtime_vals = self._shape_resolver.get_bound_symbols()
                result = 1
                for f in factors:
                    if isinstance(f, str) and f in runtime_vals:
                        result *= runtime_vals[f]
                    elif isinstance(f, int):
                        result *= f
                    else:
                        # Symbol not bound - fallback to trace value
                        return trace_value
                return result
            return trace_value  # Fallback for graphs without resolver
        return resolve_product

    # ========================================================================
    # ARG COMPILATION (compile time only)
    # ========================================================================

    def _compile_args(self, args: List[Any], tensors: Dict[str, Any]) -> Tuple[Any, ...]:
        """Pre-compile positional arguments to typed objects."""
        return tuple(self._compile_arg(arg, tensors) for arg in args)

    def _compile_kwargs(self, kwargs: Dict[str, Any], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compile keyword arguments to typed objects."""
        return {k: self._compile_arg(v, tensors) for k, v in kwargs.items()}

    def _compile_arg(self, arg: Any, tensors: Dict[str, Any]) -> Any:
        """Pre-compile a single argument to a typed object."""
        if arg is None:
            return ScalarArg(None)

        if isinstance(arg, dict):
            arg_type = arg.get("type")

            # Handle tensor references
            if arg_type in ("tensor", "tensor_ref"):
                tensor_id = arg.get("tensor_id")
                if tensor_id in self._tensor_id_to_slot:
                    return TensorSlot(self._tensor_id_to_slot[tensor_id])
                return arg

            if arg_type == "tensor_tuple":
                # Bundle of tensors for ops like cat, stack, etc.
                tensor_ids = arg.get("tensor_ids", [])
                slots = []
                for tid in tensor_ids:
                    if tid in self._tensor_id_to_slot:
                        slots.append(TensorSlot(self._tensor_id_to_slot[tid]))
                    else:
                        # Unknown tensor - shouldn't happen
                        slots.append(tid)
                return ListArg(tuple(slots))

            if arg_type == "dtype":
                dtype_str = arg.get("value", "float32")
                return DtypeArg(self._parse_dtype(dtype_str))

            if arg_type == "device":
                return ScalarArg(self.device)

            if arg_type == "list":
                items = arg.get("value", [])
                compiled_items = tuple(self._compile_arg(item, tensors) for item in items)
                return ListArg(compiled_items)

            if arg_type == "scalar":
                return ScalarArg(arg.get("value"))

            if arg_type == "symbol":
                # Dynamic resolution: return SymbolArg for runtime resolution
                symbol_id = arg.get("symbol_id") or arg.get("id") or arg.get("name")
                if symbol_id is None:
                    raise ValueError(f"ZERO FALLBACK: symbol arg missing 'symbol_id'/'id'/'name': {arg}")
                trace_value = arg.get("trace_value", 0)
                offset = arg.get("offset", 0)
                return SymbolArg(symbol_id=symbol_id, trace_value=trace_value, offset=offset)

            if arg_type == "product":
                # Dynamic resolution: return ProductArg for runtime resolution
                # Format: {'type': 'product', 'factors': ['s1', 's2'], 'trace_value': 16384}
                factors_raw = arg.get("factors", [])
                trace_value = arg.get("trace_value", 0)
                compiled_factors = []
                for f in factors_raw:
                    if isinstance(f, dict) and f.get("type") == "symbol":
                        # Extract symbol id from nested symbol
                        compiled_factors.append(f.get("symbol_id") or f.get("id") or f.get("name"))
                    elif isinstance(f, dict):
                        # Concrete value wrapped in dict
                        compiled_factors.append(f.get("value", f.get("trace_value", 0)))
                    elif isinstance(f, str):
                        # Direct symbol reference (e.g., "s0")
                        compiled_factors.append(f)
                    else:
                        # Concrete integer
                        compiled_factors.append(f)
                return ProductArg(factors=tuple(compiled_factors), trace_value=trace_value)

            if arg_type in ("int", "float", "bool", "none"):
                return ScalarArg(arg.get("value"))

            if arg_type == "memory_format":
                return ScalarArg(self._parse_memory_format(arg.get("value")))

            if arg_type == "layout":
                return ScalarArg(self._parse_layout(arg.get("value")))

            if arg_type == "unknown":
                # Unknown type - try to parse value content
                value = arg.get("value")
                if isinstance(value, str):
                    # Memory format strings
                    if value in ("torch.contiguous_format", "torch.channels_last",
                                 "torch.channels_last_3d", "torch.preserve_format"):
                        return ScalarArg(self._parse_memory_format(value))
                    # Layout strings
                    if value in ("torch.strided", "torch.sparse_coo"):
                        return ScalarArg(self._parse_layout(value))
                    # Dtype strings
                    if value.startswith("torch."):
                        parsed_dtype = self._parse_dtype(value.replace("torch.", ""))
                        if parsed_dtype is not None:
                            return DtypeArg(parsed_dtype)
                # Fall through to return as scalar
                return ScalarArg(value)

            return arg

        if isinstance(arg, (int, float, bool)):
            return ScalarArg(arg)

        if isinstance(arg, str):
            if arg in self._tensor_id_to_slot:
                return TensorSlot(self._tensor_id_to_slot[arg])
            return ScalarArg(arg)

        if isinstance(arg, (list, tuple)):
            compiled_items = tuple(self._compile_arg(item, tensors) for item in arg)
            return ListArg(compiled_items)

        return arg

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype with Prism remap.

        Delegates to neurobrix.core.dtype.config.parse_dtype (single source of truth).
        Handles "torch." prefix and bf16↔fp16 remap automatically.
        """
        return _cfg_parse_dtype(dtype_str, compute_dtype=self.dtype)

    def _parse_memory_format(self, fmt_str: Optional[str]) -> Any:
        """Parse memory format string to torch memory format."""
        if fmt_str is None:
            return torch.contiguous_format
        fmt_map = {
            "torch.contiguous_format": torch.contiguous_format,
            "torch.channels_last": torch.channels_last,
            "torch.channels_last_3d": torch.channels_last_3d,
            "torch.preserve_format": torch.preserve_format,
        }
        return fmt_map.get(fmt_str, torch.contiguous_format)

    def _parse_layout(self, layout_str: Optional[str]) -> Any:
        """Parse layout string to torch layout."""
        if layout_str is None:
            return torch.strided
        layout_map = {
            "torch.strided": torch.strided,
            "torch.sparse_coo": torch.sparse_coo,
        }
        return layout_map.get(layout_str, torch.strided)

    # ========================================================================
    # RUNTIME METHODS
    # ========================================================================

    def bind_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Bind weight tensors to arena slots, including constant tensors."""
        assert self._arena is not None, "compile() must be called before bind_weights()"
        for tensor_id, tensor in weights.items():
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is not None:
                self._arena[slot] = tensor

        # Pre-populate constant weight slots not provided by weight loader.
        # Only allocate for genuinely empty/constant tensors (shape [0] or constant_*).
        # Do NOT allocate for regular weight slots that weren't provided — zero3
        # block-by-block execution provides partial weight sets intentionally.
        tensors_meta = self.dag.get("tensors", {})
        for tensor_id in self._weight_tensor_ids:
            slot = self._tensor_id_to_slot[tensor_id]
            if self._arena[slot] is not None:
                continue
            meta = tensors_meta.get(tensor_id, {})
            shape = meta.get("shape", [0])
            # Only allocate for empty-state constants (e.g., KV cache init tensors)
            # Regular weights that weren't provided stay None — ops that need them
            # will be skipped or fail explicitly (ZERO FALLBACK)
            if shape == [0] or "constant_" in tensor_id:
                dtype_str = meta.get("dtype", "float32")
                t_dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
                self._arena[slot] = torch.empty(shape, dtype=t_dtype, device=self.device)

    def compute_op_devices(self) -> None:
        """
        Derive per-op device from weight tensor placement in the arena.

        Called AFTER bind_weights() to inspect actual tensor devices.
        For FGP components, weights span multiple GPUs — this method sets
        each op's device based on its weight inputs, enabling cross-device
        activation transfer at block boundaries.

        OPTIMIZATION: Pre-computes needs_transfer flag per op. Only ops at
        GPU boundaries (where input device != op device) need the expensive
        device alignment path. For Qwen3-30B (115K ops, 4 GPUs), this reduces
        device checks from 115K to ~100 per decode step.

        Single-device components: all weights on same device → _is_multi_device=False → zero overhead.
        """
        assert self._arena is not None, "compile() must be called before compute_op_devices()"
        arena = self._arena
        devices_seen = set()

        # Phase 1: Assign device to weighted ops from weight placement
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, 'device'):
                    op.device = tensor.device
                    devices_seen.add(str(tensor.device))
                    break

        # Multi-device if weights span >1 device, OR if any weight is on a
        # different device than the executor (e.g., weights on CPU, compute on CUDA)
        if self.device is not None:
            devices_seen.add(str(self.device))
        self._is_multi_device = len(devices_seen) > 1

        if not self._is_multi_device:
            return

        # Phase 2: Propagate device through the graph and mark boundary ops.
        # Build slot→device map: track which device each output slot lives on.
        # An op needs_transfer only when its inputs come from a different device.
        slot_device: dict = {}  # slot_idx → torch.device

        # Seed slot_device from weight/buffer tensors already in arena
        for op in self._ops:
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, 'device'):
                    slot_device[ws] = tensor.device

        # Also seed from input tensors (bound later, but we know input slots)
        # Input slots will be set at runtime — skip for now, handle in phase 3

        # Phase 3: Forward pass — propagate device and detect boundaries
        for op in self._ops:
            # Determine this op's target device
            target = op.device  # From weights (phase 1)

            if target is None:
                # Weightless op: inherits device from its input data
                # Check weight_input_slots first, then scan all input slots
                # from the args resolver is impractical, so use slot_device
                # from predecessor ops via output_slots
                continue  # No device = no transfer needed

            # Check if any input slot comes from a different device
            # We only need to check weight_input_slots predecessors
            # The actual boundary detection happens for ops WITH weights
            # whose predecessors produced output on a different GPU
            for ws in op.weight_input_slots:
                if ws in slot_device and slot_device[ws] != target:
                    op.needs_transfer = True
                    break

            # Record this op's output device for downstream ops
            for s in op.output_slots:
                slot_device[s] = target

        # Phase 4: Comprehensive cross-device detection using all_input_slots.
        # Track which device each slot's tensor lives on, then mark ANY op
        # (weighted or weightless) that has inputs from multiple devices.
        # This catches residual connections that cross device boundaries.
        current_activation_device = None
        for op in self._ops:
            if op.device is not None:
                if current_activation_device is None:
                    op.needs_transfer = True
                elif op.device != current_activation_device:
                    op.needs_transfer = True
                current_activation_device = op.device
            else:
                # Weightless op: inherit current activation device
                op.device = current_activation_device

            # Check ALL input slots for cross-device inputs (catches residuals)
            if op.all_input_slots and current_activation_device is not None:
                for s in op.all_input_slots:
                    src_dev = slot_device.get(s)
                    if src_dev is not None and src_dev != current_activation_device:
                        op.needs_transfer = True
                        break

            # Record output device for downstream ops
            out_dev = op.device or current_activation_device
            if out_dev is not None:
                for s in op.output_slots:
                    slot_device[s] = out_dev

    def bind_inputs(self, inputs: Dict[str, torch.Tensor]) -> None:
        """Bind input tensors to arena slots."""
        assert self._arena is not None, "compile() must be called before bind_inputs()"
        for tensor_id, tensor in inputs.items():
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is not None:
                self._arena[slot] = tensor

    def bind_tensor(self, tensor_id: str, tensor: torch.Tensor) -> None:
        """Bind a single tensor by ID."""
        assert self._arena is not None, "compile() must be called before bind_tensor()"
        slot = self._tensor_id_to_slot.get(tensor_id)
        if slot is not None:
            self._arena[slot] = tensor

    def bind_symbols(self, resolver) -> None:
        """
        Bind the symbolic shape resolver for runtime symbol resolution.

        This must be called AFTER bind_inputs() but BEFORE run() when
        the graph uses symbolic dimensions (V3+ format).

        The resolver should have been populated via bind_from_inputs()
        with actual input tensor shapes.

        Args:
            resolver: SymbolicShapeResolver with bound _runtime_values
        """
        self._shape_resolver = resolver

    def peek_tensor(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Peek at a tensor in the arena by ID. Returns None if not found."""
        slot = self._tensor_id_to_slot.get(tensor_id)
        if slot is None or self._arena is None:
            return None
        return self._arena[slot]

    def run(self, debug: bool = False) -> None:
        """
        Execute the compiled sequence.

        THIS IS THE HOT LOOP - ZERO OVERHEAD:
        - No isinstance() checks
        - No dict lookups
        - No string operations
        - Just: resolve args, call func, store result

        Args:
            debug: If True, enable verbose error handling (slower)
        """
        arena = self._arena

        # Use inference_mode to disable autograd and reduce memory usage
        with torch.inference_mode():
            self._run_inner(arena, debug)

    def _run_inner(self, arena, debug: bool = False) -> None:
        """Inner loop extracted for inference_mode wrapping."""
        trace_nan = _TRACE_NAN
        nan_guard = _NAN_GUARD
        nan_guard_verbose = _NAN_GUARD_VERBOSE
        trace_zeros = _TRACE_ZEROS

        if self._is_multi_device:
            self._run_inner_multi_device(arena, debug)
            return

        # NaN-guard counters for summary (only if verbose)
        nan_guard_triggers = [] if nan_guard_verbose else None

        for op_idx, op in enumerate(self._ops):
            # Resolve args via pre-compiled closures (no isinstance!)
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            try:
                # Call function directly
                result = op.func(*args, **kwargs)
            except Exception as e:
                # ============================================================
                # NOP PROPAGATION for dynamic MoE routing
                # When an expert is deactivated, unbind produces fewer outputs
                # → downstream ops receive None → crash. Detect and propagate.
                # CRITICAL: Only when args[0] is None (primary operand = dead path).
                # None in non-primary args (attn_bias=None) is legitimate.
                # ============================================================
                if args and args[0] is None:
                    if op.op_type in _ACCUMULATOR_OPS:
                        result = args[0]
                    else:
                        for s in op.output_slots:
                            arena[s] = None
                        for kill_slot in op.kill_slots:
                            arena[kill_slot] = None
                        continue
                elif _has_none_arg(args):
                    # None inside list args (e.g. index(t, [None])) — MoE-related
                    has_none_in_list = any(
                        isinstance(a, (list, tuple)) and any(item is None for item in a)
                        for a in args
                    )
                    if has_none_in_list:
                        if op.op_type in _ACCUMULATOR_OPS and args[0] is not None:
                            result = args[0]
                        else:
                            for s in op.output_slots:
                                arena[s] = None
                            for kill_slot in op.kill_slots:
                                arena[kill_slot] = None
                            continue
                    else:
                        raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e
                else:
                    raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e

            # ================================================================
            # NaN/Inf-GUARD (OFF by default — engine.py inf-fix handles overflow)
            # Enable with NBX_NAN_GUARD=1 for debugging (2.5x slower)
            # Use NBX_NAN_GUARD_VERBOSE=1 for detailed diagnostics
            # ================================================================
            if nan_guard and isinstance(result, torch.Tensor) and result.is_floating_point():
                has_nan = torch.isnan(result).any().item()
                has_inf = torch.isinf(result).any().item()
                if has_nan or has_inf:
                    # Count NaN/Inf before replacement
                    nan_count = torch.isnan(result).sum().item()
                    inf_count = torch.isinf(result).sum().item()
                    out_shape = list(result.shape)
                    out_dtype = str(result.dtype)

                    # Analyze inputs to determine if this op CREATED or PROPAGATED NaN
                    def analyze_input(x: Any, depth: int = 0) -> Optional[Union[Dict[str, Any], List[Any]]]:
                        if isinstance(x, torch.Tensor) and x.is_floating_point():
                            has_nan = torch.isnan(x).any().item()
                            has_inf = torch.isinf(x).any().item()
                            return {
                                'shape': list(x.shape),
                                'dtype': str(x.dtype),
                                'min': float(x.min().item()) if x.numel() > 0 else 0,
                                'max': float(x.max().item()) if x.numel() > 0 else 0,
                                'has_nan': has_nan,
                                'has_inf': has_inf,
                            }
                        elif isinstance(x, (list, tuple)) and depth < 1:
                            return [analyze_input(item, depth+1) for item in x[:3]]
                        return None

                    input_infos = [analyze_input(a) for a in args]
                    # Check if inputs had NaN or Inf (both can propagate to NaN)
                    inputs_had_bad = any(
                        (info and (info.get('has_nan') or info.get('has_inf'))) if isinstance(info, dict)
                        else any(i and (i.get('has_nan') or i.get('has_inf')) for i in info if isinstance(i, dict))
                        if isinstance(info, list) else False
                        for info in input_infos
                    )

                    is_creator = not inputs_had_bad

                    # Replace NaN→0 and Inf→max (preserves sign)
                    result = torch.nan_to_num(result, nan=0.0, posinf=65504.0, neginf=-65504.0)

                    # ALWAYS log when verbose mode is on
                    if nan_guard_verbose:
                        marker = "★ CREATOR" if is_creator else "propagator"
                        issue = []
                        if nan_count > 0:
                            issue.append(f"NaN={nan_count}")
                        if inf_count > 0:
                            issue.append(f"Inf={inf_count}")
                        print(f"\n[NaN/Inf-GUARD] Op {op_idx}/{len(self._ops)}: {op.op_type} [{marker}]")
                        print(f"  UID: {op.op_uid}")
                        print(f"  Output: shape={out_shape}, dtype={out_dtype}, {', '.join(issue)}")
                        for i, info in enumerate(input_infos):
                            if info is None:
                                continue
                            if isinstance(info, dict):
                                print(f"  Input[{i}]: shape={info['shape']}, dtype={info['dtype']}, "
                                      f"range=[{info['min']:.4g}, {info['max']:.4g}], "
                                      f"nan={info['has_nan']}, inf={info['has_inf']}")
                            elif isinstance(info, list):
                                print(f"  Input[{i}]: list of {len(info)} tensors")
                                for j, sub in enumerate(info[:2]):
                                    if isinstance(sub, dict):
                                        print(f"    [{j}]: shape={sub['shape']}, range=[{sub['min']:.4g}, {sub['max']:.4g}], "
                                              f"nan={sub['has_nan']}, inf={sub['has_inf']}")

                    # Track for summary
                    if nan_guard_triggers is not None:
                        nan_guard_triggers.append({
                            'op_idx': op_idx,
                            'op_type': op.op_type,
                            'op_uid': op.op_uid,
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                            'is_creator': is_creator,
                        })

            # Store output(s)
            slots = op.output_slots
            if len(slots) == 1:
                # ================================================================
                # NaN-GUARD for single tensor output (handled above)
                # ================================================================
                arena[slots[0]] = result
            elif len(slots) > 1:
                # ================================================================
                # NaN-GUARD for TUPLE outputs (split, chunk, attention, etc.)
                # Dynamic output: result may have fewer elements than slots
                # (e.g., MoE unbind with variable expert activation counts)
                # ================================================================
                result_len = len(result) if isinstance(result, (tuple, list)) else 0
                for i, s in enumerate(slots):
                    if i >= result_len:
                        arena[s] = None  # Dynamic: fewer outputs than trace-time
                        continue
                    item = result[i]
                    if nan_guard and isinstance(item, torch.Tensor) and item.is_floating_point():
                        if torch.isnan(item).any().item():
                            nan_count = torch.isnan(item).sum().item()

                            # Check if this op CREATED the NaN
                            inputs_had_nan = any(
                                isinstance(a, torch.Tensor) and a.is_floating_point() and torch.isnan(a).any().item()
                                for a in args
                            )
                            is_creator = not inputs_had_nan

                            # Replace NaN
                            item = torch.nan_to_num(item, nan=0.0)

                            if nan_guard_verbose:
                                marker = "★ CREATOR" if is_creator else "propagator"
                                print(f"\n[NaN-GUARD TUPLE] Op {op_idx}/{len(self._ops)}: {op.op_type} output[{i}] [{marker}]")
                                print(f"  UID: {op.op_uid}")
                                print(f"  Output[{i}]: shape={list(item.shape)}, dtype={item.dtype}, NaN={nan_count}")
                                # Show input info
                                for j, a in enumerate(args):
                                    if isinstance(a, torch.Tensor) and a.is_floating_point():
                                        print(f"  Input[{j}]: shape={list(a.shape)}, "
                                              f"range=[{a.min().item():.4g}, {a.max().item():.4g}], "
                                              f"nan={torch.isnan(a).any().item()}, inf={torch.isinf(a).any().item()}")

                            if nan_guard_triggers is not None:
                                nan_guard_triggers.append({
                                    'op_idx': op_idx,
                                    'op_type': f"{op.op_type}[{i}]",
                                    'op_uid': op.op_uid,
                                    'nan_count': nan_count,
                                    'is_creator': is_creator,
                                })

                    arena[s] = item

            # NaN/Inf TRACING: Find ALL CREATORS of NaN or Inf (NBX_TRACE_NAN=1)
            # This is diagnostic-only, doesn't modify values
            if trace_nan and isinstance(result, torch.Tensor) and result.is_floating_point():
                has_nan = torch.isnan(result).any()
                has_inf = torch.isinf(result).any()

                if has_nan or has_inf:
                    # Check if ANY input (including in lists) has NaN or Inf
                    def has_bad_recursive(x):
                        if isinstance(x, torch.Tensor) and x.is_floating_point():
                            return (torch.isnan(x).any().item() or torch.isinf(x).any().item())
                        if isinstance(x, (list, tuple)):
                            return any(has_bad_recursive(item) for item in x)
                        return False

                    inputs_have_bad = any(has_bad_recursive(a) for a in args)

                    if not inputs_have_bad:
                        # This op CREATED NaN/Inf from clean inputs!
                        issue = "NaN" if has_nan else "Inf"
                        print(f"\n[{issue} SOURCE] Op {op_idx}/{len(self._ops)}: {op.op_type}")
                        print(f"  UID: {op.op_uid}")
                        print(f"  Output: shape={list(result.shape)}, dtype={result.dtype}")
                        if has_nan:
                            nan_count = torch.isnan(result).sum().item()
                            print(f"  NaN: {nan_count}/{result.numel()} ({100*nan_count/result.numel():.1f}%)")
                        if has_inf:
                            inf_count = torch.isinf(result).sum().item()
                            print(f"  Inf: {inf_count}/{result.numel()} ({100*inf_count/result.numel():.1f}%)")
                        # Show input info
                        def describe_arg(a, depth=0):
                            if isinstance(a, torch.Tensor):
                                if a.is_floating_point():
                                    a_inf = torch.isinf(a).any().item()
                                    return f"Tensor{list(a.shape)} {a.dtype}, inf={a_inf}, range=[{a.min().item():.4g}, {a.max().item():.4g}]"
                                return f"Tensor{list(a.shape)} {a.dtype}"
                            if isinstance(a, (list, tuple)) and depth < 2:
                                inner = [describe_arg(x, depth+1) for x in a[:3]]
                                if len(a) > 3:
                                    inner.append(f"...+{len(a)-3} more")
                                return f"[{', '.join(inner)}]"
                            return f"{type(a).__name__}"
                        for i, arg in enumerate(args):
                            print(f"  Input[{i}]: {describe_arg(arg)}")

            # ZERO-TRACE: Find first op that produces all-zero from non-zero input
            # Enable with NBX_TRACE_ZEROS=1 (one-time diagnostic, expensive)
            if trace_zeros and isinstance(result, torch.Tensor) and result.is_floating_point():
                if result.numel() > 0 and (result == 0).all().item():
                    # Output is zero — check if any tensor input was non-zero
                    has_nonzero_input = False
                    for a in args:
                        if isinstance(a, torch.Tensor) and a.is_floating_point() and a.numel() > 0:
                            if not (a == 0).all().item():
                                has_nonzero_input = True
                                break
                    if has_nonzero_input:
                        print(f"\n[ZERO-SOURCE] Op {op_idx}/{len(self._ops)}: {op.op_type} ({op.op_uid})")
                        print(f"  Output: shape={list(result.shape)}, dtype={result.dtype} → ALL ZERO")
                        for i, a in enumerate(args):
                            if isinstance(a, torch.Tensor):
                                print(f"  Input[{i}]: shape={list(a.shape)}, dtype={a.dtype}, "
                                      f"mean={a.float().mean().item():.6f}, range=[{a.min().item():.6g}, {a.max().item():.6g}]")
                            else:
                                print(f"  Input[{i}]: {type(a).__name__} = {a}")
                        trace_zeros = False  # Only report first occurrence
                # Also check for ops that produce non-finite values (may cause downstream zeros)
                elif result.numel() > 0 and (torch.isinf(result).any().item() or torch.isnan(result).any().item()):
                    nan_ct = torch.isnan(result).sum().item()
                    inf_ct = torch.isinf(result).sum().item()
                    if nan_ct > 0 or inf_ct > result.numel() * 0.5:
                        print(f"[ZERO-TRACE-WARN] Op {op_idx}: {op.op_type} ({op.op_uid}) "
                              f"output NaN={nan_ct}, Inf={inf_ct}, numel={result.numel()}")

            # FREE DEAD TENSORS - Critical for memory management
            # These slots are no longer needed after this op
            # Setting to None allows Python GC to release GPU memory
            for kill_slot in op.kill_slots:
                arena[kill_slot] = None

        # NaN-guard summary at end of execution (verbose mode only)
        if nan_guard_triggers:
            from collections import Counter
            creators = [t for t in nan_guard_triggers if t.get('is_creator')]
            propagators = [t for t in nan_guard_triggers if not t.get('is_creator')]

            print(f"\n{'='*60}")
            print(f"[NaN-GUARD SUMMARY]")
            print(f"  Total ops with NaN: {len(nan_guard_triggers)}")
            print(f"  ★ CREATORS (from clean inputs): {len(creators)}")
            print(f"  Propagators: {len(propagators)}")

            if creators:
                creator_types = Counter(t['op_type'] for t in creators)
                print(f"\n  CREATOR OPS (these are the culprits!):")
                for c in creators[:10]:
                    print(f"    Op {c['op_idx']}: {c['op_type']} ({c['op_uid']}) → {c.get('nan_count', '?')} NaN")
                print(f"  Creator op types: {dict(creator_types)}")

            total_nans = sum(t.get('nan_count', 0) for t in nan_guard_triggers)
            print(f"  Total NaN values replaced: {total_nans}")
            print(f"{'='*60}")

    def _run_inner_multi_device(self, arena, _debug: bool = False) -> None:  # noqa: ARG002
        """
        FGP multi-device hot loop with cross-device activation transfer.

        OPTIMIZATION: Only ops marked needs_transfer=True at compile time
        go through the expensive device alignment path. For Qwen3-30B with
        115K ops across 4 GPUs, only ~100 ops at block boundaries need
        transfers — the other 99.9% use the fast single-device path.

        Cross-device .to() queues the copy on the source stream, but the
        target stream doesn't implicitly wait. We use CUDA events for
        fine-grained sync: record on source, wait on target.
        """
        import torch
        _current_device_idx = torch.cuda.current_device()

        for op in self._ops:
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            # NOP propagation for deactivated MoE expert paths
            if args and args[0] is None:
                for s in op.output_slots:
                    arena[s] = None
                for kill_slot in op.kill_slots:
                    arena[kill_slot] = None
                continue

            # ── FAST PATH: No device transfer needed (99%+ of ops) ──
            if not op.needs_transfer:
                # Still need to set CUDA device context for ops that allocate
                if op.device is not None and op.device.type == "cuda" and op.device.index != _current_device_idx:
                    torch.cuda.set_device(op.device)
                    _current_device_idx = op.device.index

                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    if args and args[0] is None:
                        if op.op_type in _ACCUMULATOR_OPS:
                            result = args[0]
                        else:
                            for s in op.output_slots:
                                arena[s] = None
                            for kill_slot in op.kill_slots:
                                arena[kill_slot] = None
                            continue
                    elif _has_none_arg(args):
                        has_none_in_list = any(
                            isinstance(a, (list, tuple)) and any(item is None for item in a)
                            for a in args
                        )
                        if has_none_in_list:
                            if op.op_type in _ACCUMULATOR_OPS and args and args[0] is not None:
                                result = args[0]
                            else:
                                for s in op.output_slots:
                                    arena[s] = None
                                for kill_slot in op.kill_slots:
                                    arena[kill_slot] = None
                                continue
                        else:
                            raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e
                    else:
                        raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e
            else:
                # ── SLOW PATH: GPU boundary — device alignment needed ──
                # Rule: CUDA always wins over CPU — compute happens on GPU
                target = op.device
                if target is not None and target.type == "cpu":
                    # Weight on CPU (zero3 sub-component) — find CUDA arg for compute
                    for a in args:
                        if isinstance(a, torch.Tensor) and a.device.type == "cuda":
                            target = a.device
                            break
                elif target is None:
                    for a in args:
                        if isinstance(a, torch.Tensor):
                            target = a.device
                            break
                        elif isinstance(a, (list, tuple)):
                            for item in a:
                                if isinstance(item, torch.Tensor):
                                    target = item.device
                                    break
                            if target is not None:
                                break

                if target is not None:
                    new_args = []
                    for a in args:
                        if isinstance(a, torch.Tensor) and a.device != target:
                            new_args.append(a.to(target))
                        elif isinstance(a, (list, tuple)):
                            moved = []
                            any_moved = False
                            for item in a:
                                if isinstance(item, torch.Tensor) and item.device != target:
                                    moved.append(item.to(target))
                                    any_moved = True
                                else:
                                    moved.append(item)
                            new_args.append(type(a)(moved) if any_moved else a)
                        else:
                            new_args.append(a)
                    args = new_args
                    if kwargs:
                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor) and v.device != target:
                                new_kwargs[k] = v.to(target)
                            else:
                                new_kwargs[k] = v
                        kwargs = new_kwargs

                if target is not None and target.type == "cuda" and target.index != _current_device_idx:
                    torch.cuda.set_device(target)
                    _current_device_idx = target.index

                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    if args and args[0] is None:
                        if op.op_type in _ACCUMULATOR_OPS:
                            result = args[0]
                        else:
                            for s in op.output_slots:
                                arena[s] = None
                            for kill_slot in op.kill_slots:
                                arena[kill_slot] = None
                            continue
                    elif _has_none_arg(args):
                        has_none_in_list = any(
                            isinstance(a, (list, tuple)) and any(item is None for item in a)
                            for a in args
                        )
                        if has_none_in_list:
                            if op.op_type in _ACCUMULATOR_OPS and args and args[0] is not None:
                                result = args[0]
                            else:
                                for s in op.output_slots:
                                    arena[s] = None
                                for kill_slot in op.kill_slots:
                                    arena[kill_slot] = None
                                continue
                        else:
                            raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e
                    else:
                        raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e

            slots = op.output_slots
            if len(slots) == 1:
                arena[slots[0]] = result
            elif len(slots) > 1:
                result_len = len(result) if isinstance(result, (tuple, list)) else 0
                for i, s in enumerate(slots):
                    if i < result_len:
                        arena[s] = result[i]
                    else:
                        arena[s] = None

            for kill_slot in op.kill_slots:
                arena[kill_slot] = None

    def gather_outputs(self, output_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Gather output tensors from the arena."""
        assert self._arena is not None, "compile() must be called before gather_outputs()"
        ids_to_gather = output_ids or self._output_tensor_ids
        outputs = {}
        for tensor_id in ids_to_gather:
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is not None and self._arena[slot] is not None:
                outputs[tensor_id] = self._arena[slot]
        return outputs

    def protect_tensor(self, tensor_id: str) -> bool:
        """
        Protect a tensor from liveness GC (post-compile).

        Removes the tensor's slot from any op's kill_slots so it survives execution.
        Also protects the storage chain — view/slice/reshape ops share underlying
        storage with their source tensor. If the source is freed, the view's data
        is invalidated even though the view's slot is preserved.

        Must be called BEFORE run().

        Args:
            tensor_id: The tensor ID to protect

        Returns:
            True if the tensor was found and protected
        """
        slot = self._tensor_id_to_slot.get(tensor_id)
        if slot is None:
            return False

        self._persistent_tensor_ids.add(tensor_id)

        # Remove this slot from any op's kill_slots
        self._remove_slot_from_kill_slots(slot)

        # Protect storage chain: trace back through view-like ops to protect
        # source tensors that share the same underlying storage.
        self._protect_storage_chain(tensor_id)

        return True

    # View-like ATen ops that share storage with their first input
    _VIEW_OPS = frozenset({
        "aten::view", "aten::_unsafe_view", "aten::reshape",
        "aten::slice", "aten::select", "aten::narrow",
        "aten::permute", "aten::transpose", "aten::t",
        "aten::expand", "aten::unsqueeze", "aten::squeeze",
        "aten::contiguous", "aten::as_strided",
    })

    def _remove_slot_from_kill_slots(self, slot: int) -> None:
        """Remove a slot from all ops' kill_slots."""
        from dataclasses import replace
        for i, op in enumerate(self._ops):
            if slot in op.kill_slots:
                new_kills = tuple(s for s in op.kill_slots if s != slot)
                self._ops[i] = replace(op, kill_slots=new_kills)

    def _protect_storage_chain(self, tensor_id: str) -> None:
        """
        Trace back through view-like ops to protect source tensors that
        share the same underlying PyTorch storage.

        View ops (view, slice, reshape, transpose, etc.) return tensors that
        alias the source tensor's memory. If liveness GC frees the source,
        the view's data becomes invalid. This method protects all tensors
        in the view chain up to the first non-view (allocating) op.
        """
        ops_metadata = self.dag.get("ops", {})

        # Find which op produces this tensor
        for op_uid, op_data in ops_metadata.items():
            output_ids = op_data.get("output_tensor_ids", [])
            if tensor_id not in output_ids:
                continue

            op_type = op_data.get("op_type", "")
            if op_type not in self._VIEW_OPS:
                return  # Hit a non-view op — storage is owned here, stop

            # This op is a view — its first input shares storage
            input_tids = op_data.get("input_tensor_ids", [])
            if not input_tids:
                return

            source_tid = input_tids[0]
            source_slot = self._tensor_id_to_slot.get(source_tid)
            if source_slot is not None:
                self._remove_slot_from_kill_slots(source_slot)
                # Recurse up the chain
                self._protect_storage_chain(source_tid)
            return

    def clear_intermediates(self) -> None:
        """Clear intermediate tensors from arena (keep weights and inputs)."""
        assert self._arena is not None, "compile() must be called before clear_intermediates()"
        self._arena.clear_intermediates()

    def clear_inputs(self) -> None:
        """Clear input tensors for next inference."""
        assert self._arena is not None, "compile() must be called before clear_inputs()"
        self._arena.clear_inputs()

    @property
    def num_ops(self) -> int:
        """Number of compiled ops."""
        return len(self._ops)

    @property
    def num_slots(self) -> int:
        """Number of tensor slots in arena."""
        return self._num_weights + self._num_inputs + self._num_intermediates

    @property
    def arena(self) -> TensorArena:
        """Direct access to arena for advanced use cases."""
        assert self._arena is not None, "compile() must be called before accessing arena"
        return self._arena
