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
import copy
import os
import re
import torch

from neurobrix.core.dtype.config import parse_dtype as _cfg_parse_dtype, DTYPE_MAP as _DTYPE_MAP
from .compiled_ops import CompiledOpResolver

# TEMP diagnostic state for _maybe_dump_tid_native (CompiledSequence has
# __slots__, so per-instance dump state lives in this module-level dict
# keyed by id(self)).
_DUMP_STATE_NATIVE: Dict[int, Dict[str, Any]] = {}

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

# ============================================================================
# TRANSFORMER BLOCK REGEX — used by get_op_blocks() introspection and shared
# with execution strategies (zero3 pipelining) that need to group weights or
# ops by transformer layer. Pattern extracts the integer block index from
# tensor/weight names following both NeuroTax (NeuroBrix standard, singular
# "block.N.") and HuggingFace / native conventions (plural "blocks.N.",
# "layers.N.", "model.layers.N.", etc.).
#   "block.0.attn.q"          → 0  (NeuroTax — the canonical form in NBX)
#   "blocks.0.attn.wq"        → 0  (vendor plural)
#   "model.layers.12.mlp.gate" → 12
#   "encoder.layers.5.norm"   → 5
# Non-block names (embeddings, final norms, lm_head) return no match — these
# are classified as block_idx = -1 by get_op_blocks().
# ============================================================================
_BLOCK_RE = re.compile(r'(?:blocks?|layers|model\.layers|encoder\.layers|decoder\.layers)\.(\d+)\.')


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


def _concrete_product_match(target: int, sym_dims: list) -> bool:
    """True if ``target`` equals the product of some contiguous run of CONCRETE
    (plain-int) entries in ``sym_dims`` — i.e. the value is fully explained by the
    input's concrete dims, with no symbolic dim contributing. Mirror of Forge
    windowing.py.
    """
    n = len(sym_dims)
    for a in range(n):
        prod = 1
        for b in range(a, n):
            d = sym_dims[b]
            if not isinstance(d, int):  # symbolic dim breaks the all-concrete run
                break
            prod *= d
            if prod == target:
                return True
            if prod > target:
                break
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


@dataclass(frozen=True)
class ExprArg:
    """
    Symbolic expression that resolves at runtime via SymbolicShapeResolver.

    Handles arbitrary expression trees from SymInt.to_json():
    - floordiv: (s1 + pad) // stride + 1
    - add/sub/mul/mod/neg: nested expressions

    Used for spatial dimensions in view/reshape ops that are derived from
    input spatial dims through conv chains.

    Fields:
        expr_dict: Raw expression dict from graph.json (evaluated recursively)
        trace_value: Value at trace time (fallback when symbols not bound)
    """
    expr_dict: dict
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

    def __setitem__(self, idx: int, value: Optional[torch.Tensor]) -> None:
        self._memory[idx] = value

    def __len__(self) -> int:
        return len(self._memory)

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
        '_op_uid_interceptors',  # Fine-grained per-op_uid interceptors for op-level tiling
        '_seq_dependent_constants',  # Constants with trace-time seq_len dim: [(slot, axis, sym_id, trace_val)]
        '_seq_constant_originals',  # Original full-size constants: {slot: tensor} — never narrowed
        '_pretranspose_weights',  # Weight tensor IDs that need .t().contiguous() at bind time
        '_op_blocks_cache',  # Cache for get_op_blocks() — immutable post-compile
    )

    def __init__(
        self,
        dag: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        amp_enabled: bool = True,
        use_triton: bool = False,
    ):
        """
        Initialize CompiledSequence.

        NOTE: This is 100% AUTONOMOUS - no dependency on NativeATenDispatcher.
        All op resolution is handled by CompiledOpResolver.

        Args:
            dag: The TensorDAG dict from graph.json
            device: The target device (e.g., torch.device("cuda:0"))
            dtype: The target dtype (e.g., torch.float16)
            amp_enabled: Whether to apply AMP autocast rules.
            use_triton: Use Triton kernels instead of PyTorch native ops.
        """
        self.dag = dag
        self.device = device
        self.dtype = dtype
        # use_triton parameter kept for backward compat but triton mode uses triton/ package

        # Extract graph's traced dtype for AMP policy decision
        graph_dtype_str = dag.get("torch_dtype", "")
        graph_dtype = _cfg_parse_dtype(graph_dtype_str) if graph_dtype_str else None

        # 100% Autonomous op resolution - no sequential_dispatcher dependency
        self.op_resolver = CompiledOpResolver(device, dtype, graph_dtype=graph_dtype,
                                              amp_enabled=amp_enabled,
                                              use_triton=use_triton)

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
        # Fine-grained per-op_uid interceptors (op-level tiling: targets a single
        # op instance, e.g. only aten.convolution::62 for Sana 4Kpx fusion).
        # Checked BEFORE op_type interceptors so a per-uid hook wins.
        self._op_uid_interceptors: Dict[str, Callable] = {}

        # Weight tensor IDs that need pre-transposition (set by _eliminate_weight_transpose_ops)
        self._pretranspose_weights: set = set()

        # get_op_blocks() cache — immutable after compile(), lazy populated
        self._op_blocks_cache: Optional[Dict[int, Dict[str, Any]]] = None

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

    def register_op_uid_interceptor(self, op_uid: str, interceptor: Callable) -> None:
        """
        Register a fine-grained interceptor for ONE specific op instance by uid.

        Distinct from register_op_interceptor (op_type-wide): this targets a
        single op_uid (e.g. "aten.convolution::62"). Used by the op-level
        tiling engine to intercept exactly the ops Prism flagged as VRAM
        overflows, not every conv/upsample in the DAG.

        Must be called BEFORE compile() OR followed by update_op_uid_interceptors()
        for hot-swap on an already-compiled sequence.
        """
        self._op_uid_interceptors[op_uid] = interceptor

    def update_op_uid_interceptors(self, interceptors: Dict[str, Callable]) -> None:
        """
        Hot-swap per-op_uid interceptors on an already-compiled sequence.

        Same pattern as update_op_interceptors but matches by op_uid.
        """
        self._op_uid_interceptors.update(interceptors)
        if not self._ops:
            return
        for op in self._ops:
            if op.op_uid in interceptors:
                op.func = interceptors[op.op_uid]

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

        # Phase -0.5: Eliminate aten::t on weight tensors (pre-transpose at bind time)
        # Removes ~5K ops/token for LLMs (weight.t() before every mm)
        self._eliminate_weight_transpose_ops(tensors, ops_metadata, execution_order)

        # Phase -0.25: (retired) Windowing symbolization — pad→view num_blocks
        # = ceil(seq/W) — is now emitted at trace/build time, so the graph
        # arrives with symbolic windowing dims and no runtime pre-compilation
        # pass is needed. The cross-branch propagation below still consumes the
        # trace-emitted symbolic expressions.

        # Phase -0.1: Propagate symbolic expressions across branches
        # Handles expand broadcast dims and view/reshape dims that depend on
        # symbolic values from a different data-flow branch (e.g., CFormer
        # windowed attention where Q branch expands to match KV window count).
        self._propagate_cross_branch_expressions(tensors, ops_metadata)

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

    def _eliminate_weight_transpose_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """
        Remove aten::t ops on weight tensors by pre-transposing at bind time.

        Pattern: param::weight → aten::t → aten::mm
        After:   param::weight (pre-transposed) → aten::mm

        Same rewire logic as _eliminate_detach_ops. The weight tensor gets
        transposed in bind_weights() using self._pretranspose_weights set.

        Impact: Eliminates ~5K ops/token for LLMs (each weight.t() before mm).
        """
        rewire: Dict[str, str] = {}
        transpose_uids: set = set()
        pretranspose_tids: set = set()

        # Collect expert weight IDs from moe_fused ops — must NOT be pre-transposed
        # because moe_fused_dispatch() calls .t() on them at runtime.
        moe_weight_ids: set = set()
        for op_data in ops_metadata.values():
            if op_data.get("op_type") == "custom::moe_fused":
                attrs = op_data.get("attributes", {})
                for key in ("expert_gate_weight_ids", "expert_up_weight_ids", "expert_down_weight_ids"):
                    moe_weight_ids.update(attrs.get(key, []))

        for op_uid in execution_order:
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue
            if op_data.get("op_type") != "aten::t":
                continue

            # aten::t has exactly 1 input tensor and 1 output tensor
            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])
            in_tid = None
            for arg in args:
                if isinstance(arg, dict) and arg.get("type") == "tensor":
                    in_tid = arg.get("tensor_id")
                    break
            if in_tid is None:
                input_tids = op_data.get("input_tensor_ids", [])
                if input_tids:
                    in_tid = input_tids[0]

            out_tids = op_data.get("output_tensor_ids", [])
            if in_tid is None or not out_tids:
                continue

            # Only eliminate if input is a weight tensor (param:: or buffer::)
            # Chase rewire chains first
            root_tid = in_tid
            while root_tid in rewire:
                root_tid = rewire[root_tid]

            if not (root_tid.startswith("param::") or root_tid.startswith("buffer::")):
                continue

            # Skip MoE expert weights — moe_fused_dispatch() transposes them at runtime
            if root_tid in moe_weight_ids:
                continue

            out_tid = out_tids[0]
            rewire[out_tid] = root_tid
            transpose_uids.add(op_uid)
            pretranspose_tids.add(root_tid)

        if not transpose_uids:
            return

        # Store set of weight tensor IDs that need pre-transposition
        self._pretranspose_weights = pretranspose_tids

        # Update tensor metadata shape: consumers expect the transposed shape
        for tid in pretranspose_tids:
            meta = tensors.get(tid, {})
            shape = meta.get("shape", [])
            if len(shape) == 2:
                meta["shape"] = [shape[1], shape[0]]

        # Reuse the same rewire logic as detach elimination
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
            if op_uid in transpose_uids:
                continue
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue

            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])
            new_args = [_rewire_arg(a) for a in args]
            if new_args != args:
                attrs["args"] = new_args

            kwargs = attrs.get("kwargs", {})
            if kwargs:
                new_kwargs = {k: _rewire_arg(v) for k, v in kwargs.items()}
                if new_kwargs != kwargs:
                    attrs["kwargs"] = new_kwargs

            input_tids = op_data.get("input_tensor_ids", [])
            if input_tids:
                new_input_tids = [rewire.get(t, t) for t in input_tids]
                if new_input_tids != input_tids:
                    op_data["input_tensor_ids"] = new_input_tids

        # Remove transpose ops from execution_order
        new_order = [uid for uid in execution_order if uid not in transpose_uids]
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
            # No seq_len symbols (diffusion models, audio): skip the LLM
            # path but STILL run spatial promotion below — image VAEs have
            # named "height"/"width" symbols that need rebinding when the
            # runtime spatial size differs from the trace size (Sana 4Kpx).
            from neurobrix.triton.promotion import _spatial_promotion_pass
            _spatial_promotion_pass(self.dag, tensors, ops_metadata,
                                    symbols, set(), set())
            return

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

        # Skip promotion for trace_values shared by multiple seq_len symbols.
        # When s1, s3, s6 all have trace_value=23 (e.g., traced with trace_seq_len=23
        # for all inputs), we can't determine which symbol a concrete "23" refers to.
        # Promoting to the wrong symbol causes runtime shape mismatches.
        trace_val_counts: Dict[int, int] = {}
        for _sid, tv in safe_symbols.items():
            trace_val_counts[tv] = trace_val_counts.get(tv, 0) + 1
        ambiguous_vals = {tv for tv, count in trace_val_counts.items() if count > 1}
        if ambiguous_vals:
            distinct_vals = set(safe_symbols.values())
            if len(distinct_vals) == 1:
                # Every seq_len symbol shares ONE trace value → the component has a
                # single sequence length (Kokoro: the phoneme count flows through
                # bert→text_encoder→predictor; also plain LLMs). They all bind to the
                # same runtime length, so the "which symbol does this 23 mean" worry
                # is moot — collapse to one canonical symbol and promote. Models with
                # genuinely distinct lengths (FLUX img=256 + txt=512) have distinct
                # trace values and fall through to the conservative skip below.
                canonical = next(iter(safe_symbols))
                safe_symbols = {canonical: next(iter(distinct_vals))}
            else:
                safe_symbols = {
                    sid: tv for sid, tv in safe_symbols.items()
                    if tv not in ambiguous_vals
                }

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

        # Pre-computed RoPE table fix (runs regardless of safe_symbols).
        # TinyLlama-style models have cos_cached[2048,64] sliced to [trace_seq_len, 64]
        # at trace time. The slice end must be replaced with the full table size so the
        # table is always available for absolute position indexing during decode.
        for op_uid, op_data in ops_metadata.items():
            op_type = op_data.get("op_type", "")
            if op_type != "aten::slice":
                continue
            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])
            if len(args) < 4:
                continue
            input_tids = op_data.get("input_tensor_ids", [])
            if not input_tids:
                continue
            first_input = input_tids[0]
            if not isinstance(first_input, str) or not first_input.startswith("param::"):
                continue
            wname = first_input[7:]
            if not any(k in wname for k in ("cos_cached", "sin_cached", "cos_cache", "sin_cache")):
                continue
            # This is a RoPE table slice — replace end with full table size
            table_shape = tensors.get(first_input, {}).get("shape", [])
            dim_arg = args[1] if isinstance(args[1], int) else (
                args[1].get("value") if isinstance(args[1], dict) else 0
            )
            if isinstance(dim_arg, int) and dim_arg < len(table_shape):
                full_size = table_shape[dim_arg]
                args[3] = {"type": "scalar", "value": full_size}

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
            # RoPE table slices already handled above (before safe_symbols check).
            # Only promote non-RoPE slices here.
            if op_type == "aten::slice" and len(args) >= 4:
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
                    # The shape may be a raw list OR the wrapped
                    # {"type":"list","value":[...]} form — unwrap it (the wrapped
                    # form was silently skipped, leaving e.g. zeros([B, seq, H])
                    # frozen at the trace seq_len → the x_pad/mask buffers in
                    # Kokoro's text_encoder failed at runtime).
                    sa_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                    shape_items = shape_arg.get("value", []) if sa_wrapped else shape_arg
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
                            args[shape_idx] = ({"type": "list", "value": shape_list}
                                               if sa_wrapped else shape_list)

            # aten::expand(tensor, size) — promote seq_len in size list.
            # size may be a raw list OR the wrapped {"type":"list","value":[...]}
            # form (same as view/reshape below); unwrap it so a seq_len element
            # frozen in the size (e.g. RoPE position expand size [1,1,seq]) is
            # promoted. Without the unwrap the wrapped form was silently skipped,
            # leaving the trace seq_len literal — at decode it broadcast the
            # single position to the trace length (orpheus RoPE degeneracy).
            elif op_type == "aten::expand" and len(args) >= 2:
                size_arg = args[1]
                size_is_wrapped = isinstance(size_arg, dict) and size_arg.get("type") == "list"
                size_items = size_arg.get("value", []) if size_is_wrapped else size_arg
                if isinstance(size_items, (list, tuple)):
                    size_list = list(size_items)
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
                        args[1] = {"type": "list", "value": size_list} if size_is_wrapped else size_list

            # aten::view / aten::reshape / aten::_unsafe_view
            # Shape args may contain seq_len — promote matching elements.
            # SKIP if the tracer already did algebraic symbolization (shape list
            # has symbol refs). The tracer's algebraic propagation is authoritative:
            # it derives symbolic dims from input shapes, so concrete values in the
            # shape list were intentionally left concrete (e.g., head_dim=128 that
            # happens to match trace_seq_len=128 but is NOT dynamic).
            elif op_type in ("aten::view", "aten::reshape", "aten::_unsafe_view") and len(args) >= 2:
                shape_arg = args[1]
                is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                shape_items = shape_arg.get("value", []) if is_wrapped else shape_arg
                if isinstance(shape_items, (list, tuple)):
                    # Check if tracer already symbolized this op (has symbol refs)
                    has_tracer_symbols = any(
                        isinstance(elem, dict) and elem.get("type") == "symbol"
                        for elem in shape_items
                    )
                    if has_tracer_symbols:
                        pass  # Tracer's algebraic symbolization is authoritative
                    else:
                        # Legacy graph without algebraic symbolization — fall back
                        # to brute-force promotion (collision check still applies)
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

        # Layer 8 — spatial-symbol promotion for diffusion VAEs.
        #
        # The seq_len pass above only promotes symbols named "seq_len" — by
        # design, that's the LLM contract. Image-diffusion graphs use named
        # symbols "height" and "width" (and occasionally "depth"); they
        # were left out. For Sana 1024 / PixArt 1024 the trace size happens
        # to equal the runtime size, so missing rebind is a silent no-op.
        # For Sana 4Kpx (trace 64×64 latent → runtime 128×128 latent) the
        # missing rebind surfaces as a literal `4096` (= trace H*W) / `64`
        # (= trace H or W) baked into `aten::expand` / `aten::view` shape
        # args that should have been `s_h * s_w` / `s_h` / `s_w`. This
        # pass walks view/expand/reshape/ones/zeros/full/new_zeros/new_ones
        # and promotes adjacent (H, W) pairs at any cascade scale plus
        # H*W flattened products. Bit-perfect on trace == runtime models
        # (resolver substitutes the symbol with its own trace_value).
        # Same implementation as triton/promotion.py — single source of
        # truth, mirrored across the native CompiledSequence and the
        # triton sequential dispatcher.
        from neurobrix.triton.promotion import _spatial_promotion_pass
        _spatial_promotion_pass(self.dag, tensors, ops_metadata,
                                symbols, set(), set())

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
                # An op output that IS a graph output but is absent from the
                # `tensors` dict — e.g. a pattern-reassembled custom::rms_norm
                # whose output keeps the ORIGINAL `aten.mul::N::out_0` tid (the
                # tensors entry belongs to the eliminated mul) — never reached the
                # is_output/graph_output_ids branch above, so it was assigned a
                # slot but NOT registered as a protected output. Liveness GC then
                # frees its slot before gather_outputs (component output never
                # consumed by a downstream op). Register it here so its slot joins
                # protected_slots. Surfaced by Wan UMT5 text_encoder (final_norm
                # = custom.rms_norm::48 -> aten.mul::266::out_0).
                if out_id in graph_output_ids and out_id not in self._output_tensor_ids:
                    self._output_tensor_ids.append(out_id)

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

        # Step 1b: Dead outputs — slots produced but never read by any op
        # have no entry above and were NEVER killed. Their last use IS the
        # producing op (freed right after production). R30 mirror of the
        # sequential paths' dead-output liveness: the CogVideoX VAE
        # all-at-once decode captures a never-consumed conv-cache clone per
        # causal conv at full pixel resolution — ~28 GiB of dead arena slots
        # accumulated and OOM'd the compiled f9 decode.
        for op_idx, op_uid in enumerate(execution_order):
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue
            for out_tid in op_data.get("output_tensor_ids", []):
                slot = self._tensor_id_to_slot.get(out_tid)
                if slot is not None and slot not in slot_last_use:
                    slot_last_use[slot] = op_idx

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

        # Interceptor priority: op_uid (fine-grained, op-level tiling) > op_type
        # (broad, KV cache) > native op resolver. The per-uid hook wins so a
        # specific instance can be tiled while siblings of the same op_type
        # keep the native path.
        if op_uid in self._op_uid_interceptors:
            func = self._op_uid_interceptors[op_uid]
        elif op_type in self._op_interceptors:
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
    # CROSS-BRANCH SYMBOLIC EXPRESSION PROPAGATION
    # ========================================================================

    def _propagate_cross_branch_expressions(
        self,
        tensors: Dict[str, Any],
        ops: Dict[str, Any],
    ) -> None:
        """
        Pre-compilation pass: inject symbolic expressions into ops that have
        hardcoded dims derived from a different data-flow branch.

        Problem: In windowed attention (CFormer), Q and K/V flow through
        separate branches. The K/V branch goes through a view that creates
        [num_windows, window_size, D] with symbolic num_windows. The Q branch
        uses expand([num_windows, ...]) to broadcast — but the tracer captures
        num_windows as a concrete int since it comes from Python code, not from
        the Q tensor's shape.

        Fix: Collect all symbolic expressions from tensor symbolic_shape fields.
        For expand broadcast dims and view/reshape dims that match a known
        expression's trace_value, inject the expression dict so ExprArg handles
        it at runtime.

        Safety: Only injects when trace_value maps to exactly ONE expression
        (avoids ambiguity). Only targets expand broadcast dims (input_dim=1)
        and view/reshape dims that merge/split symbolic dimensions.
        """
        # Step 1: Collect symbolic expressions from tensor symbolic_shapes
        # Build map: trace_value → expression_dict (only unique values)
        expr_map: Dict[int, Any] = {}  # trace_value → expression dict
        ambiguous: set = set()  # trace_values with multiple expressions

        # Architectural-constant dims from WEIGHT/parameter shapes (mirror of Forge
        # windowing.py). Used by the collision guard below — kept concrete only when
        # a value is BOTH a weight dim AND fully explained by concrete input dims.
        weight_dims: set = set()
        for _tid, tdata in tensors.items():
            if tdata.get("is_parameter") or tdata.get("weight_name"):
                for d in (tdata.get("shape") or []):
                    if isinstance(d, int) and d > 1:
                        weight_dims.add(d)

        for _tid, tdata in tensors.items():
            sym_shape = tdata.get("symbolic_shape", {})
            dims = sym_shape.get("dims", []) if isinstance(sym_shape, dict) else []
            for dim in dims:
                if not isinstance(dim, dict):
                    continue
                dim_type = dim.get("type", "")
                if dim_type not in ("floordiv", "add", "sub", "mul", "mod", "neg", "symbol"):
                    continue
                # Skip bare symbols — those are handled by _promote_seq_len_scalars
                if dim_type == "symbol":
                    continue
                trace_val = dim.get("trace", dim.get("trace_value"))
                if not isinstance(trace_val, int) or trace_val <= 1:
                    continue
                if trace_val in ambiguous:
                    continue
                if trace_val in expr_map:
                    # Check if it's the same expression (same dict structure)
                    if expr_map[trace_val] != dim:
                        ambiguous.add(trace_val)
                        del expr_map[trace_val]
                else:
                    expr_map[trace_val] = dim

        if not expr_map:
            return

        injected = 0

        # Step 2: Inject into expand/view/reshape/creation ops.
        #
        # Fixpoint loop: the dim-merge branch synthesizes product expressions
        # (e.g. num_windows * num_queries) for a flatten whose merged dim the
        # trace baked concrete. A LATER flatten in the same chain
        # (view → view → view) re-uses that same merged value, but the trace
        # bakes it concrete at every step AND it appears in that op's recorded
        # input_shape — so the dim-merge passthrough-safety skips it and only a
        # DIRECT match (which has no passthrough-safety) can fix it. Persisting
        # each synthesized product back into expr_map and re-running lets the
        # whole chain propagate a windowed dim consistently. Without it, the
        # head of the chain symbolizes but the tail keeps the trace value
        # (granite Q-Former: projector output [7,6,4096] instead of [1,42,4096]).
        for _fixpoint_iter in range(8):  # bounded: each pass injects strictly more
            injected_before = injected
            new_products: Dict[int, Any] = {}  # trace_val → synthesized product expr

            for _op_uid, op_data in ops.items():
                op_type = op_data.get("op_type", "")
                attrs = op_data.get("attributes", {})
                args = attrs.get("args", [])

                if op_type == "aten::expand" and len(args) >= 2:
                    # Expand: check broadcast dims (input_dim=1 → target_dim=N)
                    size_arg = args[1]
                    if not isinstance(size_arg, dict) or size_arg.get("type") != "list":
                        continue
                    size_list = size_arg.get("value", [])
                    input_shapes = op_data.get("input_shapes", [[]])
                    if not input_shapes:
                        continue
                    input_shape = input_shapes[0]

                    changed = False
                    new_size = list(size_list)
                    for i, (target, actual) in enumerate(zip(size_list, input_shape)):
                        if (isinstance(target, int) and actual == 1
                                and target > 1 and target in expr_map):
                            new_size[i] = expr_map[target]
                            changed = True
                            injected += 1

                    if changed:
                        new_args = list(args)
                        new_args[1] = {"type": "list", "value": new_size}
                        new_attrs = dict(attrs)
                        new_attrs["args"] = new_args
                        if "size" in new_attrs:
                            new_attrs["size"] = new_size
                        op_data["attributes"] = new_attrs

                elif op_type in ("aten::view", "aten::reshape", "aten::_unsafe_view"):
                    # View/reshape: check for hardcoded dims matching expressions
                    shape_key = "shape" if "shape" in attrs else "size" if "size" in attrs else None
                    if shape_key is None:
                        continue
                    old_shape = attrs[shape_key]
                    if not isinstance(old_shape, list):
                        continue

                    # Skip if already fully symbolized
                    has_expr = any(isinstance(s, dict) and s.get("type") in
                                  ("floordiv", "add", "sub", "mul", "mod", "neg")
                                  for s in old_shape)
                    if has_expr:
                        continue

                    # Get input shape for dim-merge product detection
                    input_shapes = op_data.get("input_shapes", [[]])
                    input_shape = input_shapes[0] if input_shapes else []
                    in_tids = op_data.get("input_tensor_ids", [])
                    in_sym_dims = []
                    if in_tids:
                        _iss = tensors.get(in_tids[0], {}).get("symbolic_shape", {})
                        if isinstance(_iss, dict):
                            in_sym_dims = _iss.get("dims", [])

                    changed = False
                    new_shape = list(old_shape)
                    for i, dim_val in enumerate(old_shape):
                        if not isinstance(dim_val, int) or dim_val <= 1:
                            continue
                        if dim_val in expr_map:
                            # Trace-value collision guard (mirror of Forge
                            # windowing.py): keep the dim CONCRETE only when it is BOTH
                            # a weight/architectural constant AND fully explained by
                            # concrete input dims. Both required: weight-dim alone
                            # over-fires on a spatial dim equal to a hidden size (Sana
                            # VAE s1*2==64); concrete-product alone over-fires on a
                            # baked windowed dim (granite view::65 num_blocks==6==1*6).
                            # The conjunction isolates the channel-collision case
                            # (openaudio decoder 192 == (s1-1)*8+16 at s1=23).
                            if (dim_val in weight_dims
                                    and _concrete_product_match(dim_val, in_sym_dims)):
                                continue
                            # Direct match (no passthrough-safety — an exact known
                            # windowed value is always the windowed value)
                            new_shape[i] = expr_map[dim_val]
                            changed = True
                            injected += 1
                        elif input_shape and dim_val not in input_shape:
                            # Dim-merge detection: dim_val = expr_val * constant
                            # E.g., 15 = 5 * 3 where 5 is symbolic and 3 is from input
                            # Safety: skip if dim_val appears in input_shape (passthrough,
                            # not a merge — e.g., window_size=15 passing through unchanged)
                            for expr_val, expr_dict in expr_map.items():
                                if dim_val % expr_val == 0:
                                    quotient = dim_val // expr_val
                                    if quotient > 1 and quotient in input_shape:
                                        # Create product expression
                                        product_expr = {
                                            "type": "mul",
                                            "left": expr_dict,
                                            "right": quotient,
                                            "trace": dim_val,
                                        }
                                        new_shape[i] = product_expr
                                        changed = True
                                        injected += 1
                                        # Persist so later flattens DIRECT-match it.
                                        if dim_val not in new_products:
                                            new_products[dim_val] = product_expr
                                        break

                    if changed:
                        new_attrs = dict(attrs)
                        new_attrs[shape_key] = new_shape
                        # Also update args list
                        new_args = list(new_attrs.get("args", []))
                        for ai, arg in enumerate(new_args):
                            if isinstance(arg, dict) and arg.get("type") == "list":
                                orig = arg.get("value", [])
                                if len(orig) == len(new_shape):
                                    new_args[ai] = {"type": "list", "value": new_shape}
                                    break
                        new_attrs["args"] = new_args
                        op_data["attributes"] = new_attrs

                elif op_type in ("aten::ones", "aten::zeros", "aten::full",
                                 "aten::empty", "aten::ones_like", "aten::zeros_like"):
                    # Creation ops: size list may contain hardcoded symbolic values
                    # E.g., ones([5, 15]) for windowed attention mask
                    if not args:
                        continue
                    size_arg = args[0]
                    if not isinstance(size_arg, dict) or size_arg.get("type") != "list":
                        continue
                    size_list = size_arg.get("value", [])

                    changed = False
                    new_size = list(size_list)
                    for i, dim_val in enumerate(size_list):
                        if isinstance(dim_val, int) and dim_val > 1 and dim_val in expr_map:
                            new_size[i] = expr_map[dim_val]
                            changed = True
                            injected += 1

                    if changed:
                        new_args = list(args)
                        new_args[0] = {"type": "list", "value": new_size}
                        new_attrs = dict(attrs)
                        new_attrs["args"] = new_args
                        if "size" in new_attrs:
                            new_attrs["size"] = new_size
                        op_data["attributes"] = new_attrs

            # Merge synthesized products into expr_map so the next pass can
            # DIRECT-match later flattens. Skip values that are ambiguous or
            # already mapped (don't override a tensor-derived expression).
            grew = False
            for _v, _e in new_products.items():
                if _v in ambiguous or _v in expr_map:
                    continue
                expr_map[_v] = _e
                grew = True

            if injected == injected_before and not grew:
                break

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
        from neurobrix.core.dtype.engine import routing_upcast_fp32

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

        # TEMP DIAG: identify block.1 from the first gate weight id
        _moe_diag_block1 = ("block.1." in gate_weight_ids[0]
                            and not any(f"block.1{d}." in gate_weight_ids[0]
                                        for d in "0123456789"))

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

            def _dnat(_label, _tensor): pass

            # DTYPE CONTRACT: resolve weight dtype once, cache for all subsequent calls
            w_dtype = _cached_w_dtype[0]
            if w_dtype is None:
                w_dtype = arena[_gate_w_slots[0]].dtype
                _cached_w_dtype[0] = w_dtype
            if hidden_states.dtype != w_dtype:
                hidden_states = hidden_states.to(w_dtype)

            # ROUTING IN FP32: precision-critical for expert selection. The
            # fp32-upcast policy is owned by the dtype engine (single source);
            # see routing_upcast_fp32 for the rationale (vLLM PR #14027).
            gate_scores = routing_upcast_fp32(gate_scores)

            # Ensure ALL routing tensors on hidden_states device
            _compute_dev = hidden_states.device
            if gate_scores.device != _compute_dev:
                gate_scores = gate_scores.to(_compute_dev)

            # Dynamic routing in fp32 (replaces hardcoded topk+sort+bincount+slice)
            scores, indices = gate_scores.topk(_top_k, dim=-1)
            _dnat("topk_scores", scores)
            _dnat("topk_indices", indices)

            if _norm_topk_prob:
                scores = scores / scores.sum(dim=-1, keepdim=True)
                _dnat("scores_norm", scores)

            flat_indices = indices.flatten()
            _dnat("flat_scores", scores.flatten())
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

            _dnat("result_final", output)
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
            elif isinstance(arg, ExprArg):
                # Dynamic expression resolution (spatial dims from conv chains)
                expr_resolver = self._make_expr_resolver(arg.expr_dict, arg.trace_value)
                resolvers.append(expr_resolver)
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
            elif isinstance(arg, ExprArg):
                expr_resolver = self._make_expr_resolver(arg.expr_dict, arg.trace_value)
                resolvers.append(expr_resolver)
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
            elif isinstance(item, ExprArg):
                expr_resolver = self._make_expr_resolver(item.expr_dict, item.trace_value)
                item_resolvers.append(expr_resolver)
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

    def _make_expr_resolver(self, expr_dict: dict, trace_value: int) -> Callable[[TensorArena], int]:
        """
        Generate closure for dynamic expression resolution.

        Handles arbitrary SymInt expression trees (floordiv, add, sub, mul, etc.)
        by delegating to SymbolicShapeResolver._resolve_symint_dict() at runtime.

        Used for spatial dimensions derived from conv chains:
        e.g., (s1 + 2*pad - dilation*(k-1) - 1) // stride + 1

        Args:
            expr_dict: Expression dict from SymInt.to_json()
            trace_value: Fallback value if symbols cannot be resolved

        Returns:
            Closure that evaluates expression at runtime
        """
        def resolve_expr(_arena: TensorArena) -> int:
            if self._shape_resolver is not None:
                try:
                    return self._shape_resolver._resolve_symint_dict(expr_dict)
                except Exception:
                    return trace_value
            return trace_value  # Fallback for graphs without resolver
        return resolve_expr

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

            # Expression types from SymInt.to_json() — spatial dim expressions
            if arg_type in ("floordiv", "add", "sub", "mul", "mod", "neg"):
                trace = arg.get("trace", arg.get("trace_value", 0))
                return ExprArg(expr_dict=arg, trace_value=trace)

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
                    # Complex number literals (e.g., "1j", "0.5j", "-1j")
                    if value.endswith("j"):
                        try:
                            return ScalarArg(complex(value))
                        except ValueError:
                            pass
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
                # Pre-transpose weights that had their aten::t ops eliminated.
                # Use .t() view (no copy) — BLAS handles transposed strides natively.
                if tensor_id in self._pretranspose_weights and tensor.ndim == 2:
                    tensor = tensor.t()
                self._arena[slot] = tensor

        # Pre-populate constant weight slots NOT provided by the weight
        # loader. These are trace-time orphan constants (e.g. the Python
        # scalar `cnt` from a `mask[slice] = cnt` loop captured as
        # `param::constant_T_*` with no constant_data, or KV-cache init
        # tensors with shape [0]). Without this, their arena slot stays
        # None and the consuming op (e.g. aten::fill) receives None →
        # crash. Mirrors the working sequential path in
        # tensor_resolver.py step 4 (`torch.empty(shape)` with a 0-dim
        # default). Only constant_*/shape-[0]/missing-norm slots are
        # touched — regular weight slots left unprovided by zero3
        # block-by-block streaming are intentionally NOT allocated.
        tensors_meta = self.dag.get("tensors", {})
        for tensor_id in self._weight_tensor_ids:
            slot = self._tensor_id_to_slot[tensor_id]
            if self._arena[slot] is not None:
                continue
            meta = tensors_meta.get(tensor_id, {})
            # 0-dim scalar default (matches the trace shape `[]` for
            # orphan scalars and tensor_resolver.py). A 1-dim `[0]`
            # default produced an ndim-1 tensor that broke aten::fill,
            # which requires a 0-dim value.
            shape = meta.get("shape", [])
            dtype_str = meta.get("dtype", "float32")
            t_dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
            if shape == [0] or "constant_" in tensor_id:
                self._arena[slot] = torch.empty(shape, dtype=t_dtype, device=self.device)
            elif not meta.get("weight_key") and ".norm." in tensor_id:
                resolved_shape = [s if isinstance(s, int) else s.get("trace_value", 1)
                                  for s in shape] if isinstance(shape, list) else shape
                if tensor_id.endswith(".weight"):
                    self._arena[slot] = torch.ones(resolved_shape, dtype=t_dtype, device=self.device)
                elif tensor_id.endswith(".bias"):
                    self._arena[slot] = torch.zeros(resolved_shape, dtype=t_dtype, device=self.device)

    def rebind_partial(self, partial_map: Dict[str, torch.Tensor]) -> List[int]:
        """Replace a subset of weights in the arena without touching the rest.

        Zero3 block-wise pipelining calls this to swap a block's weights
        between CPU (pinned, inert) and GPU (hot, used by compute) during
        a single component run. Unlike bind_weights(), this method is
        designed for the hot path — it only touches the arena slots for
        the provided tensor_ids and returns the list of modified slot
        indices so the caller can pass them to
        recompute_op_devices_for_slots() without rescanning the full op
        list.

        The same _pretranspose_weights contract as bind_weights() applies:
        2-D weights whose aten::t op was eliminated at compile time get a
        .t() view here too — otherwise a rebind to the transposed GPU
        tensor would break matmul shapes on the second block onwards.

        Args:
            partial_map: tensor_id → tensor. Tensor IDs not in
                self._tensor_id_to_slot are silently skipped (e.g. the
                caller may supply the full block weight list even when
                some names weren't in the graph).

        Returns:
            List of slot indices that were modified. Order is insertion
            order of partial_map but with unresolved tensor_ids skipped.
        """
        assert self._arena is not None, "compile() must be called before rebind_partial()"
        modified: List[int] = []
        for tensor_id, tensor in partial_map.items():
            slot = self._tensor_id_to_slot.get(tensor_id)
            if slot is None:
                continue
            if tensor_id in self._pretranspose_weights and tensor.ndim == 2:
                tensor = tensor.t()
            self._arena[slot] = tensor
            modified.append(slot)
        return modified

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
                    dev = tensor.device
                    op.device = dev
                    devices_seen.add(str(dev))
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

            # Record output device for downstream ops.
            # Tensor-creation ops (arange, scalar_tensor) may run before any
            # weighted op sets current_activation_device. Default to self.device
            # so downstream cross-device detection catches mismatches.
            out_dev = op.device or current_activation_device or self.device
            if out_dev is not None:
                for s in op.output_slots:
                    slot_device[s] = out_dev

    def recompute_op_devices_for_slots(self, modified_slots: List[int]) -> None:
        """Patch per-op device flags after a partial arena rebind.

        Called by zero3 block-wise pipelining: after rebind_partial()
        swaps a block's weights between CPU and GPU, the ops that read
        those weight slots need their op.device and op.needs_transfer
        flags re-derived from the new arena contents. The global
        activation-device graph built by compute_op_devices() stays
        valid (graph topology is immutable), so we only revisit the ops
        whose weight inputs intersect modified_slots.

        Contract — matches the semantics of Phase 1 + the zero3 slow
        path rule (CUDA always wins over CPU, so needs_transfer is True
        iff op.device differs from self.device):

            op.device        = device of first weight tensor currently in arena
            op.needs_transfer = (op.device != self.device)

        For weightless ops we skip — they inherit their activation
        device from the surrounding graph, which is unchanged.

        Args:
            modified_slots: output of rebind_partial(). Empty list is
                valid (no-op). Duplicates are tolerated.
        """
        assert self._arena is not None, "compile() must be called before recompute_op_devices_for_slots()"
        if not modified_slots:
            return
        arena = self._arena
        touched = set(modified_slots)
        exec_dev = self.device
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            if not any(ws in touched for ws in op.weight_input_slots):
                continue
            # Re-derive op.device from the first non-None weight tensor.
            new_dev = None
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, 'device'):
                    new_dev = tensor.device
                    break
            if new_dev is not None:
                op.device = new_dev
                op.needs_transfer = (exec_dev is not None and new_dev != exec_dev)

    def mark_cpu_weighted_ops_for_transfer(self, exec_device: torch.device) -> int:
        """Zero3 correctness: flag every CPU-weighted op to go slow-path.

        compute_op_devices() sets op.needs_transfer=True only for the
        FIRST weighted op in Phase 4's device-transition scan. For
        zero3 that's wrong: every CPU-weighted op needs the slow path
        because its weight is on CPU while activations are on GPU. The
        slow path in _run_inner_multi_device moves args to the GPU
        target per-op, so setting op.device=CPU + needs_transfer=True
        makes the allocator's per-op scratch tensor pattern kick in —
        VRAM stays bounded to one op's working set at a time.

        Idempotent. Returns the number of ops whose flag was flipped,
        for diagnostic logging at install time.
        """
        assert self._arena is not None, (
            "compile() must be called before mark_cpu_weighted_ops_for_transfer()")
        flipped = 0
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            for ws in op.weight_input_slots:
                t = self._arena[ws]
                if t is not None and hasattr(t, 'device') and t.device != exec_device:
                    op.device = t.device
                    op.needs_transfer = True
                    flipped += 1
                    break
        return flipped

    def materialize_slots_depending_on(self, weight_slot_ids) -> int:
        """Materialize every intermediate slot whose tensor aliases one of
        the given weight slots, so evicting those weights is safe.

        Mirror of TritonSequence.materialize_slots_depending_on.

        Under zero3 pipelining, a block's weights may be referenced by
        arena intermediates through torch's view semantics (shared storage
        via `_base`). Dropping the weight tensor while dependents still
        reference its storage would leave stale pointers. Calling this
        primitive before the evict breaks the alias by copying each
        dependent intermediate into fresh storage via `.contiguous()`,
        then replacing the slot.

        Intended to be called rarely — under a correctly-fused MoE graph
        (see moe_fusion.py Pass 2 output-sweep), dead views never land in
        the arena so this typically returns 0 on zero3.

        Args:
            weight_slot_ids: iterable of slot indices whose tensors are
                about to be freed.

        Returns:
            Count of slots that were materialized.
        """
        assert self._arena is not None, (
            "compile() must be called before materialize_slots_depending_on()")
        weight_slots = set(weight_slot_ids)
        if not weight_slots:
            return 0
        arena = self._arena

        # Collect the storage identities of weights being evicted. Views
        # in PyTorch share a `Storage` object with their base — the storage
        # address is the canonical identity.
        forbidden_storage_ids = set()
        for ws in weight_slots:
            t = arena[ws]
            if t is None or not hasattr(t, 'untyped_storage'):
                continue
            try:
                forbidden_storage_ids.add(
                    t.untyped_storage().data_ptr())
            except Exception:
                continue

        if not forbidden_storage_ids:
            return 0

        materialized = 0
        for slot_idx in range(len(arena)):
            if slot_idx in weight_slots:
                continue
            t = arena[slot_idx]
            if t is None or not hasattr(t, 'untyped_storage'):
                continue
            try:
                storage_ptr = t.untyped_storage().data_ptr()
            except Exception:
                continue
            if storage_ptr not in forbidden_storage_ids:
                continue
            # Materialize: fresh storage, break the alias.
            try:
                arena[slot_idx] = t.contiguous().clone()
            except Exception:
                arena[slot_idx] = None
            materialized += 1
        return materialized

    def override_weightless_op_devices(self, device: torch.device) -> None:
        """Force op.device = device for every op without weight inputs.

        compute_op_devices() derives each op's device from its weight
        inputs, and weightless ops (arange, scalar_tensor, full, casts
        on activation-only tensors) inherit device from the preceding
        weighted op. For zero3 that inheritance is WRONG: the weighted
        op is on CPU (weights CPU-offloaded) while the actual compute
        must happen on the GPU execution device. Tensor-creation ops
        then allocate on CPU (via kwargs['device'] = op.device patched
        in _run_inner_multi_device) and downstream SDPA crashes with
        "attn_bias on cpu".

        Zero3Strategy calls this once per component, right after
        compute_op_devices has run, to override the inherited device
        for every weightless op. Weighted ops stay untouched — their
        op.device still reflects weight placement and the normal rebind
        flow (rebind_partial + recompute_op_devices_for_slots) keeps
        them in sync as blocks swap in and out.
        """
        assert self._arena is not None, (
            "compile() must be called before override_weightless_op_devices()")
        for op in self._ops:
            if not op.weight_input_slots:
                op.device = device
                op.needs_transfer = False

    def get_op_blocks(self) -> Dict[int, Dict[str, Any]]:
        """Group the compiled op list by transformer block for pipelining.

        Each block is identified by an integer index extracted from the
        weight tensor names via _BLOCK_RE (e.g. "blocks.3.attn.wq" → 3).
        Weights that don't match the regex (embeddings, final norm,
        lm_head) go into block -1 — these are "non-block" weights that
        zero3 keeps GPU-resident for the whole component run.

        Ops without weight_input_slots inherit the block assignment of
        their predecessor in topological order. This keeps weightless
        ops (add/norm/activation) in the same boundary group as the
        weighted op they feed, so rebinding happens at the right moment.

        The result is cached on self._op_blocks_cache — safe because the
        compiled op list and slot↔tensor_id mapping are immutable after
        compile().

        Returns:
            Dict[int, Dict[str, Any]]:
                block_idx → {
                    'first_op': int,          # first op_idx in block (inclusive)
                    'last_op':  int,          # last op_idx in block (inclusive)
                    'weight_tensor_ids': List[str],  # tensor_ids in this block
                }
        """
        if self._op_blocks_cache is not None:
            return self._op_blocks_cache

        blocks: Dict[int, Dict[str, Any]] = {}
        last_assigned: int = -1
        for op_idx, op in enumerate(self._ops):
            block_idx: Optional[int] = None
            op_weight_tids: List[str] = []
            for ws in op.weight_input_slots:
                tid = self._slot_to_tensor_id.get(ws)
                if not tid:
                    continue
                op_weight_tids.append(tid)
                if block_idx is None:
                    # Strip "param::"/"buffer::" prefix before regex match.
                    if tid.startswith("param::"):
                        name = tid[7:]
                    elif tid.startswith("buffer::"):
                        name = tid[8:]
                    else:
                        name = tid
                    m = _BLOCK_RE.search(name)
                    block_idx = int(m.group(1)) if m else -1

            if block_idx is None:
                # Weightless op — inherit predecessor's block.
                block_idx = last_assigned

            last_assigned = block_idx
            entry = blocks.get(block_idx)
            if entry is None:
                entry = {
                    'first_op': op_idx,
                    'last_op': op_idx,
                    'weight_tensor_ids': [],
                }
                blocks[block_idx] = entry
            else:
                entry['last_op'] = op_idx
            if op_weight_tids:
                entry['weight_tensor_ids'].extend(op_weight_tids)

        # Dedupe weight_tensor_ids per block (preserve first-seen order).
        for entry in blocks.values():
            seen: Dict[str, None] = {}
            for tid in entry['weight_tensor_ids']:
                seen.setdefault(tid, None)
            entry['weight_tensor_ids'] = list(seen.keys())

        self._op_blocks_cache = blocks
        return blocks

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

    def run(
        self,
        debug: bool = False,
        pre_op_callback: Optional[Callable[[int, "CompiledOp"], None]] = None,
    ) -> None:
        """
        Execute the compiled sequence.

        THIS IS THE HOT LOOP - ZERO OVERHEAD:
        - No isinstance() checks
        - No dict lookups
        - No string operations
        - Just: resolve args, call func, store result

        Args:
            debug: If True, enable verbose error handling (slower)
            pre_op_callback: Optional (op_idx, op) -> None hook fired
                before each op dispatch. Used by zero3 pipelining to
                rebind/evict block weights at block boundaries. Default
                None → no overhead. Only honored by the multi-device
                path; single-device runs ignore it (zero3 is always
                multi-device-classified because weights on CPU differ
                from the CUDA executor device).
        """
        arena = self._arena

        # Use inference_mode to disable autograd and reduce memory usage
        with torch.inference_mode():
            self._run_inner(arena, debug, pre_op_callback)

    def _maybe_dump_tid_native(self, op, out_slot: int, tensor) -> None:
        """TEMP diagnostic: mirror of TritonSequence._maybe_dump_tid.
        Gated by NBX_DUMP_TIDS=/path and NBX_DUMP_TIDS_FILTER=sub1,sub2.
        Uses module-level state (CompiledSequence has __slots__).
        """
        import os as _os_d, json as _json_d
        global _DUMP_STATE_NATIVE
        try:
            _DUMP_STATE_NATIVE
        except NameError:
            return
        dump_path = _os_d.environ.get("NBX_DUMP_TIDS")
        if not dump_path:
            return
        state = _DUMP_STATE_NATIVE.setdefault(id(self),
            {"seen": set(), "records": [],
             "slot_to_tid": {s: t for t, s in self._tensor_id_to_slot.items()}})
        tid = state["slot_to_tid"].get(out_slot, f"slot::{out_slot}")
        filters = [f for f in _os_d.environ.get(
            "NBX_DUMP_TIDS_FILTER", "").split(",") if f]
        # Match filter against tid OR op_uid (op_uid lets us select by op,
        # e.g. all custom.rms_norm outputs across the network).
        if filters and not any(f in tid or f in op.op_uid for f in filters):
            return
        # Dedup by op_uid, NOT tid: CompiledSequence reuses arena slots, so many
        # distinct ops share one slot → one tid. Keying by tid silently dropped
        # every reused-slot op (the whole patch-embed/modulation/RoPE chain),
        # making cross-engine diffs blind. op_uid is unique per op.
        if op.op_uid in state["seen"]:
            return
        state["seen"].add(op.op_uid)
        try:
            if not isinstance(tensor, torch.Tensor):
                return
            # Complex tensors: read via the real view ([...,2] re/im) so head
            # casting to float succeeds and the l2 covers BOTH components. Mirrors
            # the triton dump convention so engine-vs-engine complex diffs align;
            # without this, `flat[:10].float()` throws on complex and the whole
            # record is silently skipped (compiled was missing every complex
            # transformer op — RoPE q/k — making cross-engine RoPE diffs blind).
            src = torch.view_as_real(tensor) if tensor.is_complex() else tensor
            # Memory-safe: avoid `tensor.detach().float()` which
            # would materialize a 2× fp32 copy of large bf16/fp16
            # tensors (would OOM on 16g for [1,128,4096,4096]
            # = 4 GiB bf16 → 8 GiB fp32). Sample head in-place,
            # compute L2 norm with fp32 accumulator without
            # casting the full tensor.
            flat = src.detach().reshape(-1)
            head = flat[:10].float().cpu().tolist()
            try:
                norm = float(torch.linalg.vector_norm(
                    flat, dtype=torch.float32).item())
            except Exception:
                # Fallback for older torch / unsupported dtype.
                norm = float(flat[:1024].float().norm().item())
            _bn = None
            if src.dim() >= 2 and src.shape[0] in (2, 3):
                try:
                    _bn = [float(torch.linalg.vector_norm(
                        src[bi].detach().reshape(-1), dtype=torch.float32).item())
                        for bi in range(src.shape[0])]
                except Exception:
                    _bn = None
            new_record = {
                "component": self.dag.get("component_name", "?"),
                "tid": tid, "op_uid": op.op_uid, "op_type": op.op_type,
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "head10": head, "l2_norm": norm, "batch_norms": _bn,
            }
            state["records"].append(new_record)
            # JSONL append-mode write — O(1) IO per call vs the
            # earlier O(N²) read-then-append. Each record is one JSON
            # line. Bit-diff readers must use the `.jsonl` extension
            # to consume line-by-line. Cross-instance merge happens
            # naturally because every CompiledSequence appends to the
            # same file.
            with open(dump_path, "a") as f:
                _json_d.dump({"engine": "compiled",
                              "record": new_record}, f)
                f.write("\n")
        except Exception as e:
            print(f"[NBX_DUMP_TIDS compiled] failed on {tid}: {e}", flush=True)

    def _run_inner(
        self,
        arena,
        debug: bool = False,
        pre_op_callback: Optional[Callable[[int, "CompiledOp"], None]] = None,
    ) -> None:
        """Inner loop extracted for inference_mode wrapping."""
        trace_nan = _TRACE_NAN
        nan_guard = _NAN_GUARD
        nan_guard_verbose = _NAN_GUARD_VERBOSE
        trace_zeros = _TRACE_ZEROS
        # NBX_TRACE_RANGES=op1,op2,... — log per-op output max(abs) to
        # NBX_RANGE_LOG (default /tmp/nbx_ranges.tsv). Used by
        # neurobrix.tools.measure_activation_ranges to decide the
        # per-component activations_fp16_safe flag opt-in.
        _trace_ranges_env = os.environ.get("NBX_TRACE_RANGES", "")
        _trace_ranges_set = set(_trace_ranges_env.split(",")) if _trace_ranges_env else set()
        _range_log_path = os.environ.get("NBX_RANGE_LOG", "/tmp/nbx_ranges.tsv")
        _range_log_fh = None
        if _trace_ranges_set:
            _range_log_fh = open(_range_log_path, "a")
            _range_log_fh.write(f"# new run, ops={_trace_ranges_env}\n")
            _range_log_fh.write("op_idx\top_uid\top_type\tmax_abs\tdtype\tnumel\n")

        # Set CUDA device to match component placement
        if self._is_multi_device:
            self._run_inner_multi_device(arena, debug, pre_op_callback)
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
                # DEBUG: catch CPU tensor errors in triton mode
                if "cpu tensor" in str(e).lower() or "Pointer argument" in str(e):
                    import sys
                    print(f"[TRITON-DEBUG] CPU tensor at op {op.op_uid} ({op.op_type})", file=sys.stderr)
                    for ai, a in enumerate(args):
                        if isinstance(a, torch.Tensor):
                            print(f"  arg[{ai}]: device={a.device} dtype={a.dtype} shape={list(a.shape)}", file=sys.stderr)
                    raise
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
                    import os as _os_fe
                    if _os_fe.environ.get("NBX_DEBUG") == "1":
                        _rb = (self._shape_resolver.get_bound_symbols()
                               if self._shape_resolver is not None else None)
                        try:
                            _ra = op.args_resolver(arena)
                            _ras = [(tuple(a.shape) if hasattr(a, "shape") else a)
                                    for a in _ra]
                        except Exception as _e_ra:
                            _ras = f"<resolver raised {_e_ra}>"
                        print(f"[FAIL-CTX] op={op.op_uid} resolver_bound={_rb} "
                              f"resolver_is={id(self._shape_resolver)} "
                              f"re-resolved_args={_ras}", flush=True)
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
            # === NBX_OPLOG: every-op execution log (None vs l2) — covers the
            # tid-dump's blind spots (NOP'd ops, aliases). Gated, append-only.
            import os as _os_ol1
            _olp1 = _os_ol1.environ.get("NBX_OPLOG")
            if _olp1:
                try:
                    if result is None:
                        _ols1 = "None"
                    elif isinstance(result, torch.Tensor):
                        _ols1 = (f"l2={float(torch.linalg.vector_norm(result.detach().reshape(-1), dtype=torch.float32)):.4f}"
                                 if result.is_floating_point() else f"dtype={result.dtype}")
                    else:
                        _ols1 = type(result).__name__
                    with open(_olp1, "a") as _olf1:
                        _olf1.write(f"{op.op_uid}\t{op.op_type}\t{_ols1}\n")
                except Exception:
                    pass
            # ============================================================
            if len(slots) == 1:
                # ================================================================
                # NaN-GUARD for single tensor output (handled above)
                # ================================================================
                arena[slots[0]] = result
                # === TEMP TID DUMP: compare native vs triton per-op output ===
                import os as _os_d
                if _os_d.environ.get("NBX_DUMP_TIDS"):
                    self._maybe_dump_tid_native(op, slots[0], result)
                # ==============================================================
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

            # === TEMP RANGE TRACE ===
            if _range_log_fh is not None and isinstance(result, torch.Tensor) and result.is_floating_point():
                bare = op.op_type.split("::")[-1] if "::" in op.op_type else op.op_type
                if bare in _trace_ranges_set:
                    _ma = result.detach().abs().max().item() if result.numel() > 0 else 0.0
                    _range_log_fh.write(f"{op_idx}\t{op.op_uid}\t{op.op_type}\t{_ma:.6g}\t{result.dtype}\t{result.numel()}\n")
            # =========================

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

    def _run_inner_multi_device(
        self,
        arena,
        _debug: bool = False,  # noqa: ARG002
        pre_op_callback: Optional[Callable[[int, "CompiledOp"], None]] = None,
    ) -> None:
        """
        FGP multi-device hot loop with cross-device activation transfer.

        OPTIMIZATION: Only ops marked needs_transfer=True at compile time
        go through the expensive device alignment path. For Qwen3-30B with
        115K ops across 4 GPUs, only ~100 ops at block boundaries need
        transfers — the other 99.9% use the fast single-device path.

        Cross-device .to() queues the copy on the source stream, but the
        target stream doesn't implicitly wait. We use CUDA events for
        fine-grained sync: record on source, wait on target.

        pre_op_callback is invoked with (op_idx, op) BEFORE each op's
        args are resolved. Used by zero3 pipelining: the callback checks
        whether op_idx crosses a block boundary, and if so it synchronizes
        the transfer stream, rebinds the next block's weights via
        rebind_partial + recompute_op_devices_for_slots, and kicks off
        the prefetch of the block after that. The callback is skipped
        entirely (no branch cost beyond the None check) when None.
        """
        import torch
        _current_device_idx = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(_current_device_idx)
        for op_idx, op in enumerate(self._ops):
            if pre_op_callback is not None:
                pre_op_callback(op_idx, op)
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            # Fix device kwargs for multi-device execution.
            # _compile_arg bakes in self.device for ALL device kwargs, but in
            # FGP mode tensor-creation ops (arange, scalar_tensor, full) must
            # create tensors on op.device (from compute_op_devices placement).
            if kwargs and 'device' in kwargs and op.device is not None:
                kwargs['device'] = op.device

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
                    # Triton mode: also set CUDA runtime device (Triton uses runtime, not PyTorch)
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
                            # Pinned CPU→GPU: use non_blocking DMA (~2x faster)
                            nb = a.is_pinned() and target.type == "cuda"
                            new_args.append(a.to(target, non_blocking=nb))
                        elif isinstance(a, (list, tuple)):
                            moved = []
                            any_moved = False
                            for item in a:
                                if isinstance(item, torch.Tensor) and item.device != target:
                                    nb = item.is_pinned() and target.type == "cuda"
                                    moved.append(item.to(target, non_blocking=nb))
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
                                nb = v.is_pinned() and target.type == "cuda"
                                new_kwargs[k] = v.to(target, non_blocking=nb)
                            elif isinstance(v, (list, tuple)):
                                moved = []
                                any_moved = False
                                for item in v:
                                    if isinstance(item, torch.Tensor) and item.device != target:
                                        nb = item.is_pinned() and target.type == "cuda"
                                        moved.append(item.to(target, non_blocking=nb))
                                        any_moved = True
                                    else:
                                        moved.append(item)
                                new_kwargs[k] = type(v)(moved) if any_moved else v
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
            # === NBX_OPLOG: every-op execution log (None vs l2) — covers the
            # tid-dump's blind spots (NOP'd ops, aliases). Gated, append-only.
            import os as _os_ol
            _olp = _os_ol.environ.get("NBX_OPLOG")
            if _olp:
                try:
                    if result is None:
                        _ols = "None"
                    elif isinstance(result, torch.Tensor):
                        _ols = (f"l2={float(torch.linalg.vector_norm(result.detach().reshape(-1), dtype=torch.float32)):.4f}"
                                if result.is_floating_point() else f"dtype={result.dtype}")
                    else:
                        _ols = type(result).__name__
                    with open(_olp, "a") as _olf:
                        _olf.write(f"{op.op_uid}\t{op.op_type}\t{_ols}\n")
                except Exception:
                    pass
            # ============================================================
            if len(slots) == 1:
                arena[slots[0]] = result
                # === TEMP TID DUMP (multi-device branch) ===
                import os as _os_d2
                if _os_d2.environ.get("NBX_DUMP_TIDS"):
                    self._maybe_dump_tid_native(op, slots[0], result)
                # ============================================
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
