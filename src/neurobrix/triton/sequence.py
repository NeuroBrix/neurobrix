"""Triton Compiled Sequence — zero-overhead execution hot loop.

Ported from compiled_sequence.py. Same proven logic for arg compilation,
closure-based resolution, and liveness analysis. Replaces torch.dtype
with NBXDtype and uses Arena + SymbolResolver for triton mode.

Zero torch dependency in the hot loop.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurobrix.kernels.dispatch import dispatch
from neurobrix.kernels.nbx_tensor import (
    NBXTensor, NBXDtype, DeviceAllocator, parse_dtype,
    invalidate_current_device_cache,
)
from neurobrix.kernels.wrappers import (
    deferred_drain_policy, _DEFERRED_DRAIN_FLOOR_DEFAULT)

from .arena import Arena
from .symbols import SymbolResolver
from .dtype import TritonDtypeEngine


# ----------------------------------------------------------------------------
# Periodic _deferred drain — Route A (April 2026)
# ----------------------------------------------------------------------------
# The hot loops in _run_single_device / _run_multi_device push any tensor
# evicted from an arena slot (output overwrite + kill_slot) onto a local
# `_deferred` list. Until April 2026 that list was drained exactly once
# per run(), after a device sync. On a DiT with ~5600 ops per forward
# pass (PixArt-Sigma 28 layers × ~200 ops/layer, CFG batch=2), `_deferred`
# grew to ~950 live tensors × ~40 MB avg = 30 GB — on a V100 32 GB that
# hit OOM inside the transformer, not because anything leaked but because
# the peak concurrency was bounded only by the run's total allocation
# volume.
#
# Fix: drain `_deferred` the moment it crosses a bytes threshold. The drain
# uses `sync_device()` before clearing the list (cudaFree is synchronous and
# releasing while a kernel is still reading would UAF), so correctness is
# identical to the end-of-run drain — we just do it more often.
#
# Pressure-sensitive policy (audit #2 F1, P-TRITON-LIVE-SET): a flat 2 GB cap
# let the deferred queue carry up to ~2 GB of dead tensors above the compiled
# free-at-last-use watermark. But draining is not free — with the caching
# pool off (default) every drain is a cudaFree→cudaMalloc cycle, so draining
# eagerly on a model that HAS headroom is pure churn (a fixed 650 MiB cap was
# measured to regress CogVideoX-2b VAE decode ~49% with ~21 GB free — no
# benefit). So `wrappers.deferred_drain_policy()` returns
# (cliff, floor, pressure_reserve, freecheck_interval): the queue always
# drains at `cliff` (2 GB, throughput ceiling — unchanged when there is
# headroom, hence NO runtime regression), and drains EARLY (at `floor`) only
# when a throttled driver-free probe shows the device is within
# `pressure_reserve` of capacity — i.e. only when a lower watermark is needed
# to avoid OOM, where a slower run that fits beats an OOM. Unconfigured arch
# → cliff-only (prior behavior preserved).
#
# Tuning via env vars (lookup once, not in the hot loop):
#   NBX_DEFERRED_DRAIN_BYTES   pins the cliff AND disables the pressure branch
#      (floor := cliff) — reproduces a fixed cap exactly (debug / A-B).
#   NBX_DEFERRED_DRAIN_RESERVE  pins the pressure reserve (force/relax the
#      early-drain pressure gate for demonstration).
#   NBX_DEFERRED_DRAIN_COUNT  default 512
#      Safety net against pathologies where many small tensors accumulate
#      without ever crossing the bytes threshold (hypothetical: many-heads
#      SDPA with strided workspace). OR-ed with the bytes threshold —
#      the first trigger drains.
#   NBX_DEFERRED_DRAIN_DIAG   default "0"
#      Set to "1" to print drain events (count, bytes, trigger reason).
#
# Known limit (April 2026): Route A plafonne le peak concurrency
# INTRA-run. PixArt-Sigma/Alpha triton crash further along with an
# independent arena-state bug between runs (cross-run slot corruption
# — a tensor retained by `_base` chain across the first run's final
# drain has a dangling data_ptr on run 3). Tracked in
# docs/follow-ups/pixart_triton_arena_inter_run_bug.md.
# ----------------------------------------------------------------------------

def _parse_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# Bytes/pressure thresholds are data-driven (wrappers.deferred_drain_policy);
# the count safety-net stays a flat cap (guards pathological many-small-
# tensor accumulation that never crosses the bytes budget).
_DEFERRED_DRAIN_COUNT_DEFAULT = 512


def _resolve_drain_policy():
    """Resolve the per-run deferred-drain policy once, applying env overrides.

    Returns `(cliff, floor, reserve, interval, count_limit)`. The config-driven
    base comes from `wrappers.deferred_drain_policy()`; two env overrides layer
    on top for reproducibility/demonstration (looked up once per run, never in
    the hot loop):
      - NBX_DEFERRED_DRAIN_BYTES pins `cliff` AND disables the pressure branch
        (`floor := cliff`, `reserve := 0`) → reproduces a fixed cap exactly.
      - NBX_DEFERRED_DRAIN_RESERVE pins `reserve`; if the pressure branch was
        disabled it is re-enabled at the default floor.
    Shared by both hot loops so the policy is identical single- and multi-device
    (R30 by construction). No torch, no compute (R33).
    """
    cliff, floor, reserve, interval = deferred_drain_policy()
    _env_cap = _parse_env_int("NBX_DEFERRED_DRAIN_BYTES", 0)
    if _env_cap > 0:
        cliff = _env_cap
        floor = _env_cap        # disable the pressure branch
        reserve = 0
    _env_res = _parse_env_int("NBX_DEFERRED_DRAIN_RESERVE", -1)
    if _env_res >= 0:
        reserve = _env_res
        if floor >= cliff:      # pressure branch was disabled — re-enable it
            floor = min(_DEFERRED_DRAIN_FLOOR_DEFAULT, cliff)
    count_limit = _parse_env_int(
        "NBX_DEFERRED_DRAIN_COUNT", _DEFERRED_DRAIN_COUNT_DEFAULT)
    return cliff, floor, reserve, interval, count_limit


# Same pattern as core/runtime/graph/compiled_sequence._BLOCK_RE and
# triton/weight_loader._BLOCK_RE. Used by TritonSequence.get_op_blocks
# to partition ops by transformer layer for zero3 pipelining and the
# correctness slow-path priming.
_BLOCK_RE = re.compile(
    r'(?:blocks?|layers|model\.layers|encoder\.layers|decoder\.layers)\.(\d+)\.')


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

# NOP-propagation accumulator ops (deactivated-MoE-expert path: when
# args[0] is None the op passes the base accumulator through instead of
# nulling the output slot). R30: must mirror the semantics of
# core/runtime/graph/compiled_sequence.py:_ACCUMULATOR_OPS (the
# reference oracle), which lists the aten:: MoE-aggregation ops
# {scatter_reduce, scatter_add, index_add, scatter, index_put}. The
# triton NOP check compares the BARE op name (op_type.split("::")[-1]),
# so the compiled set is mirrored here as bare names. `add`/`mul` are
# kept (triton-specific accumulator NOP handling — extend, do not
# replace, to avoid regressing models that rely on them).
_ACCUMULATOR_OPS = frozenset({
    "add", "mul",
    "scatter_reduce", "scatter_add", "index_add", "scatter", "index_put",
})


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
    weight_input_slots: Tuple[int, ...] = ()
    all_input_slots: Tuple[int, ...] = ()
    device_idx: Optional[int] = None
    needs_transfer: bool = False


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
        from neurobrix.kernels.wrappers import has_native_bf16 as _has_bf16
        self._dtype_engine = TritonDtypeEngine(
            compute_dtype, has_native_bf16=_has_bf16())
        self._compute_dtype = compute_dtype
        # Phase 1 — per-component opt-in flag from the model registry
        # `activations_fp16_safe`. Default False = conservative (rms_norm/div
        # output stays fp32). Set via set_activations_fp16_safe() after
        # construction, before compile/run, by the registry-driven plumbing
        # in graph_executor (or equivalent component init site).
        self._activations_fp16_safe: bool = False
        self._compiled = False

        # Multi-device support (pipeline_parallel)
        self._is_multi_device = False

        # Op interceptors: op_type → callable (for KV cache injection)
        self._op_interceptors: Dict[str, Callable] = {}
        # Per-op_uid interceptors (op-level tiling). Checked BEFORE op_type
        # interceptors so a per-uid hook (e.g. only aten.convolution::62)
        # overrides any op_type-wide interceptor.
        self._op_uid_interceptors: Dict[str, Callable] = {}

        # Weight tensor IDs that need .t() at bind time (from
        # _eliminate_weight_transpose_ops pass).
        self._pretranspose_weights: set = set()

        # Cache for get_op_blocks — lazy populated on first call, safe
        # because the compiled op list is immutable after compile().
        self._op_blocks_cache: Optional[Dict[int, Dict[str, Any]]] = None

    def set_activations_fp16_safe(self, safe: bool) -> None:
        """Set the per-component activations_fp16_safe opt-in flag.

        Read from the model registry at component init via
        RegistryV2. When True, AMP_FP32_OPS in
        _AMP_FP32_OPS_OPT_IN_CAST_BACK (currently rms_norm + div) cast
        their output back to compute_dtype after the fp32 internal
        compute. Conservative default (False) preserves fp32 output.
        """
        self._activations_fp16_safe = bool(safe)

    def register_op_interceptor(self, op_type: str, interceptor: Callable):
        """Register an interceptor for a specific op type (e.g., SDPA for KV cache)."""
        self._op_interceptors[op_type] = interceptor
        # Hot-patch already compiled ops if sequence is compiled
        if self._compiled:
            for op in self._ops:
                if op.op_type == op_type:
                    op.func = interceptor

    def register_op_uid_interceptor(self, op_uid: str, interceptor: Callable):
        """Register a fine-grained interceptor for one specific op instance.

        Targets a single op_uid rather than every op of a given type — used
        by op-level tiling to intercept exactly the ops Prism flagged as
        VRAM overflows while leaving sibling ops on the native Triton path.
        Op_uid hooks take priority over op_type hooks at compile time.
        """
        self._op_uid_interceptors[op_uid] = interceptor
        if self._compiled:
            for op in self._ops:
                if op.op_uid == op_uid:
                    op.func = interceptor

    def update_op_uid_interceptors(self, interceptors: Dict[str, Callable]):
        """Hot-swap per-op_uid interceptors on an already-compiled sequence."""
        self._op_uid_interceptors.update(interceptors)
        if not self._compiled:
            return
        for op in self._ops:
            if op.op_uid in interceptors:
                op.func = interceptors[op.op_uid]

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

        # Phase -1: Eliminate aten::detach ops (identity at inference time).
        # Mirrors CompiledSequence._eliminate_detach_ops. Heavy impact on
        # models where the tracer captured a detach per parameter access
        # (DeepSeek: 19,428/44,634 ops, T5: ~36%).
        self._eliminate_detach_ops(tensors, ops_by_uid, exec_order)

        # Phase -0.5: Eliminate aten::t on weight tensors (pre-transpose at
        # bind time). Mirrors CompiledSequence._eliminate_weight_transpose_ops.
        # Removes ~154 ops/step for TinyLlama (one `t` before every weight `mm`).
        self._eliminate_weight_transpose_ops(tensors, ops_by_uid, exec_order)

        # Phase -0.4: Eliminate dead causal-mask chains (ones→tril→logical_not
        # →where→SDPA.attn_mask). The triton flash kernel handles causality
        # internally via IS_CAUSAL=True, so the whole chain is dead code.
        # Removes ~132 ops/step for TinyLlama (6 ops × 22 layers).
        self._eliminate_dead_causal_mask_ops(tensors, ops_by_uid, exec_order)

        # Phase -0.3: Fuse SwiGLU (silu + mul) into custom::swiglu_fused.
        # Collapses 2 elem ops → 1 kernel launch, and the fused kernel reads
        # gate/up once each + writes output once (vs silu writing intermediate
        # + mul reading/writing). Saves ~22 ops/step for Llama-style FFNs.
        # NBX_DISABLE_SWIGLU_FUSION=1 — diagnostic gate to isolate fused-swiglu
        # numerical divergence (custom::swiglu_fused fp32-intermediate vs the
        # unfused silu→fp16-store→mul chain the op-by-op path runs).
        if os.environ.get("NBX_DISABLE_SWIGLU_FUSION") != "1":
            self._fuse_swiglu_ops(tensors, ops_by_uid, exec_order)

        # Phase -0.2: Fuse HF-Llama rotate_half RoPE chains on Q+K into a
        # single custom::rope_fused op backed by Liger's rope_forward_kernel.
        # Collapses 14 ops per layer (slice×4, neg×2, cat×2, mul×4, add×2).
        # Saves ~308 ops/step for TinyLlama (22 layers).
        # NBX_DISABLE_ROPE_FUSION=1 — diagnostic gate to isolate fused-rope
        # numerical divergence (custom::rope_fused vs the unfused ATen chain
        # the op-by-op path runs); leaves the rotate_half ops native.
        if os.environ.get("NBX_DISABLE_ROPE_FUSION") != "1":
            self._fuse_rope_ops(tensors, ops_by_uid, exec_order)

        # Phase 0: Promote trace-time seq_len scalars to symbolic references.
        # Shared logic in triton/promotion.py — used by both compiled and sequential.
        from .promotion import promote_seq_len_scalars
        promote_seq_len_scalars(self.dag, tensors, ops_by_uid)

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

        # Phase 6: Identify seq-dependent constants (RoPE cos/sin buffers)
        self._seq_dependent_constants = []
        self._seq_constant_originals = {}
        self._identify_seq_dependent_constants(tensors)

        self._compiled = True

    # ========================================================================
    # OP ELIMINATION — ported from compiled_sequence
    # ========================================================================

    @staticmethod
    def _rewire_arg(arg: Any, rewire: Dict[str, str]) -> Any:
        """Rewire tensor_id references in a single arg dict.

        Handles type=tensor, type=tensor_tuple, and nested type=list.
        Shared by _eliminate_detach_ops and _eliminate_weight_transpose_ops.
        """
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
            new_items = [TritonSequence._rewire_arg(item, rewire)
                         for item in items]
            arg = dict(arg)
            arg["value"] = new_items
        return arg

    def _apply_rewire_to_remaining_ops(
        self,
        rewire: Dict[str, str],
        dropped_uids: set,
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Apply rewire map to args/kwargs/input_tensor_ids of every op
        not in dropped_uids, and remove dropped ops from execution_order.
        """
        for op_uid in execution_order:
            if op_uid in dropped_uids:
                continue
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue

            attrs = op_data.get("attributes", {})
            args = attrs.get("args", [])
            new_args = [self._rewire_arg(a, rewire) for a in args]
            if new_args != args:
                attrs["args"] = new_args

            kwargs = attrs.get("kwargs", {})
            if kwargs:
                new_kwargs = {k: self._rewire_arg(v, rewire)
                              for k, v in kwargs.items()}
                if new_kwargs != kwargs:
                    attrs["kwargs"] = new_kwargs

            input_tids = op_data.get("input_tensor_ids", [])
            if input_tids:
                new_input_tids = [rewire.get(t, t) for t in input_tids]
                if new_input_tids != input_tids:
                    op_data["input_tensor_ids"] = new_input_tids

        # Remove dropped ops from execution_order (in place).
        new_order = [uid for uid in execution_order if uid not in dropped_uids]
        execution_order.clear()
        execution_order.extend(new_order)

    def _eliminate_detach_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Remove aten::detach ops — identity at inference time (no autograd).

        Mirrors CompiledSequence._eliminate_detach_ops. Builds a rewire map
        from detach output → detach input, chases rewire chains, rewrites
        every remaining op's args/kwargs/input_tensor_ids, transfers the
        is_output flag, and removes the detach ops from execution_order.
        Universal: works for any family (LLM, diffusion, audio, video, SR).
        """
        rewire: Dict[str, str] = {}
        detach_uids: set = set()

        for op_uid in execution_order:
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue
            if op_data.get("op_type") != "aten::detach":
                continue

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

            out_tid = out_tids[0]

            # Chase rewire chains (earlier detach may have rewired in_tid).
            while in_tid in rewire:
                in_tid = rewire[in_tid]

            rewire[out_tid] = in_tid
            detach_uids.add(op_uid)

        if not detach_uids:
            return

        self._apply_rewire_to_remaining_ops(
            rewire, detach_uids, ops_metadata, execution_order)

        # Rewire DAG-level outputs.
        dag_outputs = self.dag.get("output_tensor_ids", [])
        if dag_outputs:
            new_dag_outputs = [rewire.get(t, t) for t in dag_outputs]
            if new_dag_outputs != dag_outputs:
                self.dag["output_tensor_ids"] = new_dag_outputs

        # Transfer is_output flag on tensor metadata.
        for out_tid, in_tid in rewire.items():
            out_meta = tensors.get(out_tid, {})
            if out_meta.get("is_output") and in_tid in tensors:
                tensors[in_tid]["is_output"] = True

    def _eliminate_weight_transpose_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Remove aten::t on weight tensors by pre-transposing at bind time.

        Mirrors CompiledSequence._eliminate_weight_transpose_ops. For every
        aten::t whose (rewire-chased) input is a param::/buffer:: tensor,
        rewire the output to the root weight, record the weight id in
        self._pretranspose_weights, and swap its shape metadata (dims 0↔1).
        bind_weights() then calls .t() on those weights.

        MoE expert weights consumed by custom::moe_fused are excluded —
        moe_fused_dispatch() performs its own transpose at runtime.
        Universal: applies to any family.
        """
        rewire: Dict[str, str] = {}
        transpose_uids: set = set()
        pretranspose_tids: set = set()

        # Collect expert weight IDs from moe_fused ops.
        moe_weight_ids: set = set()
        for op_data in ops_metadata.values():
            if op_data.get("op_type") == "custom::moe_fused":
                attrs = op_data.get("attributes", {})
                for key in ("expert_gate_weight_ids",
                            "expert_up_weight_ids",
                            "expert_down_weight_ids"):
                    moe_weight_ids.update(attrs.get(key, []))

        # The elimination MUTATES the shared in-memory DAG (t ops removed,
        # weight shapes swapped). A recompile over the already-mutated DAG
        # finds no aten::t and would lose the bind-time transpose contract
        # (second-request cold-serving crash, R30 mirror of the native
        # pass). The mutation carries its own contract: eliminated weights
        # are stamped `pretransposed`; every compile seeds from the stamps.
        seeded: set = {
            tid for tid, meta in tensors.items()
            if isinstance(meta, dict) and meta.get("pretransposed")
        }

        for op_uid in execution_order:
            op_data = ops_metadata.get(op_uid)
            if op_data is None:
                continue
            if op_data.get("op_type") != "aten::t":
                continue

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

            # Chase rewire chains (a prior detach-elim may have rewired in_tid
            # onto the underlying weight).
            root_tid = in_tid
            while root_tid in rewire:
                root_tid = rewire[root_tid]

            # Only eliminate transposes on weight tensors.
            if not (root_tid.startswith("param::")
                    or root_tid.startswith("buffer::")):
                continue

            # Skip MoE expert weights — moe_fused_dispatch transposes at runtime.
            if root_tid in moe_weight_ids:
                continue

            out_tid = out_tids[0]
            rewire[out_tid] = root_tid
            transpose_uids.add(op_uid)
            pretranspose_tids.add(root_tid)

        if not transpose_uids:
            self._pretranspose_weights = seeded
            return

        self._pretranspose_weights = seeded | pretranspose_tids

        # Swap shape metadata: consumers expect the transposed shape.
        # Newly eliminated tids only — swap and stamp exactly once per
        # DAG lifetime.
        for tid in pretranspose_tids:
            meta = tensors.setdefault(tid, {})
            meta["pretransposed"] = True
            shape = meta.get("shape", [])
            if len(shape) == 2:
                meta["shape"] = [shape[1], shape[0]]

        self._apply_rewire_to_remaining_ops(
            rewire, transpose_uids, ops_metadata, execution_order)

    def _eliminate_dead_causal_mask_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Eliminate dead causal-mask chains feeding SDPA attn_mask.

        Pattern (per attention layer):
            ones([N,N], bool) -> tril -> logical_not
                -> where(cond, scalar_tensor(-inf), scalar_tensor(0.0))
                -> SDPA as attn_mask

        The triton flash_attention kernel handles causal masking internally
        via IS_CAUSAL=True, so the entire chain is dead code. We detect
        matching chains, rewrite each SDPA to set is_causal=True and drop
        the mask argument, then remove the chain ops from execution_order.

        Safety:
        - Only fires when where->SDPA is the ONLY consumer of the where
          output, AND the where's fill values are exactly -inf (True branch)
          and 0.0 (False branch).
        - scalar_tensor / ones / tril / logical_not ops are dropped only if
          ALL their consumers are in the to-be-eliminated set. Shared
          non-dead consumers keep the op alive.
        - Universal: works for any LLM/audio/image model with this causal
          bias pattern. Non-causal masks (diffusion cross-attn, arbitrary
          where ops) never match because of the -inf/0.0 fill check and the
          tril->logical_not->where topology requirement.
        """
        # Step 1: build producer + consumers index.
        producer: Dict[str, str] = {}
        consumers: Dict[str, List[str]] = {}
        for op_uid in execution_order:
            od = ops_metadata.get(op_uid)
            if od is None:
                continue
            for t in od.get("output_tensor_ids", []):
                producer[t] = op_uid
            for t in od.get("input_tensor_ids", []):
                consumers.setdefault(t, []).append(op_uid)

        def _scalar_tensor_value(op_uid: str):
            od = ops_metadata.get(op_uid) or {}
            if od.get("op_type") != "aten::scalar_tensor":
                return None
            args = od.get("attributes", {}).get("args", [])
            for a in args:
                if isinstance(a, dict) and a.get("type") == "scalar":
                    return a.get("value")
            return None

        # Step 2: for each SDPA, test mask chain and collect eliminable ops.
        sdpa_rewrites: List[str] = []
        mask_chain_candidates: set = set()

        for op_uid in execution_order:
            od = ops_metadata.get(op_uid)
            if od is None:
                continue
            if od.get("op_type") != "aten::scaled_dot_product_attention":
                continue
            inputs = od.get("input_tensor_ids", [])
            if len(inputs) < 4:
                continue
            mask_tid = inputs[3]
            where_uid = producer.get(mask_tid)
            if where_uid is None:
                continue
            where_op = ops_metadata.get(where_uid) or {}
            if where_op.get("op_type") != "aten::where":
                continue
            # Where must not be consumed by anything else.
            if consumers.get(mask_tid, []) != [op_uid]:
                continue
            w_inputs = where_op.get("input_tensor_ids", [])
            if len(w_inputs) < 3:
                continue
            cond_tid, true_tid, false_tid = w_inputs[0], w_inputs[1], w_inputs[2]
            true_v = _scalar_tensor_value(producer.get(true_tid, ""))
            false_v = _scalar_tensor_value(producer.get(false_tid, ""))
            # Classic causal bias: True → -inf, False → 0.0.
            if not (true_v is not None and float(true_v) == float("-inf")):
                continue
            if not (false_v is not None and float(false_v) == 0.0):
                continue
            # Condition chain: where.cond <- logical_not <- tril <- ones(bool).
            lnot_uid = producer.get(cond_tid)
            lnot_op = ops_metadata.get(lnot_uid) if lnot_uid else None
            if lnot_op is None or lnot_op.get("op_type") != "aten::logical_not":
                continue
            lnot_in = (lnot_op.get("input_tensor_ids") or [None])[0]
            tril_uid = producer.get(lnot_in) if lnot_in else None
            tril_op = ops_metadata.get(tril_uid) if tril_uid else None
            if tril_op is None or tril_op.get("op_type") != "aten::tril":
                continue
            tril_in = (tril_op.get("input_tensor_ids") or [None])[0]
            ones_uid = producer.get(tril_in) if tril_in else None
            ones_op = ops_metadata.get(ones_uid) if ones_uid else None
            if ones_op is None or ones_op.get("op_type") != "aten::ones":
                continue

            # This SDPA qualifies.
            sdpa_rewrites.append(op_uid)
            mask_chain_candidates.update([
                where_uid, lnot_uid, tril_uid, ones_uid,
                producer[true_tid], producer[false_tid],
            ])

        if not sdpa_rewrites:
            return

        # Step 3: rewrite SDPA ops first (drops mask from their inputs).
        for op_uid in sdpa_rewrites:
            od = ops_metadata[op_uid]
            inputs = od.get("input_tensor_ids", [])
            od["input_tensor_ids"] = list(inputs[:3])
            attrs = od.setdefault("attributes", {})
            args = attrs.get("args", [])
            if args:
                attrs["args"] = [a for a in args if not (
                    isinstance(a, dict) and a.get("type") == "tensor"
                    and a.get("tensor_id") == inputs[3])]
            kwargs = attrs.setdefault("kwargs", {})
            kwargs["is_causal"] = True

        # Step 4: drop chain ops only if ALL their consumers are eliminated
        # (or were the SDPA's dropped mask position, now removed).
        # Recompute consumers after SDPA rewrite.
        consumers2: Dict[str, List[str]] = {}
        for op_uid in execution_order:
            if op_uid in sdpa_rewrites:
                od = ops_metadata[op_uid]
            else:
                od = ops_metadata.get(op_uid)
                if od is None:
                    continue
            for t in od.get("input_tensor_ids", []):
                consumers2.setdefault(t, []).append(op_uid)

        to_drop: set = set()
        # Iterate to a fixed point — dropping one op may free its inputs.
        changed = True
        while changed:
            changed = False
            for cand in list(mask_chain_candidates):
                if cand in to_drop:
                    continue
                od = ops_metadata.get(cand)
                if od is None:
                    continue
                out_tids = od.get("output_tensor_ids", [])
                all_dead = True
                for t in out_tids:
                    for c in consumers2.get(t, []):
                        if c not in to_drop:
                            all_dead = False
                            break
                    if not all_dead:
                        break
                if all_dead:
                    to_drop.add(cand)
                    changed = True

        if not to_drop:
            return

        # Remove dropped ops from execution_order.
        new_order = [u for u in execution_order if u not in to_drop]
        execution_order.clear()
        execution_order.extend(new_order)

    def _fuse_swiglu_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Fuse aten::silu + aten::mul into custom::swiglu_fused.

        Pattern (per FFN layer):
            gate = ...                  # typically _unsafe_view / mm
            up   = ...                  # typically _unsafe_view / mm
            silu_out = silu(gate)
            result   = mul(silu_out, up)

        Replaced by a single custom::swiglu_fused op backed by
        silu_mul_split_kernel (see kernels/wrappers.py:swiglu_fused_wrapper).

        Safety:
        - silu output must be consumed ONLY by one aten::mul
        - that mul must have exactly 2 tensor inputs, one of which is the
          silu output
        - silu op is dropped only if its output has no other consumer
        - universal: works for Llama-family (TinyLlama, Qwen, DeepSeek dense)
          and any model with the silu(gate) * up SwiGLU pattern.
        """
        # Index producer + consumers.
        producer: Dict[str, str] = {}
        consumers: Dict[str, List[str]] = {}
        for op_uid in execution_order:
            od = ops_metadata.get(op_uid)
            if od is None:
                continue
            for t in od.get("output_tensor_ids", []):
                producer[t] = op_uid
            for t in od.get("input_tensor_ids", []):
                consumers.setdefault(t, []).append(op_uid)

        to_drop: set = set()
        new_ops: List[Tuple[str, str, Dict[str, Any]]] = []  # (silu_uid, new_uid, op_data)

        fuse_idx = 0
        for silu_uid in list(execution_order):
            silu_op = ops_metadata.get(silu_uid)
            if silu_op is None or silu_op.get("op_type") != "aten::silu":
                continue
            silu_outs = silu_op.get("output_tensor_ids", [])
            if len(silu_outs) != 1:
                continue
            silu_out_tid = silu_outs[0]

            # silu input (gate).
            silu_ins = silu_op.get("input_tensor_ids", [])
            if len(silu_ins) != 1:
                continue
            gate_tid = silu_ins[0]

            # silu output must have exactly one consumer — an aten::mul.
            cs = consumers.get(silu_out_tid, [])
            if len(cs) != 1:
                continue
            mul_uid = cs[0]
            mul_op = ops_metadata.get(mul_uid)
            if mul_op is None or mul_op.get("op_type") != "aten::mul":
                continue
            mul_ins = mul_op.get("input_tensor_ids", [])
            if len(mul_ins) != 2:
                continue
            if silu_out_tid not in mul_ins:
                continue
            up_tid = mul_ins[0] if mul_ins[1] == silu_out_tid else mul_ins[1]

            # Output of the mul becomes output of the fused op.
            mul_outs = mul_op.get("output_tensor_ids", [])
            if len(mul_outs) != 1:
                continue
            fused_out_tid = mul_outs[0]

            # Build fused op. Reuse mul's device/dtype + output tensor id.
            new_uid = f"custom.swiglu_fused::{fuse_idx}"
            fuse_idx += 1
            fused_op: Dict[str, Any] = {
                "op_uid": new_uid,
                "op_type": "custom::swiglu_fused",
                "input_tensor_ids": [gate_tid, up_tid],
                "output_tensor_ids": [fused_out_tid],
                "input_shapes": [
                    tensors.get(gate_tid, {}).get("shape", []),
                    tensors.get(up_tid, {}).get("shape", []),
                ],
                "output_shapes": [mul_op.get("output_shapes", [None])[0]],
                "input_dtypes": [
                    tensors.get(gate_tid, {}).get("dtype"),
                    tensors.get(up_tid, {}).get("dtype"),
                ],
                "output_dtypes": mul_op.get("output_dtypes", []),
                "device": mul_op.get("device") or silu_op.get("device"),
                "attributes": {
                    "args": [
                        {"type": "tensor", "tensor_id": gate_tid},
                        {"type": "tensor", "tensor_id": up_tid},
                    ],
                    "kwargs": {},
                },
            }

            fused_op["_pair_mul_uid"] = mul_uid
            to_drop.add(silu_uid)
            to_drop.add(mul_uid)
            new_ops.append((silu_uid, new_uid, fused_op))

        if not new_ops:
            return

        # Register new ops in metadata.
        for _, new_uid, fused_op in new_ops:
            ops_metadata[new_uid] = fused_op

        # Rebuild execution_order: drop the silu, replace the mul with the
        # fused op. The mul is AFTER both gate and up are produced (silu
        # position may be before up_mm completes), so scheduling the fused
        # op at the mul's slot is safe.
        mul_to_new: Dict[str, str] = {}
        silu_drops: set = set()
        for silu_uid, new_uid, fused_op in new_ops:
            mul_to_new[fused_op["_pair_mul_uid"]] = new_uid
            silu_drops.add(silu_uid)
        new_order: List[str] = []
        for uid in execution_order:
            if uid in silu_drops:
                continue
            if uid in mul_to_new:
                new_order.append(mul_to_new[uid])
            else:
                new_order.append(uid)
        execution_order.clear()
        execution_order.extend(new_order)

    def _fuse_rope_ops(
        self,
        tensors: Dict[str, Any],
        ops_metadata: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Fuse HF-Llama rotate_half RoPE chains into custom::rope_fused.

        Pattern per RoPE branch (applied separately to Q and K):

            X = raw_tensor                              # [B, H, S, D] view
            x_left  = slice(X, dim=-1, 0, D/2)
            x_right = slice(X, dim=-1, D/2, D)
            x_rot   = cat([neg(x_right), x_left], -1)
            y = mul(X, cos_unsq) + mul(x_rot, sin_unsq)

        Q and K branches share cos_unsq and sin_unsq (same layer). We pair
        Q-side and K-side adds by matching (cos_producer, sin_producer),
        then emit ONE custom::rope_fused op per pair consuming
        (q_raw, k_raw, cos_unsq, sin_unsq) and producing (q_add_out, k_add_out).

        Dropped per layer: 2×slice(Q) + 2×slice(K) + neg(Q) + neg(K) +
        cat(Q) + cat(K) + 2×mul(Q) + 2×mul(K) + add(Q) + add(K) = 14 ops.
        The downstream _to_copy dtype casts remain, consuming the fused op's
        outputs.

        Safety / universality:
          * Each of the chain ops must have EXACTLY the fused op in its
            consumer set (after pairing) — shared intermediates skip fusion.
          * Only the canonical rotate_half topology matches. Any deviation
            (interleaved RoPE, complex64 RoPE, ALiBi, NoPE) is left alone.
          * Works for any HF-Llama family model (TinyLlama, Qwen, DeepSeek
            dense, Mistral, Gemma, Phi) — the ATen fingerprint is identical.
        """
        producer: Dict[str, str] = {}
        consumers: Dict[str, List[str]] = {}
        for op_uid in execution_order:
            od = ops_metadata.get(op_uid)
            if od is None:
                continue
            for t in od.get("output_tensor_ids", []):
                producer[t] = op_uid
            for t in od.get("input_tensor_ids", []):
                consumers.setdefault(t, []).append(op_uid)

        def _op(uid):
            return ops_metadata.get(uid) if uid else None

        # Step 1: detect single-branch RoPE patterns.
        # Returns a tuple of (chain_info_dict) or None.
        def _detect_branch(add_uid: str):
            add_op = _op(add_uid)
            if add_op is None or add_op.get("op_type") != "aten::add":
                return None
            add_ins = add_op.get("input_tensor_ids", [])
            if len(add_ins) != 2:
                return None
            mul_a_op = _op(producer.get(add_ins[0], ""))
            mul_b_op = _op(producer.get(add_ins[1], ""))
            if not (mul_a_op and mul_b_op):
                return None
            if mul_a_op.get("op_type") != "aten::mul":
                return None
            if mul_b_op.get("op_type") != "aten::mul":
                return None
            # Identify which side is X*cos vs x_rot*sin by finding the cat.
            for x_mul_op, rot_mul_op in (
                    (mul_a_op, mul_b_op), (mul_b_op, mul_a_op)):
                rot_ins = rot_mul_op.get("input_tensor_ids", [])
                if len(rot_ins) != 2:
                    continue
                # One of rot_ins is a cat output, the other is sin_unsq.
                cat_side, sin_side = None, None
                for tid in rot_ins:
                    if _op(producer.get(tid, "")) and _op(producer[tid]).get("op_type") == "aten::cat":
                        cat_side = tid
                    else:
                        sin_side = tid
                if cat_side is None or sin_side is None:
                    continue
                cat_uid = producer[cat_side]
                cat_op = _op(cat_uid)
                cat_ins = cat_op.get("input_tensor_ids", [])
                if len(cat_ins) != 2:
                    continue
                # First cat input must be neg(slice_right), second must be slice_left.
                neg_op = _op(producer.get(cat_ins[0], ""))
                left_slice_op = _op(producer.get(cat_ins[1], ""))
                if not (neg_op and left_slice_op):
                    continue
                if neg_op.get("op_type") != "aten::neg":
                    continue
                if left_slice_op.get("op_type") != "aten::slice":
                    continue
                neg_in = (neg_op.get("input_tensor_ids") or [None])[0]
                right_slice_op = _op(producer.get(neg_in, ""))
                if right_slice_op is None or right_slice_op.get("op_type") != "aten::slice":
                    continue
                # Both slices must have the same X input (raw tensor).
                x_from_right = (right_slice_op.get("input_tensor_ids") or [None])[0]
                x_from_left = (left_slice_op.get("input_tensor_ids") or [None])[0]
                if x_from_right is None or x_from_right != x_from_left:
                    continue
                # x_mul inputs: (X, cos_unsq).
                x_mul_ins = x_mul_op.get("input_tensor_ids", [])
                if len(x_mul_ins) != 2:
                    continue
                x_tid, cos_tid = None, None
                for tid in x_mul_ins:
                    if tid == x_from_right:
                        x_tid = tid
                    else:
                        cos_tid = tid
                if x_tid is None or cos_tid is None:
                    continue
                sin_tid = sin_side
                # Collect chain uids to drop (if their outputs are SOLELY
                # consumed within the chain).
                chain_uids = {
                    add_uid,
                    producer[add_ins[0]], producer[add_ins[1]],  # two muls
                    cat_uid,
                    producer[neg_in],  # right slice
                    producer[cat_ins[0]],  # neg
                    producer[cat_ins[1]],  # left slice
                }
                return {
                    "add_uid": add_uid,
                    "x_tid": x_tid,
                    "cos_tid": cos_tid,
                    "sin_tid": sin_tid,
                    "chain_uids": chain_uids,
                    "add_out_tid": add_op.get("output_tensor_ids", [None])[0],
                }
            return None

        # Step 2: collect all RoPE branches.
        branches: List[Dict[str, Any]] = []
        for op_uid in execution_order:
            info = _detect_branch(op_uid)
            if info is not None:
                branches.append(info)

        if not branches:
            return

        # Step 3: pair branches sharing the SAME (cos, sin) producer pair.
        # Two branches with identical cos_tid and sin_tid → Q and K of one layer.
        from collections import defaultdict
        pair_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for b in branches:
            pair_map[(b["cos_tid"], b["sin_tid"])].append(b)

        fuse_idx = 0
        to_drop: set = set()
        # Position of each op in the ORIGINAL order — used to schedule the fused
        # op strictly after its inputs and before its outputs' consumers.
        pos = {uid: i for i, uid in enumerate(execution_order)}
        insert_before: Dict[str, List[str]] = defaultdict(list)
        new_ops_by_uid: Dict[str, Dict[str, Any]] = {}
        import os as _os_rs
        _RS_DIAG = _os_rs.environ.get("NBX_ROPE_SCHED_DIAG")

        for (cos_tid, sin_tid), group in pair_map.items():
            if len(group) != 2:
                # Unpaired (single branch or >2 sharing cos/sin) — skip.
                continue
            b_q, b_k = group

            # Safety: every chain op's output must be consumed ONLY inside
            # the chain (or by the fused replacement). Outputs that escape
            # to other ops mean the intermediate is live elsewhere.
            chain_all = b_q["chain_uids"] | b_k["chain_uids"]
            chain_out_tids = set()
            for cu in chain_all:
                od = ops_metadata.get(cu)
                if od is None:
                    continue
                for t in od.get("output_tensor_ids", []):
                    chain_out_tids.add(t)
            # The two add outputs are EXTERNALLY consumed (by _to_copy), so
            # exempt them from the check. All other chain outputs must be
            # consumed only by other chain ops.
            exempt = {b_q["add_out_tid"], b_k["add_out_tid"]}
            ok = True
            for t in chain_out_tids:
                if t in exempt:
                    continue
                for c in consumers.get(t, []):
                    if c not in chain_all:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Sanity: Q.x_tid should correspond to the tensor with MORE
            # heads than K.x_tid (GQA). Not a hard requirement for
            # correctness — the kernel handles any (n_qh, n_kh) — but it
            # disambiguates Q vs K for readable op ordering.
            q_shape = tensors.get(b_q["x_tid"], {}).get("shape", [])
            k_shape = tensors.get(b_k["x_tid"], {}).get("shape", [])

            # Layout guard: the fused kernel is built for the 4D HF-Llama
            # [B, H, S, D] layout. Packed-varlen 3D ropes (ViT towers of
            # the GLM-4.1V/Qwen-VL class: [S, H, D], no batch dim) keep
            # the correct unfused ATen chain — fusing them is an L2 perf
            # item for a dedicated kernel, not a layout reinterpretation.
            if len(q_shape) != 4 or len(k_shape) != 4:
                continue
            # If K has more heads than Q, swap so "Q" really is Q.
            if (len(q_shape) >= 4 and len(k_shape) >= 4
                    and isinstance(q_shape[1], int) and isinstance(k_shape[1], int)
                    and q_shape[1] < k_shape[1]):
                b_q, b_k = b_k, b_q
                q_shape, k_shape = k_shape, q_shape

            new_uid = f"custom.rope_fused::{fuse_idx}"
            q_out_tid = b_q["add_out_tid"]
            k_out_tid = b_k["add_out_tid"]

            # SCHEDULING (data-driven, topology-general). The single fused op
            # replaces BOTH branches' chains, so it must run strictly AFTER all
            # four inputs are produced and strictly BEFORE any consumer of either
            # output. HF-Llama applies Q and K rope adjacently, so the later of
            # the two adds happens to satisfy both — but that heuristic is not
            # general: Open-Sora's MMDiT feeds one rope output into the joint-
            # attention cat before the OTHER branch's rope add completes, so a
            # fused op anchored at the later add produces the first output AFTER
            # its consumer already ran → the consumer reads a not-yet-written
            # slot = None (NOP-cascades the whole attention chain). Compute the
            # valid window [max_input_pos, min_consumer_pos) from real positions;
            # if a consumer precedes an input the pair is UNFUSABLE as one op →
            # leave it unfused (correct, just unoptimized). Never a consumer
            # without its producer.
            _in_prod = [producer.get(t) for t in
                        (b_q["x_tid"], b_k["x_tid"], cos_tid, sin_tid)]
            _max_in = max((pos[p] for p in _in_prod if p in pos), default=-1)
            _out_cons = [c for c in (consumers.get(q_out_tid, [])
                                     + consumers.get(k_out_tid, []))
                         if c in pos and c not in chain_all]
            _min_con = min((pos[c] for c in _out_cons),
                           default=len(execution_order))
            if _max_in >= _min_con or _min_con >= len(execution_order):
                # A consumer precedes an input, or no in-order consumer exists —
                # unfusable as one op; leave the chain unfused (correct, slower).
                if _RS_DIAG:
                    print(f"[ROPE_SCHED] UNFUSABLE cos={cos_tid[:24]} "
                          f"max_in={_max_in} min_con={_min_con} — left unfused",
                          flush=True)
                continue
            anchor = execution_order[_min_con]  # insert fused op just before it
            fuse_idx += 1

            fused_op: Dict[str, Any] = {
                "op_uid": new_uid,
                "op_type": "custom::rope_fused",
                "input_tensor_ids": [
                    b_q["x_tid"], b_k["x_tid"], cos_tid, sin_tid],
                "output_tensor_ids": [q_out_tid, k_out_tid],
                "input_shapes": [
                    tensors.get(b_q["x_tid"], {}).get("shape", []),
                    tensors.get(b_k["x_tid"], {}).get("shape", []),
                    tensors.get(cos_tid, {}).get("shape", []),
                    tensors.get(sin_tid, {}).get("shape", []),
                ],
                "output_shapes": [
                    tensors.get(q_out_tid, {}).get("shape", []),
                    tensors.get(k_out_tid, {}).get("shape", []),
                ],
                "input_dtypes": [
                    tensors.get(b_q["x_tid"], {}).get("dtype"),
                    tensors.get(b_k["x_tid"], {}).get("dtype"),
                    tensors.get(cos_tid, {}).get("dtype"),
                    tensors.get(sin_tid, {}).get("dtype"),
                ],
                "output_dtypes": [
                    tensors.get(q_out_tid, {}).get("dtype"),
                    tensors.get(k_out_tid, {}).get("dtype"),
                ],
                "device": ops_metadata[b_q["add_uid"]].get("device"),
                "attributes": {
                    "args": [
                        {"type": "tensor", "tensor_id": b_q["x_tid"]},
                        {"type": "tensor", "tensor_id": b_k["x_tid"]},
                        {"type": "tensor", "tensor_id": cos_tid},
                        {"type": "tensor", "tensor_id": sin_tid},
                    ],
                    "kwargs": {},
                },
            }

            to_drop.update(chain_all)
            insert_before[anchor].append(new_uid)
            new_ops_by_uid[new_uid] = fused_op

        if not new_ops_by_uid:
            return

        for _uid, fused_op in new_ops_by_uid.items():
            ops_metadata[_uid] = fused_op

        # Rebuild the order: drop the fused chain ops, and insert each fused op
        # immediately before the earliest consumer of its outputs (the window
        # check above guarantees all its inputs are already produced by then).
        new_order: List[str] = []
        for uid in execution_order:
            for _fu in insert_before.get(uid, ()):
                new_order.append(_fu)
            if uid in to_drop:
                continue
            new_order.append(uid)
        execution_order.clear()
        execution_order.extend(new_order)

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

        # Step 1b: Dead outputs — slots produced but never read by any op
        # have no entry above and were NEVER killed, surviving in the arena
        # for the whole pass (R30 mirror of the other engines' dead-output
        # rules; the CogVideoX VAE all-at-once decode accumulates ~27 GB of
        # never-consumed conv-cache clones at full pixel resolution, and the
        # same class plausibly contributes to the long-standing triton
        # live-watermark gap vs compiled). Their last use is the producing op.
        for op_idx, op_uid in enumerate(exec_order):
            op_data = ops_meta.get(op_uid)
            if op_data is None:
                continue
            for tid in op_data.get("output_tensor_ids", []):
                s = self._tid_to_slot.get(tid)
                if s is not None and s not in slot_last_use:
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

        # P-TRITON-LIVE-WATERMARK-AUDIT L2: persist slot_last_use on
        # self so the OOM dump in `_run_single_device` can compare each
        # surviving slot's last_use against the current op_idx and
        # decide whether the liveness analysis missed a kill (last_use
        # <= op_idx) or the tensor is legitimately needed downstream
        # (last_use > op_idx).
        self._slot_last_use = dict(slot_last_use)
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

        # MoE fused op — custom compilation with arena-based weight access
        if op_type == "custom::moe_fused":
            return self._compile_moe_fused_op(op_uid, op_data, kill_slots)

        # Op_uid match (op-level tiling) wins over op_type match (KV cache).
        # Phase 1 — interceptors are wrapped by DtypeEngine UNLESS they
        # declare `self_manages_dtype = True` on the callable (opt-in
        # cleanup, replaces the previous implicit bypass). Documented
        # interceptors that self-manage:
        #   - TritonAttentionInterceptor.intercept (kv_cache.py):
        #     Flash Attention works in fp16/bf16, casts internally
        #   - fused_upsample_conv2d / _tiled_conv2d_spatial_nbx /
        #     _tiled_rms_norm_nbx (op-level tiling): delegate to
        #     conv2d_wrapper which is itself self-managed
        if op_uid in self._op_uid_interceptors:
            func = self._op_uid_interceptors[op_uid]
            if not getattr(func, 'self_manages_dtype', getattr(getattr(func, '__func__', None), 'self_manages_dtype', False)):
                bare_name = op_type.split("::")[-1] if "::" in op_type else op_type
                func = self._dtype_engine.wrap_op(bare_name, func)
        elif op_type in self._op_interceptors:
            func = self._op_interceptors[op_type]
            if not getattr(func, 'self_manages_dtype', getattr(getattr(func, '__func__', None), 'self_manages_dtype', False)):
                bare_name = op_type.split("::")[-1] if "::" in op_type else op_type
                func = self._dtype_engine.wrap_op(bare_name, func)
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

        # Extract input slots for cross-device tracking
        all_slots = self._extract_all_input_slots(compiled_args, compiled_kwargs)
        weight_slots = self._extract_weight_slots(op_data)

        return CompiledOp(
            op_uid=op_uid, op_type=op_type, func=func,
            args_resolver=args_resolver,
            kwargs_resolver=kwargs_resolver,
            output_slots=tuple(output_slots),
            kill_slots=kill_slots,
            weight_input_slots=tuple(weight_slots),
            all_input_slots=tuple(all_slots),
        )

    def _compile_moe_fused_op(self, op_uid: str, op_data: dict,
                              kill_slots: Tuple[int, ...]) -> CompiledOp:
        """Compile fused MoE dispatch — delegates to triton/moe.py."""
        from .moe import execute_moe_fused as _moe_exec

        attrs = op_data.get("attributes", {})
        output_tids = op_data.get("output_tensor_ids", [])

        gate_scores_tid = attrs["gate_scores_tid"]
        hidden_states_tid = attrs["hidden_states_tid"]
        gate_weight_ids = attrs["expert_gate_weight_ids"]
        up_weight_ids = attrs["expert_up_weight_ids"]
        down_weight_ids = attrs["expert_down_weight_ids"]
        top_k = attrs["top_k"]
        num_experts = attrs["num_experts"]
        norm_topk_prob = attrs.get("norm_topk_prob", True)

        # Resolve arena slots (compile-time)
        gate_scores_slot = self._tid_to_slot[gate_scores_tid]
        hidden_states_slot = self._tid_to_slot[hidden_states_tid]

        gate_w_slots = []
        up_w_slots = []
        down_w_slots = []
        for i in range(num_experts):
            gs = self._tid_to_slot.get(gate_weight_ids[i])
            us = self._tid_to_slot.get(up_weight_ids[i])
            ds = self._tid_to_slot.get(down_weight_ids[i])
            if gs is None or us is None or ds is None:
                raise RuntimeError(
                    f"[MoE Triton] Missing weight slot for expert {i} in {op_uid}")
            gate_w_slots.append(gs)
            up_w_slots.append(us)
            down_w_slots.append(ds)

        # Freeze for closure
        _gs_slot = gate_scores_slot
        _hs_slot = hidden_states_slot
        _gw = tuple(gate_w_slots)
        _uw = tuple(up_w_slots)
        _dw = tuple(down_w_slots)
        _k = top_k
        _ne = num_experts
        _norm = norm_topk_prob

        _cache_key = f"triton_compiled_{op_uid}"

        def moe_fused_dispatch(arena):
            return _moe_exec(
                gate_scores=arena[_gs_slot],
                hidden_states=arena[_hs_slot],
                gate_weights=[arena[s] for s in _gw],
                up_weights=[arena[s] for s in _uw],
                down_weights=[arena[s] for s in _dw],
                top_k=_k, num_experts=_ne, norm_topk_prob=_norm,
                cache_key=_cache_key,
            )

        # Output slots
        output_slots = []
        for tid in output_tids:
            if tid not in self._tid_to_slot:
                s = len(self._tid_to_slot)
                self._tid_to_slot[tid] = s
            output_slots.append(self._tid_to_slot[tid])

        def args_resolver(arena):
            return [arena]

        def kwargs_resolver(_arena):
            return {}

        def func_wrapper(arena):
            return moe_fused_dispatch(arena)

        return CompiledOp(
            op_uid=op_uid, op_type="custom::moe_fused",
            func=func_wrapper,
            args_resolver=args_resolver,
            kwargs_resolver=kwargs_resolver,
            output_slots=tuple(output_slots),
            kill_slots=kill_slots,
            weight_input_slots=tuple(list(_gw) + list(_uw) + list(_dw)),
            all_input_slots=tuple([_gs_slot, _hs_slot] + list(_gw) + list(_uw) + list(_dw)),
        )

    # ========================================================================
    # ========================================================================
    # INPUT SLOT EXTRACTION — for cross-device detection
    # ========================================================================

    def _extract_all_input_slots(self, compiled_args, compiled_kwargs):
        """Extract all TensorSlot indices from compiled args."""
        slots = []
        def _scan(item):
            if isinstance(item, TensorSlot):
                slots.append(item.slot)
            elif isinstance(item, ListArg):
                for sub in item.items:
                    _scan(sub)
        for a in compiled_args:
            _scan(a)
        for v in compiled_kwargs.values():
            _scan(v)
        return slots

    def _extract_weight_slots(self, op_data):
        """Extract weight/buffer slots from op_data args."""
        slots = []
        raw_args = op_data.get("attributes", {}).get("args", [])
        for arg in raw_args:
            if isinstance(arg, dict) and arg.get("type") in ("tensor", "tensor_ref"):
                tid = arg.get("tensor_id", "")
                if tid.startswith("param::") or tid.startswith("buffer::"):
                    slot = self._tid_to_slot.get(tid)
                    if slot is not None:
                        slots.append(slot)
        return slots

    # ========================================================================
    # CROSS-DEVICE — ported from compiled_sequence.compute_op_devices
    # ========================================================================

    def compute_op_devices(self):
        """Ported from compiled_sequence.py:2430. Sets op.device_idx and
        op.needs_transfer from weight placement in the arena.

        Zero3 (CPU offload) interaction: CPU NBXTensors have
        `_device_idx=0` (default placeholder) which matches cuda:0's
        index, so a naive device_idx-only set-intersection would
        classify everything as single-device and skip the multi-device
        hot loop entirely — callback never fires, slow path never
        kicks in. We also collect each tensor's `_device` type string
        and force multi-device classification if any weight is on CPU
        while the exec device is on CUDA (or vice versa).
        """
        arena = self._arena
        devices_seen = set()
        has_cpu_weight = False

        # Phase 1: assign device to weighted ops. Scan ALL weight slots (NOT
        # just the first): under weight_sharding an op can read weights on
        # DIFFERENT devices — a Linear whose weight matrix is a round-robin shard
        # on cuda:3 while its bias landed on cuda:2. Stopping at the first slot
        # recorded only one device (and could pick the small bias), so
        # devices_seen was incomplete -> _is_multi_device wrong -> the op ran
        # single-device and crashed reading the other shard ("Pointer argument
        # cannot be accessed from Triton" on addmm's mat2). Run the op on its
        # LARGEST weight's device (the compute matrix, not the bias) and record
        # every weight device. R30 mirror of the compiled compute_op_devices fix.
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            best_dev = None
            best_numel = -1
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, '_device_idx'):
                    devices_seen.add(tensor._device_idx)
                    if getattr(tensor, '_device', 'cuda') == 'cpu':
                        has_cpu_weight = True
                    _numel = getattr(tensor, '_numel', None)
                    if _numel is None:
                        _numel = tensor.numel() if hasattr(tensor, 'numel') else 0
                    if _numel > best_numel:
                        best_numel = _numel
                        best_dev = tensor._device_idx
            if best_dev is not None:
                op.device_idx = best_dev

        devices_seen.add(self.device_idx)
        # Multi-device iff we have more than one distinct device idx OR
        # at least one CPU-backed weight (the CPU/GPU split that zero3
        # introduces is invisible to device_idx alone).
        self._is_multi_device = len(devices_seen) > 1 or has_cpu_weight
        if not self._is_multi_device:
            return

        # Phase 2+4: propagate current_activation_device, flag needs_transfer
        slot_device: Dict[int, int] = {}
        for op in self._ops:
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, '_device_idx'):
                    slot_device[ws] = tensor._device_idx

        # Runtime-bound INPUT slots (flow globals) occupy the contiguous range
        # [num_weights, num_weights + num_inputs). Their device is unknowable
        # at compile time, so any op consuming one takes the slow path in a
        # multi-device sequence. R30 mirror of the compiled Phase-4 fix (the
        # DETTE D1 op-input co-location gap, first real trigger: Wan2.2
        # dual-denoiser prologue mul(timestep, freq) across GPUs, 2026-07-03).
        input_slot_lo = self._num_weights
        input_slot_hi = self._num_weights + self._num_inputs
        current_activation_device = None
        for op in self._ops:
            if op.device_idx is not None:
                if current_activation_device is None:
                    op.needs_transfer = True
                elif op.device_idx != current_activation_device:
                    op.needs_transfer = True
                current_activation_device = op.device_idx
            else:
                op.device_idx = current_activation_device

            if op.all_input_slots:
                for s in op.all_input_slots:
                    if input_slot_lo <= s < input_slot_hi:
                        op.needs_transfer = True
                        break
                    if current_activation_device is not None:
                        src_dev = slot_device.get(s)
                        if src_dev is not None and src_dev != current_activation_device:
                            op.needs_transfer = True
                            break

            out_dev = op.device_idx or current_activation_device or self.device_idx
            if out_dev is not None:
                for s in op.output_slots:
                    slot_device[s] = out_dev

    # ========================================================================
    # ZERO3 PARITY API — mirrors the methods added to CompiledSequence in
    # commit ea90d66 so the correctness-only zero3 priming (and future
    # pipelining) works in triton mode. Semantics are exactly equivalent
    # on the observable side (arena slot swap, op.device / op.needs_transfer
    # flag updates, block partition by transformer layer). Zero torch
    # throughout — all manipulations go through NBXTensor.
    # ========================================================================

    def rebind_partial(self, partial_map: Dict[str, NBXTensor]) -> List[int]:
        """Replace a subset of weights in the arena without touching the rest.

        Mirror of CompiledSequence.rebind_partial. Used by zero3 to swap
        weights between their CPU home and a freshly-allocated GPU copy
        during block-wise execution. Honors the same _pretranspose_weights
        contract as bind_weights — 2-D weights whose aten::t op was
        eliminated at compile time get a .t() view before being placed in
        the arena.

        Args:
            partial_map: tensor_id → NBXTensor. Tensor IDs not in
                self._tid_to_slot are silently skipped.

        Returns:
            List of slot indices that were modified.
        """
        assert self._arena is not None, "compile() must be called before rebind_partial()"
        modified: List[int] = []
        for tensor_id, tensor in partial_map.items():
            slot = self._tid_to_slot.get(tensor_id)
            if slot is None:
                continue
            if tensor_id in self._pretranspose_weights and tensor.ndim == 2:
                tensor = tensor.t()
            self._arena[slot] = tensor
            modified.append(slot)
        return modified

    def recompute_op_devices_for_slots(self, modified_slots: List[int]) -> None:
        """Patch per-op device flags after a partial arena rebind.

        Mirror of CompiledSequence.recompute_op_devices_for_slots.
        Revisits only the ops that read the modified slots via their
        weight_input_slots, rederives op.device_idx from the current
        arena contents, and flips op.needs_transfer based on whether
        the weight lives on the execution device.

        In triton mode, CPU-backed weights (zero3) have _device='cpu'
        and _device_idx=0 which is indistinguishable from "on cuda:0"
        by device_idx alone, so we also check _device type. If the
        weight is CPU we force op.device_idx to the exec device and
        mark needs_transfer=True so the slow path does the H2D copy.
        """
        assert self._arena is not None, (
            "compile() must be called before recompute_op_devices_for_slots()")
        if not modified_slots:
            return
        arena = self._arena
        touched = set(modified_slots)
        exec_dev = self.device_idx
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            if not any(ws in touched for ws in op.weight_input_slots):
                continue
            new_dev = None
            is_cpu = False
            for ws in op.weight_input_slots:
                tensor = arena[ws]
                if tensor is not None and hasattr(tensor, '_device_idx'):
                    new_dev = tensor._device_idx
                    # NBXTensor has explicit _device; we must not treat
                    # a CPU tensor with _device_idx=0 as "on cuda:0".
                    is_cpu = getattr(tensor, '_device', 'cuda') == 'cpu'
                    break
            if new_dev is not None:
                if is_cpu:
                    # Weight is CPU — compute on exec_dev, transfer
                    # per-op via _run_multi_device's slow path.
                    op.device_idx = exec_dev
                    op.needs_transfer = True
                else:
                    op.device_idx = new_dev
                    op.needs_transfer = (new_dev != exec_dev)

    def override_weightless_op_devices(self, device_idx: int) -> None:
        """Force op.device_idx = device_idx for every op without weight inputs.

        Mirror of CompiledSequence.override_weightless_op_devices.
        Tensor-creation ops (arange, scalar_tensor, full, attn-mask
        casts) inherit device from the activation-device chain built by
        compute_op_devices. For zero3 that chain is wrong — weighted
        ops "live" on CPU in the device graph but compute must happen
        on the exec GPU. This override sets the right device for
        weightless ops directly.
        """
        assert self._arena is not None, (
            "compile() must be called before override_weightless_op_devices()")
        for op in self._ops:
            if not op.weight_input_slots:
                op.device_idx = device_idx
                op.needs_transfer = False

    def mark_cpu_weighted_ops_for_transfer(self, exec_device_idx: int) -> int:
        """Zero3 correctness: flag every CPU-weighted op to go slow-path.

        Mirror of CompiledSequence.mark_cpu_weighted_ops_for_transfer.
        compute_op_devices's Phase 4 only marks the first weighted op
        as needs_transfer=True (its scan was designed for FGP device
        transitions). For zero3 every CPU-weighted op needs the slow
        path so the per-op H2D transfer fires.

        Returns the count of ops flipped.
        """
        assert self._arena is not None, (
            "compile() must be called before mark_cpu_weighted_ops_for_transfer()")
        flipped = 0
        for op in self._ops:
            if not op.weight_input_slots:
                continue
            for ws in op.weight_input_slots:
                t = self._arena[ws]
                # CPU NBXTensor — _device attribute is the truth source;
                # _device_idx alone is ambiguous (CPU tensors report 0).
                if t is not None and getattr(t, '_device', None) == 'cpu':
                    op.device_idx = exec_device_idx
                    op.needs_transfer = True
                    flipped += 1
                    break
        return flipped

    def materialize_slots_depending_on(self, weight_slot_ids) -> int:
        """Materialize every intermediate slot whose tensor aliases one of
        the given weight slots, so evicting those weights is safe.

        Mirror of CompiledSequence.materialize_slots_depending_on.

        Context: under zero3 pipelining, a block's weights may be referenced
        by arena intermediates via `_base` (views created by compute ops
        that produce sliced/transposed projections of a weight). Dropping
        the weight before these intermediates are consumed would leave the
        views pointing at freed GPU memory. Calling this primitive before
        the evict breaks the alias by copying each dependent intermediate
        into fresh storage via `.contiguous()`, then replacing the slot.

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

        # Collect the set of root-tensor identities for the weights being
        # evicted. NBXTensor.base is flattened to root on construction so
        # a single-level check is sufficient, but we walk defensively.
        forbidden_roots = set()
        for ws in weight_slots:
            t = arena[ws]
            if t is None:
                continue
            root = getattr(t, '_base', None) or t
            forbidden_roots.add(id(root))

        if not forbidden_roots:
            return 0

        materialized = 0
        for slot_idx in range(len(arena)):
            if slot_idx in weight_slots:
                continue
            t = arena[slot_idx]
            if t is None:
                continue
            # Skip anything that doesn't alias another tensor's storage.
            base = getattr(t, '_base', None)
            if base is None:
                continue
            depends = False
            node = base
            while node is not None:
                if id(node) in forbidden_roots:
                    depends = True
                    break
                node = getattr(node, '_base', None)
            if not depends:
                continue
            # Materialize: copy into fresh storage, break the alias.
            try:
                arena[slot_idx] = t.contiguous()
            except Exception:
                # Defensive: if contiguous() fails (e.g. CPU NBXTensor
                # without a matching path), drop the slot so the evict
                # can proceed. Re-run of the graph will refill the slot.
                arena[slot_idx] = None
            materialized += 1
        return materialized
        return flipped

    def get_op_blocks(self) -> Dict[int, Dict[str, Any]]:
        """Group the compiled op list by transformer block.

        Mirror of CompiledSequence.get_op_blocks. Uses _BLOCK_RE to
        extract the block index from weight tensor names. Weightless
        ops inherit the block of their predecessor. Non-block weights
        (embeddings, final norm, lm_head) go into block -1.

        Result is cached on self._op_blocks_cache — immutable after
        compile().
        """
        cache = getattr(self, '_op_blocks_cache', None)
        if cache is not None:
            return cache

        # Invert _tid_to_slot for fast slot → tid lookup.
        slot_to_tid: Dict[int, str] = {
            slot: tid for tid, slot in self._tid_to_slot.items()}

        blocks: Dict[int, Dict[str, Any]] = {}
        last_assigned: int = -1
        for op_idx, op in enumerate(self._ops):
            block_idx: Optional[int] = None
            op_weight_tids: List[str] = []
            for ws in op.weight_input_slots:
                tid = slot_to_tid.get(ws)
                if not tid:
                    continue
                op_weight_tids.append(tid)
                if block_idx is None:
                    if tid.startswith("param::"):
                        name = tid[7:]
                    elif tid.startswith("buffer::"):
                        name = tid[8:]
                    else:
                        name = tid
                    m = _BLOCK_RE.search(name)
                    block_idx = int(m.group(1)) if m else -1

            if block_idx is None:
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

        # Dedupe weight_tensor_ids per block.
        for entry in blocks.values():
            seen: Dict[str, None] = {}
            for tid in entry['weight_tensor_ids']:
                seen.setdefault(tid, None)
            entry['weight_tensor_ids'] = list(seen.keys())

        self._op_blocks_cache = blocks
        return blocks

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
        # Narrow fp64/complex128 to the triton-supported fp32/complex64. The
        # NeuroBrix triton kernels are fp32-max (no native fp64 on V100, and the
        # elementwise/index kernels read at fp32 stride); an explicit fp64 cast
        # in the graph (e.g. the Wan rotary-embedding chain, which the vendor
        # runs in float64/complex128 for precision) must be honoured at fp32
        # precision, which is numerically ample for RoPE. complex128 → complex64
        # keeps the interleaved-pair invariant the complex kernels rely on.
        if parsed == NBXDtype.float64:
            return NBXDtype.float32
        if parsed == NBXDtype.complex128:
            return NBXDtype.complex64
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

    # ========================================================================
    # SEQ-DEPENDENT CONSTANTS — ported from compiled_sequence
    # ========================================================================

    def _identify_seq_dependent_constants(self, tensors: dict):
        """Tag constant_T_* tensors with a dim matching trace seq_len.

        RoPE cos/sin are captured at trace time with shape [1, trace_seq_len, dim].
        At runtime they must be narrowed to [1, runtime_seq_len, dim].
        """
        sym_ctx = self.dag.get("symbolic_context", {})
        symbols = sym_ctx.get("symbols", {})

        seq_info = {}
        for sid, sinfo in symbols.items():
            if sinfo.get("name") == "seq_len":
                tv = sinfo.get("trace_value")
                if tv is not None:
                    seq_info[sid] = tv
        if not seq_info:
            return

        sym_id, trace_seq_len = next(iter(seq_info.items()))

        for tid in self._weight_ids:
            tdata = tensors.get(tid, {})
            wname = tdata.get("weight_name", "")
            if not wname.startswith("constant_T_"):
                continue
            shape = tdata.get("shape", [])
            for axis, dim in enumerate(shape):
                if dim == trace_seq_len:
                    slot = self._tid_to_slot.get(tid)
                    if slot is not None:
                        self._seq_dependent_constants.append(
                            (slot, axis, sym_id, trace_seq_len))
                    break

    def update_seq_dependent_constants(self):
        """Narrow seq-dependent constants to runtime seq_len.

        Called after bind_symbols() so the symbol resolver has the actual value.
        Always narrows from the ORIGINAL full-size constant, not a previous narrow.
        """
        if not self._seq_dependent_constants or self._symbol_resolver is None:
            return

        arena = self._arena
        for slot, axis, sym_id, trace_val in self._seq_dependent_constants:
            runtime_val = self._symbol_resolver.get(sym_id)
            if runtime_val is None or runtime_val == trace_val:
                # Restore original if previously narrowed
                if slot in self._seq_constant_originals:
                    arena[slot] = self._seq_constant_originals[slot]
                continue

            # Save original on first encounter
            if slot not in self._seq_constant_originals:
                current = arena[slot]
                if current is not None:
                    self._seq_constant_originals[slot] = current

            original = self._seq_constant_originals.get(slot)
            if original is None:
                continue

            if runtime_val <= original.shape[axis]:
                arena[slot] = original.narrow(axis, 0, runtime_val)

    def bind_weights(self, weights: Dict[str, NBXTensor]):
        """Bind weight tensors to arena slots, then compute per-op devices."""
        # A fresh weight bind invalidates every cached seq-dependent-constant
        # original: they are NBXTensor views into the PREVIOUS bind's
        # component arena, which is explicitly freed when weights unload
        # (zero3/lazy streaming reloads per request — "setup persists,
        # weights do not"). Re-narrowing from a stale original reads freed
        # arena memory. Mirror of the compiled bind_weights (R30).
        self._seq_constant_originals.clear()
        for tid in self._weight_ids:
            tdata = self.dag.get("tensors", {}).get(tid, {})
            wname = tdata.get("weight_name", "")
            tensor = weights.get(wname) if wname else None
            if tensor is None and wname:
                # Trailing-suffix fallback (mirror of compiled graph_executor
                # _resolve_weight and the sequential resolver, R30): a build can
                # store a weight under a SHORTER name than the graph param when
                # an `encoder.`/`model.` prefix is applied inconsistently across a
                # component (Wan UMT5: graph param `encoder.token_embed.weight` vs
                # `token_embed.weight` in the .nbx). Without this the embed bound
                # to nothing and the whole encoder NOP-propagated to empty -> the
                # triton gather returned no outputs at all (text_encoder produced
                # []), failing CFG with a missing last_hidden_state. Strip leading
                # prefix segments, longest-suffix first; the trailing suffix is
                # unique per tensor so this cannot mis-bind.
                _parts = wname.split('.')
                for _i in range(1, len(_parts)):
                    tensor = weights.get('.'.join(_parts[_i:]))
                    if tensor is not None:
                        break
            if tensor is not None:
                # Pre-transpose weights whose aten::t was eliminated.
                # .t() is a stride-only view (no copy) on NBXTensor.
                if tid in self._pretranspose_weights and tensor.ndim == 2:
                    tensor = tensor.t()
                self._arena[self._tid_to_slot[tid]] = tensor
        self.compute_op_devices()

    def bind_inputs(self, inputs: Dict[str, NBXTensor]):
        """Bind input tensors to arena slots.

        Casts each input to its expected runtime dtype via the engine
        (mirrors PyTorch DtypeEngine path at component entry). Graph
        floating-point dtype → compute_dtype; non-floating → preserved.
        Data-driven: graph metadata + compute_dtype.
        """
        graph_tensors = self.dag.get("tensors", {})
        cast = self._dtype_engine.cast_runtime_inputs(inputs, graph_tensors)
        for tid, tensor in cast.items():
            slot = self._tid_to_slot.get(tid)
            if slot is not None:
                # Inputs must live on THIS component's device. When the component
                # is placed on a non-default GPU (e.g. a 31 GB transformer placed
                # on cuda:2 while the upstream noise/embeddings were created on the
                # default cuda:0), the input tensor is on the wrong device and the
                # first op's Triton kernel cannot read it from the component's CUDA
                # context (surfaces as "Pointer argument cannot be accessed from
                # Triton (cpu tensor?)" at the patch-embed conv). Move it D2D to the
                # component device. Inert when already co-located (single-GPU on the
                # default device) and for CPU sources (zero3 handles those). R30
                # mirror of the compiled cross-device input handling.
                if (hasattr(tensor, "_device_idx")
                        and tensor._device_idx != self.device_idx
                        and getattr(tensor, "_device", "cuda") != "cpu"):
                    from neurobrix.triton.device_transfer import transfer_tensor
                    tensor = transfer_tensor(tensor, self.device_idx)
                self._arena[slot] = tensor

    def bind_symbols(self, inputs: dict):
        """Bind symbolic shapes from actual input tensors."""
        if self._symbol_resolver:
            self._symbol_resolver.bind_from_inputs(
                inputs, self._input_ids, self.dag.get("tensors", {}))

    def _ensure_oom_reclaim_hook(self) -> None:
        """Register (once per instance) a last-chance OOM reclaimer that
        drains this sequence's CURRENT deferred dead-tensor queue
        (`self._oom_deferred_queue`, rebound at the start of each run by
        both hot loops). Called by DeviceAllocator.malloc_cuda ONLY after a
        device malloc fails — dead code on every run that never OOMs. The
        drain mirrors the in-loop drains exactly: device sync BEFORE the
        frees (UAF-safe, sync-before-free doctrine), then clear. The loop's
        local `_deferred_bytes` counter is deliberately left stale after a
        hook drain: it only OVERSTATES, so the next cliff check performs
        one drain of an already-empty queue (a wasted sync on the
        OOM-recovery path only) and resets — never an early free. Weakref-
        bound: a dead sequence's hook frees nothing and unregisters itself.
        P-TRITON-LIVE-SET: the drain policy's pressure gate is blind to the
        incoming request size (an 8 GiB ask with 7.4 GiB free trips
        nothing), so the allocator-side retry is the only chokepoint that
        sees both the request and the reclaimable queue.
        """
        if getattr(self, "_oom_hook_registered", False):
            return
        import weakref
        wself = weakref.ref(self)

        def _seq_oom_reclaim(dev_idx, nbytes_requested):
            seq = wself()
            if seq is None:
                DeviceAllocator.unregister_oom_reclaim_hook(_seq_oom_reclaim)
                return 0
            queue = getattr(seq, "_oom_deferred_queue", None)
            if not queue:
                return 0
            DeviceAllocator.sync_device()
            freed = 0
            for t in queue:
                freed += getattr(t, "_nbytes", 0) or 0
            queue.clear()
            return freed

        DeviceAllocator.register_oom_reclaim_hook(_seq_oom_reclaim)
        # Eager deregistration when this sequence is collected — the
        # weakref guard inside the hook stays as belt-and-braces, but the
        # registry entry itself must not accumulate across many sequence
        # lifetimes in a long-lived serving process (gardien MEDIUM,
        # 2026-07-11).
        weakref.finalize(
            self, DeviceAllocator.unregister_oom_reclaim_hook, _seq_oom_reclaim)
        self._oom_hook_registered = True

    def run(self, skip_kills: bool = False,
            pre_op_callback: Optional[Callable[[int, 'CompiledOp'], None]] = None):
        """Execute all ops. Zero overhead hot loop.

        Args:
            skip_kills: When True, skip kill_slots entirely (no cudaFree, no sync).
                Decode steps reuse the same arena slots every step, so intermediates
                are always overwritten before being read. No UAF possible.
            pre_op_callback: Optional (op_idx, op) -> None hook invoked
                before each op's args are resolved. Used by zero3 for
                its first-tick priming (mark_cpu_weighted_ops_for_transfer
                + override_weightless_op_devices) and by future block-
                wise pipelining to rebind weights at block boundaries.
                None → no overhead. Honored only by the multi-device
                path; single-device skips (zero3 is always multi-device-
                classified because CPU weights differ from the exec GPU).
        """
        # C2: drop the ensure_triton_device fast-path cache at run
        # entry so the line below performs the FULL runtime + Triton
        # sync exactly as it always did — this neutralises any device
        # change made outside NeuroBrix control between forward passes.
        # Inside the run the cache then legitimately skips redundant
        # per-launch switches (all in-run device changes go through
        # DeviceAllocator and maintain the cache invariant).
        invalidate_current_device_cache()
        DeviceAllocator.ensure_triton_device(self.device_idx)

        # Phase 1 — set per-component dtype context for self-managed wrappers
        # (conv2d_wrapper reads _NBX_COMPUTE_DTYPE for output dtype, the
        # opt-in cast-back wrap variant reads _NBX_ACTIVATIONS_FP16_SAFE
        # for rms_norm/div). Restore prior values after to keep nested or
        # multi-component runs correct. Single-threaded NeuroBrix runtime,
        # safe to use module-level state.
        from neurobrix.kernels import wrappers as _w
        _prev_dt = _w.get_compute_dtype()
        _prev_safe = _w.get_activations_fp16_safe()
        _w.set_compute_dtype(self._compute_dtype)
        _w.set_activations_fp16_safe(self._activations_fp16_safe)
        try:
            if self._is_multi_device:
                self._run_multi_device(skip_kills, pre_op_callback)
            else:
                self._run_single_device(skip_kills, pre_op_callback)
        finally:
            _w.set_compute_dtype(_prev_dt)
            _w.set_activations_fp16_safe(_prev_safe)

    def _maybe_trace_nan(self, op: 'CompiledOp', arena) -> None:
        """Scan op output(s) for Inf/NaN. Print the first offender on
        stderr with its (op_uid, op_type, shape, dtype) so the caller
        can localise the first op that introduced the NaN/Inf."""
        if not hasattr(self, "_trace_nan_seen"):
            self._trace_nan_seen = False
        if self._trace_nan_seen:
            return
        import sys as _sys_tn
        for s in op.output_slots:
            t = arena[s] if arena else None
            if t is None:
                continue
            try:
                from neurobrix.kernels.nbx_tensor import NBXDtype, DeviceAllocator
                import ctypes as _ct
                # Complex tensors: check finiteness on the real VIEW (interleaved
                # real/imag floats), not by casting the complex elements to a
                # single fp32 (which reads half the bytes -> false NaN/Inf). The
                # view_as_real path reads all real+imag components correctly.
                src = t.view_as_real() if t.is_complex() else t
                full = src if src._dtype == NBXDtype.float32 else src.to(NBXDtype.float32)
                full = full.contiguous()
                DeviceAllocator.set_device(full._device_idx)
                n = full.numel()
                if n == 0:
                    continue
                buf = (_ct.c_float * n)()
                DeviceAllocator.memcpy(_ct.addressof(buf), full.data_ptr(),
                                       n * 4, kind=2)
                import math as _math
                has_nan = any(_math.isnan(v) for v in buf)
                has_inf = any(_math.isinf(v) for v in buf)
                if has_nan or has_inf:
                    self._trace_nan_seen = True
                    flag = "NaN" if has_nan else "Inf"
                    inp_info = []
                    for si in op.all_input_slots[:4]:
                        ti = arena[si] if arena else None
                        if ti is not None:
                            inp_info.append(
                                f"in{si}=<{ti.dtype}, shape={list(ti.shape)}>")
                    print(
                        f"[NBX_TRITON_TRACE_NAN] FIRST {flag} at "
                        f"op_uid={op.op_uid} op_type={op.op_type} "
                        f"out=<{t.dtype}, shape={list(t.shape)}> "
                        f"{'; '.join(inp_info)}",
                        file=_sys_tn.stderr, flush=True)
                    return
            except Exception as e:
                print(f"[NBX_TRITON_TRACE_NAN] failed on {op.op_uid}: {e}",
                      file=_sys_tn.stderr, flush=True)
                return

    @staticmethod
    def nbx_tid_stats(tensor) -> dict:
        """Stats for one NBXTensor output for the NBX_DUMP_TIDS diagnostic:
        {shape, dtype, is_complex, head10, l2_norm, batch_norms}.

        Single source of truth shared by the compiled hot loop
        (_maybe_dump_tid below) and the triton-sequential op loop in
        graph_executor — same record schema in every engine so cross-engine
        diffs align. Complex tensors are read via the real view ([...,2]
        re/im) so l2/head cover BOTH components; memcpy'ing a complex buffer
        as N floats would read only the real parts (half the energy).
        """
        from neurobrix.kernels.nbx_tensor import NBXDtype, DeviceAllocator
        import ctypes as _ct
        # Zero-numel guard: an empty tensor (e.g. a VACE all-generate control
        # prefix new_zeros [1,0,1536]) has a 0-size dim. The reduction path below
        # would index into that 0-dim (`flat[0]`) and launch contiguous/cast
        # kernels over 0 elements → CUDA illegal address (error 700) that, async,
        # corrupts the context and only surfaces at a later op's malloc — making
        # the dump itself crash a run that computes fine without it. Empty
        # tensors carry no values to diff, so return a trivial record.
        if getattr(tensor, "_numel", None) == 0 or 0 in list(tensor.shape):
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "is_complex": bool(getattr(
                    tensor, "is_complex", lambda: False)()),
                "head10": [],
                "l2_norm": 0.0,
                "batch_norms": None,
            }
        # Multi-device: switch to the tensor's device before D2H memcpy.
        if hasattr(tensor, '_device_idx'):
            DeviceAllocator.set_device(tensor._device_idx)
        src = tensor.view_as_real() if getattr(
            tensor, "is_complex", lambda: False)() else tensor
        flat = src
        while flat.ndim > 1:
            flat = flat[0]
        # Cast to fp32 for comparable dumps across engines.
        if flat.dtype != NBXDtype.float32:
            flat = flat.to(NBXDtype.float32)
        flat = flat.contiguous()
        n = min(10, flat.numel())
        buf = (_ct.c_float * n)()
        DeviceAllocator.memcpy(_ct.addressof(buf), flat.data_ptr(),
                               n * 4, kind=2)
        head = list(buf)
        # L2 norm over the full (flattened) tensor.
        full = src
        if full.dtype != NBXDtype.float32:
            full = full.to(NBXDtype.float32)
        full = full.contiguous()
        N = full.numel()
        fbuf = (_ct.c_float * N)()
        DeviceAllocator.memcpy(_ct.addressof(fbuf), full.data_ptr(),
                               N * 4, kind=2)
        vals = list(fbuf)
        norm = (sum(v * v for v in vals)) ** 0.5
        _batch_norms = None
        _shp = list(tensor.shape)
        if len(_shp) >= 2 and _shp[0] in (2, 3) and N % _shp[0] == 0:
            _per = N // _shp[0]
            _batch_norms = [
                (sum(v * v for v in vals[bi * _per:(bi + 1) * _per])) ** 0.5
                for bi in range(_shp[0])]
        return {
            "shape": _shp,
            "dtype": str(tensor.dtype),
            "is_complex": bool(getattr(tensor, "is_complex", lambda: False)()),
            "head10": head,
            "l2_norm": norm,
            "batch_norms": _batch_norms,
        }

    def _maybe_dump_tid(self, op: 'CompiledOp', dump_path: str) -> None:
        """TEMP diagnostic: dump op output tensor to JSON if its tid matches.

        Gated by env var NBX_DUMP_TIDS_FILTER (optional comma-separated tid
        substrings). Only first-pass dumps are kept per tid (first decode
        step is enough for layer-by-layer comparison).
        """
        import os as _os_d, json as _json_d
        if not hasattr(self, "_dump_seen"):
            self._dump_seen = set()
            self._dump_records = []
        filter_env = _os_d.environ.get("NBX_DUMP_TIDS_FILTER", "")
        filters = [f for f in filter_env.split(",") if f] if filter_env else []
        # Invert slot → tid map lazily.
        if not hasattr(self, "_slot_to_tid"):
            self._slot_to_tid = {s: t for t, s in self._tid_to_slot.items()}
        for out_slot in op.output_slots:
            tid = self._slot_to_tid.get(out_slot, f"slot::{out_slot}")
            # Match filter against tid OR op_uid (op_uid lets us select by op,
            # e.g. all custom.rms_norm outputs across the network).
            if filters and not any(f in tid or f in op.op_uid for f in filters):
                continue
            if tid in self._dump_seen:
                continue
            self._dump_seen.add(tid)
            tensor = self._arena[out_slot] if self._arena else None
            if tensor is None:
                continue
            # NBXTensor → first 10 floats + norm + shape
            try:
                new_record = {
                    "component": self.dag.get("component_name", "?"),
                    "tid": tid,
                    "op_uid": op.op_uid,
                    "op_type": op.op_type,
                    **self.nbx_tid_stats(tensor),
                }
                self._dump_records.append(new_record)
                # JSONL append-mode write — O(1) IO per call vs the
                # earlier O(N²) read-then-append. Each record is one
                # JSON line; bit-diff readers consume line-by-line.
                with open(dump_path, "a") as f:
                    _json_d.dump({"engine": "triton",
                                  "record": new_record}, f)
                    f.write("\n")
            except Exception as e:
                print(f"[NBX_DUMP_TIDS] failed on {tid}: {e}", flush=True)

    def _maybe_fingerprint(self, op: 'CompiledOp', arena, path: str) -> None:
        """Run-to-run op-output fingerprint (P-TRITON-MOE-DETERMINISM-RESIDUAL,
        P-SANA differential methodology). Gated by env NBX_OP_FINGERPRINT=
        <jsonl-path>. Appends one line per op output:
        {op_idx, op_uid, op_type, shape, dtype, sha256(full raw bytes)}.
        Diff the JSONL across 3 runs: the FIRST op whose sha256 differs is
        the causal non-determinism site. Diagnostic only, same retained
        class as NBX_DUMP_TIDS / NBX_TRITON_TRACE_NAN."""
        import os as _os_f, json as _json_f, hashlib as _hl
        import ctypes as _ct_f
        if not hasattr(self, "_fp_idx"):
            self._fp_idx = 0
        # Optional global cap on number of fingerprinted op-outputs
        # (NBX_OP_FINGERPRINT_MAX). Lets a full-hash (CAP=0) run stay
        # cheap by stopping after the suspect region. Shared across all
        # TritonSequence instances via the class.
        _fp_max = int(_os_f.environ.get("NBX_OP_FINGERPRINT_MAX", "0"))
        if _fp_max:
            _cls = type(self)
            _g = getattr(_cls, "_fp_global", 0)
            if _g >= _fp_max:
                return
        if not hasattr(self, "_slot_to_tid"):
            self._slot_to_tid = {s: t for t, s in self._tid_to_slot.items()}
        try:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            recs = []
            for out_slot in op.output_slots:
                t = arena[out_slot] if arena else None
                self._fp_idx += 1
                _clsg = type(self)
                _clsg._fp_global = getattr(_clsg, "_fp_global", 0) + 1
                tid = self._slot_to_tid.get(out_slot, f"slot::{out_slot}")
                if t is None or not hasattr(t, "data_ptr"):
                    recs.append({"i": self._fp_idx, "op_uid": op.op_uid,
                                 "op_type": op.op_type, "tid": tid,
                                 "sha": "none"})
                    continue
                if hasattr(t, "_device_idx"):
                    DeviceAllocator.set_device(t._device_idx)
                c = t.contiguous()
                nb = c._nbytes
                # Byte-cap (default 8 KiB): non-deterministic-reduction
                # bit-noise is distributed across the tensor, so the
                # leading bytes reflect any divergence while keeping
                # 115k-op model traces feasible. NBX_OP_FINGERPRINT_CAP=0
                # disables the cap (full hash).
                cap = int(_os_f.environ.get("NBX_OP_FINGERPRINT_CAP", "8192"))
                hb = nb if cap == 0 else min(nb, cap)
                buf = (_ct_f.c_char * hb)()
                DeviceAllocator.memcpy(_ct_f.addressof(buf), c.data_ptr(),
                                       hb, kind=2)
                recs.append({
                    "i": self._fp_idx, "op_uid": op.op_uid,
                    "op_type": op.op_type, "tid": tid,
                    "shape": list(t.shape), "dtype": str(t.dtype),
                    "nbytes": nb, "hashed": hb,
                    "sha": _hl.sha256(bytes(buf)).hexdigest()})
            with open(path, "a") as f:
                for r in recs:
                    _json_f.dump(r, f)
                    f.write("\n")
        except Exception as e:
            print(f"[NBX_OP_FINGERPRINT] failed at {op.op_uid}: {e}",
                  flush=True)

    def _run_single_device(self, skip_kills: bool = False,
                            pre_op_callback: Optional[Callable[[int, 'CompiledOp'], None]] = None):
        """Single-device fast path — no device switching.

        Both output-slot overwrites and kill_slots defer their old tensors
        to a single list, released after a sync at the end of the run.
        Without this, NBXTensor.__del__ frees GPU memory while async kernels
        may still be reading from it.

        pre_op_callback is accepted for signature parity with the multi-
        device path but is not invoked here: zero3 is always classified
        multi-device (CPU weights), so this branch never runs under zero3.
        """
        arena = self._arena
        _deferred: List[NBXTensor] = []
        _deferred_bytes: int = 0
        (_drain_cliff, _drain_floor, _drain_reserve, _drain_interval,
         _drain_count_limit) = _resolve_drain_policy()
        _last_free_check = -(1 << 30)
        _drain_diag = os.environ.get("NBX_DEFERRED_DRAIN_DIAG") == "1"
        _drain_stats = [0, 0, 0] if _drain_diag else None  # [drains, cliff/count, pressure]

        # OOM last-chance reclaim: expose THIS run's deferred queue to the
        # allocator (drained+retried only after a failed device malloc).
        self._oom_deferred_queue = _deferred
        self._ensure_oom_reclaim_hook()

        # C1 (P-KERNEL-LAUNCH-HYGIENE): diagnostic env gates hoisted out
        # of the per-op hot loop — read ONCE per run. Same gate
        # semantics as the previous per-op reads (diagnostics are set
        # before launch; changing an env var mid-run no longer applies).
        _live_dump_env = os.environ.get("NBX_LIVE_DUMP_EVERY", "0")
        _live_dump_n = int(_live_dump_env) if _live_dump_env != "0" else 0
        _wm_path = os.environ.get("NBX_LIVE_WATERMARK_TRACE", "")
        _trace_nan_on = os.environ.get("NBX_TRITON_TRACE_NAN") == "1"
        _fp_path = os.environ.get("NBX_OP_FINGERPRINT", "")
        _dump_tids_env = os.environ.get("NBX_DUMP_TIDS")
        _trace_tids_env = os.environ.get("NBX_TRACE_TIDS", "")
        _trace_tids = (set(t.strip() for t in _trace_tids_env.split(","))
                       if _trace_tids_env else set())
        _gc_env = os.environ.get("NBX_FORCE_GC", "0")
        _gc_int = int(_gc_env) if _gc_env != "0" else 0

        # ===== TEMP PROFILING INSTRUMENTATION =====
        import time as _time
        _PROF = os.environ.get("NBX_TRITON_PROF") == "1"
        if _PROF:
            _MATMUL = {"aten::mm", "aten::bmm", "aten::addmm"}
            _SDPA = {"aten::scaled_dot_product_attention"}
            _META = {"aten::view", "aten::reshape", "aten::_unsafe_view",
                     "aten::transpose", "aten::t", "aten::permute",
                     "aten::unsqueeze", "aten::expand", "aten::contiguous",
                     "aten::slice", "aten::select", "aten::cat", "aten::split",
                     "aten::unbind", "aten::narrow", "aten::flatten",
                     "aten::squeeze"}
            _EMBED = {"aten::embedding", "aten::index"}
            _timings = {"matmul": 0.0, "sdpa": 0.0, "elem": 0.0,
                        "meta": 0.0, "embed": 0.0, "other": 0.0}
            _counts = {"matmul": 0, "sdpa": 0, "elem": 0,
                       "meta": 0, "embed": 0, "other": 0}
            # Per-op_type breakdown inside the `elem` bucket — lets us rank
            # fusion targets ("what dominates element-wise time?").
            _elem_timings: Dict[str, float] = {}
            _elem_counts: Dict[str, int] = {}
            if not hasattr(self, "_prof_call_idx"):
                self._prof_call_idx = 0
            self._prof_call_idx += 1
        # ===========================================

        for op_idx, op in enumerate(self._ops):
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            # NOP propagation (deactivated MoE paths).
            if args and args[0] is None:
                bare = op.op_type.split("::")[-1]
                if bare in _ACCUMULATOR_OPS:
                    result = args[0]
                else:
                    for s in op.output_slots:
                        old = arena[s]
                        if old is not None:
                            _deferred.append(old)
                            _deferred_bytes += old._nbytes
                        arena[s] = None
                    if not skip_kills:
                        for s in op.kill_slots:
                            old = arena[s]
                            if old is not None:
                                _deferred.append(old)
                                _deferred_bytes += old._nbytes
                            arena[s] = None
                    # Drop the loop variable so it does not retain the
                    # last killed tensor across subsequent outer-loop
                    # iterations. See P-SANA-4KPX-RUNTIME 2026-05-06.
                    old = None  # noqa: F841
                    continue
            else:
                if _PROF:
                    DeviceAllocator.sync_device()
                    _t0 = _time.perf_counter()
                # Periodic live tracking — every NBX_LIVE_DUMP_EVERY ops,
                # log the NBX live_tracked counter so we can scrub a log to
                # find when leakage begins. Cheap: just a counter read.
                # Gate hoisted to run start (C1).
                if _live_dump_n > 0 and op_idx % _live_dump_n == 0:
                    from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA_p
                    _live = _DA_p._cuda_live_bytes.get(_DA_p.get_device(), 0)
                    print(f"[LIVE_TRACK op_idx={op_idx} {op.op_uid}] live={_live/1024/1024:.0f}MB",
                          flush=True)
                # P-TRITON-LIVE-WATERMARK-AUDIT 2026-05-14 L1: extended
                # periodic trace. NBX_LIVE_WATERMARK_TRACE=path/to/file.jsonl
                # samples every NBX_LIVE_WATERMARK_EVERY ops (default 50)
                # the FULL memory picture: nbx_live, nbx_pool_cached,
                # driver_free, driver_total, and computes the untracked
                # gap (= driver_used - nbx_live - nbx_pool_cached). This
                # gap is what we need to identify and reduce on 16g
                # triton where conv::64 OOMs at driver_free=208MB while
                # nbx_live=12898MB on a 16151MB GPU.
                # _wm_path hoisted to run start (C1).
                if _wm_path:
                    _wm_every = int(os.environ.get(
                        "NBX_LIVE_WATERMARK_EVERY", "50"))
                    if _wm_every > 0 and op_idx % _wm_every == 0:
                        try:
                            import ctypes as _ct_wm
                            import json as _json_wm
                            from neurobrix.kernels.nbx_tensor import (
                                DeviceAllocator as _DA_wm,
                                _gpu_runtime as _gpu_rt_wm,
                                _active_backend as _bk_fn_wm,
                            )
                            _rt_wm = _gpu_rt_wm()
                            _bk_wm = _bk_fn_wm()
                            _dev_wm = _DA_wm.get_device()
                            _nbx_live = _DA_wm._cuda_live_bytes.get(_dev_wm, 0)
                            _per_dev_wm = _DA_wm._pool_free.get(_dev_wm, {})
                            _pool_cached = 0
                            for _sz_wm, _ptrs_wm in _per_dev_wm.items():
                                _pool_cached += _sz_wm * len(_ptrs_wm)
                            _free_b_wm = _ct_wm.c_size_t()
                            _total_b_wm = _ct_wm.c_size_t()
                            _mgi_name = _bk_wm.get(
                                "mem_get_info", "cudaMemGetInfo")
                            if hasattr(_rt_wm, _mgi_name):
                                getattr(_rt_wm, _mgi_name)(
                                    _ct_wm.byref(_free_b_wm),
                                    _ct_wm.byref(_total_b_wm))
                                _drv_free = _free_b_wm.value
                                _drv_total = _total_b_wm.value
                            else:
                                _drv_free = _drv_total = 0
                            _drv_used = _drv_total - _drv_free
                            _untracked = (
                                _drv_used - _nbx_live - _pool_cached)
                            _rec = {
                                "op_idx": op_idx,
                                "op_uid": op.op_uid,
                                "op_type": getattr(op, "op_type", ""),
                                "nbx_live_mb": _nbx_live / 1024 / 1024,
                                "nbx_pool_cached_mb":
                                    _pool_cached / 1024 / 1024,
                                "driver_free_mb": _drv_free / 1024 / 1024,
                                "driver_used_mb": _drv_used / 1024 / 1024,
                                "driver_total_mb":
                                    _drv_total / 1024 / 1024,
                                "untracked_mb": _untracked / 1024 / 1024,
                            }
                            with open(_wm_path, "a") as _wm_f:
                                _wm_f.write(_json_wm.dumps(_rec) + "\n")
                        except Exception as _wm_e:
                            print(f"[LIVE_WATERMARK_TRACE error] {_wm_e}",
                                  flush=True)
                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    import os as _os_oom
                    if _os_oom.environ.get("NBX_LIVE_DUMP_ON_OOM") == "1" and "malloc failed" in str(e):
                        try:
                            w_n = self._num_weights
                            i_n = self._num_inputs
                            n_w = n_i = n_m = n_other = n_none = 0
                            b_w = b_i = b_m = 0
                            type_counts: Dict[str, int] = {}
                            for s, t in enumerate(arena):
                                if t is None:
                                    n_none += 1
                                    continue
                                tn = type(t).__name__
                                type_counts[tn] = type_counts.get(tn, 0) + 1
                                if not hasattr(t, '_nbytes'):
                                    n_other += 1
                                    continue
                                nb = t._nbytes
                                if s < w_n:
                                    n_w += 1; b_w += nb
                                elif s < w_n + i_n:
                                    n_i += 1; b_i += nb
                                else:
                                    n_m += 1; b_m += nb
                            print(f"[LIVE_DUMP at {op.op_uid}] arena_size={len(arena)} "
                                  f"weights={n_w}/{b_w/1024/1024:.0f}MB "
                                  f"inputs={n_i}/{b_i/1024/1024:.0f}MB "
                                  f"intermediates={n_m}/{b_m/1024/1024:.0f}MB "
                                  f"none_slots={n_none} other_slots={n_other} "
                                  f"types={type_counts} "
                                  f"deferred_queue={len(_deferred)}/{_deferred_bytes/1024/1024:.0f}MB",
                                  flush=True)
                            # P-TRITON-LIVE-WATERMARK-AUDIT L2: per-intermediate
                            # detail. For every non-None, non-weight, non-input
                            # slot at OOM, log: slot id, nbytes, shape, tid,
                            # producing op_uid, slot_last_use (from the liveness
                            # pass cached on self), and current op_idx. This
                            # is the dispositive fact: if last_use <= op_idx
                            # the liveness analysis missed a kill; if
                            # last_use > op_idx the tensor is legitimately
                            # needed downstream.
                            try:
                                if not hasattr(self, '_slot_to_tid'):
                                    self._slot_to_tid = {
                                        sl: t for t, sl in self._tid_to_slot.items()}
                                # Build slot -> producing op_uid map
                                if not hasattr(self, '_slot_to_producer'):
                                    _p_map: Dict[int, str] = {}
                                    for _o_idx, _o in enumerate(self._ops):
                                        for _s_p in _o.output_slots:
                                            _p_map[_s_p] = _o.op_uid
                                    self._slot_to_producer = _p_map
                                for _s_l, _t_l in enumerate(arena):
                                    if _t_l is None:
                                        continue
                                    if _s_l < w_n or _s_l < w_n + i_n:
                                        continue
                                    _nb_l = getattr(_t_l, '_nbytes', 0)
                                    if _nb_l < 100 * 1024 * 1024:
                                        continue  # only large slots
                                    _sh_l = tuple(getattr(_t_l, '_shape', ()))
                                    _tid_l = self._slot_to_tid.get(_s_l, "?")
                                    _prod_l = self._slot_to_producer.get(
                                        _s_l, "?")
                                    _lu_l = getattr(
                                        self, '_slot_last_use', {}).get(_s_l, "?")
                                    print(f"[LIVE_DUMP_SLOT slot={_s_l} "
                                          f"tid={_tid_l} "
                                          f"nbytes={_nb_l/1024/1024:.0f}MB "
                                          f"shape={_sh_l} "
                                          f"producer={_prod_l} "
                                          f"last_use={_lu_l} "
                                          f"current_op_idx={op_idx}]",
                                          flush=True)
                            except Exception as _slot_e:
                                print(f"[LIVE_DUMP_SLOT error] {_slot_e}",
                                      flush=True)
                            # Walk DeviceAllocator._cuda_ptr_size directly
                            # to show ALL live cudaMalloc blocks per device,
                            # bucketed by size. Exposes blocks that NBX
                            # allocated but no NBXTensor in this arena
                            # references (the 21 GB gap mystery).
                            from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DAW
                            from collections import Counter as _Counter
                            per_dev_buckets: Dict[int, "_Counter[int]"] = {}
                            per_dev_total: Dict[int, int] = {}
                            for _ptr, _nb in _DAW._cuda_ptr_size.items():
                                _d = _DAW._cuda_ptr_device.get(_ptr, -1)
                                bucket = (_nb // (1024 * 1024)) * 1024 * 1024
                                if bucket >= 100 * 1024 * 1024:  # only buckets >= 100 MB
                                    per_dev_buckets.setdefault(_d, _Counter())[bucket] += 1
                                per_dev_total[_d] = per_dev_total.get(_d, 0) + _nb
                            for _d, _t in sorted(per_dev_total.items()):
                                buckets = per_dev_buckets.get(_d, _Counter())
                                top = sorted(buckets.items(), key=lambda x: -x[0]*x[1])[:8]
                                print(f"[LIVE_BLOCKS dev={_d}] total={_t/1024/1024:.0f}MB "
                                      f"big_blocks(>=100MB): "
                                      + " ".join(f"{c}x{sz/1024/1024:.0f}MB" for sz, c in top),
                                      flush=True)
                            # Walk gc.get_objects() to find NBXTensor objects
                            # whose data_ptr matches the >= 1 GB allocations.
                            # Reports the type/repr of OTHER referrers (besides
                            # the tensor itself) — the source of the leak.
                            try:
                                import gc as _gc_h
                                from neurobrix.kernels.nbx_tensor import NBXTensor as _NBX_H
                                # Force a cycle-collecting gc.collect() BEFORE the
                                # scan. If the ORPHAN tensors are part of a Python
                                # ref cycle that prevents __del__ from firing,
                                # this resolves them. Compare big-block count
                                # before/after to confirm cycle vs leak.
                                pre_collect_big = sum(
                                    1 for p in _DAW._cuda_ptr_size
                                    if _DAW._cuda_ptr_size[p] >= 1024*1024*1024)
                                _collected = _gc_h.collect()
                                post_collect_big = sum(
                                    1 for p in _DAW._cuda_ptr_size
                                    if _DAW._cuda_ptr_size[p] >= 1024*1024*1024)
                                print(f"  [GC_COLLECT_TEST] gc.collect() collected {_collected} cycle objects, "
                                      f"big_blocks(>=1GB) before={pre_collect_big} after={post_collect_big}",
                                      flush=True)
                                big_ptrs = {p: _DAW._cuda_ptr_size[p]
                                            for p in _DAW._cuda_ptr_size
                                            if _DAW._cuda_ptr_size[p] >= 1024*1024*1024}
                                big_tensors = []
                                for o in _gc_h.get_objects():
                                    if isinstance(o, _NBX_H) and getattr(o, '_data_ptr', 0) in big_ptrs:
                                        big_tensors.append(o)
                                print(f"[BIG_TENSORS] {len(big_tensors)} NBXTensors with data_ptr in big blocks (>=1GB)",
                                      flush=True)
                                # Build reverse arena lookup: data_ptr -> tid
                                # so each big tensor can be identified by its
                                # graph tensor_id (or "ORPHAN" if not in arena).
                                ptr_to_tid: Dict[int, str] = {}
                                if arena is not None:
                                    if not hasattr(self, '_slot_to_tid'):
                                        self._slot_to_tid = {
                                            s: tid for tid, s in self._tid_to_slot.items()}
                                    for slot_idx, arena_t in enumerate(arena):
                                        if arena_t is not None and hasattr(arena_t, '_data_ptr'):
                                            ptr_to_tid.setdefault(arena_t._data_ptr,
                                                self._slot_to_tid.get(slot_idx, f"slot::{slot_idx}"))
                                for i, t in enumerate(big_tensors[:10]):
                                    refs = _gc_h.get_referrers(t)
                                    refs_summary = []
                                    for r in refs[:5]:
                                        rt = type(r).__name__
                                        info = ""
                                        if rt == "list":
                                            info = f"[len={len(r)}]"
                                            # For small lists/tuples, sample their contents
                                            # to identify the data structure (NBXTensor count,
                                            # other types). Useful for ORPHAN diagnosis.
                                            if len(r) <= 10:
                                                contents = [type(item).__name__
                                                            for item in r]
                                                info += f" contents={contents}"
                                        elif rt == "dict":
                                            info = f"{{len={len(r)}}}"
                                            if len(r) <= 10:
                                                key_types = [(type(k).__name__,
                                                              type(v).__name__)
                                                             for k, v in list(r.items())[:5]]
                                                info += f" sample_kv={key_types}"
                                        elif rt == "tuple":
                                            info = f"(len={len(r)})"
                                            if len(r) <= 10:
                                                contents = [type(item).__name__
                                                            for item in r]
                                                info += f" contents={contents}"
                                        elif rt == "frame":
                                            # Frame referrer — try to get the function name
                                            try:
                                                fname = r.f_code.co_name
                                                fline = r.f_lineno
                                                ffile = r.f_code.co_filename.split('/')[-1]
                                                info = f"@{fname}:{ffile}:{fline}"
                                            except Exception:
                                                pass
                                        refs_summary.append(f"{rt}{info}")
                                    tid = ptr_to_tid.get(t._data_ptr, "ORPHAN(not in arena)")
                                    # Refcount + gc.is_tracked diagnostic
                                    import sys as _sys_rc
                                    rc = _sys_rc.getrefcount(t)
                                    is_tracked = _gc_h.is_tracked(t)
                                    print(f"  tensor#{i} tid={tid} shape={tuple(t._shape)} "
                                          f"nbytes={t._nbytes/1024/1024:.0f}MB "
                                          f"owns={t._owns_data} "
                                          f"data_ptr={t._data_ptr} "
                                          f"refcount={rc} gc_tracked={is_tracked} "
                                          f"referrers({len(refs)}): {refs_summary}",
                                          flush=True)
                                    # For ORPHAN tensors specifically, walk one
                                    # level deeper: who holds the list[len=3]
                                    # and tuple(len=2) referrers? That gives the
                                    # actual data structure name.
                                    if tid.startswith("ORPHAN"):
                                        for j, r in enumerate(refs[:5]):
                                            grand = _gc_h.get_referrers(r)
                                            grand_summary = []
                                            for gr in grand[:3]:
                                                grt = type(gr).__name__
                                                ginfo = ""
                                                if grt == "frame":
                                                    try:
                                                        ginfo = f"@{gr.f_code.co_name}:{gr.f_code.co_filename.split('/')[-1]}:{gr.f_lineno}"
                                                    except Exception:
                                                        pass
                                                elif grt in ("list", "tuple", "dict"):
                                                    ginfo = f"[len={len(gr)}]"
                                                elif hasattr(gr, '__class__') and gr.__class__.__module__ != 'builtins':
                                                    ginfo = f"<{gr.__class__.__module__}.{grt}>"
                                                grand_summary.append(f"{grt}{ginfo}")
                                            print(f"    └─ ref#{j} {type(r).__name__} held by ({len(grand)}): {grand_summary}",
                                                  flush=True)
                                        # Find any 'cell' referrer and trace it
                                        for r in refs:
                                            if type(r).__name__ != 'cell':
                                                continue
                                            # Verify cell.cell_contents IS the tensor
                                            try:
                                                cc = r.cell_contents
                                                cc_match = (cc is t)
                                            except Exception:
                                                cc_match = False
                                            cell_holders = _gc_h.get_referrers(r)
                                            print(f"    [CELL_TRACE] cell.cell_contents is t: {cc_match}, "
                                                  f"cell id={id(r)}, cell holders ({len(cell_holders)}):",
                                                  flush=True)
                                            for ch in cell_holders[:5]:
                                                cht = type(ch).__name__
                                                cinfo = f" id={id(ch)} len={len(ch) if hasattr(ch,'__len__') else '?'}"
                                                # Identify if this list/tuple is our diagnostic structures
                                                if ch is big_tensors:
                                                    cinfo += " [DIAG=big_tensors]"
                                                # Print first few elements
                                                try:
                                                    sample = [type(x).__name__ for x in list(ch)[:5]]
                                                    cinfo += f" sample={sample}"
                                                except Exception:
                                                    pass
                                                # Find who holds this cell_holder
                                                gh = _gc_h.get_referrers(ch)
                                                gh_summary = []
                                                for ghi in gh[:5]:
                                                    ght = type(ghi).__name__
                                                    if ght == "function":
                                                        try:
                                                            gh_summary.append(f"fn:{ghi.__name__}@{ghi.__code__.co_filename.split('/')[-1]}:{ghi.__code__.co_firstlineno}")
                                                        except Exception:
                                                            gh_summary.append("fn:?")
                                                    elif ght == "frame":
                                                        try:
                                                            gh_summary.append(f"frame@{ghi.f_code.co_name}:{ghi.f_code.co_filename.split('/')[-1]}:{ghi.f_lineno}")
                                                        except Exception:
                                                            gh_summary.append("frame:?")
                                                    elif ght == "module":
                                                        gh_summary.append(f"module:{getattr(ghi,'__name__','?')}")
                                                    elif ght in ("list", "tuple", "dict"):
                                                        gh_summary.append(f"{ght}[{len(ghi) if hasattr(ghi,'__len__') else '?'}]")
                                                    else:
                                                        gh_summary.append(ght)
                                                cinfo += f" holders={gh_summary}"
                                                print(f"      cell_holder: {cht}{cinfo}", flush=True)
                                        # EXHAUSTIVE scan: walk ALL gc.get_objects
                                        # to find any object that refers to t via
                                        # any attribute (including __slots__).
                                        # gc.get_referrers might miss C-extension
                                        # or slot-based refs; this walks every
                                        # object's __dict__/__slots__/contents.
                                        candidate_holders = []
                                        for obj in _gc_h.get_objects():
                                            if obj is t or obj is refs:
                                                continue
                                            if isinstance(obj, (list, tuple, dict, set, frozenset)):
                                                continue  # already covered
                                            try:
                                                # Check __dict__
                                                d = getattr(obj, '__dict__', None)
                                                if d is not None and t in d.values():
                                                    # Find the specific attr name
                                                    attr_names = [k for k, v in d.items() if v is t]
                                                    obj_label = type(obj).__name__
                                                    # If obj is a module, get its name
                                                    if obj_label == 'module':
                                                        obj_label = f"module:{getattr(obj, '__name__', '?')}"
                                                    candidate_holders.append(
                                                        (type(obj).__module__, obj_label,
                                                         f'attr:{attr_names[:3]}'))
                                                    continue
                                                # Check __slots__
                                                slots = getattr(type(obj), '__slots__', ())
                                                if isinstance(slots, str):
                                                    slots = (slots,)
                                                for slot in slots:
                                                    try:
                                                        if getattr(obj, slot, None) is t:
                                                            candidate_holders.append(
                                                                (type(obj).__module__, type(obj).__name__, f'slot:{slot}'))
                                                            break
                                                    except Exception:
                                                        pass
                                            except Exception:
                                                pass
                                        print(f"    [EXHAUSTIVE] {len(candidate_holders)} non-container holders:",
                                              flush=True)
                                        for mod, cls, where in candidate_holders[:10]:
                                            print(f"      {mod}.{cls} via {where}", flush=True)
                                # Print _seq_constant_originals size for context
                                try:
                                    sco_size = len(getattr(self, '_seq_constant_originals', {}))
                                    print(f"  [DIAG] _seq_constant_originals dict size: {sco_size}",
                                          flush=True)
                                except Exception:
                                    pass
                                # Scan Triton Autotuner instances for nargs leaks
                                # (autotune's self.nargs = None is only reached
                                # on success; on exception, args dict persists
                                # holding tensor refs as C-level hidden refs).
                                try:
                                    from triton.runtime.autotuner import Autotuner as _AT
                                    autotuners = [o for o in _gc_h.get_objects()
                                                  if isinstance(o, _AT)]
                                    leaks = []
                                    for at_inst in autotuners:
                                        nargs = getattr(at_inst, 'nargs', None)
                                        if nargs is not None and isinstance(nargs, dict):
                                            tensor_count = sum(1 for v in nargs.values()
                                                               if isinstance(v, _NBX_H))
                                            if tensor_count > 0:
                                                fn_name = getattr(getattr(at_inst, 'base_fn', None),
                                                                  '__name__', '?')
                                                leaks.append((fn_name, len(nargs), tensor_count))
                                    print(f"  [AUTOTUNE_LEAK_SCAN] {len(autotuners)} Autotuner instances; "
                                          f"{len(leaks)} with non-None nargs holding NBXTensors:",
                                          flush=True)
                                    for fn_name, n_args, n_tensors in leaks:
                                        print(f"    LEAK: kernel={fn_name} nargs_count={n_args} "
                                              f"nbxtensor_count={n_tensors}", flush=True)
                                except Exception as _e:
                                    print(f"  [AUTOTUNE_LEAK_SCAN failed: {_e}]", flush=True)
                            except Exception as _ge:
                                print(f"[BIG_TENSORS scan failed: {_ge}]", flush=True)
                        except Exception as _dump_e:
                            print(f"[LIVE_DUMP failed: {_dump_e}]", flush=True)
                    _none_pos = [i for i, a in enumerate(args) if a is None]
                    raise RuntimeError(
                        f"Failed at {op.op_uid} ({op.op_type}): {e} | None args at "
                        f"positions {_none_pos} of {len(args)}") from e
                if _PROF:
                    DeviceAllocator.sync_device()
                    _dt = _time.perf_counter() - _t0
                    if op.op_type in _MATMUL:
                        _cat = "matmul"
                    elif op.op_type in _SDPA:
                        _cat = "sdpa"
                    elif op.op_type in _META:
                        _cat = "meta"
                    elif op.op_type in _EMBED:
                        _cat = "embed"
                    elif "::" in op.op_type:
                        _cat = "elem"
                    else:
                        _cat = "other"
                    _timings[_cat] += _dt
                    _counts[_cat] += 1
                    if _cat == "elem":
                        _elem_timings[op.op_type] = _elem_timings.get(op.op_type, 0.0) + _dt
                        _elem_counts[op.op_type] = _elem_counts.get(op.op_type, 0) + 1

            # Defer old slot tensors before overwriting — cudaFree is
            # synchronous so immediate release while a kernel may still
            # be reading the memory would UAF. The deferred list is
            # drained when it crosses a bytes/count threshold (Route A)
            # and one final time after the loop to catch the tail.
            for s in op.output_slots:
                old = arena[s]
                if old is not None:
                    _deferred.append(old)
                    _deferred_bytes += old._nbytes
            # Drop the loop variable so it does not retain the
            # last seen NBXTensor across subsequent outer-loop
            # iterations. See P-SANA-4KPX-RUNTIME 2026-05-06.
            old = None  # noqa: F841

            # Store outputs
            if not op.output_slots:
                pass
            elif len(op.output_slots) == 1:
                # A multi-output op (split / chunk / unbind) can return a
                # 1-element tuple when a data-dependent split yields a single
                # chunk at the runtime resolution (e.g. HunyuanVAE
                # chunk_nearest_interpolate -> n_chunks == 1). One traced output
                # slot => unwrap; single-tensor ops never return a tuple/list, so
                # this is unambiguous. Mirrors compiled_sequence (R30).
                if isinstance(result, (tuple, list)):
                    result = result[0] if len(result) >= 1 else None
                arena[op.output_slots[0]] = result
            elif isinstance(result, tuple):
                for i, s in enumerate(op.output_slots):
                    arena[s] = result[i] if i < len(result) else None
            else:
                arena[op.output_slots[0]] = result

            # === TEMP NBX_TRITON_TRACE_NAN=1 : log first Inf/NaN op ===
            # Gate hoisted to run start (C1).
            if _trace_nan_on and op.output_slots:
                self._maybe_trace_nan(op, arena)
            # =============================================================

            # === NBX_OP_FINGERPRINT=<jsonl> : run-to-run op-output hash
            #     (P-TRITON-MOE-DETERMINISM-RESIDUAL differential) ===
            # Gate hoisted to run start (C1).
            if _fp_path and op.output_slots:
                self._maybe_fingerprint(op, arena, _fp_path)
            # =============================================================

            # === TEMP TID DUMP: compare triton vs native per-op output ===
            # Gate hoisted to run start (C1).
            if _dump_tids_env and op.output_slots:
                self._maybe_dump_tid(op, _dump_tids_env)
            # =============================================================

            # === Targeted tid lifecycle trace (P-SANA-4KPX-RUNTIME) ===
            # NBX_TRACE_TIDS="tid1,tid2" prints [ALLOC] when a slot for a
            # matching tid receives a tensor, and [KILL] when that slot
            # is killed by op.kill_slots. Refcount snapshot at each event
            # tells us whether the NBXTensor is held only by the arena
            # (refcount==2: 1 from arena, 1 from sys.getrefcount's own
            # arg) or by some other reference too (refcount>2 = leak).
            # Gate + tid set hoisted to run start (C1).
            if _trace_tids:
                import sys as _sys_tt
                if not hasattr(self, '_slot_to_tid'):
                    self._slot_to_tid = {
                        sl: t for t, sl in self._tid_to_slot.items()}
                for s in op.output_slots:
                    tid = self._slot_to_tid.get(s)
                    if tid in _trace_tids:
                        t = arena[s]
                        if t is not None:
                            print(f"[TID_TRACE ALLOC op_idx={op_idx} {op.op_uid}] "
                                  f"tid={tid} slot={s} "
                                  f"data_ptr={getattr(t, '_data_ptr', '?')} "
                                  f"refcount={_sys_tt.getrefcount(t)} "
                                  f"shape={tuple(getattr(t, '_shape', ()))} "
                                  f"nbytes={getattr(t, '_nbytes', 0)/1024/1024:.0f}MB",
                                  flush=True)

            if not skip_kills:
                for s in op.kill_slots:
                    old = arena[s]
                    if old is not None:
                        # Targeted lifecycle trace: snapshot refcount
                        # BEFORE kill_slots assigns arena[s]=None. If
                        # refcount > 2 (arena + sys.getrefcount arg),
                        # the tensor has additional Python refs that
                        # will keep it alive after the kill — that is
                        # exactly the orphan-leak signature.
                        if _trace_tids:
                            tid = self._slot_to_tid.get(s)
                            if tid in _trace_tids:
                                import sys as _sys_tk
                                print(f"[TID_TRACE KILL op_idx={op_idx} {op.op_uid}] "
                                      f"tid={tid} slot={s} "
                                      f"data_ptr={getattr(old, '_data_ptr', '?')} "
                                      f"refcount_before_kill={_sys_tk.getrefcount(old)} "
                                      f"(if >2 = orphan-leak; arena+sys.getrefcount=2)",
                                      flush=True)
                        _deferred.append(old)
                        _deferred_bytes += old._nbytes
                    arena[s] = None
                # Python `for` loop variables persist after the loop,
                # bound to the last iteration's value. If the last
                # killed slot held a large NBXTensor, `old` retains
                # that reference across subsequent outer-loop iterations
                # until the next op with kill_slots rebinds it. On
                # Sana 4Kpx VAE this leaked an 8 GiB conv::63 tensor
                # past silu::24's kill_slots into conv::64's allocation
                # attempt, triggering OOM at live=25GB despite the
                # arena and the deferred queue both having released.
                # Drop the binding explicitly so the reference chain
                # breaks here. P-SANA-4KPX-RUNTIME 2026-05-06.
                old = None  # noqa: F841

            # Diagnostic: NBX_FORCE_GC=N forces Python gc.collect() every N
            # ops to test if the live_tracked watermark gap is from stale
            # Python references holding NBXTensor __del__ from firing.
            # Per-op (N=1) is too slow for production but useful for diag.
            # Gate hoisted to run start (C1).
            if _gc_int > 0 and op_idx % _gc_int == 0:
                import gc as _gc_force
                _gc_force.collect()

            # Pressure-sensitive drain (see module header + deferred_drain_policy).
            # Always drain at the cliff/count ceiling; drain EARLY (queue >=
            # floor) only when a throttled driver-free probe shows the device
            # within `reserve` of capacity. sync-before-free correctness is
            # unchanged — draining more often is never a UAF risk, only a
            # throughput cost, which the pressure gate confines to the case
            # where the alternative is OOM.
            _drain_now = (_deferred_bytes >= _drain_cliff
                          or len(_deferred) >= _drain_count_limit)
            _pressure = False
            if (not _drain_now and _drain_reserve > 0
                    and _deferred_bytes >= _drain_floor
                    and (op_idx - _last_free_check) >= _drain_interval):
                _last_free_check = op_idx
                _free_b = DeviceAllocator.device_free_bytes()
                if 0 <= _free_b < _drain_reserve:
                    _drain_now = True
                    _pressure = True
            if _drain_now:
                if _drain_stats is not None:
                    _drain_stats[0] += 1
                    _drain_stats[2 if _pressure else 1] += 1
                    print(f"[NBX_DEFERRED_DRAIN] single-dev drain "
                          f"#{_drain_stats[0]}: {len(_deferred)} tensors, "
                          f"{_deferred_bytes/1e9:.2f} GB, "
                          f"trigger={'pressure' if _pressure else 'cliff'}",
                          flush=True)
                DeviceAllocator.sync_device()
                _deferred.clear()
                _deferred_bytes = 0

        # All kernels submitted — sync GPU then release deferred tensors
        if _deferred:
            DeviceAllocator.sync_device()
            _deferred.clear()
            _deferred_bytes = 0

        if _drain_stats is not None and _drain_stats[0] > 0:
            print(f"[NBX_DEFERRED_DRAIN] single-dev totals: "
                  f"{_drain_stats[0]} drains "
                  f"({_drain_stats[1]} cliff/count, {_drain_stats[2]} pressure)",
                  flush=True)

        # ===== TEMP PROFILING REPORT =====
        if _PROF:
            _total = sum(_timings.values())
            print(f"\n[NBX_TRITON_PROF] call #{self._prof_call_idx} "
                  f"(skip_kills={skip_kills})")
            for _cat in ["matmul", "sdpa", "meta", "elem", "embed", "other"]:
                _pct = 100 * _timings[_cat] / _total if _total > 0 else 0
                _avg = (1000 * _timings[_cat] / _counts[_cat]
                        if _counts[_cat] > 0 else 0)
                print(f"  {_cat:8s}: {_timings[_cat]*1000:8.1f}ms  "
                      f"({_counts[_cat]:4d} ops, {_avg:.3f}ms/op, "
                      f"{_pct:5.1f}%)")
            print(f"  TOTAL:    {_total*1000:.1f}ms")
            if _elem_timings:
                _elem_total = sum(_elem_timings.values())
                _sorted = sorted(_elem_timings.items(), key=lambda kv: kv[1], reverse=True)
                print(f"  [elem breakdown — {len(_sorted)} distinct op_types, "
                      f"{sum(_elem_counts.values())} total ops]")
                for _op_type, _ms in _sorted[:20]:
                    _n = _elem_counts[_op_type]
                    _p = 100 * _ms / _elem_total if _elem_total > 0 else 0
                    _avg = 1000 * _ms / _n if _n > 0 else 0
                    print(f"    {_op_type:40s} {_ms*1000:7.2f}ms  "
                          f"({_n:4d} ops, {_avg:.3f}ms/op, {_p:5.1f}% of elem)")
        # ==================================

    def _run_multi_device(self, skip_kills: bool = False,
                            pre_op_callback: Optional[Callable[[int, 'CompiledOp'], None]] = None):
        """Multi-device hot loop. Ported from compiled_sequence._run_inner_multi_device.

        - Fast path (99%+ ops): just switch device context if op.device_idx changed
        - Slow path (needs_transfer ops): D2D memcpy inputs to target device.
          For CPU-backed weight inputs (zero3), _transfer_tensor handles
          H2D (kind=1) instead of D2D.

        Uses cudaMalloc/cudaFree (sync) underneath, so both output-slot
        overwrites and kill_slots batch their old tensors into a
        `_deferred` list. The list is drained when its bytes or count
        cross the Route A thresholds (NBX_DEFERRED_DRAIN_BYTES /
        NBX_DEFERRED_DRAIN_COUNT) and one final time after the loop —
        same correctness as the original end-of-run-only drain (sync
        before free), just more often.

        pre_op_callback is invoked with (op_idx, op) BEFORE each op's
        args are resolved. Used by zero3 for first-tick priming and by
        block-wise pipelining (deferred) for boundary rebinds. Skipped
        (no branch cost beyond the None check) when callback is None.
        """
        arena = self._arena
        _current_dev = self.device_idx
        _deferred: List[NBXTensor] = []
        _deferred_bytes: int = 0
        (_drain_cliff, _drain_floor, _drain_reserve, _drain_interval,
         _drain_count_limit) = _resolve_drain_policy()
        _last_free_check = -(1 << 30)
        _drain_diag = os.environ.get("NBX_DEFERRED_DRAIN_DIAG") == "1"
        _drain_stats = [0, 0, 0] if _drain_diag else None  # [drains, cliff/count, pressure]

        # OOM last-chance reclaim: expose THIS run's deferred queue to the
        # allocator (drained+retried only after a failed device malloc).
        self._oom_deferred_queue = _deferred
        self._ensure_oom_reclaim_hook()

        # C1 (P-KERNEL-LAUNCH-HYGIENE): diagnostic env gates hoisted out
        # of the per-op hot loop — read ONCE per run (same gate
        # semantics; env changes mid-run no longer apply).
        _dtids = os.environ.get("NBX_DUMP_TIDS")
        _fp_path_md = os.environ.get("NBX_OP_FINGERPRINT", "")

        for op_idx, op in enumerate(self._ops):
            if pre_op_callback is not None:
                pre_op_callback(op_idx, op)
            args = op.args_resolver(arena)
            kwargs = op.kwargs_resolver(arena)

            # Fix device kwargs for creation ops on the right device
            if kwargs and 'device' in kwargs and op.device_idx is not None:
                kwargs['device'] = f"cuda:{op.device_idx}"

            # NOP propagation — deactivated MoE expert path returns None.
            if args and args[0] is None:
                for s in op.output_slots:
                    old = arena[s]
                    if old is not None:
                        _deferred.append(old)
                        _deferred_bytes += old._nbytes
                    arena[s] = None
                if not skip_kills:
                    for s in op.kill_slots:
                        old = arena[s]
                        if old is not None:
                            _deferred.append(old)
                            _deferred_bytes += old._nbytes
                        arena[s] = None
                old = None  # noqa: F841 — see P-SANA-4KPX-RUNTIME 2026-05-06
                continue

            # FAST PATH: no transfer needed
            if not op.needs_transfer:
                if op.device_idx is not None and op.device_idx != _current_dev:
                    # _current_dev only gates WHEN this pair fires; the
                    # C2 device cache in nbx_tensor stays coherent by
                    # construction (set_device invalidates on switch,
                    # ensure_triton_device re-syncs + recaches), so the
                    # two trackers cannot desynchronize.
                    DeviceAllocator.set_device(op.device_idx)
                    DeviceAllocator.ensure_triton_device(op.device_idx)
                    _current_dev = op.device_idx
                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    _none_pos = [i for i, a in enumerate(args) if a is None]
                    raise RuntimeError(
                        f"Failed at {op.op_uid} ({op.op_type}): {e} | None args at "
                        f"positions {_none_pos} of {len(args)}") from e
            else:
                # SLOW PATH: transfer inputs to target device
                target = op.device_idx
                if target is not None:
                    args = self._transfer_args(args, target)
                    if kwargs:
                        kwargs = self._transfer_kwargs(kwargs, target)
                if target is not None and target != _current_dev:
                    DeviceAllocator.set_device(target)
                    DeviceAllocator.ensure_triton_device(target)
                    _current_dev = target
                try:
                    result = op.func(*args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed at {op.op_uid} ({op.op_type}): {e}") from e

            # Defer any tensors currently in the output slots before overwriting.
            for s in op.output_slots:
                old = arena[s]
                if old is not None:
                    _deferred.append(old)
                    _deferred_bytes += old._nbytes

            # Store outputs
            if not op.output_slots:
                pass
            elif len(op.output_slots) == 1:
                # A multi-output op (split / chunk / unbind) can return a
                # 1-element tuple when a data-dependent split yields a single
                # chunk at the runtime resolution (e.g. HunyuanVAE
                # chunk_nearest_interpolate -> n_chunks == 1). One traced output
                # slot => unwrap; single-tensor ops never return a tuple/list, so
                # this is unambiguous. Mirrors compiled_sequence (R30).
                if isinstance(result, (tuple, list)):
                    result = result[0] if len(result) >= 1 else None
                arena[op.output_slots[0]] = result
            elif isinstance(result, tuple):
                for i, s in enumerate(op.output_slots):
                    arena[s] = result[i] if i < len(result) else None
            else:
                arena[op.output_slots[0]] = result

            # === TEMP TID DUMP (multi-device branch) ===
            # Gate hoisted to run start (C1).
            if _dtids and op.output_slots:
                self._maybe_dump_tid(op, _dtids)
            # ===========================================

            # === NBX_OP_FINGERPRINT (multi-device branch) ===
            # Gate hoisted to run start (C1).
            if _fp_path_md and op.output_slots:
                self._maybe_fingerprint(op, arena, _fp_path_md)
            # ================================================

            if not skip_kills:
                for s in op.kill_slots:
                    old = arena[s]
                    if old is not None:
                        _deferred.append(old)
                        _deferred_bytes += old._nbytes
                    arena[s] = None
            old = None  # noqa: F841 — see P-SANA-4KPX-RUNTIME 2026-05-06

            # Pressure-sensitive drain — mirror of the single-device path
            # (see module header + deferred_drain_policy). The free probe uses
            # the current device (no-arg → DeviceAllocator.get_device()); the
            # cliff is the hard guarantee, the pressure gate is best-effort
            # across devices.
            _drain_now = (_deferred_bytes >= _drain_cliff
                          or len(_deferred) >= _drain_count_limit)
            _pressure = False
            if (not _drain_now and _drain_reserve > 0
                    and _deferred_bytes >= _drain_floor
                    and (op_idx - _last_free_check) >= _drain_interval):
                _last_free_check = op_idx
                _free_b = DeviceAllocator.device_free_bytes()
                if 0 <= _free_b < _drain_reserve:
                    _drain_now = True
                    _pressure = True
            if _drain_now:
                if _drain_stats is not None:
                    _drain_stats[0] += 1
                    _drain_stats[2 if _pressure else 1] += 1
                    print(f"[NBX_DEFERRED_DRAIN] multi-dev drain "
                          f"#{_drain_stats[0]}: {len(_deferred)} tensors, "
                          f"{_deferred_bytes/1e9:.2f} GB, "
                          f"trigger={'pressure' if _pressure else 'cliff'}",
                          flush=True)
                DeviceAllocator.sync_device()
                _deferred.clear()
                _deferred_bytes = 0

        # All kernels submitted — sync GPU then release deferred tensors
        if _deferred:
            DeviceAllocator.sync_device()
            _deferred.clear()
            _deferred_bytes = 0

        if _drain_stats is not None and _drain_stats[0] > 0:
            print(f"[NBX_DEFERRED_DRAIN] multi-dev totals: "
                  f"{_drain_stats[0]} drains "
                  f"({_drain_stats[1]} cliff/count, {_drain_stats[2]} pressure)",
                  flush=True)


    @staticmethod
    def _needs_move(t: NBXTensor, target_dev: int) -> bool:
        """Delegates to the shared cross-device helper (R30: same impl as the
        op-by-op triton_sequential path)."""
        from neurobrix.triton.device_transfer import needs_move
        return needs_move(t, target_dev)

    @staticmethod
    def _find_cuda_arg(args) -> 'Optional[int]':
        """Return the device_idx of the first CUDA NBXTensor in args (scanning
        nested lists/tuples one level deep), or None if args contains no
        CUDA tensor.

        Used by _transfer_args to mimic native's slow-path rule: only
        promote CPU args if there's already a CUDA arg in the list. For
        metadata ops (aten::t, view, reshape, permute) on a CPU weight
        with no CUDA activation, this returns None → no promotion → the
        op runs as a pure-Python NBXTensor view on the CPU weight, which
        is correctness-equivalent to the native `cpu_weight.t()` path
        and crucially avoids allocating a 3-MB GPU temp per call (which
        is then retained by the view's _base chain in the arena,
        growing residency by ~1.28 GB per transformer block — the
        mechanism documented in LEAK_PINPOINT_REPORT.md).
        """
        for a in args:
            if isinstance(a, NBXTensor) and getattr(a, '_device', 'cuda') == 'cuda':
                return a._device_idx
            if isinstance(a, (list, tuple)):
                for item in a:
                    if isinstance(item, NBXTensor) and getattr(item, '_device', 'cuda') == 'cuda':
                        return item._device_idx
        return None

    def _transfer_args(self, args, target_dev: int):
        """Transfer NBXTensor args to target CUDA device.

        Handles both cross-GPU (D2D) and CPU-offload (H2D via zero3).

        Zero3 metadata-op preservation: if ALL NBXTensor args are CPU
        (no CUDA tensor in the list), returns args unchanged instead
        of promoting. Matches native compiled_sequence behavior — the
        slow path only promotes CPU→GPU when compute on GPU is actually
        required, not for metadata ops that return views. Without this,
        every aten::t / aten::view / aten::permute on a CPU-resident
        zero3 weight produces a fresh GPU copy that the arena's view
        retains via _base chain, leaking block-sized residency per
        forward pass.
        """
        # If no CUDA arg is present, there is no target to promote TO.
        # Run the op on the CPU args as-is (metadata ops produce CPU
        # views, which is exactly what native does).
        if self._find_cuda_arg(args) is None:
            return args

        new_args = []
        for a in args:
            if isinstance(a, NBXTensor) and self._needs_move(a, target_dev):
                new_args.append(self._transfer_tensor(a, target_dev))
            elif isinstance(a, (list, tuple)):
                moved = [
                    self._transfer_tensor(item, target_dev)
                    if isinstance(item, NBXTensor) and self._needs_move(item, target_dev)
                    else item
                    for item in a
                ]
                new_args.append(type(a)(moved))
            else:
                new_args.append(a)
        return new_args

    def _transfer_kwargs(self, kwargs, target_dev: int):
        """Transfer NBXTensor kwargs to target CUDA device."""
        new_kw = {}
        for k, v in kwargs.items():
            if isinstance(v, NBXTensor) and self._needs_move(v, target_dev):
                new_kw[k] = self._transfer_tensor(v, target_dev)
            else:
                new_kw[k] = v
        return new_kw

    def _transfer_tensor(self, tensor: NBXTensor, target_dev: int) -> NBXTensor:
        """Copy NBXTensor to target CUDA device, preserving strides.

        Picks the memcpy kind based on the source device:
          - CPU source (zero3 offload): kind=1 (H2D)
          - CUDA source (cross-GPU transfer): kind=3 (D2D)

        Stride preservation (critical for zero3 correctness): the
        destination NBXTensor is constructed with the source's shape
        AND strides, not forced to contiguous. When
        _eliminate_weight_transpose_ops pre-transposes a linear weight
        at bind time, the arena holds a .t() view with swapped strides
        over the original row-major bytes. A previous implementation
        used NBXTensor.empty() for the destination, which hard-sets
        _contiguous_strides(shape) — the memcpy'd bytes (still the
        original layout) were then re-indexed as if they were the
        transposed layout, silently producing act @ W instead of
        act @ W.t() inside every mm under zero3. Logit cosine vs
        native was near zero. See
        tests/scratch/divergence_inv/DIVERGENCE_REPORT.md for the
        bytes-vs-strides walkthrough and reproducer. Fix: construct
        the GPU NBXTensor directly so the view's stride semantics
        carry over; downstream .contiguous() inside mm/bmm then
        correctly materialises via strided_copy, matching what the
        native torch.Tensor.to(device) contract does.
        """
        # Delegates to the shared cross-device helper so the compiled and
        # op-by-op triton_sequential paths share ONE transfer impl (R30).
        from neurobrix.triton.device_transfer import transfer_tensor
        return transfer_tensor(tensor, target_dev)

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
