"""The width each activation is EXECUTED at — the dtype pass Prism prices a plan with.

Why this exists. `ActivationProfiler` sized every floating activation at the component's
compute dtype C (`force_compute_dtype_for_fp=True`). The engines do not execute every
tensor at C. Measured on Sana_1600M_4Kpx_BF16 at 3072x4096 with no calibration record
(the Mac, and the same log line on this rack's CUDA): the VAE decoder's full-resolution
tail ran in fp32 — every `custom::rms_norm` (AMP_FP32 class) kept an fp32 output, the
binary ops took the wider operand so the residual stream stayed fp32, and the fused
upsample+conv allocated its output at its pre-input's width. The allocator trace at the
crash held 15 360 MB of activations in three buffers against the plan's 6 144 MB for the
whole VAE: the plan was accepted and ran out of memory.

What it does. Given a component DAG, the compute dtype C, the engine, the hardware's
`has_native_bf16`, the component's precision contract and (when known) the op-level tiling
plan, `runtime_dtypes` walks the execution order with THE ENGINE'S rules and returns the
dtype every tensor holds at runtime; `runtime_widths` returns the same as bytes per element.
Non-floating tensors keep their traced dtype.

Where the rules come from. Everything that can be CALLED is called: the Triton engine's AMP
classes and its fp32-constant rule (`triton/dtype.py`), the stored-weight rule
(`triton/weight_loader.stored_dtype_in_compute`), the seam predicate
(`core/prism/layer_partition.is_seam_tensor`), the op-name aliases
(`kernels/classification.canonical_aten`), and the precision contract's own functions
(`core/runtime/precision_contract`, `core/dtype/calibration`). What cannot be called without
a device — the per-op wrap decisions of `TritonDtypeEngine.wrap_op`, the wrappers' output
allocations, the ATen `DtypeEngine.compile_op` (whose module imports torch; Prism is
torch-free) — is MIRRORED here, each rule beside its source `file:line`. This module is the
ONE place Prism holds them; `tests/unit/prism/test_the_estimator_prices_the_width_the_runtime_executes.py`
compares the mirrored ATen sets with the engine's own (drift door).

What it does NOT know, and the choice made each time (never a narrower width):
  * which upsample->conv pairs the runtime fuses and which ops it tiles standalone: the
    op-level tiling plan is computed AFTER placement (`PrismSolver._detect_op_level_tiling_pairs`,
    on its own traced-dtype estimate). Without a plan, every structurally eligible pair / op is
    priced at the WIDER of its tiled and untiled width.
  * S5 residual chains write their result in place into the chain's base tensor
    (tiling_engine.py:2221, residual_chain.py): the chain's ops are priced by the op rules,
    whose binary rule is never narrower than the base — an upper bound of the in-place write.
  * the `sequential` (ATen op-by-op) engine is priced with the compiled engine's table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from neurobrix.core.prism.memory_estimator import get_dtype_bytes_per_element
from neurobrix.core.prism.layer_partition import is_seam_tensor
from neurobrix.kernels.classification import canonical_aten
from neurobrix.triton import dtype as _tdt
from neurobrix.triton.weight_loader import stored_dtype_in_compute


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

#: The execution-mode strings the solver receives (cli/commands/run.py:247-284,
#: cli/commands/upscale.py:23-32). Anything else is refused by name.
TRITON_ENGINES = frozenset({"triton", "triton_sequential"})
ATEN_ENGINES = frozenset({"compiled", "sequential"})

_FLOAT = frozenset({"float16", "bfloat16", "float32", "float64"})
_HALF = frozenset({"float16", "bfloat16"})
_COMPLEX = frozenset({"complex32", "complex64", "complex128"})
_KNOWN = _FLOAT | _COMPLEX | frozenset({"int8", "int16", "int32", "int64", "uint8", "bool"})

# Rank for "the wider of two" — wrappers.py:545-574 `_DTYPE_PRIORITY` / `_wider_dtype`
# (fp16 and bf16 share a rank there; the first operand wins a tie, which `_wider` mirrors).
_RANK = {"bool": 0, "uint8": 1, "int8": 2, "int16": 3, "int32": 4, "int64": 5,
         "float16": 6, "bfloat16": 6, "float32": 7, "float64": 8,
         "complex64": 9, "complex128": 10}


# ---------------------------------------------------------------------------
# The ATen engine's AMP classes — MIRRORED from core/dtype/engine.py (it imports torch).
# A test reads the engine's own sets and fails on any drift.
# ---------------------------------------------------------------------------

# core/dtype/engine.py:420-484 AMP_FP32_OPS. Unlike the Triton set it keeps the nearest
# upsamples and holds no `rms_norm` (the compiled rms_norm is `rms_norm_fp32`, below).
ATEN_AMP_FP32_OPS: FrozenSet[str] = frozenset({
    "acos", "asin", "cosh", "erfinv", "exp", "expm1",
    "log", "log10", "log2", "log1p", "reciprocal", "rsqrt",
    "sinh", "tan", "pow", "softplus",
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "cudnn_batch_norm", "instance_norm",
    "frobenius_norm", "nuclear_norm", "cosine_similarity",
    "poisson_nll_loss", "cosine_embedding_loss", "nll_loss", "nll_loss2d",
    "hinge_embedding_loss", "kl_div", "l1_loss", "smooth_l1_loss",
    "huber_loss", "mse_loss", "margin_ranking_loss",
    "multilabel_margin_loss", "soft_margin_loss", "triplet_margin_loss",
    "multi_margin_loss", "binary_cross_entropy_with_logits",
    "dist", "pdist", "cdist",
    "polar", "view_as_complex",
    "renorm", "logsumexp",
    "upsample_nearest1d", "_upsample_nearest_exact1d",
    "upsample_nearest2d", "_upsample_nearest_exact2d",
    "upsample_nearest3d", "_upsample_nearest_exact3d",
    "upsample_linear1d", "upsample_bilinear2d", "_upsample_bilinear2d_aa",
    "upsample_trilinear3d", "upsample_bicubic2d", "_upsample_bicubic2d_aa",
    "prod", "softmax", "_softmax", "log_softmax",
    "cumprod", "cumsum", "sum",
    "linalg_vector_norm", "linalg_matrix_norm",
})
# core/dtype/engine.py:488-514 AMP_FP16_OPS.
ATEN_AMP_FP16_OPS: FrozenSet[str] = frozenset({
    "_convolution", "conv1d", "conv2d", "conv3d", "conv_tbc", "convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "addmm", "addmv", "addr", "matmul", "einsum",
    "mm", "mv", "bmm", "addbmm", "baddbmm",
    "linalg_vecdot", "linear", "chain_matmul", "linalg_multi_dot",
    "_thnn_fused_lstm_cell", "_thnn_fused_gru_cell",
    "lstm_cell", "gru_cell", "rnn_tanh_cell", "rnn_relu_cell",
    "lstm", "gru", "rnn_tanh", "rnn_relu",
    "prelu",
    "div",
})
# core/dtype/engine.py:519 _FP16_NEED_FP32, :525 _FP16_GEMM_OPS, :531-534 _FP32_OPS_HALF_IO.
ATEN_FP16_NEED_FP32: FrozenSet[str] = frozenset({"mm", "bmm", "div", "addmm"})
ATEN_FP16_GEMM_OPS: FrozenSet[str] = frozenset({"mm", "bmm", "addmm"})
ATEN_FP32_OPS_HALF_IO: FrozenSet[str] = frozenset({
    "native_layer_norm", "layer_norm", "native_group_norm", "group_norm",
    "_softmax", "softmax", "_log_softmax", "log_softmax",
})
# core/dtype/engine.py:538-552 AMP_PROMOTE_OPS.
ATEN_AMP_PROMOTE_OPS: FrozenSet[str] = frozenset({
    "addcdiv", "addcmul", "atan2", "bilinear", "cross",
    "dot", "vdot", "grid_sampler", "grid_sampler_2d", "grid_sampler_3d", "index_put",
    "tensordot", "scatter_add", "index_add",
})
# core/dtype/engine.py:577-579 AMP_CREATION_FILL_OPS (the clamp; output = the resolved dtype).
ATEN_AMP_CREATION_FILL_OPS: FrozenSet[str] = frozenset({"full", "new_full", "full_like"})


# ---------------------------------------------------------------------------
# Op families read by the wrapper rules (names, not models)
# ---------------------------------------------------------------------------

# Every conv the Triton branch lands in `conv2d_wrapper` / `conv_transpose_wrapper` /
# `_conv3d_via_conv2d`, whose output is `_NBX_COMPUTE_DTYPE` (wrappers.py:4187, :3844, :3905).
_CONV = frozenset({"_convolution", "convolution", "conv1d", "conv2d", "conv3d",
                   "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                   "conv_depthwise2d"})
_UPSAMPLE_NEAREST = frozenset({"upsample_nearest1d", "upsample_nearest2d",
                               "upsample_nearest3d", "_upsample_nearest_exact2d"})
_SDPA = frozenset({"scaled_dot_product_attention", "_scaled_dot_product_efficient_attention",
                   "_scaled_dot_product_flash_attention", "_scaled_dot_product_cudnn_attention",
                   "_scaled_dot_product_attention_math"})
_CAT = frozenset({"cat", "concat", "stack"})
_CASTS = frozenset({"_to_copy", "to"})
# Matmul-class ops whose Triton wrapper is not `mm`/`addmm`/`bmm` and whose store dtype
# was not traced op by op here: priced fp32, the widest a matmul stores (conservative; one
# `baddbmm` in the whole catalogue, the rest absent, 2026-09-28).
_MATMUL_OTHER = frozenset({"mv", "addmv", "addr", "einsum", "baddbmm", "addbmm",
                           "dot", "vdot", "linalg_vecdot", "chain_matmul", "linalg_multi_dot"})


@dataclass(frozen=True)
class PrecisionContract:
    """(activations_fp16_safe, fp32_op_uids, narrow_op_uids) — the triple
    `core/runtime/precision_contract.resolve` hands the engines — plus WHY, for a reader."""
    safe: bool
    fp32_op_uids: FrozenSet[str]
    narrow_op_uids: FrozenSet[str]
    why: str = ""


def conservative_contract(why: str) -> PrecisionContract:
    return PrecisionContract(False, frozenset(), frozenset(), why)


def plan_time_contract(cache_path, component_name: str, dag: Optional[Dict[str, Any]],
                       compute_dtype: str) -> PrecisionContract:
    """The contract `precision_contract.resolve` (precision_contract.py:213-283) would hand
    this component, computed WITHOUT its side effects (it prints, binds a calibration census
    and reads the arch through the launcher) — by calling the same functions it calls.

    Choices the plan makes where the runtime reads something the plan cannot see, each in
    the direction that prices WIDER (the conservative path is the wider one):
      * the record is looked up with `dag=None` (precision_contract.py:188-189 returns it
        unchecked and silent) and its two validity tests are asked here (:190-208): a record
        that observed nothing, or whose signature does not match THIS graph, is not applied.
        The runtime matches against the graph after its rewrite passes: a record signed on the
        rewritten graph fails here and the plan prices the conservative (wider) path.
      * `record.prefer` (:250-259) is keyed by the launcher's arch fingerprint, read from the
        driver: ANY 'conservative' preference makes the plan price the conservative path.
    """
    c = str(compute_dtype).replace("torch.", "")
    if c != "float16":                                   # precision_contract.py:231-233
        return conservative_contract(f"compute dtype {c}: the contract exists for float16 only")
    from neurobrix.core.runtime import precision_contract as _pc
    from neurobrix.core.dtype import calibration as _cal
    forced = _pc._env_force()                            # :234-236
    if forced is False:
        return conservative_contract(f"{_pc.FLAG_ENV}=0")
    record = _pc.load_calibration(cache_path, component_name, None)
    why_none = "no calibration record"
    if record is not None and (record.passes < 1 or not record.max_abs):
        record, why_none = None, "a calibration record that observed no op"
    if record is not None and dag is not None and not record.matches(dag):
        record, why_none = None, (f"calibration {record.graph_signature} measured on another "
                                  f"graph than this one")
    if record is None:
        if forced is True:                               # :238-243
            return PrecisionContract(True, frozenset(), _pc.narrowable_op_uids(dag),
                                     f"{_pc.FLAG_ENV}=1 with {why_none}")
        return conservative_contract(why_none)
    if forced is not True and any(v == "conservative" for v in (record.prefer or {}).values()):
        return conservative_contract(
            f"calibration {record.graph_signature} prefers the conservative path on "
            f"{sorted(k for k, v in record.prefer.items() if v == 'conservative')}")
    import os
    from neurobrix.core.config.loader import get_precision_calibration_policy
    headroom = os.environ.get(_pc.HEADROOM_ENV)          # :263-266
    bound = _cal.island_bound(c, int(headroom) if headroom
                              else get_precision_calibration_policy()["headroom_bits"])
    pinned = _cal.islands_from_calibration(dag or {}, record.max_abs, bound)
    return PrecisionContract(True, frozenset(pinned), _pc.narrowable_op_uids(dag),
                             f"calibration {record.graph_signature}")


@dataclass(frozen=True)
class TilingView:
    """The op-level tiling plan as far as the widths care. `fusion_convs` maps each fused
    conv uid to its upsample uid; `tiled_ops` are the standalone tiled op uids. Built from
    an `OpLevelTilingPlan` by `from_plan`. Absent (None) = the plan is not known yet."""
    fusion_convs: Mapping[str, str]
    tiled_ops: FrozenSet[str]

    @classmethod
    def from_plan(cls, plan) -> "TilingView":
        return cls(fusion_convs={c: u for u, c, _tf in plan.fusion_pairs},
                   tiled_ops=frozenset(uid for uid, _t, _tf in plan.tiled_ops)
                   | frozenset(getattr(plan, "conv3d_chunks", ()) or ()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name(dtype) -> str:
    """A dtype NAME from a graph/meta string ("torch.float32" -> "float32"). Refuses what
    it does not know: a width guessed for an unknown name is the defect this module ends."""
    if dtype is None:
        raise ValueError("runtime_widths: a tensor with no traced dtype — its width is unknown")
    s = str(dtype).rsplit(".", 1)[-1]
    if s == "bool_":
        s = "bool"
    if s not in _KNOWN:
        raise ValueError(f"runtime_widths: unknown dtype name {dtype!r}")
    return s


def _nbx(name: str):
    from neurobrix.kernels.nbx_tensor import parse_dtype
    return parse_dtype(name)


def _nbx_name(nbx) -> str:
    """An NBXDtype's NAME. NBXDtype is an IntEnum: `str()` of a member is "NBXDtype.bfloat16" on
    Python 3.10 but the bare number ("1") on 3.11+, where IntEnum.__str__ became int.__str__ — the
    Mac's interpreter; the name is read from the member, never from its string."""
    name = getattr(nbx, "name", None)
    return _name(name if isinstance(name, str) else str(nbx))


def _wider(a: str, b: str) -> str:
    """wrappers.py:567-574 `_wider_dtype`: the higher rank; the first on a tie."""
    return a if _RANK[a] >= _RANK[b] else b


def _triton_remap(name: str, c: str, stores_fp64: bool) -> str:
    """An explicit dtype argument as the Triton branch resolves it —
    `TritonSequence._parse_dtype` (`_compile_arg`): the half dtypes follow the compute half,
    fp64 / complex128 are held as the branch holds them (`storage_dtype_name`)."""
    if name == "bfloat16" and c == "float16":
        return "float16"
    if name == "float16" and c == "bfloat16":
        return "bfloat16"
    return _tdt.storage_dtype_name(name, stores_fp64)


def _aten_kwarg_remap(name: str, c: str) -> str:
    """An explicit dtype kwarg on the ATen branch — `core/dtype/config.parse_dtype(s,
    compute_dtype=C)` (config.py:103-133, read by sequential_dispatcher.py:81-85): only the
    two half dtypes swap to the compute half."""
    if name == "bfloat16" and c == "float16":
        return "float16"
    if name == "float16" and c == "bfloat16":
        return "bfloat16"
    return name


def _aten_to_copy(name: str, c: str) -> str:
    """`DtypeEngine._make_to_copy` (core/dtype/engine.py:1055-1071): a half target becomes
    the compute dtype (whatever it is); fp32 and every other target are preserved."""
    return c if name in _HALF else name


def _explicit_dtype(op: Dict[str, Any]) -> Optional[str]:
    attrs = op.get("attributes") or {}
    for src in ((attrs.get("kwargs") or {}).get("dtype"), attrs.get("dtype")):
        if isinstance(src, dict) and src.get("type") == "dtype" and src.get("value"):
            return _name(src["value"])
    return None


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def runtime_widths(dag: Dict[str, Any], compute_dtype: str, engine: str, *,
                   has_native_bf16: bool, contract: PrecisionContract, stores_fp64: bool,
                   tiling: Optional[TilingView] = None,
                   shape_of: Optional[Callable[[str], List[int]]] = None) -> Dict[str, int]:
    """{tensor_id: bytes per element at runtime} for every tensor of `dag` — see
    `runtime_dtypes`."""
    return {tid: get_dtype_bytes_per_element(n) for tid, n in runtime_dtypes(
        dag, compute_dtype, engine, has_native_bf16=has_native_bf16, contract=contract,
        stores_fp64=stores_fp64, tiling=tiling, shape_of=shape_of).items()}


def runtime_dtypes(dag: Dict[str, Any], compute_dtype: str, engine: str, *,
                   has_native_bf16: bool, contract: PrecisionContract, stores_fp64: bool,
                   tiling: Optional[TilingView] = None,
                   shape_of: Optional[Callable[[str], List[int]]] = None) -> Dict[str, str]:
    """{tensor_id: dtype name at runtime}. `shape_of(tid)` answers a tensor's shape at the
    request (the matmul store rule reads M); absent, the traced shape answers. `stores_fp64`:
    whether the engine priced holds float64 / complex128 on the profile, read by its OWN branch
    (`triton.dtype.triton_stores_fp64` for the Triton engines, `core.dtype.config.
    device_supports_fp64` for ATen). The ATen rules do not narrow on it yet: complex128 is priced
    wide where the compiled branch narrows on a device without fp64 (Apple) — named, not hidden."""
    c = str(compute_dtype).replace("torch.", "")
    if c not in _FLOAT:
        raise ValueError(f"runtime_widths: compute dtype {compute_dtype!r} is not a float dtype")
    if engine in TRITON_ENGINES:
        rule = _TritonRules(dag, c, engine, has_native_bf16, contract, tiling, stores_fp64)
    elif engine in ATEN_ENGINES:
        rule = _AtenRules(dag, c, engine, has_native_bf16, contract, tiling, stores_fp64)
    else:
        raise ValueError(f"runtime_widths: engine {engine!r} is none of "
                         f"{sorted(TRITON_ENGINES | ATEN_ENGINES)}")
    tensors = dag.get("tensors") or {}
    ops = dag.get("ops") or {}

    def traced(tid: str) -> str:
        meta = tensors.get(tid)
        if meta is None:
            raise ValueError(f"runtime_widths: tensor {tid!r} is not in the graph's tensor table")
        return _name(meta.get("dtype"))

    def traced_shape(tid: str) -> List[int]:
        return list((tensors.get(tid) or {}).get("shape") or [])

    rule.shape_of = shape_of or traced_shape
    rule.traced_shape = traced_shape

    out: Dict[str, str] = {}
    # SEEDS — tensors no op produces.
    produced = {t for op in ops.values() for t in (op.get("output_tensor_ids") or [])}
    graph_inputs = set(dag.get("input_tensor_ids") or [])
    for tid, meta in tensors.items():
        if tid in produced:
            continue
        name = traced(tid)
        if name not in _FLOAT:
            out[tid] = name
        elif tid in graph_inputs or meta.get("is_input"):
            # A component input is cast to C (TritonDtypeEngine.cast_runtime_inputs,
            # triton/dtype.py:427-487; GraphExecutor._prepare_execution on the ATen branch);
            # a SEAM enters in the dtype its producer gave it (dtype.py:467-476).
            out[tid] = name if is_seam_tensor(meta) else c
        else:
            out[tid] = rule.weight_dtype(tid, name)

    for uid in dag.get("execution_order") or []:
        op = ops.get(uid)
        if op is None:
            raise ValueError(f"runtime_widths: execution_order names {uid!r}, absent from ops")
        outs = op.get("output_tensor_ids") or []
        ins = [t for t in (op.get("input_tensor_ids") or [])]
        for t in ins:
            if t not in out:
                raise ValueError(f"runtime_widths: op {uid} reads {t!r} before any op produced it")
        result = None
        for tid in outs:
            name = traced(tid)
            if name not in _FLOAT and name not in _COMPLEX:
                out[tid] = name
                continue
            if name in _COMPLEX:
                out[tid] = rule.complex_dtype(name)
                continue
            if result is None:
                result = rule.op_dtype(uid, op, ins, out)
            out[tid] = result
    return out


class _Rules:
    """What both engines share: the operand views every rule reads."""

    def __init__(self, dag, c, engine, has_native_bf16, contract, tiling, stores_fp64):
        self.dag, self.c, self.engine = dag, c, engine
        self.stores_fp64 = bool(stores_fp64)
        self.half = c in _HALF
        # The component's GRAPH dtype — the container's traced `torch_dtype`, the source
        # both engines read (TritonSequence / TritonSequentialDispatcher / DtypeEngine
        # `graph_dtype`): the AMP_FP32 cast-back rule reads it under bf16 compute.
        self.graph_dtype = _tdt.graph_dtype_name(dag.get("torch_dtype"))
        self.native_bf16 = bool(has_native_bf16)
        self.contract = contract
        self.tiling = tiling
        self.tensors = dag.get("tensors") or {}
        self.ops = dag.get("ops") or {}
        # Structural fusion candidates: an upsample whose ONLY consumer is a conv — the
        # eligibility test of `_detect_op_level_tiling_pairs` minus its size gate
        # (solver.py:1644-1660), used only when no plan is known.
        consumers: Dict[str, List[str]] = {}
        for u, op in self.ops.items():
            for t in op.get("input_tensor_ids") or []:
                consumers.setdefault(t, []).append(u)
        self.fusable: Dict[str, str] = {}
        for u, op in self.ops.items():
            if "upsample" not in op.get("op_type", ""):
                continue
            o = op.get("output_tensor_ids") or []
            if len(o) != 1:
                continue
            cons = consumers.get(o[0], [])
            if len(cons) == 1 and "convolution" in self.ops[cons[0]].get("op_type", ""):
                self.fusable.setdefault(cons[0], u)

    # -- operand views ---------------------------------------------------
    def floats(self, ins: Iterable[str], w: Dict[str, str]) -> List[Tuple[str, str, bool]]:
        """[(tid, runtime dtype, is 0-dim)] of the float tensor operands, in order."""
        r = []
        for t in ins:
            d = w[t]
            if d in _FLOAT:
                r.append((t, d, len(self.traced_shape(t)) == 0))
        return r

    def widest(self, fl) -> Optional[str]:
        """The widest float operand, 0-dim operands excluded while a dimensioned one exists
        (wrappers.py:524-530 `_is_scalar`: a 0-d tensor is the scalar of a binary op and the
        output is `empty_like` of the tensor, :596-615; torch's promotion does the same)."""
        dims = [d for _t, d, z in fl if not z] or [d for _t, d, _z in fl]
        if not dims:
            return None
        acc = dims[0]
        for d in dims[1:]:
            acc = _wider(acc, d)
        return acc

    def default(self, op, fl) -> str:
        """An op no rule below names: the widest float operand (the elementwise/unary/layout
        wrappers allocate like their input, the binary ones like the wider operand). An op
        with no float operand writes its traced dtype through the engine's remap. This is
        the CONSERVATIVE choice for an unclassified op: never narrower than an operand."""
        w = self.widest(fl)
        if w is not None:
            return w
        return self.remap_explicit(_name(self.tensors[op["output_tensor_ids"][0]]["dtype"]))

    def first(self, fl) -> Optional[str]:
        return fl[0][1] if fl else None

    def first_dimensioned(self, fl) -> Optional[str]:
        for _t, d, z in fl:
            if not z:
                return d
        return self.first(fl)

    def first_nonempty(self, fl) -> Optional[str]:
        """The first operand `_cat_inputs_or_refuse` keeps: 0-dim AND empty operands dropped, the
        extent read at the binding (`shape_of`). An empty fp16 KV-cache constant concatenated with
        a bf16 projection made K and V fp16 here while the run kept them bf16 (Gemma's encoder in
        SANA-Video under bf16 compute; the SDPA's operands then widened to fp32 in the census)."""
        for t, d, z in fl:
            if not z and all(int(e) != 0 for e in self.shape_of(t)):
                return d
        return self.first_dimensioned(fl)

    def weight_dtype(self, tid: str, traced: str) -> str:
        """A param/buffer/constant: `stored_dtype_in_compute` (triton/weight_loader.py:609-627),
        the loader's rule for every weight of a shard — the documented mirror of the ATen
        branch's WeightLoader(torch_dtype=C)."""
        return _nbx_name(stored_dtype_in_compute(_nbx(traced), _nbx(self.c)))

    def mm_store(self, a: str, b: str, m: Optional[int]) -> str:
        """The Triton mm/addmm store dtype — `mm` operand alignment (wrappers.py:2178-2224)
        then `_matmul_out_dtype` (:1798-1861): fp32 if either operand is fp32; fp16 on
        hardware without native bf16 -> fp32; a half with M <= 4 -> fp32; else the half."""
        if "float32" in (a, b) or "float64" in (a, b):
            return "float32"
        a_eff = "bfloat16" if "bfloat16" in (a, b) else a      # :2217-2224 widest of the halves
        if a_eff == "float16" and not self.native_bf16:
            return "float32"
        if m is not None and m <= 4:
            return "float32"
        return a_eff

    def rows(self, tid: str) -> Optional[int]:
        sh = self.shape_of(tid)
        if not sh:
            return None
        m = 1
        for d in sh[:-1]:
            m *= int(d)
        return m

    def fused_input(self, uid: str) -> Optional[str]:
        """The upsample of a fused conv: the plan's pair when a plan is known, else the
        structural candidate (priced at the wider of fused and unfused by the caller)."""
        if self.tiling is not None:
            return self.tiling.fusion_convs.get(uid)
        return self.fusable.get(uid)

    def is_tiled(self, uid: str, op: Dict[str, Any]) -> bool:
        """A standalone tiled op: the plan's when known; else every op the detector could
        tile — a rank-4 non-depthwise conv (solver.py:1729-1768) or a custom::rms_norm
        (:1777-1807) — without its size gate, which reads the placement."""
        if self.tiling is not None:
            return uid in self.tiling.tiled_ops
        t = op.get("op_type", "")
        if t == "custom::rms_norm":
            return True
        if "convolution" in t:
            shp = op.get("input_shapes") or []
            if not shp or shp[0] is None or len(shp[0]) != 4:
                return False
            return not (len(shp) >= 2 and len(shp[1]) == 4 and int(shp[1][1]) == 1)
        return False


class _TritonRules(_Rules):
    """`TritonDtypeEngine.wrap_op` (triton/dtype.py:505-586) in its order, then the wrapper
    the op lands in. Both Triton engines; their differences are named where they occur."""

    def __init__(self, *a):
        super().__init__(*a)
        self.tseq = self.engine == "triton_sequential"
        islands = self.contract.fp32_op_uids if self.half else ()
        narrow = self.contract.narrow_op_uids if self.half else ()
        self.fp32_constants = {
            n for n in _tdt.fp32_constant_names(self.dag, narrow, islands)}

    def complex_dtype(self, name: str) -> str:
        # A complex output keeps the traced complex type; NBX complex is complex64 at most
        # (sequence.py:2688-2689; `_wrap_complex_output`, dtype.py:527-528, 624-654).
        return _triton_remap(name, self.c, self.stores_fp64)

    def remap_explicit(self, name: str) -> str:
        return _triton_remap(name, self.c, self.stores_fp64)

    def weight_dtype(self, tid: str, traced: str) -> str:
        # GraphExecutor._bind_fp32_constants (graph_executor.py:2063-2107) binds the
        # constants of fp32-computing consumers in fp32, once, Triton engines only.
        name = tid.split("::", 1)[1] if "::" in tid else tid
        if name in self.fp32_constants:
            return "float32"
        meta = self.tensors.get(tid) or {}
        if meta.get("constant") and meta.get("constant_data") and not meta.get("is_computable"):
            # An embedded constant is bound by the loader's own rule (`constant_load_dtype`,
            # GraphExecutor._load_constant_triton): its traced dtype, a bf16 one decoded to the
            # half compute dtype — swin2SR's fp32 coordinates table stayed fp32, canary's bf16
            # positional table became fp16 (the census walk).
            return _tdt.constant_load_dtype(traced, self.c, self.stores_fp64)
        return super().weight_dtype(tid, traced)

    def amp_fp32_out(self, op=None, narrowed: bool = False) -> str:
        # `_wrap_fp32_internal_compute_dtype_output`: fp32 compute; the output is the engine's
        # own rule, CALLED — `amp_fp32_output_dtype`: C = bf16 -> bf16 on a bf16 graph, fp32
        # on another; C = fp16 -> C only when the component is activations_fp16_safe
        # (`_NBX_ACTIVATIONS_FP16_SAFE`, set from the contract, sequence.py:3025-3029) or the
        # op is in the record's narrow set, else fp32. Under fp16 the cast back is the
        # single-tensor one: a TUPLE result (`native_layer_norm`, `native_group_norm` — the
        # op's traced outputs are several) keeps the fp32 its kernel wrote from the fp32
        # inputs (triton/dtype.py `cast_back`, the fp16 contract held byte-identical by
        # decision 2026-09-28). Measured by the census walk: Kokoro's albert projections and
        # PixArt's VAE attention read the norm's output in fp32.
        # The op's traced output dtype is the floor of the cast back: the vendor's own fp32
        # island in a half graph is narrowed by the narrow set only, never by the flag.
        r = _tdt.amp_fp32_output_dtype(self.c, self.graph_dtype, self.contract.safe, narrowed,
                                       _tdt.traced_output_dtype_name(op))
        if (self.c == "float16" and op is not None
                and len(op.get("output_tensor_ids") or []) > 1):
            return "float32"
        return r

    def op_dtype(self, uid, op, ins, w) -> str:
        c = self.c
        op_type = op.get("op_type", "")
        name = canonical_aten(op_type.split("::")[-1].split(".")[0])
        fl = self.floats(ins, w)
        explicit = _explicit_dtype(op)

        # A fused conv writes its output at its PRE-INPUT's dtype:
        # `_fused_upsample_conv2d_nbx` (kernels/ops/fused_upsample_conv.py:804-807). Its
        # interceptor (tiling_engine.py:1494-1504) is not self_manages_dtype, so wrap_op
        # applies — but the proxy is no float tensor and passes every cast untouched.
        up = self.fused_input(uid) if name in _CONV else None
        fused = None
        if up is not None:
            pre = (self.ops[up].get("input_tensor_ids") or [None])[0]
            fused = w[pre] if pre in w and w[pre] in _FLOAT else c
            if self.tiling is not None:
                return fused

        r = self._rule(uid, op, name, fl, explicit, ins)
        if fused is not None:                     # plan unknown: the wider of the two
            r = _wider(r, fused)
        return r

    def _rule(self, uid, op, name, fl, explicit, ins) -> str:
        c = self.c
        # Casts and creations carry the dtype they are told (sequence.py:2580-2582).
        if explicit is not None and (name in _CASTS or not fl or name.endswith("_like")
                                     or name.startswith("new_")):
            return self.remap_explicit(explicit)
        # dtype.py:533-535 — a contract island computes AND stores fp32 whatever its class
        # (`_wrap_fp32` overrides `_NBX_COMPUTE_DTYPE` for a self-managed conv, :573-610).
        # triton_sequential wraps `custom::rms_norm` with no op_uid (sequential.py:176-179):
        # islands and the narrow set never reach it there.
        seq_rms = self.tseq and op.get("op_type") == "custom::rms_norm"
        if self.half and not seq_rms and uid in self.contract.fp32_op_uids:
            return "float32"
        if (self.half and not seq_rms and uid in self.contract.narrow_op_uids
                and name in _tdt.AMP_FP32_OPS):                            # :536-537
            return self.amp_fp32_out(op, narrowed=True)
        r = self._class_rule(uid, op, name, fl, ins)
        if explicit is not None:
            r = _wider(r, self.remap_explicit(explicit))
        return r

    def _class_rule(self, uid, op, name, fl, ins) -> str:
        c = self.c
        if name in _tdt.AMP_SCALAR_FILL_OPS:                               # :545-546
            return self.first(fl) or self.default(op, fl)
        if name == "where":
            # `where_wrapper` allocates `empty_like(x)` (wrappers.py:1491-1504).
            return self.first(fl) or self.default(op, fl)
        if name in _tdt._SELF_MANAGED_OPS:                                  # :551-552
            return self._self_managed(uid, op, name, fl, ins)
        if not self.half:                                                   # :555-556
            return self._unwrapped(uid, op, name, fl, ins)
        if name in _tdt.AMP_FP32_OPS:                                       # :558-564
            r = self.amp_fp32_out(op)
            if self.tseq and name == "rms_norm" and self.is_tiled(uid, op):
                # A tiled rms_norm in triton_sequential runs unwrapped: the NBX wrapper at
                # x's dtype (fused_upsample_conv.py:574-615, graph_executor.py:3217-3230).
                r = _wider(r, self.first(fl) or r)
            return r
        if name in _tdt.AMP_FP16_OPS:                                       # :566-573
            if c == "float16" and name in _tdt._FP16_NEED_FP32:
                return self.amp_fp32_out(op)
            return self._lower_precision(uid, op, name, fl, ins)
        if name in _tdt.AMP_PROMOTE_OPS:                                    # :575-576, 761-781
            return self.widest(fl) or self.default(op, fl)
        return self._unwrapped(uid, op, name, fl, ins)

    def _conv_out(self, uid, op, fl) -> str:
        # conv2d_wrapper writes `_NBX_COMPUTE_DTYPE` (wrappers.py:4171-4187). A tiled conv
        # in triton_sequential runs unwrapped at its INPUT's dtype
        # (fused_upsample_conv.py:691-694, graph_executor.py:3217-3230); in triton the
        # wrap narrows the input to C first, so the tiled conv writes C as well.
        if self.tseq and self.is_tiled(uid, op):
            return _wider(self.c, self.first(fl) or self.c)
        return self.c

    def _self_managed(self, uid, op, name, fl, ins) -> str:
        if name in ("conv2d", "_convolution"):
            return self._conv_out(uid, op, fl)
        if name in ("upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d"):
            return self.first(fl) or self.default(op, fl)      # wrappers.py:3375-3425
        if name == "bmm":
            return "float32"                                   # wrappers.py:2389 force_fp32
        if name in ("mm", "mm_epilogue"):
            a, b = (fl + [(None, self.c, False)] * 2)[:2]
            return self.mm_store(a[1], b[1], self.rows(ins[0]))
        if name in ("addmm", "addmm_epilogue"):
            # addmm(bias, a, b): the store follows a (wrappers.py:2563-2680)
            fa = [d for t, d, _z in fl if t in ins[1:3]]
            a = fa[0] if fa else self.c
            b = fa[1] if len(fa) > 1 else a
            return self.mm_store(a, b, self.rows(ins[1]) if len(ins) > 1 else None)
        return self.default(op, fl)

    def _lower_precision(self, uid, op, name, fl, ins) -> str:
        """`_wrap_lower_precision` (dtype.py:747-759): float operands cast to C, then the
        wrapper decides the store."""
        c = self.c
        if name in _CONV:
            return self._conv_out(uid, op, fl)
        if name in ("matmul", "linear"):
            # matmul_wrapper (wrappers.py:2432-2475): 2-D x 2-D -> mm; 2-D x 1-D -> mv
            # (M = 1, fp32 for a half); every batched form -> bmm (force_fp32).
            # linear_wrapper (:2540-2560) goes through matmul_wrapper with a 2-D weight.
            ranks = [len(self.traced_shape(t)) for t in ins[:2]]
            if name == "linear" and ranks and ranks[0] == 2:
                return self.mm_store(c, c, self.rows(ins[0]))
            if name == "matmul" and ranks == [2, 2]:
                return self.mm_store(c, c, self.rows(ins[0]))
            return "float32"
        if name in _MATMUL_OTHER:
            return "float32"
        return c

    def _unwrapped(self, uid, op, name, fl, ins) -> str:
        if name in ("mm", "addmm", "bmm", "matmul", "linear") or name in _MATMUL_OTHER:
            # C fp32: no AMP; the wrappers store as their operands (fp32 in -> fp32).
            return self._self_managed(uid, op, name, fl, ins) if name in (
                "mm", "addmm", "bmm") else (self.widest(fl) or self.c)
        if name in _CONV:
            return self._conv_out(uid, op, fl)
        if name in _CAT:
            # NBXTensor.cat aligns every operand to the FIRST one's dtype
            # (kernels/nbx_tensor.py:4251-4253), after the empty and 0-dim operands are
            # dropped (triton/sequential.py `_cat_inputs_or_refuse`).
            return self.first_nonempty(fl) or self.default(op, fl)
        if name in _SDPA:
            # The flash path allocates `empty_like(q)`, the math path casts its result to q's
            # dtype — q AFTER the wrapper's operand alignment, read from the one function the
            # wrapper calls (`launch_keys.sdpa_operand_dtypes`: q, k, v that disagree all take the
            # narrowest), so Wan's cross-attention (fp16 q/k, fp32 v) writes fp16.
            if len(fl) >= 3 and len({d for _t, d, _z in fl[:3]}) > 1:
                from neurobrix.kernels import launch_keys as _lk
                from neurobrix.kernels.nbx_tensor import NBXDtype
                return _lk.sdpa_operand_dtypes(*(NBXDtype[d] for _t, d, _z in fl[:3]))[0].name
            return self.first(fl) or self.default(op, fl)
        if name == "rms_norm":                                  # C fp32: unwrapped wrapper
            return self.first(fl) or self.default(op, fl)
        return self.default(op, fl)


class _AtenRules(_Rules):
    """`DtypeEngine.compile_op` (core/dtype/engine.py:665-817) in its order, then torch's own
    promotion for what it leaves unwrapped."""

    def complex_dtype(self, name: str) -> str:
        # `_make_complex_output_wrapper` is a floor, never a leveller (engine.py:859-887):
        # the traced complex type is kept, complex128 included.
        return name

    def remap_explicit(self, name: str) -> str:
        return _aten_kwarg_remap(name, self.c)

    def op_dtype(self, uid, op, ins, w) -> str:
        c = self.c
        op_type = op.get("op_type", "")
        name = op_type.split("::")[-1].split(".")[0]      # config.strip_aten_prefix (:136-150)
        fl = self.floats(ins, w)
        explicit = _explicit_dtype(op)
        if name == "_to_copy":                                              # :680-681
            traced = _name((op.get("output_dtypes") or [None])[0]
                           or self.tensors[op["output_tensor_ids"][0]]["dtype"])
            return _aten_to_copy(traced, c)
        if explicit is not None and (name in _CASTS or not fl or name.endswith("_like")
                                     or name.startswith("new_")
                                     or name in ATEN_AMP_CREATION_FILL_OPS):
            return self.remap_explicit(explicit)                            # :700-701
        if uid in self.contract.fp32_op_uids:                               # :708-711
            return "float32"
        r = self._class_rule(uid, op, name, fl, ins)
        if explicit is not None:
            r = _wider(r, self.remap_explicit(explicit))
        return r

    def _class_rule(self, uid, op, name, fl, ins) -> str:
        c = self.c
        if name in _CONV and self.fused_input(uid) is not None:
            # `_fused_upsample_conv2d_torch` allocates at weight.dtype
            # (kernels/ops/fused_upsample_conv.py:299-303): the weights are at C.
            return c
        if name == "rms_norm":
            # compiled_ops.py:294-295, 551-568 -> engine.py:26-50 `rms_norm_fp32`: fp32
            # inside, the output at the WEIGHT's dtype. It is in no AMP class
            # (strip_aten_prefix gives "rms_norm"). A tiled one allocates at x's dtype
            # (fused_upsample_conv.py:638).
            weight = [d for t, d, _z in fl[1:2]]
            r = weight[0] if weight else (self.first(fl) or c)
            if self.is_tiled(uid, op):
                r = _wider(r, self.first(fl) or r)
            return r
        if self.half:
            contract = self.contract.safe and c == "float16"               # :725-727
            if name in ATEN_AMP_FP32_OPS:                                   # :728-753
                if contract and name in ATEN_FP32_OPS_HALF_IO:
                    return c
                # The store is the engine's `amp_fp32_output_dtype(c, graph, False, narrowed)`
                # (engine.py; its torch-free twin in triton/dtype.py is CALLED here — a test
                # holds the twins equal): bf16 -> bf16 on a bf16 graph, fp32 on another;
                # fp16 -> C only when narrowed.
                return _tdt.amp_fp32_output_dtype(
                    c, self.graph_dtype, False, contract and uid in self.contract.narrow_op_uids)
            if name in ATEN_AMP_FP16_OPS:                                   # :754-791
                if c == "float16" and name in ATEN_FP16_NEED_FP32:
                    if contract and name in ATEN_FP16_GEMM_OPS:
                        return c
                    if contract and uid in self.contract.narrow_op_uids:
                        return c                                            # div, narrowed
                    return "float32"
                return c
            if name in ATEN_AMP_PROMOTE_OPS:                                # :792-793
                return self.widest(fl) or self.default(op, fl)
            if name == "mul" and c == "float16":                            # :799-810, 919-972
                a = (op.get("input_tensor_ids") or [None, None])
                if len(a) >= 2 and a[0] == a[1] and fl and fl[0][1] == "float16":
                    return "float32"
        if name in _SDPA:
            # compiled_ops `_make_attention` hands q, k, v to F.sdpa: the output is q's.
            return self.first(fl) or self.default(op, fl)
        # Everything else is unwrapped: torch's promotion (the widest operand; cat promotes).
        return self.default(op, fl)
