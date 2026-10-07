"""Triton DtypeEngine — AMP rules for triton mode. Zero torch dependency.

Same AMP logic as core/dtype/engine.py but uses NBXDtype instead of torch.dtype.
Wraps op functions with input casting for numerical stability.

Rules (from PyTorch AT_FORALL_FP32 / AT_FORALL_LOWER_PRECISION_FP):
  - FP32 ops: upcast inputs to fp32 (pow, rsqrt, softmax, layernorm, ...)
  - FP16 ops: cast inputs to compute_dtype (mm, conv, bmm, ...)
  - FP16_NEED_FP32: on fp16 hardware, mm/bmm/div/addmm need fp32
  - Promote ops: promote to widest input dtype
"""

from typing import Callable, FrozenSet, Optional

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor


# ============================================================================
# AMP OP SETS — identical to core/dtype/engine.py
# ============================================================================

AMP_FP32_OPS: FrozenSet[str] = frozenset({
    "acos", "asin", "cosh", "erfinv", "exp", "expm1",
    "log", "log10", "log2", "log1p", "reciprocal", "rsqrt",
    "sinh", "tan", "pow", "softplus",
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "cudnn_batch_norm", "instance_norm",
    # Phase 2 — rms_norm is a NeuroBrix custom reduction op (not in
    # PyTorch's AT_FORALL_FP32 because PyTorch has no rms_norm). Its
    # internal pow→mean→rsqrt chain is overflow-prone in fp16 (squared
    # values overflow fp16 max above ~256), so it MUST run fp32-internal.
    # Treated as AMP_FP32 here so the unified cast-back path (read
    # activations_fp16_safe at call time) governs whether the output is
    # cast back to compute_dtype or stays fp32.
    "rms_norm",
    "frobenius_norm", "nuclear_norm", "cosine_similarity",
    "poisson_nll_loss", "cosine_embedding_loss", "nll_loss",
    "mse_loss", "smooth_l1_loss", "huber_loss",
    # FALLBACK only — the primary protection is `traced_output_is_complex`
    # (container output_dtypes). See the note in core/dtype/engine.py: do not
    # try to complete this pair, and note this set LEVELS to fp32 while the
    # complex rule is a FLOOR that preserves float64.
    "polar", "view_as_complex",
    "renorm", "logsumexp",
    # Phase 1 — Removed nearest variants from AMP_FP32_OPS (mirror PyTorch
    # convention was gratuitous: pure index lookup, dtype passthrough, no
    # compute = no precision risk). Kept linear/bilinear/bicubic which do
    # interpolate in float and may benefit from fp32 precision.
    "upsample_linear1d", "upsample_bilinear2d", "upsample_bicubic2d",
    "prod", "softmax", "_softmax", "log_softmax",
    "cumprod", "cumsum", "sum",
    "linalg_vector_norm", "linalg_matrix_norm",
})

# UNIFORM cast-back doctrine: every AMP_FP32_OPS goes through the
# fp32-internal-compute-then-cast-back wrap, and ONE rule decides its output
# dtype — `amp_fp32_output_dtype` below (the compiled DtypeEngine carries its
# torch-free twin; a test holds the two equal):
#   * compute dtype bf16 on a bf16 GRAPH (the container's `torch_dtype`): the
#     output is cast back to bf16, always. The op still computes in fp32; its
#     fp32 result always fits bf16's range (bf16 has fp32's exponent), so the
#     fp16 overflow reason for keeping it fp32 does not apply — and the
#     vendor's graph returns x.dtype, bf16 there (Sana's DC-AE RMSNorm computes
#     fp32 inside and returns x.dtype). Keeping it fp32 spread fp32 through the
#     residual stream: twice the activation memory and not the vendor's
#     numerics.
#   * compute dtype bf16 on a graph of ANOTHER dtype (an fp32 graph the
#     profile's preferred dtype coerced to bf16 compute): x.dtype was fp32 in
#     the vendor's forward, so the output stays fp32 (Kokoro moved farther from
#     its fp32 reference with the cast-back, Apple gate 2026-09-28).
#   * compute dtype fp16: the output is cast back only under the precision
#     contract — the per-component `activations_fp16_safe` flag (the
#     component's calibration record, resolved by
#     core/runtime/precision_contract.resolve, read at call time via
#     _w._NBX_ACTIVATIONS_FP16_SAFE) or the op's membership of the record's
#     narrow set. Otherwise (default, conservative — no record, or a record
#     whose islands this engine cannot pin per op) the output stays fp32.
# A contract island (`_wrap_fp32`) is decided BEFORE this rule and keeps its
# fp32 output; under bf16 there is no contract, hence no island.
# The previous `_AMP_FP32_OPS_OPT_IN_CAST_BACK` set was an additional
# membership gate that fragmented the doctrine: rms_norm and div had
# the cast-back hook but rsqrt/exp/layer_norm/batch_norm/etc did not.
# Removed in favor of a single uniform gate.

AMP_FP16_OPS: FrozenSet[str] = frozenset({
    "_convolution", "conv1d", "conv2d", "conv3d", "convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "addmm", "addmv", "addr", "matmul", "einsum",
    "mm", "mv", "bmm", "addbmm", "baddbmm",
    "linear", "prelu", "div",
})

# Must stay identical to core/dtype/engine.py _FP16_NEED_FP32.
# mm / bmm / addmm handle their own fp32 output internally (see wrappers.py):
# the kernel accumulates in fp32 and is instructed to store into an fp32 output
# buffer when inputs are half-precision. This keeps inputs in fp16 (zero copy,
# pre-transpose intact, M<=4 mv routing stays valid) while avoiding the
# fp32-accumulator→fp16-store overflow that broke Qwen3-30B on V100. So these
# ops no longer need the wrapper's blanket input-upcast path.
# Keeping div for epsilon underflow (1e-15 → 0 in fp16).
_FP16_NEED_FP32: FrozenSet[str] = frozenset({"div"})

AMP_PROMOTE_OPS: FrozenSet[str] = frozenset({
    "addcdiv", "addcmul", "atan2", "bilinear", "cross",
    "dot", "vdot", "grid_sampler", "grid_sampler_2d", "grid_sampler_3d", "index_put",
    # grid_sampler_2d/3d: the ATen names the trace carries (the autocast
    # table lists the composite `grid_sampler`); input and grid must agree
    # on the widest float type — under the contract the sampled features
    # arrive fp16 and the grid fp32 (GLM-4.1V visual, 2026-09-05).
    "scatter_add", "tensordot", "linalg_cross",
})

_FLOATING = frozenset({NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64})

# ---------------------------------------------------------------------------
# Saturating narrowing of scalars and one-element constants — the R30 mirror of
# core/dtype/engine.py's AMP_SCALAR_FILL_OPS / AMP_CREATION_FILL_OPS clamp
# (duplicated, not imported: that module imports torch, R33), extended to the
# one-element operand of `where`.
#
# A finite value beyond a half dtype's range (a finfo(fp32).min mask sentinel a
# graph traced in fp32 carries as a literal) becomes +-inf when it is written
# into that dtype. The vendor, running in the half dtype, uses finfo(half).min,
# which is finite. Measured 2026-09-26 on PixArt-XL-2-1024-MS's T5: the mask's
# `aten.where::0` filled 13 200 positions with -inf in bf16 where the ATen branch
# kept -3.4028e38 in fp32; ten components of the catalogue carry such literals
# (six text encoders at finfo(fp32).min, four at finfo(bf16).min, infinite in fp16).
# A fully masked row then softmaxes to NaN. Saturation writes the half dtype's own
# extreme instead, which is what the vendor's value is.
# ---------------------------------------------------------------------------
AMP_SCALAR_FILL_OPS: FrozenSet[str] = frozenset({
    "masked_fill", "masked_fill_", "fill", "fill_", "index_fill", "index_fill_",
})
AMP_CREATION_FILL_OPS: FrozenSet[str] = frozenset({"full", "new_full", "full_like"})
_HALF_MAX = {NBXDtype.float16: 65504.0, NBXDtype.bfloat16: 3.3895313892515355e38}


def saturate_scalar(value, dtype):
    """A Python number clamped to the finite range of a half `dtype`; anything else unchanged
    (a bool, a non-half dtype, a NaN, an infinity the graph wrote on purpose)."""
    top = _HALF_MAX.get(dtype)
    if top is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        return value
    return max(-top, min(top, v))


def _saturated_one_element(t, out_dt):
    """`t` itself, unless it is a one-element float tensor wider than `out_dt` holding a finite
    value beyond `out_dt`'s range — then a one-element tensor of `t`'s dtype holding the
    saturated value, same shape."""
    if not _is_float_tensor(t) or t.numel() != 1:
        return t
    if _get_nbx_dtype(t) not in (NBXDtype.float32, NBXDtype.float64):
        return t
    import numpy as np
    v = float(np.asarray(t.numpy()).reshape(-1)[0])
    sv = saturate_scalar(v, out_dt)
    if sv == v:
        return t
    arr = np.full(tuple(t.shape), sv, dtype=np.float64 if _get_nbx_dtype(t) == NBXDtype.float64 else np.float32)
    return NBXTensor.from_numpy(arr)



# Mirror of core/dtype/engine.py's predicate. Duplicated, not imported: that
# module imports torch, and the Triton branch loads no torch at import or at
# execution (R33). The two-modes doctrine keeps these paths parallel by
# construction — AMP_FP32_OPS / AMP_FP16_OPS above are duplicated for the same
# reason. Dtype crosses this boundary as a STRING, which is all the predicate
# reads, so the two copies cannot drift on representation.
_COMPLEX_DTYPE_NAMES = frozenset({
    "complex32", "complex64", "complex128", "chalf", "cfloat", "cdouble",
})


def traced_output_dtype_name(op_meta) -> Optional[str]:
    """The plain name of the op's first traced output dtype (`output_dtypes[0]` at the
    record's TOP level, as `traced_output_is_complex` reads it), or None when the record
    states none."""
    if not isinstance(op_meta, dict):
        return None
    dts = op_meta.get("output_dtypes") or ()
    if not dts:
        return None
    return str(dts[0]).rsplit(".", 1)[-1]


def traced_output_is_complex(op_meta) -> bool:
    """True when the CONTAINER types this op's output as complex.

    See core/dtype/engine.traced_output_is_complex for the full rationale: the
    op's NAME cannot answer this (a plain `aten::mul` produces Kokoro's
    complex64) and neither can its INPUTS (the other operand is the Python
    scalar `1j`, absent from input_dtypes). NBXDtype carries complex64 and
    complex128 — Wan2.1's RoPE freqs are complex128 — so the Triton branch
    needs the same floor as the ATen branch.

    `op_meta` carries the op's traced dtypes — the op's graph record here.
    Only `output_dtypes` is read, at that dict's TOP level.
    """
    if not isinstance(op_meta, dict):
        return False
    # Last dotted segment — see the core copy: it makes the two branches read
    # a plain string, a torch.dtype and an NBXDtype identically, so the
    # duplication cannot drift on representation.
    return any(str(d).rsplit(".", 1)[-1] in _COMPLEX_DTYPE_NAMES
               for d in (op_meta.get("output_dtypes") or ()))


def _is_float_tensor(a) -> bool:
    """Check if a is a floating-point tensor (NBXTensor or duck-typed)."""
    if not hasattr(a, 'is_floating_point'):
        return False
    return a.is_floating_point()


def _get_nbx_dtype(a) -> NBXDtype:
    """Get NBXDtype from tensor."""
    if hasattr(a, 'nbx_dtype'):
        return a.nbx_dtype
    return NBXDtype.float32


_GRAPH_FLOAT_NAMES: FrozenSet[str] = frozenset({"float16", "bfloat16", "float32", "float64"})
_HALF_GRAPH_NAMES: FrozenSet[str] = frozenset({"float16", "bfloat16"})


def amp_fp32_output_dtype(compute_dtype: str, graph_dtype: Optional[str], safe: bool,
                          narrowed: bool, traced: Optional[str] = None) -> str:
    """The OUTPUT dtype (a name) of an op computed fp32-internal under the half compute
    dtype `compute_dtype` (a name: "bfloat16" or "float16") — THE cast-back rule of the
    AMP_FP32 class (see the doctrine above AMP_FP16_OPS).

    `graph_dtype` is the component's GRAPH dtype — the container's traced `torch_dtype`
    (graph.json top level), the dtype its vendor's forward ran in. Read under bf16 only:
    the op returns bf16 when the graph is bf16 (the vendor's forward returns x.dtype, and
    x is bf16 there — Sana's DC-AE RMSNorm); on a graph of another dtype that the
    profile's preferred dtype coerced to bf16 compute, x.dtype was fp32 in the vendor's
    forward and the output stays fp32 (measured 2026-09-28 on the Mac's Apple gate:
    Kokoro's fp32 graph moved FARTHER from its fp32 reference with a bf16 cast-back).
    A graph dtype the container does not state is refused by name under bf16.

    `safe` is the component's `activations_fp16_safe` flag as THIS engine reads it for the
    cast back; `narrowed` says the op is in the record's narrow set (the compiled engine
    passes `safe=False`: its flag alone never narrows an fp32-class output). Both are read
    only under fp16 — the contract does not exist under bf16
    (precision_contract.resolve returns (False, set(), set())). `traced` is the op's traced
    output dtype name (its graph record's `output_dtypes[0]`): float32 on a half graph is
    the vendor's own fp32 island, which `safe` never narrows — only `narrowed` does.

    Pure, torch-free and device-free: the Triton wrapper, its compiled twin
    (core/dtype/engine.py `amp_fp32_output_dtype`) and Prism's width pass
    (core/prism/runtime_widths.py) all ask it. A compute dtype that is not half is refused
    by name — no AMP wrap exists there, so a caller asking is a caller in error."""
    if compute_dtype == "bfloat16":
        if graph_dtype not in _GRAPH_FLOAT_NAMES:
            raise ValueError(
                f"amp_fp32_output_dtype: under bfloat16 compute the output follows the "
                f"component's graph dtype, and {graph_dtype!r} is none of "
                f"{sorted(_GRAPH_FLOAT_NAMES)} — the container's graph.json `torch_dtype` "
                f"must state it")
        return "bfloat16" if graph_dtype == "bfloat16" else "float32"
    if compute_dtype == "float16":
        if traced == "float32" and graph_dtype in _HALF_GRAPH_NAMES:
            # The vendor's own fp32 island inside a half forward (a timestep frequency table:
            # exp(-log(10000) * arange / half) times t up to 999). The component's flag does
            # not narrow it; only the record's per-op narrow set does, the compiled engine's
            # rule. Narrowed to fp16 by the flag, CogVideoX-2b's frequency table moved
            # sin(t * f) by 0.21 at t = 999 (drift walk, 2026-10-05).
            return "float16" if narrowed else "float32"
        return "float16" if (safe or narrowed) else "float32"
    raise ValueError(f"amp_fp32_output_dtype: compute dtype {compute_dtype!r} is not a half "
                     f"dtype — the AMP_FP32 cast-back rule exists for bfloat16 and float16 only")


def contraction_accumulator_dtype(dtype: str) -> str:
    """The dtype (a name) the partial results of a contraction split over the axis it reduces
    are summed in, when a stretch runs in slices of that axis (`core/strategies/chunked_piece`):
    the split contraction is the SAME contraction, so its partials are summed in the accumulator
    its kernel keeps — float32 for a half dtype (every mm/bmm/conv kernel accumulates fp32, and
    `sum` is an AMP_FP32 op), the dtype itself for float32 and float64 — and the sum is stored
    once, in the dtype the contraction's own output was given. A partial summed at the pass
    dtype rounds once per slice where the whole op rounds once. Any other dtype is refused by
    name: a contraction over a token axis yields a floating value.

    Torch-free; its twin is core/dtype/engine.py `contraction_accumulator_dtype` (a unit
    test holds the two equal)."""
    if dtype in ("float16", "bfloat16"):
        return "float32"
    if dtype in ("float32", "float64"):
        return dtype
    raise ValueError(f"contraction_accumulator_dtype: {dtype!r} is not a floating dtype a "
                     f"contraction's partials are summed in")


def contraction_accumulator_bytes(store_bytes: int) -> int:
    """`contraction_accumulator_dtype` read on a WIDTH (bytes per element), for Prism, which prices
    a plan in widths: a 2-byte store (fp16 or bf16 alike) accumulates in 4 bytes; 4 and 8 are their
    own accumulator. The same rule (a unit test holds the two equal over every floating dtype)."""
    if store_bytes == 2:
        return 4
    if store_bytes in (4, 8):
        return store_bytes
    raise ValueError(f"contraction_accumulator_bytes: a {store_bytes}-byte store is no floating "
                     f"dtype a contraction's partials are summed in")


def matrix_unit_representation(dtype: str, unit: dict) -> Optional[str]:
    """How the hardware profile's matrix unit (`matrix_unit`, kernels/ops/_configs.matrix_unit) carries one GEMM
    operand whose dtype in memory is `dtype` (a name): "native" when it IS the unit's operand dtype (one MMA term
    per operand), "split" when the profile declares `matrix_unit.fp32_split` with this dtype among its `operands`
    (the value, widened to fp32, is carried as hi + lo of the operand dtype under a power-of-two scale, see
    `matrix_unit_split_scale`; hi*hi + hi*lo + lo*hi with fp32 accumulation outside the unit, the vendor's fp32
    arithmetic kept), else None (the GEMM keeps its tl.dot kernel). The ONE decision: the wrappers launch with it
    and the derived census keys with it (kernels/launch_keys.matrix_unit_operands)."""
    if not unit or not unit.get("mm"):
        return None
    if dtype == unit["operand_dtype"]:
        return "native"
    split = unit.get("fp32_split") or {}
    if split.get("mm") and dtype in (split.get("operands") or ()):
        return "split"
    return None


def matrix_unit_split_scale(unit: dict) -> tuple:
    """(hi_exp, lo_shift) of the split representation, derived from the unit's operand dtype (never written in a
    kernel): each operand tile is scaled by the power of two that brings its largest magnitude into
    [2^hi_exp, 2^(hi_exp+1)) — the top binade below the operand dtype's overflow, so no hi overflows and every
    value down to 2^-(hi_exp+1+|min exponent|) of the tile's largest stays normal — and lo, the residual
    x - hi, is stored times 2^lo_shift (the operand dtype's significand width) so it does not underflow where hi
    does not (Ootomo & Yokota 2022, arXiv 2203.03341, eq. 19-24)."""
    import numpy as np
    fi = np.finfo(np.dtype(unit["operand_dtype"]))
    return int(fi.maxexp) - 2, int(fi.nmant) + 1


def constant_load_dtype(traced: str, compute: str) -> str:
    """The dtype (a name) an embedded graph constant is bound in by the Triton engines
    (`GraphExecutor._load_constant_triton`): a bfloat16 constant is decoded to the compute dtype
    when that is half (fp16 bits from bf16 on hardware computing fp16, the bf16 bits kept under
    bf16), to float32 otherwise; float64 narrows to float32, complex128 to complex64; every other
    dtype is kept as traced. The loader and Prism's width pass both ask it."""
    if traced == "bfloat16":
        return compute if compute in ("float16", "bfloat16") else "float32"
    if traced == "float64":
        return "float32"
    if traced == "complex128":
        return "complex64"
    return traced


def graph_dtype_name(graph_dtype) -> Optional[str]:
    """The component's graph dtype as a plain name ("bfloat16"), from what the container
    carries (`torch_dtype`, "torch."-prefixed or not); None when it states none."""
    if graph_dtype is None or graph_dtype == "":
        return None
    if isinstance(graph_dtype, NBXDtype):
        return graph_dtype.name
    return str(graph_dtype).rsplit(".", 1)[-1]


def _cast_floats_to(result, dtype: NBXDtype):
    """Every floating tensor of an op's result — one tensor, or the tuple/list a
    `native_*_norm` returns (out, mean, rstd) — in `dtype`; everything else as is. The
    Triton twin of the compiled engine's `DtypeEngine._to_compute_dtype`."""
    if _is_float_tensor(result):
        return result.to(dtype) if _get_nbx_dtype(result) != dtype else result
    if isinstance(result, (tuple, list)):
        return type(result)(_cast_floats_to(r, dtype) for r in result)
    return result


def numpy_staging_dtype(dtype: NBXDtype):
    """Host (numpy) staging dtype for uploading CONSTANTS destined for
    an NBX buffer of `dtype` — the single authority for that mapping
    (gardien 2026-08-17: ad-hoc fp16/fp32 ternaries in kv_cache mask
    staging bypassed the dtype boundary). bfloat16 has no numpy
    representation: stage fp32 and cast device-side (NBXTensor.to,
    R33-pure) after upload — callers must compare the uploaded tensor's
    nbx_dtype against the target and cast when they differ."""
    import numpy as np
    return {
        NBXDtype.float16: np.float16,
        NBXDtype.float32: np.float32,
        NBXDtype.bfloat16: np.float32,   # no numpy bf16 — cast on device
        NBXDtype.int32: np.int32,
        NBXDtype.int64: np.int64,
    }.get(dtype, np.float32)


def resolve_compute_dtype(ctx, component: str = None) -> str:
    """Prism-RESOLVED compute dtype for triton flow-level tensor synthesis.

    SINGLE triton-side resolver (brick-consolidation E2). Returns the dtype
    as a STRING ("float16" / "bfloat16" / "float32") — no torch.dtype ever
    crosses into triton/ (R33 string-dtype boundary). The Prism plan is the
    authority for the dtype that actually executes; the manifest carries the
    pre-Prism vendor declaration and is only a last-resort fallback when no
    plan is attached.

    Resolution order (mirror of the compiled-side
    `FlowContext.compute_dtype` — separate implementation by design):
      1. `plan.components[component].dtype` when the caller names the
         component it synthesises for;
      2. first allocation carrying a dtype (single-dtype plans agree);
      3. `plan.target_dtype`;
      4. `manifest["dtype"]`.
    """
    plan = getattr(ctx, "plan", None)
    if plan is not None:
        comps = getattr(plan, "components", None)
        if comps:
            alloc = comps.get(component) if component else None
            if alloc is None or not getattr(alloc, "dtype", None):
                alloc = next((a for a in comps.values()
                              if getattr(a, "dtype", None)), None)
            if alloc is not None and getattr(alloc, "dtype", None):
                return alloc.dtype
        target = getattr(plan, "target_dtype", None)
        if target:
            return target
    return ctx.pkg.manifest.get("dtype", "float16")


# ============================================================================
# TRITON DTYPE ENGINE
# ============================================================================

# Ops whose wrappers in kernels/wrappers.py self-manage dtype. Each
# wrapper implements a doctrine-specific cast policy:
#
# - mm/bmm/addmm: accumulation-overflow doctrine. Pre-Ampere fp16 input
#   upcast to fp32, output force_fp32 via _matmul_out_dtype. DtypeEngine
#   lower_precision wrap would silently downcast back to fp16 on V100
#   and undo the bind-time fp32 weight cache.
#
# - conv2d/_convolution: VRAM-preserving doctrine (Phase 1). Skip Step 1
#   upcast (kernel already accumulates fp32 internally), narrow input/
#   weight to common dtype on mismatch, set output = compute_dtype from
#   _NBX_COMPUTE_DTYPE. Wrap by DtypeEngine would shadow that policy.
#
# - upsample_nearest{1,2,3}d: pure index lookup. Wrapper is dtype-
#   passthrough trivially. Listed here so DtypeEngine doesn't apply
#   AMP_PROMOTE_OPS or any other transform that would inflate dtype.
#
# Universal hardware: each wrapper internally gates on
# _NBX_HAS_NATIVE_BF16 so the policy is no-op on Ampere+ for the
# matmul family, and the conv/upsample doctrines operate on dtype tags
# only (no hardware gate). The Phase 1 cleanup removed the
# `not self.has_native_bf16` gate that previously restricted self-
# management to pre-Ampere only — the policy is hardware-universal by
# construction.
_SELF_MANAGED_OPS: FrozenSet[str] = frozenset({
    "mm", "bmm", "addmm",
    # fusion_vertical fused anchors: route through mm()/addmm() and
    # inherit their self-managed accumulation-overflow doctrine. A
    # wrap_op passthrough here would silently strip the mm-class dtype
    # protection from the fused op (the scouting-named trap).
    "mm_epilogue", "addmm_epilogue",
    "conv2d", "_convolution",
    "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d",
})


def fp32_constant_names(dag: dict, narrow_op_uids=(), fp32_op_uids=()) -> set:
    """The graph's params and buffers whose every consumer computes in fp32 —
    an AMP_FP32 op the engine wraps fp32-internal, or an op the precision
    contract islands: the wrap cast each of them to fp32 at EVERY call
    (whisper-large-v3-turbo: the fp32 wrap's per-call cast of layer_norm's
    (1280,) weight and bias, 900 a transcription; the copy census of
    2026-09-08). A consumer the contract NARROWS still pre-casts its inputs
    (the narrowing is its output's dtype), so its constants qualify alike.
    Bound in fp32 once at load, the cast the wrap finds nothing to do, the
    kernel sees the fp32 pointers every certified row ran it with, and the
    bytes are those of the per-call cast (the same conversion, once). Returns
    the names without their `param::` / `buffer::` prefix, as the executor
    keys its weights. `narrow_op_uids` is accepted for the call sites' symmetry
    with the contract and does not exclude."""
    islands = set(fp32_op_uids or ())
    ops = (dag or {}).get("ops") or {}
    if isinstance(ops, list):
        ops = {op.get("op_uid", str(i)): op for i, op in enumerate(ops)}
    consumers: dict = {}
    for uid, op in ops.items():
        for tid in op.get("input_tensor_ids") or []:
            if tid.startswith("param::") or tid.startswith("buffer::"):
                name = tid.split("::", 1)[1]
                consumers.setdefault(name, []).append((uid, op.get("op_type", "")))
    out = set()
    for name, uses in consumers.items():
        ok = True
        for uid, op_type in uses:
            short = op_type.split("::")[-1]
            if not (short in AMP_FP32_OPS or uid in islands):
                ok = False
                break
        if ok:
            out.add(name)
    return out


class TritonDtypeEngine:
    """AMP-driven dtype engine for triton mode. Zero torch dependency.

    Same logic as core/dtype/engine.py DtypeEngine but operates on
    NBXDtype instead of torch.dtype.
    """

    def __init__(self, compute_dtype: NBXDtype, has_native_bf16: bool = True,
                 graph_dtype=None):
        self.compute_dtype = compute_dtype
        # The component's GRAPH dtype — the container's traced `torch_dtype` — read by
        # the AMP_FP32 cast-back rule under bf16 (`amp_fp32_output_dtype`).
        self.graph_dtype: Optional[str] = graph_dtype_name(graph_dtype)
        # On pre-Ampere (no native bf16) the weight loader upcasts fp16
        # weights to fp32 at bind time. mm/bmm/addmm wrappers consume those
        # fp32 weights directly and only upcast the activation per-call.
        # Wrapping them in lower_precision here would silently re-downcast
        # the weights to fp16, defeating the bind-time cache.
        self.has_native_bf16 = has_native_bf16

    def cast_runtime_inputs(self, input_map, graph_tensors):
        """Cast component-entry runtime inputs to expected dtype.

        Mirrors PyTorch DtypeEngine path at GraphExecutor._prepare_execution
        (core/runtime/graph_executor.py:1965-1970): for each input::* tensor,
        consult graph metadata for its declared dtype. If the graph dtype is
        floating-point, cast to compute_dtype (Prism's per-component dtype).
        If the graph dtype is non-floating (int64, bool), preserve the
        graph dtype (cast if needed).

        Data-driven: no per-model hardcode. The cast decision derives from
        graph_tensors metadata + compute_dtype, both of which are already
        engine inputs.

        Args:
            input_map: {tensor_id → NBXTensor} of fresh component inputs.
            graph_tensors: dag["tensors"] dict (tid → metadata with
                "dtype" string and "is_input" flag).

        Returns:
            Cast input_map (new dict, original NBXTensors unchanged where
            no cast was needed).
        """
        cast = {}
        for tid, tensor in input_map.items():
            if not isinstance(tensor, NBXTensor):
                cast[tid] = tensor
                continue
            target = self._target_dtype_for_input(tid, graph_tensors)
            if target is not None and tensor._dtype != target:
                tensor = tensor.to(target)
            cast[tid] = tensor
        return cast

    def _target_dtype_for_input(self, tid, graph_tensors):
        """Resolve the expected NBXDtype for an input tensor id.

        Floating graph dtype → compute_dtype (Prism).
        Non-floating graph dtype → preserve graph dtype.
        Unknown / missing → None (no cast).
        """
        meta = graph_tensors.get(tid) or {}
        # A SEAM tensor (a streamed piece's input, `layer_partition.build_segment_graph`) is not a
        # component input: it is an intermediate the previous piece produced, in the dtype its
        # producing op gave it — an fp32 island stays fp32. Cast to the compute dtype, it lost the
        # precision the whole graph keeps: PixArt-XL-1024's T5 in 8 pieces, each fed the whole
        # run's own seam values, diverged 0.15-0.25 % from the same ops run whole, and the first
        # op to differ read an fp32 seam tensor the piece had narrowed to bf16. It enters as it
        # arrived, exactly as it would have flowed inside the whole graph.
        from neurobrix.core.prism.layer_partition import is_seam_tensor
        if is_seam_tensor(meta):
            return None
        dtype_str = meta.get("dtype")
        if not dtype_str:
            return None
        from neurobrix.kernels.nbx_tensor import parse_dtype
        try:
            graph_dt = parse_dtype(dtype_str.replace("torch.", ""))
        except Exception:
            return None
        if graph_dt is None:
            return None
        if graph_dt in _FLOATING:
            return self.compute_dtype
        return graph_dt

    def accumulation_dtype(self, dtype: NBXDtype) -> NBXDtype:
        """The dtype the partials of a contraction split over its reduced axis are summed in
        (`contraction_accumulator_dtype`), for a partial of `dtype` — the stitch of a stretch run
        in slices (`core/strategies/chunked_piece`) asks the engine, never the pass dtype. The
        twin of `DtypeEngine.accumulation_dtype` (R30)."""
        from neurobrix.kernels.nbx_tensor import parse_dtype
        return parse_dtype(contraction_accumulator_dtype(NBXDtype(dtype).name))

    def set_precision_contract(self, safe: bool, fp32_op_uids=(), narrow_op_uids=()) -> None:
        """The component's precision contract (core/runtime/precision_contract):
        `fp32_op_uids` are the calibration record's islands — ops whose finite
        magnitude exceeds the half-precision bound — computed in fp32 with
        an fp32 output whatever their class; `narrow_op_uids` are the fp32-
        class ops the record lets narrow back to the compute dtype. The
        same sets the compiled DtypeEngine honours (R30; closes
        D-PRECISION-CONTRACT-TRITON-PARITY, 2026-09-06)."""
        self._activations_fp16_safe = bool(safe)
        self._fp32_op_uids = frozenset(fp32_op_uids or ())
        self._narrow_op_uids = frozenset(narrow_op_uids or ())

    def wrap_op(self, op_name: str, func: Callable, op_uid: Optional[str] = None,
                op_record: Optional[dict] = None) -> Callable:
        """Wrap an op function with AMP casting rules.

        Args:
            op_name: bare op name (e.g., "mm", "pow", "add")
            func: the raw kernel wrapper function
            op_uid: the op's uid in the graph — the precision contract's
                per-op islands are keyed on it
            op_record: the op's graph entry (`op_data`). Only one key is read,
                `output_dtypes`, and it sits at the record's TOP level — NOT
                inside `op_data["attributes"]`. Handing the attributes dict
                instead reads as "not complex" and loses the floor silently,
                which is why the parameter is not called `attrs`.

        Returns:
            Wrapped function with dtype casting applied
        """
        # A complex-producing op never receives a half input — mirror of the
        # ATen branch's first rule, ahead of the islands for the same reason:
        # an island is a precision choice among REAL dtypes and cannot retype
        # a traced complex128 to complex64 by levelling its float64 input.
        if traced_output_is_complex(op_record):
            return self._wrap_complex_output(func)
        # The op's traced output dtype — the vendor's dtype for this op (the cast-back floor).
        traced = traced_output_dtype_name(op_record)

        # The calibration record's islands come FIRST: a pinned op computes
        # in fp32 and keeps its fp32 output whatever its AMP class — a
        # self-managed conv included (its wrapper follows the fp32 inputs).
        if op_uid is not None and self.compute_dtype in (NBXDtype.float16, NBXDtype.bfloat16):
            if op_uid in getattr(self, "_fp32_op_uids", ()):
                return self._wrap_fp32(func)
            if op_uid in getattr(self, "_narrow_op_uids", ()) and op_name in AMP_FP32_OPS:
                return self._wrap_fp32_internal_compute_dtype_output(func, force_cast_back=True,
                                                                     traced=traced)
        # Self-managed wrappers are NEVER wrapped — universal hardware
        # (mm/bmm/addmm self-gate on _NBX_HAS_NATIVE_BF16 internally;
        # conv2d/upsample_nearest are dtype-tag-driven). See _SELF_MANAGED_OPS
        # docstring for the per-op doctrine.
        # Scalars and one-element constants narrowed into a half dtype saturate
        # (see saturate_scalar). Ahead of the self-managed return: none of these
        # ops is self-managed, and the rule holds whatever the compute dtype.
        if op_name in AMP_SCALAR_FILL_OPS:
            return self._wrap_scalar_fill(func)
        if op_name in AMP_CREATION_FILL_OPS:
            return self._wrap_creation_fill(func)
        if op_name == "where":
            return self._wrap_where_saturate(func)
        if op_name in _SELF_MANAGED_OPS:
            return func

        # AMP only applies for half-precision compute
        if self.compute_dtype not in (NBXDtype.float16, NBXDtype.bfloat16):
            return func

        if op_name in AMP_FP32_OPS:
            # Uniform cast-back: all AMP_FP32_OPS go through the fp32-internal
            # wrap; its output dtype is `amp_fp32_output_dtype` — bf16: back to
            # bf16 on a bf16 graph, fp32 on another; fp16: back to fp16 only
            # under the contract flag (read at call time via the _w global),
            # else fp32.
            return self._wrap_fp32_internal_compute_dtype_output(func, traced=traced)

        if op_name in AMP_FP16_OPS:
            if self.compute_dtype == NBXDtype.float16 and op_name in _FP16_NEED_FP32:
                # div is in _FP16_NEED_FP32 (FP16 op needing fp32 protection
                # on V100, epsilon underflow). Same uniform cast-back wrap —
                # under fp16 only: under bf16 div is a plain AMP_FP16 op
                # (inputs cast to bf16, output bf16) and never reaches it.
                return self._wrap_fp32_internal_compute_dtype_output(func, traced=traced)
            return self._wrap_lower_precision(func)

        if op_name in AMP_PROMOTE_OPS:
            return self._wrap_promote(func)

        return func

    @staticmethod
    def _wrap_scalar_fill(func: Callable) -> Callable:
        """masked_fill / fill / index_fill: the Python scalar is clamped to the finite range of the
        tensor it fills, as the ATen branch does (AMP_SCALAR_FILL_OPS)."""
        def fill_func(*args, **kwargs):
            target = next((_get_nbx_dtype(a) for a in args if _is_float_tensor(a)), None)
            if target not in _HALF_MAX:
                return func(*args, **kwargs)
            args = tuple(saturate_scalar(a, target) if not hasattr(a, "data_ptr") else a for a in args)
            kwargs = {k: (saturate_scalar(v, target) if not hasattr(v, "data_ptr") else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)
        return fill_func

    @staticmethod
    def _wrap_creation_fill(func: Callable) -> Callable:
        """full / new_full / full_like: the fill value is clamped to the finite range of the dtype
        created — the dtype kwarg, else an NBXDtype argument, else the template tensor's dtype
        (the ATen branch's clamp_creation_fill_args)."""
        def creation_func(*args, **kwargs):
            target = kwargs.get("dtype")
            if not isinstance(target, NBXDtype):
                target = next((a for a in args if isinstance(a, NBXDtype)), None)
            if target is None:
                target = next((_get_nbx_dtype(a) for a in args if _is_float_tensor(a)), None)
            if target not in _HALF_MAX:
                return func(*args, **kwargs)
            args = tuple(saturate_scalar(a, target) if isinstance(a, float) else a for a in args)
            if isinstance(kwargs.get("fill_value"), float):
                kwargs = {**kwargs, "fill_value": saturate_scalar(kwargs["fill_value"], target)}
            return func(*args, **kwargs)
        return creation_func

    @staticmethod
    def _wrap_where_saturate(func: Callable) -> Callable:
        """where(cond, x, y) writes into x's dtype. A one-element operand of a wider float dtype
        whose finite value lies beyond that half dtype's range is replaced by the saturated value
        before the kernel writes it (a mask sentinel, read on the host: one element, once)."""
        def where_func(cond, x, y, *rest, **kwargs):
            out_dt = _get_nbx_dtype(x) if _is_float_tensor(x) else None
            if out_dt in _HALF_MAX:
                x, y = (_saturated_one_element(t, out_dt) for t in (x, y))
            return func(cond, x, y, *rest, **kwargs)
        return where_func

    def _wrap_complex_output(self, func: Callable) -> Callable:
        """Raise fp16/bf16 operands to fp32 and leave every other dtype alone.

        A FLOOR, not a leveller: `_wrap_fp32` casts every non-fp32 float to
        fp32, which would downcast the float64 that Wan2.1's RoPE feeds to
        `view_as_complex` and retype its complex128 output to complex64. Its
        contiguity normalisation IS kept (contiguous-guard pattern).

        This rule concerns TENSOR OPERANDS. It does not contradict the
        deliberate narrowing of complex128/float64 CONSTANTS at attribute
        resolution (`sequence.py`, `sequential.py`): a constant is materialised
        by this branch, and NBX complex64 is a pair of fp32, while an operand's
        width is whatever the producing op handed over.

        Defensive rather than load-bearing today: NBXTensor has no complex32
        representation at all (complex64 = fp32 pairs), and `complex_wrapper`
        / `fft_r2c_wrapper` / `NBXTensor.view_as_complex` already force fp32.
        It is here so the rule is the same rule in all four modes (R30).
        """
        def complex_out_func(*args, **kwargs):
            new_args = tuple(
                a.to(NBXDtype.float32).contiguous()
                if _is_float_tensor(a) and _get_nbx_dtype(a) in (
                    NBXDtype.float16, NBXDtype.bfloat16)
                else (a.contiguous()
                      if hasattr(a, "contiguous") and hasattr(a, "is_contiguous")
                      and not a.is_contiguous() else a)
                for a in args
            )
            return func(*new_args, **kwargs)
        return complex_out_func

    def _wrap_fp32(self, func: Callable) -> Callable:
        """Upcast float inputs to fp32, and run the op with fp32 as the
        active compute dtype.

        The second half is what makes an island hold on a self-managed
        wrapper. `conv2d_wrapper` (and the other self-managed ops) do not
        follow their inputs: they narrow the inputs to the narrowest common
        dtype and write their output in the per-component compute dtype
        they read from `kernels.wrappers` — so an island that only upcast
        the inputs was undone inside the wrapper and the fp16 output
        overflowed exactly where the calibration record said it would
        (swin2SR-x2 `aten.convolution::13`, fp32 norm 2.6e7: inf, NaN, a
        black render). Overriding the compute dtype for the duration of the
        pinned op makes the wrapper's own policy produce fp32, the same
        thing ATen does for fp32 inputs on the compiled engine. Restored
        after the call, nested-safe."""
        widens = bool(getattr(func, "_nbx_widens_on_load", False))
        def fp32_func(*args, **kwargs):
            from neurobrix.kernels import wrappers as _w
            if widens:
                # the island holds through the store dtype asked of a wrapper whose kernel
                # widens its loads: no materialised fp32 input (the copy lever, 2026-09-07)
                new_args = args
                kwargs = {**kwargs, "out_dtype": NBXDtype.float32}
            else:
                new_args = tuple(
                    a.to(NBXDtype.float32).contiguous()
                        if _is_float_tensor(a) and _get_nbx_dtype(a) != NBXDtype.float32
                    else (a.contiguous() if hasattr(a, 'contiguous') and hasattr(a, 'is_contiguous') and not a.is_contiguous() else a)
                    for a in args
                )
            prev = _w.get_compute_dtype()
            _w.set_compute_dtype(NBXDtype.float32)
            try:
                return func(*new_args, **kwargs)
            finally:
                _w.set_compute_dtype(prev)
        return fp32_func

    def _wrap_fp32_internal_compute_dtype_output(self, func: Callable, force_cast_back: bool = False,
                                                 traced: Optional[str] = None) -> Callable:
        """fp32 compute inside; the OUTPUT dtype is `amp_fp32_output_dtype`'s answer.

        The op's internal precision rationale (RMSNorm pow->mean->rsqrt overflow risk, div
        epsilon underflow) is kept by upcasting the inputs to fp32; whether the result is
        brought back to compute_dtype is the one rule:
          * compute dtype bf16 on a bf16 GRAPH (`self.graph_dtype`, the container's
            `torch_dtype`): always — every floating tensor of the result, a
            `native_*_norm` tuple included (the compiled engine's `_to_compute_dtype`);
            on a graph of another dtype coerced to bf16 compute: never (fp32 kept);
          * compute dtype fp16: only under the precision contract — the per-component
            `_NBX_ACTIVATIONS_FP16_SAFE` flag (read at call time, so an engine compiled
            before the flag was applied still sees it) or `force_cast_back` (the op is in
            the record's narrow set). The fp16 cast is the single-tensor one it always
            was: a tuple result keeps the dtypes its kernel wrote (the fp16 contract is
            held byte-identical by decision, 2026-09-28).
        """
        compute = self.compute_dtype
        cname = compute.name
        gname = self.graph_dtype
        if compute == NBXDtype.bfloat16:
            # decided now, not at the first call: an unstated graph dtype refuses at compile
            amp_fp32_output_dtype(cname, gname, False, False)
        widens = bool(getattr(func, "_nbx_widens_on_load", False))

        def cast_back(result):
            from neurobrix.kernels import wrappers as _w
            if amp_fp32_output_dtype(cname, gname, _w._NBX_ACTIVATIONS_FP16_SAFE,
                                     force_cast_back, traced) != cname:
                return result          # fp16 without the contract, or a non-bf16 graph: fp32
            if compute == NBXDtype.bfloat16:
                return _cast_floats_to(result, compute)
            if _is_float_tensor(result) and _get_nbx_dtype(result) != compute:
                result = result.to(compute)
            return result

        def cast_back_func(*args, **kwargs):
            if widens:
                # The wrapper's kernel widens its loads to fp32: no materialised fp32 input.
                # It STORES fp32, as every certified row ran it — asked to store fp16, the
                # kernel compiled for an fp16 output pointer schedules its reduction
                # differently and rounds a rare element one ulp apart (VibeVoice, the final
                # gate of 2026-09-08: 2 of 75,776 at feat 2048) — so the cast back to the
                # compute dtype stays the copy it was, exact by construction.
                return cast_back(func(*args, out_dtype=NBXDtype.float32, **kwargs))
            new_args = tuple(
                a.to(NBXDtype.float32).contiguous()
                    if _is_float_tensor(a) and _get_nbx_dtype(a) != NBXDtype.float32
                else (a.contiguous() if hasattr(a, 'contiguous') and hasattr(a, 'is_contiguous') and not a.is_contiguous() else a)
                for a in args
            )
            return cast_back(func(*new_args, **kwargs))
        return cast_back_func

    def _wrap_lower_precision(self, func: Callable) -> Callable:
        """Cast float inputs to compute_dtype."""
        compute = self.compute_dtype
        def lower_func(*args, **kwargs):
            new_args = tuple(
                a.to(compute) if _is_float_tensor(a) and _get_nbx_dtype(a) != compute
                else a
                for a in args
            )
            return func(*new_args, **kwargs)
        return lower_func

    def _wrap_promote(self, func: Callable) -> Callable:
        """Promote to widest input dtype."""
        def promote_func(*args, **kwargs):
            max_size = 0
            max_dtype = None
            for a in args:
                if _is_float_tensor(a):
                    esz = a.element_size()
                    if esz > max_size:
                        max_size = esz
                        max_dtype = _get_nbx_dtype(a)

            if max_dtype is not None and max_size > 2:
                new_args = tuple(
                    a.to(max_dtype) if _is_float_tensor(a) and _get_nbx_dtype(a) != max_dtype
                    else a
                    for a in args
                )
                return func(*new_args, **kwargs)
            return func(*args, **kwargs)
        return promote_func
