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

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.promotion import promoted as _promoted
from nbxtensor.dtype import itemsize as _itemsize
from nbxtensor.dtype.operands import attention_operand_alignment, contraction_accumulator_dtype  # the library's rules
from nbxtensor.dtype.convert import saturate_scalar  # noqa: F401 — the library's rule, used below


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
# dtype — `amp_fp32_output_dtype` (the library's, nbxtensor/dtype/outputs.py; the
# compiled DtypeEngine imports the same one):
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
def _is_half(dtype) -> bool:
    """A half floating dtype (the dtype table: floating, two bytes) — what `saturate_scalar` clamps to."""
    if dtype is None:
        return False
    name = _itemsize.canonical_name(dtype)
    return _itemsize.kind(name) == "float" and _itemsize.itemsize(name) == 2


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



# `traced_output_dtype_name`, `traced_output_is_complex` and `amp_fp32_output_dtype` are the library's
# (nbxtensor/dtype/outputs.py), one definition for both engines.
from nbxtensor.dtype.outputs import (  # noqa: F401 — the library's rules, one definition for both engines
    amp_fp32_output_dtype, traced_output_dtype_name, traced_output_is_complex)


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


def _triton_fp64_keys(vendor: str, architecture: str, keys) -> bool:
    from neurobrix.core.config import loader
    precision = loader.get_vendor_config(vendor, architecture).get("precision") or {}
    values = [precision.get("supports_fp64")] + [(precision.get(k) or {}).get("triton") for k in keys]
    if not all(isinstance(v, bool) for v in values):
        names = ", ".join(["precision.supports_fp64"] + [f"precision.{k}.triton" for k in keys])
        raise ValueError(f"ZERO FALLBACK: the {vendor}/{architecture} profile does not declare "
                         f"{names} (found {precision!r}); a device's fp64 is never assumed")
    return all(values)


def triton_has_fp64(vendor: str, architecture: str) -> bool:
    """Whether the NeuroBrix Triton kernels COMPUTE float64 / complex128 elements on this hardware —
    a capability: the device's (`precision.supports_fp64`) AND the kernels' there
    (`precision.kernels_carry_fp64.triton`), both from the vendor profile, a missing key refused.
    Bound into nbx_tensor's backend by `kernels.wrappers.set_hardware_profile`
    (`nbx_tensor.set_backend_has_fp64`): the integer-division widening and the cast door ask it."""
    return _triton_fp64_keys(vendor, architecture, ("kernels_carry_fp64",))


def triton_stores_fp64(vendor: str, architecture: str) -> bool:
    """Whether the Triton branch HOLDS float64 / complex128 tensors at that width: its kernels'
    capability (`triton_has_fp64`) AND the branch's storage policy (`precision.stores_fp64.triton`),
    a missing key refused. The mirror of `core.dtype.config.device_supports_fp64`, which reads the
    `.compiled` keys. Every Triton storage site narrows through `storage_dtype` with this answer."""
    return _triton_fp64_keys(vendor, architecture, ("kernels_carry_fp64", "stores_fp64"))


def _profile_fp64(profile, reader) -> bool:
    if profile is None or not getattr(profile, "devices", None):
        raise ValueError("ZERO FALLBACK: no hardware profile with a device; the Triton branch's "
                         "fp64 is read from its vendor profile, never assumed")
    answers = {}
    for dev in profile.devices:
        key = (getattr(dev.brand, "value", dev.brand), dev.architecture)
        answers[key] = reader(*key)
    if len(set(answers.values())) > 1:
        raise ValueError(f"ZERO FALLBACK: the profile's devices disagree on the Triton branch's "
                         f"fp64 ({answers!r}); one process binds one answer, so a mixed profile "
                         f"is refused rather than decided by its first device")
    return next(iter(answers.values()))


def profile_triton_has_fp64(profile) -> bool:
    """`triton_has_fp64` of a hardware profile (a PrismProfile): every device's, refused if they
    disagree."""
    return _profile_fp64(profile, triton_has_fp64)


def declare_hardware_profile(profile) -> None:
    """Hand the hardware profile a run plans for to nbxtensor's wrappers (`set_hardware_profile`), then declare the
    float64 capability of its kernels as the profile's vendor files state it (`precision.kernels_carry_fp64.triton`,
    `profile_triton_has_fp64`) and each device's memory (`device_facts.declare_device_memory`) through the library's
    setters — the library's own default is its backend's fact, and a
    NeuroBrix profile is the engine's to read (rule 1: the profile is the machine's identity card)."""
    if profile is None:
        return
    from nbxtensor.ops.wrappers import set_hardware_profile
    from nbxtensor.backend.capabilities import set_backend_has_fp64
    set_hardware_profile(profile)
    set_backend_has_fp64(profile_triton_has_fp64(profile))
    # the library's CPU pool (host work: conversions, pre-processing, the parallel loop) is sized by the profile's
    # compute threads — the performance cores of a hybrid CPU, else the physical ones (Hocine 2026-10-29 18:11, rule 1)
    cpu = getattr(profile, "cpu", None)
    if cpu is None:
        raise ValueError("declare_hardware_profile: the hardware profile declares no cpu: section — the library's CPU "
                         "pool is sized by the profile's compute threads (rule 1), never by a count read off the machine")
    from nbxtensor.backend.cpu.pool import set_threads
    set_threads(int(cpu.compute_threads))
    # each device's memory as the profile states it: the size the library's certified store serves entries for —
    # a census shadow sees no card, and a plan made for a profile is keyed by ITS cards
    from nbxtensor.backend.device_facts import declare_device_memory
    for device in getattr(profile, "devices", None) or []:
        if getattr(device, "memory_mb", None):
            declare_device_memory(int(getattr(device, "index", 0)), int(device.memory_mb) << 20)


def profile_triton_stores_fp64(profile) -> bool:
    """`triton_stores_fp64` of a hardware profile (a PrismProfile): every device's, refused if they
    disagree."""
    return _profile_fp64(profile, triton_stores_fp64)


def profile_attention_operand_alignment(profile) -> Optional[str]:
    """`precision.attention_operands` of the hardware profile a plan TARGETS (a PrismProfile),
    read from every device's vendor profile and refused if they disagree; None for a profile with
    no device (it places on the host, where no vendor profile declares one, and a width pass that
    then meets operands that disagree refuses by name).

    The plan's answer, never the process's: the launcher's `active_vendor_profile()` is the card
    THIS process drives — nothing at all on a host with no visible card — while Prism prices the
    hardware it was handed (a census of the other machine, the 16 GB card planned from the 32 GB
    one, a Mac profile planned on the rack). Read off the driver, the plan's width estimate
    depended on which card the planning process could see (2026-10-12: 30 plans refused under
    CUDA_VISIBLE_DEVICES='', "hardware profile None")."""
    if profile is None or not getattr(profile, "devices", None):
        return None
    from neurobrix.core.config import loader
    answers = {}
    for dev in profile.devices:
        key = (getattr(dev.brand, "value", dev.brand), dev.architecture)
        answers[key] = attention_operand_alignment(loader.get_vendor_config(*key))
    if len(set(answers.values())) > 1:
        raise ValueError(f"ZERO FALLBACK: the profile's devices disagree on "
                         f"precision.attention_operands ({answers!r}); one plan prices one "
                         f"answer, so a mixed profile is refused rather than decided by its "
                         f"first device")
    return next(iter(answers.values()))


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
                 graph_dtype=None, *, stores_fp64: bool):
        self.compute_dtype = compute_dtype
        # Whether this branch computes and stores float64 / complex128 on its device
        # (`triton_has_fp64`, from the vendor profile); False holds them in float32 / complex64.
        self.stores_fp64 = bool(stores_fp64)
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

        A floating graph input traced at or below the compute dtype's width → compute_dtype (Prism);
        one traced WIDER enters at its traced width, as this branch stores it (`storage_dtype`:
        float64 → float32 where the branch holds no fp64) — its first consumer casts it as that op's
        class dictates, exactly as for any wide tensor inside the graph. Narrowing every float
        input to the compute dtype rounded a value the graph needs whole: a timestep traced float64
        fed to a bf16 component on Metal entered as bf16, which holds integers exactly only to 256
        (777 → 776; Allegro's halo, 2026-10-18). Non-floating graph dtype → preserved.
        Unknown / missing → None (no cast). The compiled branch's rule is the same sentence
        (`core/runtime/graph/tensor_resolver.get_input_dtype`), and Prism prices the seed alike
        (`core/prism/runtime_widths`).
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
        from nbxtensor.dtype.nbx_dtype import parse_dtype
        try:
            graph_dt = parse_dtype(dtype_str.replace("torch.", ""))
        except Exception:
            return None
        if graph_dt is None:
            return None
        if graph_dt in _FLOATING:
            # Wider than the graph's OWN parameters (core/dtype/entry_width): a model traced float32
            # throughout is at its width, not wider (orpheus-3b, 2026-10-18).
            from neurobrix.core.dtype.entry_width import enters_at_traced_width
            if not enters_at_traced_width(_itemsize.itemsize(graph_dt), _itemsize.itemsize(self.compute_dtype),
                                          graph_tensors):
                return self.compute_dtype
            return self.storage_dtype(graph_dt)
        return graph_dt

    def storage_dtype(self, dtype: NBXDtype) -> NBXDtype:
        """`dtype` as this branch holds it: float64 -> float32 and complex128 -> complex64 where it
        carries no fp64, unchanged otherwise — `DtypeEngine.storage_dtype`'s mirror, through the
        library's `itemsize.storage_dtype` (the one narrowing, at this branch's fp64 fact)."""
        # The MEMBER goes to the table, never its `.name`: `NBXDtype.bool_` is Python's keyword escape,
        # which only `canonical_name` maps to the table's `bool` (mq24 Allegro triton, 2026-10-11).
        from nbxtensor.dtype.nbx_dtype import parse_dtype
        return parse_dtype(_itemsize.storage_dtype(dtype, self.stores_fp64))

    def accumulation_dtype(self, dtype: NBXDtype) -> NBXDtype:
        """The dtype the partials of a contraction split over its reduced axis are summed in
        (`contraction_accumulator_dtype`), for a partial of `dtype` — the stitch of a stretch run
        in slices (`core/strategies/chunked_piece`) asks the engine, never the pass dtype. The
        twin of `DtypeEngine.accumulation_dtype` (R30). `dtype` is what the partial hands: an
        NBXTensor's `.dtype` is its triton.language dtype, parsed as the module parses any dtype."""
        from nbxtensor.dtype.nbx_dtype import parse_dtype
        return parse_dtype(contraction_accumulator_dtype(parse_dtype(dtype).name))

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
        if (op_name in _SELF_MANAGED_OPS and op_name in AMP_FP16_OPS
                and self.compute_dtype == NBXDtype.bfloat16 and self.has_native_bf16):
            # ONE precision contract (the owner, 2026-10-25 15:26): self-management covers the LAUNCH (the in-kernel
            # promotion `launch_keys.mm_dtypes` and its kin decide), not the operands' dtype nor the store. On hardware
            # that computes bf16 natively the compiled branch casts a GEMM's or a conv's float operands to bf16
            # (`_wrap_lower_precision`); the Triton wrapper took them as they came, so an island's fp32 output stayed
            # fp32 through every self-managed op after it (2 017 policy sites on Sana's transformer, the Mac, 06:40).
            # An island — the trace's or the calibration's (returned above) — keeps fp32. Under fp16 compute without
            # native bf16 nothing changes: the in-kernel widening IS the compiled branch's fp32-internal rule there.
            from neurobrix.core.dtype.traced_island import is_traced_fp32_island
            if is_traced_fp32_island(op_record, self.graph_dtype):
                return self._wrap_fp32(func)
            return self._wrap_lower_precision(func)
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
            # The trace's own fp32 island — the compiled engine's rule, one predicate (R30).
            from neurobrix.core.dtype.traced_island import is_traced_fp32_island
            if is_traced_fp32_island(op_record, self.graph_dtype):
                return self._wrap_fp32(func)
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
            if not _is_half(target):
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
            if not _is_half(target):
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
            if _is_half(out_dt):
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
            from nbxtensor.ops import wrappers as _w
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
            the record's narrow set). Every floating tensor of the result is cast, a tuple included (the single-tensor cast of
            2026-09-28 kept a half graph's norm tuples fp32 on this branch; aligned 2026-10-25).
        """
        compute = self.compute_dtype
        cname = compute.name
        gname = self.graph_dtype
        if compute == NBXDtype.bfloat16:
            # decided now, not at the first call: an unstated graph dtype refuses at compile
            amp_fp32_output_dtype(cname, gname, False, False)
        widens = bool(getattr(func, "_nbx_widens_on_load", False))

        def cast_back(result):
            from nbxtensor.ops import wrappers as _w
            if amp_fp32_output_dtype(cname, gname, _w._NBX_ACTIVATIONS_FP16_SAFE,
                                     force_cast_back, traced) != cname:
                return result          # fp16 without the contract, or a non-bf16 graph: fp32
            # every floating tensor of the result, a `native_*_norm` tuple included — the compiled engine's
            # `_to_compute_dtype` / half-IO norms under the contract. The fp16 cast was the single-tensor one
            # (a tuple kept its kernel's dtypes, 2026-09-28): a half graph's layer norms then stored fp32 on this
            # branch and spread it through the whole encoder (whisper: 1 283 of 1 545 outputs; the owner, 2026-10-25
            # 20:36: a half-stored graph is served half everywhere except islands, on both engines).
            return _cast_floats_to(result, compute)

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
        def promoted_op(*args, **kwargs):
            # the rule is nbxtensor's (`dtype.promotion.promoted`), read on the float inputs' dtype names
            target = _promoted([_get_nbx_dtype(a).name for a in args if _is_float_tensor(a)])
            if target is not None:
                max_dtype = NBXDtype[target]
                new_args = tuple(
                    a.to(max_dtype) if _is_float_tensor(a) and _get_nbx_dtype(a) != max_dtype
                    else a
                    for a in args
                )
                return func(*new_args, **kwargs)
            return func(*args, **kwargs)
        return promoted_op


