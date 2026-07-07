"""Triton DtypeEngine — AMP rules for triton mode. Zero torch dependency.

Same AMP logic as core/dtype/engine.py but uses NBXDtype instead of torch.dtype.
Wraps op functions with input casting for numerical stability.

Rules (from PyTorch AT_FORALL_FP32 / AT_FORALL_LOWER_PRECISION_FP):
  - FP32 ops: upcast inputs to fp32 (pow, rsqrt, softmax, layernorm, ...)
  - FP16 ops: cast inputs to compute_dtype (mm, conv, bmm, ...)
  - FP16_NEED_FP32: on fp16 hardware, mm/bmm/div/addmm need fp32
  - Promote ops: promote to widest input dtype
"""

from typing import Callable, FrozenSet

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

# Phase 2 — UNIFORM cast-back doctrine: every AMP_FP32_OPS goes through
# the fp32-internal-compute-then-conditional-cast-back wrap. The cast-back
# is gated SOLELY by the per-component `activations_fp16_safe` flag from
# model_registry.yml (read at call time via _w._NBX_ACTIVATIONS_FP16_SAFE).
# When False (default, conservative — typical LLMs without an explicit
# fp16-safe annotation), output stays fp32 (matches PyTorch oracle).
# When True (annotated VAE/UNet on diffusion models), output is cast back
# to compute_dtype for VRAM-preserving fp16-throughout flow.
# The previous `_AMP_FP32_OPS_OPT_IN_CAST_BACK` set was an additional
# membership gate that fragmented the doctrine: rms_norm and div had
# the cast-back hook but rsqrt/exp/layer_norm/batch_norm/etc did not.
# Removed in favor of a single uniform gate (the registry flag).

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
    "dot", "vdot", "grid_sampler", "index_put",
    "scatter_add", "tensordot", "linalg_cross",
})

_FLOATING = frozenset({NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64})


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
    "conv2d", "_convolution",
    "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d",
})


class TritonDtypeEngine:
    """AMP-driven dtype engine for triton mode. Zero torch dependency.

    Same logic as core/dtype/engine.py DtypeEngine but operates on
    NBXDtype instead of torch.dtype.
    """

    def __init__(self, compute_dtype: NBXDtype, has_native_bf16: bool = True):
        self.compute_dtype = compute_dtype
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

    def wrap_op(self, op_name: str, func: Callable) -> Callable:
        """Wrap an op function with AMP casting rules.

        Args:
            op_name: bare op name (e.g., "mm", "pow", "add")
            func: the raw kernel wrapper function

        Returns:
            Wrapped function with dtype casting applied
        """
        # Self-managed wrappers are NEVER wrapped — universal hardware
        # (mm/bmm/addmm self-gate on _NBX_HAS_NATIVE_BF16 internally;
        # conv2d/upsample_nearest are dtype-tag-driven). See _SELF_MANAGED_OPS
        # docstring for the per-op doctrine.
        if op_name in _SELF_MANAGED_OPS:
            return func

        # AMP only applies for half-precision compute
        if self.compute_dtype not in (NBXDtype.float16, NBXDtype.bfloat16):
            return func

        if op_name in AMP_FP32_OPS:
            # Phase 2 — uniform cast-back: all AMP_FP32_OPS go through the
            # fp32-internal-then-conditional-cast-back wrap. The cast-back
            # decision is gated solely by the `activations_fp16_safe`
            # registry flag (read at call time via _w global). When False
            # (default), output stays fp32 (PyTorch-oracle parity). When
            # True (annotated fp16-safe model), output cast to compute_dtype.
            return self._wrap_fp32_internal_compute_dtype_output(func)

        if op_name in AMP_FP16_OPS:
            if self.compute_dtype == NBXDtype.float16 and op_name in _FP16_NEED_FP32:
                # div is in _FP16_NEED_FP32 (FP16 op needing fp32 protection
                # on V100, epsilon underflow). Same uniform cast-back wrap.
                return self._wrap_fp32_internal_compute_dtype_output(func)
            return self._wrap_lower_precision(func)

        if op_name in AMP_PROMOTE_OPS:
            return self._wrap_promote(func)

        return func

    def _wrap_fp32(self, func: Callable) -> Callable:
        """Upcast float inputs to fp32."""
        def fp32_func(*args, **kwargs):
            new_args = tuple(
                a.to(NBXDtype.float32).contiguous()
                    if _is_float_tensor(a) and _get_nbx_dtype(a) != NBXDtype.float32
                else (a.contiguous() if hasattr(a, 'contiguous') and hasattr(a, 'is_contiguous') and not a.is_contiguous() else a)
                for a in args
            )
            return func(*new_args, **kwargs)
        return fp32_func

    def _wrap_fp32_internal_compute_dtype_output(self, func: Callable) -> Callable:
        """Phase 1 opt-in cast-back: compute fp32 internally, output back to
        compute_dtype.

        Used when `_NBX_ACTIVATIONS_FP16_SAFE` is True (per-component flag
        from model_registry.yml). The op's internal precision rationale
        (RMSNorm pow→mean→rsqrt overflow risk, div epsilon underflow) is
        preserved by upcasting inputs to fp32, but the output is brought
        back to compute_dtype so downstream ops don't propagate fp32 in
        the activation chain — VRAM-preserving for diffusion VAE/UNet
        chains where ranges are confirmed fp16-safe.

        When the flag is False (default conservative), this wrapper is
        not selected; ops fall back to `_wrap_fp32` which leaves output
        in fp32. Read flag at call time so a single TritonDtypeEngine
        instance compiled before the registry-driven flag was applied
        still picks up the new value.
        """
        compute = self.compute_dtype
        def cast_back_func(*args, **kwargs):
            from neurobrix.kernels import wrappers as _w
            new_args = tuple(
                a.to(NBXDtype.float32).contiguous()
                    if _is_float_tensor(a) and _get_nbx_dtype(a) != NBXDtype.float32
                else (a.contiguous() if hasattr(a, 'contiguous') and hasattr(a, 'is_contiguous') and not a.is_contiguous() else a)
                for a in args
            )
            result = func(*new_args, **kwargs)
            # Cast back ONLY when the per-component opt-in flag is True.
            # Default False = conservative behavior (output stays fp32).
            if (_w._NBX_ACTIVATIONS_FP16_SAFE
                    and _is_float_tensor(result)
                    and _get_nbx_dtype(result) != compute):
                result = result.to(compute)
            return result
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
