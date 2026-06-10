"""Triton Sequential Dispatcher — zero torch, op-by-op execution.

Ported from core/runtime/graph/sequential_dispatcher.py.
Resolves graph.json args dynamically and dispatches to Triton kernels.
No pre-compilation (no arena, no closures). Useful for debugging
individual ops and validating graph correctness.

Usage: --triton-sequential flag routes here.
"""

from typing import Any, Dict, List

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype
from neurobrix.kernels.dispatch import dispatch as kernel_dispatch
from neurobrix.kernels import wrappers as w

from .symbols import SymbolResolver
from .dtype import TritonDtypeEngine


class TritonSequentialDispatcher:
    """Op-by-op dispatcher using Triton kernels. Zero torch.

    Resolves graph.json arg dicts at runtime (no pre-compilation).
    Applies AMP rules via TritonDtypeEngine (same as compiled mode).
    Simpler than TritonSequence — easier to debug individual ops.
    """

    def __init__(self, device_idx: int = 0, compute_dtype: NBXDtype = NBXDtype.float16,
                 activations_fp16_safe: bool = False):
        self.device_idx = device_idx
        self.compute_dtype = compute_dtype
        self.activations_fp16_safe = activations_fp16_safe
        from neurobrix.kernels.wrappers import has_native_bf16 as _has_bf16
        from neurobrix.kernels import wrappers as _w
        self._dtype_engine = TritonDtypeEngine(
            compute_dtype, has_native_bf16=_has_bf16())
        self._op_cache: Dict[str, Any] = {}
        # Phase 2 — propagate per-component dtype context to wrappers
        # global state, mirroring TritonSequence.run() but without the
        # try/finally restore (sequential mode doesn't nest within
        # compiled mode within a single component invocation; if a
        # later compiled run happens, its own try/finally will
        # save/restore around its run).
        # The cast-back wrap in TritonDtypeEngine reads
        # _w._NBX_ACTIVATIONS_FP16_SAFE at call time; setting it once
        # here makes Phase 2 uniform cast-back functional in
        # triton_sequential mode (mirror of compiled mode flag init).
        _w.set_compute_dtype(compute_dtype)
        _w.set_activations_fp16_safe(activations_fp16_safe)

    def bind_inputs(self, input_map, graph_tensors):
        """Cast component-entry runtime inputs through the dtype engine.

        Mirrors TritonSequence.bind_inputs (compiled mode) and
        DtypeEngine path at GraphExecutor._prepare_execution
        (sequential mode oracle). Graph floating-point dtype →
        compute_dtype; non-floating → preserved.

        Args:
            input_map: {tensor_id → NBXTensor}.
            graph_tensors: dag["tensors"] dict.

        Returns:
            Cast input_map dict (new dict; tensors unchanged where no
            cast was needed).
        """
        return self._dtype_engine.cast_runtime_inputs(input_map, graph_tensors)

    def resolve_attr(self, attr: Any) -> Any:
        """Resolve a single attribute from graph.json format."""
        if not isinstance(attr, dict):
            return attr

        atype = attr.get("type")
        value = attr.get("value")

        if atype == "dtype":
            if isinstance(value, str):
                s = value.replace("torch.", "")
                try:
                    parsed = parse_dtype(s)
                    # Remap bf16↔fp16 based on Prism compute_dtype
                    if parsed == NBXDtype.bfloat16 and self.compute_dtype == NBXDtype.float16:
                        return NBXDtype.float16
                    if parsed == NBXDtype.float16 and self.compute_dtype == NBXDtype.bfloat16:
                        return NBXDtype.bfloat16
                    # Narrow fp64/complex128 to the triton-supported
                    # fp32/complex64 — R30 mirror of the compiled hot loop
                    # (TritonSequence._parse_dtype). The constant loader
                    # already narrows stored complex128 tables to complex64;
                    # honouring a graph `_to_copy` to complex128 here would
                    # reinterpret the interleaved fp32 pairs as fp64 (Wan
                    # RoPE freqs became near-zero garbage → gray output).
                    if parsed == NBXDtype.float64:
                        return NBXDtype.float32
                    if parsed == NBXDtype.complex128:
                        return NBXDtype.complex64
                    return parsed
                except Exception:
                    return None
            return value

        if atype == "device":
            return f"cuda:{self.device_idx}"

        if atype in ("int", "float", "bool", "str"):
            return value

        if atype in ("None",) or value is None:
            return None

        if atype == "layout":
            return None  # Not used in triton

        if atype == "memory_format":
            return None  # Not used in triton

        if atype == "scalar":
            return value

        if atype == "unknown":
            if isinstance(value, str) and value.startswith("torch."):
                s = value.replace("torch.", "")
                try:
                    parsed = parse_dtype(s)
                    # Same fp64/complex128 narrowing as the "dtype" branch.
                    if parsed == NBXDtype.float64:
                        return NBXDtype.float32
                    if parsed == NBXDtype.complex128:
                        return NBXDtype.complex64
                    return parsed
                except Exception:
                    pass
            return value

        if atype == "tensor":
            return value  # Already resolved by caller

        return value

    def resolve_kwargs(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve kwargs from graph attributes."""
        kwargs_raw = attributes.get("kwargs", {})
        resolved = {}
        for key, value in kwargs_raw.items():
            rv = self.resolve_attr(value)
            if rv is not None:
                resolved[key] = rv
        return resolved

    def dispatch(self, op_type: str, inputs: List[Any],
                 attributes: Dict[str, Any]) -> Any:
        """Dispatch a single op to Triton kernel."""
        clean = op_type.replace("aten::", "").replace("custom::", "")
        base = clean.split(".")[0]

        # Custom ops — apply AMP wrapping.
        # MUST forward graph attribute kwargs (epsilon) — the rms_norm
        # wrapper defaults to eps=1e-6, but the model's real rms_norm_eps
        # (e.g. 1e-5 for Llama/TinyLlama) lives in the op's `kwargs`
        # (graph custom::rms_norm attributes: {"epsilon": 1e-05}). Dropping
        # it silently fell back to 1e-6, diverging from the other three
        # modes (PyTorch-seq/compiled + triton-compiled all forward it).
        # On the first RMSNorm over small-magnitude embeddings mean(x^2)
        # is near the eps scale, so 1e-5 vs 1e-6 is a ~10% denominator
        # swing that compounds through every layer. Mirror the generic
        # path's kwargs forwarding below.
        if op_type == "custom::rms_norm":
            kwargs = self.resolve_kwargs(attributes)
            func = self._dtype_engine.wrap_op("rms_norm", w.rms_norm)
            if kwargs:
                return func(*inputs, **kwargs)
            return func(*inputs)

        # SDPA variants → unified wrapper (AMP wrapping via the wrapper itself)
        if "scaled_dot_product" in base and "attention" in base:
            return self._dispatch_sdpa(base, inputs, attributes)

        # Resolve kwargs
        kwargs = self.resolve_kwargs(attributes)

        # Index casting: embedding/gather/index_select need int64 indices
        if base in ("embedding", "gather", "index_select", "index_add",
                     "scatter", "scatter_add"):
            inputs = self._fix_index_dtypes(base, inputs)

        # Cat: filter empty/scalar tensors
        if base == "cat" and inputs and isinstance(inputs[0], (list, tuple)):
            valid = [t for t in inputs[0]
                     if hasattr(t, 'ndim') and t.ndim > 0 and t.numel() > 0]
            if len(valid) == 0:
                return NBXTensor.empty((0,), self.compute_dtype,
                                      f"cuda:{self.device_idx}")
            if len(valid) == 1:
                return valid[0]
            inputs = [valid] + list(inputs[1:])

        # Lookup kernel and wrap with AMP rules
        func = kernel_dispatch(base)
        if func is None:
            raise RuntimeError(f"[triton-sequential] No kernel for: {op_type}")
        func = self._dtype_engine.wrap_op(base, func)

        if kwargs:
            return func(*inputs, **kwargs)
        return func(*inputs)

    def _dispatch_sdpa(self, base: str, inputs: List[Any],
                       attributes: Dict[str, Any]) -> Any:
        """Handle SDPA variants — route to our unified wrapper.

        SDPA's `attn_mask` / `dropout_p` / `is_causal` / `scale` may arrive
        EITHER positionally OR as graph kwargs. Whisper-style encoders carry
        `scale=1.0` and `is_causal=False` as kwargs with only q/k/v positional
        (the encoder pre-scales Q so SDPA scale must be 1.0, not the wrapper's
        1/sqrt(head_dim) default). Reading these positionally only — as this
        path used to — silently fell back to the wrapper defaults: e.g.
        scale=1/sqrt(64)=0.125 instead of 1.0 → 8x-wrong attention → garbage
        encoder output (Voxtral audio_tower → generic "You're welcome!"
        transcription). The compiled path forwards compiled_kwargs, so seq MUST
        honour the kwargs too (same data-driven discipline as the custom::rms_norm
        epsilon forwarding above). `_pos_or_kw` takes the positional value when
        present, else the resolved kwarg, else the default."""
        q = inputs[0].contiguous() if hasattr(inputs[0], 'contiguous') else inputs[0]
        k = inputs[1].contiguous() if hasattr(inputs[1], 'contiguous') else inputs[1]
        v = inputs[2].contiguous() if hasattr(inputs[2], 'contiguous') else inputs[2]

        kw = self.resolve_kwargs(attributes)

        def _pos_or_kw(idx, key, default):
            if len(inputs) > idx and inputs[idx] is not None:
                return inputs[idx]
            return kw.get(key, default)

        if base == "_scaled_dot_product_flash_attention_for_cpu":
            attn_mask = None
            dropout_p = float(_pos_or_kw(3, "dropout_p", 0.0))
            is_causal = bool(_pos_or_kw(4, "is_causal", False))
            scale = kw.get("scale", None)
            output = w.scaled_dot_product_attention_wrapper(
                q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            lse = NBXTensor.zeros((q.shape[0], q.shape[1], q.shape[2]),
                                  dtype=NBXDtype.float32,
                                  device=f"cuda:{self.device_idx}")
            return output, lse

        if base in ("_scaled_dot_product_efficient_attention",
                     "_scaled_dot_product_flash_attention"):
            # ATen efficient/flash signature has an extra compute_log_sumexp at
            # arg[4]: (q,k,v,attn_bias,compute_log_sumexp,dropout_p,is_causal,scale)
            attn_mask = _pos_or_kw(3, "attn_bias", None)
            if attn_mask is None:
                attn_mask = kw.get("attn_mask")
            dropout_p = _pos_or_kw(5, "dropout_p", 0.0)
            is_causal = _pos_or_kw(6, "is_causal", False)
            scale = _pos_or_kw(7, "scale", None)
            output = w.scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=attn_mask,
                dropout_p=float(dropout_p) if not isinstance(dropout_p, float) else dropout_p,
                is_causal=bool(is_causal) if not isinstance(is_causal, bool) else is_causal,
                scale=scale)
            lse = NBXTensor.zeros((q.shape[0], q.shape[1], q.shape[2]),
                                  dtype=NBXDtype.float32,
                                  device=f"cuda:{self.device_idx}")
            seed = 0
            offset = 0
            return output, lse, seed, offset

        # Standard scaled_dot_product_attention
        # (q,k,v,attn_mask,dropout_p,is_causal,scale)
        attn_mask = _pos_or_kw(3, "attn_mask", None)
        dropout_p = _pos_or_kw(4, "dropout_p", 0.0)
        is_causal = _pos_or_kw(5, "is_causal", False)
        scale = _pos_or_kw(6, "scale", None)
        return w.scaled_dot_product_attention_wrapper(
            q, k, v, attn_mask=attn_mask,
            dropout_p=float(dropout_p) if not isinstance(dropout_p, float) else dropout_p,
            is_causal=bool(is_causal) if not isinstance(is_causal, bool) else is_causal,
            scale=scale)

    def _fix_index_dtypes(self, base: str, inputs: List[Any]) -> List[Any]:
        """Cast floating-point index args to int64."""
        result = list(inputs)
        for idx, inp in enumerate(result):
            is_index = ((base == "embedding" and idx == 1)
                        or (base in ("gather", "index_select", "index_add") and idx == 2)
                        or (base in ("scatter", "scatter_add") and idx == 2))
            if is_index and hasattr(inp, 'dtype') and inp.dtype in (
                    NBXDtype.float16, NBXDtype.float32, NBXDtype.bfloat16):
                result[idx] = inp.to(NBXDtype.int64)
        return result
