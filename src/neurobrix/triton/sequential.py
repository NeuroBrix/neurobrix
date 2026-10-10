"""Triton Sequential Dispatcher — zero torch, op-by-op execution.

Ported from core/runtime/graph/sequential_dispatcher.py.
Resolves graph.json args dynamically and dispatches to Triton kernels.
No pre-compilation (no arena, no closures). Useful for debugging
individual ops and validating graph correctness.

Usage: --triton-sequential flag routes here.
"""

from typing import Optional, Any, Dict, List

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype
from neurobrix.kernels.dispatch import dispatch as kernel_dispatch
from neurobrix.kernels import wrappers as w

from .symbols import SymbolResolver
from neurobrix.core.runtime import symexpr as _symexpr
from .dtype import TritonDtypeEngine


class TritonSequentialDispatcher:
    """Op-by-op dispatcher using Triton kernels. Zero torch.

    Resolves graph.json arg dicts at runtime (no pre-compilation).
    Applies AMP rules via TritonDtypeEngine (same as compiled mode).
    Simpler than TritonSequence — easier to debug individual ops.
    """

    def __init__(self, device_idx: int = 0, compute_dtype: NBXDtype = NBXDtype.float16,
                 activations_fp16_safe: bool = False, precision_contract=None,
                 graph_dtype=None):
        self.device_idx = device_idx
        self.compute_dtype = compute_dtype
        self.activations_fp16_safe = activations_fp16_safe
        from neurobrix.kernels.wrappers import has_native_bf16 as _has_bf16
        from neurobrix.kernels import wrappers as _w
        # graph_dtype: the component's traced `torch_dtype` (the AMP_FP32 cast-back rule
        # reads it under bf16 compute) — the caller hands the DAG's own.
        self._dtype_engine = TritonDtypeEngine(
            compute_dtype, has_native_bf16=_has_bf16(), graph_dtype=graph_dtype)
        if precision_contract is not None:
            # (safe, fp32_op_uids, narrow_op_uids) — the same islands the
            # compiled and Triton-compiled engines honour (R30).
            self._dtype_engine.set_precision_contract(*precision_contract)
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
        _w.begin_run()                      # the per-run caches empty (the step's rotary tables widened once)

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

    # The run's SymbolResolver (set by the executor once it is bound): a symbolic keyword
    # attribute evaluates through it in the scalar slot (core/runtime/symexpr.py).
    symbol_resolver: Optional[SymbolResolver] = None

    def resolve_attr(self, attr: Any) -> Any:
        """Resolve a single attribute from graph.json format."""
        if not isinstance(attr, dict):
            return attr

        if _symexpr.is_symbolic(attr):
            # A symbol or an expression (a linspace bound passed by keyword): evaluated, never
            # dropped — an unknown dict used to resolve to its absent "value" and vanish.
            if self.symbol_resolver is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: symbolic attribute of type {attr.get('type')!r} met with no "
                    "symbol resolver; its trace value is a witnessed extent, not a value")
            return self.symbol_resolver.resolve_scalar(attr)

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
                 attributes: Dict[str, Any], op_uid: Optional[str] = None,
                 op_record: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch one op; while keys are recorded, name it for every key it forms, and no longer
        once it returns (a key a flow forms between ops must not be charged to the last one)."""
        from neurobrix.kernels import census as _key_census
        if not _key_census.recording():
            return self._dispatch(op_type, inputs, attributes, op_uid=op_uid, op_record=op_record)
        _key_census.set_op(op_uid)
        try:
            return self._dispatch(op_type, inputs, attributes, op_uid=op_uid, op_record=op_record)
        finally:
            _key_census.set_op(None)

    def _dispatch(self, op_type: str, inputs: List[Any],
                  attributes: Dict[str, Any], op_uid: Optional[str] = None,
                  op_record: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch a single op to Triton kernel (`op_uid` keys the precision
        contract's per-op islands; `op_record` is the op's graph entry, read
        for the traced `output_dtypes` — see TritonDtypeEngine.wrap_op)."""
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
            func = self._dtype_engine.wrap_op("rms_norm", w.rms_norm,
                                              op_record=op_record)
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
            kind, value = self._cat_inputs_or_refuse(inputs)
            if kind == "single":
                return value
            inputs = value

        # Lookup kernel and wrap with AMP rules (alias-canonical: the AMP
        # sets key on canonical names — aten::multiply must behave as mul).
        from neurobrix.kernels.classification import canonical_aten
        base = canonical_aten(base)
        func = kernel_dispatch(base)
        if func is None:
            raise RuntimeError(f"[triton-sequential] No kernel for: {op_type}")
        func = self._dtype_engine.wrap_op(base, func, op_uid=op_uid,
                                          op_record=op_record)

        if kwargs:
            return func(*inputs, **kwargs)
        return func(*inputs)

    def _cat_inputs_or_refuse(self, inputs):
        """Drop empty and 0-dim operands of a cat. ("single", t) when one operand remains,
        ("inputs", inputs) otherwise.

        When EVERY operand is empty there is no operand to return, and the old answer, a fresh
        `(0,)`, changed the rank of the result (rank 5 -> 1 on Allegro-TI2V's VAE encoder,
        2026-09-24) and let the failure surface 40 ops later as an IndexError. An all-empty cat
        of tensors that have a rank is refused here, with their shapes."""
        valid = [t for t in inputs[0]
                 if hasattr(t, 'ndim') and t.ndim > 0 and t.numel() > 0]
        if len(valid) == 0:
            from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
            shapes = [tuple(getattr(t, "shape", ())) for t in inputs[0]]
            dim = inputs[1] if len(inputs) > 1 else 0
            raise ImpossibleExtentError(
                f"cat of {len(shapes)} operand(s) that are all empty, along dim {dim}: "
                f"{shapes}. There is no operand to return, and an empty result of a "
                f"different rank would travel on as an impossible tensor; an upstream extent "
                f"reached 0")
        if len(valid) == 1:
            return "single", valid[0]
        return "inputs", [valid] + list(inputs[1:])

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

        # The graph's own answer to "is K already transposed", recorded by
        # GraphExecutor._mark_sdpa_k_layout. Passed down rather than
        # re-derived: at seq_len == head_dim the shapes cannot say.
        k_pre_transposed = attributes.get("nbx_k_pre_transposed")

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
                q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                k_pre_transposed=k_pre_transposed)
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
                scale=scale, k_pre_transposed=k_pre_transposed)
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
            scale=scale, k_pre_transposed=k_pre_transposed)

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


class SequentialWeightView:
    """The op-by-op engine's weight store seen as the sequence surface zero3 drives.

    Zero3 pipelines a component's host-resident blocks through six sequence calls
    (get_op_blocks, rebind_partial, recompute_op_devices_for_slots,
    override_weightless_op_devices, mark_cpu_weighted_ops_for_transfer,
    materialize_slots_depending_on). The arena engines answer them over arena slots;
    triton-sequential has no arena, so the ratchet never built there and every host
    weight crossed to the card at every use (Janus-Pro-7B on a 16 GB card, stage B).
    Here a "slot" is the tensor id itself and the arena is the forward's `store`, which
    the executor hands over before each pass (`bind_store`).

    The three device calls change nothing: the dispatcher derives every op's device from
    the weight it reads at that op, so a rebound weight moves its op by construction.
    """

    def __init__(self, execution_order: List[str], ops: Dict[str, Any]):
        from neurobrix.core.runtime import liveness as _liveness
        from .sequence import _BLOCK_RE
        self.store: Dict[str, Any] = {}
        self._op_weights: List[List[str]] = []
        for uid in execution_order:
            op = ops.get(uid) or {}
            self._op_weights.append([
                t for t in _liveness.op_tensor_refs(op)
                if t.startswith("param::") or t.startswith("buffer::")])
        self._tid_to_slot: Dict[str, str] = {
            t: t for tids in self._op_weights for t in tids}
        self._block_re = _BLOCK_RE
        self._op_blocks_cache: Optional[Dict[int, Dict[str, Any]]] = None

    def bind_store(self, store: Dict[str, Any]) -> None:
        self.store = store

    def get_op_blocks(self) -> Dict[int, Dict[str, Any]]:
        """Ops grouped by transformer block, indexed by position in the execution order
        (the index the dispatcher hands the pre-op callback). Same rule as the arena
        engines: an op's block is its first weight's, a weightless op inherits its
        predecessor's, a non-block weight goes to block -1."""
        if self._op_blocks_cache is not None:
            return self._op_blocks_cache
        blocks: Dict[int, Dict[str, Any]] = {}
        last_assigned = -1
        for op_idx, tids in enumerate(self._op_weights):
            block_idx = None
            if tids:
                name = tids[0].split("::", 1)[1]
                m = self._block_re.search(name)
                block_idx = int(m.group(1)) if m else -1
            if block_idx is None:
                block_idx = last_assigned
            last_assigned = block_idx
            entry = blocks.get(block_idx)
            if entry is None:
                entry = blocks[block_idx] = {
                    'first_op': op_idx, 'last_op': op_idx, 'weight_tensor_ids': []}
            else:
                entry['last_op'] = op_idx
            entry['weight_tensor_ids'].extend(tids)
        for entry in blocks.values():
            entry['weight_tensor_ids'] = list(dict.fromkeys(entry['weight_tensor_ids']))
        self._op_blocks_cache = blocks
        return blocks

    def rebind_partial(self, partial_map: Dict[str, Any]) -> List[str]:
        modified: List[str] = []
        for tid, tensor in partial_map.items():
            if tid in self._tid_to_slot:
                self.store[tid] = tensor
                modified.append(tid)
        return modified

    def recompute_op_devices_for_slots(self, modified_slots: List[str]) -> None:
        return None

    def override_weightless_op_devices(self, device_idx: int) -> None:
        return None

    def mark_cpu_weighted_ops_for_transfer(self, exec_device_idx: int) -> int:
        """The count of ops reading a host weight (the dispatcher moves those per use)."""
        store = self.store
        return sum(1 for tids in self._op_weights
                   if any(getattr(store.get(t), '_device', None) == 'cpu' for t in tids))

    def materialize_slots_depending_on(self, weight_slot_ids) -> int:
        """Copy out every live store entry that views one of the weights about to be
        evicted (mirror of TritonSequence.materialize_slots_depending_on)."""
        store = self.store
        roots = set()
        for tid in weight_slot_ids:
            t = store.get(tid)
            if t is not None:
                roots.add(id(getattr(t, '_base', None) or t))
        if not roots:
            return 0
        evicted = set(weight_slot_ids)
        n = 0
        for tid, t in list(store.items()):
            if tid in evicted or t is None:
                continue
            node = getattr(t, '_base', None)
            while node is not None and id(node) not in roots:
                node = getattr(node, '_base', None)
            if node is None:
                continue
            store[tid] = t.contiguous()
            n += 1
        return n
