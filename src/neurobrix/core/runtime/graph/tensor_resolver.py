"""
Tensor Resolver - Resolve tensor IDs to live torch.Tensors.

Extracted from GraphExecutor for clarity and testability.

Responsibilities:
- Resolve tensor_id to live tensor from store/weights/inputs
- Normalize tensors (device, contiguity, dtype conversion)
- Resolve op arguments from DAG attributes
- Handle symbolic shape resolution

ZERO HARDCODE: All tensor metadata comes from DAG.
ZERO FALLBACK: Missing tensors raise explicit errors.
"""

import torch
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from neurobrix.core.dtype.config import parse_dtype as _parse_dtype

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class TensorResolver:
    """
    Resolve tensor IDs to live torch.Tensors.

    This class encapsulates all tensor resolution logic:
    - Looking up tensors by ID from store/weights/inputs
    - Normalizing tensors for kernel execution
    - Resolving op arguments from DAG attributes

    Usage:
        ctx = ExecutionContext(...)
        resolver = TensorResolver(ctx)
        tensor = resolver.resolve("tensor_123")
        args = resolver.resolve_args(op_uid, attrs, op_type, op_data)
    """

    def __init__(self, ctx: "ExecutionContext"):
        """
        Initialize resolver with execution context.

        Args:
            ctx: ExecutionContext with dag, tensor_store, weights, inputs
        """
        self._ctx = ctx

    def resolve(self, tensor_id: str) -> torch.Tensor:
        """
        Resolve a tensor ID to a live torch.Tensor.

        Resolution order:
        1. Check tensor_store (op outputs, cached weights/inputs)
        2. Check inputs (model inputs)
        3. Check weights (model parameters)

        Args:
            tensor_id: Tensor ID from DAG

        Returns:
            Live torch.Tensor

        Raises:
            RuntimeError: If tensor cannot be resolved
        """
        # 1. Check if already in store (op outputs, cached weights/inputs)
        if tensor_id in self._ctx.tensor_store:
            return self._ctx.tensor_store[tensor_id]

        tensor_data = self._ctx.tensors_metadata.get(tensor_id)
        if not tensor_data:
            raise RuntimeError(f"ZERO FALLBACK: Tensor '{tensor_id}' not found in DAG metadata.")

        # 2. Check model inputs (if not already stored)
        if tensor_data.get("is_input") and tensor_data.get("producer_op_uid") is None:
            input_name = tensor_data.get("input_name")
            if input_name is None:
                raise ValueError(f"ZERO FALLBACK: INPUT tensor {tensor_id} missing 'input_name'")
            # Direct lookup first
            if input_name in self._ctx.inputs:
                tensor = self._ctx.inputs[input_name]
                self._ctx.tensor_store[tensor_id] = tensor
                return tensor
            # Nested dict lookup for dotted names (e.g., "added_cond_kwargs.resolution")
            if "." in input_name:
                parts = input_name.split(".")
                value = self._ctx.inputs
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if value is not None:
                    # Convert to tensor if needed
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=self._ctx.device, dtype=self._ctx.dtype)
                    self._ctx.tensor_store[tensor_id] = value
                    return value

        # 3. Check weights/parameters (if not already stored)
        if tensor_data.get("is_parameter") and tensor_data.get("producer_op_uid") is None:
            weight_name = tensor_data.get("weight_name")
            tensor = self._ctx.weights.get(weight_name) if weight_name else None
            if tensor is None and weight_name:
                # Trailing-suffix fallback (mirror of graph_executor._resolve_weight,
                # the compiled-path bind): a build can store a weight under a SHORTER
                # name than the graph param when the `encoder.`/`model.` prefix is
                # applied inconsistently across a component (Wan UMT5: graph param
                # `encoder.token_embed.weight` vs `token_embed.weight` in the .nbx).
                # _reconcile_weight_keys can miss it when the trailing suffix was
                # ambiguous at reconcile time; strip leading prefix segments,
                # longest-suffix first (each trailing suffix is unique per tensor so
                # this cannot mis-bind). Without this the op-by-op (sequential) path
                # left the embed unbound and NOP-propagated the whole encoder to None,
                # while the compiled path bound it correctly (R30 asymmetry).
                _parts = weight_name.split('.')
                for _i in range(1, len(_parts)):
                    tensor = self._ctx.weights.get('.'.join(_parts[_i:]))
                    if tensor is not None:
                        break
            if tensor is not None:
                self._ctx.tensor_store[tensor_id] = tensor
                return tensor

        # 4. Runtime creation of constant empty tensors
        # Some graphs have constant empty tensors (e.g., for aten.lift_fresh) that weren't embedded
        if "constant" in tensor_id.lower():
            shape = tensor_data.get("shape", [])
            dtype_str = tensor_data.get("dtype", "float32")
            target_dtype = self.parse_dtype(dtype_str) or torch.float32

            # Create empty tensor with the right shape
            const_tensor = torch.empty(shape, dtype=target_dtype, device=self._ctx.device)
            self._ctx.tensor_store[tensor_id] = const_tensor
            return const_tensor

        # 5. Load embedded constant from graph (base64-encoded)
        # Buffers like inv_freq are embedded during trace via embed_constants_in_graph()
        if tensor_data.get("constant") and tensor_data.get("constant_data"):
            import io
            import base64
            encoded = tensor_data["constant_data"]
            buffer = io.BytesIO(base64.b64decode(encoded))
            const_tensor = torch.load(buffer, map_location=self._ctx.device, weights_only=True)
            self._ctx.tensor_store[tensor_id] = const_tensor
            return const_tensor

        # 6. Critical Failure: Not found
        raise RuntimeError(f"ZERO FALLBACK: Tensor '{tensor_id}' could not be resolved.")

    def resolve_normalized(self, tensor_id: str) -> torch.Tensor:
        """
        Resolve tensor and normalize (device + contiguity + dtype).

        MEMORY STRATEGY:
        - Device transfer: cached (one-time cost)
        - Contiguity: cached (avoids repeated .contiguous() copies)
        - Dtype conversion: NOT cached (temporary per-op, weights stay fp32)

        This ensures weights stay as fp32 in VRAM,
        with fp16 conversion only happening temporarily during ops.

        Args:
            tensor_id: Tensor ID from DAG

        Returns:
            Normalized torch.Tensor
        """
        tensor = self.resolve(tensor_id)

        if not isinstance(tensor, torch.Tensor):
            return tensor

        # Track if we need to cache (device/contiguity only, NOT dtype)
        needs_cache_update = False

        # 1. Ensure on correct CUDA device (cache this)
        if not tensor.is_cuda or str(tensor.device) != str(self._ctx.device):
            tensor = tensor.to(self._ctx.device)
            needs_cache_update = True

        # 2. Ensure contiguous (cache this to avoid repeated copies)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            needs_cache_update = True

        # Cache device/contiguity normalized version
        if needs_cache_update:
            self._ctx.tensor_store[tensor_id] = tensor

        # 3. Dtype alignment — WEIGHTS / CONSTANTS / INPUTS (leaves) ONLY.
        # An op-output INTERMEDIATE carries the dtype the execution engine
        # (DtypeEngine AMP) deliberately produced — e.g. an mm/bmm/addmm/div
        # output upcast to fp32 for fp16-overflow protection. The graph-recorded
        # dtype is the STALE trace dtype (fp16) and must NOT override the live
        # AMP dtype: downcasting an AMP-upcast fp32 intermediate back to fp16
        # re-introduces the very overflow the upcast prevented. Witnessed on
        # Open-Sora-v2: double_blocks.18 modulation (scale × hidden ~1.0e4 → mul
        # ~1.0e5) overflowed fp16 → inf → black video, because the fp32 addmm
        # output was downcast back to fp16 at the consuming view::594. Compiled
        # mode (arena slots, no per-op resolver) keeps the fp32 through view→mul
        # and renders correctly — this guard makes sequential mirror it. Only
        # LEAVES (producer_op_uid is None) get aligned to the Prism compute dtype.
        _meta_align = self._ctx.tensors_metadata.get(tensor_id) or {}
        _is_op_output = _meta_align.get("producer_op_uid") is not None
        if tensor.is_floating_point() and not _is_op_output:
            prism_dtype = self._ctx.dtype
            if prism_dtype is not None:
                # If tensor is in graph's half-precision but Prism wants different half,
                # convert to Prism dtype
                if (tensor.dtype in (torch.float16, torch.bfloat16)
                        and tensor.dtype != prism_dtype
                        and prism_dtype in (torch.float16, torch.bfloat16)):
                    tensor = tensor.to(prism_dtype)
                # If tensor is fp32 and graph wants half-precision, convert to Prism dtype
                elif tensor.dtype == torch.float32:
                    tensor_info = self._ctx.tensors_metadata.get(tensor_id, {})
                    graph_dtype_str = tensor_info.get("dtype")
                    if graph_dtype_str:
                        target_dtype = self.parse_dtype(graph_dtype_str)
                        if target_dtype and target_dtype in (torch.float16, torch.bfloat16):
                            tensor = tensor.to(prism_dtype)
                        elif target_dtype and target_dtype != tensor.dtype:
                            tensor = tensor.to(target_dtype)

        return tensor

    def resolve_args(
        self,
        op_uid: str,
        attrs: Dict[str, Any],
        op_type: str,
        op_data: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Resolve positional args from attributes.args.

        Args:
            op_uid: Operation UID
            attrs: Operation attributes dict
            op_type: Operation type string
            op_data: Full operation data (for shape resolution)

        Returns:
            List of resolved arguments

        Raises:
            RuntimeError: If attributes.args is missing
        """
        args_list = attrs.get("args")
        if args_list is None:
            raise RuntimeError(f"ZERO FALLBACK: Op '{op_uid}' ({op_type}) has no attributes.args.")
        resolved = []
        for i, arg in enumerate(args_list):
            # [ALIAS-PRESERVING COPY] aten::copy's first arg is the mutation
            # DESTINATION — a strided view (functionalised in-place slice
            # assignment, e.g. ``out[..., 0::2] = ...`` in SanaVideo's
            # rotate-half RoPE). resolve_normalized() would .contiguous() the
            # non-contiguous view AND replace the store entry with the clone,
            # permanently detaching it from its base buffer — the write then
            # lands in a dead tensor and every consumer of the base reads
            # uninitialized empty_like memory. Resolve RAW: torch ``copy_``
            # handles non-contiguous destinations natively, and the base
            # buffer is engine-produced (already on-device, compute dtype).
            if (i == 0 and op_type.startswith("aten::copy")
                    and isinstance(arg, dict) and arg.get("type") == "tensor"):
                resolved.append(self.resolve(arg["tensor_id"]))
                continue
            resolved.append(self._resolve_arg_info(op_uid, op_type, arg, op_data))
        return resolved

    def resolve_kwargs(
        self,
        op_uid: str,
        attrs: Dict[str, Any],
        op_type: str,
        op_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve keyword args from attributes.kwargs.

        Args:
            op_uid: Operation UID
            attrs: Operation attributes dict
            op_type: Operation type string
            op_data: Full operation data (for shape resolution)

        Returns:
            Dict of resolved keyword arguments
        """
        kwargs_dict = attrs.get("kwargs", {})
        return {k: self._resolve_arg_info(op_uid, op_type, v, op_data) for k, v in kwargs_dict.items()}

    def normalize_inputs(self, tensors: List[Any]) -> List[Any]:
        """
        Normalize ALL inputs before ANY kernel call.

        Handles:
        - CPU -> CUDA device transfer
        - Non-contiguous -> contiguous memory layout

        ZERO FALLBACK: Integer tensors are NOT auto-converted.
        The graph must contain explicit aten::to / aten::_to_copy.

        Args:
            tensors: List of tensors to normalize

        Returns:
            List of normalized tensors
        """
        result = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                result.append(t)
                continue

            # 1. Ensure on correct CUDA device
            if not t.is_cuda or str(t.device) != str(self._ctx.device):
                t = t.to(self._ctx.device)

            # Note: Integer tensors (int64) are valid for embedding, gather, index_select
            # Kernels that don't support integers will crash with clear errors

            # 2. Ensure contiguous for Triton kernels
            if not t.is_contiguous():
                t = t.contiguous()

            result.append(t)
        return result

    def get_input_dtype(self, input_name: str) -> Optional[torch.dtype]:
        """
        Get the expected dtype for an input.

        PRISM DTYPE IS SOURCE OF TRUTH for floating-point tensors.
        Graph dtypes may be stale (traced under autocast with different dtype).

        Only return graph dtype for non-floating-point inputs (int64, bool, etc.)
        to handle cases like timestep where int64 -> float conversion is needed.

        Args:
            input_name: The input name (e.g., "timestep", "hidden_states")

        Returns:
            Expected torch dtype, or None if not found
        """
        tensors = self._ctx.tensors_metadata

        # Search for input tensor with matching input_name
        for tensor_id, tensor_info in tensors.items():
            if tensor_info.get("is_input") and tensor_info.get("input_name") == input_name:
                dtype_str = tensor_info.get("dtype")
                if dtype_str:
                    graph_dtype = self.parse_dtype(dtype_str)
                    if graph_dtype is not None:
                        # For floating-point types, use Prism dtype as source of truth
                        # Graph may have been traced under autocast with different dtype
                        if graph_dtype.is_floating_point:
                            return self._ctx.dtype  # Prism dtype
                        else:
                            # For non-floating types (int64, bool), use graph dtype
                            return graph_dtype
        return None

    def parse_dtype(self, dtype_str: str) -> Optional[torch.dtype]:
        """
        Parse dtype string to torch.dtype.

        Delegates to neurobrix.core.dtype.config.parse_dtype (single source of truth).

        Args:
            dtype_str: Dtype string (e.g., "float16", "float32")

        Returns:
            torch.dtype (defaults to float32 if unknown)
        """
        return _parse_dtype(dtype_str)

    def _resolve_arg_info(
        self,
        op_uid: str,
        op_type: str,
        arg_info: Any,
        op_data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Resolve a single argument info object.

        Args:
            op_uid: Operation UID
            op_type: Operation type string
            arg_info: Argument info (dict with type/value)
            op_data: Full operation data (for shape resolution)

        Returns:
            Resolved argument value
        """
        if not isinstance(arg_info, dict):
            return arg_info

        arg_type = arg_info.get("type")

        if arg_type == "tensor":
            tid = arg_info.get("tensor_id")
            if tid is None:
                raise ValueError(f"ZERO FALLBACK: tensor arg missing 'tensor_id' in {arg_info}")
            return self.resolve_normalized(tid)

        elif arg_type == "scalar":
            return arg_info.get("value")

        elif arg_type == "symbol":
            # Symbolic scalar value (e.g., seq_len for arange)
            # Resolve using shape_resolver to get runtime value
            symbol_id = arg_info.get("symbol_id")
            trace_value = arg_info.get("trace_value")

            if self._ctx.symbolic_shapes_enabled and self._ctx.shape_resolver:
                resolved = self._ctx.shape_resolver.resolve(arg_info)

                # HEURISTIC: Detect config-derived constants that were incorrectly
                # marked as symbolic due to coincidental value match at trace time.
                # Example: frequency_embedding_size/2 = 128 == latent_size at trace
                # but the frequency dim is a model config, not dependent on resolution.
                #
                # Detection: If op is inside "time_embed" or "timestep" module,
                # and the resolved value differs from trace_value, use trace_value.
                # These are model configs that should not change with resolution.
                if op_data and resolved != trace_value:
                    parent_module = op_data.get("parent_module", "")
                    if any(x in parent_module.lower() for x in ["time_embed", "timestep", "freq"]):
                        return trace_value

                return resolved

            # Fallback to trace value if no shape resolver
            return trace_value

        elif arg_type in ("add", "sub", "mul", "floordiv", "div", "mod", "neg"):
            # Symbolic ARITHMETIC expression as a scalar arg — e.g. the
            # MochiRoPE positional-grid `linspace` steps = ((s2 - 2)//2 + 2),
            # an expression over the symbolic height/width dim s2. The compiled
            # sequence folds such expressions to a concrete int at compile time,
            # but the per-op sequential path resolves args LIVE and must evaluate
            # the expression here. The shape_resolver already implements the
            # add/sub/mul/floordiv/mod/neg algebra over runtime symbol bindings
            # (it is the same evaluator used for symbolic dims), so we reuse it —
            # no duplicate arithmetic. Without this the raw expr dict reaches the
            # torch op (linspace steps) and fails the schema cast (dict -> int).
            # R30: mirrors the compiled-mode fold; pure fix (only fires on
            # arithmetic-expr scalar args, which previously crashed).
            if self._ctx.symbolic_shapes_enabled and self._ctx.shape_resolver:
                return self._ctx.shape_resolver.resolve(arg_info)
            # No resolver: fall back to the trace value baked into the expr node.
            return arg_info.get("trace")

        elif arg_type == "list":
            # For view/reshape, check if OUTPUT tensor has symbolic_shape
            # This handles unflatten patterns where trace-time shape differs from runtime
            if op_data and op_type in ("view", "reshape", "_unsafe_view", "aten::view", "aten::reshape", "aten::_unsafe_view"):
                resolved_shape = self._resolve_unflatten_shape(op_uid, op_data)
                if resolved_shape is not None:
                    return resolved_shape

                # Fallback: Check if target shape matches input numel
                # If not, substitute -1 for ONE dimension to infer
                raw_list = arg_info.get("value", [])

                # First resolve any symbolic values in the list
                if self._ctx.symbolic_shapes_enabled and self._ctx.shape_resolver:
                    resolved_list = [self._ctx.shape_resolver.resolve(v) for v in raw_list]
                else:
                    resolved_list = raw_list

                # Only apply fallback for 2D views (flatten/unflatten batch*seq)
                # These are the common cases where pos_embed scaling causes mismatch
                if len(resolved_list) == 2 and -1 not in resolved_list:
                    input_ids = op_data.get("input_tensor_ids", [])
                    if input_ids and input_ids[0] in self._ctx.tensor_store:
                        input_tensor = self._ctx.tensor_store[input_ids[0]]
                        input_numel = input_tensor.numel()
                        target_numel = 1
                        for dim in resolved_list:
                            if isinstance(dim, int) and dim > 0:
                                target_numel *= dim
                        if target_numel != input_numel and target_numel > 0:
                            # Use -1 for first dimension, keep second
                            inferred_shape = [-1, resolved_list[1]]
                            return inferred_shape

                elif len(resolved_list) >= 3 and -1 not in resolved_list:
                    input_ids = op_data.get("input_tensor_ids", [])
                    if input_ids and input_ids[0] in self._ctx.tensor_store:
                        input_tensor = self._ctx.tensor_store[input_ids[0]]
                        input_numel = input_tensor.numel()
                        target_numel = 1
                        for dim in resolved_list:
                            if isinstance(dim, int) and dim > 0:
                                target_numel *= dim
                        if target_numel != input_numel and target_numel > 0:
                            # Infer the axis that ACTUALLY changed between trace and
                            # runtime, not blindly the last one. Mirror of the
                            # compiled _make_view_reshape fallback (R30 parity):
                            # try each axis as -1; a position is a valid candidate
                            # when the product of the OTHER dims divides the input
                            # numel. Selection order: (1) input-dim-match with a
                            # change guard, (2) legacy last-dim when valid,
                            # (3) first valid axis.
                            #
                            # Blind last-dim inference broke two Wan sequential
                            # cases: (1) modulation views — input [2,9216], frozen
                            # target [1,6,1536] gave [1,6,3072] (CFG batch folded
                            # into features) instead of [2,6,1536]; (2) hidden-state
                            # views whose traced shape duplicated the seq expression
                            # into the batch slot ([seq,seq,1536]) — last-dim gave
                            # the invalid [4680,4680,-1] instead of [2,4680,1536].
                            candidates = []  # (axis, inferred_value)
                            for i in range(len(resolved_list)):
                                rest, ok = 1, True
                                for j, d in enumerate(resolved_list):
                                    if j == i:
                                        continue
                                    if isinstance(d, int) and d > 0:
                                        rest *= d
                                    else:
                                        ok = False
                                        break
                                if ok and rest > 0 and input_numel % rest == 0:
                                    candidates.append((i, input_numel // rest))
                            if candidates:
                                chosen = None
                                # 1. Prefer an axis whose recovered value matches
                                #    the input tensor's real dim AND differs from
                                #    the frozen target dim — that axis is the
                                #    changed (e.g. CFG batch 1→2) one. The change
                                #    guard keeps this from firing on dims that did
                                #    not actually move.
                                for i, inferred in candidates:
                                    if (i < input_tensor.ndim
                                            and inferred == input_tensor.shape[i]
                                            and inferred != resolved_list[i]):
                                        chosen = i
                                        break
                                # 1b. Unflatten with a PRESERVED trailing dim: the
                                #    input was flattened by a prior view (e.g.
                                #    [B*tokens, dim] feeding an addmm), so the batch
                                #    that changed is folded into the input's leading
                                #    axis and is NOT visible as a distinct input dim
                                #    (rule 1 cannot see it). But the target's last dim
                                #    still equals the input's last dim — features were
                                #    not touched — so the changed axis must be a
                                #    LEADING one, never the preserved last dim. Legacy
                                #    last-dim inference would fold the batch into
                                #    features here ([1,18720,5120] -> [1,18720,10240]
                                #    under CFG 1->2, crashing norm_q). Prefer the
                                #    earliest candidate (the batch) whose inferred
                                #    value differs from its frozen target dim. Guarded
                                #    by the preserved-last-dim test, so spatial multi-
                                #    resolution (last dim genuinely changed) still
                                #    falls through to rule 2. R30 parity with compiled.
                                if (chosen is None
                                        and input_tensor.ndim >= 1
                                        and isinstance(resolved_list[-1], int)
                                        and resolved_list[-1] == input_tensor.shape[-1]):
                                    last = len(resolved_list) - 1
                                    for i, inferred in candidates:
                                        if i != last and inferred != resolved_list[i]:
                                            chosen = i
                                            break
                                # 2. Else keep the legacy LAST-dim inference when it
                                #    is itself a valid split — the common spatial-
                                #    flatten / LLM attention-reshape case the
                                #    sequential decode path has always used. This
                                #    keeps the fix strictly additive: behavior
                                #    changes ONLY where the old default was provably
                                #    wrong (rule 1 fired) or impossible (rule 3).
                                #    Verified: TinyLlama greedy sequential stays
                                #    byte-identical to compiled.
                                if chosen is None:
                                    last = len(resolved_list) - 1
                                    if any(c[0] == last for c in candidates):
                                        chosen = last
                                    else:
                                        # 3. Last-dim is not inferable (e.g. the
                                        #    traced shape duplicated the seq expr
                                        #    into the batch slot → [4680,4680,-1]
                                        #    invalid); take the first valid axis
                                        #    (mirrors compiled candidates[0]).
                                        chosen = candidates[0][0]
                                trial = list(resolved_list)
                                trial[chosen] = -1
                                return trial
                            # No single -1 yields an integer split — fall back to
                            # the legacy last-dim inference (spatial flattening).
                            inferred_shape = list(resolved_list)
                            inferred_shape[-1] = -1
                            return inferred_shape

            # Resolve symbolic values in lists (e.g., shape = ["s0", 768])
            raw_list = arg_info.get("value", [])
            if self._ctx.symbolic_shapes_enabled and self._ctx.shape_resolver:
                return [self._ctx.shape_resolver.resolve(v) for v in raw_list]
            return raw_list

        elif arg_type == "tensor_tuple":
            tensor_ids = arg_info.get("tensor_ids", [])
            return tuple(self.resolve_normalized(tid) for tid in tensor_ids)

        elif arg_type == "dtype":
            dtype_str = arg_info.get("value", "torch.float32")
            # Single source of truth: parse_dtype handles "torch." prefix + Prism remap
            return _parse_dtype(dtype_str, compute_dtype=self._ctx.dtype)

        elif arg_type == "device":
            return torch.device(arg_info.get("value", "cpu"))

        elif arg_type == "slice":
            return slice(arg_info.get("start"), arg_info.get("stop"), arg_info.get("step"))

        elif arg_type == "memory_format":
            return None

        elif arg_type == "generator":
            return None

        elif arg_type == "layout":
            val = arg_info.get("value", "torch.strided")
            if "strided" in val:
                return torch.strided
            elif "sparse_coo" in val:
                return torch.sparse_coo
            return torch.strided

        elif arg_type == "unknown":
            # Some traces serialize standard torch enum/format singletons with
            # type="unknown" because the trace introspection didn't classify
            # them as `memory_format` / `dtype` / `layout`. When the value is a
            # `torch.<attr>` string and `torch` exposes that attribute, resolve
            # it directly. This unblocks ops like `aten::clone(memory_format=
            # torch.contiguous_format)` whose kwarg lands in this branch when
            # an op_uid interceptor forces kwargs resolution in sequential
            # mode (see graph_executor._execute_native_op). The strict raise
            # is preserved for genuinely unknown values.
            val = arg_info.get("value")
            if isinstance(val, str) and val.startswith("torch."):
                attr_name = val.split(".", 1)[1]
                resolved = getattr(torch, attr_name, None)
                if resolved is not None:
                    return resolved
            # Complex scalar (e.g. the imaginary unit '1j' in the istftnet iSTFT
            # `mul`/`exp`). PyTorch ATen handles complex natively; the other
            # engines already parse it, so the op-by-op sequential dispatcher must
            # too (R30). Serialized as a Python complex or a complex-literal str.
            if isinstance(val, complex):
                return val
            if isinstance(val, str) and val.endswith("j"):
                try:
                    return complex(val)
                except (ValueError, TypeError):
                    pass
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown argument type in {op_type} ({op_uid}) "
                f"value={val!r}")

        return arg_info

    def _resolve_unflatten_shape(
        self,
        op_uid: str,
        op_data: Dict[str, Any]
    ) -> Optional[List[int]]:
        """
        For view/reshape, use OUTPUT tensor's symbolic_shape if available.

        This handles unflatten: [s0*s1, H] -> [s0, s1, H] at runtime.
        The trace-time shape [2, 2, 1152] must be resolved to [2, 120, 1152]
        when s0=2 and s1=120 at runtime.

        Args:
            op_uid: Operation UID
            op_data: Full operation data

        Returns:
            Resolved shape list, or None if no symbolic resolution needed
        """
        if not self._ctx.symbolic_shapes_enabled or not self._ctx.shape_resolver:
            return None

        output_ids = op_data.get("output_tensor_ids", [])
        if not output_ids:
            return None

        out_tid = output_ids[0]
        out_tensor_data = self._ctx.tensors_metadata.get(out_tid, {})
        sym_shape = out_tensor_data.get("symbolic_shape", {})

        if not sym_shape:
            return None

        dims = sym_shape.get("dims", [])
        if not dims:
            return None

        # Check if any dimension is symbolic (not just concrete ints)
        has_symbolic = any(
            isinstance(d, dict) or isinstance(d, str)
            for d in dims
        )

        if not has_symbolic:
            return None

        # Resolve each dimension using the shape resolver
        result = []
        for d in dims:
            resolved = self._ctx.shape_resolver.resolve(d)
            result.append(resolved)

        # SANITY CHECK: Verify the resolved shape has correct total elements
        # If symbolic propagation gave wrong result (e.g., split patterns), fall back
        input_ids = op_data.get("input_tensor_ids", [])
        if input_ids:
            in_tid = input_ids[0]
            if in_tid in self._ctx.tensor_store:
                input_tensor = self._ctx.tensor_store[in_tid]
                input_numel = input_tensor.numel()
                resolved_numel = 1
                for r in result:
                    if r > 0:  # Skip -1 if present
                        resolved_numel *= r
                    else:
                        resolved_numel = -1  # Has infer dim
                        break

                if resolved_numel > 0 and resolved_numel != input_numel:
                    # Symbolic resolution is WRONG! Fall back to original shape with -1
                    return None

        # SANITY CHECK 2: Verify resolved shape order matches concrete shape
        # Catches bugs where symbolic dims are in wrong order (e.g., [1024, 4096] vs [4096, 1024])
        concrete = sym_shape.get("concrete", [])
        if concrete and len(concrete) == len(result):
            # Simple check: if total elements match but shape differs, dimensions are wrong
            if result != list(concrete):
                result_numel = 1
                concrete_numel = 1
                for r in result:
                    result_numel *= r
                for c in concrete:
                    concrete_numel *= c
                # Same numel but different shape order = symbolic dims captured wrong
                if result_numel == concrete_numel:
                    return None

        return result
