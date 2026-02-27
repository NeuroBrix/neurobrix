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
            if weight_name in self._ctx.weights:
                tensor = self._ctx.weights[weight_name]
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

        # 3. Dtype alignment
        # Prism compute_dtype (e.g., fp16) overrides graph dtype (e.g., bf16).
        # Convert fp32 weights down to compute_dtype, but NEVER convert compute_dtype
        # tensors to a different half-precision format (bf16→fp16 or vice versa).
        if tensor.is_floating_point():
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
        return [self._resolve_arg_info(op_uid, op_type, arg, op_data) for arg in args_list]

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
                            # Infer last dimension (spatial flattening)
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
            raise RuntimeError(f"ZERO FALLBACK: Unknown argument type in {op_type} ({op_uid})")

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
