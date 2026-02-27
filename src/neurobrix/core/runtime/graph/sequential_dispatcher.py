import torch
import torch.nn.functional as F
import logging
from typing import List, Any, Dict, Optional, Callable

from neurobrix.core.dtype.config import parse_dtype

logger = logging.getLogger(__name__)


def _fix_sdpa_kv_layout(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Fix K/V tensor layout for SDPA.

    Some text encoders (Gemma-2) produce K with transposed seq/head_dim:
      K: [batch, heads, head_dim, seq_len] instead of [batch, heads, seq_len, head_dim]

    Detect by comparing against Q's known-good layout [batch, heads, seq_q, head_dim].
    If K's last dim != Q's last dim (head_dim), transpose K (and V if needed).
    """
    if q.ndim == 4 and k.ndim == 4:
        q_head_dim = q.shape[-1]
        k_last_dim = k.shape[-1]
        # If K's last dim doesn't match Q's head_dim, K is likely transposed
        if k_last_dim != q_head_dim and k.shape[-2] == q_head_dim:
            k = k.transpose(-2, -1).contiguous()
        v_last_dim = v.shape[-1]
        if v_last_dim != q_head_dim and v.shape[-2] == q_head_dim:
            v = v.transpose(-2, -1).contiguous()
    return q, k, v


class NativeATenDispatcher:
    """
    Native ATen Dispatcher - Bypass Triton for validation.

    This class executes graph operations using pure PyTorch ATen ops.
    Used to validate graph topology and weights when Triton kernels are suspect.
    """

    # No local DTYPE_MAP — uses neurobrix.core.dtype.config.parse_dtype()

    def __init__(self, device: Optional[str] = None, compute_dtype: Optional[torch.dtype] = None):
        # Cache for resolved operations
        self._op_cache = {}
        # Runtime device from Prism - overrides hardcoded graph devices
        self._runtime_device = device
        # Prism compute dtype - remaps graph dtype (e.g., bf16→fp16 on V100)
        self._compute_dtype = compute_dtype

    def _resolve_attr_value(self, attr_value: Any) -> Any:
        """
        Resolve attribute value from graph format to PyTorch native format.

        Graph stores typed values like:
            {"type": "dtype", "value": "torch.float16"}
            {"type": "device", "value": "cuda:0"}
            {"type": "int", "value": 1}

        This resolves them to actual PyTorch types.
        SOURCE OF TRUTH: The graph attributes dict.
        """
        if not isinstance(attr_value, dict):
            return attr_value

        attr_type = attr_value.get("type")
        value = attr_value.get("value")

        if attr_type == "dtype":
            # Resolve dtype string to torch.dtype with Prism remap
            if isinstance(value, str):
                return parse_dtype(value, compute_dtype=self._compute_dtype)
            return value

        elif attr_type == "device":
            # Use runtime device from Prism (overrides hardcoded graph device)
            if self._runtime_device:
                return torch.device(self._runtime_device)
            # Fallback to graph value
            if isinstance(value, str):
                return torch.device(value)
            return value

        elif attr_type == "layout":
            # Resolve layout (usually strided)
            if value == "torch.strided":
                return torch.strided
            elif value == "torch.sparse_coo":
                return torch.sparse_coo
            return value

        elif attr_type == "memory_format":
            if value == "torch.contiguous_format":
                return torch.contiguous_format
            elif value == "torch.channels_last":
                return torch.channels_last
            return value

        elif attr_type in ("int", "float", "bool", "str"):
            return value

        elif attr_type == "tensor":
            # Tensor references should already be resolved by graph executor
            return value

        elif attr_type == "None" or value is None:
            return None

        elif attr_type == "unknown":
            # Graph stores some values as "unknown" type - try to resolve by value content
            if isinstance(value, str):
                # Memory format strings
                if value == "torch.contiguous_format":
                    return torch.contiguous_format
                elif value == "torch.channels_last":
                    return torch.channels_last
                elif value == "torch.channels_last_3d":
                    return torch.channels_last_3d
                elif value == "torch.preserve_format":
                    return torch.preserve_format
                # Layout strings
                elif value == "torch.strided":
                    return torch.strided
                elif value == "torch.sparse_coo":
                    return torch.sparse_coo
                # Dtype strings
                elif value.startswith("torch."):
                    return parse_dtype(value)
            return value

        # Unknown type - return as-is
        return value

    def _extract_kwargs(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and resolve kwargs from graph attributes.

        SOURCE OF TRUTH: attributes["kwargs"] from graph.
        """
        kwargs_raw = attributes.get("kwargs", {})
        resolved = {}

        for key, value in kwargs_raw.items():
            resolved_value = self._resolve_attr_value(value)
            # Skip None values to use op defaults
            if resolved_value is not None:
                resolved[key] = resolved_value

        return resolved

    def dispatch(self, op_type: str, inputs: List[Any], attributes: Dict[str, Any]) -> Any:
        """
        Dispatch operation to native PyTorch.
        """
        # [CUSTOM OPS] Handle pattern-reassembled ops before standard dispatch
        if op_type == "custom::rms_norm":
            return self._dispatch_rms_norm(inputs, attributes)

        # 1. Clean operation name
        clean_name = op_type.replace("aten::", "")
        # Remove variant suffixes if present (e.g. .default, .int)
        base_name = clean_name.split(".")[0]

        # [IDENTITY OPS] These ops just pass through their input unchanged
        # lift_fresh: Used by FakeTensorMode to mark tensors as "fresh"
        # Note: Graph capture sometimes incorrectly links lift_fresh to wrong tensor.
        if base_name == "lift_fresh":
            if not inputs:
                return None
            inp = inputs[0]
            # If input is already scalar, return as-is
            if not isinstance(inp, torch.Tensor) or inp.ndim == 0:
                return inp

            # Check for graph capture bug: expected scalar/empty but got tensor
            # Compare graph-recorded shapes (what tracer expected) vs actual input shape
            graph_input_shapes = attributes.get("input_shapes", [])
            graph_output_shapes = attributes.get("output_shapes", [])

            if not graph_output_shapes:
                return inp

            expected_output_shape = graph_output_shapes[0] if graph_output_shapes else None

            # Case 1: Graph expected scalar (shape=[]) but got multi-dim tensor
            if expected_output_shape == []:
                # For Gemma2/T5/etc: embedding *= sqrt(hidden_size)
                import math
                if len(inp.shape) >= 2:
                    # UNIVERSAL: derive hidden_size from tensor shape, not a lookup table.
                    hidden_size = inp.shape[-1]
                    scale = math.sqrt(hidden_size)
                    return torch.tensor(scale, dtype=inp.dtype, device=inp.device)
                return torch.tensor(1.0, dtype=inp.dtype, device=inp.device)

            # Case 2: Graph expected empty tensor (shape=[0]) but got real tensor
            if expected_output_shape == [0]:
                # Return empty tensor with expected shape
                return torch.empty(0, dtype=inp.dtype, device=inp.device)

            # Case 3: Other shape mismatches - just return input
            return inp

        # [REDIRECTION LAYER] Map device-specific variants to universal ops
        if "scaled_dot_product" in base_name and "attention" in base_name:
            if base_name == "_scaled_dot_product_flash_attention_for_cpu":
                # CPU variant signature: (q, k, v, dropout_p, is_causal, *, return_debug_mask, scale)
                # Returns (output, logsumexp) — no attn_mask parameter
                q = inputs[0]
                k = inputs[1]
                v = inputs[2]
                q, k, v = _fix_sdpa_kv_layout(q, k, v)
                dropout_p = float(inputs[3]) if len(inputs) > 3 and isinstance(inputs[3], (int, float)) else 0.0
                is_causal = bool(inputs[4]) if len(inputs) > 4 else False
                scale = attributes.get("kwargs", {}).get("scale", None)

                output = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )

                # Create dummy logsumexp [batch, heads, seq_len]
                # Based on T5 flash attention signature
                lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)

                return output, lse

            if base_name in ("_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention"):
                # Efficient/Flash attention returns 4 values: output, logsumexp, philox_seed, philox_offset
                q = inputs[0].contiguous()
                k = inputs[1].contiguous()
                v = inputs[2].contiguous()
                q, k, v = _fix_sdpa_kv_layout(q, k, v)

                # Extract optional args - correct indices from PyTorch internal signature:
                # _scaled_dot_product_efficient_attention(query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, scale)
                # inputs[3] = attn_bias (None)
                # inputs[4] = compute_log_sumexp (bool)
                # inputs[5] = dropout_p (float)
                # inputs[6] = is_causal (bool) <-- CRITICAL for LLMs
                # inputs[7] or kwargs['scale'] = scale
                attn_mask = inputs[3] if len(inputs) > 3 and inputs[3] is not None else None
                # inputs[4] is compute_log_sumexp, skip it
                dropout_p = inputs[5] if len(inputs) > 5 else 0.0
                is_causal = inputs[6] if len(inputs) > 6 else False
                scale = inputs[7] if len(inputs) > 7 else None

                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p if isinstance(dropout_p, float) else 0.0,
                    is_causal=is_causal if isinstance(is_causal, bool) else False,
                    scale=scale
                )

                # Create dummy outputs for logsumexp, philox_seed, philox_offset
                lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)
                philox_seed = torch.tensor(0, device=q.device, dtype=torch.int64)
                philox_offset = torch.tensor(0, device=q.device, dtype=torch.int64)

                return output, lse, philox_seed, philox_offset

            base_name = "scaled_dot_product_attention"
            if len(inputs) > 7:
                inputs = inputs[:7]
            # Fix K/V layout for standard SDPA path (Gemma-2 text encoder K transposition)
            if len(inputs) >= 3:
                q, k, v = _fix_sdpa_kv_layout(inputs[0], inputs[1], inputs[2])
                inputs = [q, k, v] + list(inputs[3:])

        # [MULTI-RESOLUTION FIX] Upsample ops have hardcoded output_size from trace time.
        # When scale factors (scales_h, scales_w) are available, recompute output_size
        # from actual input tensor dimensions to support multi-resolution inference.
        if base_name in ("upsample_nearest2d", "upsample_bilinear2d", "upsample_bicubic2d"):
            # inputs[0] = input tensor
            # inputs[1] = output_size (hardcoded from trace time)
            # inputs[2] = scales_h (if available)
            # inputs[3] = scales_w (if available)
            if len(inputs) >= 4 and isinstance(inputs[0], torch.Tensor):
                input_tensor = inputs[0]
                scales_h = inputs[2]
                scales_w = inputs[3]

                # If scale factors are available and valid, recompute output_size
                if scales_h is not None and scales_w is not None:
                    if isinstance(scales_h, (int, float)) and isinstance(scales_w, (int, float)):
                        # Get actual input spatial dimensions
                        h_in = input_tensor.shape[2]
                        w_in = input_tensor.shape[3]

                        # Compute correct output size from actual input
                        new_h = int(h_in * scales_h)
                        new_w = int(w_in * scales_w)

                        # Replace hardcoded output_size with computed value
                        inputs[1] = [new_h, new_w]

        # [MULTI-RESOLUTION FIX] Expand ops may have hardcoded spatial dimensions
        # When the tensor has non-singleton dimensions that don't match the expand size,
        # we should use the actual tensor dimensions (expand can only broadcast from 1).
        if base_name == "expand":
            if len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor):
                input_tensor = inputs[0]
                expand_size = inputs[1]
                if isinstance(expand_size, (list, tuple)) and len(expand_size) == len(input_tensor.shape):
                    new_size = list(expand_size)
                    fixed = False
                    for i, (actual, target) in enumerate(zip(input_tensor.shape, expand_size)):
                        # expand can only broadcast from size 1
                        # If actual != 1 and actual != target, use actual
                        if actual != 1 and actual != target and target != -1:
                            new_size[i] = actual
                            fixed = True
                    if fixed:
                        inputs[1] = new_size

        try:
            # 2. Find the operator
            op_fn = self._resolve_op(base_name)
            
            # 3. Execute
            # Execute via PyTorch Natif
            
            # [STABILITY FIX] Some ATen ops are very strict about dtypes for indices
            if base_name in ("embedding", "gather", "index_select", "index_add", "scatter", "scatter_add"):
                new_inputs = []
                for idx, inp in enumerate(inputs):
                    is_index_arg = (base_name == "embedding" and idx == 1) or \
                                  (base_name in ("gather", "index_select", "index_add") and idx == 2) or \
                                  (base_name in ("scatter", "scatter_add") and idx == 2)

                    if is_index_arg and isinstance(inp, torch.Tensor) and inp.is_floating_point():
                        new_inputs.append(inp.long())
                    else:
                        new_inputs.append(inp)
                inputs = new_inputs

            # [STABILITY FIX] Handle cat with scalar/empty tensors (Gemma2 attention pattern)
            # Gemma2 creates scalar -inf tensors for attention masking, then tries to cat them.
            # A 0-dimensional tensor cannot be concatenated - filter them out.
            if base_name == "cat":
                # Input[0] is tensor tuple, Input[1] is dim
                if len(inputs) >= 1:
                    tensor_tuple = inputs[0]
                    if isinstance(tensor_tuple, (list, tuple)):
                        # Filter out 0-dimensional and empty tensors
                        valid_tensors = []
                        for t in tensor_tuple:
                            if isinstance(t, torch.Tensor):
                                # Skip 0-dim scalars (ndim=0) and empty 1D tensors (numel=0)
                                if t.ndim == 0 or t.numel() == 0:
                                    continue
                            valid_tensors.append(t)

                        if len(valid_tensors) == 0:
                            # All tensors were filtered - return empty tensor
                            dim = inputs[1] if len(inputs) > 1 else 0
                            return torch.tensor([], device=tensor_tuple[0].device if tensor_tuple else 'cpu')
                        elif len(valid_tensors) == 1:
                            # Only one tensor remains - no cat needed
                            return valid_tensors[0]
                        else:
                            # Update inputs with filtered tensors
                            inputs = [valid_tensors] + list(inputs[1:])

            # [UNIVERSAL KWARGS] Extract kwargs from graph attributes (SOURCE OF TRUTH)
            # This handles dtype, device, layout, memory_format for ALL ops universally
            kwargs = self._extract_kwargs(attributes)

            if kwargs:
                result = op_fn(*inputs, **kwargs)
            else:
                result = op_fn(*inputs)

            return result

        except Exception as e:
            logger.error(f"[Sandbox Error] Failed to dispatch {op_type}")
            logger.error(f"  Error: {str(e)}")
            raise

    def _resolve_op(self, op_name: str) -> Any:
        """Dynamically find the PyTorch function."""
        if op_name in self._op_cache:
            return self._op_cache[op_name]

        # Priority 1: torch.ops.aten
        if hasattr(torch.ops.aten, op_name):
            op = getattr(torch.ops.aten, op_name)
            self._op_cache[op_name] = op
            return op

        # Priority 2: torch namespace
        if hasattr(torch, op_name):
            op = getattr(torch, op_name)
            self._op_cache[op_name] = op
            return op

        # Priority 3: torch.nn.functional
        if hasattr(F, op_name):
            op = getattr(F, op_name)
            self._op_cache[op_name] = op
            return op

        raise AttributeError(f"Native operator '{op_name}' not found in torch.ops.aten, torch, or torch.nn.functional")

    def get_native_op(self, op_name: str) -> Any:
        """
        Public API for getting a native PyTorch function reference.

        Used by CompiledSequence to pre-bind function references at compile time.
        Returns wrapped functions for ops that need tensor-to-scalar conversion.

        Args:
            op_name: The operation name (e.g., "mm", "add", "relu")

        Returns:
            The PyTorch function callable (possibly wrapped), or None if not found
        """
        try:
            raw_op = self._resolve_op(op_name)

            # Wrap ops that need tensor-to-scalar conversion
            # slice.Tensor expects Python int for dim, start, end, step
            if op_name == "slice":
                return self._make_slice_wrapper(raw_op)

            # narrow expects Python int for dim, start, length
            if op_name == "narrow":
                return self._make_narrow_wrapper(raw_op)

            # select expects Python int for dim, index
            if op_name == "select":
                return self._make_select_wrapper(raw_op)

            # squeeze/unsqueeze expect Python int for dim
            if op_name in ("squeeze", "unsqueeze"):
                return self._make_dim_wrapper(raw_op)

            # transpose expects Python int for dim0, dim1
            if op_name == "transpose":
                return self._make_transpose_wrapper(raw_op)

            # split/chunk expect Python int for split_size/chunks, dim
            if op_name in ("split", "chunk"):
                return self._make_split_wrapper(raw_op)

            # embedding/gather/index_select expect int64 indices (not float)
            if op_name in ("embedding", "gather", "index_select", "index_add", "scatter", "scatter_add"):
                return self._make_index_wrapper(raw_op, op_name)

            return raw_op
        except AttributeError:
            return None

    def _tensor_to_int(self, val: Any) -> Any:
        """Convert 0-d tensor to Python int if needed."""
        if isinstance(val, torch.Tensor) and val.ndim == 0:
            return int(val.item())
        return val

    def _tensor_to_int_or_none(self, val: Any) -> Any:
        """Convert 0-d tensor to Python int, or return None if None."""
        if val is None:
            return None
        return self._tensor_to_int(val)

    def _make_slice_wrapper(self, raw_op: Any) -> Callable:
        """Wrap slice.Tensor to convert tensor args to Python ints."""
        def wrapper(input_tensor, dim, start=None, end=None, step=1):
            dim = self._tensor_to_int(dim)
            start = self._tensor_to_int_or_none(start)
            end = self._tensor_to_int_or_none(end)
            step = self._tensor_to_int(step)
            return raw_op(input_tensor, dim, start, end, step)
        return wrapper

    def _make_narrow_wrapper(self, raw_op: Any) -> Callable:
        """Wrap narrow to convert tensor args to Python ints."""
        def wrapper(input_tensor, dim, start, length):
            dim = self._tensor_to_int(dim)
            start = self._tensor_to_int(start)
            length = self._tensor_to_int(length)
            return raw_op(input_tensor, dim, start, length)
        return wrapper

    def _make_select_wrapper(self, raw_op: Any) -> Callable:
        """Wrap select to convert tensor args to Python ints."""
        def wrapper(input_tensor, dim, index):
            dim = self._tensor_to_int(dim)
            index = self._tensor_to_int(index)
            return raw_op(input_tensor, dim, index)
        return wrapper

    def _make_dim_wrapper(self, raw_op: Any) -> Callable:
        """Wrap squeeze/unsqueeze to convert dim tensor to Python int."""
        def wrapper(input_tensor, dim=None):
            if dim is not None:
                dim = self._tensor_to_int(dim)
            return raw_op(input_tensor, dim) if dim is not None else raw_op(input_tensor)
        return wrapper

    def _make_transpose_wrapper(self, raw_op: Any) -> Callable:
        """Wrap transpose to convert dim tensors to Python ints."""
        def wrapper(input_tensor, dim0, dim1):
            dim0 = self._tensor_to_int(dim0)
            dim1 = self._tensor_to_int(dim1)
            return raw_op(input_tensor, dim0, dim1)
        return wrapper

    def _make_split_wrapper(self, raw_op: Any) -> Callable:
        """Wrap split/chunk to convert tensor args to Python ints."""
        def wrapper(input_tensor, split_size_or_sections, dim=0):
            if isinstance(split_size_or_sections, torch.Tensor):
                split_size_or_sections = int(split_size_or_sections.item())
            dim = self._tensor_to_int(dim)
            return raw_op(input_tensor, split_size_or_sections, dim)
        return wrapper

    def _make_index_wrapper(self, raw_op: Any, op_name: str) -> Callable:
        """
        Wrap embedding/gather/index_select/scatter ops to convert float indices to int64.

        These ops require integer indices but graph execution may flow float tensors
        through arithmetic operations before reaching these ops.
        """
        if op_name == "embedding":
            # embedding(weight, indices, ...) - indices is arg 1
            def embedding_wrapper(weight, indices, *args, **kwargs):
                if isinstance(indices, torch.Tensor) and indices.is_floating_point():
                    indices = indices.long()
                return raw_op(weight, indices, *args, **kwargs)
            return embedding_wrapper

        elif op_name in ("gather", "index_select", "index_add"):
            # gather(input, dim, index) - index is arg 2
            # index_select(input, dim, index) - index is arg 2
            # index_add(input, dim, index, source) - index is arg 2
            def gather_wrapper(input_tensor, dim, index, *args, **kwargs):
                dim = self._tensor_to_int(dim)
                if isinstance(index, torch.Tensor) and index.is_floating_point():
                    index = index.long()
                return raw_op(input_tensor, dim, index, *args, **kwargs)
            return gather_wrapper

        elif op_name in ("scatter", "scatter_add"):
            # scatter(input, dim, index, src) - index is arg 2
            # scatter_add(input, dim, index, src) - index is arg 2
            def scatter_wrapper(input_tensor, dim, index, src, *args, **kwargs):
                dim = self._tensor_to_int(dim)
                if isinstance(index, torch.Tensor) and index.is_floating_point():
                    index = index.long()
                return raw_op(input_tensor, dim, index, src, *args, **kwargs)
            return scatter_wrapper

        # Fallback - shouldn't happen
        return raw_op

    def _dispatch_rms_norm(self, inputs: List[Any], attributes: Dict[str, Any]) -> Any:
        """Dispatch custom::rms_norm (pattern-reassembled RMSNorm)."""
        x = inputs[0]
        weight = inputs[1]
        epsilon = attributes.get("epsilon", 1e-6)
        if not epsilon:
            epsilon = attributes.get("kwargs", {}).get("epsilon", 1e-6)
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + epsilon)
        return x_normed.to(weight.dtype) * weight