"""
Metadata Operations Implementation - NeuroBrix
ATen Native Operations - Pure PyTorch

Ces opérations manipulent les shapes, les indices et les constantes.
ZERO type hardening dangereux - on préserve les dtypes originaux.
Toutes les ops sont en format aten::* (trace ATen native).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Callable, Union, Optional, Tuple

# ==============================================================================
# HELPERS
# ==============================================================================

def _flatten_shape(shape) -> List[int]:
    """
    Flatten nested lists/tensors to a flat list of ints.
    Ex: [[1], [2], [3]] → [1, 2, 3]
    Ex: tensor([1, 2]) → [1, 2]
    """
    if isinstance(shape, torch.Tensor):
        shape = shape.tolist()

    if not isinstance(shape, (list, tuple)):
        return [int(shape)]

    result = []
    for item in shape:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten_shape(item))
        elif isinstance(item, torch.Tensor):
            result.extend(_flatten_shape(item.tolist()))
        else:
            result.append(int(item))
    return result


def _to_int_list(t) -> List[int]:
    """Convert tensor or list to list of ints."""
    if isinstance(t, torch.Tensor):
        return [int(x) for x in t.flatten().tolist()]
    elif isinstance(t, (list, tuple)):
        return [int(x) for x in t]
    elif isinstance(t, (int, float)):
        return [int(t)]
    else:
        raise TypeError(f"Cannot convert {type(t)} to int list")


# =============================================================================
# UNIVERSAL PARSING HELPERS - ZERO FALLBACK
# =============================================================================

def _parse_dtype(kwargs: Dict[str, Any]) -> Optional[torch.dtype]:
    """
    Parse dtype from attrs['kwargs'].
    ZERO FALLBACK: unknown dtype → explicit error.
    """
    if not kwargs or "dtype" not in kwargs:
        return None

    dtype_val = kwargs["dtype"].get("value")
    if dtype_val is None:
        raise RuntimeError("dtype specified but value is None")

    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float64": torch.float64,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.bool": torch.bool,
    }

    if dtype_val not in dtype_map:
        raise RuntimeError(f"Unsupported dtype: {dtype_val}")

    return dtype_map[dtype_val]


def _parse_device(kwargs: Dict[str, Any]) -> Optional[torch.device]:
    """
    Parse device from attrs['kwargs'].
    """
    if not kwargs or "device" not in kwargs:
        return None

    dev = kwargs["device"].get("value")
    if dev is None:
        raise RuntimeError("device specified but value is None")

    return torch.device(dev)


def _parse_shape(arg: Any) -> Tuple[int, ...]:
    """
    Parse shape argument for creation ops.
    Accepted formats:
      - {"type": "list" | "int_list", "value": [..]}
      - {"value": [..]}
      - list / tuple of ints
    ZERO FALLBACK otherwise.
    """
    if isinstance(arg, dict):
        if "value" not in arg:
            raise RuntimeError(f"Invalid shape arg (no value): {arg}")
        shape = arg["value"]
    else:
        shape = arg

    if not isinstance(shape, (list, tuple)):
        raise RuntimeError(f"Shape must be list/tuple, got {shape}")

    try:
        return tuple(int(x) for x in shape)
    except Exception:
        raise RuntimeError(f"Invalid shape contents: {shape}")


def _parse_scalar(arg: Any) -> Any:
    """
    Parse scalar argument.
    """
    if isinstance(arg, dict):
        if "value" not in arg:
            raise RuntimeError(f"Scalar arg missing value: {arg}")
        return arg["value"]
    return arg


# ==============================================================================
# SHAPE OPERATIONS (ATen Native)
# ==============================================================================

def _shape(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen size - Returns shape as Int64 Tensor.
    aten::size(Tensor self) -> int[]
    """
    data = inputs[0]
    start = attrs.get("start", 0)
    end = attrs.get("end", None)
    dims = list(data.shape)
    sliced = dims[start:end] if end is not None else dims[start:]
    # Shape output is always int64 on CPU
    return torch.tensor(sliced, dtype=torch.int64, device='cpu')


def _gather(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen gather/index_select.
    aten::gather(Tensor self, int dim, Tensor index) -> Tensor
    aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
    """
    data = inputs[0]
    indices = inputs[1]
    axis = attrs.get("axis", 0)

    # Handle negative axis
    if axis < 0:
        axis = data.ndim + axis

    # Save original indices shape for output reshape
    indices_shape = list(indices.shape)

    # Flatten indices for index_select
    indices_flat = indices.flatten().to(torch.int64)

    # Handle negative indices (range [-s, s-1])
    s = data.shape[axis]
    indices_flat = torch.where(indices_flat < 0, indices_flat + s, indices_flat)

    # Move to same device as data
    indices_flat = indices_flat.to(data.device)

    # Gather using index_select
    result = torch.index_select(data, dim=axis, index=indices_flat)

    # Reshape to output shape
    output_shape = list(data.shape[:axis]) + indices_shape + list(data.shape[axis+1:])
    return result.reshape(output_shape)


def _slice(inputs: List, attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen slice - torch.Tensor.slice(dim, start, end, step)

    ATen signature: aten::slice(Tensor self, int dim, int start, int end, int step=1)
    Args from capture:
    - inputs[0]: tensor
    - inputs[1]: dim (int)
    - inputs[2]: start (int)
    - inputs[3]: end (int)
    - inputs[4]: step (int, optional, default=1)
    """
    data = inputs[0]

    # ATen format: dim, start, end, [step]
    dim = int(inputs[1]) if len(inputs) > 1 else 0
    start = int(inputs[2]) if len(inputs) > 2 else 0
    end = int(inputs[3]) if len(inputs) > 3 else data.shape[dim]
    step = int(inputs[4]) if len(inputs) > 4 else 1

    # Handle negative dim
    if dim < 0:
        dim = data.ndim + dim

    # Handle INT64_MAX (Python's max int from torch capture)
    if end > data.shape[dim] or end == 9223372036854775807:
        end = data.shape[dim]

    # Build slice tuple
    slices = [slice(None)] * data.ndim
    slices[dim] = slice(start, end, step)

    return data[tuple(slices)]


def _select(inputs: List, attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen select - Select single index along dimension (removes dim).

    ATen signature: aten::select.int(Tensor self, int dim, int index) -> Tensor
    Args from capture:
    - inputs[0]: tensor
    - inputs[1]: dim (int)
    - inputs[2]: index (int)
    """
    data = inputs[0]

    # Get dim and index
    dim = int(inputs[1]) if len(inputs) > 1 else attrs.get("dim", 0)
    index = int(inputs[2]) if len(inputs) > 2 else attrs.get("index", 0)

    # Handle negative dim
    if dim < 0:
        dim = data.ndim + dim

    # Handle negative index
    if index < 0:
        index = data.shape[dim] + index

    # Select removes the dimension (unlike slice which keeps it)
    return data.select(dim, index)


def _concat(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen cat - concatenate tensors along axis.
    aten::cat(Tensor[] tensors, int dim=0) -> Tensor

    Input format from graph:
    - inputs[0]: tuple/list of tensors (from tensor_tuple in args)
    - inputs[1]: dim (optional, also in attrs)
    """
    if not inputs:
        raise ValueError("Concat requires at least one input")

    # Handle tensor_tuple format: first arg is tuple/list of tensors
    if isinstance(inputs[0], (tuple, list)):
        tensors = list(inputs[0])
        # Get dim from second arg or attrs
        axis = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], int) else attrs.get("dim", attrs.get("axis", 0))
    else:
        # Legacy format: inputs are individual tensors
        tensors = [t for t in inputs if isinstance(t, torch.Tensor)]
        axis = attrs.get("dim", attrs.get("axis", 0))

    if not tensors:
        raise ValueError("Concat requires at least one tensor")

    # Handle negative axis
    if axis < 0:
        axis = tensors[0].ndim + axis

    # Align devices to first tensor's device
    target_device = tensors[0].device
    aligned = [t.to(target_device) if t.device != target_device else t for t in tensors]

    return torch.cat(aligned, dim=axis)


def _reshape(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen reshape/view - reshape tensor to new shape.
    aten::reshape(Tensor self, int[] shape) -> Tensor
    aten::view(Tensor self, int[] size) -> Tensor

    RELATIVE RESHAPE - ZERO HARDCODE:
    Graph captures absolute shapes. At runtime, we adapt by computing
    the ratio between graph input numel and runtime input numel, then
    using -1 to let PyTorch infer the dynamic dimension.

    BATCH-AWARE FALLBACK:
    If total elements don't match but the mismatch is cleanly at dim 0
    (batch dimension), we scale dim 0 proportionally. This handles cases
    where CFG creates batch=2 during trace but runtime uses batch=1.
    """
    data = inputs[0]

    # Get shape from input tensor or attrs
    if len(inputs) > 1:
        shape_list = _to_int_list(inputs[1])
    else:
        shape_list = _flatten_shape(attrs.get("shape", []))

    if not shape_list:
        return data

    allowzero = attrs.get("allowzero", 0)

    # =========================================================================
    # BATCH-AWARE FALLBACK (Universal)
    # =========================================================================
    # Check if elements match or can be resolved via batch scaling
    input_elements = data.numel()
    static_elements = 1
    for d in shape_list:
        if d > 0:
            static_elements *= d

    # Case 1: Elements match exactly → use static shape directly
    if input_elements == static_elements:
        return data.reshape(shape_list)

    # Case 2: Batch Merge Pattern (V2 graphs with CFG)
    # Pattern: [batch*k, features] -> [batch, features*k] during trace
    # At runtime: [runtime_batch*k, features] -> [runtime_batch, features*k]
    # Example: [4, 384] -> [2, 768] (trace) becomes [2, 384] -> [1, 768] (runtime)
    # NOTE: For proper seq_len handling, re-trace with symbolic_tracker that marks
    # seq_len as dynamic at SOURCE - not patched at runtime.
    if len(shape_list) == 2 and data.ndim == 2:
        trace_batch = shape_list[0]  # e.g., 2
        trace_features = shape_list[1]  # e.g., 768
        runtime_dim0 = data.shape[0]  # e.g., 2
        runtime_dim1 = data.shape[1]  # e.g., 384

        # Check if this is a merge pattern: dim1 doubles, dim0 halves
        if trace_features > 0 and runtime_dim1 > 0 and trace_features % runtime_dim1 == 0:
            k = trace_features // runtime_dim1  # merge factor (e.g., 2)
            if k > 1 and runtime_dim0 % k == 0:
                # This is a merge pattern
                runtime_batch = runtime_dim0 // k  # e.g., 2/2 = 1
                try:
                    return data.reshape(runtime_batch, trace_features)
                except RuntimeError:
                    pass  # Fall through

    # Case 3: Mismatch at dim 0 (batch) → scale dim 0 dynamically
    # STRICT: Only apply if:
    #   1. dim 0 is positive and small (typical batch sizes are 1-16)
    #   2. The ratio is a clean integer
    #   3. The ratio equals input's actual batch dimension (dim 0)
    # This prevents false positives on position embeddings and other large dim-0 tensors
    if len(shape_list) > 0 and shape_list[0] > 0 and shape_list[0] <= 16:
        non_batch_elements = static_elements // shape_list[0]
        if non_batch_elements > 0 and input_elements % non_batch_elements == 0:
            dynamic_batch = input_elements // non_batch_elements
            # Only apply if dynamic_batch matches input's dim 0 (actual batch)
            if data.ndim > 0 and dynamic_batch == data.shape[0]:
                dynamic_shape = [dynamic_batch] + list(shape_list[1:])
                try:
                    return data.reshape(dynamic_shape)
                except RuntimeError:
                    pass  # Fall through to sophisticated logic below

    # Get graph shapes for relative computation
    graph_input_shapes = attrs.get("input_shapes", [])
    graph_input_shape = graph_input_shapes[0] if graph_input_shapes else None
    graph_output_shape = shape_list

    # If no graph input shape, use simple reshape
    if not graph_input_shape or len(graph_input_shape) == 0:
        return _simple_reshape(data, shape_list, allowzero)

    data_shape = list(data.shape)
    data_numel = data.numel()

    # Calculate graph input numel
    graph_input_numel = 1
    for d in graph_input_shape:
        graph_input_numel *= d

    # If same numel, use graph output shape directly
    if data_numel == graph_input_numel:
        return data.reshape(graph_output_shape)

    # Different numel - compute relative shape
    # Strategy: Try to preserve aspect ratio for transformed dimensions
    
    # 1. Identify "fixed" vs "dynamic" dimensions
    # A dimension is fixed if it matches the input dimension at the same index
    # (and input index exists)
    
    final_shape = []
    dynamic_indices = []
    
    # Track consumed input volume to verify what's left
    # But input shape rank might differ.
    
    # Heuristic: Match dimensions from left to right.
    # If they match graph input & output, they are preserved.
    # Everything else is "the reshaped block".
    
    # Find prefix of preserved dimensions
    prefix_len = 0
    min_len = min(len(data_shape), len(graph_output_shape), len(graph_input_shape) if graph_input_shape else 0)
    
    for i in range(min_len):
        # If input dim == output dim, assume it's preserved
        # Check against graph shapes
        g_in = graph_input_shape[i] if graph_input_shape else -999
        g_out = graph_output_shape[i]
        
        if g_in == g_out and g_in == data_shape[i]:
             final_shape.append(g_out)
             prefix_len += 1
        else:
             break
             
    # The rest of the output shape is the "reshaped block"
    reshaped_block_graph = graph_output_shape[prefix_len:]
    
    # Calculate volume available for this block
    preserved_vol = 1
    for d in final_shape:
        preserved_vol *= d
        
    available_vol = data_numel // preserved_vol
    
    # Calculate graph volume for this block
    graph_block_vol = 1
    for d in reshaped_block_graph:
        if d != -1: 
             graph_block_vol *= d
             
    # If explicit -1 in graph block, just use it
    if -1 in reshaped_block_graph:
         final_shape.extend(reshaped_block_graph)
         return data.reshape(final_shape)
         
    # If volumes match (miracle), just copy
    if available_vol == graph_block_vol:
         final_shape.extend(reshaped_block_graph)
         return data.reshape(final_shape)
         
    # HOMOTHETIC SCALING REFINED
    # Identify which dims in the block are actually dynamic vs fixed (present in input)
    
    dynamic_dims_indices = []
    fixed_vol = 1
    
    # We check if dimensions in reshaped_block_graph exist in data_shape
    # To avoid double counting, we use a multiset approach (frequency count)
    from collections import Counter
    data_dims_counts = Counter(data_shape)
    
    # Reduce counts for dimensions already consumed by prefix
    for d in final_shape:
        if d in data_dims_counts and data_dims_counts[d] > 0:
            data_dims_counts[d] -= 1
            
    final_block = list(reshaped_block_graph)
    
    for i, d in enumerate(reshaped_block_graph):
        if d == -1:
            dynamic_dims_indices.append(i)
        elif d == 1:
            # Dimension 1 is ALWAYS Fixed (Batch or Singleton)
            # Never scale a 1.
            pass
        elif d > 1 and data_dims_counts[d] > 0:
            # Dimension exists in input -> Assume it's Fixed (Channel, etc.)
            data_dims_counts[d] -= 1
            fixed_vol *= d
        else:
            # Dimension not in input -> Dynamic (Spatial)
            dynamic_dims_indices.append(i)
            
    # Calculate volume available for dynamic dims
    dynamic_vol = available_vol // fixed_vol
    
    # Calculate graph volume for dynamic dims
    graph_dynamic_vol = 1
    for i in dynamic_dims_indices:
        d = reshaped_block_graph[i]
        if d != -1:
            graph_dynamic_vol *= d
            
    if not dynamic_dims_indices:
        # All fixed?
        final_shape.extend(final_block)
        return data.reshape(final_shape)

    # Scale ratio
    if graph_dynamic_vol > 0:
        scale_ratio = dynamic_vol / graph_dynamic_vol
    else:
        scale_ratio = 1 # Should not happen if logic is correct
        
    ndim_dynamic = len(dynamic_dims_indices)
    linear_scale = scale_ratio ** (1/ndim_dynamic) if ndim_dynamic > 0 else 1
    
    # Apply scale
    scale = round(linear_scale)

    # ZERO FALLBACK: scale cannot be 0 - would produce invalid shape
    if scale == 0:
        # This happens when runtime tensor has fewer elements than graph expects
        # Fall back to -1 inference instead of producing [batch, 0, ...]
        if len(dynamic_dims_indices) == 1:
            # Single dynamic dim - let PyTorch infer it
            final_block[dynamic_dims_indices[0]] = -1
            final_shape.extend(final_block)
            return data.reshape(final_shape)
        else:
            # Multiple dynamic dims - can't infer, raise explicit error
            raise RuntimeError(
                f"ZERO FALLBACK: Cannot reshape tensor.\n"
                f"  Input shape: {data_shape} (numel={data_numel})\n"
                f"  Target shape: {graph_output_shape}\n"
                f"  Graph input shape: {graph_input_shape}\n"
                f"  Scale ratio {scale_ratio:.4f} rounds to 0.\n"
                f"  This indicates a shape mismatch between trace and runtime."
            )

    # Robustness: If scale is 1, maybe it's just a reshape of dynamic dims without scaling?
    # e.g. 4096 -> 64x64.
    
    for i in range(len(final_block)):
        if i in dynamic_dims_indices:
            d = final_block[i]
            if d == -1:
                final_block[i] = -1 # Let PyTorch infer one -1
            else:
                final_block[i] = int(d * scale)
                
    final_shape.extend(final_block)
    
    # Last check: if we have multiple -1 (from logic error), fix it
    neg_count = sum(1 for x in final_shape if x == -1)
    if neg_count > 1:
        # Fallback: remove all but last -1, replace with graph dim * scale (best guess)
        # Or just fail. Let's try to be smart.
        # Actually, if we have homothety, we shouldn't have -1s left ideally.
        pass

    # print(f"[Reshape] Homothetic Result: {final_shape}")
    return data.reshape(final_shape)

    # OLD LOGIC REMOVED (Legacy relative shape)
    """
    neg_used = neg_pos >= 0

    for i, dim in enumerate(graph_output_shape):
    ...
    """


def _simple_reshape(data: torch.Tensor, shape_list: List[int], allowzero: int) -> torch.Tensor:
    """Simple reshape without graph-relative computation."""
    neg_count = sum(1 for d in shape_list if d == -1)
    if neg_count > 1:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::reshape has {neg_count} dimensions with -1.\n"
            f"Only one -1 dimension is allowed.\n"
            f"shape_list={shape_list}"
        )

    final_shape = []
    for i, dim in enumerate(shape_list):
        if dim == 0 and not allowzero:
            if i < data.ndim:
                final_shape.append(data.shape[i])
            else:
                final_shape.append(0)
        else:
            final_shape.append(dim)

    return data.reshape(final_shape)


def _unsqueeze(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen unsqueeze - add dimension of size 1.
    aten::unsqueeze(Tensor self, int dim) -> Tensor
    """
    data = inputs[0]

    # Get axes from input (v13+) or attrs (v11-)
    if len(inputs) > 1:
        axes = _to_int_list(inputs[1])
    else:
        axes = attrs.get("axes", [0])

    if isinstance(axes, int):
        axes = [axes]

    # Output rank = input rank + number of axes
    output_rank = data.ndim + len(axes)

    # Normalize negative axes to positive (relative to output)
    normalized_axes = []
    for ax in axes:
        if ax < 0:
            ax = output_rank + ax
        normalized_axes.append(ax)

    # Sort axes and apply unsqueeze in order
    result = data
    for ax in sorted(normalized_axes):
        result = result.unsqueeze(ax)

    return result


def _squeeze(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen squeeze - remove dimensions of size 1.
    aten::squeeze(Tensor self) -> Tensor
    aten::squeeze.dim(Tensor self, int dim) -> Tensor
    """
    data = inputs[0]

    # Get axes from input (v13+) or attrs (v11-)
    if len(inputs) > 1:
        axes = _to_int_list(inputs[1])
    elif "axes" in attrs and attrs["axes"] is not None:
        axes = attrs["axes"]
        if isinstance(axes, int):
            axes = [axes]
    else:
        # No axes specified: squeeze all dims of size 1
        return data.squeeze()

    # Normalize negative axes
    axes = [ax if ax >= 0 else data.ndim + ax for ax in axes]

    # Sort in reverse to squeeze from back (avoid index shifting)
    result = data
    for ax in sorted(axes, reverse=True):
        if result.shape[ax] == 1:
            result = result.squeeze(ax)

    return result


def _transpose(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen permute/transpose - permute tensor dimensions.
    aten::permute(Tensor self, int[] dims) -> Tensor
    aten::transpose.int(Tensor self, int dim0, int dim1) -> Tensor
    aten::t(Tensor self) -> Tensor  # 2D matrix transpose (no args)
    """
    data = inputs[0]

    # Check for 2-arg transpose (dim0, dim1) first
    dim0 = attrs.get("dim0")
    dim1 = attrs.get("dim1")
    if dim0 is not None and dim1 is not None:
        return data.transpose(int(dim0), int(dim1))

    # Check for permute dims
    perm = attrs.get("perm") or attrs.get("dims")
    if perm is not None:
        return data.permute(perm)

    # Handle aten::t (2D matrix transpose) - no arguments needed
    # aten::t always swaps dims 0 and 1 for 2D tensors
    if data.ndim == 2:
        return data.t()

    # For higher-dim tensors with no args, reverse all dims (like numpy.T)
    if data.ndim > 2:
        return data.permute(list(range(data.ndim))[::-1])

    # 1D tensor - transpose is identity
    return data


def _split(inputs: List, attrs: Dict[str, Any]) -> List[torch.Tensor]:
    """
    ATen split - split tensor into chunks.
    aten::split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]
    aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
    """
    data = inputs[0]
    
    # 1. Resolve dimension
    dim = attrs.get("dim", attrs.get("axis", 0))
    if len(inputs) > 2:
        # ATen: split(tensor, split_size, dim)
        dim = int(inputs[2])

    # Handle negative dim
    if dim < 0:
        dim = data.ndim + dim

    # 2. Resolve split size / sizes
    if len(inputs) > 1:
        split_val = inputs[1]
        if isinstance(split_val, (list, tuple, torch.Tensor)):
            # split_with_sizes
            sizes = _to_int_list(split_val)
            return list(torch.split(data, sizes, dim=dim))
        else:
            # split_size (int)
            return list(torch.split(data, int(split_val), dim=dim))

    # 3. Fallback to attributes
    split_size = attrs.get("split_size", attrs.get("split"))
    if split_size is not None:
        if isinstance(split_size, list):
            return list(torch.split(data, split_size, dim=dim))
        return list(torch.split(data, int(split_size), dim=dim))

    # 4. Chunk fallback
    num_chunks = attrs.get("num_outputs", attrs.get("chunks", 2))
    return list(torch.chunk(data, int(num_chunks), dim=dim))


def _expand(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen expand - expand tensor to target shape.

    aten::expand(Tensor self, int[] size, *, bool implicit=False) -> Tensor

    RELATIVE EXPANSION:
    The graph captures absolute shapes during tracing. At runtime, we need to
    adapt based on actual input shapes. Strategy:
    - Compare graph_input_shape vs graph_output_shape
    - If dimension unchanged: use data.shape[i] (dynamic)
    - If dimension changed: use graph_output_shape[i] (static expand)
    - Special: -1 means "keep current size"

    Note: In ATen, -1 in size means "keep existing dimension size"
    """
    data = inputs[0]

    # Get target shape from graph
    target_shape: list[int]
    if len(inputs) > 1:
        shape_input = inputs[1]
        if isinstance(shape_input, torch.Tensor):
            target_shape = shape_input.tolist()
        elif isinstance(shape_input, (list, tuple)):
            # Type narrowing for pyright - shape_input is list or tuple here
            target_shape = [int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in shape_input]  # type: ignore[misc]
        else:
            raise RuntimeError(f"Invalid target_shape type: {type(shape_input)}")
    elif "size" in attrs:
        target_shape = attrs["size"]
    elif "shape" in attrs:
        target_shape = attrs["shape"]
    else:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::expand missing target shape.\n"
            f"inputs count: {len(inputs)}\n"
            f"attrs keys: {list(attrs.keys())}"
        )

    # Get graph input shape for relative expansion
    graph_input_shape = attrs.get("input_shapes", [[]])[0] if "input_shapes" in attrs else None

    data_shape = list(data.shape)
    final_shape = []

    # Align shapes from the right (broadcasting)
    data_idx = len(data_shape) - 1
    target_idx = len(target_shape) - 1
    graph_input_idx = len(graph_input_shape) - 1 if graph_input_shape else -1

    while target_idx >= 0:
        target_dim = target_shape[target_idx]
        data_dim = data_shape[data_idx] if data_idx >= 0 else 1
        graph_input_dim = graph_input_shape[graph_input_idx] if graph_input_idx >= 0 and graph_input_shape else None

        if target_dim == -1:
            # -1 means keep current size
            final_shape.insert(0, data_dim)
        elif graph_input_dim is not None and target_dim == graph_input_dim:
            # Same as graph input → use runtime data shape (dynamic)
            final_shape.insert(0, data_dim)
        elif data_dim == 1:
            # Singleton dim → expand to target (this is a real expand)
            final_shape.insert(0, target_dim)
        elif target_dim == data_dim:
            # Same size → keep it
            final_shape.insert(0, data_dim)
        else:
            # Mismatch - try to use data_dim if target was graph-specific
            # This handles cases where graph was traced with different batch size
            final_shape.insert(0, data_dim)

        target_idx -= 1
        data_idx -= 1
        graph_input_idx -= 1

    # Try expand
    try:
        return data.expand(final_shape)
    except RuntimeError as e:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::expand invalid broadcast.\n"
            f"data.shape={list(data.shape)}\n"
            f"target={final_shape}\n"
            f"graph_target={target_shape}\n"
            f"graph_input_shape={graph_input_shape}"
        ) from e


def _flatten(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen flatten - flatten tensor to 2D.
    aten::flatten(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
    """
    data = inputs[0]
    axis = attrs.get("axis", 1)

    # Handle negative axis
    if axis < 0:
        axis = data.ndim + axis

    # Handle edge case: axis == 0 means first dim is 1
    if axis == 0:
        return data.reshape(1, -1)

    # Calculate output shape
    first_dim = 1
    for i in range(axis):
        first_dim *= data.shape[i]

    return data.reshape(first_dim, -1)


# ==============================================================================
# STANDARD OPERATIONS
# ==============================================================================

def _constant(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen constant - Create tensor from attribute value.

    ZERO FALLBACK: value is required. No implicit zeros.
    """
    value = attrs.get("value")
    device = attrs.get("device", 'cpu')

    # ZERO FALLBACK: value must be provided
    if value is None:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::constant missing 'value' attribute.\n"
            f"attrs keys: {list(attrs.keys())}\n"
            f"Tracer must provide a valid constant value."
        )

    if isinstance(value, dict) and 'data' in value:
        data = value['data']
        shape = value.get('shape', [])
        dtype_map = {
            'float32': torch.float32, 'float16': torch.float16,
            'int64': torch.int64, 'int32': torch.int32,
            'bool': torch.bool
        }
        dtype_str = value.get('dtype')
        assert isinstance(dtype_str, str) or dtype_str is None, f"dtype must be str or None, got {type(dtype_str)}"
        if dtype_str is not None:
            dtype = dtype_map.get(dtype_str, torch.float32)
        else:
            dtype = torch.float32
        t = torch.tensor(data, dtype=dtype, device=device)
        return t.reshape(shape) if shape else t

    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, device=device)


def _scalar_tensor(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen scalar_tensor - Create tensor from scalar value.

    Unlike _constant, the value comes from inputs[0] (resolved from args[0]).
    Dtype and device come from kwargs.
    """
    # Get scalar value from inputs[0]
    if not inputs:
        raise RuntimeError("ZERO FALLBACK: aten::scalar_tensor requires a scalar input")

    scalar_val = inputs[0]

    # Get dtype from kwargs
    kwargs = attrs.get("kwargs", {})
    dtype_info = kwargs.get("dtype", {})
    if isinstance(dtype_info, dict):
        dtype_str = dtype_info.get("value", "torch.float32")
    else:
        dtype_str = str(dtype_info) if dtype_info else "torch.float32"

    # Remove torch. prefix if present
    dtype_str = dtype_str.replace("torch.", "")
    dtype_map = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16, "float64": torch.float64,
        "int64": torch.int64, "int32": torch.int32,
        "int16": torch.int16, "int8": torch.int8,
        "bool": torch.bool,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Get device from attrs
    device = attrs.get("_device", "cuda:0")

    return torch.tensor(scalar_val, dtype=dtype, device=device)


def _constant_of_shape(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen full/zeros/ones - Create tensor filled with value."""
    if inputs:
        shape = _to_int_list(inputs[0])
    else:
        shape = _flatten_shape(attrs.get("shape", [1]))

    val = attrs.get("value", 0.0)
    dtype_str = "float32"

    if isinstance(val, dict):
        dtype_str = val.get("dtype", "float32") if isinstance(val.get("value"), dict) else "float32"
        val = val.get("value", 0.0)

    if isinstance(val, (list, tuple)):
        val = val[0]

    # Determine dtype
    dtype_map = {'float32': torch.float32, 'float16': torch.float16,
                 'int64': torch.int64, 'int32': torch.int32}
    dtype = dtype_map.get(dtype_str, torch.float32)

    # At this point val should be numeric
    assert not isinstance(val, dict), f"val should not be dict at this point: {val}"
    assert val is not None, "val must not be None at this point"
    return torch.full(tuple(shape), float(val), dtype=dtype)


def _arange(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::arange - create sequential tensor."""

    # Extract from attrs
    end = attrs.get("end", None)
    start = attrs.get("start", 0)
    step = attrs.get("step", 1)
    dtype_str = attrs.get("dtype", None)
    device = attrs.get("_device", "cuda")  # Default to CUDA for Triton ops

    # Fallback: deduce end from output_shape
    if end is None:
        output_shape = attrs.get("_output_shape", [])
        if output_shape and len(output_shape) == 1:
            end = output_shape[0]

    if end is None:
        raise RuntimeError(
            f"ZERO FALLBACK: aten::arange missing 'end' parameter.\n"
            f"attrs={attrs}"
        )

    # Default dtype for indices
    dtype = torch.int64
    if dtype_str:
        dtype_map = {
            "int64": torch.int64, "int32": torch.int32,
            "float32": torch.float32, "float16": torch.float16,
        }
        dtype = dtype_map.get(dtype_str, torch.int64)

    return torch.arange(start, end, step, dtype=dtype, device=device)


def _size(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen numel - Return total number of elements."""
    return torch.tensor(inputs[0].numel(), dtype=torch.int64, device='cpu')


def _tile(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen tile/repeat - Repeat tensor along each dimension."""
    repeats = _to_int_list(inputs[1]) if len(inputs) > 1 else attrs.get("repeats")
    assert repeats is not None, "repeats must be provided"
    if not isinstance(repeats, list):
        repeats = list(repeats) if hasattr(repeats, '__iter__') else [repeats]
    return inputs[0].tile(repeats)


# REMOVED: _masked_fill
# ZERO FALLBACK: masked_fill is COMPUTE (element-wise conditional) - must use Triton kernel


def _clone(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen clone - Clone tensor (memory copy).
    CRITICAL: Ensures contiguity for subsequent views.
    """
    # memory_format is optional arg 1
    # We generally want to preserve format or force contiguous
    return torch.clone(inputs[0])


def _copy(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen copy_ - Copy elements from src to self."""
    # args: self, src, non_blocking
    dest = inputs[0]
    src = inputs[1]
    non_blocking = False
    # Check inputs/attrs for non_blocking
    if len(inputs) > 2:
        val = inputs[2]
        non_blocking = bool(val.item()) if isinstance(val, torch.Tensor) else bool(val)
    elif "non_blocking" in attrs:
        non_blocking = bool(attrs["non_blocking"])
        
    return dest.copy_(src, non_blocking=non_blocking)


def _identity(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen identity - Return input unchanged."""
    return inputs[0]


def _dropout(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen dropout - Apply dropout with training mode support.

    ZERO FALLBACK: Respects training flag for training/fine-tuning.
    """
    x = inputs[0]
    p = float(attrs.get("p", 0.5))
    training = bool(attrs.get("training", False))

    if training:
        return F.dropout(x, p=p, training=True)
    return x


def _cast(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen to/_to_copy - Convert tensor to specified type."""
    # Check for ATen kwargs.dtype format first (from _to_copy)
    kwargs = attrs.get("kwargs", {})
    if "dtype" in kwargs:
        dtype_info = kwargs["dtype"]
        if isinstance(dtype_info, dict) and dtype_info.get("type") == "dtype":
            dtype_str = dtype_info.get("value", "torch.float32")
            # Map string dtype to torch dtype
            str_dtype_map = {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.bool": torch.bool,
                "torch.float64": torch.float64,
                "torch.uint8": torch.uint8,
                "torch.int8": torch.int8,
            }
            target = str_dtype_map.get(dtype_str, torch.float32)
            return inputs[0].to(target)

    # Fallback to ONNX-style integer codes
    to = attrs.get("to")
    dtype_map = {
        1: torch.float32, 10: torch.float16, 16: torch.bfloat16,
        6: torch.int32, 7: torch.int64, 9: torch.bool,
        11: torch.float64, 2: torch.uint8, 3: torch.int8,
    }
    assert isinstance(to, int), f"to parameter must be int, got {type(to)}"
    target = dtype_map.get(to, torch.float32)
    return inputs[0].to(target)


# ==============================================================================
# CONVOLUTIONS
# ==============================================================================

# REMOVED: _conv2d, _convolution
# ZERO FALLBACK: Convolutions are COMPUTE ops - must go through Triton kernels
# These were incorrectly placed in metadata_ops (compute using F.conv2d)

# ==============================================================================
# MISC OPERATIONS
# ==============================================================================

# REMOVED: _resize (upsample), _one_hot, _triu, _tril, _pad
# ZERO FALLBACK: These are COMPUTE ops - must go through Triton kernels
# - _resize: Uses F.interpolate (compute)
# - _one_hot: Uses F.one_hot (compute)
# - _triu/_tril: Uses torch.triu/tril (compute - element-wise)
# - _pad: Uses F.pad (compute)

def _nonzero(inputs, attrs):
    """ATen nonzero - Return indices of non-zero elements (metadata: index lookup)."""
    return torch.nonzero(inputs[0], as_tuple=False).T

def _unique(inputs, attrs):
    """ATen unique - Return unique elements (metadata: deduplication)."""
    return torch.unique(inputs[0])


# =============================================================================
# CREATION OPS - DEDICATED HANDLERS
# =============================================================================

def _zeros(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::zeros - Create zero tensor with specified shape."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if not args:
        raise RuntimeError(f"aten::zeros missing shape. attrs={attrs}")

    shape = _parse_shape(args[0])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    return torch.zeros(shape, device=device, dtype=dtype)


def _ones(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::ones - Create ones tensor with specified shape."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if not args:
        raise RuntimeError(f"aten::ones missing shape. attrs={attrs}")

    shape = _parse_shape(args[0])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    return torch.ones(shape, device=device, dtype=dtype)


def _full(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::full - Create tensor filled with value."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if len(args) < 2:
        raise RuntimeError(f"aten::full missing shape or value. attrs={attrs}")

    shape = _parse_shape(args[0])
    value = _parse_scalar(args[1])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    return torch.full(shape, value, device=device, dtype=dtype)


def _empty(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::empty - Create uninitialized tensor with specified shape."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if not args:
        raise RuntimeError(f"aten::empty missing shape. attrs={attrs}")

    shape = _parse_shape(args[0])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    return torch.empty(shape, device=device, dtype=dtype)


def _zeros_like(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::zeros_like - Create zero tensor with same shape as input."""
    if not inputs:
        raise RuntimeError("aten::zeros_like requires input tensor")

    kwargs = attrs.get("kwargs", {})
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    # If no overrides, follow base tensor
    if device is None and dtype is None:
        return torch.zeros_like(inputs[0])
    return torch.zeros_like(inputs[0], device=device, dtype=dtype)


def _ones_like(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::ones_like - Create ones tensor with same shape as input."""
    if not inputs:
        raise RuntimeError("aten::ones_like requires input tensor")

    kwargs = attrs.get("kwargs", {})
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    if device is None and dtype is None:
        return torch.ones_like(inputs[0])
    return torch.ones_like(inputs[0], device=device, dtype=dtype)


def _full_like(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::full_like - Create tensor with same shape as input, filled with value."""
    if not inputs:
        raise RuntimeError("aten::full_like requires input tensor")

    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    # args[0] is the input tensor (already in inputs), args[1] is the fill value
    if len(args) < 2:
        raise RuntimeError(f"aten::full_like missing fill value. attrs={attrs}")

    value = _parse_scalar(args[1])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    if device is None and dtype is None:
        return torch.full_like(inputs[0], value)
    return torch.full_like(inputs[0], value, device=device, dtype=dtype)


def _empty_like(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::empty_like - Create uninitialized tensor with same shape as input."""
    if not inputs:
        raise RuntimeError("aten::empty_like requires input tensor")

    kwargs = attrs.get("kwargs", {})
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    if device is None and dtype is None:
        return torch.empty_like(inputs[0])
    return torch.empty_like(inputs[0], device=device, dtype=dtype)


def _new_arange(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::arange - Create range tensor."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if not args:
        raise RuntimeError(f"aten::arange missing arguments. attrs={attrs}")

    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    if len(args) == 1:
        end = _parse_scalar(args[0])
        return torch.arange(end, dtype=dtype, device=device)
    elif len(args) == 2:
        start = _parse_scalar(args[0])
        end = _parse_scalar(args[1])
        return torch.arange(start, end, dtype=dtype, device=device)
    elif len(args) >= 3:
        start = _parse_scalar(args[0])
        end = _parse_scalar(args[1])
        step = _parse_scalar(args[2])
        return torch.arange(start, end, step, dtype=dtype, device=device)
    else:
        raise RuntimeError(f"aten::arange invalid args count: {len(args)}")


def _linspace(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """aten::linspace - Create linearly spaced tensor."""
    args = attrs.get("args", [])
    kwargs = attrs.get("kwargs", {})

    if len(args) < 3:
        raise RuntimeError(f"aten::linspace requires start, end, steps. attrs={attrs}")

    start = _parse_scalar(args[0])
    end = _parse_scalar(args[1])
    steps = _parse_scalar(args[2])
    device = _parse_device(kwargs)
    dtype = _parse_dtype(kwargs)

    return torch.linspace(start, end, steps, dtype=dtype, device=device)


# ==============================================================================
# COMPLEX OPS - PyTorch Native (Triton kernels are unreliable for these)
# ==============================================================================

def _scaled_dot_product_attention(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """
    ATen scaled_dot_product_attention - Uses PyTorch's highly optimized SDPA.
    PyTorch routes to Flash Attention / cuDNN automatically.
    """
    q = inputs[0].contiguous()
    k = inputs[1].contiguous()
    v = inputs[2].contiguous()

    # Optional mask
    attn_mask = inputs[3] if len(inputs) > 3 and inputs[3] is not None else None
    is_causal = inputs[4] if len(inputs) > 4 else False
    dropout_p = inputs[5] if len(inputs) > 5 else 0.0
    scale = inputs[6] if len(inputs) > 6 else None

    # Convert mask to float if needed
    if attn_mask is not None:
        attn_mask = attn_mask.contiguous()
        if attn_mask.dtype in (torch.int64, torch.int32):
            # Check if additive format (values like -10000)
            if attn_mask.min().item() < -1:
                attn_mask = attn_mask.to(q.dtype)
            else:
                # Binary: 1=attend, 0=mask -> additive: 0=attend, -inf=mask
                mask_float = torch.zeros_like(attn_mask, dtype=q.dtype)
                mask_float.masked_fill_(attn_mask == 0, float("-inf"))
                attn_mask = mask_float
        elif attn_mask.dtype == torch.bool:
            mask_float = torch.zeros(attn_mask.shape, dtype=q.dtype, device=attn_mask.device)
            mask_float.masked_fill_(~attn_mask, float("-inf"))
            attn_mask = mask_float
        else:
            attn_mask = attn_mask.to(q.dtype)

    # Convert scale to float if it's a tensor
    scale_val: float | None = None
    if scale is not None:
        if isinstance(scale, torch.Tensor):
            scale_val = float(scale.item())
        else:
            scale_val = float(scale)

    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p) if isinstance(dropout_p, (int, float)) else 0.0,
        is_causal=bool(is_causal) if isinstance(is_causal, bool) else False,
        scale=scale_val
    )
    return output


def _scaled_dot_product_attention_4outputs(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> Tuple:
    """
    ATen efficient/flash attention variants - return 4 outputs.
    """
    output = _scaled_dot_product_attention(inputs, attrs)
    q = inputs[0]
    batch, heads, seq_q = q.shape[:3]

    # Dummy outputs for logsumexp, philox_seed, philox_offset
    lse = torch.zeros((batch, heads, seq_q), device=q.device, dtype=torch.float32)
    philox_seed = torch.tensor(0, device=q.device, dtype=torch.int64)
    philox_offset = torch.tensor(0, device=q.device, dtype=torch.int64)

    return output, lse, philox_seed, philox_offset


def _native_layer_norm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> Tuple:
    """ATen native_layer_norm - returns (output, mean, rstd)."""
    x = inputs[0]
    normalized_shape = inputs[1] if len(inputs) > 1 else attrs.get("normalized_shape", [x.shape[-1]])
    weight = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    bias = inputs[3] if len(inputs) > 3 and inputs[3] is not None else None
    eps = inputs[4] if len(inputs) > 4 else attrs.get("eps", 1e-5)

    if isinstance(normalized_shape, torch.Tensor):
        normalized_shape = normalized_shape.tolist()

    output = F.layer_norm(x, normalized_shape, weight, bias, float(eps))

    # Compute mean and rstd for the normalized dimensions
    dims = list(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + float(eps))

    return output, mean.squeeze(), rstd.squeeze()


def _native_group_norm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> Tuple:
    """ATen native_group_norm - returns (output, mean, rstd)."""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    N = inputs[3] if len(inputs) > 3 else x.shape[0]
    C = inputs[4] if len(inputs) > 4 else x.shape[1]
    HxW = inputs[5] if len(inputs) > 5 else x.numel() // (x.shape[0] * x.shape[1])
    num_groups = inputs[6] if len(inputs) > 6 else attrs.get("num_groups", 32)
    eps = inputs[7] if len(inputs) > 7 else attrs.get("eps", 1e-5)

    # Ensure contiguous for F.group_norm and reshaping
    x = x.contiguous()
    output = F.group_norm(x, int(num_groups), weight, bias, float(eps))

    # Compute mean and rstd per group
    batch = x.shape[0]
    x_reshaped = x.reshape(batch, int(num_groups), -1)
    mean = x_reshaped.mean(dim=2)
    var = x_reshaped.var(dim=2, unbiased=False)
    rstd = torch.rsqrt(var + float(eps))

    return output, mean, rstd


def _layer_norm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen layer_norm - single output."""
    result = _native_layer_norm(inputs, attrs)
    return result[0] if isinstance(result, tuple) else result


def _group_norm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen group_norm - single output."""
    result = _native_group_norm(inputs, attrs)
    return result[0] if isinstance(result, tuple) else result


def _convolution(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen convolution - uses PyTorch's cuDNN backend."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    stride = inputs[3] if len(inputs) > 3 else attrs.get("stride", [1, 1])
    padding = inputs[4] if len(inputs) > 4 else attrs.get("padding", [0, 0])
    dilation = inputs[5] if len(inputs) > 5 else attrs.get("dilation", [1, 1])
    transposed = inputs[6] if len(inputs) > 6 else attrs.get("transposed", False)
    output_padding = inputs[7] if len(inputs) > 7 else attrs.get("output_padding", [0, 0])
    groups = inputs[8] if len(inputs) > 8 else attrs.get("groups", 1)

    # Convert to tuples if needed
    def to_tuple(v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return (int(v), int(v))

    stride = to_tuple(stride)
    padding = to_tuple(padding)
    dilation = to_tuple(dilation)
    output_padding = to_tuple(output_padding)

    return torch.conv2d(x, weight, bias, stride, padding, dilation, int(groups))


def _conv2d(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen conv2d."""
    return _convolution(inputs, attrs)


def _pad(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen pad/constant_pad_nd."""
    x = inputs[0]
    pad = inputs[1] if len(inputs) > 1 else attrs.get("pad", [0, 0])
    value = inputs[2] if len(inputs) > 2 else attrs.get("value", 0.0)

    if isinstance(pad, torch.Tensor):
        pad = pad.tolist()

    return F.pad(x, pad, mode="constant", value=float(value) if value is not None else 0.0)


def _upsample_nearest2d(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen upsample_nearest2d."""
    x = inputs[0]
    output_size = inputs[1] if len(inputs) > 1 else attrs.get("output_size")
    scale_factors = inputs[2] if len(inputs) > 2 else attrs.get("scale_factor")

    if output_size is not None:
        if isinstance(output_size, torch.Tensor):
            output_size = output_size.tolist()
        return F.interpolate(x, size=output_size, mode="nearest")
    elif scale_factors is not None:
        if isinstance(scale_factors, torch.Tensor):
            scale_factors = scale_factors.tolist()
        return F.interpolate(x, scale_factor=scale_factors, mode="nearest")
    else:
        return x


def _softmax(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen softmax."""
    x = inputs[0]
    dim = inputs[1] if len(inputs) > 1 else attrs.get("dim", -1)
    return F.softmax(x, dim=int(dim))


def _gelu(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen gelu."""
    x = inputs[0]
    approximate = attrs.get("approximate", "none")
    return F.gelu(x, approximate=approximate)


def _silu(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen silu."""
    return F.silu(inputs[0])


def _where(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen where - conditional select."""
    tensors = [inputs[0], inputs[1], inputs[2]]
    target_device = inputs[0].device
    aligned = _ensure_same_device(tensors, target_device)
    return torch.where(aligned[0], aligned[1], aligned[2])


def _clamp(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen clamp."""
    x = inputs[0]
    min_val = inputs[1] if len(inputs) > 1 else attrs.get("min", None)
    max_val = inputs[2] if len(inputs) > 2 else attrs.get("max", None)
    return torch.clamp(x, min=min_val, max=max_val)


def _eq(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen eq - equality comparison."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.eq(aligned[0], aligned[1])


def _ne(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen ne - not equal."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.ne(aligned[0], aligned[1])


def _lt(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen lt - less than."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.lt(aligned[0], aligned[1])


def _le(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen le - less equal."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.le(aligned[0], aligned[1])


def _gt(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen gt - greater than."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.gt(aligned[0], aligned[1])


def _ge(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen ge - greater equal."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.ge(aligned[0], aligned[1])


def _masked_fill(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen masked_fill."""
    x = inputs[0]
    mask = inputs[1]
    value = inputs[2] if len(inputs) > 2 else attrs.get("value", 0.0)
    if isinstance(value, torch.Tensor):
        value = value.item()
    return x.masked_fill(mask, value)


def _mul(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen mul."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.mul(aligned[0], aligned[1])


def _add(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen add."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    alpha = attrs.get("alpha", 1.0)
    if alpha != 1.0:
        return torch.add(aligned[0], aligned[1], alpha=alpha)
    return torch.add(aligned[0], aligned[1])


def _sub(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen sub."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    alpha = attrs.get("alpha", 1.0)
    if alpha != 1.0:
        return torch.sub(aligned[0], aligned[1], alpha=alpha)
    return torch.sub(aligned[0], aligned[1])


def _div(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen div."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.div(aligned[0], aligned[1])


def _pow(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen pow."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.pow(aligned[0], aligned[1])


def _rsub(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen rsub - reverse subtract."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    a, b = aligned[0], aligned[1]
    alpha = attrs.get("alpha", 1.0)
    # rsub(a, b, alpha) = b - alpha * a
    if alpha != 1.0:
        return b - alpha * a
    return b - a


def _neg(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen neg."""
    return torch.neg(inputs[0])


def _exp(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen exp."""
    return torch.exp(inputs[0])


def _log(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen log."""
    return torch.log(inputs[0])


def _sqrt(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen sqrt."""
    return torch.sqrt(inputs[0])


def _rsqrt(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen rsqrt."""
    return torch.rsqrt(inputs[0])


def _sin(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen sin."""
    return torch.sin(inputs[0])


def _cos(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen cos."""
    return torch.cos(inputs[0])


def _tanh(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen tanh."""
    return torch.tanh(inputs[0])


def _sigmoid(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen sigmoid."""
    return torch.sigmoid(inputs[0])


def _relu(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen relu."""
    return F.relu(inputs[0])


def _erf(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen erf."""
    return torch.erf(inputs[0])


def _abs(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen abs."""
    return torch.abs(inputs[0])


def _isinf(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen isinf."""
    return torch.isinf(inputs[0])


def _isnan(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen isnan."""
    return torch.isnan(inputs[0])


def _isfinite(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen isfinite."""
    return torch.isfinite(inputs[0])


def _maximum(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen maximum."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.maximum(aligned[0], aligned[1])


def _minimum(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen minimum."""
    target_device = inputs[0].device
    aligned = _ensure_same_device([inputs[0], inputs[1]], target_device)
    return torch.minimum(aligned[0], aligned[1])


def _reciprocal(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen reciprocal."""
    return torch.reciprocal(inputs[0])


def _sum(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen sum."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.sum(x, dim=dim, keepdim=keepdim)
    return torch.sum(x)


def _mean(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen mean."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.mean(x, dim=dim, keepdim=keepdim)
    return torch.mean(x)


def _max(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen max."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        result = torch.max(x, dim=dim, keepdim=keepdim)
        return result.values if hasattr(result, 'values') else result  # type: ignore[return-value]
    return torch.max(x)


def _min(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen min."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        result = torch.min(x, dim=dim, keepdim=keepdim)
        return result.values if hasattr(result, 'values') else result  # type: ignore[return-value]
    return torch.min(x)


def _prod(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen prod."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.prod(x, dim=dim, keepdim=keepdim)
    return torch.prod(x)


def _var(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen var."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    unbiased = attrs.get("unbiased", True)
    if dim is not None:
        return torch.var(x, dim=dim, keepdim=keepdim, unbiased=unbiased)
    return torch.var(x, unbiased=unbiased)


def _std(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen std."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    unbiased = attrs.get("unbiased", True)
    if dim is not None:
        return torch.std(x, dim=dim, keepdim=keepdim, unbiased=unbiased)
    return torch.std(x, unbiased=unbiased)


def _argmax(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen argmax."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.argmax(x, dim=dim, keepdim=keepdim)
    return torch.argmax(x)


def _argmin(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen argmin."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.argmin(x, dim=dim, keepdim=keepdim)
    return torch.argmin(x)


def _any(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen any."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.any(x, dim=dim, keepdim=keepdim)
    return torch.any(x)


def _all(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen all."""
    x = inputs[0]
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return torch.all(x, dim=dim, keepdim=keepdim)
    return torch.all(x)


def _cumsum(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen cumsum."""
    x = inputs[0]
    dim = inputs[1] if len(inputs) > 1 else attrs.get("dim", 0)
    return torch.cumsum(x, dim=int(dim))


def _cumprod(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen cumprod."""
    x = inputs[0]
    dim = inputs[1] if len(inputs) > 1 else attrs.get("dim", 0)
    return torch.cumprod(x, dim=int(dim))


def _embedding(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen embedding."""
    weight = inputs[0]
    indices = inputs[1]
    # Ensure indices on same device as weight
    if indices.device != weight.device:
        indices = indices.to(weight.device)
    return F.embedding(indices, weight)


def _bmm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen bmm - batched matrix multiplication."""
    return torch.bmm(inputs[0], inputs[1])


def _mm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen mm - matrix multiplication."""
    return torch.mm(inputs[0], inputs[1])


def _matmul(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen matmul."""
    return torch.matmul(inputs[0], inputs[1])


def _linear(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen linear."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    return F.linear(x, weight, bias)


def _addmm(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> torch.Tensor:
    """ATen addmm."""
    bias = inputs[0]
    mat1 = inputs[1]
    mat2 = inputs[2]
    beta = attrs.get("beta", 1.0)
    alpha = attrs.get("alpha", 1.0)
    return torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)


# ==============================================================================
# REGISTRY & DISPATCHER
# ==============================================================================

METADATA_OPS_IMPL: Dict[str, Callable] = {
    # === COMPLEX OPS - PyTorch Native (Triton unreliable) ===
    "aten::scaled_dot_product_attention": _scaled_dot_product_attention,
    "aten::_scaled_dot_product_attention": _scaled_dot_product_attention,
    "aten::_scaled_dot_product_efficient_attention": _scaled_dot_product_attention_4outputs,
    "aten::_scaled_dot_product_flash_attention": _scaled_dot_product_attention_4outputs,
    "aten::_scaled_dot_product_flash_attention_for_cpu": _scaled_dot_product_attention_4outputs,
    "aten::native_layer_norm": _native_layer_norm,
    "aten::layer_norm": _layer_norm,
    "aten::native_group_norm": _native_group_norm,
    "aten::group_norm": _group_norm,
    "aten::convolution": _convolution,
    "aten::conv2d": _conv2d,
    "aten::pad": _pad,
    "aten::constant_pad_nd": _pad,
    "aten::upsample_nearest2d": _upsample_nearest2d,
    "aten::softmax": _softmax,
    "aten::_softmax": _softmax,
    "aten::gelu": _gelu,
    "aten::silu": _silu,

    # === CREATION - dedicated handlers (ZERO FALLBACK) ===
    # NOTE: Element-wise, comparison, reduction, matmul ops are handled by
    # dynamic resolution (_resolve_op) - no need for manual wrappers
    "aten::zeros": _zeros,
    "aten::ones": _ones,
    "aten::full": _full,
    "aten::empty": _empty,
    "aten::zeros_like": _zeros_like,
    "aten::ones_like": _ones_like,
    "aten::full_like": _full_like,
    "aten::empty_like": _empty_like,
    "aten::arange": _new_arange,
    "aten::linspace": _linspace,
    # Keep _constant only for true scalar constants
    "aten::constant": _constant,
    "aten::scalar_tensor": _scalar_tensor,
    "aten::tensor": _constant,
    "aten::_tensor_constant_from_buffer": _constant,

    # === SHAPE & VIEWS ===
    "aten::view": _reshape,
    "aten::reshape": _reshape,
    "aten::_unsafe_view": _reshape,
    "aten::unsqueeze": _unsqueeze,
    "aten::squeeze": _squeeze,
    "aten::flatten": _flatten,
    "aten::expand": _expand,
    "aten::expand_as": _expand,
    "aten::repeat": _expand,
    "aten::permute": _transpose,
    "aten::transpose": _transpose,
    "aten::t": _transpose,
    "aten::contiguous": _identity,
    "aten::clone": _clone,
    "aten::copy": _copy,
    "aten::detach": _identity,
    "aten::alias": _identity,

    # === INDEXING & SLICING ===
    "aten::slice": _slice,
    "aten::narrow": _slice,
    "aten::select": _select,
    "aten::index_select": _gather,
    "aten::gather": _gather,
    "aten::split": _split,
    "aten::split_with_sizes": _split,
    "aten::chunk": _split,
    "aten::unbind": _split,

    # === CONCATENATION ===
    "aten::cat": _concat,
    "aten::concat": _concat,
    "aten::stack": _concat,

    # === CAST & TYPE ===
    "aten::to": _cast,
    "aten::type_as": _cast,
    "aten::float": _cast,
    "aten::half": _cast,
    "aten::bfloat16": _cast,
    "aten::int": _cast,
    "aten::long": _cast,
    "aten::bool": _cast,
    "aten::_to_copy": _cast,

    # REMOVED: pad, convolutions - COMPUTE ops go through Triton

    # === MISC (metadata only - no compute) ===
    "aten::size": _size,
    "aten::_shape_as_tensor": _shape,
    "aten::dropout": _dropout,  # Inference: identity (no compute)
    "aten::feature_dropout": _dropout,  # Inference: identity
    "aten::native_dropout": _dropout,  # Inference: identity
    "aten::nonzero": _nonzero,  # Index lookup
    "aten::unique": _unique,  # Deduplication
    "aten::tile": _tile,  # Memory layout

    # REMOVED: triu, tril, masked_fill, one_hot, upsample (all COMPUTE)
    # ZERO FALLBACK: Must use Triton kernels for compute
}


# ==============================================================================
# DYNAMIC OP RESOLUTION - Like sequential_dispatcher.py
# ==============================================================================

_OP_CACHE: Dict[str, Callable] = {}


def _resolve_op(op_name: str) -> Callable:
    """
    Dynamically find the PyTorch function.
    Priority: torch.ops.aten > torch > F
    """
    if op_name in _OP_CACHE:
        return _OP_CACHE[op_name]

    # Clean the name: "aten::add" -> "add", "aten::add.Tensor" -> "add"
    clean_name = op_name.replace("aten::", "").split(".")[0]

    # Priority 1: torch.ops.aten (most accurate for ATen ops)
    if hasattr(torch.ops.aten, clean_name):
        op = getattr(torch.ops.aten, clean_name)
        _OP_CACHE[op_name] = op
        return op

    # Priority 2: torch namespace
    if hasattr(torch, clean_name):
        op = getattr(torch, clean_name)
        _OP_CACHE[op_name] = op
        return op

    # Priority 3: torch.nn.functional
    if hasattr(F, clean_name):
        op = getattr(F, clean_name)
        _OP_CACHE[op_name] = op
        return op

    raise AttributeError(f"Op '{op_name}' not found in torch.ops.aten, torch, or F")


def _ensure_same_device(tensors: List[Any], target_device: torch.device) -> List[Any]:
    """Move all tensors to target device."""
    result = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device != target_device:
            result.append(t.to(target_device))
        else:
            result.append(t)
    return result


def execute_metadata_op(op_type: str, inputs: List[torch.Tensor], attrs: Dict[str, Any], outputs: List | None = None) -> Any:
    """
    Execute metadata op - ATen native format.

    1. Check custom handlers (for ops with special signatures)
    2. Fall back to dynamic resolution (torch.ops.aten / torch / F)
    """
    # Get target device from attrs or first tensor input
    target_device = attrs.get("_device")
    if target_device is None:
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                target_device = inp.device
                break

    # Ensure all tensor inputs are on same device
    if target_device is not None:
        inputs = _ensure_same_device(inputs, target_device)

    # 1. Check custom handlers first (for special signatures)
    if op_type in METADATA_OPS_IMPL:
        return METADATA_OPS_IMPL[op_type](inputs, attrs)

    # 2. Dynamic resolution - call PyTorch directly
    try:
        op_fn = _resolve_op(op_type)

        # Pass ALL inputs (tensors, scalars, etc.) - don't filter
        # Filter out None values only
        valid_inputs = [inp for inp in inputs if inp is not None]

        # Call the op directly with all valid inputs
        return op_fn(*valid_inputs)

    except Exception as e:
        raise RuntimeError(f"Failed to execute '{op_type}' via dynamic resolution: {e}")


# --- Compatibility ---
def is_implemented_metadata_op(op_type: str) -> bool:
    """Check if op_type can be executed (custom handler OR dynamic resolution)."""
    if op_type in METADATA_OPS_IMPL:
        return True
    # Try dynamic resolution
    try:
        _resolve_op(op_type)
        return True
    except AttributeError:
        return False

def list_implemented_ops() -> List[str]:
    """List all ops with custom handlers."""
    return list(METADATA_OPS_IMPL.keys())
