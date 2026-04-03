"""Triton Constants Loader — base64-encoded graph constants → NBXTensor.

Loads constant tensors (RoPE tables, position embeddings, etc.) embedded
in graph.json as base64 data. Uses numpy for deserialization, cudaMemcpy
for GPU transfer. Zero torch dependency.
"""

import base64
import io
import struct
from typing import Dict

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


# numpy dtype from safetensors/graph dtype strings
_DTYPE_MAP = {
    "float16": (np.float16, NBXDtype.float16),
    "bfloat16": (None, NBXDtype.bfloat16),  # numpy has no bfloat16
    "float32": (np.float32, NBXDtype.float32),
    "float64": (np.float64, NBXDtype.float64),
    "int8": (np.int8, NBXDtype.int8),
    "int16": (np.int16, NBXDtype.int16),
    "int32": (np.int32, NBXDtype.int32),
    "int64": (np.int64, NBXDtype.int64),
    "uint8": (np.uint8, NBXDtype.uint8),
    "bool": (np.bool_, NBXDtype.bool_),
}


def load_constants_from_graph(tensors: dict, device_idx: int = 0) -> Dict[str, NBXTensor]:
    """Load base64-encoded constant tensors from graph.json.

    Args:
        tensors: The "tensors" dict from graph.json
        device_idx: CUDA device for allocation

    Returns:
        Dict mapping weight_name → NBXTensor
    """
    DeviceAllocator.set_device(device_idx)
    constants = {}

    for tid, tdata in tensors.items():
        if not tdata.get("constant"):
            continue
        if tdata.get("is_computable"):
            continue

        b64_data = tdata.get("constant_data")
        if not b64_data:
            continue

        weight_name = tdata.get("weight_name")
        if not weight_name:
            continue

        dtype_str = tdata.get("dtype", "float32")
        shape = tdata.get("shape", [])

        # Decode base64 → raw bytes
        raw_bytes = base64.b64decode(b64_data)

        # Convert to numpy array
        np_dtype, nbx_dtype = _DTYPE_MAP.get(dtype_str, (np.float32, NBXDtype.float32))

        if np_dtype is None:
            # bfloat16: no numpy support, load as uint16 then treat as bfloat16
            arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
            nbx = NBXTensor.from_numpy(arr)
            # Override dtype to bfloat16
            nbx._dtype = NBXDtype.bfloat16
        else:
            arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
            nbx = NBXTensor.from_numpy(arr)

        constants[weight_name] = nbx

    return constants
