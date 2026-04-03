"""Triton Weight Loader — safetensors → NBXTensor via numpy + cudaMemcpy.

Zero torch dependency. Loads weights directly to GPU memory.
"""

import os
from typing import Dict

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


# safetensors dtype string → NBXDtype
_SF_DTYPE_MAP = {
    "F16": NBXDtype.float16,
    "BF16": NBXDtype.bfloat16,
    "F32": NBXDtype.float32,
    "F64": NBXDtype.float64,
    "I8": NBXDtype.int8,
    "I16": NBXDtype.int16,
    "I32": NBXDtype.int32,
    "I64": NBXDtype.int64,
    "U8": NBXDtype.uint8,
    "BOOL": NBXDtype.bool_,
}


def load_safetensors(path: str, device_idx: int = 0) -> Dict[str, NBXTensor]:
    """Load all tensors from a safetensors file directly to GPU.

    Uses safetensors' numpy API for header parsing and raw byte access,
    then cudaMemcpy H2D for GPU transfer. Zero torch dependency.

    Args:
        path: Path to .safetensors file
        device_idx: CUDA device index for allocation

    Returns:
        Dict mapping weight names to NBXTensor on GPU
    """
    from safetensors import safe_open

    DeviceAllocator.set_device(device_idx)
    weights = {}

    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            np_tensor = f.get_tensor(key)
            shape = tuple(np_tensor.shape)
            dtype_name = str(np_tensor.dtype)

            # Map numpy dtype → NBXDtype
            nbx_dt = _numpy_to_nbx(dtype_name)

            # Allocate GPU memory and copy H2D
            nbx = NBXTensor.from_numpy(np_tensor)
            weights[key] = nbx

    return weights


def _numpy_to_nbx(dtype_str: str) -> NBXDtype:
    """Map numpy dtype string to NBXDtype."""
    mapping = {
        "float16": NBXDtype.float16,
        "float32": NBXDtype.float32,
        "float64": NBXDtype.float64,
        "int8": NBXDtype.int8,
        "int16": NBXDtype.int16,
        "int32": NBXDtype.int32,
        "int64": NBXDtype.int64,
        "uint8": NBXDtype.uint8,
        "bool": NBXDtype.bool_,
    }
    return mapping.get(dtype_str, NBXDtype.float32)
