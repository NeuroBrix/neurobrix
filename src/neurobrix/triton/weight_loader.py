"""Triton Weight Loader — safetensors → NBXTensor via arena + cudaMemcpy.

Zero torch dependency. Uses ComponentArena for one-shot GPU allocation
per device (zero fragmentation). Loads weights via raw byte parsing.
"""

import json
import math
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from neurobrix.kernels.nbx_tensor import (
    NBXTensor, NBXDtype, DeviceAllocator, _contiguous_strides, dtype_size,
)
from .memory_pool import ComponentArena


# safetensors dtype → (numpy dtype for reading, NBXDtype, bytes per element)
_SF_DTYPE_INFO = {
    "F16":  (np.float16,  NBXDtype.float16,  2),
    "BF16": (None,        NBXDtype.bfloat16, 2),
    "F32":  (np.float32,  NBXDtype.float32,  4),
    "F64":  (np.float64,  NBXDtype.float64,  8),
    "I8":   (np.int8,     NBXDtype.int8,     1),
    "I16":  (np.int16,    NBXDtype.int16,    2),
    "I32":  (np.int32,    NBXDtype.int32,    4),
    "I64":  (np.int64,    NBXDtype.int64,    8),
    "U8":   (np.uint8,    NBXDtype.uint8,    1),
    "BOOL": (np.bool_,    NBXDtype.bool_,    1),
}


def load_component_weights(
    cache_path: str,
    component: str,
    device_idx: int,
    compute_dtype: NBXDtype = NBXDtype.float16,
    shard_map: Optional[Dict[str, str]] = None,
) -> Dict[str, NBXTensor]:
    """Load all weights for a component as NBXTensor. Zero torch.

    Uses ComponentArena: one cudaMalloc per device, sub-allocate inside.
    Zero fragmentation vs 18866 individual cudaMalloc calls.

    Args:
        cache_path: Model cache path
        component: Component name (e.g. "model")
        device_idx: Default GPU device
        compute_dtype: Target dtype from Prism (bf16↔fp16 remap)
        shard_map: weight_name → "cuda:N" from Prism strategy
    """
    comp_dir = Path(cache_path) / "components" / component
    weights_dir = comp_dir / "weights"

    if not weights_dir.exists():
        return {}

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)

    # Parse shard_map → per-weight device index
    weight_device: Dict[str, int] = {}
    if shard_map:
        for wname, dev_str in shard_map.items():
            if isinstance(dev_str, str) and ':' in dev_str:
                weight_device[wname] = int(dev_str.split(':')[-1])
            elif isinstance(dev_str, int):
                weight_device[wname] = dev_str

    # Phase 1: Scan all shards to compute total bytes per device
    shard_files = sorted(weights_dir.glob("*.safetensors"))
    dev_bytes: Dict[int, int] = {}
    shard_headers = []

    for shard_path in shard_files:
        header, data_offset = _read_header(str(shard_path))
        shard_headers.append((str(shard_path), header, data_offset))

        for key, info in header.items():
            if key == '__metadata__':
                continue
            target_dev = weight_device.get(key, device_idx)
            nbytes = _target_nbytes(info, compute_dtype)
            # Align each sub-allocation
            aligned = (nbytes + ComponentArena.ALIGNMENT - 1) & ~(ComponentArena.ALIGNMENT - 1)
            dev_bytes[target_dev] = dev_bytes.get(target_dev, 0) + aligned

    # Phase 2: Create one arena per device
    arenas: Dict[int, ComponentArena] = {}
    for dev, total in dev_bytes.items():
        arenas[dev] = ComponentArena(total, dev)

    # Phase 3: Load tensors into arenas
    weights: Dict[str, NBXTensor] = {}

    for shard_path, header, data_offset in shard_headers:
        _load_shard_into_arenas(
            shard_path, header, data_offset,
            weights, device_idx, compute_dtype,
            weight_device, arenas)

    # Store arenas on the dict so they stay alive (prevent GC of GPU memory)
    weights['_arenas'] = arenas  # type: ignore

    return weights


def _read_header(path: str):
    """Read safetensors header without loading data."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size
    return header, data_offset


def _target_nbytes(info: dict, compute_dtype: NBXDtype) -> int:
    """Compute target byte count for a tensor after dtype remap."""
    shape = info["shape"]
    sf_dtype = info["dtype"]
    dtype_info = _SF_DTYPE_INFO.get(sf_dtype)
    if dtype_info is None:
        return 0
    _, nbx_dtype, elem_size = dtype_info

    # bf16→fp16 or fp16→bf16 keeps same element size (2 bytes)
    numel = math.prod(shape) if shape else 1
    return numel * elem_size


def _bf16_to_fp16_inplace(ptr: int, numel: int, device_idx: int):
    """Run bf16→fp16 kernel in-place on a GPU buffer.

    Both bf16 and fp16 are 2 bytes/element, so the kernel reads uint16
    (bf16 raw bits) and writes fp16 to the same memory locations.

    Creates typed NBXTensor wrappers around the raw pointer for Triton.
    """
    import triton
    from neurobrix.kernels.ops.dtype_convert import bf16_to_fp16_kernel

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)

    # Wrap raw pointer as typed NBXTensor for Triton kernel.
    # int16 and uint16 are same 2-byte memory layout — Triton kernel
    # treats bits as unsigned via .to(tl.uint32) anyway.
    flat = (numel,)
    strides = _contiguous_strides(flat)
    src = NBXTensor(ptr, flat, strides, NBXDtype.int16, 'cuda',
                    owns_data=False, device_idx=device_idx)
    dst = NBXTensor(ptr, flat, strides, NBXDtype.float16, 'cuda',
                    owns_data=False, device_idx=device_idx)

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    bf16_to_fp16_kernel[grid](src, dst, numel, BLOCK=BLOCK, num_warps=4)


def _load_shard_into_arenas(
    path: str,
    header: dict,
    data_offset: int,
    weights: Dict[str, NBXTensor],
    device_idx: int,
    compute_dtype: NBXDtype,
    weight_device: Dict[str, int],
    arenas: Dict[int, ComponentArena],
) -> None:
    """Load tensors from one shard, sub-allocating from arenas."""
    with open(path, 'rb') as f:
        for key, info in header.items():
            if key == '__metadata__':
                continue

            sf_dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            nbytes = end - start

            dtype_info = _SF_DTYPE_INFO.get(sf_dtype)
            if dtype_info is None:
                raise RuntimeError(f"Unknown safetensors dtype: {sf_dtype}")

            np_dtype, nbx_dtype, elem_size = dtype_info

            # Read raw bytes from file
            f.seek(data_offset + start)
            raw = f.read(nbytes)

            # Determine target dtype after remap
            target_dtype = nbx_dtype
            if nbx_dtype == NBXDtype.bfloat16 and compute_dtype == NBXDtype.float16:
                target_dtype = NBXDtype.float16
            elif nbx_dtype == NBXDtype.float16 and compute_dtype == NBXDtype.bfloat16:
                target_dtype = NBXDtype.bfloat16

            target_dev = weight_device.get(key, device_idx)
            arena = arenas[target_dev]
            DeviceAllocator.set_device(target_dev)

            # GPU-accelerated bf16→fp16: H2D raw bytes, convert in-place on GPU.
            # bf16 and fp16 are both 2 bytes/element → same buffer, zero extra memory.
            if nbx_dtype == NBXDtype.bfloat16 and target_dtype == NBXDtype.float16:
                ptr = arena.alloc(nbytes)
                # H2D: raw bf16 bits → GPU as uint16
                arr_u16 = np.frombuffer(raw, dtype=np.uint16)
                DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
                # In-place GPU kernel: bf16 bits → fp16 (same 2-byte slots)
                _bf16_to_fp16_inplace(ptr, arr_u16.shape[0], target_dev)
                strides = _contiguous_strides(shape)
                nbx = NBXTensor(ptr, shape, strides, NBXDtype.float16, 'cuda',
                                owns_data=False, device_idx=target_dev)
                weights[key] = nbx
                continue

            # Standard path: numpy → H2D
            if nbx_dtype == NBXDtype.bfloat16:
                arr = np.ascontiguousarray(
                    np.frombuffer(raw, dtype=np.uint16).reshape(shape))
            else:
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                arr = np.ascontiguousarray(arr)
                if target_dtype != nbx_dtype:
                    fp32 = arr.astype(np.float32)
                    arr = np.ascontiguousarray(
                        (fp32.view(np.uint32) >> 16).astype(np.uint16))

            arr_bytes = arr.nbytes
            ptr = arena.alloc(arr_bytes)
            DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

            # Create NBXTensor wrapping the arena sub-allocation
            strides = _contiguous_strides(shape)
            nbx = NBXTensor(ptr, shape, strides, target_dtype, 'cuda',
                            owns_data=False, device_idx=target_dev)
            weights[key] = nbx
