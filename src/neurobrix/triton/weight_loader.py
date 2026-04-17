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
    upcast_fp16_to_fp32: bool = False,
    per_device_vram_budget: Optional[Dict[int, int]] = None,
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
        upcast_fp16_to_fp32: If True AND the fp32 footprint fits the
            per-device budget, store fp16-target weights as fp32 at load
            time. Used on pre-Ampere hardware (no native bf16) to avoid
            per-call weight upcast in mm/bmm/addmm.
        per_device_vram_budget: Per-device byte budget reserved for
            weights (typically ~50% of device memory, leaving room for
            activations + KV cache). If the doubled fp32 footprint for a
            given device exceeds its budget, upcast is globally disabled
            to keep dtype consistency across the model.
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

    # Phase 1: Scan all shards to compute per-device bytes for both the
    # regular (fp16/bf16-targeted) path and the hypothetical fp32 upcast
    # path. We then decide globally whether to upcast.
    shard_files = sorted(weights_dir.glob("*.safetensors"))
    dev_bytes_regular: Dict[int, int] = {}
    dev_bytes_upcast: Dict[int, int] = {}
    shard_headers = []

    for shard_path in shard_files:
        header, data_offset = _read_header(str(shard_path))
        shard_headers.append((str(shard_path), header, data_offset))

        for key, info in header.items():
            if key == '__metadata__':
                continue
            target_dev = weight_device.get(key, device_idx)
            regular = _target_nbytes(info, compute_dtype)
            upcast = _target_nbytes(info, compute_dtype,
                                    upcast_fp16_to_fp32=True)
            reg_aligned = (regular + ComponentArena.ALIGNMENT - 1) & ~(
                ComponentArena.ALIGNMENT - 1)
            up_aligned = (upcast + ComponentArena.ALIGNMENT - 1) & ~(
                ComponentArena.ALIGNMENT - 1)
            dev_bytes_regular[target_dev] = (
                dev_bytes_regular.get(target_dev, 0) + reg_aligned)
            dev_bytes_upcast[target_dev] = (
                dev_bytes_upcast.get(target_dev, 0) + up_aligned)

    # Decide: upcast only if every device's fp32 footprint fits its budget.
    upcast_effective = upcast_fp16_to_fp32
    if upcast_effective and per_device_vram_budget is not None:
        for dev, up in dev_bytes_upcast.items():
            budget = per_device_vram_budget.get(dev, 0)
            if budget <= 0 or up > budget:
                upcast_effective = False
                print(f"[weight_loader] bind-time fp16→fp32 upcast skipped "
                      f"for {component}: device {dev} needs {up/1e9:.1f}GB "
                      f"fp32, budget {budget/1e9:.1f}GB — per-call fallback "
                      f"will handle overflow protection")
                break
    elif upcast_effective and per_device_vram_budget is None:
        # No budget means we can't verify fit — be conservative.
        upcast_effective = False

    dev_bytes = dev_bytes_upcast if upcast_effective else dev_bytes_regular

    if upcast_effective:
        total_up = sum(dev_bytes_upcast.values())
        total_reg = sum(dev_bytes_regular.values())
        print(f"[weight_loader] bind-time fp16→fp32 upcast ENABLED for "
              f"{component}: {total_reg/1e9:.2f}GB → {total_up/1e9:.2f}GB")

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
            weight_device, arenas,
            upcast_effective=upcast_effective)

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


def _target_nbytes(info: dict, compute_dtype: NBXDtype,
                   upcast_fp16_to_fp32: bool = False) -> int:
    """Compute target byte count for a tensor after dtype remap.

    When upcast_fp16_to_fp32 is set, weights whose EFFECTIVE target
    dtype is fp16 (either native fp16 or bf16 remapped to fp16 via the
    standard bf16→fp16 path) are treated as fp32 for sizing purposes.
    """
    shape = info["shape"]
    sf_dtype = info["dtype"]
    dtype_info = _SF_DTYPE_INFO.get(sf_dtype)
    if dtype_info is None:
        return 0
    _, nbx_dtype, elem_size = dtype_info

    numel = math.prod(shape) if shape else 1

    if upcast_fp16_to_fp32:
        # Replicate the effective-target-dtype logic from the loader.
        target = nbx_dtype
        if nbx_dtype == NBXDtype.bfloat16 and compute_dtype == NBXDtype.float16:
            target = NBXDtype.float16
        elif nbx_dtype == NBXDtype.float16 and compute_dtype == NBXDtype.bfloat16:
            target = NBXDtype.bfloat16
        if target == NBXDtype.float16:
            return numel * 4  # fp32

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
    upcast_effective: bool = False,
) -> None:
    """Load tensors from one shard, sub-allocating from arenas.

    When upcast_effective is True, weights whose target dtype resolves
    to fp16 (via the standard remap chain) are stored as fp32 instead.
    This is decided upstream based on per-device VRAM budget.
    """
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

            # Bind-time fp16→fp32 upcast for pre-Ampere overflow protection.
            # Only promote weights whose final target is fp16; bf16 targets,
            # integer types, and already-fp32 weights pass through unchanged.
            upcast_this = upcast_effective and target_dtype == NBXDtype.float16

            target_dev = weight_device.get(key, device_idx)
            arena = arenas[target_dev]
            DeviceAllocator.set_device(target_dev)

            # GPU-accelerated bf16→fp16 path (same buffer, 2 bytes/elem).
            # Only used when we're NOT also upcasting to fp32 — otherwise
            # the staged bf16→fp32 conversion happens via numpy below.
            if (nbx_dtype == NBXDtype.bfloat16
                    and target_dtype == NBXDtype.float16
                    and not upcast_this):
                ptr = arena.alloc(nbytes)
                arr_u16 = np.frombuffer(raw, dtype=np.uint16)
                DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
                _bf16_to_fp16_inplace(ptr, arr_u16.shape[0], target_dev)
                strides = _contiguous_strides(shape)
                nbx = NBXTensor(ptr, shape, strides, NBXDtype.float16, 'cuda',
                                owns_data=False, device_idx=target_dev)
                weights[key] = nbx
                continue

            # Standard path: numpy → H2D, with optional fp32 upcast.
            if nbx_dtype == NBXDtype.bfloat16:
                # bf16 bits in uint16 containers. If upcast_this, expand
                # them to fp32 via the standard bit-shift: bf16 is the top
                # 16 bits of a fp32, so (bf16_bits << 16) reinterpret-cast
                # as fp32 reproduces the value exactly (no approximation).
                raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                if upcast_this:
                    u32 = raw_u16.astype(np.uint32) << 16
                    arr = np.ascontiguousarray(u32.view(np.float32))
                    final_dtype = NBXDtype.float32
                else:
                    arr = np.ascontiguousarray(raw_u16)
                    final_dtype = target_dtype  # bf16 (kept as uint16)
            else:
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                arr = np.ascontiguousarray(arr)
                if upcast_this:
                    # Native fp16 → fp32 widening (exact, no precision loss).
                    arr = np.ascontiguousarray(arr.astype(np.float32))
                    final_dtype = NBXDtype.float32
                elif target_dtype != nbx_dtype:
                    # fp16 → bf16 (truncation of mantissa, standard path).
                    fp32 = arr.astype(np.float32)
                    arr = np.ascontiguousarray(
                        (fp32.view(np.uint32) >> 16).astype(np.uint16))
                    final_dtype = target_dtype
                else:
                    final_dtype = target_dtype

            arr_bytes = arr.nbytes
            ptr = arena.alloc(arr_bytes)
            DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

            # Create NBXTensor wrapping the arena sub-allocation
            strides = _contiguous_strides(shape)
            nbx = NBXTensor(ptr, shape, strides, final_dtype, 'cuda',
                            owns_data=False, device_idx=target_dev)
            weights[key] = nbx
