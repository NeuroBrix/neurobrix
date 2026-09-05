"""
Device utilities — unified abstraction for CUDA, MPS, XPU, CPU.

ZERO HARDCODE: All device operations dispatch based on device.type.
ZERO FALLBACK: If a device op fails, it crashes. No silent degradation.

Usage:
    from neurobrix.core.device_utils import device_sync, device_empty_cache, device_seed

    device_sync(device)          # Waits for async ops to complete
    device_empty_cache(device)   # Frees unused cached memory
    device_seed(device, seed)    # Sets RNG seed for GPU
    device_memory_stats(device)  # Returns allocated/reserved/total MB
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from neurobrix.core.runtime.tensor_compat import aten


# R33: this module imports no torch. The device belongs to whichever engine
# runs in this process — the ATen branch when it is loaded (its caching
# allocator holds the memory, its RNG seeds the draws), the NBX allocator
# otherwise (a --triton process never loads torch). Every function asks
# ``aten()`` and takes the branch that owns the device.


def _parse_device(device) -> Tuple[str, int]:
    """("cuda", 2) from "cuda:2" / "cuda" / a device object with .type/.index."""
    if isinstance(device, str):
        kind, _, idx = device.partition(":")
        return kind, int(idx) if idx else 0
    kind = getattr(device, "type", str(device))
    idx = getattr(device, "index", None)
    return kind, int(idx) if idx is not None else 0


def device_sync(device: Optional[str] = None) -> None:
    """
    Synchronize device — wait for all async operations to complete.
    Used for timing accuracy at execution boundaries.
    NOT in the compute hot path.
    """
    if device is None:
        return
    kind, idx = _parse_device(device)
    torch = aten()
    if torch is not None:
        if kind == "cuda":
            torch.cuda.synchronize(torch.device(device))
        elif kind == "mps":
            torch.mps.synchronize()
        # CPU, XPU: no explicit sync needed (operations are synchronous)
        return
    if kind == "cuda":
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.device_synchronize(idx)


def device_empty_cache(device: Optional[str] = None) -> None:
    """
    Release unused cached memory on device.
    Called at phase transitions (component unload, flow boundaries).
    """
    if device is None:
        return
    kind, idx = _parse_device(device)
    torch = aten()
    if torch is not None:
        if kind == "cuda":
            torch.cuda.empty_cache()
        elif kind == "mps":
            torch.mps.empty_cache()
        # CPU: no cache to clear
        return
    if kind == "cuda":
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.empty_cache_pool()


def device_seed(device: Optional[str], seed: int) -> None:
    """
    Set RNG seed on device for reproducibility (the ATen branch's RNG).
    Always sets CPU seed. Additionally sets GPU seed if device is GPU.
    The Triton branch seeds its own streams from the same data-driven
    seed (``rng_stream.set_run_seed`` in its flows); nothing to do here.
    """
    torch = aten()
    if torch is None:
        return
    torch.manual_seed(seed)
    if device is None:
        return
    kind, _ = _parse_device(device)
    if kind == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif kind == "mps":
        torch.mps.manual_seed(seed)


def device_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory statistics for device in MB.
    Returns dict with: allocated_mb, reserved_mb, total_mb, free_mb.
    """
    empty = {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
    if device is None:
        return empty
    kind, idx = _parse_device(device)
    torch = aten()
    if torch is not None:
        d = torch.device(device)
        if d.type == "cuda":
            idx = d.index or 0
            allocated = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(idx) / (1024 * 1024)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
                "free_mb": round(total - reserved, 1),
            }
        elif d.type == "mps":
            allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
            # MPS doesn't expose reserved/total like CUDA
            # driver_allocated is closer to "reserved" (includes allocator overhead)
            driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
            recommended = torch.mps.recommended_max_memory() / (1024 * 1024)
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(driver, 1),
                "total_mb": round(recommended, 1),
                "free_mb": round(recommended - driver, 1),
            }
        # CPU / unknown: no GPU memory to report
        return empty
    if kind == "cuda":
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        free_b, total_b = DeviceAllocator.mem_get_info(idx)
        allocated = DeviceAllocator.memory_allocated(idx) / (1024 * 1024)
        total = total_b / (1024 * 1024)
        free = free_b / (1024 * 1024)
        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(total - free, 1),
            "total_mb": round(total, 1),
            "free_mb": round(free, 1),
        }
    return empty


def device_multinomial(probs, num_samples: int = 1):
    """
    torch.multinomial with MPS workaround (ATen branch only).
    MPS backend does not implement torch.multinomial.
    Workaround: compute on CPU, transfer result back to device.
    This is explicit — not a silent fallback. MPS simply lacks this op.
    """
    import torch
    if probs.device.type == "mps":
        result = torch.multinomial(probs.cpu(), num_samples)
        return result.to(probs.device)
    return torch.multinomial(probs, num_samples)
