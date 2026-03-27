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

import torch
from typing import Dict, Optional


def device_sync(device: Optional[str] = None) -> None:
    """
    Synchronize device — wait for all async operations to complete.

    Used for timing accuracy at execution boundaries.
    NOT in the compute hot path.
    """
    if device is None:
        return
    dtype = torch.device(device).type if isinstance(device, str) else device.type
    if dtype == "cuda":
        torch.cuda.synchronize(torch.device(device))
    elif dtype == "mps":
        torch.mps.synchronize()
    # CPU, XPU: no explicit sync needed (operations are synchronous)


def device_empty_cache(device: Optional[str] = None) -> None:
    """
    Release unused cached memory on device.

    Called at phase transitions (component unload, flow boundaries).
    """
    if device is None:
        return
    dtype = torch.device(device).type if isinstance(device, str) else device.type
    if dtype == "cuda":
        torch.cuda.empty_cache()
    elif dtype == "mps":
        torch.mps.empty_cache()
    # CPU: no cache to clear


def device_seed(device: Optional[str], seed: int) -> None:
    """
    Set RNG seed on device for reproducibility.

    Always sets CPU seed. Additionally sets GPU seed if device is GPU.
    """
    torch.manual_seed(seed)
    if device is None:
        return
    dtype = torch.device(device).type if isinstance(device, str) else device.type
    if dtype == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif dtype == "mps":
        torch.mps.manual_seed(seed)


def device_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory statistics for device in MB.

    Returns dict with: allocated_mb, reserved_mb, total_mb, free_mb.
    """
    if device is None:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}

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
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}


def device_multinomial(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """
    torch.multinomial with MPS workaround.

    MPS backend does not implement torch.multinomial.
    Workaround: compute on CPU, transfer result back to device.
    This is explicit — not a silent fallback. MPS simply lacks this op.
    """
    if probs.device.type == "mps":
        result = torch.multinomial(probs.cpu(), num_samples)
        return result.to(probs.device)
    return torch.multinomial(probs, num_samples)
