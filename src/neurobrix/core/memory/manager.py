# core/memory/manager.py
"""
Memory Manager - Single Source of Truth for Memory Operations

Consolidates memory cleanup patterns previously duplicated across:
- core/runtime/graph_executor.py (unload_weights, cleanup)
- core/runtime/strategies/base.py (unload_weights)
- core/runtime/strategies/single_gpu.py (unload_weights)

SINGLE SOURCE OF TRUTH: All memory cleanup operations defined here.
"""
from __future__ import annotations

import gc
from typing import Dict, Optional, Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
from neurobrix.core.device_utils import device_empty_cache, device_sync, device_memory_stats


class MemoryManager:
    """
    Centralized memory management for NeuroBrix runtime.

    Provides static methods for common memory operations to eliminate
    code duplication across executor and strategy classes.
    """

    @staticmethod
    def unload_weights(
        weights_dict: Dict[str, torch.Tensor],
    ) -> None:
        """
        Unload weights and free device memory: every device holding one of them synchronised BEFORE the references
        drop (a kernel still in flight on a freed buffer is the err-700 class: the context corrupted, the next malloc
        refused), then the references dropped, then each device's cache emptied. No heap collection: this runs once
        per streamed segment per step (the 30B MoE census spent 999 of 999 profile samples in a gc.collect() here,
        2026-09-29).

        The Triton engine's weights (NBXTensors, their arenas) are released by nbxtensor's `release_weights`, which
        also drops the library's caches that hold weights (attention-bias tables, the pointer tables pinning experts).
        The PyTorch engine's weights are torch's: synchronised, dropped and its cache emptied per device here.
        """
        import sys
        if "nbxtensor.tensor" in sys.modules:       # the library's weights exist only once it is loaded
            from nbxtensor.memory import release_weights
            if release_weights(weights_dict):
                return
        devices = {str(v.device) for v in weights_dict.values()
                   if getattr(getattr(v, "device", None), "type", None) in ("cuda", "mps")}
        for dev in devices:
            device_sync(dev)
        weights_dict.clear()
        for dev in devices:
            device_empty_cache(dev)

    @staticmethod
    def cleanup_context(context: Any) -> None:
        """Clear an execution context's runtime state (an object with clear_runtime_state())."""
        if context is not None and hasattr(context, "clear_runtime_state"):
            context.clear_runtime_state()

    @staticmethod
    def release_flow_memory(device: Optional[str] = None) -> None:
        """
        Flow-boundary memory release with the sync-before-free discipline.

        Consolidates the hand-rolled `gc.collect(); device_empty_cache(dev)`
        pairs that flow handlers ran at stage boundaries (after component
        unload / between phases). Those pairs skipped the manager's
        sync-before-free protection: `gc.collect()` is what triggers the
        finalizers (NBXTensor / ComponentArena → cudaFree) on cyclic or
        lingering references, and freeing memory that an in-flight kernel
        is still reading is the err-700 class documented in
        `unload_weights` above. Ordering here:

          1. device_sync(device)   — every pending kernel on the device
             completes BEFORE any reference can be finalized;
          2. gc.collect()          — drop refs, run finalizers;
          3. device_empty_cache(device) — return freed blocks.

        `device` is the flow's primary device string ("cuda:N"); None is a
        no-op sync/flush (CPU-only contexts), the collect still runs.

        The triton flows use the triton-side mirror
        (`nbxtensor.memory.memory_pool.release_flow_memory`) — separate
        implementation by design (never a shared compute helper).
        """
        device_sync(device)
        gc.collect()
        device_empty_cache(device)


# The flows' boundary release, imported by name
release_flow_memory = MemoryManager.release_flow_memory
