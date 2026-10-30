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
        clear_cuda_cache: bool = True,
        log_prefix: str = "[MemoryManager]",
        verbose: bool = False,
    ) -> None:
        """
        Unload weights and free GPU memory.

        CRITICAL: Sync every device that holds memory in this dict BEFORE
        clearing references, then drop refs (which triggers ComponentArena
        and NBXTensor finalizers that call cudaFree), then empty_cache. No
        heap collection: this runs once per streamed segment per step.

        Why the pre-clear sync matters: ComponentArena.free() (triton path)
        calls DeviceAllocator.free_cuda directly with no internal sync.
        If a kernel issued by the executor is still in-flight on the
        stream when we free its input/output buffer, the kernel reads
        from freed memory → cudaErrorIllegalAddress (err 700) silently
        corrupts the CUDA context. The next cudaMalloc on that context
        fails with err 700, which our wrapper misreports as "GPU malloc
        failed". The path is only exposed when unload actually runs
        between phases (lifecycle / lazy strategies) — eager runs never
        triggered it.

        The previous implementation called device_sync() AFTER clear()
        and without an explicit device argument, making it a global
        no-op (device_sync(None) returns immediately, see
        core/device_utils.py:27). device_empty_cache() was called the
        same way and returns early on None as well (device_utils.py:43).
        Both sync and cache flush are now per-device, driven by the
        same enumerated set.

        Args:
            weights_dict: Dictionary of weight tensors to clear
            clear_cuda_cache: Whether to call empty_cache after clear
            log_prefix: Prefix for log messages
            verbose: Print debug messages
        """
        if verbose:
            print(f"{log_prefix} Unloading {len(weights_dict)} weight tensors")

        # Step 1: collect every device holding memory in this dict, so we
        # can sync each one before freeing. Multi-GPU safe.
        # - "_arenas" key (triton path): Dict[int, ComponentArena] keyed
        #   by device_idx. The arena owns the cudaMalloc'd block.
        # - NBXTensor (triton path): exposes private _device_idx; its
        #   public .device property returns self (a NBXTensor is its own
        #   device context), so we MUST NOT call str(val.device) here.
        # - torch.Tensor (native path): standard .device → torch.device.
        devices_to_sync = set()
        for key, val in weights_dict.items():
            if key == "_arenas":
                if isinstance(val, dict):
                    for arena in val.values():
                        idx = getattr(arena, "device_idx", None)
                        if idx is not None:
                            devices_to_sync.add(f"cuda:{idx}")
            elif hasattr(val, "_device_idx"):
                # NBXTensor: private attr is the source of truth
                devices_to_sync.add(f"cuda:{val._device_idx}")
            elif hasattr(val, "device") and hasattr(val.device, "type"):
                # torch.Tensor: .device is a torch.device instance
                dev = val.device
                if dev.type in ("cuda", "mps"):
                    devices_to_sync.add(str(dev))
        for dev in devices_to_sync:
            device_sync(dev)

        # Step 1b: evict the triton attention bias caches (audit #2 F7).
        # `wrappers._causal_bias_cache` holds MATERIALIZED [Sq, Sk] additive
        # masks keyed by (device, Sq, Sk, dtype); a warm serve sweeping many
        # sequence lengths grows it unbounded because nothing reclaimed it.
        # Unload boundaries are the right eviction point (a diffusion loop
        # never unloads per step, so hot reuse of one bias table is intact).
        # After the per-device sync above, no in-flight kernel is still
        # reading these tensors, so dropping the refs (→ finalizer cudaFree)
        # is UAF-safe. Lazy import + guard: a triton-substrate cleanup must
        # never break a compiled-path unload, and never raise.
        try:
            from nbxtensor.ops.elementwise_wrappers import clear_bias_caches
            clear_bias_caches()
        except Exception:
            pass

        # Step 1c: release the MoE pointer tables that PIN weights (Metal: each holds its expert
        # tensors and a whole-allocation wrap for as long as it is cached). Without this the
        # reference drop below frees nothing an MoE table pins, and a streamed segment stays in
        # memory beside the next one (`nbxtensor.memory.pointer_tables.release_pinned_tables`). Only when
        # the pointer-table module is already loaded: a run that never built a table has none to release, and a
        # compiled-path unload imports nothing. On CUDA a table pins nothing and none is dropped.
        # Never swallowed: a table that could not release its pins keeps the unloaded weights in
        # memory — the very defect this step removes — and that must be seen, not survived.
        import sys
        _tables = sys.modules.get("nbxtensor.memory.pointer_tables")
        if _tables is not None:
            _tables.release_pinned_tables()

        # Step 2: drop references — ComponentArena.__del__ / NBXTensor
        # finalizers run cudaFree here, but the kernels that touched
        # this memory are now guaranteed to have completed.
        weights_dict.clear()

        # Step 3 is the reference drop above, not a heap collection. A full gc.collect() stood
        # here from v0.1.0, with no measurement of its need, and layer streaming calls this for
        # every segment of every decode step: the 30B MoE census shadows spent 999 of 999 profile
        # samples in it (2026-09-29). NBXTensor and ComponentArena free their device memory at
        # refcount zero; the flow boundary (`release_flow_memory`) still collects once per stage.
        # The zero3 MoE path dropped its own per-call collect the same way (84f2cb88).

        # Step 4: per-device cache flush. device_empty_cache(None) is a
        # no-op (core/device_utils.py:43) — same trap as device_sync. We
        # reuse the set collected above so the flush targets exactly the
        # devices we just touched. Empty set → nothing was allocated →
        # nothing to flush, skip cleanly.
        # Note: device strings are "cuda:N" here, coherent with NBXTensor's
        # current cuda hardcode. Portability to ROCm/MPS will be addressed
        # as a single coherent pass when NBXTensor factory methods are
        # migrated — see NBXTensor debt.
        if clear_cuda_cache:
            for dev in devices_to_sync:
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
def release_flow_memory(device: Optional[str] = None) -> None:
    """Convenience function for MemoryManager.release_flow_memory()."""
    MemoryManager.release_flow_memory(device)
