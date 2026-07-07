# core/memory/manager.py
"""
Memory Manager - Single Source of Truth for Memory Operations

Consolidates memory cleanup patterns previously duplicated across:
- core/runtime/graph_executor.py (unload_weights, cleanup)
- core/runtime/strategies/base.py (unload_weights)
- core/runtime/strategies/single_gpu.py (unload_weights)

SINGLE SOURCE OF TRUTH: All memory cleanup operations defined here.
"""

import gc
from typing import Dict, Optional, Any

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
        and NBXTensor finalizers that call cudaFree), then GC, then
        empty_cache.

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
            from neurobrix.kernels.wrappers import clear_bias_caches
            clear_bias_caches()
        except Exception:
            pass

        # Step 2: drop references — ComponentArena.__del__ / NBXTensor
        # finalizers run cudaFree here, but the kernels that touched
        # this memory are now guaranteed to have completed.
        weights_dict.clear()

        # Step 3: force garbage collection
        gc.collect()

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
    def cleanup_context(
        context: Any,
        log_prefix: str = "[MemoryManager]",
        verbose: bool = False,
    ) -> None:
        """
        Cleanup execution context runtime state.

        Args:
            context: ExecutionContext or similar object with clear_runtime_state()
            log_prefix: Prefix for log messages
            verbose: Print debug messages
        """
        if context is not None and hasattr(context, "clear_runtime_state"):
            if verbose:
                print(f"{log_prefix} Clearing context runtime state")
            context.clear_runtime_state()

    @staticmethod
    def cleanup_tensors(
        *tensor_dicts: Dict[str, torch.Tensor],
        clear_cuda_cache: bool = True,
        log_prefix: str = "[MemoryManager]",
        verbose: bool = False,
    ) -> None:
        """
        Cleanup multiple tensor dictionaries at once.

        Args:
            *tensor_dicts: Variable number of tensor dictionaries to clear
            clear_cuda_cache: Whether to call torch.cuda.empty_cache()
            log_prefix: Prefix for log messages
            verbose: Print debug messages
        """
        total_cleared = 0
        for tensor_dict in tensor_dicts:
            if tensor_dict is not None:
                total_cleared += len(tensor_dict)
                tensor_dict.clear()

        if verbose:
            print(f"{log_prefix} Cleared {total_cleared} tensors from {len(tensor_dicts)} dicts")

        # Single GC and cache clear at the end
        gc.collect()
        if clear_cuda_cache:
            device_empty_cache()

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
        (`neurobrix.triton.memory_pool.release_flow_memory`) — separate
        implementation by design (never a shared compute helper).
        """
        device_sync(device)
        gc.collect()
        device_empty_cache(device)

    @staticmethod
    def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
        """
        Get current GPU memory statistics.

        Args:
            device: Device string (e.g., "cuda:0"). If None, uses default.

        Returns:
            Dict with allocated_mb, reserved_mb, free_mb
        """
        return device_memory_stats(device)


# Convenience functions (for simpler imports)
def unload_weights(
    weights_dict: Dict[str, torch.Tensor],
    clear_cuda_cache: bool = True,
    log_prefix: str = "[Memory]",
    verbose: bool = False,
) -> None:
    """Convenience function for MemoryManager.unload_weights()."""
    MemoryManager.unload_weights(weights_dict, clear_cuda_cache, log_prefix, verbose)


def cleanup_tensors(
    *tensor_dicts: Dict[str, torch.Tensor],
    clear_cuda_cache: bool = True,
    log_prefix: str = "[Memory]",
    verbose: bool = False,
) -> None:
    """Convenience function for MemoryManager.cleanup_tensors()."""
    MemoryManager.cleanup_tensors(*tensor_dicts, clear_cuda_cache=clear_cuda_cache,
                                   log_prefix=log_prefix, verbose=verbose)


def release_flow_memory(device: Optional[str] = None) -> None:
    """Convenience function for MemoryManager.release_flow_memory()."""
    MemoryManager.release_flow_memory(device)
