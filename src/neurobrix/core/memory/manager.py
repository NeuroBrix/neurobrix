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

        CRITICAL: Clear the dict first to release tensor references,
        then run garbage collection, then clear CUDA cache.

        Args:
            weights_dict: Dictionary of weight tensors to clear
            clear_cuda_cache: Whether to call torch.cuda.empty_cache()
            log_prefix: Prefix for log messages
            verbose: Print debug messages
        """
        if verbose:
            print(f"{log_prefix} Unloading {len(weights_dict)} weight tensors")

        # Step 1: Clear the dictionary (releases references)
        weights_dict.clear()

        # Step 2: Force garbage collection
        gc.collect()

        # Step 3: Clear CUDA cache if requested and available
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
        """
        Get current GPU memory statistics.

        Args:
            device: Device string (e.g., "cuda:0"). If None, uses default.

        Returns:
            Dict with allocated_mb, reserved_mb, free_mb
        """
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0}

        device_idx = 0
        if device and ":" in device:
            device_idx = int(device.split(":")[1])

        allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
        total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)
        free = total - reserved

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total,
        }


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
