"""
NeuroBrix Memory Management for I/O.

Provides:
1. IOConfig - Configuration for I/O operations
2. PinnedMemoryManager - Pinned memory for PCIe DMA transfers
3. PrefetchQueue - Overlapping I/O and compute

ZERO HARDCODE: Configuration from environment variables
"""

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Any

import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IOConfig:
    """I/O configuration for loaders."""

    # Number of parallel workers for disk I/O
    num_workers: int = 8

    # Use pinned memory for DMA transfers
    use_pinned_memory: bool = True

    # Enable prefetching (load next component while current executes)
    enable_prefetch: bool = True

    # Prefetch queue size (number of components to prefetch)
    prefetch_queue_size: int = 2

    # Log performance metrics
    log_performance: bool = True

    @classmethod
    def from_env(cls) -> "IOConfig":
        """Create config from environment variables."""
        return cls(
            num_workers=int(os.environ.get("NBX_IO_WORKERS", "8")),
            use_pinned_memory=os.environ.get("NBX_PINNED_MEMORY", "1") == "1",
            enable_prefetch=os.environ.get("NBX_PREFETCH", "1") == "1",
            prefetch_queue_size=int(os.environ.get("NBX_PREFETCH_SIZE", "2")),
            log_performance=os.environ.get("NBX_IO_LOG", "1") == "1",
        )


# Global config instance
_io_config: Optional[IOConfig] = None


def get_io_config() -> IOConfig:
    """Get global I/O configuration."""
    global _io_config
    if _io_config is None:
        _io_config = IOConfig.from_env()
    return _io_config


def set_io_config(config: IOConfig) -> None:
    """Set global I/O configuration."""
    global _io_config
    _io_config = config


# ============================================================================
# Pinned Memory Manager
# ============================================================================

class PinnedMemoryManager:
    """
    Manages pinned memory buffers for DMA transfers.

    Pinned memory enables PCIe DMA (16GB/s) instead of CPU copy (~8GB/s).

    Usage:
        manager = PinnedMemoryManager()
        pinned_dict = manager.to_pinned(tensors)
        gpu_dict = manager.to_device(pinned_dict, "cuda:0", non_blocking=True)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._stats = {"pinned_bytes": 0, "transfers": 0}

    def to_pinned(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move tensors to pinned memory for fast GPU transfer.

        Args:
            tensors: Dict of name -> tensor (on CPU)

        Returns:
            Dict of name -> tensor (pinned memory)
        """
        if not self.enabled:
            return tensors

        pinned = {}
        for name, tensor in tensors.items():
            if tensor.device.type == "cpu" and not tensor.is_pinned():
                # Create pinned tensor and copy data
                pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                pinned_tensor.copy_(tensor)
                pinned[name] = pinned_tensor
                self._stats["pinned_bytes"] += tensor.numel() * tensor.element_size()
            else:
                pinned[name] = tensor

        return pinned

    def to_device(
        self,
        tensors: Dict[str, torch.Tensor],
        device: str,
        non_blocking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer tensors to GPU device using DMA (non-blocking).

        Args:
            tensors: Dict of name -> tensor (pinned memory)
            device: Target device (e.g., "cuda:0")
            non_blocking: Use async transfer (True for DMA)

        Returns:
            Dict of name -> tensor (on GPU)
        """
        result = {}
        for name, tensor in tensors.items():
            result[name] = tensor.to(device, non_blocking=non_blocking)
            self._stats["transfers"] += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pinned memory statistics."""
        return {
            **self._stats,
            "pinned_mb": self._stats["pinned_bytes"] / 1e6,
        }


# ============================================================================
# Prefetch Queue
# ============================================================================

@dataclass
class PrefetchItem:
    """Item in the prefetch queue."""
    component_name: str
    shard_paths: List[Path]
    dtype: Optional[torch.dtype] = None
    device: str = "cpu"  # Prefetch to CPU/pinned, NOT GPU
    tensors: Optional[Dict[str, torch.Tensor]] = None
    ready: threading.Event = field(default_factory=threading.Event)


class PrefetchQueue:
    """
    Prefetch queue for overlapping I/O and compute.

    While GPU executes Component A, this loads Component B to pinned RAM.

    Usage:
        queue = PrefetchQueue()
        queue.start()

        # Queue next component for prefetch
        queue.prefetch("transformer", shard_paths, dtype)

        # Get prefetched weights (blocks if not ready)
        weights = queue.get("transformer", device="cuda:0")

        queue.stop()

    CRITICAL: Prefetch happens to CPU/pinned RAM only, never to GPU.
    This avoids VRAM OOM from holding two components simultaneously.
    """

    def __init__(self, config: Optional[IOConfig] = None):
        self.config = config or get_io_config()
        # Lazy import to avoid circular dependency
        self._loader = None

        self._queue: Queue[PrefetchItem] = Queue(
            maxsize=self.config.prefetch_queue_size
        )
        self._cache: Dict[str, PrefetchItem] = {}
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def _get_loader(self):
        """Lazy load SmartLoader to avoid circular import."""
        if self._loader is None:
            from neurobrix.core.io.loader import SmartLoader
            self._loader = SmartLoader(self.config)
        return self._loader

    def start(self) -> None:
        """Start the prefetch worker thread."""
        if self._worker is not None:
            return

        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()


    def stop(self) -> None:
        """Stop the prefetch worker thread."""
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def prefetch(
        self,
        component_name: str,
        shard_paths: List[Path],
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Queue a component for prefetch.

        IMPORTANT: Prefetches to CPU/pinned RAM, not GPU.

        Args:
            component_name: Name of component to prefetch
            shard_paths: Paths to weight shard files
            dtype: Target dtype for conversion
        """
        if not self.config.enable_prefetch:
            return

        # Check if already prefetched/queued
        with self._lock:
            if component_name in self._cache:
                return

        item = PrefetchItem(
            component_name=component_name,
            shard_paths=shard_paths,
            dtype=dtype,
            device="cpu",  # Always prefetch to CPU, not GPU
        )

        try:
            self._queue.put_nowait(item)
            with self._lock:
                self._cache[component_name] = item

        except:
            # Queue full, skip prefetch
            pass

    def get(
        self,
        component_name: str,
        device: str,
        timeout: float = 60.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get prefetched weights, transferring to GPU.

        If component is not prefetched, returns None.

        Args:
            component_name: Name of component
            device: Target GPU device
            timeout: Max wait time for prefetch completion

        Returns:
            Dict of tensors on GPU, or None if not prefetched
        """
        with self._lock:
            item = self._cache.get(component_name)

        if item is None:
            return None

        # Wait for prefetch to complete
        if not item.ready.wait(timeout=timeout):
            return None

        # Transfer from pinned CPU to GPU
        if item.tensors is not None:
            loader = self._get_loader()
            result = loader.transfer_to_device(
                item.tensors, device, non_blocking=True
            )

            # Clear from cache to free pinned memory
            with self._lock:
                del self._cache[component_name]

            return result

        return None

    def clear(self) -> None:
        """Clear all prefetched data."""
        with self._lock:
            self._cache.clear()
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _worker_loop(self) -> None:
        """Background worker that performs prefetch operations."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                # Load to pinned memory (NOT to GPU)
                loader = self._get_loader()
                item.tensors = loader.load_shards_parallel(
                    item.shard_paths,
                    to_pinned=True,
                    dtype=item.dtype,
                )

            except Exception as e:
                # Log error but don't crash — prefetch is best-effort
                import sys
                print(f"[PrefetchQueue] Error prefetching {item.component_name}: {e}", file=sys.stderr)
                item.tensors = None

            finally:
                item.ready.set()


# ============================================================================
# Global Instances
# ============================================================================

_prefetch_queue: Optional[PrefetchQueue] = None


def get_prefetch_queue() -> PrefetchQueue:
    """Get global PrefetchQueue instance."""
    global _prefetch_queue
    if _prefetch_queue is None:
        _prefetch_queue = PrefetchQueue()
        _prefetch_queue.start()
    return _prefetch_queue


def shutdown_io() -> None:
    """Shutdown I/O system (stop prefetch worker)."""
    global _prefetch_queue
    if _prefetch_queue is not None:
        _prefetch_queue.stop()
        _prefetch_queue = None
