"""
NeuroBrix I/O Module.

High-performance file loading with:
- Parallel shard loading (ThreadPoolExecutor)
- Pinned memory for DMA transfers (16GB/s PCIe)
- Prefetch queue for overlapping I/O and compute

ARCHITECTURE (after consolidation):
- loader.py: Unified parallel loading (replaces smart_loader + parallel_loader)
- memory.py: IOConfig, PinnedMemoryManager, PrefetchQueue
- weight_loader.py: NBX-specific weight loading (from NBX container/cache)
"""

# Memory management and config
from neurobrix.core.io.memory import (
    IOConfig,
    get_io_config,
    set_io_config,
    PinnedMemoryManager,
    PrefetchQueue,
    get_prefetch_queue,
    shutdown_io,
)

# Unified loader (canonical parallel loading)
from neurobrix.core.io.loader import (
    # Core functions
    load_files_parallel,
    load_shards_with_devices,
    execute_parallel,
    # File loaders
    load_safetensors_file,
    load_pytorch_file,
    load_weight_file_auto,
    create_device_loader,
    # Smart loader with pinned memory
    SmartLoader,
    get_smart_loader,
    # Constants
    DEFAULT_NUM_WORKERS,
)

# NBX weight loader (moved from core/runtime/)
from neurobrix.core.io.weight_loader import (
    WeightLoader,
    safe_dtype_convert,
    PARALLEL_SHARD_WORKERS,
)

__all__ = [
    # Config
    "IOConfig",
    "get_io_config",
    "set_io_config",
    # Memory
    "PinnedMemoryManager",
    "PrefetchQueue",
    "get_prefetch_queue",
    "shutdown_io",
    # Core parallel loading
    "load_files_parallel",
    "load_shards_with_devices",
    "execute_parallel",
    # File loaders
    "load_safetensors_file",
    "load_pytorch_file",
    "load_weight_file_auto",
    "create_device_loader",
    # Smart loader
    "SmartLoader",
    "get_smart_loader",
    # NBX weight loader
    "WeightLoader",
    "safe_dtype_convert",
    # Constants
    "DEFAULT_NUM_WORKERS",
    "PARALLEL_SHARD_WORKERS",
]
