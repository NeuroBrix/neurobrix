"""
NeuroBrix Unified Loader - Single Source of Truth for Parallel I/O.

Consolidates parallel_loader.py + smart_loader.py eliminating duplication.

Features:
1. Parallel file loading (ThreadPoolExecutor)
2. Pinned memory for DMA transfers (16GB/s PCIe)
3. Prefetch queue for overlapping I/O and compute
4. Device-targeted loading for FGP

ZERO HARDCODE: Configuration from environment or IOConfig
ZERO FALLBACK: Explicit errors for missing files
"""

import time
from pathlib import Path
from typing import Dict, List, Callable, TypeVar, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file  # type: ignore[attr-defined]
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    safetensors_load_file = None  # type: ignore[assignment]

# Import pinned memory from dedicated module
from neurobrix.core.io.memory import PinnedMemoryManager, get_io_config, IOConfig

T = TypeVar('T')

# Default workers from config
DEFAULT_NUM_WORKERS = 8


# ============================================================================
# Core Parallel Loading Functions (Canonical Implementation)
# ============================================================================

def load_files_parallel(
    files: List[Union[str, Path]],
    loader_fn: Callable[[Union[str, Path]], Dict[str, torch.Tensor]],
    max_workers: int = DEFAULT_NUM_WORKERS,
    log_progress: bool = True,
    prefix: str = "[Loader]"
) -> Dict[str, torch.Tensor]:
    """
    Load multiple weight files in parallel using ThreadPoolExecutor.

    This is the CANONICAL implementation - all parallel loading should use this.

    Args:
        files: List of file paths to load
        loader_fn: Function that loads a single file and returns Dict[str, Tensor]
        max_workers: Maximum number of parallel workers
        log_progress: Whether to print progress logs
        prefix: Log message prefix for identification

    Returns:
        Dict[str, torch.Tensor]: Combined weights from all files

    Raises:
        RuntimeError: If any file fails to load (ZERO FALLBACK)
    """
    if not files:
        return {}

    # Single file - no parallelism needed
    if len(files) == 1:
        return loader_fn(files[0])

    # Parallel loading
    weights: Dict[str, torch.Tensor] = {}
    num_workers = min(max_workers, len(files))

    errors: List[Tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(loader_fn, f): f for f in files}

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                file_weights = future.result()
                weights.update(file_weights)
            except Exception as e:
                errors.append((str(file_path), e))

    # ZERO FALLBACK: Crash on errors
    if errors:
        error_msgs = [f"  {path}: {e}" for path, e in errors]
        raise RuntimeError(
            f"ZERO FALLBACK: Failed to load {len(errors)} files:\n" +
            "\n".join(error_msgs)
        )

    return weights


def load_shards_with_devices(
    shard_items: List[Tuple[Union[str, Path], str]],
    loader_fn: Callable[[Union[str, Path], str], Dict[str, torch.Tensor]],
    max_workers: int = DEFAULT_NUM_WORKERS,
    log_progress: bool = True,
    prefix: str = "[Loader]"
) -> Dict[str, torch.Tensor]:
    """
    Load shards in parallel with per-shard device targeting.

    Used for FGP (Fine-Grained Pipeline) where different shards go to different GPUs.

    Args:
        shard_items: List of (file_path, device_string) tuples
        loader_fn: Function that loads a file to a specific device
        max_workers: Maximum parallel workers
        log_progress: Whether to print progress
        prefix: Log message prefix

    Returns:
        Dict[str, torch.Tensor]: Combined weights from all shards
    """
    if not shard_items:
        return {}

    if len(shard_items) == 1:
        path, device = shard_items[0]
        return loader_fn(path, device)

    weights: Dict[str, torch.Tensor] = {}
    num_workers = min(max_workers, len(shard_items))

    def load_with_device(item: Tuple[Union[str, Path], str]) -> Tuple[str, str, Dict[str, torch.Tensor]]:
        path, device = item
        file_name = Path(path).name if isinstance(path, (str, Path)) else str(path)
        return file_name, device, loader_fn(path, device)

    errors: List[Tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_with_device, item): item for item in shard_items}

        for future in as_completed(futures):
            item = futures[future]
            try:
                file_name, device, shard_weights = future.result()
                weights.update(shard_weights)
            except Exception as e:
                errors.append((str(item[0]), e))

    if errors:
        error_msgs = [f"  {path}: {e}" for path, e in errors]
        raise RuntimeError(
            f"ZERO FALLBACK: Failed to load {len(errors)} shards:\n" +
            "\n".join(error_msgs)
        )

    return weights


def execute_parallel(
    items: List[T],
    worker_fn: Callable[[T], Any],
    max_workers: int = DEFAULT_NUM_WORKERS,
    log_progress: bool = False,
    prefix: str = "[Loader]"
) -> List[Any]:
    """
    Generic parallel execution for any task list.

    Args:
        items: List of items to process
        worker_fn: Function to apply to each item
        max_workers: Maximum parallel workers
        log_progress: Whether to log progress
        prefix: Log prefix

    Returns:
        List of results in completion order (NOT input order)
    """
    if not items:
        return []

    if len(items) == 1:
        return [worker_fn(items[0])]

    num_workers = min(max_workers, len(items))
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


# ============================================================================
# Standard File Loaders
# ============================================================================

def load_safetensors_file(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a safetensors file. ZERO FALLBACK: crash if file missing."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"ZERO FALLBACK: Weight file not found: {path}")

    if not HAS_SAFETENSORS or safetensors_load_file is None:
        raise RuntimeError("ZERO FALLBACK: safetensors not installed")

    return safetensors_load_file(str(path))


def load_pytorch_file(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a PyTorch .pt/.bin file. ZERO FALLBACK: crash if file missing."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"ZERO FALLBACK: Weight file not found: {path}")

    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_weight_file_auto(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Auto-detect format and load weight file. ZERO FALLBACK: crash if unsupported."""
    path = Path(path)

    if path.suffix == ".safetensors":
        return load_safetensors_file(path)
    elif path.suffix in (".pt", ".bin", ".pth"):
        return load_pytorch_file(path)
    else:
        raise RuntimeError(f"ZERO FALLBACK: Unsupported weight format: {path.suffix}")


def create_device_loader(
    device: str,
    dtype: Optional[torch.dtype] = None
) -> Callable[[Union[str, Path]], Dict[str, torch.Tensor]]:
    """
    Create a loader function that loads to a specific device with optional dtype conversion.

    Args:
        device: Target device string (e.g., "cuda:0")
        dtype: Optional dtype to convert weights to

    Returns:
        Loader function for use with load_files_parallel
    """
    def loader(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        weights = load_weight_file_auto(path)

        result = {}
        for key, tensor in weights.items():
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            result[key] = tensor.to(device=device)

        return result

    return loader


# ============================================================================
# Smart Loader with Pinned Memory (for high-performance scenarios)
# ============================================================================

class SmartLoader:
    """
    High-performance file loader with parallel I/O and pinned memory.

    Uses the unified load_files_parallel() internally, adding:
    1. Pinned memory for PCIe DMA (16GB/s)
    2. Non-blocking GPU transfers
    3. Performance logging

    Usage:
        loader = SmartLoader()
        weights = loader.load_and_transfer(shard_paths, device="cuda:0")
    """

    def __init__(self, config: Optional[IOConfig] = None):
        self.config = config or get_io_config()
        self.pinned_manager = PinnedMemoryManager(self.config.use_pinned_memory)
        self._stats = {
            "shards_loaded": 0,
            "total_bytes": 0,
            "total_time": 0.0,
        }

    def load_shard(
        self,
        shard_path: Path,
        to_pinned: bool = True,
    ) -> Tuple[str, Dict[str, torch.Tensor]]:
        """Load a single shard file to CPU (optionally pinned)."""
        shard_name = shard_path.name
        tensors = load_weight_file_auto(shard_path)

        # Move to pinned memory for fast GPU transfer
        if to_pinned and self.config.use_pinned_memory:
            tensors = self.pinned_manager.to_pinned(tensors)

        return shard_name, tensors

    def load_shards_parallel(
        self,
        shard_paths: List[Path],
        to_pinned: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load multiple shards in parallel to CPU/pinned memory."""
        if not shard_paths:
            return {}

        start_time = time.time()

        # Use canonical load_files_parallel
        def loader(path):
            _, tensors = self.load_shard(Path(path), to_pinned)
            return tensors

        shard_paths_list: List[Union[str, Path]] = list(shard_paths)
        weights = load_files_parallel(
            shard_paths_list,
            loader,
            max_workers=self.config.num_workers,
            log_progress=self.config.log_performance,
            prefix="[SmartLoader]"
        )

        # Convert dtype if specified
        if dtype is not None:
            weights = self._convert_dtype(weights, dtype)

        # Update stats
        elapsed = time.time() - start_time
        total_bytes = sum(t.numel() * t.element_size() for t in weights.values())
        self._stats["total_bytes"] += total_bytes
        self._stats["total_time"] += elapsed
        self._stats["shards_loaded"] += len(shard_paths)

        return weights

    def transfer_to_device(
        self,
        tensors: Dict[str, torch.Tensor],
        device: str,
        non_blocking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Transfer tensors from pinned memory to GPU using DMA."""
        start_time = time.time()

        result = self.pinned_manager.to_device(tensors, device, non_blocking)

        # Sync to measure actual transfer time
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

        elapsed = time.time() - start_time

        return result

    def load_and_transfer(
        self,
        shard_paths: List[Path],
        device: str,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load shards in parallel and transfer to GPU using DMA."""
        # Step 1: Load to pinned memory in parallel
        pinned = self.load_shards_parallel(shard_paths, to_pinned=True, dtype=dtype)

        # Step 2: DMA transfer to GPU
        if device.startswith("cuda") or device.startswith("hip"):
            return self.transfer_to_device(pinned, device, non_blocking=True)
        else:
            return pinned  # CPU device, no transfer needed

    def _convert_dtype(
        self,
        tensors: Dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Convert floating-point tensors to target dtype."""
        result = {}
        for name, tensor in tensors.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                # Prism validates bf16→fp16 safety upstream
                result[name] = tensor.to(dtype)
            else:
                result[name] = tensor
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        avg_speed = 0
        if self._stats["total_time"] > 0:
            avg_speed = (self._stats["total_bytes"] / 1e6) / self._stats["total_time"]

        return {
            **self._stats,
            "total_mb": self._stats["total_bytes"] / 1e6,
            "avg_speed_mbs": avg_speed,
            "pinned_stats": self.pinned_manager.get_stats(),
        }


# ============================================================================
# Global Instances
# ============================================================================

_smart_loader: Optional[SmartLoader] = None


def get_smart_loader() -> SmartLoader:
    """Get global SmartLoader instance."""
    global _smart_loader
    if _smart_loader is None:
        _smart_loader = SmartLoader()
    return _smart_loader
