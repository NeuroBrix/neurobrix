"""Async layer double/N-buffering prefetcher for Triton streaming strategies.

Wires Prism's `ResidencyPlan.prefetch_depth` to `nbxtensor`'s pure `NBXStream`,
`NBXEvent`, and `NBXTensor.to(device, non_blocking=True, stream=...)` primitives
so weight transfers for layer L+1..L+depth overlap with compute on layer L.

Zero Torch imports; zero raw allocator/driver calls; gracefully degrades to
synchronous transfers when `prefetch_depth == 0`, when tensors are unpinned, or
on unified-memory (Metal/MPS) and CPU targets.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from nbxtensor.memory import NBXStream

logger = logging.getLogger(__name__)

_UNIFIED_OR_HOST_PREFIXES = ("cpu", "mps", "metal")


def resolve_prefetch_depth(residency_plan: Any, default: int = 0) -> int:
    """Extract a non-negative `prefetch_depth` from a Prism `ResidencyPlan` or dict."""
    if residency_plan is None:
        return max(0, int(default))
    raw = (
        residency_plan.get("prefetch_depth", default)
        if isinstance(residency_plan, Mapping)
        else getattr(residency_plan, "prefetch_depth", default)
    )
    if raw is None:
        return max(0, int(default))
    return max(0, int(raw))


def _normalize_device(device: Union[str, int]) -> Tuple[str, int]:
    """Return `(device_str, device_idx)` from a string like `'cuda:1'` or int `1`."""
    if isinstance(device, int):
        return f"cuda:{device}", device
    text = str(device or "cuda:0").strip().lower()
    if ":" in text:
        _, _, tail = text.partition(":")
        return text, int(tail) if tail.isdigit() else 0
    return text, 0


def _detect_backend_name() -> str:
    """Query nbxtensor's active GPU backend name safely (empty string if unavailable)."""
    try:
        from nbxtensor.backend import runtime as nbx_rt
        return str(nbx_rt.gpu_backend() or "").strip().lower()
    except Exception:
        return ""


def _is_unified_or_cpu(device_str: str, backend_name: Optional[str]) -> bool:
    """Return True when no separate H2D copy stream is needed (CPU or unified Metal/MPS)."""
    low = device_str.lower()
    if any(low == p or low.startswith(f"{p}:") for p in _UNIFIED_OR_HOST_PREFIXES):
        return True
    resolved_backend = (backend_name if backend_name is not None else _detect_backend_name()).lower()
    return resolved_backend == "metal"


def _is_cpu_tensor(tensor: Any) -> bool:
    """Return True if `tensor` resides on CPU host memory."""
    if getattr(tensor, "is_cpu", False):
        return True
    dev = getattr(tensor, "device", None)
    return str(dev or "").lower() == "cpu"


def _is_pinned_tensor(tensor: Any) -> bool:
    """Return True if `tensor` is backed by page-locked host memory."""
    attr = getattr(tensor, "is_pinned", None)
    if callable(attr):
        return bool(attr())
    if attr is not None:
        return bool(attr)
    return bool(getattr(tensor, "pinned", False))


def _transfer_sync(tensor: Any, device_str: str, device_idx: int) -> Any:
    """Transfer a host tensor synchronously onto `device_str`."""
    if device_str.startswith("cuda") and hasattr(tensor, "to_cuda"):
        return tensor.to_cuda(device_idx)
    if hasattr(tensor, "to"):
        return tensor.to(device_str)
    return tensor


def _close_event_quietly(event: Any) -> None:
    """Close an `NBXEvent` (or compatible handle) without raising."""
    close_fn = getattr(event, "close", None)
    if callable(close_fn):
        close_fn()


class AsyncLayerPrefetcher:
    """Double/N-buffering layer weight prefetcher over `NBXStream` and `NBXEvent`.

    - `prefetch_depth == 0`: synchronous transfer path with zero stream overhead.
    - `prefetch_depth >= 1`: opens one dedicated `NBXStream` copy stream on discrete
      GPUs and overlaps `prefetch(L+1..L+depth)` with compute on layer `L`.
    """

    def __init__(
        self,
        prefetch_depth: int = 0,
        device: Union[str, int] = "cuda:0",
        *,
        stream_factory: Optional[Callable[[int], Any]] = None,
        backend_name: Optional[str] = None,
    ) -> None:
        depth = int(prefetch_depth)
        if depth < 0:
            raise ValueError(f"prefetch_depth must be >= 0, got {depth}")
        self._prefetch_depth: int = depth
        self._device_str, self._device_idx = _normalize_device(device)
        self._buffers: Dict[Hashable, Dict[str, Any]] = {}
        self._events: Dict[Hashable, Any] = {}
        self._copy_stream: Any = None
        self._closed: bool = False

        if depth >= 1 and not _is_unified_or_cpu(self._device_str, backend_name):
            factory = stream_factory if stream_factory is not None else NBXStream
            try:
                self._copy_stream = factory(self._device_idx)
            except Exception as exc:
                logger.warning(
                    f"[AsyncLayerPrefetcher] copy stream create failed on {self._device_str} "
                    f"({exc}); falling back to synchronous prefetch"
                )
                self._copy_stream = None

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def device_idx(self) -> int:
        return self._device_idx

    @property
    def copy_stream(self) -> Any:
        return self._copy_stream

    @property
    def copy_stream_handle(self) -> int:
        return int(self._copy_stream) if self._copy_stream is not None else 0

    @property
    def is_async(self) -> bool:
        return self._copy_stream is not None

    @property
    def cached_keys(self) -> Tuple[Hashable, ...]:
        return tuple(self._buffers.keys())

    @property
    def pending_keys(self) -> Tuple[Hashable, ...]:
        return tuple(self._events.keys())

    def __contains__(self, key: Hashable) -> bool:
        return key in self._buffers

    def __len__(self) -> int:
        return len(self._buffers)

    def next_keys(self, ordered_keys: Sequence[Hashable], current_key: Hashable) -> List[Hashable]:
        """Return up to `prefetch_depth` keys following `current_key` in `ordered_keys`."""
        if self._prefetch_depth <= 0:
            return []
        upcoming: List[Hashable] = []
        for candidate in ordered_keys:
            if candidate > current_key:
                upcoming.append(candidate)
                if len(upcoming) >= self._prefetch_depth:
                    break
        return upcoming

    def prefetch(
        self,
        key: Hashable,
        host_tensors: Mapping[str, Any],
        device: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Stage `host_tensors` for `key` onto `device` (async on copy stream when active)."""
        if key in self._buffers:
            return self._buffers[key]
        target_str, target_idx = (
            _normalize_device(device) if device is not None else (self._device_str, self._device_idx)
        )
        moved: Dict[str, Any] = {}
        enqueued_async = False
        for name, tensor in host_tensors.items():
            if tensor is None:
                continue
            if not _is_cpu_tensor(tensor):
                moved[name] = tensor
                continue
            if self._copy_stream is not None and _is_pinned_tensor(tensor):
                moved[name] = tensor.to(target_str, non_blocking=True, stream=self._copy_stream)
                enqueued_async = True
            else:
                moved[name] = _transfer_sync(tensor, target_str, target_idx)
        self._buffers[key] = moved
        if enqueued_async and self._copy_stream is not None:
            self._events[key] = self._copy_stream.record_event()
        return moved

    def await_layer(self, key: Hashable, compute_stream: Any = None) -> Dict[str, Any]:
        """Ensure `key`'s async H2D transfer is visible to `compute_stream` and return GPU tensors."""
        event = self._events.pop(key, None)
        if event is not None:
            try:
                wait_fn = getattr(compute_stream, "wait_event", None) if compute_stream is not None else None
                if callable(wait_fn):
                    wait_fn(event)
                else:
                    event.synchronize()
            finally:
                _close_event_quietly(event)
        return self._buffers.get(key, {})

    def evict(self, key: Hashable) -> None:
        """Drop cached GPU buffers and any unwaited event for `key`."""
        event = self._events.pop(key, None)
        if event is not None:
            _close_event_quietly(event)
        self._buffers.pop(key, None)

    def clear(self, keep: Optional[Iterable[Hashable]] = None) -> None:
        """Release cached GPU buffers and unwaited events (optionally preserving `keep` keys)."""
        keep_set = set(keep) if keep is not None else set()
        for key in list(self._events.keys()):
            if key not in keep_set:
                _close_event_quietly(self._events.pop(key))
        for key in list(self._buffers.keys()):
            if key not in keep_set:
                self._buffers.pop(key, None)

    def close(self) -> None:
        """Release all cached buffers, pending events, and the dedicated copy stream (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.clear()
        stream = self._copy_stream
        self._copy_stream = None
        if stream is not None:
            close_fn = getattr(stream, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> "AsyncLayerPrefetcher":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False


__all__ = ["AsyncLayerPrefetcher", "resolve_prefetch_depth"]
