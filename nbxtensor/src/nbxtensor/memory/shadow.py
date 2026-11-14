"""The shadow device: the library run with no device in sight, so an application can enumerate every kernel key a
model forms without running it (the engine's launch census) — moved from the engine's census, M2 and the shadow lot.

`install_shadow(devices)` turns the process into a shadow, once: the allocator's direct device operations, streams, events and
queries answer without a driver (`malloc` hands out addresses that address nothing), the launcher launches nothing,
an autotuned kernel's `run` is `shadow_launch` here (nothing), and a value read answers a benign value — or, for a small array
the tensor was made from on the host, its real one. Nothing is installed by itself, and a real run never records
anything here.

Every allocation the shadow hands out is recorded AS a shadow: base -> nbytes, bases sorted. A consumer that must size
an allocation (the Metal pin sizes its whole-allocation wrap) asks `shadow_range` and gets an answer for a recorded
shadow and None for anything else, so an address nobody recorded keeps failing in every run, shadow or not. Measured
2026-09-24: the 76 MoE census failures were the pin asking the allocator about addresses only the shadow had handed
out. Torch-free.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional

__all__ = ["malloc", "shadow_range", "install_shadow", "shadow_device", "shadow_launch", "DevicesVisible"]

_NEXT = [1 << 40]                       # addresses that address nothing, distinct and aligned
_RANGES: Dict[int, int] = {}
_BASES: List[int] = []


def malloc(nbytes: int, dev_idx: Optional[int] = None, pooled: bool = True) -> int:
    """`DeviceAllocator.malloc_direct` in a shadow, its signature kept whole (`pooled`: a weight arena asks for a block
    outside the pool; a shadow has no pool and answers the same address either way)."""
    ptr = _NEXT[0]
    _NEXT[0] += (int(nbytes) + 255) // 256 * 256 + 256
    _RANGES[ptr] = int(nbytes)
    _BASES.append(ptr)          # monotonic: appending keeps the list sorted
    return ptr


def shadow_range(addr: int) -> Optional[tuple]:
    """(base, nbytes) of the RECORDED shadow allocation containing `addr`, or None.

    None for every address this shadow did not hand out, including every address of a real
    run, where nothing is ever recorded here."""
    addr = int(addr)
    i = bisect.bisect_right(_BASES, addr) - 1
    if i < 0:
        return None
    base = _BASES[i]
    size = _RANGES[base]
    return (base, size) if addr < base + max(size, 1) else None


# --------------------------------------------------------------------------------------------
# The whole shadow: installed once per process (`install_shadow`).
# --------------------------------------------------------------------------------------------

_INSTALLED = [False]
_CURRENT = [0]                          # the device a `set_device` named last
_HOST_VALUES: Dict[int, Any] = {}       # shadow address -> the small host array a tensor was made from
_HOST_VALUE_CAP_BYTES = 1 << 20         # only values a flow reads back (a grid, token ids, a frame count) are kept


class DevicesVisible(RuntimeError):
    """`install_shadow` was asked with a device in sight: a shadow is proven unable to touch a card only when none is
    visible (with one visible, a path the shadow does not cover would succeed on real hardware in silence)."""


def shadow_launch(*_args, **_kwargs) -> None:
    """What stands behind an autotuned kernel's `run` in a shadow: nothing (a launch observer's `shadow_run`)."""
    return None


def shadow_device() -> int:
    """The device the shadow's `set_device` named last (0 before any)."""
    return _CURRENT[0]


def _noop(*_a, **_k):
    return None


def _dtype_name(d) -> str:
    """A dtype read by its NAME, never by `str()` of it: `NBXDtype` is an `IntEnum`, and `str()` of one renders
    `NBXDtype.bool_` on Python 3.10 and `9` on 3.11 — a match on `"bool" in str(dtype)` holds on one interpreter only."""
    return str(getattr(d, "name", d)).lower()


def _item_value(dtype):
    """The benign scalar a value read (`.item()`) answers for a value the shadow does not know. In a shadow a value
    read is a guard or a size: a bool reads True (a guard's `all(isfinite(x))` reads healthy), an integer reads ONE
    (a size read at zero shapes an empty tensor; a token id 1 runs the shapes 0 runs), anything else 0.0."""
    name = _dtype_name(dtype)
    if "bool" in name:
        return True
    return 1 if "int" in name else 0.0


def _host_of(t):
    try:
        return _HOST_VALUES.get(int(t.data_ptr()))
    except Exception:  # noqa: BLE001 — a tensor without a pointer has no host value
        return None


def install_shadow(devices: int = 1) -> None:
    """Turn this process into a shadow of `devices` devices: no device memory, no launch, no value. Idempotent.

    Refused (`DevicesVisible`) while the driver sees a device. Install before the kernels are defined: an autotuned
    kernel defined earlier keeps its benchmark (its launch observer reaches `shadow_launch` only through the wrapper the
    library puts on a kernel at its definition)."""
    if _INSTALLED[0]:
        return
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    from nbxtensor.backend import launcher as _launcher
    visible = DA.device_count()
    if visible > 0:
        raise DevicesVisible(f"{visible} device(s) visible")
    _INSTALLED[0] = True
    _shadow_allocator(DA, int(devices))
    _launcher.launch = lambda kernel, grid, *a, **k: None       # every launch is skipped
    _shadow_values()


def _shadow_allocator(DA, n_dev: int) -> None:
    """The direct device operations (an interceptor installed later, a replay recording, still stands in front of
    them as it does in a real run), and the whole driver surface: syncs, streams, events, peer access, host-pinned
    memory, device queries. A path not listed here fails LOUDLY (no device) — never silently."""
    DA.malloc_direct = staticmethod(malloc)
    for name in ("free_direct", "memset_direct", "memcpy_direct", "memcpy_async", "upload", "sync_direct",
                 "ensure_triton_device", "device_synchronize", "stream_synchronize", "event_synchronize",
                 "record_event", "stream_wait_event", "destroy_stream", "destroy_event",
                 "reset_peak_memory", "print_alloc_stats", "_maybe_init_pool", "free_host_pinned"):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(_noop))
    DA.device_count = staticmethod(lambda: n_dev)
    DA.set_device = staticmethod(lambda device_id: _CURRENT.__setitem__(0, int(device_id)))
    DA.get_device = staticmethod(shadow_device)
    _handles = [0]

    def _handle(*_a, **_k):
        _handles[0] += 1
        return _handles[0]
    for name in ("create_stream", "create_event"):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(_handle))
    for name, value in (("clear_last_error", 0), ("event_elapsed_ms", 0.0),
                        ("can_access_peer", False), ("ensure_peer_access", False),
                        ("enable_peer_access", False), ("most_free_device", 0),
                        ("device_free_bytes", -1), ("device_total_bytes", -1),
                        ("mem_get_info", (0, 0)), ("visible_device_memory", []),
                        ("holds", False), ("empty_cache_pool", 0), ("_pool_flush", 0)):
        if hasattr(DA, name):
            setattr(DA, name, staticmethod(lambda *a, _v=value, **k: _v))
    if hasattr(DA, "malloc_host_pinned"):
        DA.malloc_host_pinned = staticmethod(malloc)
    if hasattr(DA, "free_memory_mb"):
        # No room is read: a plan made in a shadow is made against the capacity the application declares.
        DA.free_memory_mb = staticmethod(lambda index=0: None)


def _shadow_values() -> None:
    """A value born on the HOST is known to the shadow: `from_numpy` keeps a small array beside the tensor (in a table
    keyed by the shadow address: the tensor class carries no spare attribute), and every host read answers from it;
    any other value read answers its benign value (`_item_value`)."""
    from nbxtensor.tensor import NBXTensor as T
    _FROM_NUMPY[0] = T.from_numpy
    T.from_numpy = staticmethod(_from_numpy_shadow)
    T.item = _item_shadow
    T.numpy = _numpy_shadow
    if hasattr(T, "tolist"):
        T.tolist = _tolist_shadow
    # A host copy of a strided view reads its bytes (numpy over the pointer): in a shadow a value parked on the host
    # is a shadow address, never memory to read (a segmentation fault, 2026-10-28). Answered unread, as every value.
    setattr(T, "_contiguous_cpu", lambda self: T.empty_cpu(self._shape, self._dtype, pinned=self._pinned))


_FROM_NUMPY: List[Any] = [None]         # the real `from_numpy`, wrapped by the shadow's


def _from_numpy_shadow(arr, dtype=None, device=None):
    t = _FROM_NUMPY[0](arr, dtype, device)
    try:
        import numpy as np
        a = np.asarray(arr)
        if a.nbytes <= _HOST_VALUE_CAP_BYTES:
            _HOST_VALUES[int(t.data_ptr())] = np.array(a, copy=True)
    except Exception:  # noqa: BLE001 — a value without a pointer is a device value
        pass
    return t


def _item_shadow(self):
    h = _host_of(self)
    if h is not None and h.size == 1:
        v = h.reshape(-1)[0]
        return bool(v) if "bool" in str(h.dtype) else (int(v) if "int" in str(h.dtype) else float(v))
    return _item_value(self._dtype)


def _numpy_shadow(self):
    import numpy as np
    h = _host_of(self)
    if h is not None and h.size == int(np.prod(self._shape)):
        return h.reshape(tuple(self._shape))
    # integer reads answer ONES for the reason `_item_value` does; float reads zero
    if "int" in _dtype_name(self._dtype):
        return np.ones(tuple(self._shape), dtype=np.int64)
    return np.zeros(tuple(self._shape), dtype=np.float32)


def _tolist_shadow(self):
    import numpy as np
    h = _host_of(self)
    if h is not None and h.size == np.prod(self._shape):
        return h.reshape(tuple(self._shape)).tolist()
    return np.ones(tuple(self._shape), dtype=np.int64).tolist()
