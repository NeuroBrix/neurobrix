"""Memory: slot storage, and (as the move proceeds) allocators, pools and transfers. The names below are the
application's door to them: each answers from the module that holds the mechanism."""


def set_limit(nbytes, device=None) -> None:
    """Hold this process to a card of `nbytes` bytes on `device` (an ordinal; None: every device), or lift the limit
    (`nbytes` None): the allocator refuses what such a card refuses and its pool parks a fraction of the limit
    (`DeviceAllocator.set_limit`). Optional; absent, the physical card is the limit."""
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.set_limit(nbytes, device)


def release_weights(weights: dict) -> bool:
    """A loaded weight set released after its devices drain (`memory_pool.release_weights`): True when it was the
    library's, False and untouched otherwise."""
    from nbxtensor.memory.memory_pool import release_weights as release
    return release(weights)


def join_weights(held: dict, loaded: dict) -> dict:
    """`loaded` with its arenas joined to `held`'s, to update `held` with (`memory_pool.join_weights`)."""
    from nbxtensor.memory.memory_pool import join_weights as join
    return join(held, loaded)


def pool_enabled() -> bool:
    """Whether the allocator parks freed blocks (`pool.parks_freed_blocks`)."""
    from nbxtensor.memory.pool import parks_freed_blocks
    return parks_freed_blocks()


def pool_parked_cap_bytes(vendor: str, architecture: str, total_bytes: float) -> int:
    """The bytes the pool may keep parked on such a device, 0 unbounded (`pool.parked_cap_bytes`)."""
    from nbxtensor.memory.pool import parked_cap_bytes
    return parked_cap_bytes(vendor, architecture, total_bytes)


def pool_fit_slack(vendor: str, architecture: str) -> float:
    """The most the pool hands out beyond a request, as a fraction of it (`pool.fit_slack`)."""
    from nbxtensor.memory.pool import fit_slack
    return fit_slack(vendor, architecture)


# --- the allocator's figures and the pool's controls (R3, 2026-11-13): an application reads and steers them here,
# never through `DeviceAllocator` itself. Each answers from `memory.allocator.DeviceAllocator`, same arguments.

def _allocator():
    from nbxtensor.memory.allocator import DeviceAllocator
    return DeviceAllocator


def memory_allocated(device_idx=None) -> int:
    """Live bytes the allocator handed out on `device_idx` (None: every device)."""
    return _allocator().memory_allocated(device_idx)


def pool_cached_bytes(device_idx=None) -> int:
    """Bytes the pool keeps parked in free blocks on `device_idx` (None: every device); counted in
    `memory_allocated`."""
    return _allocator().pool_cached_bytes(device_idx)


def blocks_in_use(device_idx=None) -> dict:
    """{address: bytes} of every block a tensor holds on `device_idx` (None: every device), the pool's free blocks
    not in it."""
    return _allocator().blocks_in_use(device_idx)


def device_free_bytes(device_idx=None) -> int:
    """The driver's free bytes on `device_idx` (None: the current device), the pool's parked bytes included; a refusal
    raises `DeviceQueryError`."""
    return _allocator().device_free_bytes(device_idx)


def device_total_bytes(device_idx=None) -> int:
    """The device's total memory on `device_idx` (None: the current device); a refusal raises `DeviceQueryError`."""
    return _allocator().device_total_bytes(device_idx)


def free_memory_mb(index: int):
    """The driver's free memory on device `index`, in MB; a refusal raises `DeviceQueryError`."""
    return _allocator().free_memory_mb(index)


def peak_device_bytes():
    """The most one device held at once since the process started (live driver bytes, the pool's cache included);
    None where the backend keeps no such count."""
    return _allocator().peak_device_bytes()


def host_pinned_allocated() -> int:
    """Live bytes of pinned host memory the allocator handed out."""
    return _allocator().host_pinned_allocated()


def empty_cache_pool() -> int:
    """Every block the pool keeps parked given back to the driver; the bytes freed (0 with the pool off)."""
    return _allocator().empty_cache_pool()


def give_back_cached(dev: int) -> None:
    """After a card's work ends, its cached free memory back to the driver (the expandable range's free chunks and
    the pool's free blocks)."""
    _allocator().give_back_cached(dev)


def open_window(device_idx: int) -> int:
    """Start watching what this process holds from the driver on `device_idx`; a token for `close_window`."""
    return _allocator().open_window(device_idx)


def window_start(token: int):
    """What this process held from the driver when the window `token` opened; None for a token never opened."""
    return _allocator().window_start(token)


def close_window(device_idx: int, token: int):
    """The most this process held from the driver on `device_idx` since `open_window` gave `token`, the window
    closed; None for a token not open there."""
    return _allocator().close_window(device_idx, token)


def register_idle_reclaimer(reclaim) -> None:
    """`reclaim(dev) -> bytes` called when a malloc on `dev` is refused after the pool's flush, to release memory the
    application holds idle; registered once."""
    _allocator().register_idle_reclaimer(reclaim)


def weight_arenas():
    """A weight set's memory, to count, allocate and pin (`memory_pool.WeightArenas`)."""
    from nbxtensor.memory.memory_pool import WeightArenas
    return WeightArenas()
