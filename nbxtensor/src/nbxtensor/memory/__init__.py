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
    """Whether the allocator parks freed blocks (`pool.enabled`)."""
    from nbxtensor.memory.pool import enabled
    return enabled()


def pool_parked_cap_bytes(vendor: str, architecture: str, total_bytes: float) -> int:
    """The bytes the pool may keep parked on such a device, 0 unbounded (`pool.parked_cap_bytes`)."""
    from nbxtensor.memory.pool import parked_cap_bytes
    return parked_cap_bytes(vendor, architecture, total_bytes)


def pool_fit_slack(vendor: str, architecture: str) -> float:
    """The most the pool hands out beyond a request, as a fraction of it (`pool.fit_slack`)."""
    from nbxtensor.memory.pool import fit_slack
    return fit_slack(vendor, architecture)
