"""Memory: slot storage, and (as the move proceeds) allocators, pools and transfers."""


def set_limit(nbytes, device=None) -> None:
    """Hold this process to a card of `nbytes` bytes on `device` (an ordinal; None: every device), or lift the limit
    (`nbytes` None): the allocator refuses what such a card refuses and its pool parks a fraction of the limit
    (`DeviceAllocator.set_limit`). Optional; absent, the physical card is the limit."""
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.set_limit(nbytes, device)
