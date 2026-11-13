"""The devices: how many the process sees, the one it works on, the one the next launch runs on, when they drain, and
what each has free (docs/API.md section 5). The application's door to them: each name answers from the module that
holds the mechanism (`memory.allocator.DeviceAllocator` and the runtime table it reads), so an application never
reaches into the allocator for a device question.

Indexes are the runtime's ordinals (0, 1, ...). A refusal of the runtime is raised with its own message, never
replaced by a default.
"""
from typing import Optional, Tuple

__all__ = ["count", "current", "select", "select_for_launch", "synchronize", "drain", "mem_info", "most_free",
           "DeviceQueryError", "NoGpuRuntime"]


def __getattr__(name):
    """`DeviceQueryError` (the refusal a device question raises) and `NoGpuRuntime` (no GPU runtime on this host),
    read from their modules only when asked: importing the door loads no device runtime."""
    if name == "DeviceQueryError":
        from nbxtensor.memory.allocator import DeviceQueryError
        return DeviceQueryError
    if name == "NoGpuRuntime":                # a host with no GPU runtime: the one refusal that means "no device"
        from nbxtensor.backend.runtime import NoGpuRuntime
        return NoGpuRuntime
    raise AttributeError(f"module 'nbxtensor.device' has no attribute {name!r}")


def _allocator():
    from nbxtensor.memory.allocator import DeviceAllocator
    return DeviceAllocator


def count() -> int:
    """How many devices this process can see (`DeviceAllocator.device_count`): 0 on a host with no GPU runtime or when
    the runtime says it has no device; any other refusal of the runtime raises `DeviceQueryError` with its code."""
    return _allocator().device_count()


def current() -> int:
    """The device the runtime works on now (`DeviceAllocator.get_device`)."""
    return _allocator().get_device()


def select(index: int) -> None:
    """Make `index` the runtime's current device for what follows: allocations, copies, synchronizations. An ordinal
    the process cannot see is refused by name (`DeviceAllocator.set_device`). The kernel launcher's own context is
    not touched: `select_for_launch` does that."""
    _allocator().set_device(index)


def select_for_launch(index: int) -> None:
    """Make `index` current for the next kernel launch too (the runtime and the launcher's context; a no-op when it
    already is: `DeviceAllocator.ensure_triton_device`)."""
    _allocator().ensure_triton_device(index)


def synchronize() -> None:
    """Wait until the current device has finished all it was given, through the interceptor in force if one is
    installed (`DeviceAllocator.sync_device`); a device's earlier failure surfaces here, raised."""
    _allocator().sync_device()


def drain(index: Optional[int] = None) -> None:
    """Wait until every stream of device `index` (None: the current one) has drained; the current device is restored
    after (`DeviceAllocator.device_synchronize`)."""
    _allocator().device_synchronize(index)


def mem_info(index: Optional[int] = None) -> Tuple[int, int]:
    """(free, total) bytes of device `index` (None: the current one), as the runtime reports them
    (`DeviceAllocator.mem_get_info`). On Metal, total is the recommended working set: never read total - free as
    the process's footprint."""
    return _allocator().mem_get_info(index)


def most_free() -> int:
    """The device with the most free memory as the runtime reports it; the current device is restored after
    (`DeviceAllocator.most_free_device`). A device the runtime refuses to describe, or a failed restore, raises
    `DeviceQueryError` naming it."""
    return _allocator().most_free_device()
