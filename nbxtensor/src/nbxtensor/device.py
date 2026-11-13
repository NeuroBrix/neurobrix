"""The devices: how many the process sees, the one it works on, the one the next launch runs on, when they drain, and
what each has free (docs/API.md section 5). The application's door to them: each name answers from the module that
holds the mechanism (`memory.allocator.DeviceAllocator` and the runtime table it reads), so an application never
reaches into the allocator for a device question.

Indexes are the runtime's ordinals (0, 1, ...). A refusal of the runtime is raised with its own message, never
replaced by a default.
"""
from typing import Optional, Tuple

__all__ = ["count", "current", "select", "select_for_launch", "synchronize", "drain", "mem_info", "most_free",
           "target_name", "DeviceQueryError", "NoGpuRuntime", "install_shadow", "DevicesVisible", "declare_target",
           "TargetNotDeclarable"]


def __getattr__(name):
    """`DeviceQueryError` (the refusal a device question raises) and `NoGpuRuntime` (no GPU runtime on this host),
    read from their modules only when asked: importing the door loads no device runtime."""
    if name == "DeviceQueryError":
        from nbxtensor.memory.allocator import DeviceQueryError
        return DeviceQueryError
    if name == "NoGpuRuntime":                # a host with no GPU runtime: the one refusal that means "no device"
        from nbxtensor.backend.runtime import NoGpuRuntime
        return NoGpuRuntime
    if name == "DevicesVisible":              # `install_shadow` asked with a device in sight
        from nbxtensor.memory.shadow import DevicesVisible
        return DevicesVisible
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


def target_name() -> str:
    """The name of what this process compiles for, as its measured and certified settings are keyed: the backend and
    the architecture, `<backend>-<arch>` ("cuda-70", "hip-gfx90a", "metal-apple-m4-pro"), read from the launcher's
    compile target (`backend.launcher.target`, `arch_name`). A driver that cannot answer raises its own error."""
    from nbxtensor.backend.launcher import arch_name, target
    t = target()
    return f"{t.backend}-{arch_name(t)}"


def install_shadow(devices: int) -> None:
    """Turn this process into a shadow of `devices` devices, once (a second call changes nothing): no device memory,
    no launch, no value, so an application can run its kernels' dispatch to learn every launch key it forms without a
    card (`memory.shadow.install_shadow`). From then on `count()` answers `devices` and `current()` the device
    `select` named last (0 before any); allocations hand out addresses that address nothing.

    Refused with `DevicesVisible` while the runtime sees a device (with one in sight, a path the shadow does not cover
    would succeed on real hardware in silence), and with `ValueError` for a count below one. Install before the
    kernels are defined."""
    if int(devices) < 1:
        raise ValueError(f"install_shadow({devices!r}): a shadow stands for at least one device; name how many the "
                         f"hardware it stands for has")
    from nbxtensor.memory.shadow import install_shadow as install
    install(int(devices))


class TargetNotDeclarable(ValueError):
    """`declare_target` was given a maker whose compile target is not formed from a capability number, or a capability
    that is not one ("major" or "major.minor")."""


# maker -> (compile backend, threads per warp) of a target formed from a capability number
_TARGETS = {"nvidia": ("cuda", 32), "amd": ("hip", 64)}


def declare_target(maker: str, capability: str) -> None:
    """Bind this process's compile target without asking a driver, from what a hardware description states: `maker`
    "nvidia" (backend cuda, 32 threads a warp) or "amd" (hip, 64), `capability` "major.minor" ("7.0" -> 70; "8" ->
    80). `backend.launcher.target()` answers it from then on, and the device description in force is read again
    under it (`backend.device_files.forget`): a derivation that compiles and launches nothing forms the keys the
    launcher forms on that device.

    Refused with `TargetNotDeclarable`, naming what was given, for any other maker (a device whose architecture is a
    name, Apple or Intel, keeps its driver's answer) or a capability that is not digits with at most one dot."""
    m = str(maker).strip().lower()
    cc = str(capability).strip()
    parts = cc.split(".")
    if m not in _TARGETS or not 1 <= len(parts) <= 2 or not all(p.isdigit() for p in parts):
        raise TargetNotDeclarable(f"declare_target({maker!r}, {capability!r}): a target is declared for "
                                  f"{sorted(_TARGETS)} with a capability 'major' or 'major.minor'")
    from triton.backends.compiler import GPUTarget
    from nbxtensor.backend import device_files, launcher
    backend, warp = _TARGETS[m]
    minor = int(parts[1]) if len(parts) == 2 else 0
    launcher.bind_target(GPUTarget(backend, int(parts[0]) * 10 + minor, warp))
    device_files.forget()     # a description resolved before the bind is read again
