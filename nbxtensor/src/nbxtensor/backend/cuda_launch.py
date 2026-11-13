"""A CUDA launch's refusals, its dimensions and its argument cells — shared by `CudaDriver.launch` (the live launch) and
`CudaDriver.prepare_launch` (a frozen plan's launch resolved once, `BoundCudaLaunch`)."""
from __future__ import annotations

import ctypes
from typing import Optional, Sequence, Tuple


def refuse(param_counts, function, params, names, trailing) -> None:
    """`param_counts()`: the driver's declared counts, read only after the trailing refusal. The live launch's refusals, BEFORE anything reaches the device (the launcher contract's ownership rules, checked
    by `verify_driver_contract`): no trailing buffer (CUDA declares none), the argument list exactly what the cubin
    declares, and every non-null pointer an address this engine's allocator handed out. A kernel launched past either
    fault reads garbage or foreign memory, and a CUDA fault is sticky for the whole context."""
    if trailing is not None:
        raise RuntimeError(
            "NeuroBrix launcher: trailing buffers were computed for this "
            f"kernel ({trailing!r}) and the CUDA driver cannot bind them. "
            "It declares none, so this is a driver/metadata mismatch, not "
            "something to launch past.")
    expected = param_counts().get(int(function.value))
    if expected is not None and len(params) != expected:
        raise TypeError(f"NeuroBrix launcher: the kernel declares {expected} parameters, "
                        f"{len(params)} given — refused, not launched")
    from nbxtensor.memory.allocator import DeviceAllocator
    for i, (kind, value) in enumerate(params):
        if kind == "ptr" and value and not DeviceAllocator.holds(int(value)):
            # The refusal names the parameter: "which buffer" is the whole question the caller then has to answer
            # (Ming's embedding, 2026-09-14, refused an address nobody could attribute from the address alone).
            what = (f"parameter '{names[i]}'" if names and i < len(names)
                    else f"parameter #{i}")
            raise ValueError(f"NeuroBrix launcher: device address {int(value):#x} ({what}) was not "
                             f"handed out by the allocator — refused, not launched")


def dims(grid, block) -> Optional[Tuple[int, ...]]:
    """(gx, gy, gz, bx, by, bz): a caller may give one or two extents (the contract's checker does), CUDA wants three;
    None for an empty grid (nothing to launch)."""
    gx, gy, gz = (tuple(int(g) for g in grid) + (1, 1, 1))[:3]
    bx, by, bz = (tuple(int(b) for b in block) + (1, 1, 1))[:3]
    return None if gx * gy * gz <= 0 else (gx, gy, gz, bx, by, bz)


def cells(kinds: Sequence[str]):
    """(one C cell per parameter kind, the pointer array over them): the argument buffer of a launch."""
    from nbxtensor.backend.launcher import ctypes_slot
    storage = [ctypes_slot(kind) for kind in kinds]
    return storage, (ctypes.c_void_p * len(storage))(*[ctypes.addressof(c) for c in storage])


class BoundCudaLaunch:
    """One CUDA launch of a frozen plan, its argument cells its own (filled once): `run(stream)` is one cuLaunchKernel."""

    __slots__ = ("launch", "check", "function", "dims", "shared", "cells", "arr")

    def __init__(self, launch, check, function, launch_dims, shared: int, storage, arr):
        self.launch, self.check, self.function, self.dims = launch, check, function, launch_dims
        self.shared, self.cells, self.arr = shared, storage, arr        # the cells kept alive: arr points into them

    def run(self, stream: int) -> None:
        rc = self.launch(self.function, *self.dims, self.shared, stream, self.arr, None)
        if rc != 0:
            self.check(rc, "cuLaunchKernel")


def prepare(launch, check, param_counts, function, grid, block, shared: int, params, trailing):
    """The launch resolved once (`BoundCudaLaunch`), or None where the live launch would refuse (it then refuses by
    name, live) or launch nothing."""
    try:
        refuse(param_counts, function, params, None, trailing)
    except (RuntimeError, TypeError, ValueError):
        return None
    launch_dims = dims(grid, block)
    if launch_dims is None:
        return None
    storage, arr = cells(tuple(kind for kind, _v in params))
    for cell, (_kind, value) in zip(storage, params):
        cell.value = value
    return BoundCudaLaunch(launch, check, function, launch_dims, int(shared or 0), storage, arr)
