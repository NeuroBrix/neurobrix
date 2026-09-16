"""NeuroBrix's launcher driver for triton-lang/triton-ext's AppleGPU backend.

Why this file exists, in one sentence: a driver implements ONE backend's launch
ABI, and triton-ext's is not the fork's.

The fork binds each scalar to its own Metal argument slot. triton-ext packs every
scalar into a SINGLE device buffer and binds it after the pointers — its own
`MetalLauncher` says so:

    # Scalars pack into one device buffer, so IR param order is
    # [pointers, packed_scalar_buf, system_values].

Launching one backend's kernel through the other's driver therefore lands the
first scalar and zeroes the rest, with nothing raised: every mask goes false,
every `tl.load` takes its `other`, and the `tl.store` is fully masked, so the
output buffer keeps the zeros it was allocated with. That is the exact-zero
signature measured on conv2d 1x1 across all 18 configs (see the campaign ledger,
`abi_scalars.py`: the kernel saw `HW,C,K = [4096, 0, 0]` for `[4096, 180, 180]`).

Two rules this file keeps:

* **We do not re-derive their ABI.** The scalar layout comes from triton-ext's own
  `_compute_scalar_layout`/`_SCALAR_PACK_INFO`, so a change on their side is a
  change here, not a silent divergence. We follow the line of Triton.
* **No torch.** triton-ext ships two dispatch runtimes and prefers the libtorch
  one whenever torch is merely installed; R33 forbids that, so this driver pins
  the torch-free `_NativeRuntime`. Upstream has no supported switch for it —
  that is an owed upstream contribution, with `spike_native_runtime.py` as its
  reproducer.
"""
from __future__ import annotations

import ctypes
import struct
from typing import Any, List, Optional, Sequence, Tuple

from neurobrix.kernels.launcher import Driver


def _ext():
    """triton-ext's driver module, with the torch-free runtime pinned.

    `_runtime()` chooses `_TorchRuntime` whenever torch can be imported and
    offers no override, so the choice is made here before anything builds one.
    """
    from triton_apple_backend import driver as D
    if not isinstance(getattr(D, "_RUNTIME", None), D._NativeRuntime):
        D._RUNTIME = D._NativeRuntime()
    return D


def _native():
    from triton_apple_backend import metal_native
    return metal_native


#: Our `_pack_param` kinds -> the width, in bytes, of the value it produced.
#: A float arrives as the bit pattern of its storage type and an int as itself,
#: so both are written as an unsigned field of this width. Re-packing through
#: `struct.pack('<f', ...)` would treat a bit pattern as a number.
_KIND_WIDTH = {
    "bits16": 2, "bits32": 4, "bits64": 8,
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "u1": 1, "u8": 1, "u16": 2, "u32": 4, "u64": 8,
}


class TritonExtDriver(Driver):
    """Loads a metallib and launches it the way triton-ext's own launcher does."""

    #: triton-ext emits a compiled `.metallib`, not text MSL (`compiler.py`
    #: stages: "msl" then "metallib"). The fork emits the text.
    artifact_kind = "metallib"

    #: Their launcher builds its argument mask from the kernel SIGNATURE, so two
    #: extra scratch pointers would trip its own arity check:
    #: "flat arg count does not match signature keep-mask length".
    wants_scratch_params = False

    _instance: Optional["TritonExtDriver"] = None

    @classmethod
    def instance(cls) -> "TritonExtDriver":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- compile target ----------------------------------------------------
    def target(self):
        # The same target the fork's driver builds: the seam owns the NAME
        # ("metal" for the fork, "mps" for triton-ext) and the device owns the
        # arch. Building it here again would be a second answer to one question.
        from neurobrix.triton.metal_driver import metal_target
        return metal_target()

    def max_shared_memory_per_block(self) -> int:
        from triton_apple_backend import hw_constants
        return int(getattr(hw_constants, "MAX_THREADGROUP_MEMORY", 32768))

    def block_for(self, metadata):
        """Threads per threadgroup, as their launcher computes it: `lx` is
        `num_warps * 32`, and ly/lz are 1 — Metal has no second launch axis
        here."""
        return (32 * int(getattr(metadata, "num_warps", 4)), 1, 1)

    # -- load --------------------------------------------------------------
    def load(self, binary: bytes, name: str, shared: int):
        module = _native().load_metallib(bytes(binary))
        fn = module.get_function(name)
        import os as _os
        if _os.environ.get("NBX_EXT_DEBUG"):
            print(f"[ext] load name={name!r} metallib={len(bytes(binary))}B "
                  f"fn={fn} max_threads="
                  f"{getattr(fn, 'max_total_threads_per_threadgroup', None)}", flush=True)
        return fn

    # -- launch ------------------------------------------------------------
    def launch(self, function, grid, block, shared: int, stream: int,
               params: Sequence[Tuple[str, Any]], names=None, types=None) -> None:
        if types is None:
            raise RuntimeError(
                "the triton-ext driver needs the Triton type of every launch "
                "parameter: its scalars share one packed buffer whose field "
                "offsets come from those types. Launching without them would "
                "pack to the wrong offsets WITHOUT failing, which is the exact "
                "silent-zero this driver exists to end.")

        D = _ext()
        ptr_args: List[Any] = []
        scalar_types: List[str] = []
        scalar_vals: List[Tuple[str, int]] = []

        for (kind, value), ty in zip(params, types):
            if kind == "ptr":
                ptr_args.append(_buffer_for(int(value), ty))
            else:
                scalar_types.append(ty)
                scalar_vals.append((kind, int(value)))

        args: Tuple[Any, ...] = tuple(ptr_args)
        if scalar_types:
            total, offsets = D._compute_scalar_layout(scalar_types)
            limit = getattr(D, "_SETBYTES_LIMIT", 4096)
            if total > limit:
                raise RuntimeError(
                    f"packed scalar args are {total} bytes, over Metal's "
                    f"{limit}-byte setBytes limit")
            buf = bytearray(total)
            for (kind, val), ty, off in zip(scalar_vals, scalar_types, offsets):
                width = _KIND_WIDTH.get(kind)
                if width is None:
                    raise RuntimeError(
                        f"the triton-ext driver cannot pack a parameter of kind "
                        f"{kind!r} (Triton type {ty!r})")
                fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width]
                struct.pack_into(fmt, buf, off, val & ((1 << (8 * width)) - 1))
            args = args + (bytes(buf),)

        gx, gy, gz = grid
        lx, ly, lz = block
        import os as _os
        if _os.environ.get("NBX_EXT_DEBUG"):
            print(f"[ext] ptrs={len(ptr_args)} scalars={list(zip(scalar_types, scalar_vals))} "
                  f"packed={(args[-1].hex() if scalar_types else None)} "
                  f"threads={[gx*lx, gy*ly, gz*lz]} group_size={[lx, ly, lz]} "
                  f"grid={grid} block={block} shared={shared}", flush=True)
        function(*args, threads=[gx * lx, gy * ly, gz * lz], group_size=[lx, ly, lz])
        # triton-ext dispatches on ITS OWN command queue, not the one NeuroBrix's
        # allocator orders its copies against, so nothing makes a host read see
        # these writes. Measured: without this the kernel computes correctly and
        # the reader still sees the buffer's initial zeros.
        _native().synchronize()


#: Metal's zero-copy wrap requires page-aligned memory; Apple Silicon pages
#: are 16 KiB.
_PAGE = 16384

#: Triton pointer type -> numpy dtype. `wrap` reads the element type off the
#: object it is given; a bare memoryview arrives as dtype None, which is not a
#: buffer triton-ext can bind.
_PTR_NP = {
    "fp16": "float16", "bf16": "uint16", "fp32": "float32", "fp64": "float64",
    "i1": "bool", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
    "fp8e4nv": "uint8", "fp8e5": "uint8", "fp8e4b15": "uint8",
}


def _buffer_for(addr: int, ty: str):
    """Alias the NBXTensor allocation containing `addr` as a MetalBuffer.

    The launcher reduces a pointer parameter to its integer address, so the
    length has to come from the allocator, which already records it. A tensor
    VIEW points into the middle of an allocation, so the base is looked up
    before the size.

    `ty` is the Triton pointer type ("*fp32"): the element type has to travel
    with the address, because `wrap` takes it from the object it is handed and a
    typeless buffer is not bindable.
    """
    import numpy as np

    elem = ty.lstrip("*k") if ty else ""
    np_dtype = _PTR_NP.get(elem)
    if np_dtype is None:
        raise RuntimeError(
            f"the triton-ext driver has no numpy element type for pointer "
            f"parameter of type {ty!r}; binding it typeless would not fail, it "
            f"would bind a buffer the kernel reads as zeros")
    if addr == 0:
        raise RuntimeError(
            "the triton-ext driver was handed a null pointer parameter; its "
            "launcher has no slot for one")
    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    size = DeviceAllocator._cuda_ptr_size.get(addr)
    base = addr
    if size is None:
        base, size = _containing_allocation(addr)
    # An interior pointer — a view, or a tensor packed inside a larger buffer —
    # binds by starting the Metal buffer AT that address and running to the end
    # of the allocation. Binding the allocation's base instead would read the
    # wrong elements silently, and `metal_native` has no offset binding
    # (`metal_native.m`: every argument goes in with `setBuffer:...offset:0`).
    #
    # Metal decides whether a given address can be wrapped with no copy. The
    # documented requirement is page alignment, but MEASURED on M4 Pro a
    # non-page-aligned interior address inside an already-mapped region is
    # accepted (repro_ext_no_offset_binding.py, case 3: offset 0x300, wrapped,
    # 64768 bytes). So we do not pre-judge it: we ask, and turn Metal's own
    # refusal into a named one. Guessing a binding is what this driver exists
    # to stop; refusing a binding that would have worked is merely wrong.
    offset = addr - base
    raw = (ctypes.c_byte * (size - offset)).from_address(addr)
    view = np.frombuffer(memoryview(raw), dtype=np_dtype)
    try:
        return _native().wrap(view)
    except Exception as exc:
        raise RuntimeError(
            f"the triton-ext driver could not bind pointer 0x{addr:x} "
            f"(0x{offset:x} into an allocation at 0x{base:x}, {size - offset} "
            f"bytes): Metal declined a zero-copy view of it. `metal_native` "
            f"binds every buffer at offset 0, so there is no other correct "
            f"binding here — binding the allocation base would read the wrong "
            f"elements WITHOUT failing. Metal said: {exc}") from exc


def _containing_allocation(addr: int) -> Tuple[int, int]:
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    for base, size in DeviceAllocator._range_size.items():
        if base <= addr < base + size:
            return base, size
    raise RuntimeError(
        f"the triton-ext driver cannot bind pointer 0x{addr:x}: the allocator "
        f"does not record it, so its length is unknown and it cannot be wrapped "
        f"as a Metal buffer")


def driver() -> TritonExtDriver:
    return TritonExtDriver.instance()
