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
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

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


_SYNC_TIMING = __import__("os").environ.get("NBX_SYNC_TIMING", "0") == "1"
_SYNC_ACC = [0.0, 0.0, 0.0, 0]   # drain, submit, synchronize, count
if _SYNC_TIMING:                                       # pragma: no cover
    import atexit as _ax

    @_ax.register
    def _report_sync():
        d, s_, y, n = _SYNC_ACC
        if not n:
            return
        print(f"[sync-timing] launches={n}  drain={d:.3f}s  submit={s_:.3f}s  "
              f"synchronize={y:.3f}s  (per launch: drain {1000*d/n:.3f}ms  "
              f"submit {1000*s_/n:.3f}ms  sync {1000*y/n:.3f}ms)", flush=True)


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
        from neurobrix.triton.metal_backend import metal_target
        return metal_target()

    def max_shared_memory_per_block(self) -> int:
        """triton-ext's own threadgroup budget, by the name it actually uses.

        This read `getattr(hw_constants, "MAX_THREADGROUP_MEMORY", 32768)` until
        2026-09-17. **That name does not exist in triton-ext** — the module
        declares `TG_BUDGET_BYTES`, `WARP_SIZE`, `SG_FRAG_DIM`, `TARGET`,
        `target_arch` and nothing else — so every call took the literal, and the
        launcher pruned configurations against a number no backend had said.
        The same class as a literal standing in for a runtime value, in the one
        shape that hides best: a `getattr` default that is never not taken.

        It was right by luck. Measured the day it was found, M4 Pro:

            hw_constants.TG_BUDGET_BYTES          32768
            MTLDevice.maxThreadgroupMemoryLength  32768

        so the number does not move; only the authority for it does. The
        backend binds the same constant into its own launch metadata
        (`driver.py:238`, `"max_shared_mem": _TG_BUDGET_BYTES`), which is what
        makes it the right thing to ask.

        A backend that stops declaring it refuses here, and
        `max_shared_memory_per_block()` in the launcher turns that into None —
        "do not prune" — which is the safe answer. Inventing one is not.
        """
        from triton_apple_backend import hw_constants
        budget = getattr(hw_constants, "TG_BUDGET_BYTES", None)
        if budget is None:
            raise RuntimeError(
                "triton_apple_backend.hw_constants declares no TG_BUDGET_BYTES; "
                "the threadgroup-memory ceiling is the backend's to state and "
                "this driver will not invent one. Declared names: "
                f"{sorted(n for n in dir(hw_constants) if not n.startswith('_'))}")
        return int(budget)

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
    def trailing_buffers(self, metadata):
        """What triton-ext's emitter appends to a kernel's own parameters.

        Its compiler writes two descriptor blocks into the emitted MSL and
        carries them in the metadata (`compiler.py`: `metadata["print_layout"]`,
        `metadata["assert_layout"]`, from `AGPU-PRINT-LAYOUT` / `AGPU-ASSERT-LAYOUT`).
        When a block is present the kernel has ONE MORE buffer parameter than
        its Triton signature declares, and triton-ext's own driver binds it
        last, print before assert.

        We bound neither until 2026-09-17, and `metal_native.packArguments`
        binds the tuple it is handed at indices 0..n-1 without ever comparing
        the count to what the kernel declares — so the slot simply stayed
        unbound. Measured that day (`probe_ki19_driver_protections.py`): a
        `tl.device_assert` that FAILS on every lane ran to completion and raised
        nothing, while the same kernel's ordinary output was correct. A lost
        assert is the worst shape of that: the guard is paid for and not
        delivered, which is the same fault class as the conv-1x1 zeros.

        Four of NeuroBrix's own kernels are compiled `@triton.jit(debug=True)`
        precisely to keep such an assert (`embedding`, `index_select` x2,
        `index_put`), so this is not a hypothetical parameter.

        Returns None when the kernel declares neither block — the common case,
        and then `launch` does exactly what it did before.
        """
        from triton_apple_backend.device_assert import parse_assert_layout
        from triton_apple_backend.device_print import parse_print_layout
        pl = parse_print_layout(getattr(metadata, "print_layout", None))
        al = parse_assert_layout(getattr(metadata, "assert_layout", None))
        if pl is None and al is None:
            return None
        return _Trailing(pl, al)

    def launch(self, function, grid, block, shared: int, stream: int,
               params: Sequence[Tuple[str, Any]], names=None, types=None,
               trailing=None) -> None:
        if types is None:
            raise RuntimeError(
                "the triton-ext driver needs the Triton type of every launch "
                "parameter: its scalars share one packed buffer whose field "
                "offsets come from those types. Launching without them would "
                "pack to the wrong offsets WITHOUT failing, which is the exact "
                "silent-zero this driver exists to end.")

        # The argument list must be exactly what the kernel declares. `zip`
        # below stops at the shorter of the two, so a short list would bind
        # what it could and launch — silently, with the tail unbound. CUDA has
        # refused this since the launcher existed (`CudaDriver.launch` compares
        # against the cubin's own count); this driver did not, and the
        # launcher contract caught it the first time it was run against a
        # driver that is not the archived fork's (2026-09-17).
        if len(params) != len(types):
            raise RuntimeError(
                f"NeuroBrix triton-ext driver: {len(params)} launch parameters "
                f"against {len(types)} declared types — refused, not launched. "
                f"Binding the shorter of the two leaves the rest unbound and "
                f"says nothing.")

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
        # The emitter's own trailing parameters, bound LAST and in its order
        # (print, then assert), which is what `planKernelAbi` fixed. Both must
        # start zeroed: each block's head word is a running count the kernel
        # bumps.
        print_buf = assert_buf = None
        if trailing is not None:
            rt = D._runtime()      # D is the pinned torch-free runtime's module
            if trailing.print_layout is not None:
                print_buf = rt.zeros_i32(trailing.print_layout.nbytes // 4)
                args = args + (print_buf,)
            if trailing.assert_layout is not None:
                assert_buf = rt.zeros_i32(trailing.assert_layout.nbytes // 4)
                args = args + (assert_buf,)

        # Two queues, one device, shared buffers. Metal orders command buffers
        # within a queue; ACROSS queues nothing is ordered without an explicit
        # event (Apple: a fence cannot synchronize untracked resources accessed
        # from separate queues; MTLEvent/MTLSharedEvent is the mechanism).
        #
        # NeuroBrix enqueues its device-to-device copies as blits on ITS queue
        # and returns to the host immediately — correct while the kernels shared
        # that queue, which is what `metal_device._blit` says in as many words:
        # "a blit on the same queue as the kernels is ordered by the GPU and
        # waits for nothing". triton-ext breaks that premise: its kernels run on
        # its own queue, so a blit that has not landed is invisible to them.
        #
        # Measured on swin2SR (M4 Pro, 2026-09-17): without this drain, the conv
        # at (1,180,448,448)->64 read 63181 NaN / 356 Inf / absmax 3.39e+38 from
        # an input that is `absmax 1.492` and finite in every ordered run, and
        # the fp64 screen refused all 18 candidates. A host round trip inserted
        # before the launch made the same run clean, which is what identified
        # the ordering rather than the arithmetic.
        if _SYNC_TIMING:
            import time as _t
            _a = _t.perf_counter(); _nbx_queue_drain()
            _b = _t.perf_counter()
            function(*args, threads=[gx * lx, gy * ly, gz * lz], group_size=[lx, ly, lz])
            _c = _t.perf_counter(); _native().synchronize(); _d = _t.perf_counter()
            _SYNC_ACC[0] += _b - _a; _SYNC_ACC[1] += _c - _b
            _SYNC_ACC[2] += _d - _c; _SYNC_ACC[3] += 1
        else:
            _nbx_queue_drain()
            function(*args, threads=[gx * lx, gy * ly, gz * lz], group_size=[lx, ly, lz])
            # And the other direction: the host (and NeuroBrix's own blits) must
            # see what this kernel wrote.
            _native().synchronize()

        # Read what the kernel recorded. Prints first, so anything it printed is
        # already out when a failed assert raises.
        if print_buf is not None:
            from triton_apple_backend.device_print import format_records
            rt = D._runtime()
            for line in format_records(trailing.print_layout, rt.as_u32(print_buf)):
                print(line, flush=True)
        if assert_buf is not None:
            from triton_apple_backend.device_assert import check as _check_asserts
            rt = D._runtime()
            _check_asserts(trailing.assert_layout, rt.as_u32(assert_buf))


class _Trailing(NamedTuple):
    """The emitter-declared buffers that follow a kernel's own parameters.

    Held as the PARSED layouts rather than the raw text: parsing is a regex
    walk over the whole emitted module, and it is a property of the
    compilation, so it happens once in `trailing_buffers` and never on the
    launch path.
    """
    print_layout: Any
    assert_layout: Any


#: Binding census, printed at exit under NBX_EXT_STATS=1. An interior binding
#: takes its LENGTH from the allocator's range table rather than from an exact
#: pointer hit, which is the one place a stale entry could hand back a buffer
#: shorter than the tensor.
_STATS = {"exact": 0, "interior": 0}

import os as _os_stats
if _os_stats.environ.get("NBX_EXT_STATS"):
    import atexit as _atexit
    _atexit.register(lambda: print(
        f"[ext-stats] pointer bindings: exact={_STATS['exact']} "
        f"interior={_STATS['interior']}", flush=True))

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
        _STATS["interior"] += 1
    else:
        _STATS["exact"] += 1
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
    # Floor the span to a whole number of elements. The allocation runs to the
    # end of its block, which need not be a multiple of this tensor's element
    # size — and `np.frombuffer` refuses a remainder with "buffer size must be a
    # multiple of element size". Measured on Kokoro-82M, aten.upsample_linear1d
    # at (1, 9, 76800). Flooring is safe: the kernel addresses its own tensor,
    # which ends at or before the block's end, so no element it reads or writes
    # is cut off.
    itemsize = np.dtype(np_dtype).itemsize
    span = ((size - offset) // itemsize) * itemsize
    if span <= 0:
        raise RuntimeError(
            f"the triton-ext driver cannot bind pointer 0x{addr:x}: the "
            f"allocation leaves {size - offset} bytes from it, less than one "
            f"{np_dtype} element")
    raw = (ctypes.c_byte * span).from_address(addr)
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


def _nbx_queue_drain():
    """Wait for NeuroBrix's own Metal queue before a foreign queue reads its
    buffers. Coarse — a host wait — but correct; an MTLSharedEvent between the
    two queues is the finer instrument and is owed if this costs measurably."""
    from neurobrix.kernels.metal_device import runtime
    runtime().sync()

