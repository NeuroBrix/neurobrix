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
* **We run what they compile.** triton-ext compiles the kernel to a metallib; nbxtensor builds its pipeline and
  encodes every launch on its own Metal stream (`metal_dispatch`): one serial compute encoder, so consecutive kernels
  are ordered by Metal and the host waits only where it reads. triton-ext's runtime (a command buffer and a fresh
  wrap per launch, on a queue of its own) is not used: Metal ran two of its launches overlapped, so it had to wait
  after every one (131 493 waits, 19.58 s of a 30.36 s TinyLlama run, the Mac, 2026-11-12).
"""
from __future__ import annotations

import struct
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

from nbxtensor.backend.launcher import Driver
import logging
_log = logging.getLogger(__name__)


def _ext():
    """triton-ext's driver module: the authority for the packed-scalar layout (`_compute_scalar_layout`)."""
    from triton_apple_backend import driver as D
    return D


_SYNC_TIMING = __import__("os").environ.get("NBX_SYNC_TIMING", "0") == "1"
_SYNC_ACC = [0.0, 0]             # seconds encoding launches, launches
_SYNC_PER_KERNEL = {}            # name -> [launches, seconds encoding them]


def kernel_name(function) -> str:
    """The name a Metal kernel was compiled under (`metal_dispatch.Pipeline.name`): the counter's rows."""
    return getattr(function, "name", None) or type(function).__name__


if _SYNC_TIMING:                                       # pragma: no cover
    import atexit as _ax

    @_ax.register
    def _report_sync():
        e, n = _SYNC_ACC
        if not n:
            return
        _log.debug(f"[sync-timing] launches={n}  encode={e:.3f}s ({1000*e/n:.4f} ms a launch); the host now waits "
                   f"only where it reads (metal_device flush), never after a launch")
        rows = sorted(_SYNC_PER_KERNEL.items(), key=lambda kv: -kv[1][0])
        _log.debug(f"[sync-timing]   {'kernel':<44} {'launches':>8} {'encode s':>9}")
        for name, (c, t) in rows[:18]:
            _log.debug(f"[sync-timing]   {name[:44]:<44} {c:>8} {t:>9.3f}")


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

    #: Pointers bind as (buffer, offset) of their allocation (`metal_dispatch`): no wrap, so no extent is needed.
    wants_extents = False

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
        from nbxtensor.backend.metal_backend import metal_target
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
        """Our pipeline of the compiled kernel (`metal_dispatch.pipeline`): triton-ext compiles, nbxtensor runs."""
        from nbxtensor.backend.metal_dispatch import pipeline
        return pipeline(bytes(binary), name)

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
        # The two address flags triton-ext's compiler records (`compiler.py`:
        # `metadata["exposes_addresses"]`, `metadata["reads_addresses"]`) and its
        # own launcher passes to every call: a kernel that stores a pointer's
        # bits exposes its buffers, and one that reads through such bits must
        # name the exposed buffers to Metal, which keeps resident only what the
        # encoder names. We passed neither until 2026-10-04.
        exposes = bool(getattr(metadata, "exposes_addresses", False))
        reads = bool(getattr(metadata, "reads_addresses", False))
        # A kernel with a barrier across threadgroups needs its whole grid co-resident: Metal has no cooperative
        # launch and does not preempt a spinning threadgroup (triton-ext's compiler, `cross_tg_barrier`).
        barrier = bool(getattr(metadata, "cross_tg_barrier", False))
        if pl is None and al is None and not exposes and not reads and not barrier:
            return None
        return _Trailing(pl, al, exposes, reads, barrier)

    def launch(self, function, grid, block, shared: int, stream: int,
               params: Sequence[Tuple[str, Any]], names=None, types=None,
               trailing=None, extents=None) -> None:
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

        pointers, packed = _pointers_and_scalars(function, params, types)
        # The emitter's own trailing parameters, bound LAST and in its order (print, then assert), each a zeroed
        # block whose head word is a running count the kernel bumps: triton-ext's `planKernelAbi`.
        from nbxtensor.backend import metal_dispatch as MD
        MD.raise_landed_asserts()                       # an earlier launch's failure the host already sees
        printed = _zeroed_block(trailing.print_layout.nbytes) if trailing and trailing.print_layout else None
        asserted = MD.assert_block(function, trailing.assert_layout) if trailing and trailing.assert_layout else None
        t0 = _now()
        MD.dispatch(function, stream, pointers, packed, [t.data_ptr() for t in (printed, asserted) if t is not None],
                    grid, block, exposes=bool(trailing and trailing.exposes_addresses),
                    reads=bool(trailing and trailing.reads_addresses), shared=int(shared or 0),
                    barrier=bool(trailing and trailing.cross_tg_barrier))
        if _SYNC_TIMING:
            spent = _now() - t0
            _SYNC_ACC[0] += spent; _SYNC_ACC[1] += 1
            row = _SYNC_PER_KERNEL.setdefault(kernel_name(function), [0, 0.0])
            row[0] += 1; row[1] += spent
        if printed is not None:
            _print_records(trailing.print_layout, printed)

    def prepare_launch(self, function, grid, block, shared: int, params: Sequence[Tuple[str, Any]], names=None,
                       types=None, trailing=None, extents=None):
        """The launch resolved once for a frozen plan's replay (`metal_dispatch.bind`), or None where it must stay live:
        a kernel with trailing print / assert blocks (one block a launch, read after it). The same refusals as
        `launch`, made here once."""
        if types is None or len(params) != len(types):
            return None                                 # `launch` refuses these by name, at the live launch
        if trailing and (trailing.print_layout or trailing.assert_layout):
            return None
        from nbxtensor.backend import metal_dispatch as MD
        pointers, packed = _pointers_and_scalars(function, params, types)
        return MD.bind(function, pointers, packed, grid, block, exposes=bool(trailing and trailing.exposes_addresses),
                       reads=bool(trailing and trailing.reads_addresses), shared=int(shared or 0),
                       barrier=bool(trailing and trailing.cross_tg_barrier))


def _pointers_and_scalars(function, params, types):
    """(the pointer arguments in order, the scalars packed in one block at triton-ext's offsets or None)."""
    D = _ext()
    pointers: List[int] = []
    scalar_types: List[str] = []
    scalar_vals: List[Tuple[str, int]] = []
    for (kind, value), ty in zip(params, types):
        if kind == "ptr":
            if not int(value):
                raise RuntimeError(f"kernel {kernel_name(function)!r} was handed a null pointer parameter")
            pointers.append(int(value))
        else:
            scalar_types.append(ty)
            scalar_vals.append((kind, int(value)))
    if not scalar_types:
        return pointers, None
    total, offsets = D._compute_scalar_layout(scalar_types)
    limit = getattr(D, "_SETBYTES_LIMIT", 4096)
    if total > limit:
        raise RuntimeError(f"packed scalar args are {total} bytes, over Metal's {limit}-byte setBytes limit")
    buf = bytearray(total)
    for (kind, val), ty, off in zip(scalar_vals, scalar_types, offsets):
        width = _KIND_WIDTH.get(kind)
        if width is None:
            raise RuntimeError(f"the Metal launch cannot pack a parameter of kind {kind!r} (Triton type {ty!r})")
        struct.pack_into({1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width], buf, off, val & ((1 << (8 * width)) - 1))
    return pointers, bytes(buf)


def _now() -> float:
    return __import__("time").perf_counter() if _SYNC_TIMING else 0.0


def _zeroed_block(nbytes: int):
    """A zeroed device block for an emitter's print or assert records, from nbxtensor's own allocator."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.zeros((max(int(nbytes) // 4, 1),), dtype=NBXDtype.int32, device="cuda:0")


def _print_records(layout, block) -> None:
    """What a printing kernel recorded, after it ran (its records exist only then): a debug path, it waits."""
    from nbxtensor.backend.metal_device import runtime
    from triton_apple_backend.device_print import format_records
    runtime().flush()
    for line in format_records(layout, block.numpy().view("uint32")):
        _log.debug("%s", line)


class _Trailing(NamedTuple):
    """The emitter-declared buffers that follow a kernel's own parameters.

    Held as the PARSED layouts rather than the raw text: parsing is a regex
    walk over the whole emitted module, and it is a property of the
    compilation, so it happens once in `trailing_buffers` and never on the
    launch path.
    """
    print_layout: Any
    assert_layout: Any
    #: triton-ext's two address flags for this compilation, honoured at the launch as its own launcher does.
    exposes_addresses: bool = False
    reads_addresses: bool = False
    #: the grid must be co-resident (`metal_dispatch.dispatch` refuses one wider than the GPU holds).
    cross_tg_barrier: bool = False


class pinned_addresses:
    """Hold these tensors, so an address a kernel CAPTURES stays valid for every launch inside the scope.

    A kernel may store the bitcast of a pointer it was handed into a table, and a later kernel inside the scope may
    load that entry and read through it: that is how a mixture-of-experts band addresses its per-expert weights. The
    address is the allocation's own `gpuAddress()` plus the offset (`pinned_gpu_address`), valid as long as the
    allocation lives; the scope holds the tensors so the allocator cannot free and reissue their storage underneath
    a table (on MPS the first same-size allocation after a free reissued the exact address, and a stale table read
    the newcomer's bytes with nothing raised: repro_130_mps_reissue.py). Nests.
    """

    def __init__(self, *tensors):
        if not tensors:
            raise RuntimeError(
                "pinned_addresses with nothing to pin would promise a lifetime "
                "it does not hold; pass the tensors whose addresses kernels "
                "will capture")
        self._tensors = tuple(tensors)

    def __enter__(self):
        held = [_recorded(int(t.data_ptr())) for t in self._tensors]
        self._bases = [extent[0] for extent in held if extent is not None]     # a recorded shadow holds nothing
        for base in self._bases:
            _PIN_COUNTS[base] = _PIN_COUNTS.get(base, 0) + 1
        return self

    def __exit__(self, exc_type, exc, tb):
        for base in getattr(self, "_bases", ()):
            n = _PIN_COUNTS.get(base, 0) - 1
            if n > 0:
                _PIN_COUNTS[base] = n
            else:
                _PIN_COUNTS.pop(base, None)
        self._bases = []
        return False


#: How many open scopes hold each allocation, by its base: any address inside a held allocation is pinned.
_PIN_COUNTS: dict = {}


def _recorded(addr: int):
    """(base, size) of the allocation holding `addr`; None for a recorded census shadow (memory that exists nowhere,
    pinned as nothing). Any other address is refused by name, in a shadow and in a real run alike: a path that let an
    unrecorded address through would one day pass a real one in silence (the census of 2026-09-24)."""
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.shadow import shadow_range
    if shadow_range(addr) is not None:
        return None
    size = DeviceAllocator._cuda_ptr_size.get(addr)
    if size is not None:
        return addr, size
    hit = DeviceAllocator.containing(addr)
    if hit is None:
        raise RuntimeError(f"the Metal launch cannot pin pointer 0x{addr:x}: the allocator does not record it, so "
                           f"its length and lifetime are unknown")
    return hit


def pinned_gpu_address(addr: int) -> int:
    """The address a SHADER may read at `addr` (its allocation's `gpuAddress()` plus the offset), valid while its
    pin is held. Refuses by name an address no open scope holds: handed out without a pin, it would go stale with
    nothing raised. A recorded census shadow has no GPU address; its own address stands for it."""
    extent = _recorded(int(addr))
    if extent is None:
        return int(addr)
    if extent[0] not in _PIN_COUNTS:
        raise RuntimeError(
            f"pinned_gpu_address(0x{addr:x}): its allocation is not under any "
            f"pinned_addresses scope — an address handed out without a pin "
            f"would go stale silently, which is the defect this refuses")
    from nbxtensor.backend.metal_device import runtime
    return runtime().gpu_address(int(addr))


def driver() -> TritonExtDriver:
    return TritonExtDriver.instance()
