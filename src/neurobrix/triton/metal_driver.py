"""NeuroBrix's own Metal driver: compile, load and dispatch, without torch.

R33 applies on Metal exactly as on CUDA. That rule decides this module's
shape, so it is worth stating what it excludes before what it does.

`triton-msl` is two things bolted together. Its **lowerer** turns TTGIR into
Metal Shading Language, and that is text — a compiler pass with no runtime and
no torch, verified: importing `triton` and `triton_msl.backend.compiler` and
lowering a kernel all the way to MSL leaves `sys.modules` free of torch. Its
**driver** is the other thing: it imports torch in eight places, binds
arguments through `torch.mps`, and dispatches zero-copy against PyTorch's own
stream. Taking any of it would put torch back into the execution path through
a side door, on Apple only, which is precisely the exception R33 does not
have.

So we take the lowerer and nothing else. Compilation, loading and dispatch are
ours:

* **compile** — `MTLDevice.newLibraryWithSource:` compiles the MSL in-process.
  No `xcrun`, so no Xcode on the user's machine. Proved equivalent to the
  offline compiler at every shape of the first-light milestone, byte for byte,
  and 4-9x faster (`tools/metal_msl_path_equivalence.py`).
* **load** — `newLibraryWithData:` takes a metallib we compiled earlier,
  without recompiling anything. PyObjC cannot bridge the `dispatch_data_t`
  that selector wants, so it is built through libSystem here.
* **dispatch** — on `MTLBuffer`s the Metal `DeviceAllocator` already owns,
  fetched with `buffer_for_pointer`. The container is NBXTensor, on Apple as
  everywhere else.

The one thing this module will not do is invoke `xcrun`. Anything NeuroBrix
compiles for distribution is built to the floor declared below, which is a
build-time tool's job and not the runtime's.

R33 preserved — no torch, at the boundary included.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import threading
from typing import Dict, List, Optional, Tuple

# What NeuroBrix targets when IT compiles a metallib for distribution.
#
# Measured 2026-09-05: the metallib records its own floor, and the flags
# choose it. `-std=metal3.0 -mmacos-version-min=14.0` yields
# `air64_v26-apple-macosx14.0.0`, which loads and runs on this machine
# bit-identically to a Metal-4 build. triton-msl compiles with the HIGHEST
# `-std` the device reports, which on macOS 26 pins the artifact to macOS 26
# and would strand every user below it — NeuroBrix's floor is macOS 14.
METAL_STD_FLOOR = "metal3.0"
MACOS_DEPLOYMENT_FLOOR = "14.0"


class MetalKernelError(RuntimeError):
    """Compilation, loading or dispatch of a Metal kernel failed."""


#: Emitted MSL names a scalar's buffer parameter `<param>_buf`.
_SCALAR_BUFFER_SUFFIX = "_buf"

#: What the launcher must supply for one slot, and how the driver binds it.
POINTER = "pointer"                 # a device address, bound as a buffer
SCALAR_BY_VALUE = "scalar_value"    # an int/float, bound with setBytes
SCALAR_BY_BUFFER = "scalar_buffer"  # an int/float the MSL reads THROUGH a
                                    # pointer, so the driver must put it in a
                                    # device buffer first


def _slot_kinds(binding, signature):
    """How each MSL parameter must be bound, decided from the signature.

    The Metal emitter does not pass every scalar the same way. `rms_norm`'s
    scalars arrive by value; `matmul`'s arrive as `device int* M_buf`, read
    through a pointer inside the kernel. Both are correct MSL and neither is
    visible from the argument the launcher passes — an `int` 128 and a device
    address 0x80 are the same Python object.

    So the decision is made HERE, at compile time, where both facts are
    known: what Triton says the parameter is (`signature`), and how the
    emitter declared it (`binding`). Guessing at launch time from the value
    is not possible, and was the bug: `matmul` refused every autotune config
    with "address 0x80, which the Metal allocator did not hand out" — 0x80
    being M=128.

    Without `signature` the emitter's own view is used, which is right for
    every kernel whose scalars are passed by value and wrong in exactly the
    way described above for the rest. Callers that have the signature pass it.
    """
    kinds = []
    for index, name, mtype, is_pointer in binding:
        if not is_pointer:
            kinds.append(SCALAR_BY_VALUE)
            continue
        if signature is None:
            kinds.append(POINTER)
            continue
        triton_name = name
        if triton_name not in signature and \
                triton_name.endswith(_SCALAR_BUFFER_SUFFIX):
            triton_name = triton_name[:-len(_SCALAR_BUFFER_SUFFIX)]
        declared = signature.get(triton_name)
        if declared is None:
            raise MetalKernelError(
                f"the emitted MSL binds a parameter {name!r} that is not in "
                f"the Triton signature {sorted(signature)}; the driver will "
                f"not guess what to put in it")
        kinds.append(POINTER if declared.startswith("*")
                     else SCALAR_BY_BUFFER)
    return tuple(kinds)


def _arg_slots(binding, kinds=None):
    """The MSL binding, expressed in the launcher contract's terms.

    `is_pointer` is what the LAUNCHER must supply, not how Metal binds it: a
    scalar the emitter reads through a pointer is still a scalar to the
    engine, and declaring it a pointer would make every backend's contract
    describe Metal's code generator.
    """
    from .launcher_contract import ArgSlot

    scalar = {"int": "i32", "uint": "u32", "short": "i16", "ushort": "u16",
              "char": "i8", "long": "i64", "float": "fp32", "half": "fp16"}
    kinds = kinds or tuple(POINTER if b[3] else SCALAR_BY_VALUE
                           for b in binding)
    slots = []
    for (index, name, mtype, _emitted_pointer), kind in zip(binding, kinds):
        is_pointer = kind == POINTER
        slots.append(ArgSlot(index=index, name=name, is_pointer=is_pointer,
                             dtype="*fp32" if is_pointer
                             else scalar.get(mtype, "i32")))
    return tuple(slots)


# --- turning a JITFunction into MSL, without ever reaching xcrun ------------

class _MSLOnly:
    """Run Triton's pipeline but replace its final stage.

    triton-msl's last stage shells out to `xcrun metal`. Substituting it does
    two things at once: nothing invokes the offline compiler, and the compiled
    artifact Triton caches carries the MSL we actually want. The lowering
    itself — TTIR, TTGIR, MSL — is untouched, which is the part we are here
    for.
    """

    _MARKER = b"NBX-MSL-ONLY"

    def __enter__(self):
        from triton_msl.backend.compiler import MetalBackend

        self._backend = MetalBackend
        self._original = MetalBackend.add_stages

        def patched(backend_self, stages, options, language=None):
            self._original(backend_self, stages, options, language)
            stages["metallib"] = lambda src, metadata: _MSLOnly._MARKER

        MetalBackend.add_stages = patched
        return self

    def __exit__(self, *exc):
        self._backend.add_stages = self._original
        return False


def metal_target():
    """The Triton target for this machine, without activating a driver.

    Asking `triton.runtime.driver.active` would make Triton probe every
    registered backend, and upstream's AMD probe does `import torch` in its
    `is_active()` — torch in our process for a card that is not there. The
    target is built from the Metal device instead, which is where the answer
    lives anyway.
    """
    from triton.backends.compiler import GPUTarget

    from ..kernels.metal_device import runtime

    return GPUTarget("metal", runtime().arch_name, 32)


def _attrs_from_specialization(jit_fn, specialization):
    """Triton's `attrs` dict, from the launcher's per-argument markers.

    Triton builds this in `JITFunction.run` from the same markers, and the
    middle end reads `tt.divisibility` to decide whether a load or a store
    may be vectorized. The translation lives here rather than in the launcher
    because it is the compiler's spelling, not the engine's: another backend
    wanting a different one changes this function and nothing else.

    Dropping it is not cosmetic. Compiling rms_norm without these attributes
    changed its fp16 result at two of the four milestone shapes on 2026-09-05
    — same inputs, same driver, different vectorization, different summation
    order — while fp32 was untouched, which is exactly how it would have gone
    unnoticed.

    `BaseBackend.parse_attr` is Triton's own, a staticmethod on the base class
    every backend inherits, and imports no torch.
    """
    if not specialization:
        return None
    from triton.backends.compiler import BaseBackend

    attrs = {}
    for index, name in enumerate(jit_fn.arg_names):
        marker = specialization.get(name)
        if isinstance(marker, str):
            attrs[(index, )] = BaseBackend.parse_attr(marker)
    return attrs


def compile_to_msl(jit_fn, signature: dict, constexprs: dict,
                   num_warps: int = 4, specialization: dict | None = None,
                   num_stages: int | None = None):
    """Lower a `@triton.jit` function to MSL. Returns (msl, metadata).

    `num_stages` is accepted and forwarded, and measured to change nothing in
    the emitted MSL on this backend (2026-09-05: identical bytes at 1, 2 and
    4) because the Metal lowerer does not software-pipeline. It is forwarded
    rather than dropped so that the value the autotuner chose is what the
    compiler saw, whatever the compiler does with it.
    """
    import triton
    from triton.compiler.compiler import ASTSource

    options = {"num_warps": num_warps}
    if num_stages is not None:
        options["num_stages"] = int(num_stages)
    with _MSLOnly():
        compiled = triton.compile(
            ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs,
                      attrs=_attrs_from_specialization(jit_fn, specialization)),
            target=metal_target(), options=options)
    msl = compiled.asm.get("msl")
    if not msl:
        raise MetalKernelError(
            "the Metal backend produced no MSL for "
            f"{getattr(jit_fn, '__name__', jit_fn)!r}")
    return msl, compiled.metadata


# --- the emitted signature, read from the MSL itself ------------------------

_KERNEL_RE = re.compile(r"kernel\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)
_PARAM_RE = re.compile(
    r"(?P<qual>device|constant|threadgroup)\s+(?P<type>[\w:]+)\s*(?P<ref>[*&])\s*"
    r"(?P<name>\w+)\s*\[\[buffer\((?P<index>\d+)\)\]\]")


def parse_kernel_signature(msl: str):
    """(kernel name, [(index, name, msl_type, is_pointer), ...]).

    Read from the emitted source rather than assumed from the Triton
    signature, because the binding order is the emitter's decision and a
    silent mismatch between what we bind and what the kernel reads would be
    wrong numbers, not an error.
    """
    match = _KERNEL_RE.search(msl)
    if not match:
        raise MetalKernelError("no `kernel void` found in the emitted MSL")
    name, params = match.group(1), match.group(2)
    binding = []
    for p in _PARAM_RE.finditer(params):
        binding.append((int(p.group("index")), p.group("name"),
                        p.group("type"), p.group("ref") == "*"))
    binding.sort()
    return name, binding


# --- libraries: from source (framework) or from a prebuilt metallib ---------

#: Metal's math mode. `MTLCompileOptions` defaults to FAST (mathMode 2,
#: measured 2026-09-05), which lets the compiler reassociate float arithmetic
#: and substitute fast approximations for divide, rsqrt and the transcendental
#: functions. That is a numerical policy, and taking it by default means the
#: engine's fp32 results on Apple are decided by a flag nobody chose.
#:
#: It is not hypothetical: with fast math on, rms_norm fp32 at 2x4096 sat 3
#: ULP from the fp64 oracle where the CUDA reference sits at 1, and the
#: milestone bar — no further from the oracle than CUDA — failed at that
#: shape. Under SAFE the same kernel is BIT-IDENTICAL to CUDA. Safe is IEEE
#: semantics, which is what these kernels are written against everywhere else.
#:
#: 0 = MTLMathModeSafe, 1 = Relaxed, 2 = Fast.
METAL_MATH_MODE_SAFE = 0


def compile_options():
    """The `MTLCompileOptions` every kernel of this engine is built with.

    A function rather than an inline object so the policy is one thing that
    can be read and asserted, instead of a line inside a compile call.
    """
    import Metal

    options = Metal.MTLCompileOptions.alloc().init()
    # `mathMode` is the current spelling, `fastMathEnabled` the deprecated one
    # kept for older macOS. Setting whichever exists is not a fallback — both
    # name the same switch — and the check below refuses if neither took.
    if hasattr(options, "setMathMode_"):
        options.setMathMode_(METAL_MATH_MODE_SAFE)
    elif hasattr(options, "setFastMathEnabled_"):
        options.setFastMathEnabled_(False)
    else:
        raise MetalKernelError(
            "MTLCompileOptions exposes neither mathMode nor fastMathEnabled; "
            "the engine will not compile kernels under an unknown float "
            "policy")
    if hasattr(options, "mathMode") \
            and options.mathMode() != METAL_MATH_MODE_SAFE:
        raise MetalKernelError(
            f"asked for safe math, got mathMode {options.mathMode()}")
    return options


def library_from_source(device, msl: str):
    """Compile MSL in-process. No toolchain, no subprocess, no Xcode."""
    library, error = device.newLibraryWithSource_options_error_(
        msl, compile_options(), None)
    if library is None:
        raise MetalKernelError(f"framework MSL compile failed: {error}")
    return library


def library_from_metallib(device, blob: bytes):
    """Load a prebuilt metallib. Nothing is recompiled.

    `newLibraryWithData:` wants a `dispatch_data_t`. PyObjC cannot bridge one
    — handing the selector `bytes` or an `NSData` segfaults the process, which
    is why triton-msl routes around it through a temporary file — so it is
    built here through libSystem and wrapped as an Objective-C object.
    """
    import objc

    libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
    libsystem.dispatch_data_create.restype = ctypes.c_void_p
    libsystem.dispatch_data_create.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    holder = ctypes.create_string_buffer(blob, len(blob))
    handle = libsystem.dispatch_data_create(
        ctypes.cast(holder, ctypes.c_void_p), len(blob), None, None)
    if not handle:
        raise MetalKernelError("dispatch_data_create returned NULL")
    library, error = device.newLibraryWithData_error_(
        objc.objc_object(c_void_p=ctypes.c_void_p(handle)), None)
    if library is None:
        raise MetalKernelError(f"newLibraryWithData failed: {error}")
    return library


# --- a compiled, launchable kernel ------------------------------------------

class MetalKernel:
    """One compiled MSL kernel, ready to dispatch on allocator buffers."""

    def __init__(self, msl: str, metadata, library=None, constexprs=None,
                 specialization=None, signature=None):
        from ..kernels.metal_device import runtime

        self._runtime = runtime()
        self._msl = msl
        self.name, self._msl_binding = parse_kernel_signature(msl)
        # How each parameter must be bound — decided here, from the Triton
        # signature and the emitted declaration together. See `_slot_kinds`.
        self._slot_kinds = _slot_kinds(self._msl_binding, signature)
        # The launcher contract's view of the same thing.
        self.binding = _arg_slots(self._msl_binding, self._slot_kinds)
        #: Device buffers holding scalars the MSL reads through a pointer.
        #: One per such slot, allocated on first use and reused: launches are
        #: synchronous here, so the value is consumed before it is rewritten.
        self._scalar_buffers: dict = {}
        self.constexprs = dict(constexprs or {})
        self.specialization = dict(specialization or {})
        # This driver reloads from MSL source through the framework, so the
        # artifact it caches IS the source. A metallib is the alternative and
        # `library_from_metallib` loads one; the kind is declared so a cache
        # cannot hand a metallib to a backend expecting a cubin.
        self.binary = msl.encode("utf-8")
        self.binary_kind = "msl"
        self.shared_memory = int(getattr(metadata, "shared", 0) or 0)
        self.block_size = int(getattr(metadata, "block_size", 0)
                              or getattr(metadata, "num_warps", 4) * 32)
        library = library or library_from_source(self._runtime._device, msl)
        function = library.newFunctionWithName_(self.name)
        if function is None:
            raise MetalKernelError(
                f"library has no function {self.name!r}; it exposes "
                f"{list(library.functionNames())}")
        pipeline, error = \
            self._runtime._device.newComputePipelineStateWithFunction_error_(
                function, None)
        if pipeline is None:
            raise MetalKernelError(f"pipeline state failed: {error}")
        self._library = library
        self._pipeline = pipeline

    @property
    def msl(self) -> str:
        return self._msl

    def _scalar_buffer(self, index: int, value, mtype: str):
        """A device buffer holding one scalar the kernel reads by pointer.

        The address comes from the allocator like every other device buffer —
        the driver does not get a private allocation path — so
        `buffer_for_pointer` resolves it and the ownership rule that refuses
        foreign addresses keeps applying.
        """
        from ..kernels.nbx_tensor import DeviceAllocator

        address = self._scalar_buffers.get(index)
        if address is None:
            address = DeviceAllocator.malloc_cuda(4)
            self._scalar_buffers[index] = address

        payload = _pack_scalar(value, mtype)
        host = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
        DeviceAllocator.memcpy(address, ctypes.addressof(host), len(payload),
                               kind=1)

        buffer, offset = self._runtime.buffer_for_pointer(address)
        if buffer is None:                              # pragma: no cover
            raise MetalKernelError(
                f"{self.name}: the allocator does not recognise the scalar "
                f"buffer it just handed out for {index}")
        return buffer, offset

    def launch(self, grid, args, stream: int = 0) -> None:
        """Dispatch `grid` threadgroups on `stream`.

        `stream` is the contract's spelling and the allocator's handle: 0 is
        the allocator's own queue, anything else a queue from
        `create_stream`. It matters for more than tidiness — an event
        recorded on one queue says nothing about work submitted on another,
        so the launcher's autotune benchmark would time an empty queue if
        this ignored the argument.

        `args` is positional and must match the MSL binding order, which
        `parse_kernel_signature` read from the source. Pointer parameters take
        an NBXTensor or an integer address from the Metal allocator; scalars
        take a Python int or float and are packed to the type the kernel
        declares.
        """
        import Metal
        import objc

        # An autorelease pool per dispatch, so the transient Objective-C
        # objects go away promptly in a process that has no run loop to drain
        # one. Hygiene, not the fix for the hang below — measured: removing
        # it changes nothing over 256 launches.
        with objc.autorelease_pool():
            self._dispatch(Metal, grid, args, stream)

    def _dispatch(self, Metal, grid, args, stream: int) -> None:
        runtime = self._runtime
        encoder_queue = runtime._resolve_queue(int(stream or 0))
        if encoder_queue is None:
            raise MetalKernelError(
                f"{self.name}: stream handle {stream!r} is not one the "
                f"allocator handed out")
        command_buffer = encoder_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        try:
            self._encode_and_run(Metal, command_buffer, encoder, grid, args,
                                 runtime)
        except BaseException:
            # A command buffer counts against the queue's in-flight limit
            # from the moment it is created until it COMPLETES. One that is
            # abandoned — because encoding raised — never completes, and its
            # slot is gone for the life of the process. After 64 the queue
            # blocks in `commandBuffer()` and nothing on this device runs
            # again.
            #
            # Measured 2026-09-05: the kernels suite reached ~90% and stopped
            # dead, 25 seconds of CPU across an hour of wall clock, parked in
            # `_dispatch_semaphore_wait_slow` under
            # `[AGXG16XFamilyCommandQueue commandBuffer]`. That suite has
            # ~219 failing tests and most fail inside a launch: each one
            # leaked a slot. It surfaced now only because the launcher routes
            # every kernel launch through here.
            #
            # Committing an abandoned buffer lets it complete empty and
            # returns the slot. The encoder must be closed first — Metal
            # aborts the process on `commit command buffer with uncommitted
            # encoder`, which is a worse failure than the one being reported.
            # The exception is re-raised unchanged: a refusal must stay as
            # loud as it was.
            try:
                encoder.endEncoding()
            except Exception:                           # pragma: no cover
                pass
            try:
                command_buffer.commit()
            except Exception:                           # pragma: no cover
                pass
            raise

    def _encode_and_run(self, Metal, command_buffer, encoder, grid, args,
                        runtime) -> None:
        encoder.setComputePipelineState_(self._pipeline)

        if len(args) != len(self._msl_binding):
            raise MetalKernelError(
                f"{self.name} binds {len(self._msl_binding)} arguments, "
                f"{len(args)} given")

        for (slot, value, kind) in zip(self._msl_binding, args,
                                       self._slot_kinds):
            index, pname, mtype, _emitted = slot
            if kind == POINTER:
                address = getattr(value, "data_ptr", None)
                address = address() if callable(address) else int(value)
                buffer, offset = runtime.buffer_for_pointer(address)
                if buffer is None:
                    raise MetalKernelError(
                        f"{self.name} argument {pname!r} is address "
                        f"{address:#x}, which the Metal allocator did not "
                        f"hand out. Every device buffer must come from "
                        f"NBXTensor / DeviceAllocator.")
                encoder.setBuffer_offset_atIndex_(buffer, offset, index)
            elif kind == SCALAR_BY_BUFFER:
                buffer, offset = self._scalar_buffer(index, value, mtype)
                encoder.setBuffer_offset_atIndex_(buffer, offset, index)
            else:
                encoder.setBytes_length_atIndex_(
                    _pack_scalar(value, mtype), 4, index)

        groups = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        groups = (groups + (1, 1))[:3]
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(groups[0]), int(groups[1]), int(groups[2])),
            Metal.MTLSizeMake(int(self.block_size), 1, 1))
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        error = command_buffer.error()
        if error is not None:
            raise MetalKernelError(f"{self.name} dispatch failed: {error}")


_INT_TYPES = {"int", "uint", "int32_t", "uint32_t", "short", "ushort", "char"}


def _pack_scalar(value, msl_type: str) -> bytes:
    """Four bytes, little-endian, in the type the kernel declares."""
    import struct

    if msl_type in _INT_TYPES:
        packed = int(value)
        return struct.pack("<I" if msl_type.startswith("u") else "<i", packed)
    if msl_type in ("float", "half"):
        return struct.pack("<f", float(value))
    raise MetalKernelError(f"cannot pack a scalar of MSL type {msl_type!r}")


# --- the cache --------------------------------------------------------------

#: (MSL text, specialization) -> kernel. See `kernel_from_msl`.
_KERNEL_CACHE: Dict[tuple, MetalKernel] = {}
#: MSL text -> compiled library, shared by every kernel built from it.
_LIBRARY_CACHE: Dict[str, object] = {}
_CACHE_LOCK = threading.Lock()


def kernel_from_msl(msl: str, metadata, constexprs=None,
                    specialization=None, signature=None) -> MetalKernel:
    """A compiled kernel for this MSL, built once per process.

    Two caches, because two different things are being reused:

    * the **library** is keyed on the MSL text alone. That text already
      encodes the constexprs and whatever the divisibility attributes
      changed, so two compilations producing identical source really are the
      same compiled code and the expensive part is shared.

    * the **kernel object** is keyed on the MSL *and* the specialization it
      was compiled with. A kernel handed back from the first cache would
      otherwise report the markers of whoever compiled it first — for a
      kernel the markers do not change, `add_one` say, that is a lie the
      launcher contract catches and should catch.
    """
    spec = dict(specialization or {})
    key = (msl, tuple(sorted(spec.items())))
    with _CACHE_LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is None:
            library = _LIBRARY_CACHE.get(msl)
            cached = MetalKernel(msl, metadata, library=library,
                                 constexprs=constexprs, specialization=spec,
                                 signature=signature)
            _LIBRARY_CACHE.setdefault(msl, cached._library)
            _KERNEL_CACHE[key] = cached
        return cached


def compile_kernel(jit_fn, signature: dict, constexprs: dict,
                   num_warps: int = 4, specialization: dict | None = None,
                   num_stages: int | None = None) -> MetalKernel:
    """Lower, compile and return a launchable kernel. No torch, no xcrun."""
    msl, metadata = compile_to_msl(jit_fn, signature, constexprs, num_warps,
                                   specialization, num_stages)
    return kernel_from_msl(msl, metadata, constexprs, specialization,
                           signature)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _KERNEL_CACHE.clear()
        _LIBRARY_CACHE.clear()


# ---------------------------------------------------------------------------
# The launcher contract, implemented
# ---------------------------------------------------------------------------

class MetalDriver:
    """Metal, behind `launcher_contract.LauncherDriver`.

    The first implementation of that contract, and deliberately thin: every
    method below is either the compile path above or a call the allocator
    already owns. A CUDA implementation is the same class over
    `DeviceAllocator`'s cuda entry points, which is the point of writing the
    contract down rather than letting the launcher learn Metal's habits.
    """

    backend = "metal"

    def compile(self, jit_fn, signature, constexprs, num_warps: int = 4,
                specialization=None, num_stages=None):
        return compile_kernel(jit_fn, signature, constexprs, num_warps,
                              specialization, num_stages)

    # -- ordering: the allocator owns streams and events on every backend ----

    @staticmethod
    def _allocator():
        from ..kernels.nbx_tensor import DeviceAllocator
        return DeviceAllocator

    def create_stream(self) -> int:
        return self._allocator().create_stream()

    def destroy_stream(self, stream: int) -> None:
        self._allocator().destroy_stream(stream)

    def synchronize_stream(self, stream: int) -> None:
        self._allocator().stream_synchronize(stream)

    def create_event(self, timing: bool = False) -> int:
        return self._allocator().create_event(timing=timing)

    def destroy_event(self, event: int) -> None:
        self._allocator().destroy_event(event)

    def record_event(self, event: int, stream: int = 0) -> None:
        self._allocator().record_event(event, stream)

    def synchronize_event(self, event: int) -> None:
        self._allocator().event_synchronize(event)

    def wait_event(self, stream: int, event: int) -> None:
        self._allocator().stream_wait_event(stream, event)

    def elapsed_ms(self, start: int, end: int) -> float:
        return self._allocator().event_elapsed_ms(start, end)


def driver() -> MetalDriver:
    """The process-wide Metal driver."""
    return _DRIVER


_DRIVER = MetalDriver()
