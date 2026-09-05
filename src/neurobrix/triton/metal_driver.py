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


def compile_to_msl(jit_fn, signature: dict, constexprs: dict,
                   num_warps: int = 4):
    """Lower a `@triton.jit` function to MSL. Returns (msl, metadata)."""
    import triton
    from triton.compiler.compiler import ASTSource

    with _MSLOnly():
        compiled = triton.compile(
            ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs),
            target=metal_target(), options={"num_warps": num_warps})
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

def library_from_source(device, msl: str):
    """Compile MSL in-process. No toolchain, no subprocess, no Xcode."""
    import Metal

    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(
        msl, options, None)
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

    def __init__(self, msl: str, metadata, library=None):
        from ..kernels.metal_device import runtime

        self._runtime = runtime()
        self._msl = msl
        self.name, self.binding = parse_kernel_signature(msl)
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

    def launch(self, grid, args, queue=None) -> None:
        """Dispatch `grid` threadgroups.

        `args` is positional and must match the MSL binding order, which
        `parse_kernel_signature` read from the source. Pointer parameters take
        an NBXTensor or an integer address from the Metal allocator; scalars
        take a Python int or float and are packed to the type the kernel
        declares.
        """
        import Metal

        runtime = self._runtime
        encoder_queue = queue or runtime._queue
        command_buffer = encoder_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)

        if len(args) != len(self.binding):
            raise MetalKernelError(
                f"{self.name} binds {len(self.binding)} arguments, "
                f"{len(args)} given")

        for (index, pname, mtype, is_pointer), value in zip(self.binding, args):
            if is_pointer:
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

_KERNEL_CACHE: Dict[str, MetalKernel] = {}
_CACHE_LOCK = threading.Lock()


def kernel_from_msl(msl: str, metadata) -> MetalKernel:
    """A compiled kernel for this MSL, compiled once per process."""
    with _CACHE_LOCK:
        cached = _KERNEL_CACHE.get(msl)
        if cached is None:
            cached = MetalKernel(msl, metadata)
            _KERNEL_CACHE[msl] = cached
        return cached


def compile_kernel(jit_fn, signature: dict, constexprs: dict,
                   num_warps: int = 4) -> MetalKernel:
    """Lower, compile and return a launchable kernel. No torch, no xcrun."""
    msl, metadata = compile_to_msl(jit_fn, signature, constexprs, num_warps)
    return kernel_from_msl(msl, metadata)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _KERNEL_CACHE.clear()
