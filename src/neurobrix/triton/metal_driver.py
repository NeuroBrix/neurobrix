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
import struct
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
#: One `[[buffer(n)]]` parameter of an emitted kernel.
#:
#: The address-space qualifier may be followed by `const`, and by `volatile`:
#: `device const float* Q [[buffer(0)]]` is what the FlashAttention templates
#: emit for their read-only operands. Requiring the bare form silently matched
#: NOTHING for those kernels, so every pointer went unbound and Q, K and V
#: read as zeros — attention returning a zero tensor with no error anywhere
#: (measured 2026-09-07: l = 48, m = 0, acc = 0, which is what an all-zero
#: input looks like from inside the kernel).
_PARAM_RE = re.compile(
    r"(?P<qual>device|constant|threadgroup)\s+"
    r"(?:(?:const|volatile)\s+)*"
    r"(?P<type>[\w:]+)\s*(?P<ref>[*&])\s*"
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

#: Metal's math modes, by the name the hardware profile uses. Metal accepts
#: three; none of them is chosen implicitly here.
METAL_MATH_MODES = {"safe": 0, "relaxed": 1, "fast": 2}
METAL_MATH_MODE_SAFE = 0


def _shader_policy() -> dict:
    """How this vendor compiles shaders, from the hardware profile.

    Two facts live here — the language version and the float policy — and
    both were Metal defaults nobody chose until an Apple machine made them
    visible: fast math cost rms_norm fp32 two ULP against the CUDA reference,
    and the default language version 2.4 has no `bfloat`, so the first
    TinyLlama --triton run refused at the embedding kernel while the same
    profile's `precision.supports_bf16` said the hardware has it.

    Read from `config/vendors/apple/apple_silicon.yml`, zero-fallback: a
    profile that does not declare the policy is a profile this driver
    refuses to guess for.
    """
    from ..core.config.loader import get_vendor_config

    shader = get_vendor_config("apple", "apple_silicon").get("shader")
    if not shader:
        raise MetalKernelError(
            "the Apple hardware profile declares no `shader` policy: the "
            "engine will not compile kernels under a language version and a "
            "float mode nobody chose")
    for key in ("language_version", "math_mode"):
        if key not in shader:
            raise MetalKernelError(
                f"the Apple hardware profile's `shader` block is missing "
                f"{key!r}")
    return shader


def _language_version(text: str) -> int:
    """`"3.1"` -> Metal's packed `(major << 16) | minor`."""
    try:
        major, minor = (int(part) for part in str(text).split(".")[:2])
    except Exception:
        raise MetalKernelError(
            f"shader.language_version {text!r} is not a MAJOR.MINOR version")
    return (major << 16) | minor


def compile_options():
    """The `MTLCompileOptions` every kernel of this engine is built with.

    A function rather than an inline object so the policy is one thing that
    can be read and asserted, instead of two lines inside a compile call.
    Every value is the profile's; the defaults are refused rather than
    inherited, and each setting is read back so a silent no-op is impossible.
    """
    import Metal

    policy = _shader_policy()
    options = Metal.MTLCompileOptions.alloc().init()

    version = _language_version(policy["language_version"])
    options.setLanguageVersion_(version)
    if int(options.languageVersion()) != version:
        raise MetalKernelError(
            f"asked for Metal {policy['language_version']}, the compiler "
            f"kept {int(options.languageVersion())}")

    name = str(policy["math_mode"]).lower()
    if name not in METAL_MATH_MODES:
        raise MetalKernelError(
            f"shader.math_mode {name!r} is not one of "
            f"{sorted(METAL_MATH_MODES)}")
    mode = METAL_MATH_MODES[name]
    # `mathMode` is the current spelling, `fastMathEnabled` the deprecated
    # one kept for older macOS. Setting whichever exists is not a fallback —
    # both name the same switch — and the read-back below refuses if neither
    # took.
    if hasattr(options, "setMathMode_"):
        options.setMathMode_(mode)
    elif hasattr(options, "setFastMathEnabled_"):
        options.setFastMathEnabled_(mode == METAL_MATH_MODES["fast"])
    else:
        raise MetalKernelError(
            "MTLCompileOptions exposes neither mathMode nor fastMathEnabled; "
            "the engine will not compile kernels under an unknown float "
            "policy")
    if hasattr(options, "mathMode") and int(options.mathMode()) != mode:
        raise MetalKernelError(
            f"asked for {name} math, got mathMode {int(options.mathMode())}")
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

    def launch_params(self, grid, block, params, stream: int = 0,
                      names=None) -> None:
        """The engine launcher's entry: `(kind, value)` pairs in signature
        order, and the block the driver chose from the compiled metadata.

        `block` is the size the DRIVER read from the compiled metadata, and
        it is the one dispatched. The emitted MSL was built for exactly one
        threadgroup size and dispatching it at another is wrong results
        rather than a performance choice, so there is one source for it —
        the metadata of the compilation that produced this binary — and this
        object does not compute a second opinion. (`self.block_size` remains
        as the fallback for the direct `compile_kernel` callers, which have
        no launcher above them.)

        Metal's own limits are checked, because they are the ones that turn
        a bad size into a dispatch failure rather than a wrong answer.
        """
        import Metal
        import objc

        want = int(block[0]) if isinstance(block, (tuple, list)) else int(block)
        want = want or self.block_size
        # The bound is the PIPELINE's, not a rule of thumb. Metal has no
        # multiple-of-the-SIMD-width requirement — a 16-thread threadgroup is
        # legal, it simply leaves half a SIMD group idle, and several gather
        # and index kernels legitimately use one. Asserting a multiple of 32
        # here refused them (measured 2026-09-06: index_select and embedding).
        # `maxTotalThreadsPerThreadgroup` is per compiled kernel and accounts
        # for its register and threadgroup-memory use, which a constant 1024
        # does not.
        ceiling = int(self._pipeline.maxTotalThreadsPerThreadgroup())
        if want <= 0 or want > ceiling:
            raise MetalKernelError(
                f"{self.name}: a threadgroup of {want} threads is not "
                f"dispatchable; this kernel's pipeline allows 1..{ceiling}")

        with objc.autorelease_pool():
            self._dispatch_params(Metal, grid, params, stream, want, names)

    def _dispatch_params(self, Metal, grid, params, stream: int,
                         threads: int, names=None) -> None:
        runtime = self._runtime
        encoder_queue = runtime._resolve_queue(int(stream or 0))
        if encoder_queue is None:
            raise MetalKernelError(
                f"{self.name}: stream handle {stream!r} is not one the "
                f"allocator handed out")
        # Bind by NAME when the launcher supplies them, else by index.
        #
        # The emitted MSL declares only the arguments the compiled kernel
        # actually uses: the middle end drops ones it proved dead, and a
        # template declares the roles it reads. The launcher, correctly,
        # passes every non-constexpr parameter of the signature. So the two
        # counts differ legitimately — measured: a FlashAttention kernel whose
        # MSL declares 22 buffers while the signature carries 28 — and zipping
        # them positionally either refuses a good launch or, worse, shifts
        # every argument after the first gap.
        #
        # Each declaration carries its own `[[buffer(i)]]`, and i is the
        # argument's index in that same signature, so the mapping is exact.
        needed = max((slot[0] for slot in self._msl_binding), default=-1)
        if needed >= len(params):
            raise MetalKernelError(
                f"{self.name} declares buffer({needed}) but only "
                f"{len(params)} arguments were given")

        # The stream's open encoder, not one of ours: a NeuroBrix stream is
        # one command buffer with one serialised compute encoder, opened at
        # the first launch and committed at a synchronisation point. Creating
        # a buffer per launch put 1,505 of them in a decode step and 501 ms of
        # that step into draining them.
        command_buffer, encoder = runtime.encoder_for(int(stream or 0))
        if encoder is None:                             # pragma: no cover
            raise MetalKernelError(
                f"{self.name}: the stream has no encoder to dispatch into")
        try:
            encoder.setComputePipelineState_(self._pipeline)
            by_name = {}
            if names:
                by_name = {n: params[i] for i, n in enumerate(names)
                           if i < len(params)}
            for index, pname, mtype, emitted_pointer in self._msl_binding:
                # The emitted name is the kernel's own argument name, with the
                # `_buf` suffix the emitter adds to a scalar it passes through
                # a pointer. Name is the only mapping that survives a kernel
                # whose artifact declares fewer arguments than the signature
                # carries AND one whose buffer indices are the graph's rather
                # than the signature's — both occur.
                key = pname[:-len(_SCALAR_BUFFER_SUFFIX)] \
                    if pname.endswith(_SCALAR_BUFFER_SUFFIX) and pname not in by_name \
                    else pname
                if key in by_name:
                    kind, value = by_name[key]
                elif index < len(params):
                    kind, value = params[index]
                else:
                    raise MetalKernelError(
                        f"{self.name}: no argument for {pname!r} "
                        f"(buffer {index}); the artifact and the signature "
                        f"disagree and there is no name to match on")
                if kind == "ptr":
                    buffer, offset = runtime.buffer_for_pointer(int(value))
                    if buffer is None:
                        raise MetalKernelError(
                            f"{self.name} argument {pname!r} is address "
                            f"{int(value):#x}, which the Metal allocator did "
                            f"not hand out. Every device buffer must come "
                            f"from NBXTensor / DeviceAllocator.")
                    encoder.setBuffer_offset_atIndex_(buffer, offset, index)
                elif emitted_pointer:
                    # A scalar the emitted kernel reads THROUGH a pointer:
                    # copied into the command buffer, not into a buffer this
                    # kernel shares with its own other dispatches.
                    encoder.setBytes_length_atIndex_(
                        struct.pack("<I", int(value) & 0xFFFFFFFF), 4, index)
                else:
                    encoder.setBytes_length_atIndex_(
                        _pack_bits(kind, value), 4, index)

            groups = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
            groups = (groups + (1, 1))[:3]
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(int(groups[0]), int(groups[1]), int(groups[2])),
                Metal.MTLSizeMake(int(threads), 1, 1))
            # Encoded, not committed: the stream decides when to commit,
            # and the host waits only where it reads. Metal keeps the order
            # inside a serial compute encoder with its own barriers.
            runtime.note_dispatch(int(stream or 0))
        except BaseException:
            # A command buffer holds its queue's in-flight slot until it
            # COMPLETES; one abandoned mid-encode never does, and after 64 the
            # queue blocks forever. The stream closes its buffer — ending the
            # encoder first, because Metal aborts the process on a commit with
            # one still open.
            try:
                runtime.abandon_stream(int(stream or 0))
            except Exception:                           # pragma: no cover
                pass
            raise
        # No error to read yet: the buffer has not completed. A failure
        # surfaces at the flush that waits for it, named there.

    #: A scalar the emitted kernel reads THROUGH a pointer is bound with
    #: `setBytes:length:atIndex:`, which copies the value into the command
    #: buffer and gives the kernel a pointer to that copy. It replaced a
    #: device buffer cached per (kernel, slot) and written with an H2D
    #: memcpy, which had two faults, one old and one new:
    #:
    #: * the cache made every dispatch of a kernel share four bytes, so once
    #:   several dispatches were encoded before any of them ran, they all
    #:   read whichever value was written last. That was correct only while
    #:   every launch waited for its own kernel — the wait was holding up a
    #:   correctness property nobody had written down;
    #: * the write went through `DeviceAllocator.memcpy`, which synchronises
    #:   the stream, which ENDS the encoder being written to. The next
    #:   `setBuffer` call then touched an ended encoder and the process died
    #:   with a segmentation fault (measured 2026-09-07, at the first mm of
    #:   the first prefill).
    #:
    #: The rule it leaves behind: nothing on the encoding path may call into
    #: the allocator, because the allocator synchronises.

    @staticmethod
    def _scalar_bytes(value, mtype: str) -> bytes:
        """One scalar the kernel reads by pointer, as bytes for `setBytes`.

        See the note above `_dispatch_params`'s scalar branch for why
        this is not a device buffer.
        """
        return _pack_scalar(value, mtype)

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
        command_buffer, encoder = runtime.encoder_for(int(stream or 0))
        if encoder is None:                             # pragma: no cover
            raise MetalKernelError(
                f"{self.name}: the stream has no encoder to dispatch into")
        try:
            self._encode_and_run(Metal, command_buffer, encoder, grid, args,
                                 runtime, int(stream or 0))
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
                runtime.abandon_stream(int(stream or 0))
            except Exception:                           # pragma: no cover
                pass
            raise

    def _encode_and_run(self, Metal, command_buffer, encoder, grid, args,
                        runtime, stream: int = 0) -> None:
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
                payload = self._scalar_bytes(value, mtype)
                encoder.setBytes_length_atIndex_(payload, len(payload), index)
            else:
                encoder.setBytes_length_atIndex_(
                    _pack_scalar(value, mtype), 4, index)

        groups = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        groups = (groups + (1, 1))[:3]
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(int(groups[0]), int(groups[1]), int(groups[2])),
            Metal.MTLSizeMake(int(self.block_size), 1, 1))
        # Encoded into the stream — see `_dispatch_params`. The error is read
        # at the flush that waits for the buffer the stream commits.
        runtime.note_dispatch(stream)


#: The launcher hands scalars as (kind, integer) with floats already reduced
#: to their IEEE bit pattern — `bits16` / `bits32` / `bits64` — so nothing
#: here re-derives a type from a Python value.
def _pack_bits(kind: str, value) -> bytes:
    """Four bytes for one scalar slot, from the launcher's kind."""
    if kind in ("bits16", "bits32", "i8", "i16", "i32", "u8", "u16", "u32",
                "i1", "u1"):
        return struct.pack("<I", int(value) & 0xFFFFFFFF)
    if kind in ("i64", "u64", "bits64"):
        # Metal kernels emitted from Triton take 32-bit scalar slots; a value
        # that does not fit is a silent truncation, so it is refused.
        packed = int(value)
        if not (-(2 ** 31) <= packed < 2 ** 32):
            raise MetalKernelError(
                f"a 64-bit scalar ({packed}) does not fit the 32-bit slot the "
                f"emitted MSL declares; refusing rather than truncating")
        return struct.pack("<I", packed & 0xFFFFFFFF)
    raise MetalKernelError(f"unsupported scalar kind {kind!r}")


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

class _Metadata:
    """The two fields `load` needs from a compiled kernel, for the callers
    that have the artifact but not Triton's metadata object."""

    __slots__ = ("name", "shared", "block_size", "num_warps")

    def __init__(self, name, shared, block_size=0, num_warps=4):
        self.name = name
        self.shared = int(shared or 0)
        self.block_size = int(block_size or 0)
        self.num_warps = int(num_warps)

class MetalDriver:
    """Metal behind the engine's one launcher (`kernels/launcher.py`).

    The launcher compiles — Triton's own compiler, torch-free, with the
    target this driver names — and this loads the result and launches it.
    That division is the launcher's, not Metal's: `load`, `launch`,
    `block_for`, `target`, `artifact_kind` and `wants_scratch_params` are the
    whole vendor surface, and the CUDA driver in the launcher implements the
    same six.

    The ordering calls below (streams, events) are the allocator's on every
    backend and are here only because the contract checker exercises them
    through a driver handle.
    """

    backend = "metal"

    # Not a subclass of `launcher.Driver`: the launcher resolves its driver
    # from THIS module, so inheriting would be an import cycle, and it
    # duck-types the six members below anyway. The contract is the six, not
    # the base class — which is what makes it implementable from anywhere.

    #: Triton's Metal backend emits MSL; there is no cubin to load.
    artifact_kind = "msl"

    #: Metal's dispatch takes the kernel's own arguments and nothing else.
    #: The two trailing scratch pointers of Triton's CUDA ABI would be two
    #: extra buffer bindings the emitted kernel does not declare, and the
    #: length check in `MetalKernel.launch` would refuse the launch.
    wants_scratch_params = False

    def target(self):
        return metal_target()

    def block_for(self, metadata):
        """The threadgroup size the EMITTED kernel was built for.

        Not `num_warps * 32`. triton-msl records the MSL's own threadgroup
        size separately precisely because the C++ path can overwrite
        `metadata.block_size` with a value meant for its host metallib, and
        launching MSL at that other size is silently wrong results rather
        than an error.
        """
        size = int(getattr(metadata, "block_size", 0)
                   or getattr(metadata, "num_warps", 4) * 32)
        return (size, 1, 1)

    def load(self, binary, name: str, shared: int):
        """Compile the MSL to a pipeline and return the launchable handle."""
        msl = binary.decode("utf-8") if isinstance(binary, bytes) else binary
        return kernel_from_msl(msl, _Metadata(name, shared))

    def launch(self, function, grid, block, shared: int, stream: int,
               params, names=None) -> None:
        """Dispatch a loaded kernel. `params` is the launcher's list of
        `(kind, value)` pairs, in the compiled signature's order.

        The KIND is what makes this correct rather than lucky. Metal declares
        some kernels' scalars as `device int* M_buf` and reads them through a
        pointer, and an `int` 128 is indistinguishable from a device address
        0x80 at the call site — so the binding is decided from the kind the
        signature gave (`"ptr"` versus a scalar kind) and the slot the
        emitter declared, never from the value.
        """
        function.launch_params(grid, block, params, stream, names)

    def compile(self, jit_fn, signature, constexprs, num_warps: int = 4,
                specialization=None, num_stages=None):
        """The whole path in one call, for tools that have no launcher around
        them (the compile census, the first-light harness, the R33 proof)."""
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
