"""NeuroBrix kernel launcher — the dispatch layer launches, Triton compiles.

Contract: docs/internal/metal_launcher_contract.md. The owner's universal R33
(2026-09-05) counts upstream Triton's launch path (`kernel[grid]` →
`triton.runtime.driver.active`, whose CUDA backend imports torch for the
device, the stream and its benchmark buffers) as a violation on every
backend. This module keeps Triton as the COMPILER — its binder specialises
the arguments and its compiler produces the binary, both torch-free — and
hands the launch to a vendor driver of its own: the compiled binary, the
entry name, the grid, the block, the shared-memory size, the stream, and the
arguments as integer pointers and typed scalars.

    launch(kernel, grid, *args, **constexprs_and_options)

is the one entry point; `install()` routes every `kernel[grid](...)` of the
house library through it (the seam for the whole-zoo bit-identity gate,
`NBX_LAUNCHER=triton` restores upstream for the differential).

CUDA is the first client (`CudaDriver`, ctypes on libcuda); ROCm and Metal
implement the same four calls behind the same `Driver` interface.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import struct
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# The target: from the engine's hardware profile, never from a driver probe
# ---------------------------------------------------------------------------

_TARGET = None          # triton.backends.compiler.GPUTarget, resolved once


def _compute_capability() -> int:
    """Compute capability as an int (70 for sm_70): the hardware profile the
    process was given (`kernels.wrappers.set_hardware_profile`), else one
    driver attribute query at first use (cached, never per launch)."""
    try:
        import sys as _sys
        _w = _sys.modules.get("neurobrix.kernels.wrappers")   # never imported from here: the
        prof = getattr(_w, "_HARDWARE_PROFILE", None)          # wrapper module is heavy and above us
        devs = getattr(prof, "devices", None) if prof is not None else None
        if devs:
            cc = str(getattr(devs[0], "compute_capability", "") or "")
            if cc and "." in cc:
                major, minor = cc.split(".")[:2]
                return int(major) * 10 + int(minor)
    except Exception:
        pass
    return CudaDriver.instance().compute_capability()


_SMEM_LIMIT: Optional[int] = None


def max_shared_memory_per_block() -> Optional[int]:
    """The executing device's shared-memory ceiling per block, or None.

    The hard limit comes from the device and never from a table of names —
    sm_86 declares 99 KB where sm_80 declares 163, and a table keyed on the
    capability MAJOR gives the first the second's answer. That is what stopped
    every language model on an A40: the flash tile was sized for sm_80 and
    asked 164 352 bytes of a card that holds 101 376.

    None when the driver cannot be asked (no CUDA, a backend without the
    query). The caller must then leave its proposal alone rather than invent a
    budget — pruning on a guessed number deletes configurations that work.

    Cached for the life of the process: a driver query in a hot path is an
    anti-pattern this engine has already paid for once.
    """
    global _SMEM_LIMIT
    if _SMEM_LIMIT is None:
        try:
            _SMEM_LIMIT = int(active_driver().max_shared_memory_per_block())
        except Exception:
            return None
    return _SMEM_LIMIT


def target():
    """The compile target of this process: vendor, capability, warp size —
    from the engine's data, never from `triton.runtime.driver.active` (whose
    backend probes import torch). The one source for every module that
    used to ask Triton's driver (matmul / config spaces, the autotune cache).

    The vendor is the DRIVER's answer, not a literal here: on Apple the
    vendor is `metal`, the arch is a device name string rather than a
    compute capability, and both come from the hardware profile like
    everything else. `CudaDriver.target()` returns exactly what this used to
    return, so the CUDA path is unchanged."""
    global _TARGET
    if _TARGET is None:
        _TARGET = active_driver().target()
    return _TARGET


def arch() -> int:
    """Compute capability as an int (70 for sm_70) — `target().arch`."""
    return int(target().arch)


# ---------------------------------------------------------------------------
# The vendor driver interface
# ---------------------------------------------------------------------------

def _cubin_param_count(cubin: bytes, name: str) -> Optional[int]:
    """The number of kernel parameters recorded in a cubin — the count of
    EIATTR_KPARAM_INFO entries in its `.nv.info.<kernel>` section.

    Read from the binary because the driver on this box (535, CUDA 12.2) has
    no `cuFuncGetParamInfo`, and the driver-level `launch` must still refuse
    an argument list of the wrong length instead of launching it: a missing
    trailing parameter is read as garbage by the kernel and faults, or worse
    does not. ELF64 walk: section headers, the string table, then the
    `.nv.info.<name>` attribute stream (format byte, attribute byte, a
    2-byte size for the SVAL format). Returns None when the section is
    absent, and the launch then cannot check the count."""
    try:
        if cubin[:4] != b"\x7fELF":
            return None
        e_shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)

        def section(i):
            off = e_shoff + i * e_shentsize
            sh_name, _t, _f, _a, sh_offset, sh_size = struct.unpack_from("<IIQQQQ", cubin, off)
            return sh_name, sh_offset, sh_size

        _, str_off, str_size = section(e_shstrndx)
        strtab = cubin[str_off:str_off + str_size]
        wanted = f".nv.info.{name}".encode()
        for i in range(e_shnum):
            sh_name, off, size = section(i)
            end = strtab.index(b"\0", sh_name)
            if strtab[sh_name:end] != wanted:
                continue
            data = cubin[off:off + size]
            p, n = 0, 0
            while p + 2 <= len(data):
                fmt, attr = data[p], data[p + 1]
                p += 2
                if fmt == 0x01:            # EIFMT_NVAL
                    pass
                elif fmt in (0x02, 0x03):  # EIFMT_BVAL / EIFMT_HVAL: a 16-bit value
                    p += 2
                elif fmt == 0x04:          # EIFMT_SVAL: 16-bit size, then the payload
                    (sz,) = struct.unpack_from("<H", data, p)
                    p += 2 + sz
                else:
                    return None
                if attr == 0x17:           # EIATTR_KPARAM_INFO
                    n += 1
            return n
        return None
    except Exception:
        return None


class Driver:
    """What a backend implements: load a binary, launch it, and name the
    compile target it wants.

    Three class attributes carry the facts the launcher used to assume were
    CUDA's. Each was a CUDA literal in the launch path until the Metal port
    pressed on it, and each default is the CUDA answer, so a driver that says
    nothing behaves exactly as before.
    """

    #: The key of the compiled artifact in `CompiledKernel.asm`. CUDA emits
    #: "cubin"; Metal emits "msl"; ROCm emits "hsaco".
    artifact_kind = "cubin"

    #: Triton's ABI (>= 3.6) passes a global-scratch and a profile-scratch
    #: pointer after the kernel's own arguments. A backend whose launch ABI
    #: has no such slots says so rather than receiving two stray zeros.
    wants_scratch_params = True

    def load(self, binary: bytes, name: str, shared: int):  # pragma: no cover - interface
        raise NotImplementedError

    def launch(self, function, grid, block, shared: int, stream: int, params,
               names=None, types=None, trailing=None) -> None:  # pragma: no cover
        raise NotImplementedError

    def trailing_buffers(self, metadata):
        """Buffers the COMPILED kernel declares after the caller's arguments,
        which only this backend knows how to build and read.

        A backend whose emitter appends parameters of its own — a device-print
        record area, a device-assert status area — declares them in the compile
        metadata, and a driver that binds only the caller's arguments leaves
        those slots unbound. Metal does not complain: `packArguments` binds the
        tuple it is given at indices 0..n-1 and nothing checks the count, so a
        `tl.device_assert` the author explicitly kept writes into a slot nobody
        bound and the failure is LOST (measured 2026-09-17, triton-ext,
        `probe_ki19_driver_protections.py`: a failing assert ran to completion
        and raised nothing). CUDA has had the matching protection since this
        launcher existed — `CudaDriver.launch` refuses when the parameter count
        is not exactly what the cubin declares.

        This is the seam for that: `prepare` asks the driver ONCE per
        compilation what the kernel declares beyond the call, and `launch`
        hands the answer back. The value is opaque to the launcher — naming
        what is in it would put a backend's ABI in a file that must not know
        one. CUDA declares nothing and returns None.
        """
        return None

    def block_for(self, metadata):
        """Threads per block, from the compiled metadata.

        `num_warps * warp_size` on CUDA. Not universal: triton-msl documents
        that its C++ path can overwrite `metadata.block_size` with a value
        meant for a different launch shape, so the Metal driver reads the
        emitted kernel's own size instead of computing one.
        """
        return (32 * int(metadata.num_warps), 1, 1)

    def target(self):  # pragma: no cover - interface
        raise NotImplementedError

    def max_shared_memory_per_block(self):  # pragma: no cover - interface
        """The device's per-block shared-memory ceiling, in bytes.

        A backend that cannot ask raises, and the module-level helper turns
        that into None — which every caller reads as "do not prune"."""
        raise NotImplementedError


class CudaDriver(Driver):
    """libcuda through ctypes. The context is the primary context of the
    allocator's current device, already alive through the runtime API."""

    _inst: Optional["CudaDriver"] = None
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    # The OPT-IN maximum, not the 48 KB default: a kernel that asks for more
    # than 48 KB must opt in per function, and this is the ceiling on that ask.
    # It is the number an over-large tile is refused against.
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97

    @classmethod
    def instance(cls) -> "CudaDriver":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def target(self):
        """`GPUTarget("cuda", <capability>, 32)` — what `target()` returned
        as a literal before the driver registry existed."""
        from triton.backends.compiler import GPUTarget
        return GPUTarget("cuda", _compute_capability(), 32)

    def __init__(self) -> None:
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("NeuroBrix launcher: libcuda not found")
        self.lib = lib
        self._check(lib.cuInit(0), "cuInit")
        # ctypes infers the type of every argument at every call unless the signature is
        # declared. The launch is called six hundred times per decoded token, so its signature
        # is declared once here and the call passes plain integers (2026-09-08).
        lib.cuLaunchKernel.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._modules: Dict[Tuple[int, bytes], ctypes.c_void_p] = {}
        self._param_counts: Dict[int, Optional[int]] = {}     # function handle -> parameters the cubin declares
        # (function handle, argument kinds) -> the C cells and the pointer array reused at every
        # launch of that function: the buffer is written in place, never rebuilt (2026-09-08).
        self._arg_buffers: Dict[Tuple[int, tuple], tuple] = {}

    def _check(self, ret: int, what: str) -> None:
        if ret != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(ret, ctypes.byref(msg))
            raise RuntimeError(f"NeuroBrix launcher: {what} failed ({ret}: "
                               f"{(msg.value or b'?').decode()})")

    def _ensure_context(self) -> None:
        ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if ctx.value:
            return
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        dev = ctypes.c_int(int(DeviceAllocator.get_device()))
        self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self._check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def compute_capability(self) -> int:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        dev = ctypes.c_int(int(DeviceAllocator.get_device()))
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(major), self.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute")
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(minor), self.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute")
        return major.value * 10 + minor.value

    def max_shared_memory_per_block(self) -> int:
        """The device's opt-in shared-memory ceiling per block, in bytes.

        Asked of the DRIVER, not of a table of names. A table is right until
        the next card ships; a driver query is right forever. The vendor YAMLs
        keep their role — good defaults and the name in a certificate path —
        but a hard limit is never one of their answers.

        Read once per process and cached by the caller, never per launch.
        """
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        dev = ctypes.c_int(int(DeviceAllocator.get_device()))
        out = ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(
            ctypes.byref(out),
            self.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev),
            "cuDeviceGetAttribute")
        return int(out.value)

    def load(self, binary: bytes, name: str, shared: int):
        self._ensure_context()
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        dev = int(DeviceAllocator.get_device())
        key = (dev, binary)
        module = self._modules.get(key)
        if module is None:
            module = ctypes.c_void_p()
            buf = ctypes.create_string_buffer(binary, len(binary))
            self._check(self.lib.cuModuleLoadData(ctypes.byref(module), buf), "cuModuleLoadData")
            self._modules[key] = module
        function = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), f"cuModuleGetFunction({name})")
        self._param_counts[int(function.value)] = _cubin_param_count(binary, name)
        if shared > 48 * 1024:
            self._check(self.lib.cuFuncSetAttribute(function, self.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ctypes.c_int(shared)), "cuFuncSetAttribute")
        return function

    def launch(self, function, grid, block, shared: int, stream: int, params,
               names=None, types=None, trailing=None) -> None:
        # `types` is for a backend that packs its scalars into one buffer and
        # needs the field offsets; CUDA binds each parameter to its own slot and
        # has no use for it.
        if trailing is not None:
            raise RuntimeError(
                "NeuroBrix launcher: trailing buffers were computed for this "
                f"kernel ({trailing!r}) and the CUDA driver cannot bind them. "
                "It declares none, so this is a driver/metadata mismatch, not "
                "something to launch past.")
        # Two refusals BEFORE anything reaches the device (the launcher
        # contract's ownership rules, checked by `verify_driver_contract`):
        # the argument list must be exactly what the cubin declares, and
        # every non-null pointer must be an address this engine's allocator
        # handed out. A kernel launched past either fault reads garbage or
        # foreign memory, and a CUDA fault is sticky for the whole context.
        expected = self._param_counts.get(int(function.value))
        if expected is not None and len(params) != expected:
            raise TypeError(f"NeuroBrix launcher: the kernel declares {expected} parameters, "
                            f"{len(params)} given — refused, not launched")
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        for i, (kind, value) in enumerate(params):
            if kind == "ptr" and value and not DeviceAllocator.holds(int(value)):
                # The refusal names the parameter: "which buffer" is the
                # whole question the caller then has to answer (Ming's
                # embedding, 2026-09-14, refused an address nobody could
                # attribute from the address alone).
                what = (f"parameter '{names[i]}'" if names and i < len(names)
                        else f"parameter #{i}")
                raise ValueError(f"NeuroBrix launcher: device address {int(value):#x} ({what}) was not "
                                 f"handed out by the allocator — refused, not launched")
        # A caller may give one or two extents (the contract's checker does); CUDA wants three.
        gx, gy, gz = (tuple(int(g) for g in grid) + (1, 1, 1))[:3]
        bx, by, bz = (tuple(int(b) for b in block) + (1, 1, 1))[:3]
        if gx * gy * gz <= 0:
            return
        # The argument buffer is built ONCE per (function, argument kinds) and written in place
        # afterwards: a decode step launches six hundred kernels, and allocating one ctypes object
        # per parameter plus a pointer array per launch was a fifth of the launcher's cost
        # (2026-09-08). The buffer belongs to this driver and this function, and every launch of
        # that function overwrites it before the call — a launch is synchronous from Python's side
        # (the values are read by cuLaunchKernel before it returns), so one buffer is enough.
        kinds = tuple(k for k, _ in params)
        cache_key = (int(function.value), kinds)
        slot = self._arg_buffers.get(cache_key)
        if slot is None:
            storage = [_ctypes_slot(kind) for kind in kinds]
            arr = (ctypes.c_void_p * len(storage))(*[ctypes.addressof(c) for c in storage])
            slot = self._arg_buffers[cache_key] = (storage, arr)
        storage, arr = slot
        for cell, (_kind, v) in zip(storage, params):
            cell.value = v
        rc = self.lib.cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, arr, None)
        if rc != 0:
            self._check(rc, "cuLaunchKernel")


    # -- the contract surface (`triton/launcher_contract.py`): compile once,
    # launch many times; streams and events are the allocator's. The same
    # checker that gates the Metal driver gates this object, unchanged.
    backend = "cuda"

    def compile(self, jit_fn, signature, constexprs, num_warps: int = 4,
                specialization=None, num_stages=None):
        from triton.compiler import ASTSource, compile as triton_compile
        from neurobrix.triton.launcher_contract import ArgSlot
        specialization = dict(specialization or {})
        params = list(jit_fn.params)
        names = [kp.name for kp in params]
        unknown = set(signature) - set(names)
        if unknown:
            raise ValueError(f"NeuroBrix launcher: signature names {sorted(unknown)} are not parameters of {jit_fn.__name__}")
        spec = []
        bound = {}
        for kp in params:
            kind = signature.get(kp.name)
            if kp.is_constexpr or kind == "constexpr":
                if kp.name not in constexprs:
                    raise ValueError(f"NeuroBrix launcher: constexpr {kp.name!r} has no value")
                bound[kp.name] = constexprs[kp.name]
                spec.append(("constexpr", constexprs[kp.name]))
                continue
            if kind is None:
                raise ValueError(f"NeuroBrix launcher: parameter {kp.name!r} is missing from the signature")
            bound[kp.name] = 0
            spec.append((kind, specialization.get(kp.name)))   # None: no marker; "" / "D": Triton's own spellings
        _, _, backend = _binder(jit_fn)
        options = {"num_warps": int(num_warps)}
        if num_stages is not None:
            options["num_stages"] = int(num_stages)
        _forward_debug(jit_fn, options)
        # `_pack_args` builds the compile options from its KWARGS argument (the
        # `options` one only feeds the cache key), so the same dict goes in both.
        options, sig, cexprs, attrs = jit_fn._pack_args(backend, options, bound, spec, options)
        src = ASTSource(jit_fn, sig, cexprs, attrs)
        # A backend that cannot emit for the device may WARN and compute on the
        # CPU instead. Numbers come back, nothing raises, and the caller has an
        # answer from hardware it did not ask for. Correct-or-refuse applies to
        # the compile as much as to the launch.
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as _w:
            _warnings.simplefilter("always")
            compiled = triton_compile(src, target=target(), options=options.__dict__)
        try:
            from neurobrix.triton.metal_backend import backend_fallback_markers
            _markers = backend_fallback_markers()
        except Exception:                              # noqa: BLE001
            _markers = ()
        if _markers:
            for _warn in _w:
                _msg = str(_warn.message)
                if any(_m in _msg for _m in _markers):
                    raise RuntimeError(
                        f"the Metal backend could not emit {kernel.__name__} for "
                        f"this device and fell back to the CPU. Refusing the "
                        f"result: it would be an answer from hardware the caller "
                        f"did not ask for, and it is indistinguishable from a "
                        f"correct one. The backend said: {_msg.strip()[:400]}")
        for _warn in _w:                               # keep every other warning visible
            _warnings.warn_explicit(_warn.message, _warn.category,
                                    _warn.filename, _warn.lineno)
        md = compiled.metadata
        if getattr(md, "num_ctas", 1) != 1 or getattr(md, "global_scratch_size", 0) or getattr(md, "profile_scratch_size", 0):
            raise RuntimeError(f"NeuroBrix launcher: {jit_fn.__name__} needs clusters or scratch memory the CUDA client does not provide yet")
        cubin = compiled.asm["cubin"]
        function = CudaDriver.instance().load(cubin, md.name, md.shared)
        binding = []
        ordered = {}
        for kp in params:
            kind = signature.get(kp.name)
            if kp.is_constexpr or kind == "constexpr":
                continue
            ordered[kp.name] = kind
            binding.append(ArgSlot(index=len(binding), name=kp.name, is_pointer=kind.startswith("*"),
                                   dtype=kind if not kind.startswith("*") else kind[1:]))
        return CudaCompiledKernel(md.name, bytes(cubin), 32 * int(md.num_warps), int(md.shared), constexprs,
                                  specialization, binding, ordered, function)

    @staticmethod
    def _allocator():
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
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

# ---------------------------------------------------------------------------
# Which driver this process launches through — the one place a backend is named
# ---------------------------------------------------------------------------
#
# `launch()` used to end in `CudaDriver.instance()`. The `Driver` base class
# was already there, so the second backend was anticipated; there was just no
# way to install one. This is that way, and it is a table rather than a
# branch: adding ROCm is a row.
#
# The backend NAME is not decided here either. It comes from the seam that
# already resolves it for the allocator, so the engine has one detection
# rather than two opinions.

_DRIVER: Optional[Driver] = None



def register_driver(driver: Optional[Driver]) -> None:
    """Install the driver this process launches through. `None` clears it."""
    global _DRIVER, _TARGET
    _DRIVER = driver
    _TARGET = None          # the target is the driver's answer


def active_driver() -> Driver:
    """The driver in force, resolved once from the detected backend."""
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _resolve_driver()
    return _DRIVER


def _resolve_driver() -> Driver:
    from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
    try:
        name = _detect_gpu_backend()
    except Exception:
        return CudaDriver.instance()
    if name == "metal":
        # WHICH Metal backend is a profile selection, and a driver implements ONE
        # backend's launch ABI (ours reads the fork's MSL conventions for scalar
        # binding). So the SEAM answers which module launches, or refuses by
        # name. A static {"metal": <the fork's driver>} table was the previous
        # answer, and it is what launched triton-ext-compiled kernels through the
        # fork's ABI: scalars after the first arrived as 0, every mask went
        # false, and the kernel returned exact zeros without failing.
        from neurobrix.triton.metal_backend import nbx_driver_module
        from importlib import import_module
        return import_module(nbx_driver_module()).driver()
    # CUDA is the built-in default, which keeps the Dell's path byte-identical
    # to before this seam existed.
    return CudaDriver.instance()


def _unsupported(kind):
    raise RuntimeError(f"NeuroBrix launcher: unsupported scalar kind {kind!r}")


# ---------------------------------------------------------------------------
# Argument packing: the signature dictates the kind of every parameter
# ---------------------------------------------------------------------------

_INT_KINDS = {"i1": "i8", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
              "u1": "u8", "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64"}


_CTYPES_SLOT = {"ptr": ctypes.c_uint64, "u64": ctypes.c_uint64, "bits64": ctypes.c_uint64,
                "i64": ctypes.c_int64, "u32": ctypes.c_uint32, "bits32": ctypes.c_uint32,
                "i32": ctypes.c_int32, "u16": ctypes.c_uint16, "bits16": ctypes.c_uint16,
                "i16": ctypes.c_int16, "u8": ctypes.c_uint8, "i8": ctypes.c_int8,
                "f64": ctypes.c_double}


def _ctypes_slot(kind: str):
    """The C cell one launch parameter of this kind lives in — the same widths the packing
    produced when it built a fresh object per launch."""
    cell = _CTYPES_SLOT.get(kind)
    if cell is None:
        _unsupported(kind)
    return cell()


def _pack_param(ty: str, value: Any) -> Tuple[str, Any]:
    """One launch parameter as (kind, integer-or-float) — a pointer as its
    address, a float scalar as the bit pattern of its storage type (what
    Triton's C launcher does with pack_fp16/pack_fp32/pack_fp64)."""
    if ty.startswith("*"):
        return "ptr", int(value.data_ptr()) if hasattr(value, "data_ptr") else int(value)
    if ty in _INT_KINDS:
        return _INT_KINDS[ty], int(value)
    if ty in ("fp32", "f32"):
        return "bits32", struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if ty == "fp64":
        return "bits64", struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    if ty == "fp16":
        import numpy as np
        return "bits16", int(np.array(float(value), dtype=np.float16).view(np.uint16))
    if ty == "bf16":
        from neurobrix.kernels.nbx_tensor import float32_to_bf16_bits
        import numpy as np
        return "bits16", int(float32_to_bf16_bits(np.array([float(value)], dtype=np.float32))[0])
    raise RuntimeError(f"NeuroBrix launcher: cannot pack a parameter of type {ty!r}")


# ---------------------------------------------------------------------------
# The launch: Triton's binder + compiler, our driver
# ---------------------------------------------------------------------------

class _Prepared:
    __slots__ = ("function", "signature", "shared", "num_warps", "block", "name",
                 "trailing")

    def __init__(self, function, signature, shared, num_warps, name, block=None,
                 trailing=None):
        self.function = function
        self.signature = signature
        self.shared = shared
        self.num_warps = num_warps
        self.block = block if block is not None else (32 * num_warps, 1, 1)
        self.name = name
        #: What the compiled kernel declares AFTER the caller's arguments, as
        #: only its own driver can describe it (`Driver.trailing_buffers`).
        #: Computed once per compilation, because it is a property of the
        #: compilation and reading it per launch would cost a metadata walk on
        #: the hot path.
        self.trailing = trailing


# ---------------------------------------------------------------------------
# Specialisation: Triton's binder rules, written once in Python and measured
# against the C++ specialiser (tests/unit/kernels/test_launcher.py) — the C++
# one imports torch the moment it meets a tensor-like argument (the addendum's
# hypothesis, confirmed by the import stack on 2026-09-05).
# ---------------------------------------------------------------------------

# Triton's short element names for a pointer's dtype (torch / tl spellings in).
_POINTEE = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float64": "fp64",
            "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
            "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64", "bool": "i1",
            "float8_e4m3fn": "fp8e4nv", "float8_e5m2": "fp8e5", "float8_e4m3fnuz": "fp8e4b8", "float8_e5m2fnuz": "fp8e5b16",
            "fp16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp64": "fp64"}
_INT32_MIN, _INT32_MAX = -(2 ** 31), 2 ** 31
_INT64_MIN, _INT64_MAX = -(2 ** 63), 2 ** 63


_PTR_TY: Dict[Any, str] = {}      # a tensor dtype -> the pointee spelling Triton's binder gives it


def specialize_arg(arg, specialize: bool = True, align: bool = True):
    """(type, attr) of one runtime argument, exactly as Triton's binder:
    a pointer is `*<dtype>` with 'D' when 16-byte aligned; an int is i32 /
    i64 / u64 by range, 'D' when divisible by 16, the constant 1 when
    specialised; a float is fp32; a bool is u1; None is a constexpr. The attr
    is None when the parameter is not specialised at all."""
    if arg is None:
        return ("constexpr", None)
    if hasattr(arg, "data_ptr") and hasattr(arg, "dtype"):
        dt = arg.dtype
        # The pointee's spelling is a property of the dtype, not of the call: computing it per
        # launch cost six string operations per pointer argument, and a decode step launches six
        # hundred kernels of fifteen to twenty arguments each (2026-09-08).
        ty = _PTR_TY.get(dt)
        if ty is None:
            name = getattr(dt, "name", None) or str(dt).split(".")[-1]
            ty = _PTR_TY[dt] = "*" + _POINTEE.get(str(name), str(name))
        if not specialize:
            return (ty, None)
        return (ty, "D" if (align and int(arg.data_ptr()) % 16 == 0) else "")
    if isinstance(arg, bool):
        return ("u1", None)
    if isinstance(arg, int):
        if specialize and arg == 1:
            return ("constexpr", 1)
        if _INT32_MIN <= arg < _INT32_MAX:
            ty = "i32"
        elif _INT64_MIN <= arg < _INT64_MAX:
            ty = "i64"
        elif 0 <= arg < 2 ** 64:
            ty = "u64"
        else:
            raise RuntimeError(f"NeuroBrix launcher: integer argument out of range: {arg}")
        if not specialize:
            return (ty, None)
        return (ty, "D" if (align and arg % 16 == 0) else "")
    if isinstance(arg, float):
        return ("fp32", None)
    try:
        import triton.language as tl
        if isinstance(arg, tl.dtype):
            return ("constexpr", arg)
    except Exception:
        pass
    raise TypeError(f"NeuroBrix launcher: cannot specialise an argument of type {type(arg).__name__}")


def _param_table(kernel):
    """The kernel's parameter list as plain tuples, read once: a launch reads five attributes per
    parameter, and there are fifteen to twenty of them on the kernels a decode step calls."""
    table = _PARAM_TABLES.get(id(kernel))
    if table is None:
        table = _PARAM_TABLES[id(kernel)] = tuple(
            (kp.name, kp.is_constexpr, kp.has_default, kp.default,
             not kp.do_not_specialize, not kp.do_not_specialize_on_alignment,
             getattr(kp, "annotation_type", None))
            for kp in kernel.params)
    return table


_PARAM_TABLES: Dict[int, tuple] = {}


def nbx_binder(kernel, args, kwargs):
    """(bound_args, specialization, options) — the same triple Triton's
    generated binder returns, from the kernel's parameter list."""
    bound = {}
    spec = []
    positional = list(args)
    options = dict(kwargs)
    for i, (name, is_constexpr, has_default, default, specialize_p, align_p, ann) in enumerate(_param_table(kernel)):
        if i < len(positional):
            value = positional[i]
        elif name in options:
            value = options.pop(name)
        elif has_default:
            value = default
        else:
            raise TypeError(f"NeuroBrix launcher: {kernel.__name__}() missing argument {name!r}")
        bound[name] = value
        if is_constexpr:
            spec.append(("constexpr", value))
            continue
        specialize = specialize_p
        align = align_p
        if ann:
            if isinstance(ann, str) and (ann == "u1" or ann[:2] in ("fp", "bf")):
                specialize = False
            if specialize:
                spec.append((ann,) + tuple(specialize_arg(value, True, align)[1:]))
            else:
                spec.append((ann, None))
            continue
        spec.append(specialize_arg(value, specialize, align))
    if len(positional) > len(_param_table(kernel)):
        raise TypeError(f"NeuroBrix launcher: {kernel.__name__}() takes {len(kernel.params)} arguments, {len(positional)} given")
    return bound, spec, options


_binders: Dict[int, tuple] = {}     # id(kernel) → (kernel_cache, key_cache, backend)


def reset_caches() -> None:
    """Drop every compiled kernel this launcher is holding.

    A `_Prepared` holds a handle the DRIVER produced — a CUDA module, a Metal
    pipeline — which is bound to the device and the runtime that produced it.
    Anything that replaces the runtime underneath must drop these too, or the
    next launch drives a handle from a runtime that no longer exists and
    refuses a perfectly good tensor.

    Called by `metal_device.reset_runtime_for_tests`, which is the only thing
    that replaces a runtime today. The caches and the runtime have one
    lifetime; clearing a subset is worse than clearing none.
    """
    global _TARGET
    _binders.clear()
    _PARAM_TABLES.clear()
    _PTR_TY.clear()
    _TARGET = None


def _binder(kernel):
    b = _binders.get(id(kernel))
    if b is None:
        from triton.compiler import make_backend
        b = _binders[id(kernel)] = ({}, {}, make_backend(target()))
    return b


def _forward_debug(kernel, options: Dict[str, Any]) -> None:
    """Carry the kernel's `debug` flag into the compile options, as Triton's
    own `JITFunction.run` does (`kwargs.get("debug", self.debug) or
    knobs.runtime.debug`). `@triton.jit(debug=True)` is what keeps a
    `tl.device_assert` in the binary — the gather/scatter kernels rely on it
    to trap an out-of-range index — so a launcher that dropped it would
    compile the traps out silently."""
    if "debug" not in options:
        global _DEBUG_KNOB
        if _DEBUG_KNOB is None:
            _DEBUG_KNOB = False
            try:
                from triton import knobs
                _DEBUG_KNOB = bool(knobs.runtime.debug)
            except Exception:
                pass
        options["debug"] = bool(getattr(kernel, "debug", False)) or _DEBUG_KNOB


_DEBUG_KNOB = None      # the process's Triton debug knob, read once (a launch may not import)


def prepare(kernel, args, kwargs) -> Tuple[_Prepared, Dict[str, Any]]:
    """Specialise (our binder), compile (Triton's compiler with the engine's
    target), load (our driver) — once per specialisation and device.

    A decode step launches six hundred kernels and every one of them comes back
    here: what a repeat launch pays must be a dict lookup, nothing more. The
    specialisation IS the identity of the compilation, so it keys the cache
    directly; Triton's `compute_cache_key` (which builds and hashes a string)
    runs on a miss only, where a compile is about to happen anyway. The device
    comes from the allocator's own cache — `get_device` asks the runtime, and a
    runtime round trip per launch is what a launcher must not do (2026-09-08:
    the launcher held ten times the decode rate of the engine).
    """
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.runtime.jit import compute_cache_key
    kernel_cache, key_cache, backend = _binder(kernel)
    kwargs = dict(kwargs)
    _forward_debug(kernel, kwargs)      # into kwargs: `_pack_args` parses the compile options from THEM
    bound_args, specialization, options = nbx_binder(kernel, args, kwargs)
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, _cached_device_idx
    dev = _cached_device_idx()
    if dev is None:
        dev = int(DeviceAllocator.get_device())
    fast_key = (tuple(specialization), tuple(options.items()) if options else (), dev)
    prep = kernel_cache.get(fast_key)
    if prep is None:
        key = (compute_cache_key(key_cache, specialization, options), dev)
        prep = kernel_cache.get(key)
    if prep is None:
        options, signature, constexprs, attrs = kernel._pack_args(backend, kwargs, bound_args, specialization, options)
        src = ASTSource(kernel, signature, constexprs, attrs)
        compiled = triton_compile(src, target=target(), options=options.__dict__)
        md = compiled.metadata
        if getattr(md, "num_ctas", 1) != 1:
            raise RuntimeError("NeuroBrix launcher: cluster launches (num_ctas > 1) are not implemented")
        if getattr(md, "global_scratch_size", 0) or getattr(md, "profile_scratch_size", 0):
            raise RuntimeError(f"NeuroBrix launcher: {kernel.__name__} asks for scratch memory "
                               f"the launcher does not provide yet")
        drv = active_driver()
        artifact = compiled.asm.get(drv.artifact_kind)
        if artifact is None:
            raise RuntimeError(
                f"NeuroBrix launcher: {kernel.__name__} compiled to "
                f"{sorted(compiled.asm)} but {drv.__class__.__name__} takes "
                f"{drv.artifact_kind!r}")
        function = drv.load(artifact, md.name, md.shared)
        prep = _Prepared(function, signature, md.shared, md.num_warps, md.name,
                         drv.block_for(md), trailing=drv.trailing_buffers(md))
        kernel_cache[key] = prep
    kernel_cache[fast_key] = prep
    return prep, bound_args


_TRACE = os.environ.get("NBX_LAUNCH_TRACE")     # a file: one line per launch, "<kernel>\t<grid>"


# The decode replay (`neurobrix.triton.replay`) records one step's FINAL launches and
# replays them without the Python band above — the engine's fast decode path, worth 8x
# on a text row. It used to record by wrapping Triton's `CompiledKernel.run`; nothing on
# the Triton branch calls that any more (R33, third peel 2026-09-05), so the recorder
# needs a seam HERE or it records a step with no kernel in it. Armed for the window of
# one recorded step and cleared after: the hot path pays one global read and a None test.
_RECORDER = None


def set_launch_recorder(fn) -> None:
    """Record every launch this launcher issues: `fn(prepared, grid, params)`, with the
    launch as the DRIVER will take it — the resolved kernel, three extents, and the
    parameters already packed. `None` stops recording."""
    global _RECORDER
    _RECORDER = fn


def launch(kernel, grid, *args, **kwargs):
    """The one entry point: `launch(kernel, grid, *args, **constexprs_and_options)`."""
    prep, bound_args = prepare(kernel, args, kwargs)
    if _TRACE:
        with open(_TRACE, "a") as fh:
            fh.write(f"{kernel.__name__}\t{grid if not callable(grid) else 'callable'}\n")
    if callable(grid):
        grid = grid(bound_args)
    grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
    runtime = [(name, ty) for name, ty in prep.signature.items()
               if ty != "constexpr"]
    params = [_pack_param(ty, bound_args[name]) for name, ty in runtime]
    # The parameter NAMES, in the same order. A backend whose artifact
    # declares its arguments by name — and may declare only the ones the
    # compiled kernel kept — binds by name rather than by position, which is
    # the only mapping that stays correct when the two lists differ in length.
    names = [name for name, _ty in runtime]
    drv = active_driver()
    if drv.wants_scratch_params:
        params.append(("ptr", 0))    # global scratch (Triton ≥ 3.6 ABI)
        params.append(("ptr", 0))    # profile scratch
    if _RECORDER is not None:
        _RECORDER(prep, grid, params)
    # The Triton TYPE of every runtime parameter, in the same order. A backend
    # that packs its scalars into one buffer computes the field offsets from
    # these; our `(kind, value)` pairs have already lost the distinction between
    # an i16 and a bf16, and packing to the wrong offset is silent.
    drv.launch(prep.function, grid, prep.block, prep.shared, _stream(), params,
               names=names, types=[ty for _name, ty in runtime],
               trailing=prep.trailing)


def _stream() -> int:
    """The legacy default stream (0): every house kernel launches on it today,
    and the allocator's copies order against it."""
    return 0



# ---------------------------------------------------------------------------
# The autotune correctness screen: every candidate is checked before any is timed
# ---------------------------------------------------------------------------
#
# An autotuner ranks configs by speed. That is safe only while every config
# computes the same thing, and on a backend where one does not, speed is
# exactly the wrong tiebreak: a kernel that writes half its output does half
# the stores, so it is genuinely faster, so it wins. Measured 2026-09-07 on
# Apple — fp16 mm [64,32]@[32,64] selected a config that left 32 of 64
# columns zero, deterministically, with no error raised anywhere.
#
# Nothing here names a backend, and the tolerance is read from the hardware
# profile rather than written down: what separates "a different summation
# order" from "a different answer" is a property of the device's arithmetic,
# and the profile is where the engine keeps those.
#
# A config that diverges is excluded from the timing and recorded with its
# shape and its deviation. It is never dropped silently, because a config
# quietly missing from a sweep looks like a config that lost on speed.

class ScreenedOut(NamedTuple):
    """One config the screen refused, and why."""
    kernel: str
    key: tuple
    config: str
    dtype: str
    deviation: float
    tolerance: float


class Unscreened(NamedTuple):
    """One key where a configuration was seated WITHOUT an oracle.

    Not a failure and not a refusal: the engine must run. It is a PROVENANCE.
    "This configuration was validated" and "this configuration was the fastest
    among candidates nobody verified" are different statements, and until this
    record existed they were written the same way and read the same way.
    """
    kernel: str
    key: tuple
    candidates: int
    reason: str


_SCREENED: List[ScreenedOut] = []
_UNSCREENED: List[Unscreened] = []
_SCREEN_CACHE: Dict[int, set] = {}      # id(tuner) -> keys already screened


def screened_out() -> List[ScreenedOut]:
    """Every config the correctness screen excluded, in order."""
    return list(_SCREENED)


#: Keys the screen adjudicated, with what adjudicated them — the converse of
#: `_UNSCREENED`: a record that says only what was NOT verified lets a silent
#: entry read as verified. `capture()` stamps these `screened: True` with the
#: adjudicator's name.
_ADJUDICATED: dict = {}


def adjudicated() -> dict:
    return dict(_ADJUDICATED)


def unscreened() -> List[Unscreened]:
    """Every key whose seated configuration no oracle adjudicated.

    Read this wherever a chosen configuration is RECORDED, so the record
    carries what it is. The certified directory is filled only by
    `neurobrix autotune certify`, which runs its own fp64 oracle over every
    candidate; nothing on this path may ever reach it.
    """
    return list(_UNSCREENED)


def clear_screened() -> None:
    _SCREENED.clear()
    _UNSCREENED.clear()
    _SCREEN_CACHE.clear()


def _no_oracle_reason(oracle) -> str:
    """Why no oracle adjudicated this key — the provider, or its answer."""
    if _SCREEN_ORACLE is None:
        return "no oracle provider is installed"
    if oracle is None:
        return ("the oracle provider covers no oracle for this kernel "
                "(the GEMM class and, since 2026-09-13, the convolution family "
                "are covered; anything else is decided by the bare vote)")
    try:
        from neurobrix.kernels.screen_oracle import last_refusal
        why = last_refusal()
    except Exception:                                  # noqa: BLE001
        why = None
    return why or "the oracle produced no reference"



def _screen_on_windows(tuner, kernel_name, _rk, configs, args, meta, buffers, budget, total):
    """Screen an over-budget shape on bounded row windows of its output. None if impossible.

    The candidates' own bytes are sliced to the same rows as the oracle, so both sides are the
    same rows of the same tensor. Only the OUTPUT is windowed and compared; the inputs are not
    re-read, which is the one thing this screen checks less than the full one — said in the
    seat line rather than left to be discovered."""
    from neurobrix.kernels import screen_oracle as _so
    out_name = None
    entry = _so.ORACLES.get(getattr(getattr(tuner, "base_fn", None), "__name__", "") or "")
    if entry is not None:
        out_name = entry[1]
    named = {**dict(getattr(tuner, "nargs", None) or {}), **dict(meta or {})}
    out_tensor = named.get(out_name) if out_name else None
    if out_tensor is None:
        return None
    wins = _row_windows_for(out_tensor, budget)
    if not wins:
        return None
    try:
        shape = tuple(int(x) for x in out_tensor.shape)
        M, N = shape
        itemsize = int(out_tensor._nbytes) // max(1, M * N)
        row_bytes = N * itemsize
        out_addr = int(out_tensor.data_ptr())
    except Exception:                                   # noqa: BLE001
        return None
    ranges = _rows_to_bytes(wins, row_bytes)
    where = _describe_windows(wins, M)

    shots = []
    for config in configs:
        try:
            tuner.fn.run(*args, **{**dict(meta or {}), **config.all_kwargs()})
            shots.append((config, _snapshot_ranges(out_addr, ranges)))
        except Exception:                               # noqa: BLE001
            pass
    if len(shots) < 2:
        return _seat_unscreened(kernel_name, _rk, configs, len(shots),
                                f"over the screening budget ({total} bytes) and fewer than "
                                f"two candidates ran on the windows {where}")

    dtype_name = None
    for _a, _n, dt in buffers:
        if _a == out_addr:
            dtype_name = dt
            break

    # ONE ORACLE PER WINDOW, never over their span. Passing (first_start, last_end) asks for
    # every row between the first and last window — the whole product, which is the 18 GB
    # computation the windowing exists to avoid, and it fails silently into the consensus
    # path. Measured 2026-09-23: matmul_kernel fell through to consensus for exactly this
    # reason while sitting in ROW_WINDOWABLE.
    kept = None
    try:
        import numpy as _np
        parts = []
        for r0, r1 in wins:
            piece = _so.windowed_reference(tuner, meta, (r0, r1))
            if piece is None:
                parts = None
                break
            # CAST TO THE OUTPUT'S DTYPE before comparing, exactly as the full-screen
            # provider does. The reference is computed in float64; a candidate's bytes are
            # fp16. Comparing them raw fails with a 4x shape mismatch — which is what the
            # swallowed exception was hiding until it was made to speak.
            if dtype_name in ("bf16", "bfloat16"):
                from neurobrix.kernels.autotune_certify import f32_to_bf16_bits
                want = _np.ascontiguousarray(
                    f32_to_bf16_bits(_np.ascontiguousarray(piece, dtype=_np.float32)))
            else:
                want = _np.ascontiguousarray(piece.astype(_so._NP.get(dtype_name, _np.float32)))
            parts.append(want.tobytes())
        if parts is not None:
            ref_bytes = b"".join(parts)
            kept = [(c, blob) for c, blob in shots
                    if configs_agreeing_with_oracle([(c, blob)], ref_bytes, dtype_name)]
        else:
            print(f"[AUTOTUNE_SCREEN_WINDOWED] {kernel_name}: the windowed oracle returned "
                  f"nothing for a window of this key; falling back to consensus on the same "
                  f"windows.", flush=True)
    except Exception as exc:                            # noqa: BLE001
        # SAY why. An `except: pass` here is the exact shape this session spent a day
        # removing elsewhere: it turns "the oracle could not be computed" into "no oracle
        # covers this kernel", which is a different and false statement.
        print(f"[AUTOTUNE_SCREEN_WINDOWED] {kernel_name}: the windowed oracle raised "
              f"({type(exc).__name__}: {str(exc)[:160]}); falling back to consensus on the "
              f"same windows.", flush=True)
        kept = None
    if kept is not None:
        if not kept:
            raise RuntimeError(
                f"NeuroBrix autotune screen: {kernel_name} at key {_rk} — the fp64 oracle "
                f"contradicts EVERY candidate on {where}. Screened on windows because the "
                f"arguments ({total} bytes) exceed the screening budget; a window that "
                f"disagrees is a disagreement. Refusing to seat any of them.")
        print(f"[AUTOTUNE_SCREEN_WINDOWED] {kernel_name} at key {_rk}: arguments total "
              f"{total} bytes, over the screening budget {budget}. Screened against the fp64 "
              f"oracle on {where} — {len(kept)} of {len(shots)} candidates kept. VERIFIED ON "
              f"THOSE WINDOWS ONLY; the rest of the output and the input buffers were not "
              f"compared.", flush=True)
        _record_screen_windows(_rk, where, "fp64 oracle", len(kept), len(shots))
        return [c for c, _ in kept]

    # No windowed oracle for this kernel: cluster the candidates by agreement on the windows.
    # Strictly less than an oracle and strictly more than nothing, and it says which it is.
    groups = {}
    for c, blob in shots:
        groups.setdefault(blob, []).append(c)
    best = max(groups.values(), key=len)
    print(f"[AUTOTUNE_SCREEN_WINDOWED] {kernel_name} at key {_rk}: arguments total {total} "
          f"bytes, over the screening budget {budget}, and no row-windowed oracle covers this "
          f"kernel. Screened by CONSENSUS on {where} — {len(best)} of {len(shots)} agree. "
          f"No oracle adjudicated this key.", flush=True)
    _record_screen_windows(_rk, where, "consensus (no windowed oracle)", len(best), len(shots))
    return best


#: What each windowed screen verified, so a proof can name it.
SCREEN_WINDOWS: dict = {}


def _record_screen_windows(key, where: str, by: str, kept: int, candidates: int) -> None:
    SCREEN_WINDOWS[str(key)] = {"windows": where, "adjudicated_by": by,
                                "kept": kept, "candidates": candidates}


def screen_windows_of(key):
    """The windows a key was screened on, for the proof. None when it was screened whole."""
    return SCREEN_WINDOWS.get(str(key))


def _seat_unscreened(kernel: str, key, configs, candidates: int, reason: str):
    """Return the configs, having said plainly that nothing verified them.

    The bare screen may still rank by speed — the engine never refuses to run.
    What it may no longer do is produce a line that reads as a validation.
    """
    _UNSCREENED.append(Unscreened(kernel, key, candidates, reason))
    print(f"[AUTOTUNE_UNSCREENED] {kernel} at key {key}: {reason}. The "
          f"configuration seated here is the FASTEST AMONG {candidates} "
          f"CANDIDATES THAT NOTHING VERIFIED — it is not a validated setting "
          f"and it is never written to the certified directory.", flush=True)
    return configs


def _screen_rtol(dtype_name: str):
    """The tolerance for one dtype, from the hardware profile. None = exact."""
    from neurobrix.kernels.ops._configs import active_vendor_profile

    table = active_vendor_profile().get("autotune_screen_rtol")
    if table is None:
        raise RuntimeError(
            "the hardware profile declares no `autotune_screen_rtol`: the "
            "autotune correctness screen will not invent a tolerance, and "
            "without one it cannot tell a reordered sum from a wrong answer")
    return table.get(dtype_name)


def _writable_buffers(values):
    """Every device buffer among these arguments, with its byte length.

    Returns None when any of them is a NON-CONTIGUOUS view, and the caller
    then skips screening rather than guessing.

    The snapshot and restore below copy a CONTIGUOUS span from `data_ptr()`.
    For a strided view that span is not the tensor: it covers the gaps
    between the view's elements, which belong to other tensors in the same
    allocation. Restoring it writes stale bytes over live data somewhere
    else entirely — which is exactly what happened, and it showed up as eight
    unrelated tests failing in the full suite while passing alone.
    """
    out = []
    for value in values:
        if not (hasattr(value, "data_ptr") and hasattr(value, "_nbytes")):
            continue
        contiguous = getattr(value, "is_contiguous", None)
        if callable(contiguous) and not contiguous():
            return None
        out.append((int(value.data_ptr()), int(value._nbytes),
                    getattr(getattr(value, "dtype", None), "name", "?")))
    return out



#: How many row windows an over-budget screen takes, and where. The LAST window is anchored
#: at the final row deliberately: the failure class that motivates screening a large shape is
#: index overflow, and it shows at the largest linear index or nowhere (the rack, 2026-09-23).
_SCREEN_WINDOWS = 3


def _screen_windows() -> int:
    """How many row windows an over-budget screen takes. `NBX_SCREEN_WINDOWS` is the door.

    A tunable with no door cannot be tuned, and the instruction was to set this FROM a
    measurement. Read WHEN USED, never frozen at import — a value frozen from the environment
    at import is the same defect as a literal standing in for a runtime value, and this package
    has been bitten by exactly that (`autotune_cache._dir` says so in its own docstring).

    WHAT THE COUNT DOES, AND DOES NOT, measured 2026-09-23. `per` is
    `budget // (n_win * oracle_row_bytes)`, so more windows means proportionally smaller ones
    and the oracle's TOTAL work does not move (M=100 000, N=64, 4 MiB budget):

        windows   rows each   total rows   places
              1        8192         8192        1
              3        2730         8190        3
              8        1024         8192        8

    **So this is not a cost knob.** It changes WHERE the screen looks, not what it costs; the
    cost is set by the screening budget. A cell asserting the opposite went red on its first
    run, which is how the property was found.

    WHAT THE ORACLE COSTS, chatterbox on a 16 GB V100 where its 32g-certified keys do not serve
    so all 65 keys sweep:

        oracle OFF   379.9, 378.9 s    spread   1.0 s
        oracle ON    500.3, 395.5 s    spread 104.8 s

    Every ON run is slower than every OFF run — 1.04x at the closest bound, 1.32x at the widest,
    1.18x on the means — but the ON arm's own spread is 105 s against the OFF arm's 1 s, so the
    variance lives in the oracle path and a point factor from two reps is not a number. The
    Apple side measured 31.4x for ONE selection on a 4.6 GB shape, so the factor scales with
    what is windowed and there is no constant to adopt.
    """
    import os as _os
    raw = _os.environ.get("NBX_SCREEN_WINDOWS")
    if raw is None:
        return _SCREEN_WINDOWS
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"NBX_SCREEN_WINDOWS={raw!r} is not an integer") from None
    if n < 1:
        raise ValueError(f"NBX_SCREEN_WINDOWS={n} would screen nothing; 1 is the minimum")
    return n


def _row_windows_for(out_tensor, budget_bytes: int):
    """Row ranges of a 2-D output whose total bytes fit the budget, or None.

    None means the shape cannot be windowed by rows — the caller then says so rather than
    pretending it screened something."""
    try:
        shape = tuple(int(x) for x in out_tensor.shape)
        itemsize = int(out_tensor._nbytes) // max(1, int(np.prod(shape)))
    except Exception:                                   # noqa: BLE001
        return None
    if len(shape) != 2 or itemsize <= 0:
        return None
    M, N = shape
    row_bytes = N * itemsize
    if row_bytes <= 0 or M <= 0:
        return None
    # Size the window by what the ORACLE costs, not by what the output costs. The reference
    # is computed in float64 — 8 bytes a element against 2 for an fp16 output — so a window
    # sized to the budget in output bytes needs four times the budget to adjudicate, and the
    # oracle then raises and the screen silently degrades to consensus (measured 2026-09-23,
    # matmul_kernel falling through while sitting in ROW_WINDOWABLE). The reference row is
    # what it will actually occupy.
    oracle_row_bytes = N * 8
    n_win = _screen_windows()
    per = max(1, int(budget_bytes) // (n_win * max(oracle_row_bytes, 1)))
    if per >= M:
        return [(0, M)]
    # `n_win` windows spread across the rows, the LAST always anchored at the final row:
    # the failure class that motivates screening a large shape is index overflow, and it shows
    # at the largest linear index or nowhere.
    if n_win == 1:
        wins = [(M - per, M)]
    else:
        step = (M - per) / (n_win - 1)
        wins = [(int(round(i * step)), int(round(i * step)) + per) for i in range(n_win - 1)]
        wins.append((M - per, M))
    # de-duplicate and order; overlapping windows on a short M collapse to fewer
    out, seen = [], set()
    for r0, r1 in wins:
        r0, r1 = max(0, r0), min(M, r1)
        if r1 > r0 and (r0, r1) not in seen:
            seen.add((r0, r1)); out.append((r0, r1))
    return out or None


def _describe_windows(wins, M):
    return "rows " + "; ".join(f"{r0}-{r1}" for r0, r1 in wins) + f" of {M}"


def _rows_to_bytes(wins, row_bytes):
    return [(r0 * row_bytes, (r1 - r0) * row_bytes) for r0, r1 in wins]


def _snapshot_ranges(address, ranges):
    """Copy only the given byte ranges of one device buffer to the host."""
    import ctypes

    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    blobs = []
    for off, length in ranges:
        host = (ctypes.c_char * length)()
        DeviceAllocator.memcpy(ctypes.addressof(host), address + off, length, kind=2)
        blobs.append(bytes(host))
    return b"".join(blobs)


def _snapshot(buffers):
    import ctypes

    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    shots = []
    for address, nbytes, _name in buffers:
        host = (ctypes.c_char * nbytes)()
        DeviceAllocator.memcpy(ctypes.addressof(host), address, nbytes, kind=2)
        shots.append(bytes(host))
    return shots


def _restore(buffers, shots):
    import ctypes

    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    for (address, nbytes, _name), blob in zip(buffers, shots):
        host = (ctypes.c_char * nbytes).from_buffer_copy(blob)
        DeviceAllocator.memcpy(address, ctypes.addressof(host), nbytes, kind=1)


#: The engine spells a dtype "fp16"; the hardware profiles and numpy spell it
#: "float16". One table, so the screen cannot silently fail to find a
#: tolerance and fall back to comparing floats bit-for-bit — which it did on
#: first run, excluding seven correct configs.
_DTYPE_CANON = {
    "fp16": "float16", "float16": "float16", "half": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "float32": "float32", "f32": "float32",
    "fp64": "float64", "float64": "float64", "f64": "float64",
    "i8": "int8", "int8": "int8", "i16": "int16", "int16": "int16",
    "i32": "int32", "int32": "int32", "i64": "int64", "int64": "int64",
    "u8": "uint8", "uint8": "uint8", "i1": "bool_", "bool": "bool_",
    "bool_": "bool_",
}


def _as_float64(blob: bytes, canon: str):
    """One buffer's contents as float64, whatever the engine calls its type.

    bfloat16 is not a numpy dtype: it is read as its 16 bits and widened by
    placing them in the high half of a float32, which is exactly what the
    format is.
    """
    import numpy as np

    if canon == "bfloat16":
        bits = np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64)
    return np.frombuffer(blob, dtype=np.dtype(canon)).astype(np.float64)


def _deviation(a: bytes, b: bytes, dtype_name: str):
    """(deviation, tolerance) between two results of the same buffer.

    Integers and booleans are compared bit-identically: no valid reordering
    changes an integer. Floats are compared relative to the reference's own
    magnitude, against the profile's tolerance.
    """
    import numpy as np

    canon = _DTYPE_CANON.get(str(dtype_name).lower())
    if canon is None:
        # An unknown spelling is not licence to guess a tolerance.
        return (0.0 if a == b else float("inf")), 0.0
    rtol = _screen_rtol(canon)
    if rtol is None:
        return (0.0 if a == b else float("inf")), 0.0
    x = _as_float64(a, canon)
    y = _as_float64(b, canon)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return float("inf"), float(rtol)      # one produced NaN/Inf, one did not
    scale = float(np.abs(y[finite]).max()) if finite.any() else 0.0
    if scale == 0.0:
        return (0.0 if np.array_equal(x, y) else float("inf")), float(rtol)
    return float(np.abs(x[finite] - y[finite]).max() / scale), float(rtol)


def configs_agreeing_with_oracle(results, oracle, dtype_name):
    """The configurations an oracle does not contradict, or None if there is none.

    `screen_configs` decides by CONSENSUS among the candidates, which was the
    right answer to the 2026-09-07 incident: anchoring on a nominated reference
    inverts the moment that reference is the broken one. But consensus is a
    VOTE, and a vote has two failure modes it cannot see —

      * the majority cluster is wrong in the same way, so the minority that is
        right gets excluded;
      * every candidate agrees and all are wrong, in which case the screen
        returns the whole space and says nothing.

    Neither is hypothetical on hardware nobody has looked at. The certified
    directory covers `nvidia/volta` and nothing else, so on any other card the
    runtime sweeps and this screen is the only thing between the user and a
    wrong kernel — over a space that DIFFERS by target: the same flash tile
    needs 98 304 bytes on sm_70 and 164 352 on sm_86.

    So when an oracle exists, it overrules the vote. When it does not, this
    returns None rather than pretending: the caller keeps the consensus it had
    and knows that is what it has. Refusing on an oracle nobody computed would
    be the same silence in the other direction.

    `results` are `(config, output_bytes)` pairs; the comparison and its
    tolerance are the profile's own, the very ones the screen already applies
    between candidates.
    """
    if oracle is None:
        return None
    kept = []
    for entry in results:
        config, produced = entry[0], entry[1]
        if produced == oracle:
            kept.append(entry)
            continue
        deviation, tolerance = _deviation(produced, oracle, dtype_name)
        if deviation <= tolerance:
            kept.append(entry)
    return kept


#: A callable `(tuner, key, buffers) -> list[bytes] | None` returning the
#: CORRECT contents of the screened buffers, or None where none can be had.
#: Default None: the screen keeps the consensus it has always kept, and this
#: whole path costs one `is None`.
#:
#: `configs_agreeing_with_oracle` is the primitive and it takes ONE buffer; a
#: screened result is a SNAPSHOT, a list of buffers with a dtype each. The
#: primitive was correct and tested from the day it was written and still could
#: not be called from here, because nothing carried a snapshot to it. That
#: adapter is `_oracle_keeps` below, and its absence is why the overrule sat
#: unwired: a helper whose every test passes can still have no seam.
_SCREEN_ORACLE = None


def set_screen_oracle(provider) -> None:
    """Install (or clear, with None) the oracle provider for the screen."""
    global _SCREEN_ORACLE
    _SCREEN_ORACLE = provider


def _oracle_keeps(results, oracle, buffers):
    """The candidates that agree with the oracle on EVERY screened buffer."""
    kept = []
    for entry in results:
        agrees = True
        for (_a, _n, dtype_name), produced, reference in zip(buffers, entry[1], oracle):
            if not configs_agreeing_with_oracle([(entry[0], produced)],
                                                reference, dtype_name):
                agrees = False
                break
        if agrees:
            kept.append(entry)
    return kept


def _make_agree(buffers):
    """The screen's agreement predicate for one set of screened buffers."""
    def agree(one, other):
        worst, tol, name = 0.0, 0.0, "?"
        for (_a, _n, dtype_name), x, y in zip(buffers, one, other):
            if x == y:
                continue
            deviation, tolerance = _deviation(x, y, dtype_name)
            if deviation > worst:
                worst, tol, name = deviation, tolerance, dtype_name
        return worst <= tol, worst, tol, name
    return agree


def _cluster(results, agree):
    """Candidates grouped by mutual agreement, first-fit against each group."""
    clusters: List[list] = []
    for entry in results:
        for cluster in clusters:
            ok, _w, _t, _d = agree(entry[1], cluster[0][1])
            if ok:
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    return clusters


def _largest_agreement(results, buffers):
    """What the VOTE would have seated — asked only to say whether it was
    about to be wrong. Never used to decide anything once an oracle exists."""
    clusters = _cluster(results, _make_agree(buffers))
    if not clusters:
        return []
    return max(clusters, key=len)


#: what `do_bench` allocates on top of the arguments, once per call —
#: an L2-sized scratch flushed before every timed launch.
_BENCH_FLUSH_BYTES = 256 * 1024 * 1024


def bench_would_swap(total_bytes: int):
    """(True, available_mb) when timing candidates would measure the swap.

    The comparison carries NO TUNING MARGIN, and that is a different claim from
    "no constant at all" -- it used to say the latter, and stopped being true
    when `_BENCH_FLUSH_BYTES` was added (2026-09-17, after hat-s-x4 reached
    SIGKILL with the door open). What is added is not a cushion: it is the
    scratch `do_bench` REALLY ALLOCATES on every call, at the same name it
    allocates it with, so the two cannot drift. Nothing here is free to tune.

    `available_mb` comes from `core.host_memory` -- our own authority for this
    quantity, never `Pages free` -- and is read live, never cached: pressure is
    a state of the moment, not of the process.

    An unreadable platform (available_mb None) gates nothing and says nothing
    here: `memory_state` already names why it could not read.
    """
    try:
        from neurobrix.core.host_memory import host_shares_memory_with_device, memory_state
        if not host_shares_memory_with_device():
            return False, None                         # discrete memory: the host is not the bench
        avail_mb = memory_state().available_mb
    except Exception:                                  # noqa: BLE001
        return False, None
    if avail_mb is None:
        return False, None
    # The sweep does not only hold the arguments: `do_bench` allocates a
    # 256 MiB L2 flush buffer per call, and Triton times one candidate after
    # another. Comparing the arguments alone therefore UNDERSTATES what the
    # sweep needs, which is how hat-s-x4 reached SIGKILL with the door open
    # (2026-09-17: a baddbmm key carrying 5.92 GB of arguments, the door
    # answering False, and the machine dying).
    need = total_bytes + _BENCH_FLUSH_BYTES
    import os as _os_bws
    if _os_bws.environ.get("NBX_BENCH_DOOR_DIAG"):
        _acct = ""
        try:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
            _live = _DA.memory_allocated() / 1e6
            _cached = sum(_DA._pool_cached_bytes.values()) / 1e6
            _n = len(_DA._cuda_ptr_size)
            _acct = (f" | allocator: live {_live:.0f} MB, pooled {_cached:.0f} MB, "
                     f"{_n} live pointers")
        except Exception as _e:
            _acct = f" | allocator accounting unavailable: {type(_e).__name__}"
        print(f"[BENCH_DOOR] args {total_bytes} + flush {_BENCH_FLUSH_BYTES} "
              f"= {need} bytes vs {avail_mb} MB available -> "
              f"{'REFUSE' if need > avail_mb * 2 ** 20 else 'allow'}{_acct}", flush=True)
    return need > avail_mb * 2 ** 20, avail_mb


def autotune_shape_key(tuner, kwargs):
    """The key Triton's Autotuner will store this call's choice under.

    Built exactly as `Autotuner.run` builds it — the `keys` arguments' values
    followed by every argument's dtype — from `tuner.nargs`, which holds the
    real arguments by the time `prune_configs` runs. Triton computes this key
    before calling `prune_configs` and does not pass it, so the screen rebuilt
    a DIFFERENT key (the constexpr kwargs) and recorded its unscreened seats
    under that one. The replay cache is written under Triton's key, so the
    `screened: false` mention never landed on any entry — seen on 2026-09-13
    by the production demonstration: 3 announcements, 0 stamped records. A
    key in one space, a record in another — the same family as the weight
    dict's two key spaces.
    """
    nargs = getattr(tuner, "nargs", None) or {}
    all_args = {**nargs, **(kwargs or {})}
    arg_names = getattr(tuner, "arg_names", None) or list(all_args)
    _args = {k: v for (k, v) in all_args.items() if k in arg_names}
    key = [_args[k] for k in (getattr(tuner, "keys", None) or []) if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


def _call_screen_oracle(provider, tuner, key, buffers, meta):
    """Call the provider with the launch kwargs when its signature takes them."""
    import inspect
    try:
        n = len(inspect.signature(provider).parameters)
    except (TypeError, ValueError):
        n = 3
    if n >= 4:
        return provider(tuner, key, buffers, meta)
    return provider(tuner, key, buffers)


def screen_configs(tuner, configs, key, meta=None, record_key=None):
    """Screen a key's candidates, then RELEASE what the sweep cached.

    A sweep allocates: the candidates' own buffers, and a 256 MiB L2 flush
    buffer per `do_bench` call. The allocator pool retains freed blocks by
    design, and its release path is flush-on-OOM — which never runs on unified
    memory, because the OS KILLS the process instead of returning an allocation
    failure. So one key's sweep leaves its blocks resident and the next key
    starts poorer.

    Measured on hat-s-x4 (M4 Pro, triton-ext, 2026-09-17), sampling available
    memory once a second while the run proceeded:

        18133 MB -> 15557 -> 15882 -> 9340 -> 3478 -> ... -> 210 MB, then SIGKILL

    and the per-key door was RIGHT at each decision, because it looks at one key
    at a time:

        args 5.92 GB + flush 0.27 GB vs 12695 MB available -> allow
        args 3.27 GB + flush 0.27 GB vs  8080 MB available -> allow

    The door is not wrong; it is blind to accumulation. Releasing the pool when
    a key's sweep finishes is what makes each door's view of "available" true.
    """
    try:
        return _screen_configs(tuner, configs, key, meta=meta,
                               record_key=record_key)
    finally:
        try:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            DeviceAllocator.empty_cache_pool()
        except Exception:                              # noqa: BLE001
            pass                                       # a sweep must not fail on cleanup


def _screen_configs(tuner, configs, key, meta=None, record_key=None):
    """Run every candidate once and keep the ones that agree with each other.

    `record_key` is the key the choice will be stored under (Triton's cache
    key, see `autotune_shape_key`); `key` is the screen's own de-duplication
    key. An unscreened seat is recorded under `record_key` so the replay cache
    can carry the mention.

    Agreement is decided by CONSENSUS, not against a nominated reference.
    Anchoring on one config inverts the moment that config is the broken one:
    measured 2026-09-07, `matmul_kernel`'s first config was one of the three
    that wrote half the output, so screening against it excluded the seven
    correct ones. There is no way to know in advance which config is right —
    that is the whole problem — so the screen asks which answer the configs
    agree on, and treats the rest as the outliers they are.

    A kernel writes into buffers the caller owns, so the screen snapshots
    them first and restores that snapshot before each run; otherwise an
    accumulating kernel would be compared against its own previous output.

    Returns the configs in the consensus. Raises when there is no consensus —
    when the candidates split evenly — because then nothing here can tell
    which half is right, and choosing on speed would be choosing at random.
    """
    if len(configs) < 2:
        return configs

    # De-duplicated by the SHAPE key, not the constexpr key: ten distinct
    # conv shapes share one constexpr tuple, and keyed by it the screen ran
    # on the first and silently skipped the other nine (2026-09-13, live).
    _rk = record_key if record_key is not None else key
    seen = _SCREEN_CACHE.setdefault(id(tuner), set())
    if _rk in seen:
        return configs
    seen.add(_rk)

    kernel_name = getattr(tuner.base_fn, "__name__", str(tuner))
    named = dict(tuner.nargs or {})
    if not named:
        return _seat_unscreened(kernel_name, _rk, configs, len(configs),
                                "the tuner carries no named arguments, so the "
                                "screen has nothing to compare")
    args = [named[name] for name in tuner.arg_names if name in named]
    buffers = _writable_buffers(args)
    if buffers is None:
        return _seat_unscreened(kernel_name, _rk, configs, len(configs),
                                "a strided view among the arguments, which the "
                                "screen cannot snapshot")
    if not buffers:
        return _seat_unscreened(kernel_name, _rk, configs, len(configs),
                                "no writable buffer to compare")

    from neurobrix.kernels.ops._configs import active_vendor_profile

    budget = active_vendor_profile().get("autotune_screen_max_bytes")
    if budget is None:
        raise RuntimeError(
            "the hardware profile declares no `autotune_screen_max_bytes`: "
            "the screen will not decide for itself how much memory traffic a "
            "tuning step may cost")
    total = sum(nbytes for _a, nbytes, _d in buffers)
    if total > int(budget):
        # Two doors meet here and both stay. Beyond the SCREEN budget the compare
        # is skipped (the Dell's ruling of 2026-09-12: the seat is announced as
        # UNSCREENED, never written to the certified directory). And Triton
        # would still time every candidate on these arguments; when they alone
        # exceed the machine's available memory the sweep measures the swap
        # (the Mac, 2026-09-13: a baddbmm key carrying 5.9 GB of arguments
        # against 4.5 GB available), so the sweep is cut to the single
        # first-declared config, the cut is SAID, and the choice is marked
        # unmeasured so capture() never persists it — recorded, it would
        # outlive the pressure that forced it.
        _swaps, _avail_mb = bench_would_swap(total)
        if _swaps:
            print(f"[AUTOTUNE_BENCH] "
                  f"{getattr(tuner.base_fn, '__name__', tuner)}: arguments "
                  f"total {total} bytes against {_avail_mb} MB available -- "
                  f"timing candidates would measure the swap, not the "
                  f"kernels. Taking the first declared config WITHOUT "
                  f"measurement; the choice will not be persisted.",
                  flush=True)
            from neurobrix.triton import autotune_cache as _atc
            _atc.mark_unmeasured(tuner, key)
            return _seat_unscreened(
                kernel_name, _rk, configs[:1], 1,
                f"arguments total {total} bytes, over the profile's screening "
                f"budget {int(budget)} and over the {_avail_mb} MB available: "
                f"the sweep was cut to the first declared config, unmeasured")
        # OVER THE SCREENING BUDGET — screen on NAMED ROW WINDOWS rather than not at all.
        #
        # A fixed budget made the LARGEST shapes the LEAST verified ones, which is backwards:
        # the failure class that motivates screening a large shape is index overflow, and it
        # lives exactly where the budget stopped looking (the 2^31 addmm, 2026-09-23). The
        # budget is a rule about COST and the question is CORRECTNESS, so the cost is bounded
        # instead — a few row windows of the output, compared to the rack's row-windowed fp64
        # oracle, at a price set by the window and not by the shape (owner, 2026-09-23).
        #
        # The last window is anchored at the final row on purpose: an index that wraps shows
        # at the largest linear index or nowhere.
        windowed = _screen_on_windows(tuner, kernel_name, _rk, configs, args, meta,
                                      buffers, int(budget), total)
        if windowed is not None:
            return windowed
        return _seat_unscreened(
            kernel_name, _rk, configs, len(configs),
            f"arguments total {total} bytes, over the profile's screening "
            f"budget {int(budget)}, and the output could not be row-windowed")

    before = _snapshot(buffers)
    meta = dict(meta or {})

    # -- run each candidate once, from the same starting state --------------
    results, unrun = [], []
    for config in configs:
        try:
            _restore(buffers, before)
            tuner.fn.run(*args, **{**meta, **config.all_kwargs()})
            results.append((config, _snapshot(buffers)))
        except Exception:
            # A config the backend refuses is not a screen failure — the
            # autotuner already handles one that will not compile. Counted,
            # never swallowed: a screen that silently keeps what it could not
            # run is a screen that checked nothing.
            unrun.append(config)
    _restore(buffers, before)

    if unrun:
        print(f"[AUTOTUNE_SCREEN] {kernel_name}: {len(unrun)} of "
              f"{len(configs)} configs could not be run for screening at key "
              f"{key}; they go to the timer unchecked", flush=True)
    if len(results) < 2:
        return _seat_unscreened(kernel_name, _rk, configs, len(results),
                                "fewer than two candidates ran, so there is "
                                "nothing to compare them against")

    # -- an oracle, where one exists, OVERRULES the vote --------------------
    #
    # This is the whole point of the 2026-09-10 finding. Consensus has two
    # failure modes it cannot see: a majority wrong in the same way, and a
    # unanimous space that is wrong. Both are silent, and both are widest
    # exactly where we have never looked — outside `nvidia/volta`, where the
    # configuration space itself differs by target.
    oracle = None
    if _SCREEN_ORACLE is None:
        # A mechanism that is complete and switched off is the most expensive
        # form of a vacuous guard: it costs the price of writing it and returns
        # nothing. The only thing worse is one that is silent about being off.
        from neurobrix.kernels.screen_oracle import announce_no_oracle
        announce_no_oracle(kernel_name, key,
                           why="no oracle provider is installed at all")
    if _SCREEN_ORACLE is not None:
        try:
            # The kernel's constexpr arguments (kernel_height, stride_*, padding_*,
            # groups, fp16 …) travel as launch KWARGS, not in `tuner.nargs`; a
            # provider that reads `nargs` alone never sees them. The GEMM oracles
            # need none and worked; the convolution oracle needs all of them and
            # returned None on every live key (2026-09-13, real-esrgan-x4 on
            # card 0) while its own suite was green. So the launch kwargs go to
            # the provider too, when it accepts them.
            oracle = _call_screen_oracle(_SCREEN_ORACLE, tuner, key, buffers, meta)
        except Exception as exc:      # an oracle that fails is not a launch failure
            print(f"[AUTOTUNE_ORACLE] {kernel_name}: the oracle provider raised "
                  f"({type(exc).__name__}: {exc}); falling back to the consensus "
                  f"at key {key}", flush=True)
    if oracle is not None:
        kept = _oracle_keeps(results, oracle, buffers)
        refused = [c for c, _ in results if c not in [k for k, _ in kept]]
        if not kept:
            raise RuntimeError(
                f"NeuroBrix autotune screen: {kernel_name} at key {key} — the "
                f"fp64 oracle contradicts EVERY candidate ({len(results)} of "
                f"{len(results)}). A consensus would have returned the whole "
                f"space and said nothing. Refusing to seat any of them.")
        if len(kept) < len(results):
            print(f"[AUTOTUNE_ORACLE] {kernel_name}: the oracle refused "
                  f"{len(refused)} of {len(results)} configs at key {key}; "
                  f"{len(kept)} kept. The vote is NOT consulted for this key.",
                  flush=True)
            # Was the vote about to seat one of them? That is the finding.
            majority = _largest_agreement(results, buffers)
            wrong_majority = [c for c, _ in majority if c in refused]
            if wrong_majority:
                print(f"[AUTOTUNE_ORACLE] FINDING — {kernel_name} at key {key}: "
                      f"{len(wrong_majority)} config(s) the CONSENSUS would have "
                      f"seated are contradicted by the oracle. The majority was "
                      f"wrong in the same way. This is the failure mode the "
                      f"screen cannot see on its own.", flush=True)
        # The oracle adjudicated this key: the record says so, with the name of
        # what adjudicated it — the converse of the unscreened mention.
        _ADJUDICATED[(kernel_name, repr(_rk))] = "fp64 oracle"
        return [c for c, _ in kept] + unrun

    # -- cluster by agreement ----------------------------------------------
    agree = _make_agree(buffers)
    clusters = _cluster(results, agree)

    if len(clusters) == 1:
        return _seat_unscreened(
            kernel_name, _rk, [c for c, _ in results] + unrun, len(results),
            _no_oracle_reason(oracle)
            + ", and the candidates were unanimous — which is one of the two "
              "failure modes consensus cannot see")

    clusters.sort(key=len, reverse=True)
    if len(clusters[0]) == len(clusters[1]):
        raise RuntimeError(
            f"NeuroBrix autotune screen: {kernel_name} at key {key} splits "
            f"into {len(clusters)} groups of configs that disagree with each "
            f"other, with no majority ("
            + ", ".join(str(len(c)) for c in clusters)
            + "). Refusing to choose one on speed.")

    winners = {id(c) for c, _ in clusters[0]}
    dropped = []
    for cluster in clusters[1:]:
        for config, result in cluster:
            _ok, worst, tol, name = agree(result, clusters[0][0][1])
            dropped.append(ScreenedOut(kernel_name, key, str(config), name,
                                       worst, tol))

    _SCREENED.extend(dropped)
    # A runtime exclusion that contradicts a CERTIFIED setting is a finding,
    # reported here and persisted — never a silence.
    try:
        from neurobrix.kernels.autotune_certified import report_contradictions
        report_contradictions(tuner, dropped)
    except Exception as exc:                        # the report must not turn into a launch failure
        print(f"[AUTOTUNE_SCREEN] could not check the certified directory: {exc}", flush=True)
    for entry in dropped:
        print(f"[AUTOTUNE_SCREEN] {entry.kernel}: config excluded before "
              f"timing — it disagrees with the {len(clusters[0])}-config "
              f"consensus by {entry.deviation:.3e} ({entry.dtype}, tolerance "
              f"{entry.tolerance:.1e}) at key {entry.key}: {entry.config}",
              flush=True)
    _record_screen_exclusions(dropped)

    return _seat_unscreened(
        kernel_name, _rk, [c for c, _ in clusters[0]] + unrun, len(results),
        _no_oracle_reason(oracle)
        + f", and the winner is a {len(clusters[0])}-config majority — which "
          f"is the other failure mode consensus cannot see, a majority wrong "
          f"in the same way")


def _record_screen_exclusions(dropped) -> None:
    """Hand the exclusions to the persisted sweep, so Forge sees them."""
    try:
        from neurobrix.triton import autotune_cache
    except Exception:                                   # pragma: no cover
        return
    record = getattr(autotune_cache, "record_screen_exclusions", None)
    if record is None:
        return
    try:
        record([e._asdict() for e in dropped])
    except Exception as exc:                            # pragma: no cover
        print(f"[AUTOTUNE_SCREEN] could not persist exclusions: {exc}",
              flush=True)

# ---------------------------------------------------------------------------
# The seam: route every `kernel[grid](...)` of the process through `launch`
# ---------------------------------------------------------------------------

_installed = False


#: The package whose kernels this launcher owns. A kernel defined anywhere else —
#: torch's in-tree Triton ops, a third-party library's — is launched by Triton's own
#: path: this engine's allocator never handed out its memory and has no claim on it.
_OWN_PACKAGE = "neurobrix"


def _is_ours(jit_fn) -> bool:
    """True when this JITFunction was defined inside the engine.

    Read from the function's own module, which is the authority: a kernel's home is
    where it was written, not a name we maintain a list of. An unreadable module is
    treated as foreign — the conservative direction, since refusing a foreign launch
    breaks a working library while passing one through only forgoes a check that was
    never ours to make.
    """
    fn = getattr(jit_fn, "fn", None)
    module = getattr(fn, "__module__", None) or getattr(jit_fn, "__module__", None)
    if not isinstance(module, str):
        return False
    return module == _OWN_PACKAGE or module.startswith(_OWN_PACKAGE + ".")


def install(force: Optional[bool] = None) -> bool:
    """Route `JITFunction.__getitem__` through the NeuroBrix launcher.
    `NBX_LAUNCHER=triton` keeps upstream's (the differential arm)."""
    global _installed
    if _installed:
        return True
    if force is None and os.environ.get("NBX_LAUNCHER", "nbx").lower() == "triton":
        return False
    try:
        from triton.runtime.jit import JITFunction
    except ModuleNotFoundError:
        # A compiled-only install (no Triton wheel: a Mac without it, a CPU
        # box) has no kernel launcher to route; the package import that
        # installs the seam must not be the import that ends the engine.
        # `tests/unit/cli/test_compiled_mode_needs_no_triton.py` was red on
        # this line from c8ed017 to 2026-09-16.
        return False

    # The seam is process-wide: `JITFunction` is Triton's, so patching it routes
    # EVERY Triton kernel in the process — including ones this engine does not own.
    # torch 2.14 ships in-tree Triton implementations under `torch/_native/ops/`
    # (bmm_outer_product, foreach_mm, norm, polar, scatter_add, sum, topk) and
    # dispatches eager aten calls to them when a shape condition matches. Those
    # kernels are torch's, launched on torch's memory, and this launcher refused
    # them at its ownership rule — `aten.bmm::0`, parameter 'A_ptr', on the warm
    # compiled path of GLM-4.1V, Janus-Pro-7B and Sana-1600M (2026-09-17).
    #
    # The ownership rule is right and stays: a NeuroBrix kernel may not read memory
    # this engine did not hand out. It says nothing about a foreign library's kernel
    # on that library's own memory, and it is not ours to police. So the seam asks
    # whose kernel it is and gives a foreign one back to Triton's own path,
    # unwrapped — the differential arm `NBX_LAUNCHER=triton` already proves that
    # path works.
    _upstream_getitem = JITFunction.__getitem__
    _upstream_run = JITFunction.run

    def __getitem__(self, grid):
        if not _is_ours(self):
            return _upstream_getitem(self, grid)
        return lambda *args, **kwargs: launch(self, grid, *args, **kwargs)

    def run(self, *args, grid=None, warmup=False, **kwargs):
        """`JITFunction.run` is the Autotuner's launch path (every autotuned
        kernel: mm, bmm, addmm, conv2d, ...) and `warmup`'s — both routed here."""
        if not _is_ours(self):
            return _upstream_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        if warmup:
            prepare(self, args, kwargs)
            return None
        return launch(self, grid, *args, **kwargs)

    JITFunction.__getitem__ = __getitem__
    JITFunction.run = run

    # The correctness screen, at the seam the autotuner itself uses to narrow
    # its candidates: `prune_configs` is called once per key, immediately
    # before anything is timed, and by then `self.nargs` holds the real
    # arguments. Wrapping it means the screen sees exactly the configs that
    # were about to be benchmarked, on exactly the tensors they would run on.
    from triton.runtime.autotuner import Autotuner

    if not getattr(Autotuner.prune_configs, "_nbx_screened", False):
        _upstream_prune = Autotuner.prune_configs

        def prune_configs(self, kwargs, *rest):
            configs = _upstream_prune(self, kwargs, *rest)
            if os.environ.get("NBX_AUTOTUNE_SCREEN", "on").lower() == "off":
                return configs
            key = tuple(sorted(
                (k, str(v)) for k, v in (kwargs or {}).items()
                if isinstance(v, (int, float, bool, str))))
            return screen_configs(self, list(configs), key, kwargs,
                                  record_key=autotune_shape_key(self, kwargs))

        prune_configs._nbx_screened = True
        prune_configs._nbx_upstream = _upstream_prune
        Autotuner.prune_configs = prune_configs

    # The screen consults an fp64 oracle by default. A CERTIFIED key never
    # reaches the screen, so this costs nothing where a certification exists;
    # it is paid only on an uncertified key, once for the key and not once per
    # candidate. `NBX_SCREEN_ORACLE=off` keeps the bare consensus for the
    # differential arm.
    if os.environ.get("NBX_SCREEN_ORACLE", "on").lower() != "off":
        try:
            from neurobrix.kernels.screen_oracle import install as _install_oracle
            _install_oracle()
        except Exception as exc:                       # never block a launch
            print(f"[AUTOTUNE_ORACLE] the oracle provider could not be "
                  f"installed ({type(exc).__name__}: {exc}); the screen runs "
                  f"on the consensus alone", flush=True)

    # A backend refusal for ONE candidate config costs that config, not the
    # run. Triton's `_bench` already scores `OutOfResources` and friends `inf`
    # and carries on; `MetalNonRecoverableError` descends from `RuntimeError`,
    # so nothing catches it and it ends the sweep. Measured 2026-09-12: five
    # of six blocked models died inside the sweep on one refused config while
    # `conv2d_forward_kernel` declares eighteen, eleven of them servable.
    try:
        from neurobrix.kernels.autotune_refusals import install as _install_refusals
        if not _install_refusals():
            print("[AUTOTUNE_REFUSED] the autotuner is not importable; a "
                  "refused config will end the sweep instead of being "
                  "excluded", flush=True)
    except Exception as exc:                           # never block a launch
        print(f"[AUTOTUNE_REFUSED] the refusal policy could not be installed "
              f"({type(exc).__name__}: {exc}); a refused config will end the "
              f"sweep", flush=True)

    _installed = True
    return True


# ---------------------------------------------------------------------------
# The benchmarker the autotuner uses: CUDA events through the allocator's
# runtime handle — Triton's own asks torch for its timing and its buffers.
# ---------------------------------------------------------------------------

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean", budget_ms=None, **_):
    """Same contract as `triton.testing.do_bench` (milliseconds; quantiles or
    a mean/min/max), with the L2 flush and the timing done through the
    engine's runtime (`DeviceAllocator`: events on the legacy stream)."""
    import numpy as np
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor, NBXDtype
    fn()
    DeviceAllocator.stream_synchronize(0)
    # an L2-sized scratch flushed before every timed call, as upstream does.
    # The SIZE is `_BENCH_FLUSH_BYTES`, not a literal repeated here:
    # `bench_would_swap` counts this allocation when it decides whether a sweep
    # would measure the swap, and a gate counting a different number from the
    # one actually allocated is a gate about nothing.
    flush = NBXTensor.empty((_BENCH_FLUSH_BYTES // 4,), dtype=NBXDtype.float32, device="cuda")
    # ONE probing launch before the four that finish the estimate. The
    # estimate used to be five blind launches -- so a pathologically slow
    # candidate (2.5 min a launch was measured on a bf16 matmul sweep) cost
    # five launches, >12 minutes, before any number existed. After a single
    # launch its time IS known, and a candidate whose one launch exceeds the
    # budget cannot win the sweep anyway; benching it further buys nothing.
    #
    # For a candidate under budget the arithmetic is unchanged: one launch
    # plus four launches, and t_est is the mean of the five -- the same
    # estimator, split across two event pairs instead of one.
    if budget_ms is None:
        # Triton's `_bench -> self.do_bench(kernel_call, quantiles=...)` chain
        # is not ours to re-sign, so the sweep wrapper PUBLISHES the budget and
        # this end consults it -- explicit coupling, documented at both ends.
        try:
            from neurobrix.kernels.autotune_refusals import current_budget_ms
            budget_ms = current_budget_ms()
        except Exception:                              # noqa: BLE001
            budget_ms = None
    t_probe = _time_ms(fn, 1)
    if budget_ms is not None and t_probe > budget_ms:
        raise CandidateOverTimeBudget(t_probe, budget_ms)
    try:
        from neurobrix.kernels.autotune_refusals import note_candidate_time
        note_candidate_time(t_probe)
    except Exception:                                  # noqa: BLE001
        pass
    t_est = (t_probe + 4.0 * _time_ms(fn, 4)) / 5.0
    n_warmup = max(1, int(warmup / max(t_est, 1e-3)))
    n_repeat = max(1, int(rep / max(t_est, 1e-3)))
    for _i in range(n_warmup):
        fn()
    times = []
    for _i in range(n_repeat):
        DeviceAllocator.memset_cuda(flush.data_ptr(), 0, flush._nbytes)
        times.append(_time_ms(fn, 1))
    times = np.asarray(times, dtype=np.float64)
    if quantiles is not None:
        return [float(q) for q in np.quantile(times, quantiles)]
    return float(getattr(np, return_mode)(times)) if return_mode in ("mean", "min", "max", "median") else float(times.mean())


class CandidateOverTimeBudget(Exception):
    """One autotune candidate exceeded the sweep's measured time budget.

    Carries the two numbers so the exclusion can be ANNOUNCED with them: a
    candidate scored out for time without its time is a silent narrowing.
    """

    def __init__(self, took_ms: float, budget_ms: float):
        self.took_ms = float(took_ms)
        self.budget_ms = float(budget_ms)
        super().__init__(
            f"one launch took {took_ms:.1f} ms against a budget of "
            f"{budget_ms:.1f} ms derived from this sweep's own measurements")


def _time_ms(fn, n: int) -> float:
    """Wall time of n calls on the device, by CUDA events on the legacy stream."""
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    start = DeviceAllocator.create_event(timing=True); end = DeviceAllocator.create_event(timing=True)
    DeviceAllocator.record_event(start, 0)
    for _i in range(n):
        fn()
    DeviceAllocator.record_event(end, 0)
    DeviceAllocator.event_synchronize(end)
    ms = DeviceAllocator.event_elapsed_ms(start, end)
    DeviceAllocator.destroy_event(start); DeviceAllocator.destroy_event(end)
    return ms / n


# ---------------------------------------------------------------------------
# The launcher contract's CUDA client (`neurobrix.triton.launcher_contract`,
# Metal workstream 2026-09-05): a driver COMPILES a jit function plus an
# explicit signature / constexprs / specialization markers into a
# CompiledKernel, and LAUNCHES it with a grid and a flat argument list —
# integer addresses for pointers (each verified with the allocator), typed
# scalars otherwise. Streams and events are the allocator's. `CudaDriver`
# carries this surface; the same checker that gates the Metal driver gates it.
# ---------------------------------------------------------------------------

class CudaCompiledKernel:
    """The result of compiling once, launched many times (contract object)."""

    binary_kind = "cubin"

    def __init__(self, name, binary, block_size, shared_memory, constexprs, specialization, binding, signature, function):
        self.name = name
        self.binary = binary
        self.block_size = block_size
        self.shared_memory = shared_memory
        self.constexprs = dict(constexprs)
        self.specialization = dict(specialization)
        self.binding = tuple(binding)
        self._signature = signature          # name -> triton type, non-constexpr, in binding order
        self._function = function

    def launch(self, grid, args, stream: int = 0) -> None:
        args = list(args)
        if len(args) != len(self.binding):
            raise TypeError(f"NeuroBrix launcher: {self.name} takes {len(self.binding)} arguments, {len(args)} given")
        params = []
        for slot, value in zip(self.binding, args):
            if slot.is_pointer:
                params.append(("ptr", int(value)))     # ownership is the driver's refusal, below
            else:
                params.append(_pack_param(slot.dtype, value))
        params.append(("ptr", 0))    # global scratch (Triton >= 3.6 ABI)
        params.append(("ptr", 0))    # profile scratch
        g = tuple(int(x) for x in grid) + (1,) * (3 - len(grid))
        CudaDriver.instance().launch(self._function, g, (self.block_size, 1, 1), self.shared_memory, int(stream), params)


def driver() -> Driver:
    """The driver this process launches through — the launcher's backend
    and the contract's client are one object (`CudaDriver` on CUDA; the
    Metal driver registers itself behind the same interface)."""
    return active_driver()
