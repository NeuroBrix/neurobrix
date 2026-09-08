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

    def launch(self, function, grid, block, shared: int, stream: int, params) -> None:  # pragma: no cover
        raise NotImplementedError

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


class CudaDriver(Driver):
    """libcuda through ctypes. The context is the primary context of the
    allocator's current device, already alive through the runtime API."""

    _inst: Optional["CudaDriver"] = None
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

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

    def launch(self, function, grid, block, shared: int, stream: int, params) -> None:
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
        for kind, value in params:
            if kind == "ptr" and value and not DeviceAllocator.holds(int(value)):
                raise ValueError(f"NeuroBrix launcher: device address {int(value):#x} was not handed out "
                                 f"by the allocator — refused, not launched")
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
        compiled = triton_compile(src, target=target(), options=options.__dict__)
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

#: backend name (as the allocator seam resolves it) -> module exposing
#: `driver()`. CUDA is absent because it is the built-in default, which is
#: what keeps the CUDA path byte-identical to before this registry existed.
_DRIVER_MODULES = {"metal": "neurobrix.triton.metal_driver"}


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
    path = _DRIVER_MODULES.get(name)
    if path is None:
        return CudaDriver.instance()
    from importlib import import_module
    return import_module(path).driver()


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
        return "bits16", struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16
    raise RuntimeError(f"NeuroBrix launcher: cannot pack a parameter of type {ty!r}")


# ---------------------------------------------------------------------------
# The launch: Triton's binder + compiler, our driver
# ---------------------------------------------------------------------------

class _Prepared:
    __slots__ = ("function", "signature", "shared", "num_warps", "block", "name")

    def __init__(self, function, signature, shared, num_warps, name, block=None):
        self.function = function
        self.signature = signature
        self.shared = shared
        self.num_warps = num_warps
        self.block = block if block is not None else (32 * num_warps, 1, 1)
        self.name = name


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
                         drv.block_for(md))
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
    params = [_pack_param(ty, bound_args[name]) for name, ty in prep.signature.items() if ty != "constexpr"]
    drv = active_driver()
    if drv.wants_scratch_params:
        params.append(("ptr", 0))    # global scratch (Triton ≥ 3.6 ABI)
        params.append(("ptr", 0))    # profile scratch
    if _RECORDER is not None:
        _RECORDER(prep, grid, params)
    drv.launch(prep.function, grid, prep.block, prep.shared, _stream(), params)


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


_SCREENED: List[ScreenedOut] = []
_SCREEN_CACHE: Dict[int, set] = {}      # id(tuner) -> keys already screened


def screened_out() -> List[ScreenedOut]:
    """Every config the correctness screen excluded, in order."""
    return list(_SCREENED)


def clear_screened() -> None:
    _SCREENED.clear()
    _SCREEN_CACHE.clear()


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


def screen_configs(tuner, configs, key, meta=None):
    """Run every candidate once and keep the ones that agree with each other.

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

    seen = _SCREEN_CACHE.setdefault(id(tuner), set())
    if key in seen:
        return configs
    seen.add(key)

    named = dict(tuner.nargs or {})
    if not named:
        return configs
    args = [named[name] for name in tuner.arg_names if name in named]
    buffers = _writable_buffers(args)
    if buffers is None:
        print(f"[AUTOTUNE_SCREEN] "
              f"{getattr(tuner.base_fn, '__name__', tuner)}: a strided view "
              f"among the arguments; not screened at key {key}", flush=True)
        return configs
    if not buffers:
        return configs

    from neurobrix.kernels.ops._configs import active_vendor_profile

    budget = active_vendor_profile().get("autotune_screen_max_bytes")
    if budget is None:
        raise RuntimeError(
            "the hardware profile declares no `autotune_screen_max_bytes`: "
            "the screen will not decide for itself how much memory traffic a "
            "tuning step may cost")
    total = sum(nbytes for _a, nbytes, _d in buffers)
    if total > int(budget):
        print(f"[AUTOTUNE_SCREEN] "
              f"{getattr(tuner.base_fn, '__name__', tuner)}: arguments total "
              f"{total} bytes, over the profile's screening budget "
              f"{int(budget)}; not screened at key {key}", flush=True)
        return configs

    before = _snapshot(buffers)
    meta = dict(meta or {})
    kernel_name = getattr(tuner.base_fn, "__name__", str(tuner))

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
        return configs

    # -- cluster by agreement ----------------------------------------------
    def agree(one, other):
        worst, tol, name = 0.0, 0.0, "?"
        for (_a, _n, dtype_name), x, y in zip(buffers, one, other):
            if x == y:
                continue
            deviation, tolerance = _deviation(x, y, dtype_name)
            if deviation > worst:
                worst, tol, name = deviation, tolerance, dtype_name
        return worst <= tol, worst, tol, name

    clusters: List[list] = []
    for entry in results:
        for cluster in clusters:
            ok, _w, _t, _d = agree(entry[1], cluster[0][1])
            if ok:
                cluster.append(entry)
                break
        else:
            clusters.append([entry])

    if len(clusters) == 1:
        return [c for c, _ in results] + unrun

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

    return [c for c, _ in clusters[0]] + unrun


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


def install(force: Optional[bool] = None) -> bool:
    """Route `JITFunction.__getitem__` through the NeuroBrix launcher.
    `NBX_LAUNCHER=triton` keeps upstream's (the differential arm)."""
    global _installed
    if _installed:
        return True
    if force is None and os.environ.get("NBX_LAUNCHER", "nbx").lower() == "triton":
        return False
    from triton.runtime.jit import JITFunction

    def __getitem__(self, grid):
        return lambda *args, **kwargs: launch(self, grid, *args, **kwargs)

    def run(self, *args, grid, warmup=False, **kwargs):
        """`JITFunction.run` is the Autotuner's launch path (every autotuned
        kernel: mm, bmm, addmm, conv2d, ...) and `warmup`'s — both routed here."""
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
            return screen_configs(self, list(configs), key, kwargs)

        prune_configs._nbx_screened = True
        prune_configs._nbx_upstream = _upstream_prune
        Autotuner.prune_configs = prune_configs

    _installed = True
    return True


# ---------------------------------------------------------------------------
# The benchmarker the autotuner uses: CUDA events through the allocator's
# runtime handle — Triton's own asks torch for its timing and its buffers.
# ---------------------------------------------------------------------------

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean", **_):
    """Same contract as `triton.testing.do_bench` (milliseconds; quantiles or
    a mean/min/max), with the L2 flush and the timing done through the
    engine's runtime (`DeviceAllocator`: events on the legacy stream)."""
    import numpy as np
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor, NBXDtype
    fn()
    DeviceAllocator.stream_synchronize(0)
    # an L2-sized scratch flushed before every timed call, as upstream does
    flush = NBXTensor.empty((256 * 1024 * 1024 // 4,), dtype=NBXDtype.float32, device="cuda")
    t_est = _time_ms(fn, 5)
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
# Metal chantier 2026-09-05): a driver COMPILES a jit function plus an
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
