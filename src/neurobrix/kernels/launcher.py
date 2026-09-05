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
from typing import Any, Dict, Optional, Tuple

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
    used to ask Triton's driver (matmul / config spaces, the autotune cache)."""
    global _TARGET
    if _TARGET is None:
        from triton.backends.compiler import GPUTarget
        _TARGET = GPUTarget("cuda", _compute_capability(), 32)
    return _TARGET


def arch() -> int:
    """Compute capability as an int (70 for sm_70) — `target().arch`."""
    return int(target().arch)


# ---------------------------------------------------------------------------
# The vendor driver interface
# ---------------------------------------------------------------------------

class Driver:
    """Four calls a backend implements: load a binary, set its shared memory,
    launch, and (for the target) read the device's capability."""

    def load(self, binary: bytes, name: str, shared: int):  # pragma: no cover - interface
        raise NotImplementedError

    def launch(self, function, grid, block, shared: int, stream: int, params) -> None:  # pragma: no cover
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
        self._modules: Dict[Tuple[int, bytes], ctypes.c_void_p] = {}

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
        if shared > 48 * 1024:
            self._check(self.lib.cuFuncSetAttribute(function, self.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ctypes.c_int(shared)), "cuFuncSetAttribute")
        return function

    def launch(self, function, grid, block, shared: int, stream: int, params) -> None:
        gx, gy, gz = grid
        bx, by, bz = block
        if gx * gy * gz <= 0:
            return
        storage = [ctypes.c_uint64(v) if kind in ("ptr", "u64", "bits64") else
                   ctypes.c_int64(v) if kind == "i64" else
                   ctypes.c_uint32(v) if kind in ("u32", "bits32") else
                   ctypes.c_int32(v) if kind == "i32" else
                   ctypes.c_uint16(v) if kind in ("u16", "bits16") else
                   ctypes.c_int16(v) if kind == "i16" else
                   ctypes.c_uint8(v) if kind == "u8" else
                   ctypes.c_int8(v) if kind == "i8" else
                   ctypes.c_double(v) if kind == "f64" else
                   _unsupported(kind)
                   for kind, v in params]
        arr = (ctypes.c_void_p * len(storage))(*[ctypes.addressof(s) for s in storage])
        self._check(self.lib.cuLaunchKernel(function, gx, gy, gz, bx, by, bz, ctypes.c_uint(shared),
                                            ctypes.c_void_p(stream), arr, None), "cuLaunchKernel")


def _unsupported(kind):
    raise RuntimeError(f"NeuroBrix launcher: unsupported scalar kind {kind!r}")


# ---------------------------------------------------------------------------
# Argument packing: the signature dictates the kind of every parameter
# ---------------------------------------------------------------------------

_INT_KINDS = {"i1": "i8", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
              "u1": "u8", "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64"}


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

    def __init__(self, function, signature, shared, num_warps, name):
        self.function = function
        self.signature = signature
        self.shared = shared
        self.num_warps = num_warps
        self.block = (32 * num_warps, 1, 1)
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
        name = getattr(dt, "name", None) or str(dt).split(".")[-1]
        ty = "*" + _POINTEE.get(str(name), str(name))
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


def nbx_binder(kernel, args, kwargs):
    """(bound_args, specialization, options) — the same triple Triton's
    generated binder returns, from the kernel's parameter list."""
    bound = {}
    spec = []
    positional = list(args)
    options = dict(kwargs)
    for i, kp in enumerate(kernel.params):
        name = kp.name
        if i < len(positional):
            value = positional[i]
        elif name in options:
            value = options.pop(name)
        elif kp.has_default:
            value = kp.default
        else:
            raise TypeError(f"NeuroBrix launcher: {kernel.__name__}() missing argument {name!r}")
        bound[name] = value
        if kp.is_constexpr:
            spec.append(("constexpr", value))
            continue
        specialize = not kp.do_not_specialize
        align = not kp.do_not_specialize_on_alignment
        ann = getattr(kp, "annotation_type", None)
        if ann:
            if isinstance(ann, str) and (ann == "u1" or ann[:2] in ("fp", "bf")):
                specialize = False
            if specialize:
                spec.append((ann,) + tuple(specialize_arg(value, True, align)[1:]))
            else:
                spec.append((ann, None))
            continue
        spec.append(specialize_arg(value, specialize, align))
    if len(positional) > len(kernel.params):
        raise TypeError(f"NeuroBrix launcher: {kernel.__name__}() takes {len(kernel.params)} arguments, {len(positional)} given")
    return bound, spec, options


_binders: Dict[int, tuple] = {}     # id(kernel) → (kernel_cache, key_cache, backend)


def _binder(kernel):
    b = _binders.get(id(kernel))
    if b is None:
        from triton.compiler import make_backend
        b = _binders[id(kernel)] = ({}, {}, make_backend(target()))
    return b


def prepare(kernel, args, kwargs) -> Tuple[_Prepared, Dict[str, Any]]:
    """Specialise (our binder), compile (Triton's compiler with the engine's
    target), load (our driver) — once per specialisation and device."""
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.runtime.jit import compute_cache_key
    kernel_cache, key_cache, backend = _binder(kernel)
    bound_args, specialization, options = nbx_binder(kernel, args, kwargs)
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    key = (compute_cache_key(key_cache, specialization, options), int(DeviceAllocator.get_device()))
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
        function = CudaDriver.instance().load(compiled.asm["cubin"], md.name, md.shared)
        prep = kernel_cache[key] = _Prepared(function, signature, md.shared, md.num_warps, md.name)
    return prep, bound_args


def launch(kernel, grid, *args, **kwargs):
    """The one entry point: `launch(kernel, grid, *args, **constexprs_and_options)`."""
    prep, bound_args = prepare(kernel, args, kwargs)
    if callable(grid):
        grid = grid(bound_args)
    grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
    params = [_pack_param(ty, bound_args[name]) for name, ty in prep.signature.items() if ty != "constexpr"]
    params.append(("ptr", 0))    # global scratch (Triton ≥ 3.6 ABI)
    params.append(("ptr", 0))    # profile scratch
    CudaDriver.instance().launch(prep.function, grid, prep.block, prep.shared, _stream(), params)


def _stream() -> int:
    """The legacy default stream (0): every house kernel launches on it today,
    and the allocator's copies order against it."""
    return 0


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

    JITFunction.__getitem__ = __getitem__
    _installed = True
    return True
