#!/usr/bin/env python3
"""Does the user need Xcode? Three routes from one MSL source, compared bit
for bit.

The Metal Triton backend's compile path ends in Apple's OFFLINE shader
compiler: MSL -> `xcrun metal` -> AIR -> `xcrun metallib` -> a library loaded
from disk. That compiler is not part of the Command Line Tools; it is a
~700 MB on-demand Xcode component, and `xcodebuild`, which installs it,
requires a full Xcode. Asking every user of NeuroBrix to install Xcode to run
a kernel is a large ask, so the question is whether it is a real one.

It is not obviously real, because Metal itself compiles shader SOURCE at
runtime through the framework — `newLibraryWithSource:options:error:` — with
no toolchain installed at all. The question this tool answers is whether that
route produces the SAME NUMBERS as the toolchain route, on our kernel, at the
shapes we care about. "Probably equivalent" is not an answer anyone can ship.

Three routes, one MSL source, identical inputs and identical tiles:

    A  framework   newLibraryWithSource:  — compiles MSL in-process. NO Xcode.
    B  xcrun       xcrun metal + metallib, then loaded from the result.
    C  prebuilt    the SAME metallib bytes as B, loaded with
                   newLibraryWithData: and never recompiled — the
                   "compiled by us, shipped to them" route.

If A == B the user does not need Xcode. If C == B a metallib built once here
can be shipped, which is a different and also useful answer. Both are decided
by comparing the bytes the GPU produced, not by reading documentation.

Every buffer comes from the Metal `DeviceAllocator` this chantier delivered,
bound through `buffer_for_pointer` — the accessor a launch path should use,
rather than wrapping a raw address with `newBufferWithBytesNoCopy`.

R33 preserved — no torch anywhere in this tool.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SHAPES = [(4, 128), (2, 4096), (1024, 64), (8, 1536)]
EPS = 1e-6

_SIGNATURE = {
    "input_ptr": "*fp32", "weight_ptr": "*fp32", "output_ptr": "*fp32",
    "batch_dim": "i32", "feat_dim": "i32",
    "input_batch_stride": "i32", "input_feat_stride": "i32",
    "output_batch_stride": "i32", "output_feat_stride": "i32",
    "eps": "fp32", "scale_by_weight": "constexpr",
    "BLOCK_SIZE_BATCH": "constexpr", "BLOCK_SIZE_FEAT": "constexpr",
}


# --- route A: the framework, no toolchain ----------------------------------

def library_from_source(device, msl: str):
    """`newLibraryWithSource:` — the framework compiles it, in-process."""
    import Metal
    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(
        msl, options, None)
    if library is None:
        raise RuntimeError(f"framework compile failed: {error}")
    return library


# --- route C: a prebuilt metallib, loaded and never recompiled -------------

def library_from_data(device, blob: bytes):
    """`newLibraryWithData:` over a real `dispatch_data_t`.

    The dispatch object is built by hand through libSystem because PyObjC
    cannot bridge it: handing this selector `bytes` or an `NSData` segfaults
    the process. triton-msl hit the same wall and routed around it via
    `newLibraryWithURL:` (`backend/driver.py:131` — "to avoid a PyObjC
    segfault in NSData's interaction with Metal's internal SHA256 hashing").
    Built properly, the selector works, and it is the one that proves a
    metallib can be LOADED without a compiler present.
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
        raise RuntimeError("dispatch_data_create returned NULL")
    data = objc.objc_object(c_void_p=ctypes.c_void_p(handle))
    library, error = device.newLibraryWithData_error_(data, None)
    if library is None:
        raise RuntimeError(f"newLibraryWithData failed: {error}")
    return library


# --- route B: the offline compiler, timed and sized ------------------------

def metallib_via_xcrun(msl: str, std_flag: str):
    """MSL -> .air -> .metallib with Apple's offline compiler. Returns
    (bytes, seconds, per-step seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "k.metal"
        air = Path(tmp) / "k.air"
        lib = Path(tmp) / "k.metallib"
        source.write_text(msl)
        t0 = time.perf_counter()
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", std_flag,
                        "-c", str(source), "-o", str(air)],
                       check=True, capture_output=True)
        t1 = time.perf_counter()
        subprocess.run(["xcrun", "-sdk", "macosx", "metallib",
                        str(air), "-o", str(lib)],
                       check=True, capture_output=True)
        t2 = time.perf_counter()
        return lib.read_bytes(), t2 - t0, {"metal_c": t1 - t0,
                                           "metallib": t2 - t1}


# --- running one library ----------------------------------------------------

def run_kernel(runtime, library, name, block_size, x, w):
    """Dispatch the kernel on buffers from the Metal DeviceAllocator.

    The binding order is the emitted MSL's own: three device pointers, then
    six int scalars, then the float eps.
    """
    import Metal
    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    batch_dim, feat_dim = x.shape
    function = library.newFunctionWithName_(name)
    if function is None:
        raise RuntimeError(f"library has no function {name!r}")
    pipeline, error = runtime._device.newComputePipelineStateWithFunction_error_(
        function, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline state failed: {error}")

    x_ptr = DeviceAllocator.malloc_cuda(x.nbytes)
    w_ptr = DeviceAllocator.malloc_cuda(w.nbytes)
    o_ptr = DeviceAllocator.malloc_cuda(x.nbytes)
    try:
        DeviceAllocator.memcpy(x_ptr, x.ctypes.data, x.nbytes, kind=1)
        DeviceAllocator.memcpy(w_ptr, w.ctypes.data, w.nbytes, kind=1)
        DeviceAllocator.memset_cuda(o_ptr, 0, x.nbytes)

        queue = runtime._queue
        command_buffer = queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, pointer in enumerate((x_ptr, w_ptr, o_ptr)):
            buffer, offset = runtime.buffer_for_pointer(pointer)
            if buffer is None:
                raise RuntimeError(
                    f"pointer {pointer:#x} did not come from this allocator")
            encoder.setBuffer_offset_atIndex_(buffer, offset, index)
        # setBytes wants a bytes-like object; PyObjC will not take a ctypes
        # byref here. Little-endian, which is what the device is.
        scalars = [batch_dim, feat_dim, feat_dim, 1, feat_dim, 1]
        for offset, value in enumerate(scalars):
            encoder.setBytes_length_atIndex_(
                struct.pack("<i", value), 4, 3 + offset)
        encoder.setBytes_length_atIndex_(struct.pack("<f", EPS), 4, 9)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(batch_dim, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1))
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        if command_buffer.error() is not None:
            raise RuntimeError(f"dispatch failed: {command_buffer.error()}")

        out = np.empty_like(x)
        DeviceAllocator.memcpy(out.ctypes.data, o_ptr, x.nbytes, kind=2)
        return out
    finally:
        for pointer in (x_ptr, w_ptr, o_ptr):
            DeviceAllocator.free_cuda(pointer)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton_msl.backend.device_detect import get_device_info
    from triton_msl.backend.driver import _detect_metal_arch

    from neurobrix.kernels import metal_device
    from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel
    from neurobrix.kernels.wrappers import _batch_block

    runtime = metal_device.runtime()
    info = get_device_info()
    target = GPUTarget("metal", _detect_metal_arch(), 32)
    jit_fn = rms_norm_forward_kernel.fn
    std_flag = info.metal_std_flag

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "validation_outputs" / "metal_first_light_2026_09_05")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device          : {runtime.device_name}")
    print(f"metal std flag  : {std_flag}")
    print(f"xcrun metal     : "
          f"{subprocess.run(['xcrun', 'metal', '--version'], capture_output=True, text=True).stdout.splitlines()[0]}")
    print()

    rows = []
    failures = []
    if True:
        for batch_dim, feat_dim in SHAPES:
            tag = f"{batch_dim}x{feat_dim}"
            bsb = _batch_block(batch_dim, feat_dim)
            bsf = triton.next_power_of_2(feat_dim)
            constexprs = {"scale_by_weight": True,
                          "BLOCK_SIZE_BATCH": bsb, "BLOCK_SIZE_FEAT": bsf}
            compiled = triton.compile(
                ASTSource(fn=jit_fn, signature=_SIGNATURE,
                          constexprs=constexprs),
                target=target, options={"num_warps": 4})
            # Read from the compiled artifact rather than tracing the stage:
            # Triton caches, so on a warm cache the stage never runs and a
            # tracer captures nothing. The artifact always carries both.
            msl = compiled.asm["msl"]
            block_size = compiled.metadata.block_size
            name = compiled.metadata.name

            rng = np.random.RandomState(20260903 + batch_dim * 31 + feat_dim)
            x = rng.randn(batch_dim, feat_dim).astype(np.float32)
            w = rng.randn(feat_dim).astype(np.float32)

            # A — the framework, no toolchain
            t0 = time.perf_counter()
            lib_a = library_from_source(runtime._device, msl)
            t_a = time.perf_counter() - t0
            out_a = run_kernel(runtime, lib_a, name, block_size, x, w)

            # B — Apple's offline compiler
            blob, t_b, steps = metallib_via_xcrun(msl, std_flag)
            lib_b = library_from_data(runtime._device, blob)
            out_b = run_kernel(runtime, lib_b, name, block_size, x, w)

            # C — the same bytes, loaded again without recompiling
            t0 = time.perf_counter()
            lib_c = library_from_data(runtime._device, blob)
            t_c = time.perf_counter() - t0
            out_c = run_kernel(runtime, lib_c, name, block_size, x, w)

            ab = np.array_equal(out_a.view(np.uint32), out_b.view(np.uint32))
            cb = np.array_equal(out_c.view(np.uint32), out_b.view(np.uint32))
            triton_blob = compiled.asm["metallib"]

            rows.append({
                "shape": tag,
                "BLOCK_SIZE_BATCH": bsb, "BLOCK_SIZE_FEAT": bsf,
                "threads_per_threadgroup": block_size,
                "msl_bytes": len(msl),
                "metallib_bytes_xcrun": len(blob),
                "metallib_bytes_triton": len(triton_blob),
                "metallib_bytes_identical":
                    blob == triton_blob,
                "seconds_framework_compile": round(t_a, 4),
                "seconds_xcrun_total": round(t_b, 4),
                "seconds_xcrun_metal_c": round(steps["metal_c"], 4),
                "seconds_xcrun_metallib": round(steps["metallib"], 4),
                "seconds_load_prebuilt": round(t_c, 4),
                "framework_equals_xcrun": bool(ab),
                "prebuilt_equals_xcrun": bool(cb),
            })
            if not ab:
                failures.append(f"{tag}: framework != xcrun")
            if not cb:
                failures.append(f"{tag}: prebuilt != xcrun")
            print(f"  {tag:>10}  A==B {str(ab):>5}   C==B {str(cb):>5}   "
                  f"MSL {len(msl):>5} B   metallib {len(blob):>6} B   "
                  f"framework {t_a*1000:7.1f} ms   xcrun {t_b*1000:7.1f} ms   "
                  f"load {t_c*1000:6.2f} ms")
    report = out_dir / "msl_path_equivalence.json"
    report.write_text(json.dumps({
        "device": runtime.device_name,
        "metal_std_flag": std_flag,
        "rows": rows,
    }, indent=1))
    print(f"\nwritten: {report}")

    if failures:
        print("\nNOT EQUIVALENT:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nAll three routes produce byte-identical output at every shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
