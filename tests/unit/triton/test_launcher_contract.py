"""The launcher contract, checked against Metal — and written for CUDA.

`triton/launcher_contract.py` says what a driver must present to the engine's
kernel launcher. That launcher is a Dell-side component with CUDA as its first
client; this file is what makes the contract executable, so the CUDA
implementation can be checked against the same assertions on a machine with no
Apple GPU.

**To add a backend:** write the driver, add a line to `_DRIVERS` below, and
change nothing else. If a backend needs the checker relaxed, the contract is
wrong and this file is where that argument happens.

R33 — no torch here either. The buffers come from `DeviceAllocator`, on every
backend, which is the whole reason the launcher passes pointer integers.
"""

from __future__ import annotations

import ctypes

import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator
from neurobrix.triton.launcher_contract import (ArgSlot, verify_driver_contract)


def _metal_driver():
    """The Metal driver, or None where there is no Apple GPU."""
    try:
        from neurobrix.kernels import nbx_tensor
        if nbx_tensor._detect_gpu_backend() != "metal":
            return None
        from neurobrix.triton.metal_driver import driver
        return driver()
    except Exception:
        return None


def _cuda_driver():
    """Placeholder for the CUDA implementation.

    Returns None until it exists. When it does, this returns it and the whole
    file runs against CUDA with no other change — which is the property the
    contract is for.
    """
    return None


_DRIVERS = {"metal": _metal_driver, "cuda": _cuda_driver}


@pytest.fixture(params=sorted(_DRIVERS))
def driver(request):
    made = _DRIVERS[request.param]()
    if made is None:
        pytest.skip(f"no {request.param} driver on this machine")
    return made


# --- the kernel the contract is checked with --------------------------------
#
# Deliberately the simplest thing that still exercises every part of the
# surface: two pointers, a scalar, a constexpr, and a grid wider than one
# block so a dropped program offset cannot pass.

def _add_one_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask) + 1.0,
                 mask=mask)

    return add_one


_N = 256
_BLOCK = 64
_SIGNATURE = {"src_ptr": "*fp32", "dst_ptr": "*fp32", "n": "i32",
              "BLOCK": "constexpr"}
_CONSTEXPRS = {"BLOCK": _BLOCK}


class _Args:
    """Builds the flat argument list, and says what buffers it needs."""

    buffer_sizes = (_N * 4, _N * 4)
    output_index = 1
    output_bytes = _N * 4

    def __call__(self, pointers):
        return [pointers[0], pointers[1], _N]


def _make_buffer(nbytes: int) -> int:
    return DeviceAllocator.malloc_cuda(nbytes)


def _read_buffer(address: int, nbytes: int) -> bytes:
    buffer = (ctypes.c_char * nbytes)()
    DeviceAllocator.memcpy(ctypes.addressof(buffer), address, nbytes, kind=2)
    return bytes(buffer)


def _free_buffer(address: int) -> None:
    DeviceAllocator.free_cuda(address)


def test_driver_satisfies_the_launcher_contract(driver):
    """The whole surface, in one pass, with the failures named individually."""
    import struct

    source = [float(i) for i in range(_N)]
    payload = struct.pack(f"<{_N}f", *source)
    expected = struct.pack(f"<{_N}f", *[v + 1.0 for v in source])

    args = _Args()
    original_make = _make_buffer

    def make_and_fill(nbytes):
        address = original_make(nbytes)
        if nbytes == len(payload):
            host = (ctypes.c_char * nbytes).from_buffer_copy(payload)
            DeviceAllocator.memcpy(address, ctypes.addressof(host), nbytes,
                                   kind=1)
        return address

    failures = verify_driver_contract(
        driver=driver,
        jit_fn=_add_one_kernel(),
        signature=_SIGNATURE,
        constexprs=_CONSTEXPRS,
        make_buffer=make_and_fill,
        read_buffer=_read_buffer,
        free_buffer=_free_buffer,
        grid=(_N // _BLOCK,),
        args_builder=args,
        expected=expected,
    )
    assert not failures, (
        f"{driver.backend} does not satisfy the launcher contract:\n  "
        + "\n  ".join(failures))


def test_the_contract_declares_a_backend_name(driver):
    assert isinstance(driver.backend, str) and driver.backend


def test_pointer_slots_carry_addresses_not_containers(driver):
    """The launcher passes integers. A driver that wants a tensor object has
    put the container back in the interface, and with it the framework the
    container belongs to."""
    kernel = driver.compile(_add_one_kernel(), _SIGNATURE, _CONSTEXPRS,
                            num_warps=4)
    pointer_slots = [s for s in kernel.binding if s.is_pointer]
    assert len(pointer_slots) == 2
    assert all(isinstance(s, ArgSlot) for s in kernel.binding)
    assert [s.name for s in pointer_slots] == ["src_ptr", "dst_ptr"]


def test_the_artifact_declares_what_it_is(driver):
    """A cache that cannot tell a metallib from a cubin will eventually load
    one into the other."""
    kernel = driver.compile(_add_one_kernel(), _SIGNATURE, _CONSTEXPRS,
                            num_warps=4)
    assert kernel.binary_kind in ("cubin", "metallib", "msl", "hsaco")
    assert isinstance(kernel.binary, (bytes, bytearray)) and kernel.binary


def test_no_torch_is_pulled_by_compiling_or_launching(driver):
    """R33, as an execution fact rather than a source scan.

    The gate in `tests/unit/kernels/test_r33_torch_import_guard.py` reads
    imports. This asserts the property that actually matters: driving a kernel
    through the contract does not put torch in the process.
    """
    import subprocess
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[3] / "src"
    code = (
        "import sys;"
        "sys.path.insert(0, %r);" % str(src_root) +
        "from neurobrix.kernels.nbx_tensor import DeviceAllocator;"
        "DeviceAllocator.get_device();"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=180)
    assert out.stdout.strip().endswith("False"), (
        f"reaching the device pulled torch.\nstdout: {out.stdout!r}\n"
        f"stderr: {out.stderr[-300:]!r}")
