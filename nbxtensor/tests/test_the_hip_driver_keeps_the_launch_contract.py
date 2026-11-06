"""The HIP launch driver (`backend.launcher.HipDriver`), proven card-less against a fake libamdhip64.

Prepared for AMD Instinct cards; no AMD card has run it (2026-11-06). Its launch follows Triton 3.8's own AMD launcher
(`triton/backends/amd/driver.c`, `_launch`: `hipModuleLaunchKernel(function, gridX, gridY, gridZ, warp_size * num_warps, 1,
1, shared_memory, stream, params, 0)`, the kernel's arguments then the global- and profile-scratch pointers), its target
`GPUTarget("hip", <gfx arch>, <warp size>)` (`driver.py`, `get_current_target`), its binary `hsaco` (`compiler.py`).
What this test does if the driver were wrong: a library opened by another name, a missing library or symbol defaulted
instead of refused, a launch whose grid, block, shared memory, stream or argument array differ from upstream's order,
a large shared-memory ask launched without opting in, an error without HIP's own string — red."""
import ctypes

import pytest

from nbxtensor.backend import launcher as L
from nbxtensor.backend import runtime as R


class FakeHip:
    """libamdhip64's surface the driver uses, recording every call."""

    def __init__(self, attrs=None, fail=None):
        self.calls = []
        self.attrs = attrs or {74: 65536, 75: 65536, 87: 64}
        self.fail = fail or {}

    def _rc(self, name):
        return self.fail.get(name, 0)

    def hipModuleLoadData(self, module_ref, buf):
        ctypes.cast(module_ref, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x1000
        self.calls.append(("hipModuleLoadData",))
        return self._rc("hipModuleLoadData")

    def hipModuleGetFunction(self, function_ref, module, name):
        ctypes.cast(function_ref, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x2000
        self.calls.append(("hipModuleGetFunction", name))
        return self._rc("hipModuleGetFunction")

    def hipFuncSetAttribute(self, function, attr, value):
        self.calls.append(("hipFuncSetAttribute", attr, int(getattr(value, "value", value))))
        return self._rc("hipFuncSetAttribute")

    def hipDeviceGetAttribute(self, out_ref, attr, dev):
        ctypes.cast(out_ref, ctypes.POINTER(ctypes.c_int)).contents.value = self.attrs[attr]
        return 0

    def hipModuleLaunchKernel(self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        self.calls.append(("hipModuleLaunchKernel", (gx, gy, gz), (bx, by, bz), shared, stream,
                           [ctypes.cast(params[i], ctypes.POINTER(ctypes.c_uint64)).contents.value for i in range(3)]))
        return self._rc("hipModuleLaunchKernel")

    def hipGetErrorString(self, rc):
        return b"hipErrorOutOfMemory"


@pytest.fixture
def held(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: True))
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: 0))


def test_the_library_is_the_runtime_seams_own(monkeypatch):
    # the module API is in the HIP runtime library the seam loaded for the allocator: no second load by name
    lib = FakeHip()
    monkeypatch.setattr(R, "gpu_runtime", lambda: lib)
    assert L.HipDriver().lib is lib


def test_a_missing_symbol_is_refused_by_name():
    lib = FakeHip()
    lib.hipModuleLaunchKernel = None
    with pytest.raises(RuntimeError, match="hipModuleLaunchKernel"):
        L.HipDriver(lib=lib)


def test_a_launch_passes_upstreams_order(held):
    lib = FakeHip()
    d = L.HipDriver(lib=lib)
    f = d.load(b"hsaco", "k", 0)
    d.launch(f, (4, 2), (256, 1, 1), 1024, 7, [("ptr", 0xA0), ("i32", 5), ("ptr", 0)])
    name, grid, block, shared, stream, args = lib.calls[-1]
    assert (name, grid, block, shared, stream) == ("hipModuleLaunchKernel", (4, 2, 1), (256, 1, 1), 1024, 7)
    assert args[0] == 0xA0 and args[1] & 0xFFFFFFFF == 5 and args[2] == 0
    assert d.artifact_kind == "hsaco" and d.wants_scratch_params is True


def test_shared_memory_past_the_devices_block_maximum_is_refused_never_opted_in(held):
    # a module function takes its LDS at launch up to attribute 74; hipFuncSetAttribute takes a host stub and Triton's
    # AMD launcher calls none — past the maximum the config cannot run (OutOfResources, an autotune scores it inf)
    lib = FakeHip(attrs={74: 65536, 75: 65536, 87: 64})
    d = L.HipDriver(lib=lib)
    d.load(b"hsaco", "k", 65536)
    assert not [c for c in lib.calls if c[0] == "hipFuncSetAttribute"]
    from triton.runtime.errors import OutOfResources
    with pytest.raises(OutOfResources):
        d.load(b"hsaco", "k", 65537)


def test_the_block_is_the_code_objects_wavefront_never_a_default():
    import types
    d = L.HipDriver(lib=FakeHip())
    assert d.block_for(types.SimpleNamespace(warp_size=32, num_warps=4)) == (128, 1, 1)
    with pytest.raises(AttributeError):
        d.block_for(types.SimpleNamespace(num_warps=4))


def test_hidden_arguments_of_the_runtime_are_not_counted():
    meta = {"amdhsa.kernels": [{".name": "k", ".args": [{".value_kind": "global_buffer"}, {".value_kind": "by_value"},
                                                        {".value_kind": "hidden_block_count_x"}]}]}

    def pack(v):                                     # a minimal MessagePack writer for the test's map
        if isinstance(v, dict):
            return bytes([0x80 | len(v)]) + b"".join(pack(k) + pack(x) for k, x in v.items())
        if isinstance(v, list):
            return bytes([0x90 | len(v)]) + b"".join(pack(x) for x in v)
        e = v.encode()
        return bytes([0xA0 | len(e)]) + e if len(e) < 32 else bytes([0xD9, len(e)]) + e
    import struct
    desc = pack(meta)
    note = struct.pack("<III", 7, len(desc), 32) + b"AMDGPU\0\0" + desc + b"\0" * (-len(desc) % 4)
    shoff = 64 + len(note)
    elf = bytearray(b"\x7fELF" + b"\0" * 60) + note
    struct.pack_into("<Q", elf, 0x28, shoff)
    struct.pack_into("<HH", elf, 0x3A, 64, 2)
    elf += b"\0" * 64 + struct.pack("<IIQQQQ", 0, 7, 0, 0, 64, len(note)) + b"\0" * 16
    assert L._hsaco_param_count(bytes(elf), "k") == 2


def test_the_wavefront_of_the_target_is_the_device_files(monkeypatch):
    d = L.HipDriver(lib=FakeHip(attrs={87: 64}))     # the device would say 64: the RDNA3 file says 32 and wins
    monkeypatch.setattr(L, "_profile_arch", lambda: "gfx1100")
    assert d.target().warp_size == 32
    monkeypatch.setattr(L, "_profile_arch", lambda: "gfx9999")
    with pytest.raises(RuntimeError, match="warp_size"):
        d.target()


def test_an_error_carries_hips_own_string(held):
    lib = FakeHip(fail={"hipModuleLaunchKernel": 2})
    d = L.HipDriver(lib=lib)
    f = d.load(b"hsaco", "k", 0)
    with pytest.raises(RuntimeError, match="hipModuleLaunchKernel failed \\(2: hipErrorOutOfMemory\\)"):
        d.launch(f, (1,), (64, 1, 1), 0, 0, [("ptr", 0xA0), ("i32", 1), ("ptr", 0)])


def test_the_target_is_the_profiles_gfx_arch_and_the_devices_warp(monkeypatch):
    lib = FakeHip(attrs={87: 64})
    d = L.HipDriver(lib=lib)
    monkeypatch.setattr(L, "_profile_arch", lambda: "gfx90a")
    t = d.target()
    assert (t.backend, t.arch, t.warp_size) == ("hip", "gfx90a", 64)        # CDNA2's device file: wave64
    monkeypatch.setattr(L, "_profile_arch", lambda: "")
    with pytest.raises(RuntimeError, match="gfx"):
        d.target()


def test_hip_page_locked_host_memory_is_asked_with_its_flags_word():
    # hipHostMalloc(ptr, size, flags) takes a flags word cudaMallocHost(ptr, size) has not: left out, the runtime reads
    # whatever the register holds as flags. What this does if the code were wrong: two arguments on HIP, three on CUDA.
    from nbxtensor.memory.device_memory import AbiMemory
    seen = {}

    class Rt:
        def hipHostMalloc(self, *a):
            seen["hip"] = len(a)
            return 0

        def cudaMallocHost(self, *a):
            seen["cuda"] = len(a)
            return 0
    AbiMemory(Rt(), R._GPU_BACKENDS["hip"]).host_alloc(4096)
    AbiMemory(Rt(), R._GPU_BACKENDS["cuda"]).host_alloc(4096)
    assert seen == {"hip": 3, "cuda": 2}
