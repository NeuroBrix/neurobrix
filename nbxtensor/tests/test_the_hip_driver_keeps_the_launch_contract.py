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


def test_the_library_is_opened_by_the_runtime_tables_names(monkeypatch):
    tried = []

    def cdll(name):
        tried.append(name)
        raise OSError(name)
    monkeypatch.setattr(L.ctypes, "CDLL", cdll)
    with pytest.raises(RuntimeError, match="libamdhip64"):
        L.HipDriver()
    assert tried == list(R._GPU_BACKENDS["hip"]["rt_libs"])


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


def test_a_large_shared_memory_ask_opts_in_and_one_past_the_device_is_refused(held, monkeypatch):
    lib = FakeHip(attrs={74: 32768, 75: 65536, 87: 64})
    d = L.HipDriver(lib=lib)
    d.load(b"hsaco", "k", 49152)
    assert ("hipFuncSetAttribute", 8, 49152) in lib.calls       # hipFuncAttributeMaxDynamicSharedMemorySize
    from triton.runtime.errors import OutOfResources
    with pytest.raises(OutOfResources):
        d.load(b"hsaco", "k", 65537)


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
    assert (t.backend, t.arch, t.warp_size) == ("hip", "gfx90a", 64)
    monkeypatch.setattr(L, "_profile_arch", lambda: "")
    with pytest.raises(RuntimeError, match="gfx"):
        d.target()
