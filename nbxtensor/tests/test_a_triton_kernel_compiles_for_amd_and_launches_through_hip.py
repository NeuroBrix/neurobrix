"""A real Triton kernel compiled for AMD targets on a machine with no AMD card, then launched through the HIP driver over
a stand-in libamdhip64 — the whole launch path a ROCm machine runs, short of the card.

Triton 3.8 emits a code object for gfx90a, gfx942, gfx950, gfx1100 and gfx1201 here (read 2026-11-06). What this test
does if the code were wrong: the code object not the one handed to hipModuleLoadData, the block not warp size x
num_warps, an argument count other than the code object's own `.args` (the kernel's then the two scratch pointers), a
second module load for the same binary, the contract surface (`compile`) launching through another driver — red."""
import ctypes

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from nbxtensor.backend import launcher as L  # noqa: E402


def _kernel():
    """A fresh jit function per target: one process compiles for one target, so the launcher's compiled-kernel cache
    is keyed without it — each arch here starts from its own function."""
    @triton.jit
    def _add(x, y, o, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
    return _add


class _Buf:
    """A device address with its element type, as the launcher reads a tensor argument (data_ptr, dtype)."""
    dtype = "float32"

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


X, Y, O = _Buf(0xA000), _Buf(0xB000), _Buf(0xC000)


class FakeHip:
    def __init__(self, warp):
        self.loads, self.launches, self.warp = [], [], warp

    def hipModuleLoadData(self, module_ref, buf):
        self.loads.append(bytes(buf.raw))
        ctypes.cast(module_ref, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x1000 + len(self.loads)
        return 0

    def hipModuleGetFunction(self, function_ref, module, name):
        ctypes.cast(function_ref, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x2000
        return 0

    def hipFuncSetAttribute(self, *a):
        return 0

    def hipDeviceGetAttribute(self, out_ref, attr, dev):
        ctypes.cast(out_ref, ctypes.POINTER(ctypes.c_int)).contents.value = {74: 65536, 75: 65536, 87: self.warp}[attr]
        return 0

    def hipModuleLaunchKernel(self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        self.launches.append(((gx, gy, gz), (bx, by, bz), shared, stream))
        return 0

    def hipGetErrorString(self, rc):
        return b"?"


@pytest.fixture
def hip(monkeypatch, request):
    from triton.backends.compiler import GPUTarget
    from nbxtensor.memory.allocator import DeviceAllocator
    arch, warp = request.param
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: True))
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: 0))
    lib = FakeHip(warp)
    drv = L.HipDriver(lib=lib)
    monkeypatch.setattr(L, "_profile_arch", lambda: arch)
    L.register_driver(drv)
    yield drv, lib, GPUTarget("hip", arch, warp), _kernel()
    L.register_driver(None)


@pytest.mark.parametrize("hip", [("gfx90a", 64), ("gfx942", 64), ("gfx950", 64), ("gfx1100", 32), ("gfx1201", 32)],
                         indirect=True, ids=lambda p: p[0])
def test_the_launcher_compiles_for_the_arch_and_launches_its_code_object(hip):
    drv, lib, tgt, _add = hip
    assert L.target() == tgt                                  # the driver's answer: the profile's arch, the device's warp
    L.launch(_add, (3,), X, Y, O, 700, BLOCK=256, num_warps=4)
    L.launch(_add, (5,), X, Y, O, 1300, BLOCK=256, num_warps=4)
    assert len(lib.loads) == 1 and lib.loads[0][:4] == b"\x7fELF"          # one module, an ELF code object
    assert L._hsaco_param_count(lib.loads[0], "_add") == 6                 # x, y, o, n + the two scratch pointers
    assert lib.launches == [((3, 1, 1), (tgt.warp_size * 4, 1, 1), 0, 0), ((5, 1, 1), (tgt.warp_size * 4, 1, 1), 0, 0)]


@pytest.mark.parametrize("hip", [("gfx90a", 64)], indirect=True, ids=lambda p: p[0])
def test_the_contract_surface_compiles_and_launches_through_hip(hip):
    drv, lib, tgt, _add = hip
    k = drv.compile(_add, {"x": "*fp32", "y": "*fp32", "o": "*fp32", "n": "i32"}, {"BLOCK": 128}, num_warps=2)
    assert k.binary_kind == "hsaco" and k.block_size == 128 and k.binary == lib.loads[-1]
    k.launch((7,), [0xA000, 0xB000, 0xC000, 900], stream=11)
    assert lib.launches[-1] == ((7, 1, 1), (128, 1, 1), 0, 11)
    with pytest.raises(TypeError, match="takes 4 arguments"):
        k.launch((1,), [0xA000, 0xB000, 0xC000])


@pytest.mark.parametrize("hip", [("gfx90a", 64)], indirect=True, ids=lambda p: p[0])
def test_an_argument_list_of_another_length_is_refused(hip):
    drv, lib, tgt, _add = hip
    L.launch(_add, (1,), X, Y, O, 10, BLOCK=64, num_warps=1)
    function = ctypes.c_void_p(0x2000)
    with pytest.raises(TypeError, match="declares 6 parameters, 5 given"):
        drv.launch(function, (1,), (64, 1, 1), 0, 0, [("ptr", 1), ("ptr", 2), ("ptr", 3), ("i32", 4), ("ptr", 0)])
