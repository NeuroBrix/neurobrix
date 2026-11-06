"""Intel GPUs through Level Zero (`backend.level_zero`, `backend.level_zero_driver`), proven card-less against a stand-in
libze_loader and OpenCL ICD. No Intel card has run it (2026-11-06).

The contracts are Level Zero's (oneapi-src/level-zero include/ze.py: structure types, layouts) and Intel's Triton
launch ABI (intel-xpu-backend-for-triton v3.8.0 driver.c: SPIR-V modules with the compile's build flags, one rebuild
with 256 registers per thread on failure or a spill past 1000 bytes, the kernel's arguments then the two scratch
pointers then one shared-local-memory argument of its size and no value, a work-group of threads_per_warp x num_warps,
the grid as group count). What this test does if the code were wrong: a structure of another type or layout, a module
of another format, a lost build flag or retry, an argument at another index or size, the shared-memory argument with a
value, a group size or count from elsewhere, a count mismatch launched, a target fact guessed, a GPU-less loader naming
the backend — red."""
import ctypes
import types

import pytest

from nbxtensor.backend import level_zero as Z
from nbxtensor.backend import runtime as R

PVC = 0x0BD5


def _obj(ref):
    return ref._obj if hasattr(ref, "_obj") else ref


class FakeZe:
    """The loader's surface the runtime and driver use, recording every call."""

    def __init__(self, devices=1, gpu=True, usable=12 << 30, total=16 << 30, build_fail=0, spill=0, nargs=7):
        self.calls, self.devices, self.gpu = [], devices, gpu
        self.usable, self.total, self.build_fail, self.spill, self.nargs = usable, total, build_fail, spill, nargs
        self.next, self.args, self.mem = 0x5000, {}, {}

    def _h(self):
        self.next += 0x10
        return self.next

    def zeInit(self, flags):
        self.calls.append(("zeInit", flags.value))
        return 0

    def zeDriverGet(self, n, arr):
        _obj(n).value = 1
        if arr is not None:
            arr[0] = 0xD0
        return 0

    def zeDeviceGet(self, drv, n, arr):
        _obj(n).value = self.devices
        if arr is not None:
            for i in range(self.devices):
                arr[i] = 0xE0 + i
        return 0

    def zeDeviceGetProperties(self, dev, props):
        p = _obj(props)
        assert p.stype == Z._ST_DEVICE_PROPERTIES_1_2
        p.type = 1 if self.gpu else 2
        p.vendorId, p.deviceId, p.coreClockRate = 0x8086, PVC, 1600
        p.numSlices, p.numSubslicesPerSlice, p.numEUsPerSubslice = 1, 56, 8
        p.timerResolution, p.timestampValidBits = 1_000_000_000, 64
        p.name = b"Intel(R) Data Center GPU Max 1100"
        if p.pNext:
            ext = ctypes.cast(p.pNext, ctypes.POINTER(Z._UsableMem)).contents
            assert ext.stype == Z._ST_USABLE_MEM
            ext.currUsableMemSize = self.usable
        return 0

    def zeDeviceGetComputeProperties(self, dev, props):
        c = _obj(props)
        assert c.stype == Z._ST_DEVICE_COMPUTE
        c.maxTotalGroupSize, c.maxSharedLocalMemory, c.numSubGroupSizes = 1024, 131072, 2
        c.subGroupSizes[0], c.subGroupSizes[1] = 16, 32
        return 0

    def zeDeviceGetMemoryProperties(self, dev, n, arr):
        _obj(n).value = 1
        if arr is not None:
            assert arr[0].stype == Z._ST_DEVICE_MEMORY
            arr[0].totalSize = self.total
        return 0

    def zeDriverGetExtensionFunctionAddress(self, drv, name, out):
        return Z._ERR_UNSUPPORTED_FEATURE             # no Intel version-string extension: the encoded version

    def zeDriverGetProperties(self, drv, props):
        _obj(props).driverVersion = 0x01060000
        return 0

    def zeContextCreate(self, drv, desc, out):
        assert _obj(desc).stype == Z._ST_CONTEXT_DESC
        _obj(out).value = 0xC0
        return 0

    def zeDeviceGetCommandQueueGroupProperties(self, dev, n, arr):
        _obj(n).value = 2
        if arr is not None:
            arr[0].flags, arr[1].flags = 2, 1          # a copy-only group, then the compute group
        return 0

    def zeCommandListCreateImmediate(self, ctx, dev, desc, out):
        d = _obj(desc)
        self.calls.append(("list", d.stype, d.ordinal, d.flags))
        _obj(out).value = self._h()
        return 0

    def zeMemAllocDevice(self, ctx, desc, size, align, dev, out):
        assert _obj(desc).stype == Z._ST_DEVICE_MEM_ALLOC
        _obj(out).value = h = self._h()
        self.mem[h] = size.value
        return 0

    def zeMemAllocHost(self, ctx, desc, size, align, out):
        assert _obj(desc).stype == Z._ST_HOST_MEM_ALLOC
        buf = ctypes.create_string_buffer(size.value)
        self.mem[ctypes.addressof(buf)] = buf
        _obj(out).value = ctypes.addressof(buf)
        return 0

    def zeMemFree(self, ctx, p):
        self.calls.append(("zeMemFree", p.value))
        return 0

    def zeCommandListAppendMemoryCopy(self, lst, dst, src, size, ev, n, evs):
        self.calls.append(("copy", lst.value, dst.value, src.value, size.value))
        return 0

    def zeCommandListAppendMemoryFill(self, lst, ptr, pattern, psize, size, ev, n, evs):
        self.calls.append(("fill", ptr.value, ctypes.cast(pattern, ctypes.POINTER(ctypes.c_uint8)).contents.value,
                           psize.value, size.value))
        return 0

    def zeCommandListHostSynchronize(self, lst, timeout):
        self.calls.append(("sync", lst.value))
        return 0

    def zeModuleCreate(self, ctx, dev, desc, out, log):
        d = _obj(desc)
        self.calls.append(("module", d.stype, d.format, d.inputSize, d.pBuildFlags))
        if self.build_fail:
            self.build_fail -= 1
            return 0x70000004
        _obj(out).value = self._h()
        return 0

    def zeModuleBuildLogGetString(self, log, n, buf):
        _obj(n).value = 1
        return 0

    def zeModuleBuildLogDestroy(self, log):
        return 0

    def zeModuleDestroy(self, mod):
        return 0

    def zeKernelCreate(self, mod, desc, out):
        d = _obj(desc)
        self.calls.append(("kernel", d.stype, d.flags, d.pKernelName))
        _obj(out).value = self._h()
        return 0

    def zeKernelDestroy(self, k):
        return 0

    def zeKernelGetProperties(self, k, props):
        p = _obj(props)
        p.numKernelArgs, p.spillMemSize = self.nargs, self.spill
        self.spill = 0                                   # the rebuilt kernel no longer spills
        return 0

    def zeKernelSetArgumentValue(self, k, idx, size, value):
        v = None if value is None else bytes(ctypes.string_at(ctypes.addressof(_obj(value)), size.value))
        self.args[idx.value] = (size.value, v)
        return 0

    def zeKernelSetGroupSize(self, k, x, y, z):
        self.calls.append(("group", x.value, y.value, z.value))
        return 0

    def zeCommandListAppendLaunchKernel(self, lst, k, count, ev, n, evs):
        c = _obj(count)
        self.calls.append(("launch", c.x, c.y, c.z))
        return 0


@pytest.fixture
def ze(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: True))
    lib = FakeZe()
    return lib, Z.LevelZeroRuntime(lib)


def test_a_loader_without_a_gpu_does_not_name_the_backend():
    assert Z._gpu_devices(FakeZe(gpu=False)) == []
    assert Z._gpu_devices(FakeZe(devices=0)) == []
    assert Z._gpu_devices(FakeZe()) == [(0xD0, 0xE0)]
    assert R._DEVICE_PROBES == (("metal", "nbxtensor.backend.metal_device.metal_device_available"),
                                ("xpu", "nbxtensor.backend.level_zero.level_zero_device_available"))
    assert R._TRITON_BACKEND_BY_RUNTIME["xpu"] == "intel"


def test_the_runtime_serves_the_allocators_c_abi(ze):
    lib, rt = ze
    from nbxtensor.memory.device_memory import AbiMemory
    dm = AbiMemory(rt, Z.XPU_BACKEND)
    rc, p = dm.malloc(1 << 20)
    assert rc == 0 and lib.mem[p] == 1 << 20
    assert dm.mem_get_info() == (0, 12 << 30, 16 << 30)              # free: the usable-memory extension
    assert dm.memset(p, 0x1FF, 64) == 0 and ("fill", p, 0xFF, 1, 64) in lib.calls
    assert dm.memcpy(p, 0x9000, 128, 1) == 0
    copy = [c for c in lib.calls if c[0] == "copy"][-1]
    assert copy[2:] == (p, 0x9000, 128) and lib.calls[-1][0] == "sync"
    assert ("list", Z._ST_QUEUE_DESC, 1, Z._QUEUE_FLAG_IN_ORDER) in lib.calls   # the compute group, in order
    assert dm.supports("host_register") is False
    assert dm.free(p) == 0 and ("zeMemFree", p) in lib.calls
    lib.usable = 0
    assert dm.mem_get_info()[0] == Z._ERR_UNSUPPORTED_FEATURE        # no free-memory answer: refused, not guessed


def test_two_timing_events_measure_the_device_timestamps(ze):
    lib, rt = ze
    a, b, out = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_float()
    lib.zeEventPoolCreate = lambda ctx, d, n, devs, out_: (setattr(_obj(out_), "value", lib._h()), 0)[1]
    lib.zeEventCreate = lambda pool, d, out_: (setattr(_obj(out_), "value", lib._h()), 0)[1]
    lib.zeEventHostReset = lambda e: 0
    lib.zeEventHostSynchronize = lambda e, t: 0
    lib.zeCommandListAppendWriteGlobalTimestamp = lambda lst, slot, e, n, evs: 0
    assert rt.event_create(ctypes.byref(a)) == 0 and rt.event_create(ctypes.byref(b)) == 0
    assert rt.event_record(a) == 0 and rt.event_record(b) == 0
    ctypes.c_uint64.from_address(rt._events[a.value]["slot"]).value = 1_000_000
    ctypes.c_uint64.from_address(rt._events[b.value]["slot"]).value = 3_500_000
    assert rt.event_elapsed(ctypes.byref(out), a, b) == 0
    assert abs(out.value - 2.5) < 1e-6                                # 2.5 M cycles at 1 GHz


@pytest.fixture
def drv(ze):
    from nbxtensor.backend.level_zero_driver import LevelZeroDriver
    lib, rt = ze
    return lib, LevelZeroDriver(rt)


def test_a_spirv_module_is_built_with_its_flags_and_retried_with_256_registers(drv):
    lib, d = drv
    lib.build_fail = 1
    d.load(b"\x03\x02\x23\x07spirv", "k", 0, build_flags="-cl-opt-disable")
    mods = [c for c in lib.calls if c[0] == "module"]
    assert [m[2] for m in mods] == [Z._MODULE_FORMAT_IL_SPIRV] * 2
    assert [m[4] for m in mods] == [b"-cl-opt-disable", b"-cl-opt-disable -cl-intel-256-GRF-per-thread"]
    assert ("kernel", Z._ST_KERNEL_DESC, Z._KERNEL_FLAG_FORCE_RESIDENCY, b"k") in lib.calls
    lib.calls.clear()
    lib.spill = 2048
    d.load(b"other", "k", 0)
    assert [m[4] for m in lib.calls if m[0] == "module"] == [b"", b"-cl-intel-256-GRF-per-thread"]


def test_a_launch_sets_intels_argument_order_and_shared_local_memory(drv):
    lib, d = drv
    k = d.load(b"spv", "k", 4096)                      # 4 own arguments + 2 scratch + 1 shared local memory
    params = [("ptr", 0xA0), ("ptr", 0xB0), ("i32", 7), ("bits32", 0x3F800000), ("ptr", 0), ("ptr", 0)]
    d.launch(k, (3, 2), (512, 1, 1), 4096, 0, params)
    assert lib.args[0] == (8, (0xA0).to_bytes(8, "little")) and lib.args[2] == (4, (7).to_bytes(4, "little"))
    assert lib.args[3] == (4, (0x3F800000).to_bytes(4, "little")) and lib.args[5] == (8, bytes(8))
    assert lib.args[6] == (4096, None)                                 # the size, no value
    assert ("group", 512, 1, 1) in lib.calls and lib.calls[-1] == ("launch", 3, 2, 1)
    with pytest.raises(TypeError, match="declares 6 parameters, 5 given"):
        d.launch(k, (1,), (512, 1, 1), 4096, 0, params[:5])
    md = types.SimpleNamespace(threads_per_warp=16, num_warps=8, build_flags="-x")
    assert d.block_for(md) == (128, 1, 1) and d.load_options(md) == {"build_flags": "-x"}
    from triton.runtime.errors import OutOfResources
    with pytest.raises(OutOfResources):
        d.load(b"spv2", "k", 131073)


class FakeCl:
    def __init__(self, exts):
        self.exts = exts

    def clGetPlatformIDs(self, n, arr, out):
        if out is not None:
            _obj(out).value = 1
        if arr is not None:
            arr[0] = 0x1
        return 0

    def clGetDeviceIDs(self, plat, typ, n, arr, out):
        if out is not None:
            _obj(out).value = 1
        if arr is not None:
            arr[0] = 0x2
        return 0

    def clGetDeviceInfo(self, dev, param, size, value, out):
        p = param.value
        if p == Z._CL_DEVICE_VENDOR_ID:
            _obj(value).value = 0x8086
        elif p == Z._CL_DEVICE_ID_INTEL:
            _obj(value).value = PVC
        elif p == Z._CL_DEVICE_EXTENSIONS:
            data = self.exts.encode() + b"\0"
            if out is not None:
                _obj(out).value = len(data)
            if value is not None:
                ctypes.memmove(value, data, len(data))
        return 0


def test_the_target_is_asked_of_the_device_and_its_file_never_guessed(drv, monkeypatch):
    lib, d = drv
    exts = "cl_khr_fp64 cl_intel_subgroup_2d_block_io cl_intel_subgroup_matrix_multiply_accumulate"
    import nbxtensor.backend.level_zero_driver as LD
    monkeypatch.setattr(LD, "device_extensions", lambda did: Z.device_extensions(did, FakeCl(exts)))
    import triton
    if not hasattr(triton.knobs, "intel"):                            # the PyPI Triton: no Intel backend, refused
        with pytest.raises(RuntimeError, match="no Intel backend"):
            d.target()
        monkeypatch.setattr(triton.knobs, "intel", types.SimpleNamespace(device_arch=None), raising=False)
        d._target = None
    t = d.target()
    assert (t.backend, t.warp_size, t.arch["arch"]) == ("xpu", 32, "pvc") and triton.knobs.intel.device_arch == "pvc"
    assert t.arch["has_fp64"] and t.arch["has_2d_block_io"] and t.arch["has_subgroup_matrix_multiply_accumulate"]
    assert t.arch["has_bfloat16_conversion"] is False and t.arch["sub_group_sizes"] == [16, 32]
    assert t.arch["driver_version"] == "1.6.0"                        # | major | minor | build |, as SYCL reads it
    assert t.arch["gpu_eu_count"] == 448 and t.arch["max_num_sub_groups"] == 64 and t.arch["local_mem_size"] == 131072
    from nbxtensor.backend.launcher import arch_name
    assert arch_name(t) == "pvc"



def test_an_unknown_intel_device_is_refused_by_name():
    with pytest.raises(RuntimeError, match="no Intel device file lists PCI device 0x1234"):
        Z.device_file_for(0x1234)
    assert Z.device_file_for(0xE20B)[0] == "bmg" and Z.device_file_for(0x56A0)[0] == "dg2"


def test_install_help_names_what_a_present_gpu_lacks(tmp_path, monkeypatch):
    from nbxtensor.backend import install_help as H

    def card(name, vendor, device):
        d = tmp_path / name
        d.mkdir()
        (d / "class").write_text("0x030000\n")
        (d / "vendor").write_text(f"{vendor:#06x}\n")
        (d / "device").write_text(f"{device:#06x}\n")
    card("0000:03:00.0", 0x1002, 0x744C)
    card("0000:04:00.0", 0x8086, 0xE20B)
    monkeypatch.setattr(H, "_loads", lambda names: False)
    monkeypatch.setattr(H, "_triton_backends", lambda: ["amd", "nvidia"])
    text = H.report(str(tmp_path))
    assert "AMD GPU (0x744c)" in text and "repo.radeon.com/amdgpu-install" in text and "sudo apt install rocm" in text
    assert "INTEL GPU (0xe20b)" in text and "libze-intel-gpu1" in text and "pip install triton-xpu" in text
    assert "pip install triton\n" not in text                         # the AMD backend is there: no Triton step
    assert H._steps()["install_page"] in text
    msg = H.no_runtime_message(str(tmp_path))
    assert msg.startswith("No GPU runtime found") and "sudo apt install rocm" in msg
    empty = tmp_path / "none"
    empty.mkdir()
    assert H.no_runtime_message(str(empty)) == "No GPU runtime found (tried CUDA, ROCm/HIP, Level Zero and Metal)."


def test_the_current_device_is_per_thread_and_a_free_uses_the_allocating_context(ze):
    import threading
    lib, rt = ze
    lib.devices = 2
    rt2 = Z.LevelZeroRuntime(lib)
    assert rt2.set_device(1) == 0
    seen = []
    t = threading.Thread(target=lambda: seen.append(rt2._current))
    t.start()
    t.join()
    assert seen == [0] and rt2._current == 1                          # another thread keeps its own device
    p = ctypes.c_void_p()
    assert rt2.malloc(ctypes.byref(p), 64) == 0
    ctx = rt2._sizes[p.value][1]
    rt2.set_device(0)
    freed = []
    lib.zeMemFree = lambda c, q: (freed.append(c.value), 0)[1]
    assert rt2.free(p.value) == 0 and freed == [ctx.value]
    assert rt2.free(0xDEAD) == Z._ERR_INVALID_ARGUMENT                 # not ours: refused, never freed


def test_a_kernel_is_created_once_per_module_and_name(drv):
    lib, d = drv
    k1 = d.load(b"spv", "k", 0)
    k2 = d.load(b"spv", "k", 0)
    assert k1 is k2 and len([c for c in lib.calls if c[0] == "kernel"]) == 1


def test_the_structures_are_the_published_bindings_layouts():
    # sizes and the offset of the last field, as the loader's own ctypes bindings lay them out (oneapi-src/level-zero
    # include/ze.py, v1.20-r1.20.41, compared field by field on 2026-11-06) — independent of the stand-ins above,
    # which share this module's assumptions. A field added, dropped or retyped here moves one of these numbers.
    published = {'_Desc': (24, 16), '_QueueDesc': (40, 32), '_QueueGroupProps': (40, 32),
                 '_DeviceMemAllocDesc': (24, 20), '_UsableMem': (24, 16), '_DeviceProps': (368, 112),
                 '_ComputeProps': (88, 52), '_MemoryProps': (296, 40), '_DriverProps': (40, 32),
                 '_ModuleDesc': (56, 48), '_KernelDesc': (32, 24), '_KernelProps': (96, 76), '_GroupCount': (12, 8),
                 '_EventPoolDesc': (24, 20), '_EventDesc': (32, 24)}       # read from ze.py itself, not written by hand
    for name, (size, last) in published.items():
        cls = getattr(Z, name)
        assert (ctypes.sizeof(cls), getattr(cls, cls._fields_[-1][0]).offset) == (size, last), name
    assert (Z._ST_DEVICE_PROPERTIES_1_2, Z._ST_USABLE_MEM, Z._ST_KERNEL_PROPS) == (0x00020006, 0x00020041, 0x1E)
