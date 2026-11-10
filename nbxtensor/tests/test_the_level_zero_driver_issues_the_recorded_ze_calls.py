"""The exact Level Zero calls the launch driver issues for a load and a launch, with their arguments and their order,
RECORDED on main (02b1a38cf) against the stand-in loader. No Intel card exists in the lab: a regroup
of the driver and its runtime (who knows a ze struct, who issues a ze call) is proven by reproducing this recording
with this file untouched — never by a run. The paths a regroup most easily breaks are here by name: the build log read
and destroyed on a failed build, the spilling kernel and its module destroyed IN THAT ORDER when the rebuilt one
replaces them, nothing destroyed that was never created.
What this would do if the code were wrong: on a real chip, a leaked module or build log per failed build, a kernel
destroyed after its module, or arguments bound in another order — and no machine here to say so."""
import ctypes
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("triton")

_spec = importlib.util.spec_from_file_location(
    "_lz_contracts", Path(__file__).with_name("test_the_level_zero_backend_keeps_its_contracts.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
Z = _base.Z


def _v(x):
    x = x._obj if hasattr(x, "_obj") else x
    return getattr(x, "value", x)


class Recording(_base.FakeZe):
    """The stand-in loader, every module / kernel / launch call logged with what it was given. A failed build hands
    back a build log (as the driver's compiler does) and no module."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.log = []

    def zeModuleCreate(self, ctx, dev, desc, out, log):
        d = _base._obj(desc)
        rc = super().zeModuleCreate(ctx, dev, desc, out, log)
        if rc:
            _base._obj(log).value = self._h()
        self.log.append(("zeModuleCreate", d.format, d.inputSize, d.pBuildFlags, hex(rc), _v(out) or 0, _v(log) or 0))
        return rc

    def zeModuleBuildLogGetString(self, log, n, buf):
        self.log.append(("zeModuleBuildLogGetString", _v(log), buf is not None))
        return super().zeModuleBuildLogGetString(log, n, buf)

    def zeModuleBuildLogDestroy(self, log):
        self.log.append(("zeModuleBuildLogDestroy", _v(log)))
        return super().zeModuleBuildLogDestroy(log)

    def zeModuleDestroy(self, mod):
        self.log.append(("zeModuleDestroy", _v(mod)))
        return super().zeModuleDestroy(mod)

    def zeKernelCreate(self, mod, desc, out):
        d = _base._obj(desc)
        rc = super().zeKernelCreate(mod, desc, out)
        self.log.append(("zeKernelCreate", _v(mod), d.flags, d.pKernelName, _v(out)))
        return rc

    def zeKernelDestroy(self, k):
        self.log.append(("zeKernelDestroy", _v(k)))
        return super().zeKernelDestroy(k)

    def zeKernelGetProperties(self, k, props):
        spill = self.spill
        rc = super().zeKernelGetProperties(k, props)
        self.log.append(("zeKernelGetProperties", _v(k), self.nargs, spill))
        return rc

    def zeKernelSetArgumentValue(self, k, idx, size, value):
        v = None if value is None else bytes(ctypes.string_at(ctypes.addressof(_base._obj(value)), size.value)).hex()
        self.log.append(("zeKernelSetArgumentValue", _v(k), idx.value, size.value, v))
        return super().zeKernelSetArgumentValue(k, idx, size, value)

    def zeKernelSetGroupSize(self, k, x, y, z):
        self.log.append(("zeKernelSetGroupSize", _v(k), x.value, y.value, z.value))
        return super().zeKernelSetGroupSize(k, x, y, z)

    def zeCommandListAppendLaunchKernel(self, lst, k, count, ev, n, evs):
        c = _base._obj(count)
        self.log.append(("zeCommandListAppendLaunchKernel", _v(k), c.x, c.y, c.z, ev is None, _v(n), evs is None))
        return super().zeCommandListAppendLaunchKernel(lst, k, count, ev, n, evs)


def _driver(monkeypatch, **kw):
    from nbxtensor.backend.level_zero_driver import LevelZeroDriver
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: True))
    lib = Recording(**kw)
    return lib, LevelZeroDriver(Z.LevelZeroRuntime(lib))


PARAMS = [("ptr", 0xA0), ("ptr", 0xB0), ("i32", 7), ("bits32", 0x3F800000), ("ptr", 0), ("ptr", 0)]


def scenarios(monkeypatch):
    out = {}
    lib, d = _driver(monkeypatch)
    d.load(b"\x03\x02\x23\x07spirv", "k", 0, build_flags="-cl-opt-disable")
    out["a plain load"] = list(lib.log)
    lib.log.clear()
    d.load(b"\x03\x02\x23\x07spirv", "k", 0, build_flags="-cl-opt-disable")
    out["the same load again issues nothing"] = list(lib.log)
    d.load(b"\x03\x02\x23\x07spirv", "other", 0, build_flags="-cl-opt-disable")
    out["another kernel of the same module"] = list(lib.log)

    lib, d = _driver(monkeypatch, build_fail=1)
    d.load(b"spv", "k", 0)
    out["a failed build, rebuilt with 256 registers"] = list(lib.log)

    lib, d = _driver(monkeypatch, build_fail=2)
    with pytest.raises(Z.IntelDeviceError) as e:
        d.load(b"spv", "k", 0)
    out["two failed builds are a refusal"] = list(lib.log) + [("raised", e.value.call, e.value.result)]

    lib, d = _driver(monkeypatch, build_fail=1)
    with pytest.raises(Z.IntelDeviceError) as e:
        d.load(b"spv", "k", 0, build_flags="-cl-intel-128-GRF-per-thread")
    out["a failed build whose register mode was chosen is not retried"] = list(lib.log) + [("raised", e.value.call)]

    lib, d = _driver(monkeypatch, spill=2048)
    d.load(b"spv", "k", 0)
    out["a spilling kernel, its module rebuilt and both destroyed"] = list(lib.log)

    lib, d = _driver(monkeypatch, nargs=6)
    k = d.load(b"spv", "k", 0)
    lib.log.clear()
    d.launch(k, (3, 2), (512, 1, 1), 0, 0, PARAMS)
    out["a launch"] = list(lib.log)

    lib, d = _driver(monkeypatch)
    k = d.load(b"spv", "k", 4096)
    lib.log.clear()
    d.launch(k, (3, 2, 4), (256, 1, 1), 4096, 0, PARAMS)
    out["a launch with shared local memory"] = list(lib.log)
    lib.log.clear()
    d.launch(k, (0, 2), (256, 1, 1), 4096, 0, PARAMS)
    out["an empty grid issues nothing"] = list(lib.log)
    return out


#: recorded on main (02b1a38cf), 2026-11-10 — handles are the stand-in's counter
RECORDED = {'a plain load': [('zeModuleCreate', 0, 9, b'-cl-opt-disable', '0x0', 20496, 0),
                  ('zeKernelCreate', 20496, 1, b'k', 20512),
                  ('zeKernelGetProperties', 20512, 7, 0)],
 'the same load again issues nothing': [],
 'another kernel of the same module': [('zeKernelCreate', 20496, 1, b'other', 20528),
                                       ('zeKernelGetProperties', 20528, 7, 0)],
 'a failed build, rebuilt with 256 registers': [('zeModuleCreate', 0, 3, b'', '0x70000004', 0, 20496),
                                                ('zeModuleBuildLogGetString', 20496, False),
                                                ('zeModuleBuildLogGetString', 20496, True),
                                                ('zeModuleBuildLogDestroy', 20496),
                                                ('zeModuleCreate',
                                                 0,
                                                 3,
                                                 b'-cl-intel-256-GRF-per-thread',
                                                 '0x0',
                                                 20512,
                                                 0),
                                                ('zeKernelCreate', 20512, 1, b'k', 20528),
                                                ('zeKernelGetProperties', 20528, 7, 0)],
 'two failed builds are a refusal': [('zeModuleCreate', 0, 3, b'', '0x70000004', 0, 20496),
                                     ('zeModuleBuildLogGetString', 20496, False),
                                     ('zeModuleBuildLogGetString', 20496, True),
                                     ('zeModuleBuildLogDestroy', 20496),
                                     ('zeModuleCreate',
                                      0,
                                      3,
                                      b'-cl-intel-256-GRF-per-thread',
                                      '0x70000004',
                                      0,
                                      20512),
                                     ('zeModuleBuildLogGetString', 20512, False),
                                     ('zeModuleBuildLogGetString', 20512, True),
                                     ('zeModuleBuildLogDestroy', 20512),
                                     ('raised', 'zeModuleCreate', 'ZE_RESULT_ERROR_MODULE_BUILD_FAILURE')],
 'a failed build whose register mode was chosen is not retried': [('zeModuleCreate',
                                                                   0,
                                                                   3,
                                                                   b'-cl-intel-128-GRF-per-thread',
                                                                   '0x70000004',
                                                                   0,
                                                                   20496),
                                                                  ('zeModuleBuildLogGetString', 20496, False),
                                                                  ('zeModuleBuildLogGetString', 20496, True),
                                                                  ('zeModuleBuildLogDestroy', 20496),
                                                                  ('raised', 'zeModuleCreate')],
 'a spilling kernel, its module rebuilt and both destroyed': [('zeModuleCreate', 0, 3, b'', '0x0', 20496, 0),
                                                              ('zeKernelCreate', 20496, 1, b'k', 20512),
                                                              ('zeKernelGetProperties', 20512, 7, 2048),
                                                              ('zeModuleCreate',
                                                               0,
                                                               3,
                                                               b'-cl-intel-256-GRF-per-thread',
                                                               '0x0',
                                                               20528,
                                                               0),
                                                              ('zeKernelDestroy', 20512),
                                                              ('zeModuleDestroy', 20496),
                                                              ('zeKernelCreate', 20528, 1, b'k', 20544),
                                                              ('zeKernelGetProperties', 20544, 7, 0)],
 'a launch': [('zeKernelSetArgumentValue', 20512, 0, 8, 'a000000000000000'),
              ('zeKernelSetArgumentValue', 20512, 1, 8, 'b000000000000000'),
              ('zeKernelSetArgumentValue', 20512, 2, 4, '07000000'),
              ('zeKernelSetArgumentValue', 20512, 3, 4, '0000803f'),
              ('zeKernelSetArgumentValue', 20512, 4, 8, '0000000000000000'),
              ('zeKernelSetArgumentValue', 20512, 5, 8, '0000000000000000'),
              ('zeKernelSetGroupSize', 20512, 512, 1, 1),
              ('zeCommandListAppendLaunchKernel', 20512, 3, 2, 1, True, 0, True)],
 'a launch with shared local memory': [('zeKernelSetArgumentValue', 20512, 0, 8, 'a000000000000000'),
                                       ('zeKernelSetArgumentValue', 20512, 1, 8, 'b000000000000000'),
                                       ('zeKernelSetArgumentValue', 20512, 2, 4, '07000000'),
                                       ('zeKernelSetArgumentValue', 20512, 3, 4, '0000803f'),
                                       ('zeKernelSetArgumentValue', 20512, 4, 8, '0000000000000000'),
                                       ('zeKernelSetArgumentValue', 20512, 5, 8, '0000000000000000'),
                                       ('zeKernelSetArgumentValue', 20512, 6, 4096, None),
                                       ('zeKernelSetGroupSize', 20512, 256, 1, 1),
                                       ('zeCommandListAppendLaunchKernel', 20512, 3, 2, 4, True, 0, True)],
 'an empty grid issues nothing': []}


def test_the_driver_issues_the_recorded_calls(monkeypatch):
    got = scenarios(monkeypatch)
    for name, calls in RECORDED.items():
        assert got[name] == calls, name
    assert set(got) == set(RECORDED)
