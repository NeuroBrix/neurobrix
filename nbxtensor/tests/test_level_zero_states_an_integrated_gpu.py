"""What an Intel GPU states of itself through Level Zero, asked card-less against a stand-in loader: whether it is
integrated with the host (ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0), ze_api.h v1.20-r1.20.41), the memory its heaps
hold, its largest single allocation (maxMemAllocSize) and its PCI id — what an application's first-launch detection
describes a unified device with. The device file of a PCI id is found without raising (`device_files.for_pci_id`), the
display controllers of the PCI bus are listed for any vendor (`install_help.pci_display_devices`), and the unified fact
of the current Level Zero device is the device's (`device_facts.host_shares_memory_with_device`).

What this test does if the code were wrong: the integrated flag read from another bit, the heaps not summed, the
allocation limit from another field, a loader-less machine answered with [] instead of None, a PCI id matched to no file
or to the wrong one, a display class other than 0x0300 dropped — red."""
import pytest

from nbxtensor.backend import device_files, install_help
from nbxtensor.backend import level_zero as Z

GIB = 1 << 30


def _obj(ref):
    return ref._obj if hasattr(ref, "_obj") else ref


class FakeZe:
    def __init__(self, devices):
        self.devices = devices          # [(device id, flags, heaps, max alloc, name)]

    def zeInit(self, flags):
        return 0

    def zeDriverGet(self, n, arr):
        _obj(n).value = 1
        if arr is not None:
            arr[0] = 0xD0
        return 0

    def zeDeviceGet(self, drv, n, arr):
        _obj(n).value = len(self.devices)
        if arr is not None:
            for i in range(len(self.devices)):
                arr[i] = 0xE0 + i
        return 0

    def _dev(self, dev):
        return self.devices[int(getattr(dev, "value", dev)) - 0xE0]

    def zeDeviceGetProperties(self, dev, props):
        did, flags, _heaps, mx, name = self._dev(dev)
        p = _obj(props)
        assert p.stype == Z._ST_DEVICE_PROPERTIES_1_2
        p.type, p.vendorId, p.deviceId, p.flags, p.maxMemAllocSize, p.name = 1, 0x8086, did, flags, mx, name
        return 0

    def zeDeviceGetMemoryProperties(self, dev, n, arr):
        heaps = self._dev(dev)[2]
        _obj(n).value = len(heaps)
        if arr is not None:
            for i, size in enumerate(heaps):
                assert arr[i].stype == Z._ST_DEVICE_MEMORY
                arr[i].totalSize = size
        return 0


def test_an_integrated_and_a_discrete_gpu_state_their_own_facts():
    lib = FakeZe([(0x64A0, 0x1, [15 * GIB], 4 * GIB, b"Intel(R) Arc(TM) 140V GPU (16GB)"),
                  (0xE20B, 0x4, [8 * GIB, 4 * GIB], 12 * GIB, b"Intel(R) Arc(TM) B580 Graphics")])
    lnl, bmg = Z.gpu_facts(lib)
    assert lnl == {"pci_device_id": 0x64A0, "name": "Intel(R) Arc(TM) 140V GPU (16GB)", "integrated": True,
                   "memory_bytes": 15 * GIB, "max_allocation_bytes": 4 * GIB}
    assert bmg["integrated"] is False and bmg["memory_bytes"] == 12 * GIB and bmg["max_allocation_bytes"] == 12 * GIB


def test_no_loader_is_no_answer_not_no_gpu(monkeypatch):
    monkeypatch.setattr(Z, "load_library", lambda: None)
    assert Z.gpu_facts() is None
    assert Z.gpu_facts(FakeZe([])) == []


def test_the_device_file_of_a_pci_id_is_found_or_none():
    name, desc = device_files.for_pci_id("intel", 0x64A0)
    assert name == "lnl" and desc["compute_capability"] == "lnl"
    assert device_files.for_pci_id("intel", 0x7D55)[0] == "mtl"
    assert device_files.for_pci_id("intel", 0x7D45)[0] == "arl_s"       # MTL-U: SYCL's mtl_u = arl_s
    assert device_files.for_pci_id("intel", 0x9A49) is None             # Tiger Lake: no compile target
    assert Z.device_file_for(0x64A0)[0] == "lnl"
    with pytest.raises(RuntimeError, match="0x9a49"):
        Z.device_file_for(0x9A49)


def test_every_display_controller_of_the_bus_is_listed_whatever_its_vendor(tmp_path):
    for bus, cls, vendor, device in [("0000:00:02.0", "0x038000", "0x8086", "0x46a6"),      # ADL-P as "Display"
                                     ("0000:03:00.0", "0x030000", "0x102b", "0x0536"),      # a BMC's Matrox VGA
                                     ("0000:00:1f.0", "0x060100", "0x8086", "0xa082")]:     # an ISA bridge
        d = tmp_path / bus
        d.mkdir()
        (d / "class").write_text(cls + "\n")
        (d / "vendor").write_text(vendor + "\n")
        (d / "device").write_text(device + "\n")
    assert install_help.pci_display_devices(str(tmp_path)) == [(0x8086, 0x46A6, "0000:00:02.0"),
                                                               (0x102B, 0x0536, "0000:03:00.0")]
    assert install_help.pci_gpus(str(tmp_path)) == [("intel", 0x8086, 0x46A6)]


def test_the_current_level_zero_device_says_whether_it_shares_the_host_memory(monkeypatch):
    from nbxtensor.backend import device_facts, runtime

    class Rt:
        def __init__(self, flags):
            self.flags = flags

        def properties(self, index=None):
            return type("P", (), {"flags": self.flags})()
    monkeypatch.setattr(runtime, "gpu_backend", lambda: "xpu")
    monkeypatch.setattr(Z, "runtime", lambda: Rt(0x1))
    assert device_facts.host_shares_memory_with_device() is True
    monkeypatch.setattr(Z, "runtime", lambda: Rt(0x4))            # ECC only: a discrete card
    assert device_facts.host_shares_memory_with_device() is False
