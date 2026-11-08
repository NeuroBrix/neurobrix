"""First launch on a consumer machine with an integrated GPU: the profile is generated from what the operating system
ships (the PCI bus in sysfs, Windows' Win32_VideoController), never from a vendor tool the user did not install; a
chip a backend drives is described as unified memory with its facts asked of the device; a chip no backend drives is
refused BY NAME and the machine plans on its CPU — never refused whole.

The owner, 2026-11-08: a contributor's Tiger Lake laptop (Iris Xe, no discrete card) had every plan refused with "no
vendor profile for intel/unknown". Measured on main 13480627: the Linux cascade asked xpu-smi and sycl-ls (oneAPI
tools, absent), then lspci, which wrote `architecture: unknown` and an invented 16 384 MB; no vendor file is named
"unknown", so the whole machine was unusable, its CPU included.

No Intel machine is in the lab. Every output below is a REAL one, cited:
- TIGER_LAKE_DG1: `lspci -Dnn` of a Tiger Lake laptop with a DG1, https://discussion.fedoraproject.org/t/the-discrete-gpu-is-not-working/84088
- METEOR_LAKE: `lspci -nn`, https://bbs.archlinux.org/viewtopic.php?id=307264 (the domain `0000:` is what -D prefixes)
- PHOENIX: `lspci -nn` of an AMD Phoenix APU, https://github.com/isc30/ryzen-7000-series-proxmox/discussions/27
- UHD_730: Win32_VideoController of a UHD 730 desktop (Name, PNPDeviceID), https://gist.github.com/Lapish/f15f4dcbcd3ecdb5d82551ba4e6036bd
  — the values of that dump in ConvertTo-Json's form.
- Iris Xe on Windows: the dxdiag card name of a Dell XPS 17 (https://www.dell.com/community/en/conversations/xps/drop-in-vram-clk-from-9001mhz-to-405mhz/677ced26db356218a151c1df)
  and the hardware id PCI\\VEN_8086&DEV_9A49&SUBSYS_3F9B17AA&REV_01 (driveridentifier.com, a search snippet).
The Level Zero facts of a driven chip are not an observed output (none was found): they follow compute-runtime's own
formula for an integrated GPU's global memory, min(RAM, GTT) x 0.94 - 450 MB (shared/source/device/device.cpp,
drm_memory_manager.cpp), and its 4 GB default single allocation (programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md).

What this file does if the code were wrong: an Intel GPU written as "unknown", an invented memory, a vendor tool asked,
a chip no backend drives planned on, the machine refused whole, a driven integrated chip not unified or without its
largest allocation, a refusal without its reason, the Display-controller class (0x0380) missed — red.
"""
from __future__ import annotations

import functools
import json
import subprocess

import pytest
import yaml

from neurobrix.core.prism import autodetect

TIGER_LAKE_DG1 = ("0000:00:02.0 VGA compatible controller [0300]: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] "
                  "[8086:9a49] (rev 01)\n"
                  "0000:03:00.0 VGA compatible controller [0300]: Intel Corporation DG1 [Iris Xe MAX Graphics] "
                  "[8086:4905] (rev 01)\n")
METEOR_LAKE = ("0000:00:02.0 VGA compatible controller [0300]: Intel Corporation Meteor Lake-P [Intel Arc Graphics] "
               "[8086:7d55] (rev 08)\n")
PHOENIX = ("0000:c4:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1 [1002:15bf] "
           "(rev c2)\n")
UHD_730 = {"Name": "Intel(R) UHD Graphics 730",
           "PNPDeviceID": "PCI\\VEN_8086&DEV_4692&SUBSYS_D0001458&REV_0C\\3&11583659&0&10"}
IRIS_XE_WINDOWS = {"Name": "Intel(R) Iris(R) Xe Graphics", "PNPDeviceID": "PCI\\VEN_8086&DEV_9A49&SUBSYS_3F9B17AA&REV_01"}

_VENDOR_TOOLS = {"nvidia-smi", "rocminfo", "rocm-smi", "xpu-smi", "sycl-ls", "tt-smi", "mtsmi", "brsmi", "ixsmi",
                 "cnmon", "wmic"}
GIB = 1 << 30


def _lspci_rows(text):
    """(bus, class, vendor, device) of each `lspci -Dnn` line — the sysfs facts the same machine shows."""
    import re
    out = []
    for line in text.strip().splitlines():
        m = re.match(r"(\S+) .*\[(03[0-9a-f]{2})\]: .*\[([0-9a-f]{4}):([0-9a-f]{4})\]", line)
        out.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


def _machine(monkeypatch, tmp_path, *, lspci="", windows=None, level_zero=None, klass=None):
    """A machine seen through its operating system: its sysfs PCI tree (from the lspci lines), its lspci, its
    PowerShell CIM answer on Windows, its Level Zero loader (None: not installed). Every vendor tool is absent."""
    from nbxtensor.backend import install_help, level_zero as ze
    root = tmp_path / "sys_bus_pci_devices"
    for bus, cls, vendor, device in _lspci_rows(lspci) if lspci else []:
        d = root / bus
        d.mkdir(parents=True)
        (d / "class").write_text(f"0x{klass or cls}00\n")
        (d / "vendor").write_text(f"0x{vendor}\n")
        (d / "device").write_text(f"0x{device}\n")
    (root / "0000:00:1f.0").mkdir(parents=True)                # an ISA bridge on the same bus: not a GPU
    (root / "0000:00:1f.0" / "class").write_text("0x060100\n")
    (root / "0000:00:1f.0" / "vendor").write_text("0x8086\n")
    (root / "0000:00:1f.0" / "device").write_text("0xa082\n")
    monkeypatch.setattr(install_help, "pci_display_devices",
                        functools.partial(install_help.pci_display_devices, str(root)))
    asked = []

    def run(cmd, *a, **k):
        asked.append(cmd[0])
        if cmd[0] in _VENDOR_TOOLS:
            raise FileNotFoundError(cmd[0])
        if cmd[0] == "lspci":
            return subprocess.CompletedProcess(cmd, 0, lspci, "")
        if cmd[0] == "powershell" and "Win32_VideoController" in cmd[-1]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(windows if len(windows) != 1 else windows[0]), "")
        raise AssertionError(f"detection asked {cmd!r}")
    monkeypatch.setattr(autodetect.subprocess, "run", run)
    monkeypatch.setattr(ze, "gpu_facts", lambda lib=None: level_zero)
    monkeypatch.setattr(autodetect.platform, "system", lambda: "Windows" if windows is not None else "Linux")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return asked


def _profile(monkeypatch, os_type="Linux", ram_mb=32 * 1024):
    cpu = {"model": "laptop CPU", "cores": 4, "threads": 8, "ram_mb": ram_mb, "architecture": "x86_64", "features": []}
    monkeypatch.setattr(autodetect, "_detect_cpu", lambda os_type: dict(cpu))
    monkeypatch.setattr(autodetect, "_measure_runtime_base_mb", lambda: {"compiled": 500, "triton": 90})
    monkeypatch.setattr(autodetect, "_measure_io_workers", lambda: 4)
    monkeypatch.setattr(autodetect, "_detect_system_vendor", lambda os_type: "custom")
    return autodetect.detect_hardware()


def _loaded(monkeypatch, tmp_path, data):
    from neurobrix.core.prism import loader
    monkeypatch.setattr(loader, "GENERATED_DIR", tmp_path)
    (tmp_path / "default-igpu.yml").write_text(yaml.safe_dump(data, sort_keys=False))
    return loader.load_profile("default-igpu")


def _ze(device_id, name, ram_mb=32 * 1024, integrated=True):
    """Level Zero's facts of one integrated GPU, by compute-runtime's formula (module docstring)."""
    return [{"pci_device_id": device_id, "name": name, "integrated": integrated,
             "memory_bytes": int(ram_mb * (1 << 20) * 0.94) - 450 * (1 << 20), "max_allocation_bytes": 4 * GIB}]


def test_a_tiger_lake_laptop_is_refused_by_name_and_plans_on_its_cpu(monkeypatch, tmp_path):
    asked = _machine(monkeypatch, tmp_path, lspci=TIGER_LAKE_DG1)
    profile = _profile(monkeypatch)
    assert profile["devices"] == [] and profile["summary"]["topology"] == "CPU-Only"
    refused = {d["pci_id"]: d for d in profile["refused_devices"]}
    tgl, dg1 = refused["8086:9a49"], refused["8086:4905"]
    assert tgl["model"] == "Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics]" and tgl["bus"] == "0000:00:02.0"
    assert (tgl["platform"], tgl["integrated"]) == ("Tiger Lake (Gen12, Xe-LP)", True)
    assert (dg1["platform"], dg1["integrated"]) == ("DG1 (Gen12, Xe-LP)", False)
    assert "arch_parser.c" in tgl["refused"] and "arch_parser.c" in dg1["refused"]      # the reason, with its source
    assert not {"xpu-smi", "sycl-ls"} & set(asked)                                       # no oneAPI tool is asked
    hw = _loaded(monkeypatch, tmp_path, profile)
    assert hw.devices == [] and [d["pci_id"] for d in hw.refused_devices] == ["8086:9a49", "8086:4905"]


def test_the_plan_of_a_machine_whose_gpu_is_refused_says_it_and_its_cpu_threads(monkeypatch, tmp_path):
    from neurobrix.core.prism.solver import ExecutionPlan, PrismSolver, explain_plan, plan_record
    _machine(monkeypatch, tmp_path, lspci=TIGER_LAKE_DG1)
    hw = _loaded(monkeypatch, tmp_path, _profile(monkeypatch))
    why = PrismSolver()._explain_choice("cpu_execution", 100.0, [(100.0, "cpu_execution", None, None)], hw)
    assert "4 compute threads" in why and "TigerLake-LP GT2" in why and "8086:9a49" in why
    plan = ExecutionPlan({}, "float32", 0.0, "cpu_execution", {}, "lazy", selection_reason=why,
                         refused_devices=list(hw.refused_devices))
    said = explain_plan(plan)
    assert "device refused  Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] (8086:9a49, Tiger Lake (Gen12, " \
           "Xe-LP), integrated): Intel's Triton backend" in said
    assert [d["pci_id"] for d in plan_record(plan)["refused_devices"]] == ["8086:9a49", "8086:4905"]


def test_a_meteor_lake_laptop_with_level_zero_is_driven_as_unified_memory(monkeypatch, tmp_path):
    _machine(monkeypatch, tmp_path, lspci=METEOR_LAKE, level_zero=_ze(0x7D55, "Intel(R) Arc(TM) Graphics"))
    profile = _profile(monkeypatch)
    (dev,) = profile["devices"]
    assert "refused_devices" not in profile
    assert (dev["brand"], dev["architecture"], dev["compute_capability"]) == ("intel", "mtl", "mtl")
    assert dev["unified_memory"] is True and dev["host_memory_mb"] == 32 * 1024
    assert dev["memory_mb"] == (int(32 * 1024 * (1 << 20) * 0.94) - 450 * (1 << 20)) >> 20     # the device's figure
    assert dev["max_allocation_mb"] == 4096 and dev["pci_id"] == "8086:7d55"
    assert dev["supports_dtypes"] == ["float32", "float16"]          # mtl.yml: no bf16 declared (Xe-LPG, no XMX)
    spec = _loaded(monkeypatch, tmp_path, profile).devices[0]
    assert spec.has_unified_memory and spec.get_device_string() == "xpu:0" and spec.max_allocation_mb == 4096


def test_a_chip_a_device_file_describes_is_refused_when_its_loader_is_missing(monkeypatch, tmp_path):
    _machine(monkeypatch, tmp_path, lspci=METEOR_LAKE, level_zero=None)
    profile = _profile(monkeypatch)
    (refused,) = profile["refused_devices"]
    assert profile["devices"] == [] and "Level Zero loader" in refused["refused"] and "doctor" in refused["refused"]
    assert refused["platform"] == "Meteor Lake (Xe-LPG)" and refused["integrated"] is True


def test_a_display_controller_of_class_0380_is_a_gpu_too(monkeypatch, tmp_path):
    """An Alder Lake-P iGPU may report "Display controller [0380]" rather than VGA (0300): the bus listing takes the
    whole display class, as the kernel's `INTEL_VGA_DEVICE` matches it (class 0x03, mask 0xff)."""
    _machine(monkeypatch, tmp_path, lspci=METEOR_LAKE, klass="0380", level_zero=_ze(0x7D55, "Intel(R) Arc(TM) Graphics"))
    assert [d["architecture"] for d in autodetect.detect_devices()[0]] == ["mtl"]


def test_an_amd_apu_without_rocm_is_refused_by_name(monkeypatch, tmp_path):
    _machine(monkeypatch, tmp_path, lspci=PHOENIX)
    profile = _profile(monkeypatch)
    (refused,) = profile["refused_devices"]
    assert profile["devices"] == [] and refused["pci_id"] == "1002:15bf" and refused["brand"] == "amd"
    assert refused["model"] == "Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1" and "amd runtime" in refused["refused"]


@pytest.mark.parametrize("listed, platform", [([UHD_730], "Alder Lake-S (Gen12, Xe-LP)"),
                                              ([IRIS_XE_WINDOWS], "Tiger Lake (Gen12, Xe-LP)")])
def test_a_windows_laptop_is_read_from_win32_videocontroller_and_refused_by_name(monkeypatch, tmp_path, listed, platform):
    asked = _machine(monkeypatch, tmp_path, windows=listed)
    devices, brand = autodetect._detect_gpus("Windows")
    (refused,) = devices
    assert brand == "none" and refused["platform"] == platform and refused["model"] == listed[0]["Name"]
    assert "arch_parser.c" in refused["refused"] and "wmic" not in asked


def test_a_windows_laptop_whose_chip_a_backend_drives_is_unified(monkeypatch, tmp_path):
    lunar = {"Name": "Intel(R) Arc(TM) 140V GPU (16GB)", "PNPDeviceID": "PCI\\VEN_8086&DEV_64A0&SUBSYS_00000000&REV_04"}
    _machine(monkeypatch, tmp_path, windows=[lunar], level_zero=_ze(0x64A0, lunar["Name"], ram_mb=16 * 1024))
    (dev,), brand = autodetect._detect_gpus("Windows")
    assert brand == "intel" and dev["architecture"] == "lnl" and dev["unified_memory"] is True


def test_windows_states_its_memory_through_cim_or_the_profile_is_refused(monkeypatch):
    """wmic is disabled by default from Windows 11 24H2: the RAM was then an invented 8192 MB. TotalVisibleMemorySize is
    in kilobytes (Microsoft, Win32_OperatingSystem)."""
    def run(cmd, *a, **k):
        if cmd[0] == "wmic":
            raise FileNotFoundError("wmic")
        if "TotalVisibleMemorySize" in cmd[-1]:
            return subprocess.CompletedProcess(cmd, 0, answer, "")
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(autodetect.subprocess, "run", run)
    answer = "16541928\n"
    assert autodetect._detect_cpu_windows()["ram_mb"] == 16541928 // 1024
    answer = ""
    with pytest.raises(RuntimeError, match="TotalVisibleMemorySize"):
        autodetect._detect_cpu_windows()


def test_a_unified_profile_without_its_largest_allocation_is_refused_whatever_its_vendor(monkeypatch, tmp_path):
    _machine(monkeypatch, tmp_path, lspci=METEOR_LAKE, level_zero=_ze(0x7D55, "Intel(R) Arc(TM) Graphics"))
    profile = _profile(monkeypatch)
    profile["devices"][0].pop("max_allocation_mb")
    with pytest.raises(ValueError, match="maxMemAllocSize"):
        _loaded(monkeypatch, tmp_path, profile)


def test_the_platform_table_and_the_device_files_agree():
    """Every PCI id an Intel device file lists is in the platform table, in a row that states no reason not to drive
    it; no row that states one has an id a device file lists. The two halves of a fact never contradict each other."""
    from nbxtensor.backend import device_files
    rows = autodetect._pci_platforms()[0x8086]
    for vendor, name, _path in device_files.names():
        if vendor != "intel":
            continue
        for i in device_files.describe(vendor, name)["pci_device_ids"]:
            row = rows.get(int(str(i), 0))
            assert row is not None and row["no_target"] is None, (name, hex(int(str(i), 0)))
    assert all(device_files.for_pci_id("intel", i) is None for i, r in rows.items() if r["no_target"])


def test_a_machine_whose_vendor_tool_answers_never_reads_the_bus(monkeypatch):
    """NVIDIA's path is untouched: nvidia-smi answers and the bus is never listed."""
    card = {"index": 0, "brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "memory_mb": 16384,
            "compute_capability": "7.0", "supports_dtypes": ["float32", "float16"], "architecture": "volta",
            "pcie_version": "3.0"}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(autodetect, "_parse_nvidia_smi", lambda: [dict(card)])
    monkeypatch.setattr(autodetect, "_listed_gpus", lambda: (_ for _ in ()).throw(AssertionError("bus read")))
    assert autodetect._detect_gpus("Linux") == ([card], "nvidia")


def test_a_profile_unified_only_through_its_architecture_is_asked_the_same():
    """An older hand-written Apple profile states no `unified_memory`: it is unified through its architecture
    (`DeviceSpec.has_unified_memory`, the table of structure.py). The loader's door asks the device's answer, not the
    key (the Mac's review, 2026-11-08 21:20)."""
    import copy
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile
    old = copy.deepcopy(APPLE_M4_PRO)
    old["devices"][0].pop("unified_memory", None)
    old["devices"][0].pop("max_allocation_mb", None)
    assert old["devices"][0]["architecture"] == "apple_silicon"
    with pytest.raises(ValueError, match="max_allocation_mb"):
        profile(old)


def test_an_intel_macs_integrated_gpu_is_refused_by_name_and_the_mac_plans_on_its_cpu(monkeypatch):
    """No real Intel-Mac `system_profiler SPDisplaysDataType -json` was found to cite: this record carries only the keys
    the parser reads (`sppci_model`, `spdisplays_vendor`), with the Intel UHD 630 name a 2018 MacBook Pro reports in
    the text form of the same command (NVIDIA developer forum, "Will Macbook Pro Radeon Pro 555X or Intel UHD Graphics
    630 work with CUDA Toolkit"). It was written as architecture "xe", which no vendor file has: the Mac was refused
    whole."""
    record = {"SPDisplaysDataType": [{"sppci_model": "Intel UHD Graphics 630", "spdisplays_vendor": "Intel"}]}
    monkeypatch.setattr(autodetect.subprocess, "run",
                        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, json.dumps(record), ""))
    devices, brand = autodetect._detect_gpus("Darwin")
    (refused,) = devices
    assert brand == "none" and refused["model"] == "Intel UHD Graphics 630" and "macOS" in refused["refused"]
    assert autodetect._driven(devices) == []
