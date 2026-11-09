"""A hybrid CPU's performance and efficiency cores enter the profile on Linux and Windows as they did on macOS
(reste_apres_064 point 3; the owner's rule 1: a profile fact read nowhere is a defect). Linux: the kernel's hybrid PMU
lists (`/sys/devices/cpu_core/cpus`, `cpu_atom/cpus`), counted in physical cores; Windows: each core's EfficiencyClass.

What this test would do if the code were wrong: a 6P+8E laptop read as 12 or 14 cores of one class (red: logical CPUs
counted, or the lists ignored), a part whose cpu_atom list is empty called hybrid (red), a machine without the lists
given core classes (red), the Windows records misread (red), the profile completion skipping Linux or Windows (red)."""
from __future__ import annotations

import struct

from neurobrix.core.prism import autodetect as A


def _sysfs(root, core_cpus, atom_cpus, topology):
    for pmu, cpus in (("cpu_core", core_cpus), ("cpu_atom", atom_cpus)):
        if cpus is not None:
            (root / "devices" / pmu).mkdir(parents=True)
            (root / "devices" / pmu / "cpus").write_text(cpus + "\n")
    for n, (pkg, core) in topology.items():
        t = root / "devices" / "system" / "cpu" / f"cpu{n}" / "topology"
        t.mkdir(parents=True)
        (t / "physical_package_id").write_text(f"{pkg}\n")
        (t / "core_id").write_text(f"{core}\n")
    return root


def _laptop_topology():
    topo = {n: (0, n // 2) for n in range(12)}                 # 6 P-cores, two threads each: cpu0-11
    topo.update({12 + i: (0, 8 + i) for i in range(8)})       # 8 E-cores, one thread each: cpu12-19
    return topo


def test_an_intel_hybrid_laptop_on_linux(tmp_path):
    root = _sysfs(tmp_path, "0-11", "12-19", _laptop_topology())
    assert A._hybrid_cores_linux(root) == {"performance_cores": 6, "efficiency_cores": 8}


def test_a_part_without_e_cores_exposes_an_empty_atom_list(tmp_path):
    root = _sysfs(tmp_path, "0-11", "", {n: (0, n // 2) for n in range(12)})
    assert A._hybrid_cores_linux(root) == {}


def test_a_cpu_without_the_hybrid_lists(tmp_path):
    assert A._hybrid_cores_linux(_sysfs(tmp_path, None, None, {0: (0, 0)})) == {}


def test_the_kernel_cpu_list_format():
    assert A._cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11] and A._cpu_list("\n") == []


def _core_record(efficiency_class):
    body = bytes([0, efficiency_class]) + bytes(20) + struct.pack("<H", 1) + bytes(16)    # Flags, Class, ..., GroupMask
    return struct.pack("<II", 0, 8 + len(body)) + body


def test_windows_efficiency_classes():
    cache = struct.pack("<II", 2, 32) + bytes(24)            # a RelationCache record between the cores: skipped
    buf = b"".join(_core_record(1) for _ in range(6)) + cache + b"".join(_core_record(0) for _ in range(8))
    classes = A._core_efficiency_classes(buf)
    assert classes == [1] * 6 + [0] * 8


def test_the_profile_completion_asks_every_os(monkeypatch):
    asked = []
    monkeypatch.setattr(A, "_hybrid_cores_linux", lambda root: asked.append("Linux") or {})
    monkeypatch.setattr(A, "_hybrid_cores_windows", lambda: asked.append("Windows") or {})
    A._hybrid_cores("Linux"), A._hybrid_cores("Windows")
    assert asked == ["Linux", "Windows"]
