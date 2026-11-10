"""A laptop with an NVIDIA card and an integrated GPU: the NVIDIA card is driven and planned, and the integrated GPU is
NAMED in the profile, refused with its reason (reste_apres_064 point 2; the owner's next useful version, 2026-11-09
18:58). Before: the first detector that drove a card ended the cascade, so the integrated GPU was not named at all (the
known limit of 0.6.3 and 0.6.4).

Real outputs, cited:
- an Acer laptop, `lspci -nn`: "TigerLake-H GT1 [UHD Graphics] [8086:9a60]" at 00:02.0 and "GA107M [GeForce RTX 3050 Ti
  Mobile] [10de:25a0]" at 01:00.0, https://forums.opensuse.org/t/fails-to-start-graphical-mode-after-installing-geforce-rtx-3050-ti-driver-on-opensuse-linux/164686
- an AMD Phoenix APU, `lspci -nn` "[1002:15bf]", the source cited in test_an_integrated_gpu_is_detected_from_what_the_os_ships.py.

What this test would do if the code were wrong: the integrated GPU missing from the profile (red), planned beside the
NVIDIA card (red: not refused), refused without its reason or platform (red), the NVIDIA card no longer driven (red), on
either OS's cascade (red)."""
from __future__ import annotations

import pytest

from neurobrix.core.prism import autodetect as A

RTX = {"model": "NVIDIA GeForce RTX 3050 Ti Laptop GPU", "brand": "nvidia", "memory_mb": 4096, "index": 0}
TIGER_LAKE = {"vendor_id": 0x8086, "device_id": 0x9a60, "bus": "0000:00:02.0",
              "name": "Intel Corporation TigerLake-H GT1 [UHD Graphics]"}
RTX_PCI = {"vendor_id": 0x10de, "device_id": 0x25a0, "bus": "0000:01:00.0",
           "name": "NVIDIA Corporation GA107M [GeForce RTX 3050 Ti Mobile]"}
PHOENIX = {"vendor_id": 0x1002, "device_id": 0x15bf, "bus": "0000:c4:00.0", "name": "Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1"}


@pytest.fixture
def laptop(monkeypatch):
    def use(listed):
        monkeypatch.setattr(A, "_listed_gpus", lambda: listed)
        monkeypatch.setattr(A, "_parse_nvidia_smi", lambda: [dict(RTX)])
        monkeypatch.setattr(A, "_visible_device_filter", lambda: None)
        for name in ("_parse_rocm", "_parse_tenstorrent", "_parse_moore_threads", "_parse_biren", "_parse_iluvatar",
                     "_parse_cambricon"):
            monkeypatch.setattr(A, name, lambda: [])
    return use


def _profile_gpus(os_type):
    """What the profile's writer holds (detect_hardware): the cascade's devices, and beside driven cards the other GPUs."""
    found, brand = A._detect_gpus(os_type)
    return (found + A._beside(brand) if A._driven(found) else found), brand


@pytest.mark.parametrize("os_type", ["Linux", "Windows"])
def test_an_intel_integrated_gpu_is_named_beside_the_nvidia_card(laptop, os_type):
    laptop([TIGER_LAKE, RTX_PCI])
    devices, brand = _profile_gpus(os_type)
    assert brand == "nvidia" and A._driven(devices) == [RTX]
    (igpu,) = A._refusals(devices)
    assert igpu["pci_id"] == "8086:9a60" and igpu["integrated"] is True and igpu["platform"].startswith("Tiger Lake")
    assert "Level Zero loader" in igpu["refused"]          # Gen12 is tried since 2026-11-10; here its runtime is absent


@pytest.mark.parametrize("os_type", ["Linux", "Windows"])
def test_an_amd_apu_beside_an_nvidia_card_is_named_and_not_planned(laptop, os_type):
    laptop([PHOENIX, RTX_PCI])
    devices, brand = _profile_gpus(os_type)
    assert A._driven(devices) == [RTX]
    (apu,) = A._refusals(devices)
    assert apu["brand"] == "amd" and "one vendor's backend" in apu["refused"]


def test_a_machine_of_one_vendor_names_nothing_beside(laptop):
    laptop([RTX_PCI])
    assert _profile_gpus("Linux") == ([RTX], "nvidia")


def test_the_profile_writer_asks_beside_the_driven_cards_and_every_launch_does_not(monkeypatch, laptop):
    """detect_hardware (the profile's writer) names the other GPUs; detect_devices (every launch) never reads the bus."""
    import inspect
    laptop([TIGER_LAKE, RTX_PCI])
    assert "_beside(brand)" in inspect.getsource(A.detect_hardware)
    monkeypatch.setattr(A, "_listed_gpus", lambda: (_ for _ in ()).throw(AssertionError("bus read on every launch")))
    assert A.detect_devices() == ([RTX], "nvidia")
