"""On Windows the detection honours CUDA_VISIBLE_DEVICES, as it does on Linux (GitHub issue 6 asked for a documented
way to run on the CPU of a machine that has an NVIDIA card).

The Linux cascade filters and re-indexes the nvidia-smi devices by the variable (core/prism/autodetect.py
`_apply_visible_filter`); the Windows cascade returned nvidia-smi's whole list, so `CUDA_VISIBLE_DEVICES=-1` (the
spelling cmd.exe and PowerShell can set: both delete a variable set to an empty value) planned on cuda:0 while the
CUDA runtime saw no device. What these tests would do without the filter: the card is still detected — RED."""
import subprocess

import pytest

from neurobrix.core.prism import autodetect

SMI = ("0, NVIDIA GeForce RTX 4070, 12282, 8.9, 00000000:01:00.0\n"
       "1, NVIDIA GeForce RTX 3060, 12288, 8.6, 00000000:02:00.0\n")


@pytest.fixture(params=["Windows", "Linux"])
def windows_with_two_cards(monkeypatch, request):
    """Two cards as nvidia-smi lists them, on Windows and (the reference behaviour, and the README's command) Linux."""
    monkeypatch.setattr(autodetect.platform, "system", lambda: request.param)
    monkeypatch.setattr(autodetect.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=SMI, stderr=""))
    monkeypatch.setattr(autodetect, "_detect_pcie_version_sysfs", lambda bus: "4.0")


def test_minus_one_hides_every_card_and_the_machine_plans_on_its_cpu(windows_with_two_cards, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    devices, _ = autodetect.detect_devices()
    assert devices == []
    assert autodetect._visible_set_tag() == "cpu"


def test_one_listed_card_is_the_only_one_and_is_renumbered_from_zero(windows_with_two_cards, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    devices, brand = autodetect.detect_devices()
    assert brand == "nvidia"
    assert [(d["index"], d["model"], d["architecture"]) for d in devices] == [(0, "NVIDIA GeForce RTX 3060",
                                                                               "ampere_consumer")]


def test_with_the_variable_unset_every_card_is_detected(windows_with_two_cards, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    devices, _ = autodetect.detect_devices()
    assert [d["architecture"] for d in devices] == ["ada", "ampere_consumer"]
