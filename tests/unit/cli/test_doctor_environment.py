"""`neurobrix doctor` must not give a false all-clear.

Before 2026-09-03 it checked only whether the `neurobrix` script was on
PATH and printed "No action needed" whenever it was — including on a
machine where PyTorch could not see the GPU at all. That is the worst
thing a diagnostic can do: the user is told everything is fine, meets an
obscure failure on their first real run, and leaves without reporting it.

The fault it has to catch is the common one, not an exotic one:
`pip install neurobrix` resolves `torch` from the default index, which
ships whatever CUDA build is current. CUDA 13 requires a recent driver and
dropped Volta (sm_70) outright, so a clean install on a V100 — the very
hardware the project's own benchmarks run on — yields a torch reporting no
CUDA at all.
"""

from __future__ import annotations

import types

import pytest

from neurobrix.cli.commands import doctor


class _FakeCuda:
    def __init__(self, available, devices=(), arch_list=()):
        self._available = available
        self._devices = list(devices)
        self._arch_list = list(arch_list)

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._devices)

    def get_arch_list(self):
        return self._arch_list

    def get_device_name(self, i):
        return self._devices[i][0]

    def get_device_capability(self, i):
        return self._devices[i][1]


def _fake_torch(monkeypatch, *, available, devices=(), arch_list=(), cuda="12.1"):
    fake = types.SimpleNamespace(
        __version__="2.5.1+cu121",
        version=types.SimpleNamespace(cuda=cuda, hip=None),
        cuda=_FakeCuda(available, devices, arch_list),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    return fake


def test_healthy_environment_reports_no_problem(monkeypatch, capsys):
    _fake_torch(monkeypatch, available=True,
                devices=[("Tesla V100-SXM2-32GB", (7, 0))],
                arch_list=["sm_70", "sm_80"])
    problems = doctor.check_compute_environment()
    assert problems == []
    assert "ok" in capsys.readouterr().out


def test_gpu_present_but_torch_blind_is_a_blocking_problem(monkeypatch, capsys):
    """The exact CUDA-13-on-Volta case: the driver sees four V100s, torch
    reports no CUDA. This must be loud, and must name the fix."""
    _fake_torch(monkeypatch, available=False, cuda="13.0")
    monkeypatch.setattr(doctor, "_nvidia_smi_gpus",
                        lambda: [("Tesla V100-SXM2-32GB", "7.0")])
    monkeypatch.setattr(doctor, "_driver_version", lambda: "535.309.01")

    problems = doctor.check_compute_environment()
    assert len(problems) == 1
    text = problems[0]
    assert "PyTorch reports no CUDA" in text
    assert "index-url" in text, "a diagnosis without the fix command is half a diagnosis"
    assert "cu121" in text, "Volta must be sent to a CUDA build that still has sm_70"


def test_gpu_architecture_missing_from_the_wheel_is_caught(monkeypatch):
    """CUDA is available, but this torch has no kernels for the card —
    a failure that otherwise appears only at the first launch."""
    _fake_torch(monkeypatch, available=True,
                devices=[("Tesla V100-SXM2-32GB", (7, 0))],
                arch_list=["sm_75", "sm_80", "sm_90"])
    problems = doctor.check_compute_environment()
    assert len(problems) == 1
    assert "sm_70" in problems[0]
    assert "does not include" in problems[0]


def test_no_gpu_at_all_is_reported_but_is_not_a_torch_bug(monkeypatch):
    """No GPU and no driver is a different situation from a mismatched
    build, and must not be reported as one."""
    _fake_torch(monkeypatch, available=False)
    monkeypatch.setattr(doctor, "_nvidia_smi_gpus", lambda: [])
    monkeypatch.setattr(doctor, "_driver_version", lambda: None)
    assert doctor.check_compute_environment() == []


def test_missing_torch_is_reported(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    problems = doctor.check_compute_environment()
    assert problems and "PyTorch is not installed" in problems[0]


@pytest.mark.parametrize("cap,expected", [("7.0", "cu121"), ("8.6", "cu124"), (None, "cu124")])
def test_install_hint_matches_the_gpu_generation(cap, expected):
    assert expected in doctor._torch_install_hint(cap)
