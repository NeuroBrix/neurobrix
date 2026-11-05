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
    assert "cu126" in text, "Volta must be sent to the build measured on sm_70"


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


def test_missing_torch_is_no_problem_where_the_triton_engine_runs(monkeypatch, capsys):
    """The Triton engine runs on nbxtensor without PyTorch: a clean `pip install neurobrix` has no torch, and doctor
    said "PyTorch is not installed: pip install neurobrix" — a blocking problem and a command that installs no torch
    (the stable's install test, 2026-11-05). Red if a missing torch is a problem while the Triton engine is ready."""
    monkeypatch.setattr(doctor, "_torch_present", lambda: False)
    monkeypatch.setattr(doctor, "_triton_engine", lambda: (True, "ready on cuda"))
    monkeypatch.setattr(doctor, "_nvidia_smi_gpus", lambda: [("Tesla V100-SXM2-16GB", "7.0")])
    assert doctor.check_compute_environment() == []
    out = capsys.readouterr().out
    assert "Triton engine:  ready on cuda" in out and "PyTorch engine: not installed" in out and "neurobrix[pytorch]" in out


def test_no_engine_that_can_run_is_a_blocking_problem(monkeypatch):
    monkeypatch.setattr(doctor, "_torch_present", lambda: False)
    monkeypatch.setattr(doctor, "_triton_engine", lambda: (False, "no GPU runtime"))
    monkeypatch.setattr(doctor, "_nvidia_smi_gpus", lambda: [])
    problems = doctor.check_compute_environment()
    assert len(problems) == 1 and "Neither engine can run" in problems[0] and "no GPU runtime" in problems[0]


def test_a_gpu_without_its_runtime_library_is_named(monkeypatch):
    """The driver sees a card but nbxtensor opens no CUDA runtime: on a clean machine the engine read the card as
    CPU-only and told the user to build triton-cpu (2026-11-05)."""
    import nbxtensor.backend.runtime as RT
    monkeypatch.setattr(RT, "gpu_backend", lambda: (_ for _ in ()).throw(RuntimeError("No GPU runtime found")))
    monkeypatch.setattr(doctor, "_nvidia_smi_gpus", lambda: [("Tesla V100-SXM2-16GB", "7.0")])
    ready, says = doctor._triton_engine()
    assert not ready and "the driver sees a GPU" in says and "No GPU runtime found" in says


def test_volta_is_sent_to_the_served_stack():
    """The V100 build is the one the engine is served on (torch 2.14.0+cu126).
    Until 2026-10-12 doctor named cu121, a literal in the code that had
    outlived the stack it described."""
    hint = doctor._torch_install_hint("7.0")
    assert hint.endswith("--index-url https://download.pytorch.org/whl/cu126"), hint


def test_the_hint_is_read_from_the_wheel_table(monkeypatch):
    """Change the table and the hint follows: the build is data, not code."""
    table = doctor._torch_wheels()
    table["builds"]["volta"]["build"] = "cu999"
    monkeypatch.setattr(doctor, "_torch_wheels", lambda: table)
    assert doctor._torch_install_hint("7.0").endswith("whl/cu999")


def test_every_table_entry_names_its_source():
    table = doctor._torch_wheels()
    assert table["builds"], "an empty table would turn every hint into the selector"
    for arch, entry in table["builds"].items():
        assert entry.get("source", "").strip(), f"{arch}: a build without its source"
    assert table["cpu_only"]["source"].strip()


@pytest.mark.parametrize("cap", ["8.6", "9.0", None])
def test_an_unmeasured_gpu_is_told_so_not_given_a_guess(cap):
    hint = doctor._torch_install_hint(cap)
    assert "no PyTorch build has been measured" in hint
    assert "pytorch.org/get-started/locally" in hint
    assert "<cuda-build>" in hint


def test_doctor_carries_no_build_literal():
    """A CUDA build or a wheel URL written in doctor.py is the defect this
    table replaced."""
    import re
    from pathlib import Path
    src = Path(doctor.__file__).read_text()
    found = re.findall(r"\bcu\d{3}\b|download\.pytorch\.org", src)
    assert not found, f"doctor.py carries {found}; it belongs in config/torch_wheels.yml"
