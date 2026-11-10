"""The experimental Intel backend's policy (core/prism/intel_refusal.py; the owner 2026-11-10 13:09, the Dell's review
16:0x): a chip whose Level Zero driver is present is tried; a Level Zero result that says the chip cannot do it is SAID
by name with its cause and the experimental label, RECORDED for THAT device under the driver and nbxtensor versions,
and `neurobrix run` starts again ONCE in a fresh process that plans without it. Never said falsely: no restart with a
given profile (--hardware), with no device to name, for a result that is not a verdict on the chip, or in the process
started again. Another driver or nbxtensor version (or a loader gone) tries the chip again. Card-less: nbxtensor's
typed IntelDeviceError, stand-in versions, the exec captured.
What this test would do if the code were wrong: a restart that plans the same chip again, a sentence printed with
nothing recorded, every Intel device condemned for one, a transient error recorded, a torn record breaking every
start, the wrong profile deleted (red)."""
import json
import os
import subprocess

import pytest

from nbxtensor.backend.level_zero import IntelDeviceError
from neurobrix.core.prism import autodetect, intel_refusal as R

BUILD, LOST, ARG = 0x70000004, 0x70000001, 0x78000004
PROFILE = ("devices:\n- {index: 0, brand: intel, model: Iris Xe, pci_id: '8086:9a49'}\n"
           "- {index: 1, brand: intel, model: Arc A770, pci_id: '8086:56a0'}\n")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    versions = {"driver": "1.6.31294", "nbxtensor": "0.1.0"}
    monkeypatch.setattr(R, "_versions", lambda: dict(versions) if versions else None)
    monkeypatch.delenv(R.ENV, raising=False)
    (tmp_path / "default-abc.yml").write_text(PROFILE)
    (tmp_path / "default-cpu.yml").write_text("devices: []\n")
    (tmp_path / "default.yml").write_text(PROFILE)
    return tmp_path, versions


@pytest.fixture
def execs(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    return calls


def _failure(rc=BUILD, device=1, log="error: unsupported"):
    try:
        try:
            raise IntelDeviceError("NeuroBrix launcher: zeModuleCreate failed", "zeModuleCreate", rc, log, device=device)
        except IntelDeviceError as e:
            raise RuntimeError("Failed at op aten.add::3") from e
    except RuntimeError as outer:
        return outer


ARGV = ["-m", "neurobrix", "run", "--model", "m"]


def test_the_refusal_is_said_recorded_for_its_device_and_the_run_started_again_once(home, execs, capsys):
    tmp, _ = home
    R.handle(_failure(), "default-abc", False, ARGV)
    said = capsys.readouterr().err.strip()
    assert "Arc A770 (8086:56a0)" in said and "Iris Xe" not in said                      # the device the error names
    assert "zeModuleCreate: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in said and "error: unsupported" in said
    assert R.LABEL in said and "running again without it" in said and R.REPORT_URL in said
    (exe, argv, env), = execs
    assert argv[1:] == ARGV and env[R.ENV] == said
    assert not (tmp / "default-abc.yml").exists() and not (tmp / "default.yml").exists()  # the profile that planned it
    assert (tmp / "default-cpu.yml").exists()                                             # another one is not touched
    assert "zeModuleCreate" in R.recorded("8086:56a0") and R.recorded("8086:9a49") is None
    assert json.loads((tmp / "intel_runtime_refusals" / "8086_56a0.json").read_text())["driver"] == "1.6.31294"


@pytest.mark.parametrize("case", ["given profile", "no profile yet", "device not planned", "none of several named",
                                  "transient result", "our own argument", "already started again", "not Intel's"])
def test_nothing_is_restarted_or_recorded_when_it_would_be_said_falsely(home, execs, capsys, monkeypatch, case):
    tmp, _ = home
    args = {"given profile": (_failure(), "default-abc", True),
            "no profile yet": (_failure(), None, False),
            "device not planned": (_failure(device=7), "default-abc", False),
            "none of several named": (_failure(device=None), "default-abc", False),
            "transient result": (_failure(rc=LOST), "default-abc", False),
            "our own argument": (_failure(rc=ARG), "default-abc", False),
            "already started again": (_failure(), "default-abc", False),
            "not Intel's": (RuntimeError("CUDA out of memory"), "default-abc", False)}[case]
    if case == "already started again":
        monkeypatch.setenv(R.ENV, "Intel GPU ... refused")
    assert R.handle(*args, ARGV) is None
    err = capsys.readouterr().err
    assert not execs and not (tmp / "intel_runtime_refusals").exists() and (tmp / "default-abc.yml").exists()
    if case == "given profile":                      # said, with what to do — and no "running again"
        assert "8086:56a0" in err and "--hardware" in err and "running again" not in err
    else:
        assert err == ""


def test_on_windows_the_child_s_status_is_the_run_s(home, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "platform", "win32")
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, env: seen.append((cmd, env)) or subprocess.CompletedProcess(cmd, 3))
    with pytest.raises(SystemExit) as e:
        R.handle(_failure(), "default-abc", False, ARGV)
    assert e.value.code == 3 and seen[0][0][1:] == ARGV and R.ENV in seen[0][1]


def test_the_process_started_again_says_why_first(monkeypatch, capsys):
    monkeypatch.setenv(R.ENV, "Intel GPU Arc refused at run time: ...")
    R.announce()
    assert capsys.readouterr().err.startswith("Running again — Intel GPU Arc refused")


def test_another_driver_or_nbxtensor_version_or_a_loader_gone_tries_the_chip_again(home):
    tmp, versions = home
    dev = [{"pci_id": "8086:56a0"}]
    R.record(dev, "refused at run time: zeModuleCreate: X", "default-abc")
    assert R.recorded("8086:56a0") and R.stale() is False
    versions["driver"] = "1.6.33578"
    assert R.recorded("8086:56a0") is None and R.stale() is True and R.stale() is False
    versions["driver"] = "1.6.31294"
    R.record(dev, "refused at run time: zeModuleCreate: X", None)
    versions["nbxtensor"] = "0.1.1"
    assert R.stale() is True
    versions["nbxtensor"] = "0.1.0"
    R.record(dev, "refused at run time: zeModuleCreate: X", None)
    versions.clear()                                                   # the Level Zero loader was uninstalled
    assert R.recorded("8086:56a0") is None and R.stale() is True


def test_a_torn_record_is_said_and_removed_never_fatal(home, caplog):
    tmp, _ = home
    R.record([{"pci_id": "8086:56a0"}], "refused at run time: zeModuleCreate: X", None)
    (tmp / "intel_runtime_refusals" / "8086_9a49.json").write_text('{"pci_id": "8086:9a')
    assert set(R._read()) == {"8086:56a0"} and not (tmp / "intel_runtime_refusals" / "8086_9a49.json").exists()


def test_nothing_recorded_costs_nothing(home, monkeypatch):
    monkeypatch.setattr(R, "_versions", lambda: (_ for _ in ()).throw(AssertionError("Level Zero asked for no reason")))
    assert R.stale() is False and R.recorded("8086:56a0") is None


def test_detection_lists_a_recorded_chip_as_a_refused_device_with_its_reason(home, monkeypatch):
    """After a run-time refusal the machine is detected again: the chip Level Zero reports and a device file describes
    is no longer a planned device but a REFUSED one carrying the recorded reason (Prism then plans without it and
    names it in the plan); with nothing recorded it is planned as before."""
    from nbxtensor.backend import device_files, level_zero
    gpu = {"vendor_id": 0x8086, "device_id": 0x7D55, "bus": "0000:00:02.0", "name": "Arc Graphics"}
    monkeypatch.setattr(autodetect, "_listed_gpus", lambda: [gpu])
    monkeypatch.setattr(level_zero, "gpu_facts", lambda lib=None: [{
        "pci_device_id": 0x7D55, "name": "Intel(R) Arc(TM) Graphics", "integrated": True,
        "memory_bytes": 8 << 30, "max_allocation_bytes": 4 << 30}])
    assert device_files.for_pci_id("intel", 0x7D55) is not None, "0x7d55 is Meteor Lake's (devices/intel/mtl.yml)"
    planned = autodetect._parse_xpu()
    assert len(planned) == 1 and "refused" not in planned[0] and planned[0]["pci_id"] == "8086:7d55"
    R.record([{"pci_id": "8086:7d55"}], "refused at run time: zeModuleCreate: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE", None)
    refused = autodetect._parse_xpu()
    assert len(refused) == 1 and "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in refused[0]["refused"]
    assert "another version of either tries it again" in refused[0]["refused"]


def test_doctor_says_the_label_and_what_was_refused_without_asking_level_zero(home, monkeypatch):
    tmp, _ = home
    lines = R.doctor_lines()
    assert len(lines) == 1 and "8086:56a0" in lines[0] and "8086:9a49" in lines[0] and R.LABEL in lines[0]
    R.record([{"pci_id": "8086:56a0"}], "refused at run time: zeModuleCreate: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE",
             "default-abc")
    monkeypatch.setattr(R, "_versions", lambda: (_ for _ in ()).throw(AssertionError("Level Zero asked")))
    lines = R.doctor_lines()                                   # the profiles are gone: only the record speaks
    assert len(lines) == 1 and "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in lines[0] and "1.6.31294" in lines[0]


def test_the_report_link_names_a_template_that_exists():
    from pathlib import Path
    name = R.REPORT_URL.split("template=")[1]
    assert (Path(__file__).resolve().parents[3] / ".github" / "ISSUE_TEMPLATE" / name).is_file()
