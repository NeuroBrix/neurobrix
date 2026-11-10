"""The experimental Intel backend's policy (core/prism/intel_refusal.py; the owner 2026-11-10 13:09): a chip whose
Level Zero driver is present is tried; its first Level Zero error is SAID by name with its cause and the experimental
label, RECORDED under the driver and nbxtensor versions, and the request starts again ONCE in a fresh process that
plans on the CPU; a refusal in that process is the run's failure; another driver or nbxtensor version tries the chip
again. Card-less: the error is nbxtensor's typed IntelDeviceError, the versions and the exec are stand-ins.
What this test would do if the code were wrong: a refusal swallowed or retried in a loop, an untyped error treated as
Intel's, a record that never expires or that survives a new driver, a profile left planning the refused chip (red)."""
import json
import os

import pytest

from nbxtensor.backend.level_zero import IntelDeviceError
from neurobrix.core.prism import autodetect, intel_refusal as R

DEV = [{"brand": "intel", "model": "Intel(R) Arc(TM) Graphics", "pci_id": "8086:7d55"}]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "_visible_set_tag", lambda: "abc")
    versions = {"driver": "1.6.31294", "nbxtensor": "0.1.1"}
    monkeypatch.setattr(R, "_versions", lambda: dict(versions))
    monkeypatch.delenv(R.ENV, raising=False)
    (tmp_path / "default-abc.yml").write_text("devices:\n- {brand: intel, model: Arc, pci_id: '8086:7d55'}\n")
    (tmp_path / "default.yml").write_text("devices: []\n")
    return tmp_path, versions


def _failure():
    try:
        try:
            raise IntelDeviceError("NeuroBrix launcher: zeModuleCreate failed", "zeModuleCreate", 0x70000004, "error: unsupported")
        except IntelDeviceError as e:
            raise RuntimeError("Failed at op aten.add::3") from e
    except RuntimeError as outer:
        return outer


@pytest.fixture
def execs(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    return calls


def test_the_refusal_is_said_recorded_and_the_request_started_again_once(home, execs, capsys):
    tmp, _ = home
    calls = execs
    assert R.handle(_failure(), DEV, ["-m", "neurobrix", "run", "--model", "m"])
    said = capsys.readouterr().err
    assert "8086:7d55" in said and "zeModuleCreate" in said and "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in said
    assert "error: unsupported" in said and R.LABEL in said and "running on the CPU" in said and R.REPORT_URL in said
    (exe, argv, env), = calls
    assert argv[1:] == ["-m", "neurobrix", "run", "--model", "m"] and env[R.ENV] == said.strip()
    assert not (tmp / "default-abc.yml").exists() and not (tmp / "default.yml").exists()      # detected again
    assert "zeModuleCreate" in R.recorded("8086:7d55") and R.recorded("8086:0000") is None
    assert json.loads((tmp / "intel_runtime_refusals.json").read_text())["8086:7d55"]["driver"] == "1.6.31294"


def test_a_refusal_in_the_process_started_again_is_the_run_s_failure(home, execs, monkeypatch, capsys):
    monkeypatch.setenv(R.ENV, "Intel GPU ... refused")
    assert R.handle(_failure(), DEV, ["x"]) is False and not execs
    R.announce()
    assert capsys.readouterr().err.startswith("Running again on the CPU — Intel GPU")


def test_any_other_error_is_not_intel_s(home, execs):
    assert R.handle(RuntimeError("CUDA out of memory"), DEV, ["x"]) is False
    assert not execs and not (home[0] / "intel_runtime_refusals.json").exists()


def test_another_driver_or_nbxtensor_version_tries_the_chip_again(home):
    tmp, versions = home
    R.record(DEV, "refused at run time: zeModuleCreate: X")
    assert R.recorded("8086:7d55") and R.stale() is False
    versions["driver"] = "1.6.33578"
    assert R.recorded("8086:7d55") is None
    assert R.stale() is True and not (tmp / "intel_runtime_refusals.json").exists() and R.stale() is False
    R.record(DEV, "refused at run time: zeModuleCreate: X")
    versions["nbxtensor"] = "0.1.0"
    assert R.stale() is True


def test_nothing_recorded_costs_nothing(home, monkeypatch):
    monkeypatch.setattr(R, "_versions", lambda: (_ for _ in ()).throw(AssertionError("Level Zero asked for no reason")))
    assert R.stale() is False and R.recorded("8086:7d55") is None


def test_detection_lists_a_recorded_chip_as_a_refused_device_with_its_reason(home, monkeypatch):
    """After a run-time refusal the machine is detected again: the chip Level Zero reports and a device file describes
    is no longer a planned device but a REFUSED one carrying the recorded reason (Prism then plans on the CPU and
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
    R.record([{"pci_id": "8086:7d55"}], "refused at run time: zeModuleCreate: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE")
    refused = autodetect._parse_xpu()
    assert len(refused) == 1 and "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in refused[0]["refused"]
    assert "another version of either tries it again" in refused[0]["refused"]
