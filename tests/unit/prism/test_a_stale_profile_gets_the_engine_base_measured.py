"""A profile written by an older detector is completed with what the current one measures.

The rack's auto profiles predated `cpu.runtime_base_mb` and `get_or_create_default_profile` returned
them as they were, forever: every plan on them priced no engine base (2026-10-10). Injection: drop the
`_complete_measured_fields(path)` call in the visible-set branch -> the first test goes red.
"""
import yaml

from neurobrix.core.prism import autodetect


def _stale(tmp_path, monkeypatch, cpu):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "_visible_set_tag", lambda: "abcd1234")
    calls = []
    monkeypatch.setattr(autodetect, "_measure_runtime_base_mb",
                        lambda: calls.append(1) or {"compiled": 803, "triton": 206})
    monkeypatch.setattr(autodetect, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("re-detected")))
    path = tmp_path / "default-abcd1234.yml"
    path.write_text(yaml.safe_dump({"id": "auto-x", "cpu": cpu, "devices": [{"model": "kept"}]}))
    return path, calls


def test_a_profile_without_the_base_gets_it_measured_and_keeps_the_rest(tmp_path, monkeypatch):
    path, calls = _stale(tmp_path, monkeypatch, {"model": "xeon", "cores": 40})
    assert autodetect.get_or_create_default_profile() == "default-abcd1234"
    data = yaml.safe_load(path.read_text())
    assert data["cpu"] == {"model": "xeon", "cores": 40, "runtime_base_mb": {"compiled": 803, "triton": 206}}
    assert data["devices"] == [{"model": "kept"}] and calls == [1]


def test_a_profile_that_has_the_base_is_not_measured_again(tmp_path, monkeypatch):
    path, calls = _stale(tmp_path, monkeypatch, {"model": "xeon", "runtime_base_mb": {"triton": 1}})
    before = path.read_text()
    autodetect.get_or_create_default_profile()
    assert calls == [] and path.read_text() == before
