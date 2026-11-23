"""A profile written by an older detector is completed with what the current one measures.

The rack's auto profiles predated `cpu.runtime_base_mb` and `get_or_create_default_profile` returned
them as they were, forever: every plan on them priced no engine base (2026-10-10). Injection: drop the
`_complete_measured_fields(path)` call in the visible-set branch -> the first test goes red.
"""
import yaml

from neurobrix.pytorch.prism import autodetect


def _stale(tmp_path, monkeypatch, cpu):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "_visible_set_tag", lambda: "abcd1234")
    calls = []
    monkeypatch.setattr(autodetect, "_measure_runtime_base_mb",
                        lambda *engine: calls.append(1) or 803)
    monkeypatch.setattr(autodetect, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("re-detected")))
    path = tmp_path / "default-abcd1234.yml"
    path.write_text(yaml.safe_dump({"id": "auto-x", "cpu": cpu, "devices": [{"model": "kept"}]}))
    return path, calls


def test_a_profile_without_the_base_waits_for_its_engine_s_first_launch_and_keeps_the_rest(tmp_path, monkeypatch):
    """Since the ruling of 22/11/2026 17:43 (line 3) the base is measured per engine at that engine's first launch
    (`ensure_measured_facts`), never for both when the profile is resolved: completion measures none of it."""
    path, calls = _stale(tmp_path, monkeypatch, {"model": "xeon", "cores": 40})
    assert autodetect.get_or_create_default_profile() == "default-abcd1234"
    data = yaml.safe_load(path.read_text())
    assert "runtime_base_mb" not in data["cpu"] and data["cpu"]["model"] == "xeon" and data["cpu"]["cores"] == 40
    assert data["devices"] == [{"model": "kept"}] and calls == []


def test_a_profile_that_has_the_base_is_not_measured_again(tmp_path, monkeypatch):
    path, calls = _stale(tmp_path, monkeypatch, {"model": "xeon", "runtime_base_mb": {"triton": 1}, "io_workers": 2})
    before = path.read_text()
    autodetect.get_or_create_default_profile()
    assert calls == [] and path.read_text() == before


def test_a_profile_without_a_read_count_is_not_measured_for_one(tmp_path, monkeypatch):
    """No parallel-read count is measured ahead or kept (the weight reader measures itself on each load, 2026-11-19):
    a profile without one is complete."""
    path, calls = _stale(tmp_path, monkeypatch, {"model": "xeon", "runtime_base_mb": {"triton": 1}})
    before = path.read_text()
    autodetect.get_or_create_default_profile()
    assert calls == [] and path.read_text() == before
