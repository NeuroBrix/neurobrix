"""An empty visible GPU set is a CPU host, not a detection failure.

2026-09-13: `neurobrix upscale` under `CUDA_VISIBLE_DEVICES=""` — exactly what a
GPU-less machine looks like — died with "No CUDA GPUs are available". The
per-environment profile tag read an EMPTY device list as "detection
unavailable" and served the shared `default.yml`, which on this rack describes
four V100s; Prism then planned `cuda:0` on a process that could see no card.
The regression cell `test_upscaler_runs_with_no_gpu` had passed on 2026-09-03
and went red with the tag's arrival on 2026-09-05, unnoticed until the full
suite was read cell by cell. Register entry 51.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_an_empty_visible_set_is_a_cpu_host.py
"""
from neurobrix.core.prism import autodetect


def test_no_visible_gpu_is_its_own_environment(monkeypatch):
    monkeypatch.setattr(autodetect, "_detect_gpus", lambda _os: ([], None))
    assert autodetect._visible_set_tag() == "cpu"


def test_a_detection_failure_still_falls_back_to_the_shared_profile(monkeypatch):
    def _boom(_os):
        raise RuntimeError("nvidia-smi absent")
    monkeypatch.setattr(autodetect, "_detect_gpus", _boom)
    assert autodetect._visible_set_tag() is None


def test_two_cards_and_one_card_are_different_tags(monkeypatch):
    monkeypatch.setattr(autodetect, "_detect_gpus",
                        lambda _os: ([{"model": "V100"}, {"model": "V100"}], None))
    two = autodetect._visible_set_tag()
    monkeypatch.setattr(autodetect, "_detect_gpus", lambda _os: ([{"model": "V100"}], None))
    assert two != autodetect._visible_set_tag() != "cpu"


def test_a_cpu_detection_never_rewrites_the_shared_default(monkeypatch, tmp_path):
    # The GPU-less cell, run on this rack, overwrote default.yml with
    # `devices: []`; every reader of the shared file then saw a CPU host.
    monkeypatch.setattr(autodetect, "_detect_gpus", lambda _os: ([], None))
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    shared = tmp_path / "default.yml"
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", shared)
    shared.write_text("id: the-rack\ndevices: [{model: V100}]\n")
    monkeypatch.setattr(autodetect, "detect_hardware", lambda: {"id": "cpu-host", "devices": []})
    assert autodetect.get_or_create_default_profile() == "default-cpu"
    assert (tmp_path / "default-cpu.yml").exists()
    assert shared.read_text().startswith("id: the-rack"), "the shared profile is the machine's, not this process's"

