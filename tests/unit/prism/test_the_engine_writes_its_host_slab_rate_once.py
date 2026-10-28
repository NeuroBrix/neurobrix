"""The host slab rate (`host_slab_gbps`, the price of a host-read sliced stretch) is measured by the engine inside its
sliced runs and written by the first-launch writer (`autodetect.note_host_slab_rate`, the loader rate's precedent): on a
GENERATED profile only, where neither it nor the vendor file states it, once enough bytes moved — a probe outside the
run does not reproduce it (the Mac 2026-10-28). Red if a committed profile is written, a stated rate overwritten, or a
tiny run writes it. No card."""
import pytest
import yaml

from neurobrix.core.prism import autodetect as A

DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16", "float32"]}


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    return tmp_path


def _profile(hw, name, extra=None):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": [DEV], **(extra or {})}))


def test_a_generated_profile_gets_the_rate_once(hw):
    _profile(hw, "default-abc")
    assert A.note_host_slab_rate("default-abc", 6 * 10 ** 9, 2.0, "test stretches")
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    assert data["host_measured"]["host_slab_gbps"] == 3.0 and "measured by the engine" in data["host_measured"]["host_slab_source"]
    assert not A.note_host_slab_rate("default-abc", 6 * 10 ** 9, 1.0, "again")       # stated now: never overwritten


def test_a_committed_profile_or_a_tiny_run_is_never_written(hw):
    _profile(hw, "c4140-4xv100")
    assert not A.note_host_slab_rate("c4140-4xv100", 6 * 10 ** 9, 2.0, "x")
    _profile(hw, "default-def")
    assert not A.note_host_slab_rate("default-def", 1000, 0.001, "x")
