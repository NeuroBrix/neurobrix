"""The weight loader pins glibc's mmap threshold from config/system.yml before it reads a byte.

With glibc's dynamic threshold, the streamed loader's freed temporaries stayed in the heap: 6
deepseek-moe shards left 22.7 GB of anonymous RSS; pinned at 1 MiB, 1.07 GB peak (2026-10-10).
Injection: drop `apply_host_mmap_threshold()` from WeightLoader.__init__ -> the first test goes red.
"""
import platform

import pytest

from neurobrix.core import workspace


class _Libc:
    def __init__(self):
        self.calls = []

    def mallopt(self, param, value):
        self.calls.append((param, value))
        return 1


@pytest.fixture
def libc(monkeypatch, tmp_path):
    fake = _Libc()
    monkeypatch.setattr(workspace, "_mmap_threshold_set", False)
    monkeypatch.setattr(workspace, "_libc", lambda: fake)
    monkeypatch.setattr(platform, "libc_ver", lambda: ("glibc", "2.35"))
    return fake


def test_the_loader_sets_the_configured_threshold_before_reading(libc, tmp_path):
    from neurobrix.core.io.weight_loader import WeightLoader
    (tmp_path / "m").mkdir()
    try:
        WeightLoader(str(tmp_path / "m"))  # an empty directory: the loader refuses it after the threshold
    except FileNotFoundError:
        pass
    import yaml
    n = yaml.safe_load(workspace.SYSTEM_YML.read_text())["io"]["host_mmap_threshold_bytes"]
    assert libc.calls == [(-3, n)]


def test_a_missing_threshold_is_refused_by_name(libc, monkeypatch, tmp_path):
    y = tmp_path / "system.yml"
    y.write_text("io:\n  num_workers: 8\n")
    monkeypatch.setattr(workspace, "SYSTEM_YML", y)
    with pytest.raises(RuntimeError, match="host_mmap_threshold_bytes"):
        workspace.apply_host_mmap_threshold()


def test_not_glibc_is_left_alone(libc, monkeypatch):
    monkeypatch.setattr(platform, "libc_ver", lambda: ("", ""))
    workspace.apply_host_mmap_threshold()
    assert libc.calls == []
