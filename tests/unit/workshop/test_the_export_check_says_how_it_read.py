"""A throughput measurement that cannot say how it was taken is a number.

The export health check reads bytes off the mount and refuses under a floor. Two
ways the CLIENT can answer a question about the SERVER, and both were live:

* O_DIRECT bypasses the page cache, which is the point — but an export may refuse
  it, and a check that silently falls back measures the client's memory. The
  symptom observed on 2026-09-12 was 601 MB/s over NFS, which this link does not
  produce for a cold read.
* A buffered read of a file something just touched is a cache hit whatever the
  server is doing.

So the fallback reads from a deep offset AND labels itself, and the label travels
with the number.

Run: PYTHONPATH=src python -m pytest tests/unit/workshop/test_the_export_check_says_how_it_read.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import export_quiet as eq  # noqa: E402
from export_quiet import (  # noqa: E402
    ExportBusy, _probe_file, refuse_busy_export, throughput_mb_s,
)


#: The real probe reads 200 MB, which is right on an export and wrong in a test:
#: at six calls it spent 288 seconds moving 1.2 GB to answer questions about
#: labels and refusals. The SIZE is not what any of these tests are about.
TEST_PROBE_MB = 4


@pytest.fixture(autouse=True)
def _small_probe(monkeypatch):
    monkeypatch.setattr(eq, "PROBE_MB", TEST_PROBE_MB)


@pytest.fixture
def big_file(tmp_path):
    """A file large enough for the probe to pick and read."""
    p = tmp_path / "weights.safetensors"
    with open(p, "wb") as fh:
        fh.write(b"\0" * ((TEST_PROBE_MB + 2) * 2**20))
    return p


def test_it_reports_how_it_read(big_file):
    rate, how = throughput_mb_s(big_file, megabytes=4, timeout=60)
    assert rate > 0
    assert how.startswith("O_DIRECT") or "offset" in how


def test_the_probe_finds_a_large_file_without_walking_everything(tmp_path, big_file):
    """The first version used rglob('*') and took minutes on the very export it
    was meant to declare busy. A health check that walks a tree is the load."""
    for i in range(50):
        (tmp_path / f"noise{i}.bin").write_bytes(b"\0" * 1024)
    assert _probe_file(tmp_path) == big_file


def test_nothing_large_enough_is_not_a_pass(tmp_path, capsys):
    """A check that measured nothing must not read like one that measured health."""
    (tmp_path / "small.bin").write_bytes(b"\0" * 1024)
    assert refuse_busy_export(tmp_path) == -1.0
    assert "NOT measured" in capsys.readouterr().out


def test_a_floor_above_the_measured_rate_refuses(big_file, tmp_path):
    """Seen refusing, on a rate no local disk can beat."""
    with pytest.raises(ExportBusy) as exc:
        refuse_busy_export(tmp_path, floor=10 ** 9)
    assert "EXPORT BUSY" in str(exc.value)
    assert "--allow-busy-export" in str(exc.value)


def test_the_named_opening_proceeds(big_file, tmp_path, capsys):
    rate = refuse_busy_export(tmp_path, floor=10 ** 9, allow=True)
    assert rate > 0
    assert "--allow-busy-export" in capsys.readouterr().out


def test_a_reachable_floor_passes(big_file, tmp_path):
    """Without this the refusals prove nothing: the check must be able to pass."""
    assert refuse_busy_export(tmp_path, floor=0.001) > 0
