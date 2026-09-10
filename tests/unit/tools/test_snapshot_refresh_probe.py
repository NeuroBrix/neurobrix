"""The re-download tool stops a download by name the moment the export is under pressure."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import snapshot_refresh as SR  # noqa: E402


def test_a_download_is_stopped_when_the_export_stops_answering(tmp_path, monkeypatch):
    slow = tmp_path / "slow_snap.py"
    slow.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(SR, "FORGE", slow)
    monkeypatch.setattr(SR, "PY", sys.executable)
    monkeypatch.setattr(SR, "REPO", tmp_path)                 # cwd of the child: <tmp>/forge
    (tmp_path / "forge").mkdir()
    probes = iter([0.01, None])                                # answers once, then exceeds the limit
    monkeypatch.setattr(SR, "_export_answers", lambda dest, limit: next(probes))
    notes = []
    args = argparse.Namespace(dest=str(tmp_path), max_workers=1, probe_seconds=5.0, probe_interval=0.2, max_write_mbps=40.0)
    t = time.time()
    rc = SR._download_under_probe("org/name", args, open(tmp_path / "child.log", "w"), notes.append)
    assert rc == -1 and time.time() - t < 30
    assert notes and "STOPPED by the export probe" in notes[0] and "took more than 5 s" in notes[0]


def test_a_download_that_ends_by_itself_returns_its_code(tmp_path, monkeypatch):
    quick = tmp_path / "quick_snap.py"
    quick.write_text("import sys\nsys.exit(3)\n")
    monkeypatch.setattr(SR, "FORGE", quick)
    monkeypatch.setattr(SR, "PY", sys.executable)
    monkeypatch.setattr(SR, "REPO", tmp_path)
    (tmp_path / "forge").mkdir()
    monkeypatch.setattr(SR, "_export_answers", lambda dest, limit: 0.01)
    args = argparse.Namespace(dest=str(tmp_path), max_workers=1, probe_seconds=5.0, probe_interval=0.2, max_write_mbps=40.0)
    assert SR._download_under_probe("org/name", args, open(tmp_path / "child.log", "w"), lambda m: None) == 3


def test_the_pressure_threshold_is_half_the_limit(tmp_path, monkeypatch):
    slow = tmp_path / "slow_snap.py"
    slow.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(SR, "FORGE", slow)
    monkeypatch.setattr(SR, "PY", sys.executable)
    monkeypatch.setattr(SR, "REPO", tmp_path)
    (tmp_path / "forge").mkdir()
    monkeypatch.setattr(SR, "_export_answers", lambda dest, limit: 3.0)   # answered, but in more than limit / 2
    notes = []
    args = argparse.Namespace(dest=str(tmp_path), max_workers=1, probe_seconds=5.0, probe_interval=0.2, max_write_mbps=40.0)
    assert SR._download_under_probe("org/name", args, open(tmp_path / "child.log", "w"), notes.append) == -1
    assert "answered in 3.0 s (pressure)" in notes[0]
