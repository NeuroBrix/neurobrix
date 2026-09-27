"""Every weight loader, and Prism's host estimate, read ONE I/O worker count: $NBX_IO_WORKERS, else
config/system.yml io.num_workers; an unconfigured count is refused by name.

2026-09-27: `core/io/weight_loader.py`, `core/io/loader.py` and `nbx/loader.py` each wrote 8 beside a
comment saying it "matches system.yml io.num_workers", and none read the file — a figure the host
estimate needs (each read in flight holds a shard in host memory) had three sources and no reader.
Before this branch `core.workspace.io_workers` does not exist and the scan finds the literals: these fail.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_the_count_is_the_configured_one(monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.delenv("NBX_IO_WORKERS", raising=False)
    assert W.io_workers() == yaml.safe_load(W.SYSTEM_YML.read_text())["io"]["num_workers"]
    monkeypatch.setenv("NBX_IO_WORKERS", "3")
    assert W.io_workers() == 3


def test_an_unconfigured_count_is_refused(tmp_path, monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.delenv("NBX_IO_WORKERS", raising=False)
    cfg = tmp_path / "system.yml"
    cfg.write_text("paths: {}\n")
    monkeypatch.setattr(W, "SYSTEM_YML", cfg)
    with pytest.raises(RuntimeError, match="io.num_workers"):
        W.io_workers()


def test_no_loader_declares_a_count_of_its_own():
    for rel in ("core/io/weight_loader.py", "core/io/loader.py", "nbx/loader.py", "core/strategies/tp_sharding.py"):
        text = (SRC / rel).read_text()
        assert not re.search(r"WORKERS\s*=\s*(int\(os\.environ\.get\([^)]*\)\)|\d+)\s*$", text, re.M), rel
