"""Every weight loader, and Prism's host estimate, read ONE I/O worker count: the hardware profile's measured
`cpu.io_workers`, declared where the profile is loaded; a profile that states none is refused by name.

2026-09-27: three loaders each wrote 8 beside a comment saying it "matches system.yml io.num_workers"; until 2026-10-30
that system.yml figure (and an env override) was itself written by hand. Injection (seen red): the loaders' import-time
constants back -> the import refuses before any profile is loaded; `io.num_workers` read again -> the first test fails.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_the_count_is_the_declared_profiles(monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    W.declare_io_workers(3)
    assert W.io_workers() == 3
    assert "num_workers" not in (yaml.safe_load(W.SYSTEM_YML.read_text()).get("io") or {})


def test_a_profile_without_the_fact_is_refused(monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    with pytest.raises(RuntimeError, match="cpu.io_workers"):
        W.io_workers()


def test_loading_a_profile_declares_its_count(monkeypatch):
    from neurobrix.core import workspace as W
    from neurobrix.core.prism.cpu_config import CPUConfig
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    cpu = CPUConfig.from_yaml_dict({"model": "m", "cores": 4, "threads": 8, "ram_mb": 1, "architecture": "arm64",
                                    "io_workers": 2})
    assert cpu.io_workers == 2


def test_the_loaders_read_it_when_they_load_never_at_import():
    import importlib
    from neurobrix.core import workspace as W
    saved = W._IO_WORKERS
    try:
        W.declare_io_workers(None)
        for mod in ("neurobrix.core.io.loader", "neurobrix.core.io.weight_loader", "neurobrix.nbx.loader"):
            importlib.reload(importlib.import_module(mod))          # no profile in force: the import must not read it
    finally:
        W.declare_io_workers(saved)


def test_no_loader_declares_a_count_of_its_own():
    for rel in ("core/io/weight_loader.py", "core/io/loader.py", "nbx/loader.py", "core/strategies/tp_sharding.py"):
        text = (SRC / rel).read_text()
        assert not re.search(r"WORKERS\s*=\s*(int\(os\.environ\.get\([^)]*\)\)|\d+)\s*$", text, re.M), rel
