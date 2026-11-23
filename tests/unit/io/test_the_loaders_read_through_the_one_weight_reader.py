"""Every weight loader reads through ONE weight reader, the process's (`core.workspace.weight_reader`, nbxtensor's
`io.reader`): parallel reads that start at one and grow by their own measure on the load, up to the ceiling Prism's
host estimate priced (`declare_load_ceiling`). No worker count is stored, measured ahead, or written by hand.

History: 2026-09-27 three loaders each wrote 8; until 2026-10-30 system.yml held a hand-written count; until the
launch-time cure (2026-11-19) the profile kept a count measured ahead of the load (130 s on a quiet store, owed again at
every launch on a busy one). What each test would do if the code were wrong: a loader keeping a pool or count of its own
(red), the count read back from a profile or system.yml (red), a ceiling other than the plan's (red), a forked child
sharing its parent's reader (red)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


@pytest.fixture
def W(monkeypatch):
    from neurobrix.core import workspace
    monkeypatch.setattr(workspace, "_LOAD_CEILING", None)
    monkeypatch.setattr(workspace, "_READER", None)
    return workspace


def test_the_reader_grows_up_to_the_plan_s_ceiling(W):
    W.declare_load_ceiling(3)
    r = W.weight_reader()
    assert r.ceiling == 3 and r.width == 1
    assert W.weight_reader() is r                      # one reader for the process
    W.declare_load_ceiling(5)
    assert W.weight_reader().ceiling == 5              # another plan, another ceiling: a new reader


def test_a_replaced_reader_is_closed_and_a_forked_child_closes_none(W, monkeypatch):
    W.declare_load_ceiling(2)
    old, closed = W.weight_reader(), []
    monkeypatch.setattr(type(old), "close", lambda self: closed.append(self))
    W.declare_load_ceiling(3)
    assert W.weight_reader().ceiling == 3 and closed == [old]          # its threads end with it, never leak
    child_pid = os.getpid() + 1
    monkeypatch.setattr(W.os, "getpid", lambda: child_pid)
    W.declare_load_ceiling(4)
    W.weight_reader()
    assert closed == [old]                                             # the parent's reader is the parent's to close


def test_no_plan_the_processors_bound_it(W):
    assert W.load_ceiling() == W.processors() >= 1


@pytest.mark.parametrize("bad", [0, -2, None, 1.5])
def test_a_ceiling_is_a_count(W, bad):
    with pytest.raises(ValueError, match="ZERO FALLBACK"):
        W.declare_load_ceiling(bad)


def test_a_forked_child_builds_its_own_reader(W, monkeypatch):
    W.declare_load_ceiling(2)
    parent = W.weight_reader()
    child_pid = os.getpid() + 1
    monkeypatch.setattr(W.os, "getpid", lambda: child_pid)
    assert W.weight_reader() is not parent


def test_no_count_is_read_from_a_profile_or_the_system_file(W):
    from neurobrix.pytorch.prism.cpu_config import CPUConfig
    cpu = CPUConfig.from_yaml_dict({"model": "m", "cores": 4, "threads": 8, "ram_mb": 1, "architecture": "arm64"})
    assert not hasattr(cpu, "io_workers")
    assert "num_workers" not in (yaml.safe_load(W.SYSTEM_YML.read_text()).get("io") or {})
    for name in ("io_workers", "declare_io_workers", "UNMEASURED_IO_WORKERS", "declare_unmeasured_io_workers"):
        assert not hasattr(W, name), name


def test_the_loaders_read_it_when_they_load_never_at_import(W):
    import importlib
    for mod in ("neurobrix.core.io.weight_loader", "neurobrix.nbx.loader", "neurobrix.triton.weight_loader"):
        importlib.reload(importlib.import_module(mod))          # no plan in force: the import must not build a reader
    assert W._READER is None


def test_no_loader_keeps_a_pool_or_a_count_of_its_own():
    for rel in ("core/io/weight_loader.py", "nbx/loader.py", "triton/weight_loader.py"):
        text = (SRC / rel).read_text()
        assert "ThreadPoolExecutor" not in text, rel
        assert "weight_reader" in text, rel
        assert not re.search(r"WORKERS\s*=\s*(int\(os\.environ\.get\([^)]*\)\)|\d+)\s*$", text, re.M), rel


def test_the_launch_line_says_what_the_loads_measured(W, tmp_path):
    from neurobrix.pytorch.prism import launch_lines
    assert W.weight_reader_said() is None                       # nothing read: nothing said
    W.declare_load_ceiling(2)
    p = tmp_path / "w.bin"
    p.write_bytes(os.urandom(1 << 20))
    fd = os.open(p, os.O_RDONLY)
    try:
        W.weight_reader().read_bytes(fd, 0, 1 << 20)
    finally:
        os.close(fd)
    said = W.weight_reader_said()
    assert "read 0.00 GB" in said and "of at most 2" in said, said
    assert launch_lines.LINES["loaded"].format(what=said).startswith("Weights loaded: read ")
    assert 'launch_lines.say("loaded"' in (SRC / "serving" / "engine.py").read_text()
