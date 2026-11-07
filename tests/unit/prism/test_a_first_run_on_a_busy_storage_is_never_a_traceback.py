"""A user's first `neurobrix run` never ends in a Traceback because its storage was busy. On a clean container whose
model cache was a network mount (2026-11-07), the first-launch measure of the loaders' parallel reads found no stable
read rate and `workers_for`'s refusal ended the run: "UNEXPECTED ERROR: ZERO FALLBACK: no stable read-rate measure"
and a Traceback. Now the run reads serially (one read in flight, the least host memory, and Prism prices the same
count), its launch line says so, the profile is written WITHOUT a count, and the next run measures again (the
supervisor, 2026-11-07 02:45). No card.

What this test would do if the code were wrong: the refusal escaping the detection (red), a count written into the
profile from an unmeasured run (red), the count used without a word on the launch line (red), a later profile that
declares its count losing to the unmeasured one (red)."""
from __future__ import annotations

import pytest


def _refuse(*a, **k):
    raise RuntimeError("ZERO FALLBACK: no stable read-rate measure on /models up to 2098 MiB per trial (1973 MiB: 0/3 "
                       "steady): another reader on the storage?")


@pytest.fixture
def fresh(monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    monkeypatch.setattr(W, "_IO_UNMEASURED", None)
    import nbxtensor.io.read_rate as RR
    monkeypatch.setattr(RR, "workers_for", _refuse)
    return W


def test_an_unstable_measure_gives_one_serial_read_said_on_the_launch_line(fresh, capsys):
    from neurobrix.core.prism import autodetect as A
    assert A._measure_io_workers() is None
    out = capsys.readouterr().out
    assert "[Engine] io workers: 1 for this run — no stable read-rate measure on /models" in out, out
    assert "measured again at the next run" in out
    assert fresh.io_workers() == 1


def test_the_profile_keeps_no_count_from_an_unmeasured_run(fresh, tmp_path, monkeypatch):
    import yaml
    from neurobrix.core.prism import autodetect as A
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda: {"triton": 1})
    p = tmp_path / "default-x.yml"
    p.write_text(yaml.safe_dump({"cpu": {"model": "m", "runtime_base_mb": {"triton": 1}}}))
    A._complete_measured_fields(p)
    assert "io_workers" not in yaml.safe_load(p.read_text())["cpu"]


def test_a_declared_count_wins(fresh):
    from neurobrix.core.prism import autodetect as A
    A._measure_io_workers()
    fresh.declare_io_workers(6)
    assert fresh.io_workers() == 6


def test_without_the_failed_measure_a_missing_count_is_still_refused(monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    monkeypatch.setattr(W, "_IO_UNMEASURED", None)
    with pytest.raises(RuntimeError, match="cpu.io_workers"):
        W.io_workers()
