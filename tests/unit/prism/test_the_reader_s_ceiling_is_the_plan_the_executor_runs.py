"""1215: the weight reader's ceiling is the plan the executor runs, whatever was solved before or after it.

On engine main a serve first and a run first of the same model kept two different reader entries (width 16 after the
serve, 64 after the run): `load_ceiling.settle` declared the ceiling of every plan it priced, so a solve whose plan
nothing ran set the ceiling of the next load. Now a solve only prices its plan and the executor adopts the ceiling of
the plan it runs (`workspace.adopt_load_ceiling`). What this would do if the code were wrong: the two orders give two
ceilings, or a probe's solve moves the ceiling of the plan in force — RED. And the reader a new ceiling replaces is
closed only by the thread that made it (a daemon holding two models loads in two threads) — RED if another thread's
reader is shut under its load."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from neurobrix.core import workspace as W
from neurobrix.pytorch.prism import kept_launch_facts as kept
from neurobrix.pytorch.prism import load_ceiling


def _plan():
    return SimpleNamespace(load_ceiling=0, host_footprint=None)


def _solve(threads: int):
    """A plan priced the way the solver prices it (`settle`), on a host whose price fits every count: its ceiling is
    the profile's threads."""
    profile = SimpleNamespace(cpu=SimpleNamespace(threads=threads, ram_mb=1 << 30), devices=[])
    plan = _plan()
    load_ceiling.settle(plan, profile, lambda k: {"total_bytes": k}, lambda p: 0, lambda p, installed: installed)
    return plan


class _Reader:
    made = []

    def __init__(self, ceiling, start=1):
        self.ceiling, self.start, self.closed, self.bytes = ceiling, start, False, 0
        _Reader.made.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def fresh(monkeypatch):
    import nbxtensor.io.reader as R
    monkeypatch.setattr(W, "_LOAD_CEILING", None)
    monkeypatch.setattr(W, "_READER", None)
    monkeypatch.setattr(R, "Reader", _Reader)
    monkeypatch.setattr(kept, "kept_read_width", lambda hw, source: 1)
    _Reader.made.clear()


def test_a_solve_prices_its_plan_and_moves_no_ceiling(fresh):
    plan = _solve(64)
    assert plan.load_ceiling == 64 and plan.host_footprint == {"total_bytes": 64}
    assert W._LOAD_CEILING is None, "a solve whose plan nothing runs declared its ceiling"


@pytest.mark.parametrize("order", ["serve probe first", "run first"])
def test_both_orders_read_at_the_ceiling_of_the_plan_in_force(fresh, order):
    run, probe = _solve(64), _solve(16)
    if order == "serve probe first":
        _solve(16)
        W.adopt_load_ceiling(run)
    else:
        W.adopt_load_ceiling(run)
        _solve(16)
    assert W.load_ceiling() == 64 and W.weight_reader().ceiling == 64, order
    W.adopt_load_ceiling(probe)               # the executor of another plan adopts its own
    assert W.load_ceiling() == 16 and W.weight_reader().ceiling == 16


def test_a_plan_that_priced_no_reads_leaves_no_ceiling(fresh, monkeypatch):
    monkeypatch.setattr(W, "processors", lambda: 5)
    W.adopt_load_ceiling(_solve(64))
    W.adopt_load_ceiling(_plan())             # built before step 7.9: 0
    assert W.load_ceiling() == 5, "another plan's ceiling outlived it"


def test_a_reader_another_thread_made_is_not_closed_under_its_load(fresh):
    W.adopt_load_ceiling(_solve(8))
    made = []
    t = threading.Thread(target=lambda: made.append(W.weight_reader()))
    t.start(); t.join()
    W.adopt_load_ceiling(_solve(16))
    mine = W.weight_reader()
    assert mine.ceiling == 16 and not made[0].closed, "this thread shut the pool of another thread's load"
    W.adopt_load_ceiling(_solve(32))
    assert W.weight_reader().ceiling == 32 and mine.closed, "the reader this thread made and replaced stayed open"
