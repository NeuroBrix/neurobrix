"""The regression matrix reserves host memory per cell; two cells over half the budget never overlap.

2026-09-26: three concurrent 30B-class cells held 180 GB of a 251 GB host (1.35-1.65x their
weights), memory pressure reached 33 % "full" and the gate running beside them timed out. The
matrix now reserves 1.7x a container's weights in a flock'd ledger against 55 % of the host.
On a runner that never refused a reservation this fails.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def _child(out, need, q):
    q.put(R.reserve_host(Path(out), need))
    time.sleep(1.5)
    R.release_host(Path(out))


def test_two_cells_over_half_the_budget_never_overlap(tmp_path, monkeypatch):
    # The ledger's mutual exclusion is under test, not this host's live memory: the admission also
    # reads MemAvailable, and on a loaded rack (2026-09-28 20:20, a census and a decode beside it)
    # both cells were refused by the reading and the test failed for a reason it does not test —
    # the reading has its own test below. Forked children inherit the pin.
    monkeypatch.setattr(R, "_mem_available", lambda: R._host_bytes())
    need = int(R._host_bytes() * R.HOST_SHARE) // 2 + 1
    # FORKED, as the comment above says: macOS starts children by spawn, which re-imports the module
    # without the pin, and both cells were refused by the Mac's live reading (the Mac, 2026-10-03).
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_child, args=(str(tmp_path), need, q)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert sorted([q.get(), q.get()]) == [False, True]
    assert json.loads((tmp_path / "host_ledger.json").read_text()) == {}


def test_a_cell_the_measured_memory_cannot_cover_waits(tmp_path, monkeypatch):
    """Reservations fit, but the host's available memory does not cover the cell plus the growth the
    running cells still owe: the cell waits. On a ledger that only summed reservations it starts."""
    monkeypatch.setattr(R, "_host_bytes", lambda: 256 << 30)   # the headroom is a share of the host: pinned
    monkeypatch.setattr(R, "_mem_available", lambda: 8 << 30)
    assert R.reserve_host(tmp_path, 1 << 30) is False
    monkeypatch.setattr(R, "_mem_available", lambda: 64 << 30)
    assert R.reserve_host(tmp_path, 1 << 30) is True
    R.release_host(tmp_path)
