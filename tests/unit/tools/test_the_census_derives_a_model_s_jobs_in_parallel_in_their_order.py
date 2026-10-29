"""The census derives a model's (mode, request, rung) jobs in forked workers, consumed in their own order.

`tools/derived_census._derive_jobs` runs each derivation in a worker (one per job up to the machine's cores) and hands
the results back in the jobs' order, a refusal by name included (`plan_refused`, `system_exit`) — the table is the one
process's table, row for row (TinyLlama 16 GB, 2026-10-19: 328 rows identical, 137 s -> 10 s).

What this test would do if the code were wrong: results consumed as workers finish (a reordered pool) — red at the
order assertion (the jobs below finish in reverse order); a refusal lost or raised in the worker — red at its row.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import derived_census as DC  # noqa: E402


def _fake_derive_keys(model, hw, mode, rung, req):
    time.sleep(0.02 * (8 - rung))             # later jobs finish first
    if rung == 3:
        raise DC.PlanRefused(f"{mode} r{rung} refused")
    if rung == 5:
        raise SystemExit(f"{mode} r{rung} not derivable")
    return ([(f"op{rung}", "kern", f"({mode}, {rung})")], {"strategy": f"plan r{rung}"}, None, {f"NOTE {mode}": rung})


def test_the_parallel_results_are_the_serial_results_in_the_jobs_order(monkeypatch):
    monkeypatch.setattr(DC, "derive_keys", _fake_derive_keys)
    jobs = [(mode, 0, ["--prompt", "x"], rung) for mode in ("triton", "triton_sequential") for rung in range(1, 8)]
    # fork: the monkeypatched derive_keys reaches forked workers only — a spawned worker re-imports the real one
    # (the platform default on macOS; red there before this line, KeyError on the fake mode)
    par = DC._derive_jobs("m", "hw", jobs, start_method="fork")
    ser = [DC._derive_job(("m", "hw", mode, rung, req)) for mode, _ri, req, rung in jobs]
    # the results and the plan each was derived from; what each process executed and read differs by process
    assert [(r[0], r[1], r[2]["plan"]) for r in par] == [(r[0], r[1], r[2]["plan"]) for r in ser]
    assert [r[0] for r in par[:7]] == ["ok", "ok", "plan_refused", "ok", "system_exit", "ok", "ok"]
    assert par[0][1][0] == [("op1", "kern", "(triton, 1)")]


def test_the_workers_are_what_the_free_memory_holds_not_the_cores(monkeypatch):
    """A derivation holds a whole graph: on 24 GB of unified memory one worker per core saturated the Mac (three
    panics, 2026-10-20). The workers are bounded by the free memory at its ladder rung over what one worker measured,
    then by the cores and the jobs; unread memory runs one at a time. Red if the cores alone decide."""
    import types
    from nbxtensor.backend import host_memory as HM
    from neurobrix.core.prism import memory_budget as MB
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [4096, 8192, 16384])
    def free(mb):
        monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=mb))
    free(10_000)                                       # rung 8 GiB; a worker of 2 GiB -> 4 at once
    assert DC._workers_memory_holds(20, 2 << 30) == 4
    free(3_000)                                        # under the lowest rung: one at a time
    assert DC._workers_memory_holds(20, 2 << 30) == 1
    free(100_000)                                      # memory to spare: the cores, then the jobs
    assert DC._workers_memory_holds(20, 1 << 20) == 12 and DC._workers_memory_holds(3, 1 << 20) == 3
    free(None)
    assert DC._workers_memory_holds(20, 1 << 20) == 1


def test_the_workers_are_the_cores_the_machine_is_not_running_on(monkeypatch):
    """Eight census processes each starting a worker per core held ~300 derivations on 48 cores and a plan outran its
    explanation's timeout (2026-10-22). Red if the cores alone decide, whatever the machine already runs."""
    import types
    from nbxtensor.backend import host_memory as HM
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=1_000_000))
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (9.6, 9.0, 8.0))
    assert DC._workers_memory_holds(20, 1 << 20) == 3
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (40.0, 40.0, 40.0))
    assert DC._workers_memory_holds(20, 1 << 20) == 1


def test_the_workers_leave_what_the_gate_cells_reserved_and_say_their_own(monkeypatch, tmp_path):
    """A census beside a gate sized its workers on the free memory a just-admitted Qwen3-30B cell had reserved but
    not yet taken, and the kernel killed the cell (2026-10-23 00:10:30). The workers are what the free memory holds
    LESS the other holders' reservations in the shared ledger (a gate's `hold_c<card>` "<pid> <bytes>"; a certifier's
    hold naming no bytes reserves none), and the census writes its own `hold_host_<pid>` while they run, removed on
    exit; another census's hold is not subtracted (its workers are in the free memory read). Red if the ledger is not
    read (8 workers, not 2), if another census is subtracted too (1, not 2), or if the hold is not written or removed."""
    import os
    import types
    from nbxtensor.backend import host_memory as HM
    from neurobrix.core.prism import memory_budget as MB
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [4096, 8192, 16384])
    monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=20_000, total_mb=20_000))
    (tmp_path / "hold_c3").write_text(f"{os.getppid()} {12 << 30}\n")          # a gate cell (a live process): 12 GiB
    (tmp_path / "hold_c1").write_text("c1gemv-final 4242\n")                   # a certifier's card: no bytes
    (tmp_path / "hold_host_999999").write_text(f"999999 {64 << 30}\n")       # another census: in the free memory already
    monkeypatch.setattr(DC, "HOLDS", None)
    assert DC._workers_memory_holds(20, 2 << 30) == 7                          # no ledger: 0.8 x 19.5 GiB / 2 GiB
    monkeypatch.setattr(DC, "HOLDS", tmp_path)
    assert DC._workers_memory_holds(20, 2 << 30) == 1                          # min(16, 0.8 x 19.5 - 12) GiB / 2 GiB
    mine = tmp_path / f"hold_host_{os.getpid()}"
    with DC._host_hold(tmp_path, 4 << 30):
        assert mine.read_text().split() == [str(os.getpid()), str(4 << 30)]
        assert DC._held_by_others(tmp_path) == 12 << 30                        # its own hold is not another's
    assert not mine.exists()


def test_a_hold_whose_process_is_gone_reserves_nothing(tmp_path):
    """A census writer killed before it removed its hold left 26 GB reserved (2026-10-23 07:46). Red if a dead
    holder's bytes are still counted."""
    import os
    import subprocess
    import sys as _sys
    gone = subprocess.Popen([_sys.executable, "-c", "pass"])
    gone.wait()
    (tmp_path / "hold_c2").write_text(f"{gone.pid} {40 << 30}\n")
    (tmp_path / "hold_c3").write_text(f"{os.getppid()} {12 << 30}\n")
    assert DC._held_by_others(tmp_path) == 12 << 30


def test_a_running_gate_cell_is_not_counted_twice(monkeypatch, tmp_path):
    """A gate cell's reservation is already partly inside the free memory read: the census's room is the free memory
    bounded by what the reservations leave of the TOTAL, not the free memory minus the reservations. Counting both made
    a census beside a gate derive Qwen3-30B one job at a time into its 3 h timeout (2026-10-23). 251 GB machine, a
    132 GiB cell running (100 GB free): 119 GiB of room, not none. Red if the reservation is subtracted from the free."""
    import os
    import types
    from nbxtensor.backend import host_memory as HM
    from neurobrix.core.prism import memory_budget as MB
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [65536, 98304, 131072])
    monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=100_000, total_mb=257_000))
    (tmp_path / "hold_c3").write_text(f"{os.getppid()} {132 << 30}\n")
    monkeypatch.setattr(DC, "HOLDS", tmp_path)
    assert DC._workers_memory_holds(60, 2 << 30) == 34                        # (0.8 x 251 - 132) GiB / 2 GiB


def test_the_room_keeps_the_scheduler_s_headroom(monkeypatch, tmp_path):
    """A census that took the whole of total - reservations left an admitted 133 GiB gate cell no margin: the kernel
    killed it (2026-10-23 20:46). The room stays within the scheduler's own host share (0.8): 256 GiB machine, a 133
    GiB cell, 200 GiB free (rung 192): room = min(192, 204.8 - 133) = 71.8 GiB -> 35 workers of 2 GiB, not 61. Red
    if the share is not kept."""
    import os
    import types
    from nbxtensor.backend import host_memory as HM
    from neurobrix.core.prism import memory_budget as MB
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [65536, 131072, 196608])
    monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=204_800, total_mb=262_144))
    (tmp_path / "hold_c2").write_text(f"{os.getppid()} {133 << 30}\n")
    monkeypatch.setattr(DC, "HOLDS", tmp_path)
    assert DC._workers_memory_holds(100, 2 << 30) == 35


def test_the_census_writes_its_progress_end_and_failure(tmp_path, monkeypatch):
    """Red line of work 2 (Hocine 2026-10-23): a census writes its progress to its record at most every 10 min, each
    model written, its end and a failure. Red if a forced note is not written, a non-forced one is written inside the
    interval, or the record is not the file named."""
    rec = tmp_path / "RUN.md"
    monkeypatch.setitem(DC._PROGRESS, "file", str(rec))
    monkeypatch.setitem(DC._PROGRESS, "label", "nvidia/volta hw")
    monkeypatch.setitem(DC._PROGRESS, "last", 0.0)
    DC._note("m: 1/4 derivation(s)")                  # first: the interval since 0 has passed
    DC._note("m: 2/4 derivation(s)")                  # inside the interval: not written
    DC._note("m: 12 row(s) written (1/1 model(s))", force=True)
    lines = rec.read_text().splitlines()
    assert [l.split(" ", 3)[3].rsplit(" ", 3)[0] for l in lines] == ["m: 1/4 derivation(s)",
                                                                     "m: 12 row(s) written (1/1 model(s))"], lines


def test_a_fresh_worker_returns_its_result_and_raises_its_failure():
    """The one-job worker is a process and a pipe, no pool (a pool replacing its worker from its handler thread hung a
    forked child on the queue's lock). Red if the result or the failure does not cross."""
    import multiprocessing as mp
    import pytest
    ctx = mp.get_context("fork")
    assert DC._run_fresh(ctx, abs, -3) == 3
    with pytest.raises(ZeroDivisionError):
        DC._run_fresh(ctx, (lambda x: 1 // x), 0)
