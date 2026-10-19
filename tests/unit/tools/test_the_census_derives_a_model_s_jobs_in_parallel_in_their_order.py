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
    return ([(f"op{rung}", "kern", f"({mode}, {rung})")], None, None, {f"NOTE {mode}": rung})


def test_the_parallel_results_are_the_serial_results_in_the_jobs_order(monkeypatch):
    monkeypatch.setattr(DC, "derive_keys", _fake_derive_keys)
    jobs = [(mode, 0, ["--prompt", "x"], rung) for mode in ("triton", "triton_sequential") for rung in range(1, 8)]
    par = DC._derive_jobs("m", "hw", jobs)
    ser = [DC._derive_job(("m", "hw", mode, rung, req)) for mode, _ri, req, rung in jobs]
    assert par == ser
    assert [r[0] for r in par[:7]] == ["ok", "ok", "plan_refused", "ok", "system_exit", "ok", "ok"]
    assert par[0][1][0] == [("op1", "kern", "(triton, 1)")]
