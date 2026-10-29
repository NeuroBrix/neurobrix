"""A model's derivation workers follow the host room as it moves: asked again before every start and while they run.

The pool was sized ONCE per model: the census writer that started Qwen3-Coder's 13 parallel derivations at 21:34,
while a gate held 207 GiB, ran them one at a time — 25 to 70 min each — for hours after the gate had ended at 21:18
(2026-10-23; killed and relaunched by hand at 23:10). The width pass of one derivation, recomputed at each of ~1 200
prompt lengths of one bisection, was 76 % of its time; it reads a binding only as M <= 4 and an empty extent.

What these tests would do if the code were wrong: a pool sized once runs every job with the one worker the first
room allowed — the second case counts the most workers alive at once and fails; a `running` count ignored lets the
room be spent twice — the first case fails; a memo keyed on raw shapes misses every probe (the third case counts
the pass's calls), and one that ignored a signature change would hand an M <= 4 binding the widths of a long one
(the fourth case).
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import derived_census as DC  # noqa: E402


def _room(monkeypatch, mb):
    from nbxtensor.backend import host_memory as HM
    from neurobrix.core.prism import memory_budget as MB
    monkeypatch.setattr(DC.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(DC.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(MB, "memory_ladder_mb", lambda: [1024 * k for k in range(1, 257)])
    monkeypatch.setattr(HM, "memory_state", lambda: types.SimpleNamespace(available_mb=mb[0], total_mb=None))


def test_running_workers_are_counted_before_more_start(monkeypatch):
    mb = [10_240]                                                      # 10 GiB, workers of 2 GiB
    _room(monkeypatch, mb)
    assert DC._workers_memory_holds(20, 2 << 30) == 5
    assert DC._workers_memory_holds(20, 2 << 30, running=5) == 0       # the room is theirs: none more
    assert DC._workers_memory_holds(20, 2 << 30, running=2) == 3


def _slow_job(args):
    time.sleep(0.3)
    return ("ok", args[3])


def test_a_room_that_frees_mid_model_is_used(monkeypatch, tmp_path):
    mb = [1_024]                                                       # one worker's room at the start
    _room(monkeypatch, mb)
    monkeypatch.setattr(DC, "ROOM_EVERY_S", 0.05)
    monkeypatch.setattr(DC, "_derive_job", _slow_job)
    monkeypatch.setattr(DC, "_derive_job_priced", lambda a: (("ok", a[3]), 1 << 30))
    alive, starts = [], []
    real = DC._workers_memory_holds

    def watched(n, wb, running=0):
        if len(alive) == 2:
            mb[0] = 32_768                                             # the gate ended: 32 GiB
        more = real(n, wb, running)
        alive.append(running + more)                                   # workers alive once these start
        return more

    monkeypatch.setattr(DC, "_workers_memory_holds", watched)
    jobs = [("triton", 0, [], r) for r in range(12)]
    got = DC._derive_jobs("m", "hw", jobs, start_method="fork")
    assert [r[1] for r in got] == list(range(12))                      # in the jobs' order
    assert max(alive) >= 4, alive                                      # more than the first room's one at once


def test_the_width_pass_is_reused_where_the_binding_reads_the_same(monkeypatch):
    DC._WIDTHS.clear()
    calls = []

    def compute(shape_of, casts):
        calls.append(1)
        m = shape_of("x")
        casts["u"] = [("x", "float32")]
        return {"y": "float32" if m[0] <= 4 else "float16"}

    for n in (100, 200, 300, 4000):
        rt, casts = DC._widths_at(("k",), lambda t, n=n: [n, 64], compute)
        assert rt == {"y": "float16"} and casts == {"u": [("x", "float32")]}
    assert len(calls) == 1, calls


def test_a_binding_that_reads_differently_is_derived_again(monkeypatch):
    DC._WIDTHS.clear()
    calls = []

    def compute(shape_of, casts):
        calls.append(1)
        return {"y": "float32" if shape_of("x")[0] <= 4 else "float16"}

    assert DC._widths_at(("k",), lambda t: [100, 64], compute)[0] == {"y": "float16"}
    assert DC._widths_at(("k",), lambda t: [3, 64], compute)[0] == {"y": "float32"}
    assert DC._widths_at(("k",), lambda t: [0, 64], compute)[0] == {"y": "float32"}     # an empty extent
    assert len(calls) == 3


def test_the_width_rules_read_a_binding_only_where_the_signature_says():
    """A door on the memo's premise: the width rules read `shape_of` at two places (rows -> mm_store's M, and
    first_nonempty's zero extent). A rule that reads more turns this red before the memo can serve a wrong width."""
    src = (Path(__file__).resolve().parents[3] / "src/neurobrix/core/prism/runtime_widths.py").read_text()
    assert src.count("self.shape_of(") == 2, src.count("self.shape_of(")
    rows_uses = [ln for ln in src.splitlines() if "self.rows(" in ln and "def rows" not in ln]
    assert rows_uses and all("self.mm_store(" in ln for ln in rows_uses), rows_uses
