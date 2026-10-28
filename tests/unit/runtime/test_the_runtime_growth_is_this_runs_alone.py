"""The runtime's growth a run records (core/runtime/runtime_growth.py) is THIS run's: its height over the run (noted
after each sequence's first pass, one driver call) or its end, less what other processes hold — the larger of the run's
two ends, so a run that starts beside it is not counted — less the reading before the executor. The driver's tool runs
at the two ends only, never on the run's path. No card: the readings are injected."""
from neurobrix.core.runtime import runtime_growth as rg


def test_height_counts_and_another_processs_growth_does_not(monkeypatch, tmp_path):
    raw = {"mb": 1000.0 + 300.0}        # the driver's used figure: another process's 1 000 MB + this run's 300 context
    others = {"mb": 1000.0}
    calls = []
    monkeypatch.setattr(rg, "untracked_raw_mb", lambda d: raw["mb"])
    monkeypatch.setattr(rg, "_held_by_others_mb", lambda d: calls.append(d) or others["mb"])
    monkeypatch.setattr(rg, "_root", lambda: tmp_path)
    before = rg.begin([0])
    assert before == {0: 300.0}
    raw["mb"] = 2000.0 + 300.0 + 900.0  # the run grew 900 MB, and the other process 1 000 MB more
    others["mb"] = 2000.0
    n = len(calls)
    rg.note(0)
    assert len(calls) == n, "note() ran the driver's tool on the run's path"
    raw["mb"] = 2000.0 + 300.0 + 400.0  # the end: the run gave back 500 of its height
    assert rg.record("m", "triton", "volta", before) == 900.0
