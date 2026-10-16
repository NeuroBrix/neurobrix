"""One margin on a unified host, not two (the Mac, 2026-10-03 23:59).

Prism on unified memory plans at the highest rung whose whole host side fits the free reading — the
ladder's rounding is the margin (the owner's rule). The harness then admitted a cell only when the
available memory covered the plan's host figure AND 1/16 of the host (1.5 GB on a 24 GB Mac): Janus,
Voxtral and orpheus planned to within 0.1-0.7 GB of the free memory and were never admitted. A cell
priced by a plan that drew its device memory from the host keeps no second headroom; a discrete card's
plan and the static estimate keep it.

The owner's rule of 2026-10-12 22:45 (commit c8ddcfd36): a correctness cell is never admitted against host
memory, so `run_cell` no longer calls `reserve_host` and the cell-level tests could no longer read a headroom
from a mocked reservation (KeyError 'headroom'; the discrete/estimate test had turned vacuous, reading None
from a mock never called). What still holds: the LEDGER keeps one margin for a unified plan and the host's
share otherwise (first test here, and test_the_host_headroom_is_a_share_of_the_host), `plan_host_need` says
which kind of plan it priced (last test), and the cell's ROW carries the plan's figure as the plan stated it —
a unified plan's figure is not given a second headroom by the harness, a discrete plan's and the estimate's
are written as they are, each with its source. The two cell tests now check that, with the ledger a trap: a
run_cell admitting again is RED (seen: c8ddcfd36's run_cell hunk reverted); a harness adding a margin to the
figure is RED on `host_reserved`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _host(monkeypatch, total, available):
    monkeypatch.setattr(R, "_host_bytes", lambda: total)
    monkeypatch.setattr(R, "_mem_available", lambda: available)


def _cell(tmp_path, monkeypatch, plan):
    def trap(*a, **k):
        raise AssertionError("run_cell asked the host ledger: a correctness cell is never admitted against host memory")
    monkeypatch.setattr(R, "plan_host_need", lambda *a: plan)
    monkeypatch.setattr(R, "reserve_host", trap)
    monkeypatch.setattr(R, "release_host", trap)
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: {"model": "M"})
    return R.run_cell("M", "triton", "0", tmp_path, 60, tmp_path)


def test_a_unified_plan_is_admitted_on_the_margin_prism_left(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 11 * GiB)
    assert R.reserve_host(tmp_path, 11 * GiB - (300 << 20), headroom=0) is True


def test_the_cell_of_a_unified_plan_carries_its_figure_without_a_second_headroom(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 11 * GiB)
    row = _cell(tmp_path, monkeypatch, (10 * GiB, "plan-unified"))
    assert (row["host_reserved"], row["host_reserved_from"]) == (10 * GiB, "plan-unified")


def test_a_discrete_plan_and_the_estimate_are_written_as_they_are(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 11 * GiB)
    row = _cell(tmp_path, monkeypatch, (10 * GiB, "plan"))
    assert (row["host_reserved"], row["host_reserved_from"]) == (10 * GiB, "plan")
    monkeypatch.setattr(R, "container_bytes", lambda model: 4 * GiB)
    row = _cell(tmp_path, monkeypatch, None)
    assert (row["host_reserved"], row["host_reserved_from"]) == (int(4 * GiB * R.HOST_PER_WEIGHT_BYTE), "estimate")


def test_the_plan_figure_says_when_the_device_draws_on_the_host(tmp_path, monkeypatch):
    import json, subprocess
    for dev, want in ((5 * GiB, "plan-unified"), (0, "plan")):
        doc = {"plan": {"host_footprint": {"total_bytes": 9 * GiB, "device_bytes": dev}}}
        monkeypatch.setattr(R, "cell_request", lambda model: [])
        monkeypatch.setattr(R, "cell_profile", lambda gpu, src: ("default-x", 16384))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(doc), ""))
        assert R.plan_host_need("M", "triton", "0", tmp_path) == (9 * GiB, want)
