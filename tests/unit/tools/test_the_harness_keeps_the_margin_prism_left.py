"""One margin on a unified host, not two (the Mac, 2026-10-03 23:59).

Prism on unified memory plans at the highest rung whose whole host side fits the free reading — the
ladder's rounding is the margin (the owner's rule). The harness then admitted a cell only when the
available memory covered the plan's host figure AND 1/16 of the host (1.5 GB on a 24 GB Mac): Janus,
Voxtral and orpheus planned to within 0.1-0.7 GB of the free memory and were never admitted. A cell
priced by a plan that drew its device memory from the host keeps no second headroom; a discrete card's
plan and the static estimate keep it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _host(monkeypatch, total, available):
    monkeypatch.setattr(R, "_host_bytes", lambda: total)
    monkeypatch.setattr(R, "_mem_available", lambda: available)


def _cell(tmp_path, monkeypatch, plan):
    seen = {}
    monkeypatch.setattr(R, "plan_host_need", lambda *a: plan)
    monkeypatch.setattr(R, "reserve_host", lambda out, need, headroom=None: seen.update(need=need, headroom=headroom) or True)
    monkeypatch.setattr(R, "release_host", lambda out: None)
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: {"model": "M"})
    R.run_cell("M", "triton", "0", tmp_path, 60, tmp_path)
    return seen


def test_a_unified_plan_is_admitted_on_the_margin_prism_left(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 11 * GiB)
    assert R.reserve_host(tmp_path, 11 * GiB - (300 << 20), headroom=0) is True


def test_the_cell_of_a_unified_plan_reserves_without_a_second_headroom(tmp_path, monkeypatch):
    assert _cell(tmp_path, monkeypatch, (10 * GiB, "plan-unified"))["headroom"] == 0


def test_a_discrete_plan_and_the_estimate_keep_the_hosts_headroom(tmp_path, monkeypatch):
    assert _cell(tmp_path, monkeypatch, (10 * GiB, "plan")).get("headroom") is None
    monkeypatch.setattr(R, "container_bytes", lambda model: 4 * GiB)
    assert _cell(tmp_path, monkeypatch, None).get("headroom") is None


def test_the_plan_figure_says_when_the_device_draws_on_the_host(tmp_path, monkeypatch):
    import json, subprocess
    for dev, want in ((5 * GiB, "plan-unified"), (0, "plan")):
        doc = {"plan": {"host_footprint": {"total_bytes": 9 * GiB, "device_bytes": dev}}}
        monkeypatch.setattr(R, "cell_request", lambda model: [])
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(doc), ""))
        assert R.plan_host_need("M", "triton", "0", tmp_path) == (9 * GiB, want)
