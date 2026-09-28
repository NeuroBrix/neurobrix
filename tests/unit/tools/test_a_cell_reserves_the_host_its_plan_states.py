"""A cell reserves the host footprint its engine's plan states, not a static multiple of its weights.

The owner's rule (2026-09-27 14:27): a reservation is what the plan the engine chose says the run
will hold. The matrix reserved container bytes x 1.7 for every cell; on 2026-09-28 07:49 two such
reservations (65 GB and 97 GB of estimate) held a gate's card idle while the host used 17 GB. A tree
carrying Prism's host estimate states it in `--explain-plan --json` (`plan.host_footprint`); a tree
before it states nothing and keeps the static estimate — said in the row either way.

What each test would do if the code were wrong: a tool that ignored the plan reserves the static
estimate — the first test RED (seen: before this change `run_cell` had no plan query); a tool that
invented a figure for a tree stating none — the second RED; a tool that reserved the static figure
while the plan was known — the third RED on `host_reserved_from`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import regression_matrix as R  # noqa: E402


def _fake_tree(tmp_path: Path, plan: dict) -> Path:
    """A `src` whose `python -m neurobrix` prints an explain-plan JSON document."""
    pkg = tmp_path / "src" / "neurobrix"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(f"print('[Prism] planning...')\nprint({json.dumps(json.dumps({'plan': plan}))})\n")
    return tmp_path / "src"


@pytest.fixture
def no_request(monkeypatch):
    monkeypatch.setattr(R, "cell_request", lambda model: ["--prompt", "x"])


def test_a_tree_that_states_its_host_figure_is_reserved_by_it(tmp_path, no_request):
    src = _fake_tree(tmp_path, {"strategy": "single_gpu", "host_footprint": {"total_bytes": 12 << 30}})
    assert R.plan_host_need("M", "triton", "0", src) == (12 << 30, "plan")


def test_a_tree_that_states_none_keeps_the_static_estimate(tmp_path, no_request):
    src = _fake_tree(tmp_path, {"strategy": "single_gpu"})
    assert R.plan_host_need("M", "triton", "0", src) is None


def test_the_row_says_where_its_reservation_came_from(tmp_path, no_request, monkeypatch):
    src = _fake_tree(tmp_path, {"host_footprint": {"total_bytes": 7 << 30}})
    seen = {}
    monkeypatch.setattr(R, "container_bytes", lambda model: 100 << 30)
    monkeypatch.setattr(R, "reserve_host", lambda out, need: seen.setdefault("need", need) or True)
    monkeypatch.setattr(R, "release_host", lambda out: None)
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: {"model": "M"})
    row = R.run_cell("M", "triton", "0", tmp_path, 60, src)
    assert seen["need"] == 7 << 30
    assert row["host_reserved"] == 7 << 30 and row["host_reserved_from"] == "plan"


def test_a_tree_stating_none_is_priced_by_the_tree_named_in_price_src(tmp_path, no_request, monkeypatch):
    """A tree before prism-prices-the-host is priced by a plan-identical tree named in <out>/price_src.
    If the code ignored price_src, the static estimate would be reserved (RED)."""
    under = _fake_tree(tmp_path / "under", {"strategy": "single_gpu"})
    pricer = _fake_tree(tmp_path / "pricer", {"host_footprint": {"total_bytes": 9 << 30}})
    out = tmp_path / "out"; out.mkdir()
    (out / "price_src").write_text(str(pricer.parent))
    seen = {}
    monkeypatch.setattr(R, "container_bytes", lambda model: 100 << 30)
    monkeypatch.setattr(R, "reserve_host", lambda o, need: seen.setdefault("need", need) or True)
    monkeypatch.setattr(R, "release_host", lambda o: None)
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: {"model": "M"})
    row = R.run_cell("M", "triton", "0", out, 60, under)
    assert seen["need"] == 9 << 30 and row["host_reserved_from"] == "plan@pricer"
