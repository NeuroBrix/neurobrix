"""A cell is priced by the plan it runs on its card ALONE (2026-10-11).

The harness asked Prism for a cell's plan with the cell's card visible — while the previous cell still
held it. Prism reads a card another process holds as shared and budgets the free rung: Ming-Lite-Omni-1.5
16g asked beside a running cell planned cpu_streaming at 189.4 GiB of host, ran layer_streaming (66 515 MB
priced, 31.0 GiB measured), and its reservation held two cards idle. The price is now solved on the census
path — no card visible, the cell's own profile (resolved on its card, detection only) at that profile's
rung — so what occupies the card cannot move it."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _calls(monkeypatch, profile_out="default-ff6008b7 16\n", profile_rc=0):
    seen = []

    def run(cmd, env=None, **k):
        seen.append((cmd, dict(env or {})))
        if "-c" in cmd:   # the profile resolution
            return subprocess.CompletedProcess(cmd, profile_rc, profile_out, "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"plan": {"host_footprint": {"total_bytes": 9 * GiB}}}), "")
    R.cell_profile.cache_clear()
    monkeypatch.setattr(R, "cell_request", lambda model: [])
    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_the_plan_is_solved_with_no_card_visible_at_the_cards_profile_and_rung(tmp_path, monkeypatch):
    seen = _calls(monkeypatch)
    assert R.plan_host_need("M", "native", "1", tmp_path) == (9 * GiB, "plan")
    (pcmd, penv), (cmd, env) = seen
    assert penv["CUDA_VISIBLE_DEVICES"] == "1"                  # the profile: resolved on the cell's card
    assert env["CUDA_VISIBLE_DEVICES"] == ""                    # the plan: no card to read as shared
    assert env["NBX_PRISM_BUDGET_MB"] == "16384" and env["NBX_CENSUS"] == "1"
    assert cmd[cmd.index("--hardware") + 1] == "default-ff6008b7"


def test_a_card_whose_profile_cannot_be_read_is_said_and_never_guessed(tmp_path, monkeypatch, capsys):
    for out, rc in (("", 1), ("default-x None\n", 0), ("garbage\n", 0)):
        _calls(monkeypatch, out, rc)
        assert R.plan_host_need("M", "native", "1", tmp_path) is None
        assert "its profile could not be resolved" in capsys.readouterr().out
