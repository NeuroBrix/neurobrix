"""«The rig is free» is a statement about the rig, not about an instant.

A campaign BETWEEN two of its runs holds no compute process. Ask nvidia-smi at
that moment and the list is empty. That is how a second instance was launched
onto a live measurement on 2026-09-10, from a log that merely looked idle — and
the check written afterwards repeated it in another form on 2026-09-11: it
returned 0 while a gate held GPU0, five seconds before and five seconds after.

So the rig is free when BOTH hold: no compute process, and no driver alive that
is about to start one.

Run: PYTHONPATH=src:tools python -m pytest tests/unit/tools/test_rig_free_means_no_driver_either.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import certify_the_catalogue as C


class _Result:
    def __init__(self, stdout):
        self.stdout, self.returncode = stdout, 0


def _fake_run(smi_out, ps_out):
    def run(cmd, *a, **k):
        if "nvidia-smi" in cmd[0]:
            return _Result(smi_out)
        return _Result(ps_out)
    return run


def test_a_compute_process_is_busy(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("12345, 900 MiB\n", ""))
    assert C._rig_busy() == 1


def test_a_campaign_between_two_runs_is_still_busy(monkeypatch):
    """The case that matters: nvidia-smi is EMPTY and the rig is not free."""
    ps = ("  1 /sbin/init\n"
          " 725619 /venv/bin/python tools/precision_zoo_campaign.py run --src ...\n")
    monkeypatch.setattr(subprocess, "run", _fake_run("", ps))
    assert C._rig_busy() == 1, (
        "an empty compute-app list during a campaign's gap between runs read as "
        "'free' — which is exactly how a second instance lands on a live "
        "measurement")


def test_a_genuinely_quiet_rig_is_free(monkeypatch):
    """The control. A check that never says free is not a check."""
    monkeypatch.setattr(subprocess, "run", _fake_run("", "  1 /sbin/init\n  2 [kthreadd]\n"))
    assert C._rig_busy() == 0


def test_a_plan_run_does_not_count_itself_as_a_driver(monkeypatch):
    """`--plan` takes no card; counting it would make the tool refuse itself."""
    ps = " 999 /venv/bin/python tools/certify_the_catalogue.py --plan\n"
    monkeypatch.setattr(subprocess, "run", _fake_run("", ps))
    assert C._rig_busy() == 0


def test_the_tool_does_not_refuse_itself(monkeypatch):
    """It did, on 2026-09-11. The first version excluded `os.getpid()`, but the
    shell wrapper that launches the tool carries the same script name on its
    command line under a different pid — so the run refused itself at the door
    and the MEET phase never started. Exclusion is by process GROUP."""
    import os
    mine = os.getsid(0)
    ps = ("  SID   PID CMD\n"
          f"{mine:>5} {os.getpid():>5} /venv/bin/python tools/certify_the_catalogue.py --src ...\n"
          f"{mine:>5} {os.getpid() + 1:>5} bash -c ... tools/certify_the_catalogue.py --src ...\n")
    monkeypatch.setattr(subprocess, "run", _fake_run("", ps))
    assert C._rig_busy() == 0, (
        "the tool counted its own wrapper as a driver holding the rig")


def test_another_session_running_the_same_tool_still_counts(monkeypatch):
    """The control: self-exclusion must not blind it to a SECOND instance."""
    import os
    mine = os.getsid(0)
    ps = ("  SID   PID CMD\n"
          f"{mine:>5} {os.getpid():>5} /venv/bin/python tools/certify_the_catalogue.py --src ...\n"
          f"{mine + 12345:>5} {os.getpid() + 999:>5} /venv/bin/python tools/certify_the_catalogue.py --src ...\n")
    monkeypatch.setattr(subprocess, "run", _fake_run("", ps))
    assert C._rig_busy() == 1
