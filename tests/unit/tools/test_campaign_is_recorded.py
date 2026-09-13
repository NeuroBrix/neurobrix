"""A campaign runs under the flight recorder, or it does not run.

This rack has no UPS and loses mains — twice in nine minutes on 2026-09-11, and
the project's own doctrine calls that a known reality rather than an accident.
The recorder writes an fsync'ed record BEFORE the child starts, so a run the
power cuts leaves a record whose boot_id no longer matches the current one:
mechanical proof of an outage mid-run, and the session hook then prints the exact
resume command.

The 2026-09-11 MEET pass ran WITHOUT it. The 19:22 cut left no in_flight record,
therefore no resume block, therefore nothing that said what had been lost; the
campaign was reconstructed by hand from its own logs the following day.

That is a door to place, not a habit to acquire. Remembering to wrap the command
is precisely the discipline a power cut is under no obligation to respect, and a
rule that depends on someone remembering has already failed once here.

Run: PYTHONPATH=src:tools python -m pytest tests/unit/tools/test_campaign_is_recorded.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def _tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "precision_zoo_campaign", TOOLS / "precision_zoo_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_campaign_outside_the_recorder_is_refused(monkeypatch):
    m = _tool()
    monkeypatch.delenv("NBX_FLIGHTREC", raising=False)
    why = m.flightrec_refusal()
    assert why, "an unrecorded campaign must not start"
    # Actionable: it names the wrapper, not just the rule.
    assert "flightrec.py run" in why
    # And says WHY, so the next reader does not treat it as ceremony.
    assert "no UPS" in why


def test_a_recorded_campaign_proceeds(monkeypatch):
    m = _tool()
    monkeypatch.setenv("NBX_FLIGHTREC", "20260912_121900_some-label")
    assert m.flightrec_refusal() is None


def test_the_opening_is_deliberate_and_named(monkeypatch):
    """One way through, spelled the same everywhere, never a silent bypass."""
    m = _tool()
    monkeypatch.delenv("NBX_FLIGHTREC", raising=False)
    assert m.flightrec_refusal(allow_unrecorded=True) is None
    assert m.FLIGHTREC_OPT == "--allow-unrecorded"
    assert m.FLIGHTREC_OPT in m.flightrec_refusal()


def test_the_recorder_marks_its_child():
    """The refusal is only checkable because the recorder says so in the env.

    Without this the campaign cannot tell, and the door could only ever be a
    request. Pinned on the source because the alternative is launching a real
    recorded job from a unit test.
    """
    text = (TOOLS / "flightrec.py").read_text(encoding="utf-8")
    assert text.count('"NBX_FLIGHTREC": rec_id') == 2, (
        "both Popen paths must mark the child — the tee'd one and the bare one; "
        "a campaign launched through the unmarked path would be refused while "
        "actually being recorded")
