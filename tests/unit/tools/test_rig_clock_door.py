"""A rig is at the protocol clock when EVERY card is, not when the one you read is.

On 2026-09-11 this rack lost mains twice in nine minutes. Application clocks do
not survive a reboot, and this rack is heterogeneous: its 16 GB V100s return to
1312 MHz and its 32 GB ones to 1290 MHz -- their OWN factory defaults. The
protocol value is 1290.

So the machine came back with cards 0 and 1 at 1312 and cards 2 and 3 at 1290,
and a certification ran across all four. That two cards matched the protocol was
a manufacturer coincidence, not our lock.

The defect this pins is the SHAPE of the check, not the frequency: a check that
samples one card passes on a rig that is half wrong, and a check that samples
card 2 or 3 passes ALWAYS and would never once have fired. The tests below inject
the real 09-11 reading and require the door to refuse it and to name both
diverging cards with their values.

Run: PYTHONPATH=src:tools python -m pytest tests/unit/tools/test_rig_clock_door.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import rig_clock as RC


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def _reading(rows):
    """Install `rows` as what the driver reports (index, name, gfx, mem)."""
    out = "".join(f"{i}, {n}, {g}, {m}\n" for i, n, g, m in rows)
    return lambda cmd, *a, **k: _Result(out)


V16, V32 = "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"
ON_PROTOCOL = [(0, V16, 1290, 877), (1, V16, 1290, 877),
               (2, V32, 1290, 877), (3, V32, 1290, 877)]
# What the rack actually reported after the 2026-09-11 outage.
AS_FOUND_0911 = [(0, V16, 1312, 877), (1, V16, 1312, 877),
                 (2, V32, 1290, 877), (3, V32, 1290, 877)]


def test_every_card_at_the_protocol_clock_passes(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _reading(ON_PROTOCOL))
    state = RC.require_protocol_clock(say=lambda *a: None)
    assert state["off_protocol"] == []
    # It looked at four. A door reporting success over one card is the defect.
    assert state["cards_read"] == 4


def test_the_reading_that_followed_the_outage_is_refused(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _reading(AS_FOUND_0911))
    with pytest.raises(RC.OffProtocol) as exc:
        RC.require_protocol_clock(say=lambda *a: None)
    msg = str(exc.value)
    assert "2 of 4" in msg
    # Both diverging cards named, each with its own value: the operator must be
    # able to act without re-deriving which half is wrong.
    assert "card 0" in msg and "card 1" in msg and "1312" in msg
    # And the restore command is given, not described.
    assert "-ac 877,1290" in msg


def test_a_sampler_of_one_agreeing_card_would_have_passed(monkeypatch):
    """The trap, stated as a test so it cannot be reintroduced.

    Card 2 sat at 1290 on 09-11 because 1290 is its factory default. Any check
    reading only it saw the protocol value on a rig that was half wrong.
    """
    monkeypatch.setattr(subprocess, "run", _reading(AS_FOUND_0911))
    cards = {c["index"]: c for c in RC.rig_clocks()}
    want_gfx, _ = RC.protocol_clock()
    assert cards["2"]["graphics_mhz"] == want_gfx      # the sampler's green
    assert cards["3"]["graphics_mhz"] == want_gfx      # and still green
    with pytest.raises(RC.OffProtocol):                # the door's red
        RC.require_protocol_clock(say=lambda *a: None)


def test_zero_cards_is_a_refusal_not_a_pass(monkeypatch):
    """An instrument that examined nothing must not read as one that found nothing."""
    monkeypatch.setattr(subprocess, "run", _reading([]))
    with pytest.raises(RC.OffProtocol) as exc:
        RC.require_protocol_clock(say=lambda *a: None)
    assert "ZERO cards" in str(exc.value)


def test_a_driver_that_cannot_be_read_is_a_refusal(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Result(returncode=9, stderr="no driver"))
    with pytest.raises(RC.OffProtocol):
        RC.require_protocol_clock(say=lambda *a: None)


def test_a_missing_authority_refuses_rather_than_defaulting(monkeypatch, tmp_path):
    """No built-in fallback: a harness that invents the value stops citing it."""
    monkeypatch.setattr(RC, "PROTOCOL_FILE", tmp_path / "absent.json")
    with pytest.raises(RC.OffProtocol) as exc:
        RC.protocol_clock()
    assert "no default" in str(exc.value)


def test_the_opening_is_deliberate_and_says_so(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _reading(AS_FOUND_0911))
    said = []
    state = RC.require_protocol_clock(allow_off_protocol=True, say=said.append)
    assert state["waived"] is True
    assert len(state["off_protocol"]) == 2
    # A waiver that is quiet is a bypass. The run must state it in its own output.
    assert any("off protocol" in s for s in said)
