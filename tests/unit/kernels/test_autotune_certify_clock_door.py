"""A certification is a measurement, so it takes a measurement's entry condition.

`neurobrix autotune certify` picks a configuration BY TIMING candidates. Its
`best_ms` values therefore carry a clock regime whether or not anyone recorded
which one -- and on 2026-09-11 nobody did: this rack lost mains twice in nine
minutes, came back at 1312/1312/1290/1290 (each SKU at its OWN factory default,
protocol value 1290), and a certification ran across all four cards.

The workshop closed that on the two `tools/` harnesses the same day. It did NOT
close it on the engine's own certify command -- the one the doctrine names as the
only way to fill the certified directory -- which recorded the clocks into each
proof and refused nothing. These tests pin the refusal, at the narrowest point:
`certify()` itself, so the door holds for every caller and not only for the one
that types the documented command.

The defect pinned here is the SHAPE of the check, not the frequency. A check that
samples one card passes on a rig that is half wrong; one that samples card 2 or 3
on this rack passes ALWAYS and would never once have fired.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_autotune_certify_clock_door.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neurobrix.kernels import autotune_certify as AC

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import rig_clock as RC


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


V16, V32 = "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"
ON_PROTOCOL = [(0, V16, 1290, 877), (1, V16, 1290, 877),
               (2, V32, 1290, 877), (3, V32, 1290, 877)]
# What the rack actually reported after the 2026-09-11 outage.
AS_FOUND_0911 = [(0, V16, 1312, 877), (1, V16, 1312, 877),
                 (2, V32, 1290, 877), (3, V32, 1290, 877)]
# The memory clock alone off protocol. The engine's reader used to return only
# the graphics clock, which let this reading satisfy the engine door while
# failing the workshop's -- two doors disagreeing about one protocol.
MEMORY_OFF = [(0, V16, 1290, 877), (1, V16, 1290, 877),
              (2, V32, 1290, 810), (3, V32, 1290, 877)]


def _driver(rows):
    """Install `rows` as what nvidia-smi reports, for BOTH doors' query shapes."""
    def run(cmd, *a, **k):
        query = next((c for c in cmd if c.startswith("--query-gpu")), "")
        if "name" in query:                       # the workshop door asks for the name
            return _Result("".join(f"{i}, {n}, {g}, {m}\n" for i, n, g, m in rows))
        return _Result("".join(f"{i}, {g}, {m}\n" for i, n, g, m in rows))
    return run


@pytest.fixture(autouse=True)
def _fresh_reading(monkeypatch):
    """The engine memoises the clocks for the run; each test gets its own reading."""
    monkeypatch.setattr(AC._clocks_mhz, "cached", AC._UNREAD)
    yield
    monkeypatch.setattr(AC._clocks_mhz, "cached", AC._UNREAD)


def test_every_card_at_the_protocol_clock_passes(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _driver(ON_PROTOCOL))
    said = []
    AC.rig_protocol_refusal(say=said.append)
    # It looked at four. A door announcing success over one card is the defect.
    assert "4 card(s) read" in " ".join(said)


def test_the_reading_that_followed_the_outage_is_refused(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _driver(AS_FOUND_0911))
    with pytest.raises(RuntimeError) as exc:
        AC.rig_protocol_refusal(say=lambda *a: None)
    msg = str(exc.value)
    assert "2 of 4" in msg
    # Both diverging cards named WITH their own value: the operator has to know
    # which cards to restore, and "some card is wrong" does not say.
    assert "card 0  1312/877" in msg and "card 1  1312/877" in msg
    # And the conforming ones shown too, so a half-right rig reads as half right.
    assert msg.count("(at protocol)") == 2


def test_a_card_at_protocol_does_not_excuse_the_rig(monkeypatch):
    """The trap: 1290 is the 32 GB cards' factory default.

    A check that happened to read card 2 or card 3 would go green here on every
    run since this machine was built. The refusal must not depend on which card
    the certification will be pinned to -- a card at the wrong frequency beside a
    measurement is a fact about the rig.
    """
    monkeypatch.setattr(subprocess, "run", _driver(AS_FOUND_0911))
    with pytest.raises(RuntimeError):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_the_memory_clock_counts_too(monkeypatch):
    """`nvidia-smi -ac <mem>,<gfx>` sets both, so a protocol names both."""
    monkeypatch.setattr(subprocess, "run", _driver(MEMORY_OFF))
    with pytest.raises(RuntimeError) as exc:
        AC.rig_protocol_refusal(say=lambda *a: None)
    assert "card 2  1290/810" in str(exc.value)


def test_zero_cards_is_a_refusal_not_a_pass(monkeypatch):
    """An instrument that examined nothing must not read as one that found nothing."""
    monkeypatch.setattr(subprocess, "run", _driver([]))
    with pytest.raises(RuntimeError):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_an_unreadable_driver_is_a_refusal(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Result("", returncode=9, stderr="no driver"))
    with pytest.raises(RuntimeError):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_a_named_protocol_that_does_not_read_is_a_refusal(monkeypatch, tmp_path):
    """No built-in fallback value, ever.

    An engine that invents the number stops citing the protocol, and the
    divergence between the invented one and the machine's is silent forever.
    """
    monkeypatch.setattr(subprocess, "run", _driver(ON_PROTOCOL))
    monkeypatch.setenv(AC._PROTOCOL_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(RuntimeError):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_a_machine_declaring_no_protocol_is_told_so(monkeypatch, tmp_path):
    """An installed NeuroBrix carries no rack's decision -- and says as much.

    Not a refusal: there is nothing to diverge from, and the proof still records
    the clocks read. But not a silence either, which would be indistinguishable
    from a door that held.
    """
    monkeypatch.setattr(subprocess, "run", _driver(ON_PROTOCOL))
    monkeypatch.setattr(AC, "_protocol_file", lambda: None)
    said = []
    AC.rig_protocol_refusal(say=said.append)
    assert "declares no measurement protocol" in " ".join(said)


def test_the_opening_says_so_in_the_runs_own_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _driver(AS_FOUND_0911))
    said = []
    AC.rig_protocol_refusal(allow_off_protocol=True, say=said.append)
    text = " ".join(said)
    assert "WARNING" in text and AC.OFF_PROTOCOL_OPT in text
    # The consequence stated, not just the fact: these timings are not comparable.
    assert "not be compared" in text


def test_certify_refuses_before_it_times_anything(monkeypatch):
    """The door is inside certify(), not beside it.

    Placed in the CLI it would guard the documented command; placed here it
    guards every caller. It must also come FIRST -- a refusal after the first
    sweep has already spent the thing it exists to protect.
    """
    monkeypatch.setattr(subprocess, "run", _driver(AS_FOUND_0911))
    monkeypatch.setattr(AC.C, "active_profile",
                        lambda: pytest.fail("certify reached the profile before the clock door"))
    with pytest.raises(RuntimeError) as exc:
        AC.certify("volta", log=lambda *a: None)
    assert "protocol clock" in str(exc.value)


def test_the_two_doors_agree_on_the_same_reading(monkeypatch):
    """Two implementations, one protocol: pin them against drift.

    They are deliberately separate -- a campaign measures a FROZEN `--src`, so the
    workshop's door must not depend on whichever engine tree is being measured.
    Separateness is safe only while they answer identically, and they read the
    same authority file, so only the logic could drift. This is what notices.
    """
    for rows in (ON_PROTOCOL, AS_FOUND_0911, MEMORY_OFF):
        monkeypatch.setattr(subprocess, "run", _driver(rows))
        monkeypatch.setattr(AC._clocks_mhz, "cached", AC._UNREAD)

        def engine():
            AC.rig_protocol_refusal(say=lambda *a: None)

        def workshop():
            RC.require_protocol_clock(say=lambda *a: None)

        engine_refused = workshop_refused = None
        try:
            engine()
        except RuntimeError as exc:
            engine_refused = str(exc)
        try:
            workshop()
        except SystemExit as exc:              # RC.OffProtocol is a SystemExit
            workshop_refused = str(exc)

        assert (engine_refused is None) == (workshop_refused is None), (
            f"the two doors disagree on {rows}: engine "
            f"{'refused' if engine_refused else 'passed'}, workshop "
            f"{'refused' if workshop_refused else 'passed'}")


def test_the_protocol_authority_is_the_workshops_own_file():
    """Discovery, not a shipped copy: one file, so the values cannot drift."""
    found = AC._protocol_file()
    assert found is not None and found.name == "rig_protocol.json"
    clock = json.loads(found.read_text(encoding="utf-8"))["clock"]
    assert int(clock["application_graphics_mhz"]) == RC.protocol_clock()[0]
    assert int(clock["application_memory_mhz"]) == RC.protocol_clock()[1]
