"""A first launch READS the runtime's card facts (runtime_total_mb, runtime_context_mb), it does not start a child to
measure them (OWNER RESET 22/11 point 1; first-launch fact page line 6, 0.8 s of the Dell's 61.7 s): a process that
sees ONE card opens its context there whatever its plan names, so it reads them itself. With several cards seen, the
child that sees one card alone stays (2026-11-11: a one-card plan under two visible cards held a context on both).
Red on main a025add: `card_facts_reader` does not exist and the launch always starts the child."""
import pytest

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import facts_probe as P


@pytest.mark.parametrize("seen, reader", [(1, "runtime_card_facts"), (2, "card_facts_in_a_child"), (4, "card_facts_in_a_child")])
def test_the_reader_follows_the_cards_this_process_sees(monkeypatch, seen, reader):
    from nbxtensor import device as D
    monkeypatch.setattr(D, "count", lambda: seen)
    got = P.card_facts_reader([{"index": 0}])
    assert got is (A.runtime_card_facts if reader == "runtime_card_facts" else P.card_facts_in_a_child)


def test_a_profile_of_several_cards_keeps_the_child(monkeypatch):
    from nbxtensor import device as D
    monkeypatch.setattr(D, "count", lambda: 1)
    assert P.card_facts_reader([{"index": 0}, {"index": 1}]) is P.card_facts_in_a_child


def test_no_runtime_keeps_the_child_which_answers_nothing(monkeypatch):
    from nbxtensor import device as D

    def none():
        raise D.NoGpuRuntime("no runtime")
    monkeypatch.setattr(D, "count", none)
    assert P.card_facts_reader([{"index": 0}]) is P.card_facts_in_a_child


def test_one_seen_card_starts_no_child(monkeypatch):
    from nbxtensor import device as D
    monkeypatch.setattr(D, "count", lambda: 1)
    monkeypatch.setattr(A, "runtime_card_facts", lambda index: {"runtime_total_mb": 16151, "runtime_context_mb": 306})
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda index: pytest.fail("a child was started for a read"))
    devices = [{"index": 0, "model": "card", "memory_mb": 16384}]
    assert P.complete_runtime_card_facts(devices, P.card_facts_reader(devices)) == {"runtime_card_facts": [0]}
    assert devices[0]["runtime_context_mb"] == 306
