"""A GENERATED profile whose cards state no runtime total / context gets them measured at a launch, as its other
first-launch facts are — it is not refused for ever.

Before (2026-11-11): the two facts were written at generation only. The rack's two-card `default-*` profile, generated
before them, refused every plan — "the hardware profile declares no runtime_context_mb … delete it and run without
--hardware to regenerate it" — and a test that runs the entry point with two cards visible was red for it.

What this test does if the code is wrong: a card without the facts keeps none (red on main: no completion exists),
a card that states them is measured again, a unified-memory device is asked for a card's context, or a runtime that
cannot answer writes something — red. No card: the runtime's answer is a stub.
"""
from neurobrix.core.prism import facts_probe as P


def test_only_the_cards_that_state_none_are_completed():
    asked = []

    def facts(index):
        asked.append(index)
        return {"runtime_total_mb": 32501, "runtime_context_mb": 306}
    devices = [{"index": 0, "memory_mb": 32768},
               {"index": 1, "memory_mb": 32768, "runtime_total_mb": 32500, "runtime_context_mb": 300},
               {"index": 2, "memory_mb": 24576, "unified_memory": True}]
    assert P.complete_runtime_card_facts(devices, facts) == {"runtime_card_facts": [0]}
    assert asked == [0]
    assert (devices[0]["runtime_total_mb"], devices[0]["runtime_context_mb"]) == (32501, 306)
    assert devices[1]["runtime_context_mb"] == 300 and "runtime_context_mb" not in devices[2]


def test_a_runtime_that_cannot_answer_writes_nothing():
    devices = [{"index": 0, "memory_mb": 16384}]
    assert P.complete_runtime_card_facts(devices, lambda index: {}) == {} and "runtime_context_mb" not in devices[0]


def test_a_launch_on_a_generated_profile_calls_the_completion():
    """`ensure_measured_facts` (generated profiles only) is where the completion is asked, with autodetect's reading."""
    import inspect
    from neurobrix.core.prism import autodetect as A
    assert "complete_runtime_card_facts(devices, P.card_facts_in_a_child)" in inspect.getsource(A.ensure_measured_facts)


def test_a_card_is_read_in_its_own_process(monkeypatch):
    """Reading what the runtime takes on a card initialises it: the launching process must hold no context there
    (a launch holds its plan's cards only). The reading is a child's; red if it is taken in-process."""
    import subprocess
    import sys
    from types import SimpleNamespace as NS
    from neurobrix.core.prism import autodetect as A
    monkeypatch.setattr(A, "runtime_card_facts", lambda index: (_ for _ in ()).throw(AssertionError("read in-process")))
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return NS(returncode=0, stdout='noise\nCARD_FACTS {"runtime_total_mb": 32501, "runtime_context_mb": 306}\n')
    monkeypatch.setattr(subprocess, "run", run)
    assert P.card_facts_in_a_child(3) == {"runtime_total_mb": 32501, "runtime_context_mb": 306}
    assert seen[0][0] == sys.executable and "runtime_card_facts(3)" in seen[0][2]
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: NS(returncode=1, stdout=""))
    assert P.card_facts_in_a_child(3) == {}


def test_detection_initialises_no_card_and_generation_reads_them_in_a_child(monkeypatch):
    """`detect_devices` runs at EVERY launch without --hardware (the profile's name is a hash of what it sees): it must
    not ask the runtime about a card — that puts this process on every visible card (seen through the entry point,
    2026-11-11: a one-card plan under two visible cards held both). The cards are read once, in a child process, by
    the completion of the generated profile."""
    from neurobrix.core.prism import autodetect as A
    monkeypatch.setattr(A, "runtime_card_facts", lambda index: (_ for _ in ()).throw(AssertionError("detection asked the runtime")))
    import inspect
    for fn in (A._parse_nvidia_smi, A.detect_hardware, A.detect_devices, A._visible_set_tag):
        assert "runtime_card_facts" not in inspect.getsource(fn), fn.__name__
    # a freshly written profile gets them the same way an old one does: at the launch's ensure_measured_facts
    assert "complete_runtime_card_facts(devices, P.card_facts_in_a_child)" in inspect.getsource(A.ensure_measured_facts)
