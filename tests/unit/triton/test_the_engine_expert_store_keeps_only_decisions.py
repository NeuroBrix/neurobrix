"""The triton engine's expert store decides; the library moves the bytes (the owner's line 2026-11-12 11:05: what
touches tensors, device memory, copies and kernel launches lives in nbxtensor, never in the engine).

`triton/expert_store.NBXExpertStore` is `nbxtensor.memory.expert_slots.ExpertSlotStore` plus three decisions: the
decode's routing noted for Prism's reuse curve BEFORE the slots are acquired, the opt-in copy ahead of the next
layer's predicted experts, and the record written at release. The residency handed to the store answers what the
store asks of it (`resident()`: who sits where, least recently routed first).

What this test would do if the code were wrong: a device allocation, a copy or a kernel launch written back into the
engine's store — RED on the door (seen red on the file as it stood on main ae4b65a09: it imported the allocator, the
tensor, the dtypes, the layout); a routing noted after the acquire, or not at all — RED; a prefetch issued without the
opt-in, or for experts the residency did not plan — RED. No card."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from neurobrix.core.runtime.expert_residency import ExpertResidency

STORE = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "triton" / "expert_store.py"


def _library_imports(source: str) -> set:
    return {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module
            and n.module.split(".")[0] == "nbxtensor"} | {
        a.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import) for a in n.names
        if a.name.split(".")[0] == "nbxtensor"}


def test_the_engines_store_reaches_the_library_only_through_the_slot_store():
    assert _library_imports(STORE.read_text()) == {"nbxtensor.memory.expert_slots"}


def test_the_door_sees_a_store_that_allocates():
    was = "from nbxtensor.memory.allocator import DeviceAllocator as DA\nfrom nbxtensor.tensor import NBXTensor\n"
    assert _library_imports(was) == {"nbxtensor.memory.allocator", "nbxtensor.tensor"}


def test_the_residency_says_who_sits_where_least_recently_routed_first():
    r = ExpertResidency(3)
    r.assign("L0", [4, 1])
    r.assign("L1", [2])
    r.assign("L0", [1])                                   # expert 1 of L0 is the most recent again
    assert r.resident() == [(("L0", 4), 1), (("L1", 2), 2), (("L0", 1), 0)]
    r.assign("L1", [7])                                   # no free slot: the least recent goes
    assert r.resident() == [(("L1", 2), 2), (("L0", 1), 0), (("L1", 7), 1)]


def _store(monkeypatch, residency=None):
    from neurobrix.triton.expert_store import NBXExpertStore
    from nbxtensor.memory.expert_slots import ExpertSlotStore
    calls = []
    monkeypatch.setattr(ExpertSlotStore, "__init__", lambda self, *a, **k: calls.append(("init", a, k)))
    monkeypatch.setattr(ExpertSlotStore, "_acquire", lambda self, *a: calls.append(("acquire", a)) or "lists")
    monkeypatch.setattr(ExpertSlotStore, "prefetch", lambda self, *a: calls.append(("prefetch", a)) or len(a[1]))
    monkeypatch.setattr(ExpertSlotStore, "release", lambda self: calls.append(("release",)))
    s = NBXExpertStore(residency, 0, ("g", "u", "d"), in_place=False)
    s.residency, s.in_place = residency, False
    return s, calls


def test_the_routing_is_noted_before_the_slots_are_acquired(monkeypatch):
    s, calls = _store(monkeypatch)
    assert s.trace is None and s.hosts == {} and calls[0] == ("init", (None, 0, ("g", "u", "d")), {"in_place": False})
    s.trace = SimpleNamespace(note=lambda layer, routed: calls.append(("note", layer, routed)))
    assert s._acquire("L0", [3, 1], 8, "host") == "lists"
    assert calls[1:] == [("note", "L0", [3, 1]), ("acquire", ("L0", [3, 1], 8, "host"))]


def test_the_copy_ahead_is_opt_in_and_plans_through_the_residency(monkeypatch):
    r = ExpertResidency(4)
    r.assign("L0", [0, 1])
    r.assign("L1", [5])
    r.reset()                                             # nothing resident; the layer order and the stand-in below stay
    r.note("L0"), r.note("L1")
    r.last_routed = {"L1": [5]}
    s, calls = _store(monkeypatch, r)
    s.hosts = {"L1": "reader"}
    monkeypatch.delenv("NBX_EXPERT_PREFETCH", raising=False)
    assert s.on_enqueued("L0", [0, 1]) == 0 and not [c for c in calls if c[0] == "prefetch"]
    monkeypatch.setenv("NBX_EXPERT_PREFETCH", "1")
    assert s.on_enqueued("L0", [0, 1]) == 1
    assert calls[-1] == ("prefetch", ("L1", [5], "reader", {("L0", 0), ("L0", 1)}))
    assert s.on_enqueued("L1", [5]) == 0                # the last layer: nothing follows


def test_the_record_is_written_before_the_slots_are_released(monkeypatch):
    r = ExpertResidency(2)
    s, calls = _store(monkeypatch, r)
    s.trace = SimpleNamespace(write=lambda text: calls.append(("write", text)))
    s.release()
    assert [c[0] for c in calls[1:]] == ["write", "release"] and s.trace is None
