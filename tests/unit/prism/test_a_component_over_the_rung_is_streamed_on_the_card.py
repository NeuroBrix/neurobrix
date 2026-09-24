"""A component larger than the rung is streamed ON THE CARD, never refused, never sent to the host.

`_try_layer_streaming` decided which component to stream, and cut its segments, against the raw
`capacity_mb`. Every rung above it is judged against the RUNG — `_effective_capacity_mb`, the
free reading rounded down onto the ladder on a shared pool. A component landing between the two
was served by nothing. The Mac hit it on PixArt at 2048x1024 (e904da83):

    mps:0: unified memory — planning against 10638 MB actually free (machine: 11198 of 24576)
    ... ALL FAILED. The last rung needs only the largest single component to fit, and it does not:
      largest component: text_encoder at 9630MB

    text_encoder   9 630 MB
    rung           8 192 MB     rung_down(10 638)
    capacity      10 638 MB     11 198 x 0.95

and this rack hit it on granite-speech-3.3-8b under the same profile (16 769.6 MB against a
16 384 rung and a 17 277 capacity), which fell to `cpu_streaming`.

THESE CELLS CONSTRUCT THEIR SCENARIO
------------------------------------
The first version of this file FOUND its scenario: it planned against whatever rung the ladder
gave this rack's reading, and when the ladder was re-spaced the rung moved above the component,
the overhang vanished, and three cells went red for a reason that had nothing to do with
streaming. A gate for "a component over the rung streams" must not depend on which rungs the
ladder happens to have, nor on what this rack holds in RAM at the moment it runs (register 99).

So each cell pins the machine it plans for: the host reading is injected (the Mac's own measured
figures), the rung is imposed through the `NBX_PRISM_BUDGET_MB` door, and a precondition cell
proves from the solver's own component estimate that the largest component really does sit
strictly between the rung and the capacity. If a retrace ever moves a component out of that
window, the precondition says so instead of the verdict cells passing or failing for the wrong
reason.

Rounding the reading down onto the ladder is NOT touched: it is what makes a plan reproducible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import neurobrix.core.host_memory as host_memory
import neurobrix.core.prism.solver as solver_mod
from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import InputConfig, PrismSolver, load_profile
from neurobrix.core.prism.memory_budget import DeviceReading
from neurobrix.nbx import NBXContainer

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
APPLE = "default-9f169c79"              # Apple M4 Pro: unified, memory_mb 18 186, ram 24 576
MB = 1024 * 1024

# (model, height, width, host MB free, imposed rung MB) — the machine each case was measured on.
# PixArt is the Mac's refusal at its own reading; granite-speech and Flex are the two instances
# this rack's census found, at the rung the Mac's profile reads when its host is idle.
OVER = [
    ("PixArt-XL-2-1024-MS", 1024, 2048, 11198, 8192),
    ("granite-speech-3.3-8b", None, None, 18186, 16384),
    ("Flex.1-alpha", None, None, 18186, 16384),
]
IDS = [c[0] for c in OVER]


def _pin_machine(monkeypatch, host_free_mb: int, rung_mb: int) -> None:
    state = MemoryState(total_mb=24576, available_mb=host_free_mb,
                        source="injected by the cell: the machine this case was measured on")
    monkeypatch.setattr(solver_mod, "memory_state", lambda: state)
    monkeypatch.setattr(host_memory, "memory_state", lambda: state)
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(rung_mb))


MODES = ["compiled", "triton"]         # segments are cut on the graph each engine will run


def _plan(model, height, width, profile_id=APPLE, refusal_ok=False, mode="compiled"):
    """(plan, solver, {component: memory}). With `refusal_ok`, a refusal returns plan=None and
    still hands back the component estimate — the precondition must hold whatever the solver
    decides, including before the fix, when this exact scenario refused."""
    root = CACHE / model
    if not (root / "components").is_dir():
        pytest.skip(f"{model} is not in this cache")
    dj_path = root / "runtime" / "defaults.json"
    dj = json.loads(dj_path.read_text()) if dj_path.is_file() else {}
    kw = dict(batch_size=1, height=height or dj.get("height", 1024), width=width or dj.get("width", 1024))
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    try:
        p = s.solve_smart(NBXContainer.load(str(root)), load_profile(profile_id),
                          InputConfig(**kw), mode=mode)
    except RuntimeError:
        if not (refusal_ok and seen):
            raise
        p = None
    return p, s, seen


# ───────────────── the scenario, proved constructed before anything is judged ─────────────────

@pytest.mark.parametrize("model,h,w,host_free,rung", OVER, ids=IDS)
def test_the_largest_component_sits_between_the_rung_and_the_capacity(monkeypatch, model, h, w,
                                                                     host_free, rung):
    _pin_machine(monkeypatch, host_free, rung)
    _, s, seen = _plan(model, h, w, refusal_ok=True)
    dev = s._prepare_devices(load_profile(APPLE))[0]
    largest = max(m.total_bytes for m in seen.values()) / MB
    assert dev.budget_mb == rung, f"the door did not impose the rung: {dev.budget_mb}"
    assert rung < largest < dev.capacity_mb, (
        f"{model}: largest component {largest:.1f} MB is not strictly between the rung {rung} and "
        f"the capacity {dev.capacity_mb:.1f} — the scenario these cells judge is not the one planned")


# ───────────────── the law: streamed on the card ─────────────────

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model,h,w,host_free,rung", OVER, ids=IDS)
def test_a_component_over_the_rung_is_streamed_on_the_card(monkeypatch, model, h, w, host_free, rung,
                                                            mode):
    _pin_machine(monkeypatch, host_free, rung)
    p, _, _ = _plan(model, h, w, mode=mode)
    assert p.strategy == "layer_streaming", (
        f"{model} planned {p.strategy!r} with a component over the {rung} MB rung that the "
        f"device holds; that overhang is what layer_streaming is for")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model,h,w,host_free,rung", OVER, ids=IDS)
def test_and_its_segments_are_cut_against_the_rung_not_the_capacity(monkeypatch, model, h, w,
                                                                    host_free, rung, mode):
    """The assertion that a strategy name cannot make. Segments cut against the capacity put the
    announced peak above the rung — a plan that is a function of the live reading again.

    Looser than the solver by design of what it can see: the solver also reserves the graph's
    constants and the KV cache beside the segments; this checks peak + resident components."""
    _pin_machine(monkeypatch, host_free, rung)
    p, s, seen = _plan(model, h, w, mode=mode)
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    assert p.layer_stream_plan and parts, f"{model}: no streamed component in the plan"
    beside = sum(m.total_bytes for n, m in seen.items() if n not in parts)
    for name, part in parts.items():
        peak = (part.peak_resident_bytes + beside) / MB
        assert peak <= rung, (
            f"{model}.{name}: {len(part.segments)} segments peak at {peak:.1f} MB with what stays "
            f"resident beside them, over the {rung} MB rung the plan is budgeted at")


# ───────────────── the controls: the same model, the same machine, a rung that holds it ─────────────────

def test_the_same_component_under_the_rung_is_not_streamed(monkeypatch):
    """PixArt at the rung the Mac read earlier the same day (11 671 free -> 11 264), where it
    planned and ran. A component under the rung stays whole."""
    _pin_machine(monkeypatch, 11671, 11264)
    p, _, _ = _plan("PixArt-XL-2-1024-MS", 1024, 2048)
    assert p.strategy != "layer_streaming" and not p.strategy.startswith("cpu_"), p.strategy


@pytest.mark.parametrize("mode", MODES)
def test_a_dedicated_card_cuts_what_it_cut_before(monkeypatch, mode):
    """No door, a DEDICATED V100-16GB reading injected (driver 16 151 MB, own context 306 MB —
    this rack's card 0 as measured). The dedicated law budgets min(16 151 - 306, capacity), which
    is the capacity, so moving this rung from capacity to rung must move nothing here.

    Reaches `_try_layer_streaming` for real: DeepSeek-Coder-V2-Lite's single 17 777 MB component
    is over the card, the rung is attempted (its partition is left on the solver), and
    `lazy_sequential` then outscores it. The boundaries pinned below are the ones main's solver
    cut against the capacity, measured identical in both modes before this change. A retrace of
    this container moves these op ids (register 98): re-measure against main's cut, do not delete
    the cell."""
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    big = MemoryState(total_mb=257530, available_mb=200000, source="injected: an idle host")
    monkeypatch.setattr(solver_mod, "memory_state", lambda: big)
    monkeypatch.setattr(host_memory, "memory_state", lambda: big)
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda i: DeviceReading(
        kind="device", capacity_mb=16151, free_mb=15845, own_context_mb=306, measured=True,
        source="injected: a dedicated V100-16GB"))
    p, s, _ = _plan("DeepSeek-Coder-V2-Lite-Instruct", None, None,
                    profile_id="default-ff6008b7", mode=mode)
    dev = s._prepare_devices(load_profile("default-ff6008b7"))[0]
    assert dev.budget_mb == dev.capacity_mb, (
        f"the dedicated law no longer budgets this card at its capacity ({dev.budget_mb} vs "
        f"{dev.capacity_mb}); the premise of this control is gone, re-measure it")
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    cut = {n: [[g.first_op, g.last_op] for g in part.segments] for n, part in parts.items()}
    assert cut == {"model": [["aten.embedding::0", "aten.view::247"],
                             ["moe_fused::block.12", "aten.view::467"],
                             ["moe_fused::block.23", "custom.rms_norm::81"]]}, cut
