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

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, impose_rung,
                                              no_door, pin_dedicated_card, pin_host, profile)

# The machine each cell plans for is BUILT here (register 102): the Mac's profile and reading, a
# dedicated V100-16GB — never a profile id or a cache path this machine happens to have.
APPLE = APPLE_M4_PRO
MB = 1024 * 1024

# (model, height, width, host MB free, imposed rung MB) — the machine each case was measured on.
# PixArt-XL-1024 is the Mac's OWN refusal, its container at its reading (refused at 14 420 MB
# before the fix; after it, 6 segments peaking at 8 129.3 MB — the figures the Mac's render
# reports); PixArt-XL-2-1024-MS is the same class on the sibling container (transformer 1 642.8 MB
# against 1 464.5, the whole 178.3 MB between the two refusals); granite-speech and Flex are the two instances
# this rack's census found, at the rung the Mac's profile reads when its host is idle. The request
# is stated: Flex at 1024 x 1024 (neither Flex nor granite-speech declares a resolution, and an
# audio model has none).
OVER = [
    ("PixArt-XL-1024", 1024, 2048, 11198, 8192),
    ("PixArt-XL-2-1024-MS", 1024, 2048, 11198, 8192),
    ("granite-speech-3.3-8b", None, None, 18186, 16384),
    ("Flex.1-alpha", 1024, 1024, 18186, 16384),
]
IDS = [c[0] for c in OVER]


def _pin_machine(monkeypatch, host_free_mb: int, rung_mb: int) -> None:
    pin_host(monkeypatch, 24576, host_free_mb, "the Mac's reading this case was measured at")
    impose_rung(monkeypatch, rung_mb)


MODES = ["compiled", "triton"]         # segments are cut on the graph each engine will run


def _plan(model, height, width, spec=APPLE, refusal_ok=False, mode="compiled"):
    """(plan, solver, {component: memory}). With `refusal_ok`, a refusal returns plan=None and
    still hands back the component estimate — the precondition must hold whatever the solver
    decides, including before the fix, when this exact scenario refused."""
    root = container_root(model)
    kw = dict(batch_size=1)
    if height is not None:
        kw.update(height=height, width=width)
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    try:
        p = s.solve_smart(NBXContainer.load(str(root)), profile(spec),
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
    dev = s._prepare_devices(profile(APPLE))[0]
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
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "this rack's V100-16GB card 0 as measured")
    p, s, _ = _plan("DeepSeek-Coder-V2-Lite-Instruct", None, None, spec=V100_16GB, mode=mode)
    dev = s._prepare_devices(profile(V100_16GB))[0]
    assert dev.budget_mb == dev.capacity_mb, (
        f"the dedicated law no longer budgets this card at its capacity ({dev.budget_mb} vs "
        f"{dev.capacity_mb}); the premise of this control is gone, re-measure it")
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    cut = {n: [[g.first_op, g.last_op] for g in part.segments] for n, part in parts.items()}
    assert cut == {"model": [["aten.embedding::0", "aten.view::247"],
                             ["moe_fused::block.12", "aten.view::467"],
                             ["moe_fused::block.23", "custom.rms_norm::81"]]}, cut
