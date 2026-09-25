"""A weight's shape is architectural. It can never depend on a request dimension.

Principle 1 says a dim left CONCRETE where a symbol belongs is a tracer bug. The inverse is
equally a tracer bug and nothing looked for it: a SYMBOL where a constant belongs. An
activation's dims follow the request; a parameter's dims follow the model. A tracer that binds
one to the other has mistaken a coincidence for a relationship, and the coincidence holds only
at the trace values.

WHAT IT COST, 2026-09-23
------------------------
mochi-1-preview was retraced at `--trace-spatial 30,54`. At those values

    time(10) + width(54) == 64 == the attention head dim

so `param::pos_frequencies` axis 2 — the head dim — was bound to `s1 + s3`. At the container's
own default request that resolves to 14 + 106 = 120 against an activation of 64:

    Cannot broadcast (2, 22260, 24, 64) and (11872, 24, 120)

Seven runs were spent finding it, because a second build of the same model — retraced at
`26,30`, where 10 + 30 = 40 and nothing collides — carried the literal 64 and ran, and the two
sat in different caches with nothing anywhere saying one was malformed.

Register entry 91 named the collision guard's blind spot: it checks products and affine forms
and never sums. This gate is what that spot needed, and it is deliberately WIDER than sums —
any non-literal dim on a parameter is the same mistake whatever its shape. The census bears
that out: of the 599 offending dims across this cache, they are `symbol`, `add` AND `mul`.

WHY THE KNOWN SET IS PINNED RATHER THAN ASSERTED EMPTY
------------------------------------------------------
Eleven of 59 cached containers offend today, across 599 parameter dims.

The pin was first written from a census that printed only the FIRST offending component per
container, and this gate went red on its own first run: mochi's `vae` offends as well as its
`transformer`. That is the cheapest possible activation proof — the cell caught a component its
own author had missed, before any injection was needed. They are DATA, not code — each needs a retrace,
and several belong to models nobody is working on this week. A cell asserting zero would be red
from birth and would be disabled within a day, which is worse than no cell. So the set is
pinned: a twelfth container, or a new component inside a known one, turns this red. Shrinking
the set is also flagged, because a container that quietly stops offending has been retraced and
the pin should record it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import importlib.util
import sys

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))

#: Containers whose PARAMETERS carry at least one non-literal dim, measured 2026-09-23.
#: Each is a Forge defect awaiting a retrace, not a licence. The value is the component set,
#: so a known-bad container gaining a NEW bad component is still caught.
KNOWN_OFFENDERS = {
    "Sana_1600M_1024px_MultiLing":   {"vae"},
    "Ming-Lite-Omni-1.5":            {"image_vae"},
    "Real-ESRGAN-x4":                {"model"},
    "Real-ESRGAN-x4":                {"model"},
    "real-esrgan-x8":                {"model"},
    "SANA-Video_2B_720p_diffusers":  {"transformer"},
    "MiniCPM-o-4_5":                 {"flow_dit"},
    "Flex.1-alpha":                  {"text_encoder"},
    "Open-Sora-v2":                  {"text_encoder_2"},
    "mochi-1-preview":               {"transformer", "vae"},
    "parakeet-tdt-1.1b":             {"joint"},
}


def _tool():
    """`tools/weights_are_not_symbolic.py`, loaded by path — it is a script, not a package."""
    path = Path(__file__).resolve().parents[3] / "tools" / "weights_are_not_symbolic.py"
    spec = importlib.util.spec_from_file_location("weights_are_not_symbolic", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def found():
    if not CACHE.is_dir():
        pytest.skip("no local container cache on this machine")
    return _tool().scan(CACHE)


# ───────────────────────── the invariant ─────────────────────────

def test_no_container_outside_the_known_set_has_a_symbolic_weight_dim(found):
    """The gate. A twelfth container joining this list is a new Forge defect."""
    new = {m: sorted(c) for m, c in found.items() if m not in KNOWN_OFFENDERS}
    assert not new, (
        "a container's WEIGHT has a dim that is not a literal, which means the tracer bound an "
        "architectural constant to a request dimension. It holds only at the trace values and "
        "fails at any other request.\n"
        f"  new: {json.dumps(new, indent=2)}\n"
        "  Run `python tools/weights_are_not_symbolic.py` for the detail, and retrace at "
        "spatial values whose SUM and PRODUCT cannot equal any architectural constant.")


def test_no_known_offender_gained_a_new_component(found):
    """A known-bad container is pinned at the components that were bad. A new one is new."""
    grew = {m: sorted(set(found[m]) - KNOWN_OFFENDERS[m])
            for m in found if m in KNOWN_OFFENDERS and set(found[m]) - KNOWN_OFFENDERS[m]}
    assert not grew, f"a known offender gained a component: {json.dumps(grew, indent=2)}"


def test_a_container_that_stopped_offending_is_recorded(found):
    """Shrinking is good news and still fails, so the pin records the retrace instead of
    quietly drifting out of date. Delete the entry when this fires."""
    fixed = sorted(m for m in KNOWN_OFFENDERS if m not in found and (CACHE / m).is_dir())
    assert not fixed, (
        f"these containers no longer offend and should be removed from KNOWN_OFFENDERS: {fixed}")


# ───────────────────── the detector itself, on fixtures ─────────────────────

def test_the_detector_sees_a_sum_the_collision_guard_missed():
    """mochi's exact shape: a parameter dim bound to `s1 + s3` because 10 + 54 == 64."""
    g = {"tensors": {"param::pos_frequencies": {"is_parameter": True, "symbolic_shape": {"dims": [
        1, 24, {"type": "add", "left": {"type": "symbol", "id": "s1", "trace": 10},
                "right": {"type": "symbol", "id": "s3", "trace": 54}, "trace": 64}]}}}}
    bad = _tool().offending_parameters(g)
    assert len(bad) == 1 and bad[0][1] == 2 and bad[0][2]["type"] == "add"


def test_the_detector_is_wider_than_sums():
    """The census found `symbol` and `mul` too — 599 dims across 11 containers are not all
    sums, so a detector that only looked for sums would miss most of the class."""
    t = _tool()
    for expr in ({"type": "symbol", "id": "s2", "trace": 3},
                 {"type": "mul", "left": {"type": "symbol", "id": "s1"}, "right": 2, "trace": 1024},
                 {"type": "floordiv", "left": {"type": "symbol", "id": "s1"}, "right": 2}):
        g = {"tensors": {"param::w": {"is_parameter": True,
                                      "symbolic_shape": {"dims": [8, expr]}}}}
        assert len(t.offending_parameters(g)) == 1, expr


def test_an_ACTIVATION_with_a_symbolic_dim_is_NOT_flagged():
    """The whole point. An activation's dims SHOULD follow the request — flagging those would
    condemn every correctly traced graph in the cache and make the gate worthless."""
    g = {"tensors": {"aten.mul::0::out_0": {"is_parameter": False, "symbolic_shape": {"dims": [
        {"type": "symbol", "id": "s0", "trace": 2}, 24, 64]}}}}
    assert _tool().offending_parameters(g) == []


def test_a_weight_with_literal_dims_is_NOT_flagged():
    g = {"tensors": {"param::w": {"is_parameter": True,
                                  "symbolic_shape": {"dims": [320, 4, 3, 3]}}}}
    assert _tool().offending_parameters(g) == []


def test_a_weight_with_no_symbolic_shape_at_all_is_NOT_flagged():
    """Older graphs carry only `shape`. Absence of the record is not evidence of a defect."""
    g = {"tensors": {"param::w": {"is_parameter": True, "shape": [320, 4, 3, 3]}}}
    assert _tool().offending_parameters(g) == []


# ───────────────────── the fix, proven on the two real builds ─────────────────────

def test_the_retrace_that_avoids_the_collision_produces_a_LITERAL(): 
    """Both mochi builds exist on this rack and differ exactly here. This is the fix, measured:
    trace spatial values whose sum cannot equal the head dim, and the head dim stays a literal."""
    good = Path("/home/mlops/nbx/builds/scratch_cache/mochi-1-preview/components/transformer/graph.json")
    bad = CACHE / "mochi-1-preview" / "components" / "transformer" / "graph.json"
    if not (good.exists() and bad.exists()):
        pytest.skip("both mochi builds are needed for this comparison")
    t = _tool()
    dims_of = lambda p: ((json.loads(p.read_text())["tensors"]["param::pos_frequencies"]
                          .get("symbolic_shape") or {}).get("dims") or [])
    assert isinstance(dims_of(good)[2], int), "the 26,30 retrace should carry a literal head dim"
    assert dims_of(good)[2] == 64
    assert isinstance(dims_of(bad)[2], dict), "the 30,54 retrace should carry an expression"
    assert dims_of(bad)[2]["type"] == "add" and dims_of(bad)[2]["trace"] == 64
    assert t.offending_parameters(json.loads(good.read_text())) == []
