"""A symbol whose chain breaks on `v*2` is a QUESTION, and the census used to answer it silently.

`where_the_symbol_chain_breaks.py` classifies a break by how the literal relates to the
symbol's trace value: `v`, `v*2`, `v*4`, `v//2`, `v-1`, … Only `v` — the literal EQUAL to the
trace value — was reported by `certified_census.frozen_dims`. Everything else vanished, and
the census printed `frozen: []` and harvested the model as if it had been inspected.

The tool's reason for not calling a derived relation a defect is sound and is KEPT: an
architecture constant can coincide with an arithmetic of the symbol, and a graph alone cannot
tell the two apart. What was wrong was calling it CLEAN.

WHAT THE SILENCE COST, measured on 2026-09-22
---------------------------------------------
mochi-1-preview's VAE breaks `height` and `width` at `aten._unsafe_view::1` on `v*2` — literal
28 for a trace of 14, 44 for 22. 180 of its 353 five-dimensional activation tensors then carry
CONCRETE spatial dims. The consequence is arithmetic, not opinion:

* the profiler sizes `aten.silu::26` at `[1, 256, 84, 56, 88]` = 0.20 GiB;
* the run allocates `[1, 128, 84, 480, 848]` — and `1*128*84*480*848*2 == 8752988160`, exactly
  the byte count the allocator refused;
* so Prism plans against a figure **40x** too small, accepts a plan that cannot run, and the
  engine dies for lack of memory in a VAE the estimator called comfortable.

The census said `status: ok`, `frozen: []` over the top of that.

BLAST RADIUS, measured across this cache: **20 of 59 containers** carry at least one
derived-relation break and every one of them read `frozen: []` before this split existed —
Ming-Lite-Omni (8), the PixArt family (5, 5, 4, 4), Flex.1-alpha (4), Sana 4Kpx (4),
mochi (3), Wan, Allegro. Five carry an EXACT break, which was already caught.

So the row is reported and left UNADJUDICATED: `?` rather than a plausible reconstruction,
which is this project's own rule for a thing it cannot yet decide.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

CACHE = Path(os.environ.get("NEUROBRIX_CACHE_DIR") or (Path.home() / ".neurobrix" / "ca" "che"))

#: The cell used to pin `mochi-1-preview`, whose VAE broke `height` and `width` on `v*2`. On
#: 2026-09-22 at 17:44 that container was RETRACED and the spatial breaks went away — good news,
#: and it left this file asserting something no longer true of the cache. It stayed red from
#: then until 2026-09-23 and nothing noticed, because it was red for a reason nobody read.
#:
#: A gate pinned to ONE container is hostage to that container's next retrace. The class is what
#: matters and the class is alive: 9 of 59 containers still report a derived spatial break
#: (Flex.1-alpha v//4, Open-Sora-v2 v+2, four PixArt v//8, SANA-Video v-1, Sana-1600M v*2). So
#: the subject is DISCOVERED, and the cell skips loudly if the whole class ever disappears —
#: which would be a real event worth noticing rather than a silent green.
SPATIAL = ("height", "width")


def _census():
    spec = importlib.util.spec_from_file_location(
        "certified_census_under_test",
        Path(__file__).resolve().parents[3] / "tools" / "certified_census.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _derived(rows):
    """Rows whose break is a DERIVED relation — the ones that used to vanish silently."""
    return [r for r in rows if r.get("name") in SPATIAL
            and (r.get("first_break") or {}).get("relation", "") not in ("", "v")]


@pytest.fixture(scope="module")
def subject():
    """(model, rows) for a container that currently reports a derived spatial break."""
    m = _census()
    if not CACHE.is_dir():
        pytest.skip("no local container cache on this machine")
    for name in sorted(p.name for p in CACHE.iterdir() if (p / "components").is_dir()):
        try:
            rows = m.frozen_dims(name)
        except Exception:            # noqa: BLE001 — an unreadable container is not the subject
            continue
        if _derived(rows):
            return name, rows
    pytest.skip("no container in this cache reports a derived spatial break any more — if that "
                "is real, every one has been retraced and this cell has done its job")


@pytest.fixture(scope="module")
def rows(subject):
    return subject[1]


def test_a_derived_break_is_REPORTED_at_all(subject):
    """The rows that were silent. Without them the census prints `frozen: []` and the model is
    harvested as if it had been inspected."""
    model, rows = subject
    found = _derived(rows)
    assert found, f"{model} was selected for having a derived break and reports none"


def test_they_are_reported_as_UNADJUDICATED_not_as_a_defect(rows):
    """The tool's own caution, preserved: a derived relation is a question, not a verdict."""
    for r in _derived(rows):
        assert r["adjudicated"] is False, r
        assert r["first_break"]["relation"] != "v"


def test_the_relation_is_ARITHMETIC_on_the_trace_value(subject):
    """Whatever the relation, it must actually relate the literal to the trace value. A row
    that reports `v*2` where the literal is not twice the trace is a reporting bug, and that is
    checkable on any container rather than on one model's remembered numbers."""
    import re as _re
    model, rows = subject
    for r in _derived(rows):
        v, lit = r["trace_value"], r["first_break"]["literal"]
        rel = r["first_break"]["relation"]
        mm = _re.fullmatch(r"v([*+\-]|//)(\d+)", rel)
        assert mm, f"{model}: unrecognised relation {rel!r}"
        op, k = mm.group(1), int(mm.group(2))
        expect = {"*": v * k, "+": v + k, "-": v - k, "//": v // k}[op]
        assert lit == expect, f"{model}: {rel} on trace {v} should give {expect}, row says {lit}"


def test_split_frozen_keeps_the_two_classes_apart(rows):
    """A retrace is queued by an ADJUDICATED row only — an unadjudicated one must not stop a
    census, or 20 of 59 containers stop at once on a question nobody has answered."""
    m = _census()
    adjudicated, unadjudicated = m.split_frozen(rows)
    assert all(r.get("adjudicated") or r.get("unreadable") for r in adjudicated)
    assert len(unadjudicated) >= 2
    assert all(not r.get("adjudicated") for r in unadjudicated)


def test_the_arithmetic_that_makes_this_matter():
    """Not a graph fact — the reason the silence cost something, as one assertion.

    The estimator's shape against the shape the allocator actually refused.
    """
    predicted = 1 * 256 * 84 * 56 * 88 * 2          # what the profiler resolved
    actual = 1 * 128 * 84 * 480 * 848 * 2           # what the run asked for
    assert actual == 8752988160, "the byte count no longer matches the recorded failure"
    assert actual / predicted > 40, (
        f"the under-estimate is {actual / predicted:.1f}x; the recorded measurement was >40x")
