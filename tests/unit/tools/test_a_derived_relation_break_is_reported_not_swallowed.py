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
MODEL = "mochi-1-preview"


def _census():
    spec = importlib.util.spec_from_file_location(
        "certified_census_under_test",
        Path(__file__).resolve().parents[3] / "tools" / "certified_census.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rows():
    if not (CACHE / MODEL / "components").is_dir():
        pytest.skip(f"{MODEL} is not in this cache")
    return _census().frozen_dims(MODEL)


def _vae_spatial(rows):
    return [r for r in rows
            if r.get("component") == "vae" and r.get("name") in ("height", "width")]


def test_the_vae_spatial_breaks_are_REPORTED(rows):
    """The two rows that were silent. Without them the census prints `frozen: []`."""
    found = _vae_spatial(rows)
    assert len(found) == 2, f"expected height and width, got {[r.get('name') for r in rows]}"


def test_they_are_reported_as_UNADJUDICATED_not_as_a_defect(rows):
    """The tool's own caution, preserved: a derived relation is a question, not a verdict."""
    for r in _vae_spatial(rows):
        assert r["adjudicated"] is False, r
        assert r["first_break"]["relation"] != "v"


def test_the_relation_is_the_one_measured(rows):
    """`v*2` at `aten._unsafe_view::1`: 28 from a trace of 14, 44 from 22."""
    by_name = {r["name"]: r for r in _vae_spatial(rows)}
    assert by_name["height"]["trace_value"] == 14
    assert by_name["height"]["first_break"]["literal"] == 28
    assert by_name["width"]["trace_value"] == 22
    assert by_name["width"]["first_break"]["literal"] == 44
    for r in by_name.values():
        assert r["first_break"]["relation"] == "v*2"


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
